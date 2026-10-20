package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"os"
	"time"

	eg2 "github.com/meinside/embedding-gemma-2-runner"
	"github.com/meinside/embedding-gemma-2-runner/internal/server"
)

func runServe(ctx context.Context, args []string, stderr io.Writer) error {
	var c common
	fs := newFlagSet("serve", "serve [flags]\n\n"+
		"Serves POST /v1/embeddings (OpenAI-compatible, plus media inputs), GET /v1/models,\n"+
		"GET /health, and GET / with the input guide. Requests run one at a time.", stderr, &c)
	c.register(fs)
	addr := fs.String("addr", "127.0.0.1:8080", "listen `address`")
	allowFiles := fs.Bool("allow-local-files", false, "let inputs name files on this machine (default: data: URIs only)")
	apiKey := fs.String("api-key", "", "require this bearer `key` (default: $EG2_API_KEY, which keeps it out of ps)")
	maxBody := fs.Int64("max-body", 64, "largest request in `MiB`")
	maxInputs := fs.Int("max-inputs", 256, "most inputs per request")
	maxAudio := fs.Duration("max-audio", 10*time.Minute, "reject audio longer than this, read from its headers (0: no limit)")
	maxVideo := fs.Duration("max-video", 10*time.Minute, "reject video longer than this, read with ffprobe (0: no limit)")
	preload := fs.Bool("media", false, "load the vision and audio encoders at start, not on the first media input")
	cors := fs.String("cors", "", "let browser pages from this `origin` call the API (\"*\" for any)")
	if err := parseFlags(fs, args); err != nil {
		return err
	}
	if *apiKey == "" {
		*apiKey = os.Getenv("EG2_API_KEY")
	}
	if host, _, err := net.SplitHostPort(*addr); err != nil {
		return fmt.Errorf("%w: -addr: %v", errUsage, err)
	} else if ip := net.ParseIP(host); *apiKey == "" && (ip == nil || !ip.IsLoopback()) && host != "localhost" {
		fmt.Fprintf(stderr, "eg2: warning: %s is reachable from other machines and no API key is set\n", *addr)
	}

	opts, err := c.options(stderr)
	if err != nil {
		return err
	}
	e, err := eg2.Open(ctx, opts)
	if err != nil {
		return err
	}
	defer e.Close()
	if *preload {
		if err := e.LoadMedia(ctx); err != nil {
			return err
		}
	}

	logger := slog.New(slog.NewTextHandler(stderr, nil))
	srv := &http.Server{
		Addr: *addr,
		Handler: server.New(e, server.Options{
			AllowLocalFiles: *allowFiles,
			APIKey:          *apiKey,
			MaxBody:         *maxBody << 20,
			MaxInputs:       *maxInputs,
			MaxAudio:        *maxAudio,
			MaxVideo:        *maxVideo,
			CORSOrigin:      *cors,
			BatchTexts:      c.batch > 1,
			Help:            inputsHelp(),
			Logger:          logger,
		}),
		ReadHeaderTimeout: 10 * time.Second,
	}
	ln, err := net.Listen("tcp", *addr)
	if err != nil {
		return err
	}
	info := e.Info()
	logger.Info("serving", "addr", ln.Addr().String(), "device", info.Device, "model", info.Model,
		"media_loaded", info.MediaLoaded, "media_device", info.MediaDevice)

	errc := make(chan error, 1)
	go func() { errc <- srv.Serve(ln) }()
	select {
	case err := <-errc:
		return err
	case <-ctx.Done():
	}
	// Let a running embedding finish; a long video can take a while on CPU.
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	logger.Info("stopped")
	return nil
}
