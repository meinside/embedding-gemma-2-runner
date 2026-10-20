package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"time"

	eg2 "github.com/meinside/embedding-gemma-2-runner"
	"github.com/meinside/embedding-gemma-2-runner/internal/server"
)

type serveCmd struct {
	Common          common        `group:"Model options"`
	Addr            string        `short:"a" long:"addr" default:"127.0.0.1:8080" value-name:"ADDRESS" description:"listen address"`
	AllowLocalFiles bool          `long:"allow-local-files" description:"let inputs name files on this machine (default: data: URIs only)"`
	APIKey          string        `long:"api-key" env:"EG2_API_KEY" value-name:"KEY" description:"require this bearer key; the environment variable keeps it out of ps"`
	MaxBody         int64         `long:"max-body" default:"64" value-name:"MiB" description:"largest request"`
	MaxInputs       int           `long:"max-inputs" default:"256" value-name:"N" description:"most inputs per request"`
	MaxAudio        time.Duration `long:"max-audio" default:"10m" value-name:"DURATION" description:"reject audio longer than this, read from its headers (0: no limit)"`
	MaxVideo        time.Duration `long:"max-video" default:"10m" value-name:"DURATION" description:"reject video longer than this, read with ffprobe (0: no limit)"`
	Media           bool          `short:"m" long:"media" description:"load the vision and audio encoders at start, not on the first media input"`
	CORS            string        `long:"cors" value-name:"ORIGIN" description:"let browser pages from this origin call the API (\"*\" for any)"`
}

func runServe(ctx context.Context, c *serveCmd, stderr io.Writer) error {
	if host, _, err := net.SplitHostPort(c.Addr); err != nil {
		return fmt.Errorf("%w: --addr: %v", errUsage, err)
	} else if ip := net.ParseIP(host); c.APIKey == "" && (ip == nil || !ip.IsLoopback()) && host != "localhost" {
		fmt.Fprintf(stderr, "eg2: warning: %s is reachable from other machines and no API key is set\n", c.Addr)
	}

	e, err := eg2.Open(ctx, c.Common.options(stderr))
	if err != nil {
		return err
	}
	defer e.Close()
	if c.Media {
		if err := e.LoadMedia(ctx); err != nil {
			return err
		}
	}

	logger := slog.New(slog.NewTextHandler(stderr, nil))
	srv := &http.Server{
		Addr: c.Addr,
		Handler: server.New(e, server.Options{
			AllowLocalFiles: c.AllowLocalFiles,
			APIKey:          c.APIKey,
			MaxBody:         c.MaxBody << 20,
			MaxInputs:       c.MaxInputs,
			MaxAudio:        c.MaxAudio,
			MaxVideo:        c.MaxVideo,
			CORSOrigin:      c.CORS,
			BatchTexts:      c.Common.Batch > 1,
			Help:            inputsHelp(),
			Logger:          logger,
		}),
		ReadHeaderTimeout: 10 * time.Second,
	}
	ln, err := net.Listen("tcp", c.Addr)
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
