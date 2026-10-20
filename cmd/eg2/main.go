// Command eg2 embeds text, images, audio and video with EmbeddingGemma 2, from
// the command line or as an OpenAI-compatible HTTP server.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"syscall"

	eg2 "github.com/meinside/embedding-gemma-2-runner"
	"github.com/meinside/embedding-gemma-2-runner/internal/assets"
)

// Exit codes.
const (
	exitOK      = 0
	exitFailed  = 1 // something did not embed
	exitUsage   = 2
	exitStopped = 130
)

// errUsage marks errors that are the caller's to fix; the usage is printed.
var errUsage = errors.New("usage")

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	os.Exit(run(ctx, os.Args[1:], os.Stdin, os.Stdout, os.Stderr))
}

func run(ctx context.Context, args []string, stdin io.Reader, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		usage(stderr)
		return exitUsage
	}
	cmd, rest := args[0], args[1:]
	var err error
	switch cmd {
	case "embed":
		err = runEmbed(ctx, rest, stdin, stdout, stderr)
	case "serve":
		err = runServe(ctx, rest, stderr)
	case "setup":
		err = runSetup(ctx, rest, stdout, stderr)
	case "info":
		err = runInfo(rest, stdout, stderr)
	case "help", "-h", "-help", "--help":
		if len(rest) > 0 && rest[0] == "inputs" {
			printInputsHelp(stdout)
			return exitOK
		}
		usage(stdout)
		return exitOK
	default:
		fmt.Fprintf(stderr, "eg2: unknown command %q\n\n", cmd)
		usage(stderr)
		return exitUsage
	}
	switch {
	case err == nil:
		return exitOK
	case errors.Is(err, flag.ErrHelp):
		return exitOK
	case errors.Is(err, errUsage):
		fmt.Fprintln(stderr, "eg2:", err)
		return exitUsage
	case errors.Is(err, errSomeFailed):
		return exitFailed
	case ctx.Err() != nil:
		fmt.Fprintln(stderr, "eg2: stopped")
		return exitStopped
	}
	fmt.Fprintln(stderr, "eg2:", err)
	return exitFailed
}

func usage(w io.Writer) {
	fmt.Fprint(w, `eg2 embeds text, images, audio and video with EmbeddingGemma 2.

Usage:
  eg2 embed [flags] [FILE...]   print embeddings as JSON
  eg2 serve [flags]             OpenAI-compatible embeddings server
  eg2 setup [flags]             download the model and libraries now
  eg2 info [flags]              show versions, cache paths and what is installed
  eg2 help inputs               what an input can be: media, placeholders, tasks

Run "eg2 COMMAND -h" for the flags of a command.

`)
	printCacheDirs(w, "")
}

// printCacheDirs shows where downloads go, for usage and help output.
func printCacheDirs(w io.Writer, cacheDir string) {
	d, err := eg2.CacheDirs(cacheDir)
	if err != nil {
		fmt.Fprintf(w, "Cache: unknown (%v)\n", err)
		return
	}
	fmt.Fprintf(w, "Cache ($XDG_CACHE_HOME/eg2 when set, else the OS cache directory; -cache moves it):\n")
	fmt.Fprintf(w, "  models     %s\n", assets.ModelDir(d))
	fmt.Fprintf(w, "  libraries  %s\n", d.Libs)
	if d.Libffi != "" {
		fmt.Fprintf(w, "  libffi     %s (fixed by the ffi package)\n", d.Libffi)
	}
}

// common holds the flags every command that opens the model shares.
type common struct {
	cacheDir  string
	device    string
	processor string
	libDir    string
	weights   string
	offline   bool
	threads   int
	verbose   bool
}

func (c *common) register(fs *flag.FlagSet) {
	fs.StringVar(&c.cacheDir, "cache", "", "cache `dir` for models and libraries")
	fs.StringVar(&c.device, "device", "auto", "auto, cpu or gpu")
	fs.StringVar(&c.processor, "processor", "", "llama.cpp build to use (default: picked for this machine)")
	fs.StringVar(&c.libDir, "lib", "", "use the llama.cpp libraries in `dir` instead of downloading them")
	fs.StringVar(&c.weights, "weights", "q8_0", "q8_0, or bf16 for parity checks")
	fs.BoolVar(&c.offline, "offline", false, "fail instead of downloading anything")
	fs.IntVar(&c.threads, "threads", 0, "CPU threads (0: llama.cpp default)")
	fs.BoolVar(&c.verbose, "v", false, "show llama.cpp logs")
}

func (c *common) options(stderr io.Writer) (eg2.Options, error) {
	dev, ok := map[string]eg2.Device{"auto": eg2.DeviceAuto, "cpu": eg2.DeviceCPU, "gpu": eg2.DeviceGPU}[c.device]
	if !ok {
		return eg2.Options{}, fmt.Errorf("%w: -device %q: want auto, cpu or gpu", errUsage, c.device)
	}
	return eg2.Options{
		CacheDir:  c.cacheDir,
		Device:    dev,
		Processor: c.processor,
		LibDir:    c.libDir,
		Weights:   c.weights,
		Offline:   c.offline,
		Threads:   c.threads,
		Verbose:   c.verbose,
		Progress:  progressPrinter(stderr),
	}, nil
}

// newFlagSet returns a flag set whose -h output ends with the cache paths.
func newFlagSet(name, synopsis string, stderr io.Writer, c *common) *flag.FlagSet {
	fs := flag.NewFlagSet(name, flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.Usage = func() {
		fmt.Fprintf(stderr, "Usage: eg2 %s\n\nFlags:\n", synopsis)
		fs.PrintDefaults()
		fmt.Fprintln(stderr)
		cacheDir := ""
		if c != nil {
			cacheDir = c.cacheDir
		}
		printCacheDirs(stderr, cacheDir)
	}
	return fs
}

// parseFlags parses args and maps a bad flag to errUsage (the flag package
// has already printed why).
func parseFlags(fs *flag.FlagSet, args []string) error {
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return err
		}
		return fmt.Errorf("%w: %v", errUsage, err)
	}
	return nil
}

// progressPrinter reports downloads on stderr, keeping stdout for results.
func progressPrinter(w io.Writer) func(name string, done, total int64) {
	return func(name string, done, total int64) {
		const mib = 1 << 20
		if total > 0 {
			fmt.Fprintf(w, "\rdownloading %s: %d/%d MiB", name, done/mib, total/mib)
			if done >= total {
				fmt.Fprintln(w)
			}
			return
		}
		fmt.Fprintf(w, "\rdownloading %s: %d MiB", name, done/mib)
	}
}
