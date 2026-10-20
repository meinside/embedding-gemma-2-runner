// Command eg2 embeds text, images, audio and video with EmbeddingGemma 2, from
// the command line or as an OpenAI-compatible HTTP server.
package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/signal"
	"syscall"

	flags "github.com/jessevdk/go-flags"

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

// errUsage marks errors that are the caller's to fix.
var errUsage = errors.New("usage")

// options are the commands and their flags; go-flags reads them from the
// tags. Short letters go to the flags typed most (-t is --task); -v is
// verbose, as usual.
type options struct {
	Embed embedCmd `command:"embed" description:"print embeddings as JSON" long-description:"Each FILE is one input: an image, audio or video file, or a text file.\n--text and the media options together make one more input.\nSee \"eg2 help inputs\" for placeholders and tasks."`
	Serve serveCmd `command:"serve" description:"OpenAI-compatible embeddings server" long-description:"Serves POST /v1/embeddings (OpenAI-compatible, plus media inputs), GET /v1/models,\nGET /health, and GET / with the input guide. Requests run one at a time."`
	Setup setupCmd `command:"setup" description:"download the model and libraries now" long-description:"Downloads the llama.cpp libraries and the text model, so later runs need no\nnetwork. Images, audio and video also need --media. --prune instead removes\nwhat the cache holds for other versions."`
	Info  infoCmd  `command:"info" description:"show versions, cache paths and what is installed"`
	Help  helpCmd  `command:"help" description:"this list; \"help inputs\": what an input can be"`
}

type helpCmd struct {
	Args struct {
		Topic string `positional-arg-name:"TOPIC"`
	} `positional-args:"yes"`
}

// common holds the flags every command that opens the model shares.
type common struct {
	Cache     string `long:"cache" value-name:"DIR" description:"cache for models and libraries"`
	Device    string `long:"device" default:"auto" choice:"auto" choice:"cpu" choice:"gpu" description:"where the model runs"`
	Processor string `long:"processor" value-name:"NAME" description:"llama.cpp build to use (default: picked for this machine)"`
	Lib       string `long:"lib" value-name:"DIR" description:"use the llama.cpp libraries in DIR instead of downloading them"`
	Weights   string `long:"weights" default:"q8_0" choice:"q8_0" choice:"bf16" description:"bf16 is for parity checks"`
	Offline   bool   `long:"offline" description:"fail instead of downloading anything"`
	Threads   int    `long:"threads" value-name:"N" description:"CPU threads (0: llama.cpp default)"`
	Batch     int    `long:"batch" value-name:"N" description:"embed up to N text inputs in one pass; faster for many short texts, vectors differ from one-at-a-time ones by about 3e-4 cosine (0: one at a time)"`
	Verbose   bool   `short:"v" long:"verbose" description:"show llama.cpp logs"`
}

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	os.Exit(run(ctx, os.Args[1:], os.Stdin, os.Stdout, os.Stderr))
}

func run(ctx context.Context, args []string, stdin io.Reader, stdout, stderr io.Writer) int {
	return runWith(ctx, args, stdin, stdout, stderr, nil)
}

// runWith is run with the parsed options handed to parsed instead of the
// command, for tests of the option syntax.
func runWith(ctx context.Context, args []string, stdin io.Reader, stdout, stderr io.Writer, parsed func(*options) error) int {
	var o options
	o.Embed.init()
	p := flags.NewNamedParser("eg2", flags.Default&^flags.PrintErrors)
	p.LongDescription = "eg2 embeds text, images, audio and video with EmbeddingGemma 2."
	if _, err := p.AddGroup("Commands", "", &o); err != nil {
		panic(err) // a tag is wrong: a programming error
	}
	if opt := p.Find("embed").FindOptionByLongName("task"); opt != nil {
		opt.Description += ": " + taskList()
	}
	if _, err := p.ParseArgs(args); err != nil {
		var fe *flags.Error
		if !errors.As(err, &fe) {
			fmt.Fprintln(stderr, "eg2:", err)
			return exitUsage
		}
		switch fe.Type {
		case flags.ErrHelp:
			fmt.Fprintln(stdout, fe.Message)
			fmt.Fprintln(stdout)
			printCacheDirs(stdout, "")
			return exitOK
		case flags.ErrCommandRequired:
			p.WriteHelp(stderr)
			fmt.Fprintln(stderr)
			printCacheDirs(stderr, "")
			return exitUsage
		}
		fmt.Fprintln(stderr, "eg2:", fe.Message)
		return exitUsage
	}

	var err error
	switch {
	case parsed != nil:
		err = parsed(&o)
	case p.Active.Name == "embed":
		err = runEmbed(ctx, &o.Embed, stdin, stdout, stderr)
	case p.Active.Name == "serve":
		err = runServe(ctx, &o.Serve, stderr)
	case p.Active.Name == "setup":
		err = runSetup(ctx, &o.Setup, stdin, stdout, stderr)
	case p.Active.Name == "info":
		err = runInfo(&o.Info, stdout)
	case p.Active.Name == "help":
		switch o.Help.Args.Topic {
		case "inputs":
			printInputsHelp(stdout)
		case "":
			p.WriteHelp(stdout)
			fmt.Fprintln(stdout)
			printCacheDirs(stdout, "")
		default:
			fmt.Fprintf(stderr, "eg2: no help topic %q; there is \"eg2 help inputs\"\n", o.Help.Args.Topic)
			return exitUsage
		}
		return exitOK
	}
	switch {
	case err == nil:
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

// printCacheDirs shows where downloads go, for usage and help output.
func printCacheDirs(w io.Writer, cacheDir string) {
	d, err := eg2.CacheDirs(cacheDir)
	if err != nil {
		fmt.Fprintf(w, "Cache: unknown (%v)\n", err)
		return
	}
	fmt.Fprintf(w, "Cache ($XDG_CACHE_HOME/eg2 when set, else the OS cache directory; --cache moves it):\n")
	fmt.Fprintf(w, "  models     %s\n", assets.ModelDir(d))
	fmt.Fprintf(w, "  libraries  %s\n", d.Libs)
	if d.Libffi != "" {
		fmt.Fprintf(w, "  libffi     %s (fixed by the ffi package)\n", d.Libffi)
	}
}

func (c *common) options(stderr io.Writer) eg2.Options {
	return eg2.Options{
		CacheDir:  c.Cache,
		Device:    map[string]eg2.Device{"auto": eg2.DeviceAuto, "cpu": eg2.DeviceCPU, "gpu": eg2.DeviceGPU}[c.Device],
		Processor: c.Processor,
		LibDir:    c.Lib,
		Weights:   c.Weights,
		Offline:   c.Offline,
		Threads:   c.Threads,
		BatchSeqs: c.Batch,
		Verbose:   c.Verbose,
		Progress:  progressPrinter(stderr),
	}
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
