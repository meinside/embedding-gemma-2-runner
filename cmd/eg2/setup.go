package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	eg2 "github.com/meinside/embedding-gemma-2-runner"
	"github.com/meinside/embedding-gemma-2-runner/internal/assets"
	"github.com/meinside/embedding-gemma-2-runner/internal/media"
)

type setupCmd struct {
	Common common `group:"Model options"`
	Media  bool   `short:"m" long:"media" description:"also download the vision and audio encoders (555 MB)"`
	Prune  bool   `long:"prune" description:"remove libraries and models of other versions from the cache, and nothing else"`
	Yes    bool   `short:"y" long:"yes" description:"with --prune: do not ask first"`
}

func runSetup(ctx context.Context, c *setupCmd, stdin io.Reader, stdout, stderr io.Writer) error {
	opts := c.Common.options(stderr)
	dirs, err := assets.CacheDirs(opts.CacheDir)
	if err != nil {
		return err
	}
	if c.Prune {
		return runPrune(dirs, c.Yes, stdin, stdout)
	}
	if opts.LibDir == "" {
		lo := assets.LibOptions{Processor: assets.Processor(opts.Processor), Progress: opts.Progress}
		if lo.Processor == "" {
			lo.Processor = assets.DefaultProcessor(opts.Device != eg2.DeviceCPU)
		}
		dir, p, err := assets.Libraries(ctx, dirs, lo)
		if err != nil {
			return err
		}
		fmt.Fprintf(stdout, "llama.cpp %s (%s)  %s\n", assets.LlamaVersion, p, dir)
	}
	mo := assets.ModelOptions{Weights: assets.Weights(opts.Weights), Progress: opts.Progress}
	path, err := assets.TextModel(ctx, dirs, mo)
	if err != nil {
		return err
	}
	fmt.Fprintf(stdout, "text model  %s\n", path)
	if c.Media {
		if path, err = assets.MMProj(ctx, dirs, mo); err != nil {
			return err
		}
		fmt.Fprintf(stdout, "encoders    %s\n", path)
	}
	return nil
}

// runPrune removes what Prunable lists, after showing it and asking.
func runPrune(dirs assets.Dirs, yes bool, stdin io.Reader, stdout io.Writer) error {
	paths, err := assets.Prunable(dirs)
	if err != nil {
		return err
	}
	if len(paths) == 0 {
		fmt.Fprintln(stdout, "nothing to remove: the cache holds only the current versions")
		return nil
	}
	fmt.Fprintln(stdout, "to remove:")
	for _, p := range paths {
		fmt.Fprintf(stdout, "  %s\n", p)
	}
	if !yes {
		fmt.Fprint(stdout, "remove? [y/N] ")
		line, _ := bufio.NewReader(stdin).ReadString('\n')
		switch strings.ToLower(strings.TrimSpace(line)) {
		case "y", "yes":
		default:
			fmt.Fprintln(stdout, "nothing removed")
			return nil
		}
	}
	for _, p := range paths {
		if err := os.RemoveAll(p); err != nil {
			return err
		}
		fmt.Fprintf(stdout, "removed %s\n", p)
	}
	return nil
}

type infoCmd struct {
	Cache string `long:"cache" value-name:"DIR" description:"cache for models and libraries"`
	JSON  bool   `long:"json" description:"print as JSON"`
}

// infoReport is what `eg2 info` shows; --json prints it as is.
type infoReport struct {
	Model        string   `json:"model"`
	LlamaVersion string   `json:"llama_cpp"`
	Processor    string   `json:"processor"`
	Processors   []string `json:"processors"`
	Cache        string   `json:"cache"`
	Models       string   `json:"models"`
	Libraries    string   `json:"libraries"`
	Libffi       string   `json:"libffi,omitempty"`
	Installed    []string `json:"installed"`
	FFmpeg       bool     `json:"ffmpeg"`
}

func runInfo(c *infoCmd, stdout io.Writer) error {
	d, err := assets.CacheDirs(c.Cache)
	if err != nil {
		return err
	}
	r := infoReport{
		Model:        assets.ModelRepo + "@" + assets.ModelRevision,
		LlamaVersion: assets.LlamaVersion,
		Processor:    string(assets.DefaultProcessor(true)),
		Cache:        d.Root,
		Models:       assets.ModelDir(d),
		Libraries:    d.Libs,
		Libffi:       d.Libffi,
		FFmpeg:       media.FFmpegAvailable(),
	}
	for _, p := range assets.Processors() {
		r.Processors = append(r.Processors, string(p))
		if _, err := os.Stat(assets.LibDir(d, p)); err == nil {
			r.Installed = append(r.Installed, "llama.cpp "+string(p))
		}
	}
	if files, _ := filepath.Glob(filepath.Join(assets.ModelDir(d), "*.gguf")); len(files) > 0 {
		for _, f := range files {
			r.Installed = append(r.Installed, filepath.Base(f))
		}
	}
	if c.JSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(r)
	}
	fmt.Fprintf(stdout, "model       %s\n", r.Model)
	fmt.Fprintf(stdout, "llama.cpp   %s, build %s for this machine (available: %v)\n", r.LlamaVersion, r.Processor, r.Processors)
	fmt.Fprintf(stdout, "ffmpeg      %v (needed for video)\n", r.FFmpeg)
	fmt.Fprintf(stdout, "installed   %v\n\n", r.Installed)
	printCacheDirs(stdout, c.Cache)
	return nil
}
