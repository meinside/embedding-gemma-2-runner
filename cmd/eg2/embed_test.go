package main

import (
	"bytes"
	"context"
	"fmt"
	"image"
	"image/png"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	eg2 "github.com/meinside/embedding-gemma-2-runner"
)

func TestCombinedInput(t *testing.T) {
	tests := []struct {
		name      string
		parts     []flagPart
		task      string
		literal   bool
		wantText  string
		wantMedia int
		wantTask  eg2.Task
	}{
		{name: "flag order without placeholders",
			parts:    []flagPart{{"image", "a.jpg"}, {"text", "caption"}, {"audio", "a.wav"}},
			wantText: "<|image|>caption<|audio|>", wantMedia: 2},
		{name: "placeholders keep the text",
			parts:    []flagPart{{"text", "x <|video|> y"}, {"video", "v.mp4"}},
			wantText: "x <|video|> y", wantMedia: 1},
		{name: "task applies once the text is known",
			parts: []flagPart{{"text", "query"}}, task: "search-query",
			wantText: "query", wantTask: eg2.TaskSearchQuery},
		{name: "media alone get no task",
			parts: []flagPart{{"image", "a.jpg"}}, task: "document",
			wantText: "<|image|>", wantMedia: 1},
		{name: "literal puts media after the text",
			parts:   []flagPart{{"image", "a.jpg"}, {"text", "<|image|>"}},
			literal: true, wantText: "<|image|>", wantMedia: 1},
	}
	for _, tt := range tests {
		f := embedCmd{parts: tt.parts, Task: tt.task, Literal: tt.literal}
		in, err := f.combinedInput(nil)
		if err != nil {
			t.Fatalf("%s: %v", tt.name, err)
		}
		if in.Text != tt.wantText || len(in.Media) != tt.wantMedia || in.Task != tt.wantTask {
			t.Errorf("%s: text %q, %d media, task %q; want %q, %d, %q",
				tt.name, in.Text, len(in.Media), in.Task, tt.wantText, tt.wantMedia, tt.wantTask)
		}
	}
}

func TestFileInput(t *testing.T) {
	dir := t.TempDir()
	write := func(name string, b []byte) string {
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, b, 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}
	var img bytes.Buffer
	png.Encode(&img, image.NewRGBA(image.Rect(0, 0, 2, 2)))

	pngPath := write("a.png", img.Bytes())
	textPath := write("main.go", []byte("package main\n"))
	binPath := write("clip.wav", []byte("RIFF\x00\x00\x00\x00WAVE"))

	in, err := fileInput(pngPath)
	if err != nil || len(in.Media) != 1 || in.Media[0].Kind != eg2.MediaImage {
		t.Errorf("png: %+v, %v", in, err)
	}
	in, err = fileInput(textPath)
	if err != nil || in.Text != "package main\n" || len(in.Media) != 0 {
		t.Errorf("text: %+v, %v", in, err)
	}
	in, err = fileInput(binPath)
	if err != nil || len(in.Media) != 1 || in.Media[0].Kind != eg2.MediaAuto {
		t.Errorf("binary: %+v, %v", in, err)
	}
	if _, err := fileInput(dir); err == nil {
		t.Error("directory accepted")
	}
}

func TestRecordWriter(t *testing.T) {
	recs := []record{{ID: "a", Embedding: floatList{0.5, -0.25}}, {ID: "b", Error: "x <|image|>"}}
	tests := []struct {
		format string
		n      int
		want   string
	}{
		{"json", 1, `{"id":"a","source":"","embedding":[0.5,-0.25]}` + "\n"},
		{"json", 2, "[\n" + `{"id":"a","source":"","embedding":[0.5,-0.25]}` + ",\n" + `{"id":"b","source":"","error":"x <|image|>"}` + "\n]\n"},
		{"json", 0, "[]\n"},
		{"jsonl", 2, `{"id":"a","source":"","embedding":[0.5,-0.25]}` + "\n" + `{"id":"b","source":"","error":"x <|image|>"}` + "\n"},
	}
	for _, tt := range tests {
		var buf bytes.Buffer
		w := newRecordWriter(&buf, tt.format)
		for _, r := range recs[:tt.n] {
			if err := w.write(r); err != nil {
				t.Fatal(err)
			}
		}
		if err := w.close(); err != nil {
			t.Fatal(err)
		}
		if buf.String() != tt.want {
			t.Errorf("%s/%d:\n got %q\nwant %q", tt.format, tt.n, buf.String(), tt.want)
		}
	}
}

func TestUsageShowsCacheDirs(t *testing.T) {
	t.Setenv("XDG_CACHE_HOME", t.TempDir())
	var out bytes.Buffer
	if code := run(t.Context(), []string{"help"}, nil, &out, &out); code != exitOK {
		t.Fatalf("exit %d", code)
	}
	for _, want := range []string{"models", "libraries", os.Getenv("XDG_CACHE_HOME")} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("help lacks %q:\n%s", want, out.String())
		}
	}
}

func TestSetupPrune(t *testing.T) {
	cache := t.TempDir()
	old := filepath.Join(cache, "lib", "v0.0.1")
	if err := os.MkdirAll(old, 0o755); err != nil {
		t.Fatal(err)
	}
	run := func(stdin string, extra ...string) string {
		t.Helper()
		var out, errOut bytes.Buffer
		args := append([]string{"setup", "--cache", cache, "--prune"}, extra...)
		if code := run(context.Background(), args, strings.NewReader(stdin), &out, &errOut); code != exitOK {
			t.Fatalf("%v: exit %d\n%s", args, code, errOut.String())
		}
		return out.String()
	}
	if out := run("n\n"); !strings.Contains(out, old) || !strings.Contains(out, "nothing removed") {
		t.Errorf("answer n:\n%s", out)
	}
	if _, err := os.Stat(old); err != nil {
		t.Fatal("removed after answering n")
	}
	if out := run("", "-y"); !strings.Contains(out, "removed "+old) {
		t.Errorf("-y:\n%s", out)
	}
	if _, err := os.Stat(old); !os.IsNotExist(err) {
		t.Error("not removed with -y")
	}
	if out := run("y\n"); !strings.Contains(out, "nothing to remove") {
		t.Errorf("second run:\n%s", out)
	}
}

// The option syntax: short and long names, values, order, and the errors.
func TestParseArgs(t *testing.T) {
	parse := func(args ...string) (*options, int, string) {
		t.Helper()
		var errOut bytes.Buffer
		var got *options
		code := runWith(context.Background(), args, nil, io.Discard, &errOut, func(o *options) error {
			got = o
			return nil
		})
		return got, code, errOut.String()
	}
	o, code, _ := parse("embed", "--image", "a.jpg", "-t", "cap", "--video", "v.mp4", "--image", "b.png", "-d", "256", "-lv", "--task", "document", "f1", "f2")
	if code != exitOK {
		t.Fatal("parse failed")
	}
	if got := fmt.Sprint(o.Embed.parts); got != "[{image a.jpg} {text cap} {video v.mp4} {image b.png}]" {
		t.Errorf("parts in order: %s", got)
	}
	if o.Embed.Dim != 256 || !o.Embed.Literal || !o.Embed.Common.Verbose || o.Embed.Task != "document" || fmt.Sprint(o.Embed.Args.Files) != "[f1 f2]" {
		t.Errorf("flags: %+v", o.Embed)
	}
	if _, code, msg := parse("embed", "--format", "xml", "x"); code != exitUsage || !strings.Contains(msg, "xml") {
		t.Errorf("bad choice: %d %q", code, msg)
	}
	if _, code, msg := parse("embed", "-t", "a", "-t", "b"); code != exitUsage || !strings.Contains(msg, "given twice") {
		t.Errorf("text twice: %d %q", code, msg)
	}
	if _, code, msg := parse(); code != exitUsage || !strings.Contains(msg, "Usage") {
		t.Errorf("no command: %d %q", code, msg)
	}
	if o, code, _ := parse("serve", "-a", ":9000", "-m", "--max-audio", "90s"); code != exitOK || o.Serve.Addr != ":9000" || !o.Serve.Media || o.Serve.MaxAudio != 90*time.Second {
		t.Errorf("serve: %d %+v", code, o.Serve)
	}
	t.Setenv("EG2_API_KEY", "fromenv")
	if o, _, _ := parse("serve"); o.Serve.APIKey != "fromenv" {
		t.Errorf("api key from the environment: %q", o.Serve.APIKey)
	}
}

func TestHelpOutput(t *testing.T) {
	for _, args := range [][]string{{"-h"}, {"--help"}, {"embed", "-h"}, {"help"}} {
		var out, errOut bytes.Buffer
		if code := run(context.Background(), args, nil, &out, &errOut); code != exitOK {
			t.Errorf("%v: exit %d\n%s", args, code, errOut.String())
		}
		if !strings.Contains(out.String(), "Usage") || !strings.Contains(out.String(), "models") {
			t.Errorf("%v: help without usage and cache paths:\n%s", args, out.String())
		}
	}
	var out bytes.Buffer
	run(context.Background(), []string{"help", "inputs"}, nil, &out, io.Discard)
	if !strings.Contains(out.String(), "<|image|>") {
		t.Errorf("help inputs:\n%s", out.String())
	}
}
