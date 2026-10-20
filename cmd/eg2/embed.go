package main

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"image"
	"io"
	"os"
	"path/filepath"
	"strings"
	"unicode/utf8"

	eg2 "github.com/meinside/embedding-gemma-2-runner"
)

var errSomeFailed = errors.New("some inputs failed")

// maxTextFile is the largest file read as text. 8192 tokens are far less, so
// anything bigger fails either way; as media it gets a clearer error.
const maxTextFile = 4 << 20

// flagPart is one --text, --image, --audio or --video, in command-line order.
type flagPart struct {
	kind  string
	value string
}

// part is one of those options. Each occurrence is appended to a list the
// four share, so the order across them is kept: without placeholders it is
// the order in the vector.
type part struct {
	kind  string
	parts *[]flagPart
}

func (p *part) UnmarshalFlag(v string) error {
	if p.kind == "text" {
		for _, q := range *p.parts {
			if q.kind == "text" {
				return errors.New("--text given twice; use placeholders to place media in one text")
			}
		}
	}
	*p.parts = append(*p.parts, flagPart{p.kind, v})
	return nil
}

func (p part) MarshalFlag() (string, error) { return "", nil }

type embedCmd struct {
	Common  common `group:"Model options"`
	Text    part   `short:"t" long:"text" value-name:"TEXT" description:"text of the combined input (\"-\" reads stdin)"`
	Image   part   `long:"image" value-name:"FILE" description:"image for the combined input (repeatable)"`
	Audio   part   `long:"audio" value-name:"FILE" description:"audio for the combined input (repeatable)"`
	Video   part   `long:"video" value-name:"FILE" description:"video for the combined input (repeatable)"`
	Task    string `long:"task" value-name:"TASK" description:"task prefix"`
	Title   string `long:"title" value-name:"TITLE" description:"document title, with --task document"`
	Prompt  string `long:"prompt" value-name:"TEXT" description:"custom prefix instead of --task"`
	Literal bool   `short:"l" long:"literal" description:"keep placeholders and special-token strings in text as written"`
	Input   string `short:"i" long:"input" value-name:"FILE" description:"read JSON inputs, one per line, from FILE (\"-\": stdin)"`
	Dim     int    `short:"d" long:"dim" default:"768" value-name:"N" description:"768, 512, 256 or 128"`
	Format  string `short:"f" long:"format" choice:"json" choice:"jsonl" description:"default: json for one input, jsonl for more"`
	Args    struct {
		Files []string `positional-arg-name:"FILE"`
	} `positional-args:"yes"`

	parts []flagPart
}

// init points the four media options at the shared list.
func (c *embedCmd) init() {
	c.Text, c.Image, c.Audio, c.Video = part{"text", &c.parts}, part{"image", &c.parts}, part{"audio", &c.parts}, part{"video", &c.parts}
}

// job is one input and where it came from, for the output record.
type job struct {
	in     eg2.Input
	source string
	err    error // the input could not even be read
}

func runEmbed(ctx context.Context, f *embedCmd, stdin io.Reader, stdout, stderr io.Writer) error {
	files := f.Args.Files
	if err := f.validate(len(files)); err != nil {
		return err
	}

	jobs, err := f.jobs(files, stdin)
	if err != nil {
		return err
	}
	format := f.Format
	if format == "" {
		format = "json"
		if f.Input != "" || len(f.parts) > 0 && len(files) > 0 || len(files) > 1 {
			format = "jsonl"
		}
	}

	e, err := eg2.Open(ctx, f.Common.options(stderr))
	if err != nil {
		return err
	}
	defer e.Close()
	info := e.Info()
	model := fmt.Sprintf("%s/%s", shortModel(info.Model), info.Weights)

	w := newRecordWriter(stdout, format)
	failed, total := 0, 0
	write := func(j job, r eg2.Result, err error) error {
		rec := record{ID: j.in.ID, Source: j.source, Model: model}
		if rec.ID == "" {
			rec.ID = j.source
		}
		if err == nil {
			if err = r.Err; err == nil {
				var vec []float32
				if vec, err = eg2.Truncate(r.Embedding, f.Dim); err == nil {
					rec.Modality, rec.Tokens, rec.Dim, rec.Embedding = r.Modality, r.Tokens, f.Dim, vec
					if j.in.Task != eg2.TaskNone {
						rec.Task = string(j.in.Task)
					}
				}
			}
		}
		if err != nil {
			failed++
			rec = record{ID: rec.ID, Source: j.source, Error: err.Error()}
		}
		return w.write(rec)
	}
	// With --batch, text-only jobs wait for each other up to the batch size
	// and are embedded in one Embed call; anything else flushes them first.
	var pending []job
	flush := func() error {
		if len(pending) == 0 {
			return nil
		}
		ins := make([]eg2.Input, len(pending))
		for i, j := range pending {
			ins[i] = j.in
		}
		rs, err := e.Embed(ctx, ins...)
		if err != nil {
			return err
		}
		for i, j := range pending {
			if err := write(j, rs[i], nil); err != nil {
				return err
			}
		}
		pending = pending[:0]
		return nil
	}
	for j := range jobs {
		total++
		if j.err == nil && f.Common.Batch > 1 && len(j.in.Media) == 0 {
			pending = append(pending, j)
			if len(pending) < f.Common.Batch {
				continue
			}
		}
		if err := flush(); err != nil {
			return err // ctx done or encoders unavailable: nothing else will work
		}
		if j.err != nil {
			if err := write(j, eg2.Result{}, j.err); err != nil {
				return err
			}
			continue
		}
		if f.Common.Batch > 1 && len(j.in.Media) == 0 {
			continue // it was in the batch just flushed
		}
		rs, err := e.Embed(ctx, j.in)
		if err != nil {
			return err
		}
		if err := write(j, rs[0], nil); err != nil {
			return err
		}
	}
	if err := flush(); err != nil {
		return err
	}
	if err := w.close(); err != nil {
		return err
	}
	if failed > 0 {
		fmt.Fprintf(stderr, "eg2: %d of %d inputs failed\n", failed, total)
		return errSomeFailed
	}
	return nil
}

func (f *embedCmd) validate(nFiles int) error {
	if err := eg2.CheckDim(f.Dim); err != nil {
		return fmt.Errorf("%w: --dim: %v", errUsage, err)
	}
	if _, err := eg2.Task(f.Task).Prompt(); err != nil {
		return fmt.Errorf("%w: --task: %v", errUsage, err)
	}
	if len(f.parts) == 0 && nFiles == 0 && f.Input == "" {
		return fmt.Errorf("%w: nothing to embed: give FILEs, --text/--image/--audio/--video, or --input", errUsage)
	}
	if f.Input == "-" {
		for _, p := range f.parts {
			if p.kind == "text" && p.value == "-" {
				return fmt.Errorf("%w: --text - and --input - both read stdin", errUsage)
			}
		}
	}
	return nil
}

// defaults applies the prefix flags to an input that has text and no prefix
// of its own. Media alone get none, as the model card advises.
func (f *embedCmd) defaults(in *eg2.Input) {
	if !in.HasText() {
		return
	}
	if in.Task == eg2.TaskNone && in.Prompt == "" && in.Title == "" {
		in.Task, in.Title, in.Prompt = eg2.Task(f.Task), f.Title, f.Prompt
	}
}

// jobs yields the inputs in order: the combined input from flags, then the
// FILE arguments, then the -input lines. Reading errors become failed jobs so
// one bad file does not stop the rest.
func (f *embedCmd) jobs(files []string, stdin io.Reader) (func(yield func(job) bool), error) {
	var combined *eg2.Input
	if len(f.parts) > 0 {
		in, err := f.combinedInput(stdin)
		if err != nil {
			return nil, err
		}
		combined = &in
	}
	var lines io.Reader
	if f.Input == "-" {
		lines = stdin
	} else if f.Input != "" {
		file, err := os.Open(f.Input)
		if err != nil {
			return nil, err
		}
		lines = file // closed when the process exits
	}

	return func(yield func(job) bool) {
		if combined != nil && !yield(job{in: *combined, source: "args"}) {
			return
		}
		for _, path := range files {
			in, err := fileInput(path)
			if err == nil {
				f.defaults(&in)
				in.Literal = in.Literal || f.Literal && in.Text != ""
				if in.Task == eg2.TaskDocument && in.Title == "" && in.Text != "" {
					in.Title = filepath.Base(path)
				}
			}
			if !yield(job{in: in, source: path, err: err}) {
				return
			}
		}
		if lines == nil {
			return
		}
		sc := bufio.NewScanner(lines)
		sc.Buffer(make([]byte, 0, 1<<20), 512<<20) // base64 media make long lines
		n := 0
		for sc.Scan() {
			n++
			line := bytes.TrimSpace(sc.Bytes())
			if len(line) == 0 {
				continue
			}
			source := fmt.Sprintf("%s:%d", inputName(f.Input), n)
			in, err := eg2.ParseInput(line, eg2.ParseOptions{AllowPaths: true})
			if err == nil {
				f.defaults(&in)
			}
			if !yield(job{in: in, source: source, err: err}) {
				return
			}
		}
		if err := sc.Err(); err != nil {
			yield(job{source: inputName(f.Input), err: err})
		}
	}, nil
}

func inputName(input string) string {
	if input == "-" {
		return "stdin"
	}
	return input
}

// combinedInput builds the input of --text and the media options. Without
// placeholders the flags' order is the order in the vector, as JSON key order
// is for ParseInput.
func (f *embedCmd) combinedInput(stdin io.Reader) (in eg2.Input, err error) {
	in = eg2.Input{ID: "args", Literal: f.Literal}
	defer f.defaults(&in) // after the text is known; in is a named result
	hasPlaceholders := false
	for i, p := range f.parts {
		switch p.kind {
		case "text":
			if p.value == "-" {
				b, err := io.ReadAll(stdin)
				if err != nil {
					return in, err
				}
				f.parts[i].value = strings.TrimRight(string(b), "\n")
			}
			in.Text = f.parts[i].value
			hasPlaceholders = !f.Literal && eg2.HasPlaceholders(in.Text)
		default:
			kind := map[string]eg2.MediaKind{"image": eg2.MediaImage, "audio": eg2.MediaAudio, "video": eg2.MediaVideo}[p.kind]
			in.Media = append(in.Media, eg2.Media{Kind: kind, Path: p.value})
		}
	}
	if hasPlaceholders || f.Literal {
		return in, nil
	}
	var b strings.Builder
	for _, p := range f.parts {
		if p.kind == "text" {
			b.WriteString(p.value)
		} else {
			b.WriteString("<|" + p.kind + "|>")
		}
	}
	in.Text = b.String()
	return in, nil
}

// fileInput reads a FILE argument: an image, a text file, or other media
// whose kind (video or audio) the library decides.
func fileInput(path string) (eg2.Input, error) {
	fi, err := os.Stat(path)
	if err != nil {
		return eg2.Input{}, err
	}
	if fi.IsDir() {
		return eg2.Input{}, errors.New("is a directory")
	}
	if isImageFile(path) {
		return eg2.Input{Media: []eg2.Media{{Kind: eg2.MediaImage, Path: path}}}, nil
	}
	if fi.Size() <= maxTextFile {
		b, err := os.ReadFile(path)
		if err != nil {
			return eg2.Input{}, err
		}
		if isText(b) {
			return eg2.Input{Text: string(b)}, nil
		}
	}
	return eg2.Input{Media: []eg2.Media{{Path: path}}}, nil
}

func isImageFile(path string) bool {
	f, err := os.Open(path)
	if err != nil {
		return false
	}
	defer f.Close()
	_, _, err = image.DecodeConfig(f)
	return err == nil
}

// isText reports whether b reads as text: valid UTF-8 without NUL bytes,
// which every audio and video container has.
func isText(b []byte) bool {
	return len(b) > 0 && utf8.Valid(b) && !bytes.ContainsRune(b, 0)
}

func taskList() string {
	names := make([]string, 0, len(eg2.Tasks()))
	for _, t := range eg2.Tasks() {
		names = append(names, string(t))
	}
	return strings.Join(names, ", ")
}

// shortModel turns "repo@revision" into "repo@rev8".
func shortModel(m string) string {
	repo, rev, ok := strings.Cut(m, "@")
	if ok && len(rev) > 8 {
		return repo + "@" + rev[:8]
	}
	return m
}
