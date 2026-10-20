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

type partFlag struct {
	kind  string
	parts *[]flagPart
}

func (f partFlag) String() string { return "" }

func (f partFlag) Set(v string) error {
	if f.kind == "text" {
		for _, p := range *f.parts {
			if p.kind == "text" {
				return errors.New("--text given twice; use placeholders to place media in one text")
			}
		}
	}
	*f.parts = append(*f.parts, flagPart{f.kind, v})
	return nil
}

type embedFlags struct {
	common
	parts   []flagPart
	task    string
	title   string
	prompt  string
	literal bool
	input   string
	dim     int
	format  string
}

// job is one input and where it came from, for the output record.
type job struct {
	in     eg2.Input
	source string
	err    error // the input could not even be read
}

func runEmbed(ctx context.Context, args []string, stdin io.Reader, stdout, stderr io.Writer) error {
	var f embedFlags
	fs := newFlagSet("embed", "embed [flags] [FILE...]\n\n"+
		"Each FILE is one input: an image, audio or video file, or a text file.\n"+
		"--text and the media flags together make one more input.\n"+
		"See \"eg2 help inputs\" for placeholders and tasks.", stderr, &f.common)
	f.register(fs)
	fs.Var(partFlag{"text", &f.parts}, "text", "`text` of the combined input (\"-\" reads stdin)")
	fs.Var(partFlag{"image", &f.parts}, "image", "image `file` for the combined input (repeatable)")
	fs.Var(partFlag{"audio", &f.parts}, "audio", "audio `file` for the combined input (repeatable)")
	fs.Var(partFlag{"video", &f.parts}, "video", "video `file` for the combined input (repeatable)")
	fs.StringVar(&f.task, "task", "", "task prefix: "+taskList()+" (default: none)")
	fs.StringVar(&f.title, "title", "", "document title, with -task document")
	fs.StringVar(&f.prompt, "prompt", "", "custom prefix instead of -task")
	fs.BoolVar(&f.literal, "literal", false, "keep placeholders and special-token strings in text as written")
	fs.StringVar(&f.input, "input", "", "read JSON inputs, one per line, from `file` (\"-\": stdin)")
	fs.IntVar(&f.dim, "dim", eg2.Dims, "dimensions: 768, 512, 256 or 128")
	fs.StringVar(&f.format, "format", "", "json or jsonl (default: json for one input, jsonl for more)")
	if err := parseFlags(fs, args); err != nil {
		return err
	}
	if err := f.validate(fs.NArg()); err != nil {
		return err
	}

	jobs, err := f.jobs(fs.Args(), stdin)
	if err != nil {
		return err
	}
	format := f.format
	if format == "" {
		format = "json"
		if f.input != "" || len(f.parts) > 0 && fs.NArg() > 0 || fs.NArg() > 1 {
			format = "jsonl"
		}
	}

	opts, err := f.options(stderr)
	if err != nil {
		return err
	}
	e, err := eg2.Open(ctx, opts)
	if err != nil {
		return err
	}
	defer e.Close()
	info := e.Info()
	model := fmt.Sprintf("%s/%s", shortModel(info.Model), info.Weights)

	w := newRecordWriter(stdout, format)
	failed, total := 0, 0
	for j := range jobs {
		total++
		rec := record{ID: j.in.ID, Source: j.source, Model: model}
		if rec.ID == "" {
			rec.ID = j.source
		}
		err := j.err
		if err == nil {
			var rs []eg2.Result
			if rs, err = e.Embed(ctx, j.in); err != nil {
				return err // ctx done or encoders unavailable: nothing else will work
			}
			r := rs[0]
			if err = r.Err; err == nil {
				var vec []float32
				if vec, err = eg2.Truncate(r.Embedding, f.dim); err == nil {
					rec.Modality, rec.Tokens, rec.Dim, rec.Embedding = r.Modality, r.Tokens, f.dim, vec
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
		if err := w.write(rec); err != nil {
			return err
		}
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

func (f *embedFlags) validate(nFiles int) error {
	if err := eg2.CheckDim(f.dim); err != nil {
		return fmt.Errorf("%w: -dim: %v", errUsage, err)
	}
	if _, err := eg2.Task(f.task).Prompt(); err != nil {
		return fmt.Errorf("%w: -task: %v", errUsage, err)
	}
	if f.format != "" && f.format != "json" && f.format != "jsonl" {
		return fmt.Errorf("%w: -format %q: want json or jsonl", errUsage, f.format)
	}
	if len(f.parts) == 0 && nFiles == 0 && f.input == "" {
		return fmt.Errorf("%w: nothing to embed: give FILEs, --text/--image/--audio/--video, or --input", errUsage)
	}
	if f.input == "-" {
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
func (f *embedFlags) defaults(in *eg2.Input) {
	if !in.HasText() {
		return
	}
	if in.Task == eg2.TaskNone && in.Prompt == "" && in.Title == "" {
		in.Task, in.Title, in.Prompt = eg2.Task(f.task), f.title, f.prompt
	}
}

// jobs yields the inputs in order: the combined input from flags, then the
// FILE arguments, then the --input lines. Reading errors become failed jobs so
// one bad file does not stop the rest.
func (f *embedFlags) jobs(files []string, stdin io.Reader) (func(yield func(job) bool), error) {
	var combined *eg2.Input
	if len(f.parts) > 0 {
		in, err := f.combinedInput(stdin)
		if err != nil {
			return nil, err
		}
		combined = &in
	}
	var lines io.Reader
	if f.input == "-" {
		lines = stdin
	} else if f.input != "" {
		file, err := os.Open(f.input)
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
				in.Literal = in.Literal || f.literal && in.Text != ""
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
			source := fmt.Sprintf("%s:%d", inputName(f.input), n)
			in, err := eg2.ParseInput(line, eg2.ParseOptions{AllowPaths: true})
			if err == nil {
				f.defaults(&in)
			}
			if !yield(job{in: in, source: source, err: err}) {
				return
			}
		}
		if err := sc.Err(); err != nil {
			yield(job{source: inputName(f.input), err: err})
		}
	}, nil
}

func inputName(input string) string {
	if input == "-" {
		return "stdin"
	}
	return input
}

// combinedInput builds the input of --text and the media flags. Without
// placeholders the flags' order is the order in the vector, as JSON key order
// is for ParseInput.
func (f *embedFlags) combinedInput(stdin io.Reader) (in eg2.Input, err error) {
	in = eg2.Input{ID: "args", Literal: f.literal}
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
			hasPlaceholders = !f.literal && eg2.HasPlaceholders(in.Text)
		default:
			kind := map[string]eg2.MediaKind{"image": eg2.MediaImage, "audio": eg2.MediaAudio, "video": eg2.MediaVideo}[p.kind]
			in.Media = append(in.Media, eg2.Media{Kind: kind, Path: p.value})
		}
	}
	if hasPlaceholders || f.literal {
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
