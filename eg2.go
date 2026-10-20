// Package eg2 embeds text, images, audio and video, alone or combined, with
// Google's EmbeddingGemma 2. It runs llama.cpp loaded at run time (no cgo) and
// downloads the pinned model and libraries into a cache directory on first use.
//
// Image and video preprocessing follows the reference transformers
// implementation, so vectors are comparable with the ones it produces.
package eg2

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"image"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/meinside/embedding-gemma-2-runner/internal/assets"
	"github.com/meinside/embedding-gemma-2-runner/internal/engine"
	"github.com/meinside/embedding-gemma-2-runner/internal/media"
)

// Dims is the full embedding size. Truncate makes the shorter Matryoshka sizes.
const Dims = 768

// MaxTokens is the model's context: the most tokens one input may have.
const MaxTokens = 8192

// Device selects where the model runs.
type Device int

const (
	DeviceAuto Device = iota // the GPU if one initializes, else the CPU
	DeviceCPU
	DeviceGPU
)

// Options configure Open. The zero value downloads what is missing into the
// default cache directory and picks the device automatically.
type Options struct {
	// CacheDir holds models and libraries. Empty means $XDG_CACHE_HOME/eg2,
	// or eg2 under the OS cache directory.
	CacheDir string
	Device   Device
	// Processor names the llama.cpp build ("metal", "cuda-13", "vulkan",
	// "cpu", ...). Empty picks one for this machine.
	Processor string
	// LibDir uses llama.cpp libraries from this directory instead of the cache.
	LibDir string
	// Weights is "q8_0" (default) or "bf16", for parity checks.
	Weights string
	// SeedDir copies model files from this directory instead of downloading.
	SeedDir string
	Offline bool // fail rather than download
	Threads int  // CPU threads; 0 lets llama.cpp choose
	// MaxTokens is the most tokens one input may have, up to the model's
	// 8192 (the default). A smaller value makes the context, and so the
	// memory the model takes, smaller; longer inputs fail.
	MaxTokens int
	// BatchSeqs lets one Embed call put up to this many consecutive
	// text-only inputs in one forward pass: 2.5 times faster on an M1 GPU
	// for short texts, 1.8 on its CPU. 1 embeds them one at a time. 0, the
	// default, means 8 when the model runs on a GPU and 1 on the CPU: on the
	// GPU the vectors are the same either way, while the CPU kernels round
	// differently for other batch shapes, within about 3e-4 cosine, so a CPU
	// index should be built one way or the other, not both. Info.BatchSeqs
	// tells which applied.
	BatchSeqs int
	Verbose   bool // let llama.cpp log to stderr
	// Progress, if set, reports downloads.
	Progress func(name string, done, total int64)
}

// Embedder embeds inputs. It is safe for concurrent use; calls run one at a time.
type Embedder struct {
	mu     sync.Mutex
	eng    *engine.Engine
	opts   Options
	dirs   assets.Dirs
	libDir string
	proc   assets.Processor
}

// Result is the embedding of one Input.
type Result struct {
	ID        string
	Embedding []float32 // Dims values, unit length
	Tokens    int
	Modality  string // "text", "image", "audio", "video" or "mixed"
	Err       error  // set when this input failed; Embedding is nil then
}

// Info describes an open Embedder.
type Info struct {
	Model        string // repository@revision
	Weights      string
	Device       string // "CPU" or the GPU's description
	MediaDevice  string // the same for the vision and audio encoders; "" until they are loaded
	BatchSeqs    int    // texts per forward pass (Options.BatchSeqs resolved for the device)
	LlamaVersion string
	Processor    string
	LibDir       string
	Dirs         Dirs
	MediaLoaded  bool // vision and audio encoders loaded
	FFmpeg       bool // video available
}

// Dirs are the cache locations.
type Dirs = assets.Dirs

// CacheDirs returns where models and libraries are cached for cacheDir, or for
// the default when it is empty.
func CacheDirs(cacheDir string) (Dirs, error) { return assets.CacheDirs(cacheDir) }

// Open fetches what is missing and loads the text model. The vision and audio
// encoders are fetched and loaded on the first input with media.
func Open(ctx context.Context, o Options) (*Embedder, error) {
	dirs, err := assets.CacheDirs(o.CacheDir)
	if err != nil {
		return nil, err
	}
	if o.MaxTokens < 0 || o.MaxTokens > MaxTokens {
		return nil, fmt.Errorf("MaxTokens %d: want 1 to %d", o.MaxTokens, MaxTokens)
	}
	e := &Embedder{opts: o, dirs: dirs, libDir: o.LibDir, proc: assets.Processor(o.Processor)}
	if e.libDir == "" {
		lo := assets.LibOptions{Processor: e.proc, Offline: o.Offline, Progress: o.Progress}
		if lo.Processor == "" {
			lo.Processor = assets.DefaultProcessor(o.Device != DeviceCPU)
		}
		if e.libDir, e.proc, err = assets.Libraries(ctx, dirs, lo); err != nil {
			return nil, err
		}
	}
	textModel, err := assets.TextModel(ctx, dirs, e.modelOptions())
	if err != nil {
		return nil, err
	}
	e.eng, err = engine.Open(engine.Options{
		LibDir:    e.libDir,
		TextModel: textModel,
		Device:    map[Device]engine.Device{DeviceAuto: engine.Auto, DeviceCPU: engine.CPU, DeviceGPU: engine.GPU}[o.Device],
		Threads:   o.Threads,
		MaxTokens: o.MaxTokens,
		NSeqMax:   o.BatchSeqs,
		Quiet:     !o.Verbose,
	})
	if err != nil {
		return nil, err
	}
	return e, nil
}

func (e *Embedder) modelOptions() assets.ModelOptions {
	return assets.ModelOptions{
		Weights:  assets.Weights(e.opts.Weights),
		Offline:  e.opts.Offline,
		SeedDir:  e.opts.SeedDir,
		Progress: e.opts.Progress,
	}
}

// Close frees the model.
func (e *Embedder) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.eng != nil {
		e.eng.Close()
		e.eng = nil
	}
	return nil
}

// Info describes the Embedder.
func (e *Embedder) Info() Info {
	e.mu.Lock()
	defer e.mu.Unlock()
	w := e.opts.Weights
	if w == "" {
		w = string(assets.Q8_0)
	}
	info := Info{
		Model:        assets.ModelRepo + "@" + assets.ModelRevision,
		Weights:      w,
		LlamaVersion: assets.LlamaVersion,
		Processor:    string(e.proc),
		LibDir:       e.libDir,
		Dirs:         e.dirs,
		FFmpeg:       media.FFmpegAvailable(),
	}
	if e.eng != nil {
		info.Device = e.eng.Device()
		info.MediaDevice = e.eng.MediaDevice()
		info.BatchSeqs = e.eng.NSeqMax()
		info.MediaLoaded = e.eng.HasMMProj()
	}
	return info
}

// Embed embeds each input. A failing input gets Result.Err and the others go
// on; the returned error is for failures that stop everything (ctx done, a
// closed Embedder, encoders that cannot be fetched).
func (e *Embedder) Embed(ctx context.Context, inputs ...Input) ([]Result, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.eng == nil {
		return nil, errors.New("embedder is closed")
	}
	out := make([]Result, len(inputs))
	for i := 0; i < len(inputs); {
		if err := ctx.Err(); err != nil {
			return out[:i], err
		}
		in := inputs[i]
		out[i] = Result{ID: in.ID}
		if e.eng.NSeqMax() > 1 && len(in.Media) == 0 {
			j := i + 1
			for j < len(inputs) && len(inputs[j].Media) == 0 {
				out[j] = Result{ID: inputs[j].ID}
				j++
			}
			e.embedTextRun(inputs[i:j], out[i:j])
			i = j
			continue
		}
		if len(in.Media) > 0 {
			if err := e.loadMedia(ctx); err != nil {
				return out[:i], err
			}
		}
		vec, tokens, mod, err := e.embedOne(ctx, in)
		if err != nil {
			out[i].Err = err
		} else {
			out[i].Embedding, out[i].Tokens, out[i].Modality = vec, tokens, mod
		}
		i++
	}
	return out, nil
}

// embedTextRun embeds text-only inputs together (Options.BatchSeqs).
func (e *Embedder) embedTextRun(inputs []Input, out []Result) {
	var texts []engine.TextInput
	var idx []int // input of each text
	for i, in := range inputs {
		prefix, err := in.prefix()
		if err == nil {
			_, err = in.segments(nil)
		}
		if err != nil {
			out[i].Err = err
			continue
		}
		texts = append(texts, engine.TextInput{Text: prefix + in.Text, ParseSpecial: !in.Literal})
		idx = append(idx, i)
	}
	embs, errs := e.eng.EmbedTexts(texts)
	for k, i := range idx {
		if errs[k] != nil {
			out[i].Err = errs[k]
			continue
		}
		vec, err := normalize(embs[k].Vec)
		if err != nil {
			out[i].Err = err
			continue
		}
		out[i].Embedding, out[i].Tokens, out[i].Modality = vec, embs[k].Tokens, "text"
	}
}

// LoadMedia fetches and loads the vision and audio encoders now instead of
// on the first input with media.
func (e *Embedder) LoadMedia(ctx context.Context) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.eng == nil {
		return errors.New("embedder is closed")
	}
	return e.loadMedia(ctx)
}

func (e *Embedder) loadMedia(ctx context.Context) error {
	if e.eng.HasMMProj() {
		return nil
	}
	path, err := assets.MMProj(ctx, e.dirs, e.modelOptions())
	if err != nil {
		return err
	}
	return e.eng.LoadMMProj(path)
}

func (e *Embedder) embedOne(ctx context.Context, in Input) ([]float32, int, string, error) {
	prefix, err := in.prefix()
	if err != nil {
		return nil, 0, "", err
	}
	kinds := make([]MediaKind, len(in.Media))
	for i, m := range in.Media {
		if kinds[i], err = resolveKind(ctx, m); err != nil {
			return nil, 0, "", fmt.Errorf("media %d: %w", i+1, err)
		}
	}
	segs, err := in.segments(kinds)
	if err != nil {
		return nil, 0, "", err
	}
	mod := modality(segs, kinds)

	if len(in.Media) == 0 {
		emb, err := e.eng.EmbedText(prefix+in.Text, !in.Literal)
		if err != nil {
			return nil, 0, "", err
		}
		vec, err := normalize(emb.Vec)
		return vec, emb.Tokens, mod, err
	}

	// Text joins media without separators, the prefix first, as the reference
	// chat template renders them.
	var text strings.Builder
	text.WriteString(prefix)
	var parts []engine.Part
	for _, s := range segs {
		if s.media < 0 {
			if strings.Contains(s.text, engine.Marker) {
				return nil, 0, "", fmt.Errorf("text contains the reserved string %s", engine.Marker)
			}
			text.WriteString(s.text)
			continue
		}
		ps, err := mediaParts(ctx, in.Media[s.media], kinds[s.media])
		if err != nil {
			return nil, 0, "", fmt.Errorf("media %d: %w", s.media+1, err)
		}
		for range ps {
			text.WriteString(engine.Marker)
		}
		parts = append(parts, ps...)
	}
	emb, err := e.eng.EmbedMixed(text.String(), parts, !in.Literal)
	if err != nil {
		return nil, 0, "", err
	}
	vec, err := normalize(emb.Vec)
	return vec, emb.Tokens, mod, err
}

// audioHead is how much of an audio file MediaDuration reads: room for an
// ID3 tag with cover art before the first MP3 frame.
const audioHead = 1 << 20

// MediaDuration reads how long audio or video plays without decoding it:
// audio from its headers (WAV, FLAC, MP3), video from ffprobe. ok is false
// when it cannot tell: other audio formats, or video without ffmpeg. A
// server uses it to refuse long media before mtmd decodes them whole.
func MediaDuration(ctx context.Context, m Media) (d time.Duration, ok bool, err error) {
	kind, err := resolveKind(ctx, m)
	if err != nil {
		return 0, false, err
	}
	switch kind {
	case MediaAudio:
		head, size, err := mediaHead(m, audioHead)
		if err != nil {
			return 0, false, err
		}
		d, ok := media.AudioDuration(head, size)
		return d, ok, nil
	case MediaVideo:
		if !media.FFmpegAvailable() {
			return 0, false, nil
		}
		path, cleanup, err := mediaFile(m)
		if err != nil {
			return 0, false, err
		}
		defer cleanup()
		meta, ok, err := media.ProbeVideo(ctx, path)
		if err != nil || !ok {
			return 0, false, err
		}
		return time.Duration(meta.Duration * float64(time.Second)), true, nil
	}
	return 0, false, nil
}

// mediaHead returns the first n bytes of m and its full size.
func mediaHead(m Media, n int64) ([]byte, int64, error) {
	if m.Data != nil {
		return m.Data, int64(len(m.Data)), nil
	}
	f, err := os.Open(m.Path)
	if err != nil {
		return nil, 0, err
	}
	defer f.Close()
	fi, err := f.Stat()
	if err != nil {
		return nil, 0, err
	}
	head, err := io.ReadAll(io.LimitReader(f, n))
	return head, fi.Size(), err
}

// videoExts are file extensions of video containers, used only to word the
// error when such a file cannot be read as video.
var videoExts = map[string]bool{".mp4": true, ".m4v": true, ".mov": true, ".mkv": true, ".webm": true,
	".avi": true, ".mpg": true, ".mpeg": true, ".wmv": true, ".flv": true, ".3gp": true}

// resolveKind decides MediaAuto from the content: a decodable image, else a
// video ffprobe recognizes, else audio.
func resolveKind(ctx context.Context, m Media) (MediaKind, error) {
	if m.Kind != MediaAuto {
		return m.Kind, nil
	}
	if (m.Path == "") == (m.Data == nil) {
		return 0, errors.New("media needs exactly one of a path or data")
	}
	if isImage(m) {
		return MediaImage, nil
	}
	looksVideo := videoExts[strings.ToLower(filepath.Ext(m.Path))]
	if !media.FFmpegAvailable() {
		if looksVideo {
			return 0, errors.New("video needs ffmpeg and ffprobe in PATH")
		}
		return MediaAudio, nil
	}
	path, cleanup, err := mediaFile(m)
	if err != nil {
		return 0, err
	}
	_, ok, err := media.ProbeVideo(ctx, path)
	cleanup()
	if err != nil {
		return 0, err
	}
	if ok {
		return MediaVideo, nil
	}
	if looksVideo {
		// Otherwise mtmd would report it as unsupported audio.
		return 0, errors.New("not a video ffprobe can read")
	}
	return MediaAudio, nil
}

func isImage(m Media) bool {
	if m.Data != nil {
		_, _, err := image.DecodeConfig(bytes.NewReader(m.Data))
		return err == nil
	}
	f, err := os.Open(m.Path)
	if err != nil {
		return false
	}
	defer f.Close()
	_, _, err = image.DecodeConfig(f)
	return err == nil
}

// mediaFile returns a path for m, writing Data to a temporary file that
// cleanup removes: ffmpeg needs to seek, which a pipe does not allow for MP4
// files that keep their index at the end.
func mediaFile(m Media) (string, func(), error) {
	if m.Data == nil {
		return m.Path, func() {}, nil
	}
	f, err := os.CreateTemp("", "eg2-media-*")
	if err != nil {
		return "", nil, err
	}
	if _, err := f.Write(m.Data); err != nil {
		f.Close()
		os.Remove(f.Name())
		return "", nil, err
	}
	if err := f.Close(); err != nil {
		os.Remove(f.Name())
		return "", nil, err
	}
	return f.Name(), func() { os.Remove(f.Name()) }, nil
}

// mediaParts prepares one media item: one part for an image or audio, one per
// sampled frame for a video.
func mediaParts(ctx context.Context, m Media, kind MediaKind) ([]engine.Part, error) {
	switch kind {
	case MediaImage:
		var (
			f   media.Frame
			ok  bool
			err error
		)
		if m.Data != nil {
			f, ok, err = media.DecodeImage(m.Data)
		} else {
			f, ok, err = media.LoadImage(m.Path)
		}
		if err != nil {
			return nil, err
		}
		if !ok {
			return nil, errors.New("not a supported image (JPEG, PNG, GIF, WebP)")
		}
		return []engine.Part{{Image: &f}}, nil
	case MediaAudio:
		return []engine.Part{{AudioPath: m.Path, AudioData: m.Data}}, nil
	case MediaVideo:
		if !media.FFmpegAvailable() {
			return nil, errors.New("video needs ffmpeg and ffprobe in PATH")
		}
		path, cleanup, err := mediaFile(m)
		if err != nil {
			return nil, err
		}
		defer cleanup()
		meta, ok, err := media.ProbeVideo(ctx, path)
		if err != nil {
			return nil, err
		}
		if !ok {
			return nil, errors.New("not a video ffprobe can read")
		}
		frames, err := media.VideoFrames(ctx, path, meta)
		if err != nil {
			return nil, err
		}
		parts := make([]engine.Part, len(frames))
		for i := range frames {
			parts[i] = engine.Part{Image: &frames[i]}
		}
		return parts, nil
	}
	return nil, fmt.Errorf("unknown media kind %d", kind)
}

// Truncate shortens v to dim values (Matryoshka) and makes it unit length
// again, which slicing alone does not. dim must be 768, 512, 256 or 128, the
// sizes the model was trained for; queries and documents must use the same.
func Truncate(v []float32, dim int) ([]float32, error) {
	if err := CheckDim(dim); err != nil {
		return nil, err
	}
	if len(v) < dim {
		return nil, fmt.Errorf("vector has %d values, fewer than %d", len(v), dim)
	}
	return normalize(v[:dim])
}

// CheckDim reports whether dim is a size Truncate accepts.
func CheckDim(dim int) error {
	switch dim {
	case 768, 512, 256, 128:
		return nil
	}
	return fmt.Errorf("dimension %d: want 768, 512, 256 or 128", dim)
}

func normalize(v []float32) ([]float32, error) {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if sum == 0 || math.IsNaN(sum) || math.IsInf(sum, 0) {
		return nil, errors.New("embedding cannot be normalized")
	}
	inv := 1 / math.Sqrt(sum)
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = float32(float64(x) * inv)
	}
	return out, nil
}
