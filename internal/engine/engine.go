// Package engine runs EmbeddingGemma 2 through llama.cpp and its multimodal
// library mtmd, loaded at run time with yzma (no cgo).
//
// An Engine is not safe for concurrent use.
package engine

import (
	"errors"
	"fmt"
	"math"
	"runtime"
	"strings"
	"sync"

	"github.com/hybridgroup/yzma/pkg/llama"
	"github.com/hybridgroup/yzma/pkg/mtmd"

	"github.com/meinside/embedding-gemma-2-runner/internal/media"
)

// Device selects where the model runs.
type Device int

const (
	Auto Device = iota // GPU when one initializes, else CPU
	CPU
	GPU
)

// Marker stands for one media item in text passed to EmbedMixed. It replaces
// mtmd's default "<__media__>" so ordinary text is unlikely to contain it.
const Marker = "<__eg2_media__>"

// Options configure Open.
type Options struct {
	LibDir    string // llama.cpp shared libraries
	TextModel string // GGUF path
	Device    Device
	Threads   int // 0: llama.cpp default
	MaxTokens int // tokens per input, at most the model's 8192
	Quiet     bool
}

type Engine struct {
	model     llama.Model
	lctx      llama.Context
	mctx      mtmd.Context
	gpu       bool
	gpuName   string
	threads   int
	nEmbdInp  int32
	nEmbdOut  int32
	maxTokens int
}

// Embedding is one pooled, unnormalized output vector.
type Embedding struct {
	Vec    []float32
	Tokens int
}

// Part is one media item for EmbedMixed: an image or video frame already
// resized by package media, or audio for mtmd to decode (wav, mp3, flac).
type Part struct {
	Image     *media.Frame
	AudioPath string
	AudioData []byte
}

var (
	loadOnce sync.Once
	loadErr  error
)

// loadLibraries loads and initializes llama.cpp once per process: yzma keeps
// the libraries in package state, so a second directory cannot be loaded.
func loadLibraries(dir string, silent bool) error {
	loadOnce.Do(func() {
		if loadErr = llama.Load(dir); loadErr != nil {
			loadErr = fmt.Errorf("load llama.cpp from %s: %w", dir, loadErr)
			return
		}
		if loadErr = mtmd.Load(dir); loadErr != nil {
			loadErr = fmt.Errorf("load mtmd from %s: %w", dir, loadErr)
			return
		}
		// must follow Load: yzma calls through a nil function pointer before it
		if silent {
			llama.LogSet(llama.LogSilent())
			mtmd.LogSet(llama.LogSilent())
		}
		if err := llama.Init(); err != nil {
			loadErr = fmt.Errorf("init llama.cpp backends: %w", err)
		}
	})
	return loadErr
}

// gpuDevice returns the description of the first GPU llama.cpp found, or "".
func gpuDevice() string {
	for i := range llama.GGMLBackendDeviceCount() {
		dev := llama.GGMLBackendDeviceGet(i)
		switch llama.GGMLBackendDevType(dev) {
		case llama.GGMLBackendDeviceTypeGPU, llama.GGMLBackendDeviceTypeIGPU:
			return llama.GGMLBackendDeviceDescription(dev)
		}
	}
	return ""
}

// Open loads the text model. With Device Auto it tries the GPU first and falls
// back to the CPU when the GPU backend fails to initialize, as Metal does in a
// macOS sandbox.
func Open(o Options) (*Engine, error) {
	if err := loadLibraries(o.LibDir, o.Quiet); err != nil {
		return nil, err
	}
	if o.MaxTokens <= 0 {
		o.MaxTokens = 8192
	}
	gpuName := gpuDevice()
	if o.Device == GPU && gpuName == "" {
		return nil, errors.New("no GPU backend found by llama.cpp")
	}
	if o.Device != CPU && gpuName != "" {
		e, err := open(o, true)
		if err == nil {
			e.gpuName = gpuName
			return e, nil
		}
		if o.Device == GPU {
			return nil, err
		}
	}
	return open(o, false)
}

func open(o Options, gpu bool) (*Engine, error) {
	mp := llama.ModelDefaultParams()
	// An empty, NULL-terminated device list keeps llama.cpp off every
	// accelerator. NGpuLayers=0 alone (yzma's SetCPUOnly) still creates a
	// Metal/CUDA backend for the context, which fails where the GPU is unusable.
	noDevices := []llama.GGMLBackendDevice{0}
	if !gpu {
		mp.NGpuLayers = 0
		if err := mp.SetDevices(noDevices); err != nil {
			return nil, err
		}
	}
	model, err := llama.ModelLoadFromFile(o.TextModel, mp)
	runtime.KeepAlive(noDevices)
	if err != nil {
		return nil, fmt.Errorf("load %s: %w", o.TextModel, err)
	}
	e := &Engine{
		model: model, gpu: gpu, threads: o.Threads, maxTokens: o.MaxTokens,
		nEmbdInp: llama.ModelNEmbdInp(model), nEmbdOut: llama.ModelNEmbdOut(model),
	}

	cp := llama.ContextDefaultParams()
	n := uint32(o.MaxTokens)
	// The encoder needs the whole input in one physical batch (n_ubatch).
	cp.NCtx, cp.NBatch, cp.NUbatch = n, n, n
	cp.Embeddings = 1
	cp.PoolingType = llama.PoolingTypeMean
	if o.Threads > 0 {
		cp.NThreads, cp.NThreadsBatch = int32(o.Threads), int32(o.Threads)
	}
	if !gpu {
		cp.OpOffload, cp.Offload_kqv = 0, 0
	}
	if e.lctx, err = llama.InitFromModel(model, cp); err != nil {
		e.Close()
		return nil, fmt.Errorf("init context: %w", err)
	}
	return e, nil
}

// LoadMMProj loads the vision and audio encoders. On a GPU engine it tries the
// GPU first, then the CPU.
func (e *Engine) LoadMMProj(path string) error {
	if e.mctx != 0 {
		return nil
	}
	try := func(gpu bool) error {
		p := mtmd.ContextParamsDefault()
		p.UseGPU = gpu
		if e.threads > 0 {
			p.Threads = int32(e.threads)
		}
		marker := []byte(Marker + "\x00")
		p.MediaMarker = &marker[0]
		ctx, err := mtmd.InitFromFile(path, e.model, p)
		runtime.KeepAlive(marker)
		if err != nil {
			return fmt.Errorf("load %s: %w", path, err)
		}
		e.mctx = ctx
		return nil
	}
	if e.gpu {
		if err := try(true); err == nil {
			return nil
		}
	}
	return try(false)
}

// HasMMProj reports whether LoadMMProj succeeded.
func (e *Engine) HasMMProj() bool { return e.mctx != 0 }

// Device describes where the text model runs: the GPU's description or "CPU".
func (e *Engine) Device() string {
	if e.gpu {
		return e.gpuName
	}
	return "CPU"
}

// Close frees the model. The libraries stay loaded for the process.
func (e *Engine) Close() {
	if e.mctx != 0 {
		mtmd.Free(e.mctx)
		e.mctx = 0
	}
	if e.lctx != 0 {
		llama.Free(e.lctx)
		e.lctx = 0
	}
	if e.model != 0 {
		llama.ModelFree(e.model)
		e.model = 0
	}
}

// EmbedText embeds text alone. parseSpecial makes strings like <bos> in text
// special tokens, as the reference tokenizer does.
func (e *Engine) EmbedText(text string, parseSpecial bool) (Embedding, error) {
	toks := llama.Tokenize(llama.ModelGetVocab(e.model), text, true, parseSpecial)
	if err := e.checkLen(len(toks)); err != nil {
		return Embedding{}, err
	}
	batch, err := llama.BatchExtInit(e.lctx)
	if err != nil {
		return Embedding{}, err
	}
	defer llama.BatchExtFree(batch)
	for i, t := range toks {
		if err := addToken(batch, t, llama.Pos(i)); err != nil {
			return Embedding{}, err
		}
	}
	vec, err := e.forward(batch)
	return Embedding{Vec: vec, Tokens: len(toks)}, err
}

// EmbedMixed embeds text in which each Marker stands for the next part, in order.
func (e *Engine) EmbedMixed(text string, parts []Part, parseSpecial bool) (Embedding, error) {
	if e.mctx == 0 {
		return Embedding{}, errors.New("vision and audio encoders not loaded")
	}
	if n := strings.Count(text, Marker); n != len(parts) {
		return Embedding{}, fmt.Errorf("%d media markers for %d media parts", n, len(parts))
	}

	bitmaps := make([]mtmd.Bitmap, 0, len(parts))
	defer func() {
		for _, b := range bitmaps {
			mtmd.BitmapFree(b)
		}
	}()
	for i, p := range parts {
		b, err := e.bitmap(p)
		if err != nil {
			return Embedding{}, fmt.Errorf("media %d: %w", i+1, err)
		}
		bitmaps = append(bitmaps, b)
	}

	chunks := mtmd.InputChunksInit()
	defer mtmd.InputChunksFree(chunks)
	if ret := mtmd.Tokenize(e.mctx, chunks, mtmd.NewInputText(text, true, parseSpecial), bitmaps); ret != 0 {
		return Embedding{}, fmt.Errorf("mtmd tokenize returned %d", ret)
	}
	n := mtmd.InputChunksSize(chunks)
	tokens := 0
	for i := range n {
		tokens += int(mtmd.InputChunkGetNTokens(mtmd.InputChunksGet(chunks, i)))
	}
	if err := e.checkLen(tokens); err != nil {
		return Embedding{}, err
	}

	batch, err := llama.BatchExtInit(e.lctx)
	if err != nil {
		return Embedding{}, err
	}
	defer llama.BatchExtFree(batch)

	// The architecture has no KV cache, so every Process call is independent
	// and pools only over its own tokens. Text and media must therefore go
	// through the backbone in one batch; mtmd_helper_eval_chunks would split
	// them per chunk.
	var pos llama.Pos
	for i := range n {
		c := mtmd.InputChunksGet(chunks, i)
		switch typ := mtmd.InputChunkGetType(c); typ {
		case mtmd.InputChunkTypeText:
			for _, t := range mtmd.InputChunkGetTokensText(c) {
				if err := addToken(batch, t, pos); err != nil {
					return Embedding{}, err
				}
				pos++
			}
		case mtmd.InputChunkTypeImage, mtmd.InputChunkTypeAudio:
			nt := int(mtmd.InputChunkGetNTokens(c))
			if err := mtmd.EncodeChunk(e.mctx, c); err != nil {
				return Embedding{}, err
			}
			emb, err := mtmd.GetOutputEmbd(e.mctx, int32(nt)*e.nEmbdInp)
			if err != nil {
				return Embedding{}, err
			}
			w := int(e.nEmbdInp)
			for j := range nt {
				idx, err := llama.BatchExtAddEmbd(batch, 0, emb[j*w:(j+1)*w], w)
				if err != nil {
					return Embedding{}, err
				}
				if err := finishEntry(batch, idx, pos+llama.Pos(j)); err != nil {
					return Embedding{}, err
				}
			}
			pos += mtmd.InputChunkGetNPos(c)
		default:
			return Embedding{}, fmt.Errorf("unknown chunk type %d", typ)
		}
	}
	vec, err := e.forward(batch)
	return Embedding{Vec: vec, Tokens: tokens}, err
}

// bitmap turns a part into an mtmd bitmap the caller frees.
func (e *Engine) bitmap(p Part) (mtmd.Bitmap, error) {
	if p.Image != nil {
		b := mtmd.BitmapInit(uint32(p.Image.W), uint32(p.Image.H), p.Image.Pix)
		if b == 0 {
			return 0, errors.New("cannot create image bitmap")
		}
		return b, nil
	}
	var w mtmd.BitmapWrapper
	switch {
	case p.AudioPath != "":
		w = mtmd.BitmapInitFromFile(e.mctx, p.AudioPath, false, mtmd.InitOptDefault())
	case len(p.AudioData) > 0:
		w = mtmd.BitmapInitFromBuf(e.mctx, &p.AudioData[0], uint64(len(p.AudioData)), false, mtmd.InitOptDefault())
		runtime.KeepAlive(p.AudioData)
	default:
		return 0, errors.New("empty media part")
	}
	// mtmd's loader also accepts images and, through its ffmpeg helper,
	// video; only audio may come through here.
	if w.VideoContext != 0 {
		mtmd.VideoFree(w.VideoContext)
	}
	if w.Bitmap == 0 || !mtmd.BitmapIsAudio(w.Bitmap) {
		if w.Bitmap != 0 {
			mtmd.BitmapFree(w.Bitmap)
		}
		return 0, errors.New("not a supported audio file (wav, mp3 or flac)")
	}
	return w.Bitmap, nil
}

// checkLen guards llama.cpp's GGML_ASSERT(n_ubatch >= n_tokens), which aborts
// the whole process instead of returning an error.
func (e *Engine) checkLen(n int) error {
	if n > e.maxTokens {
		return fmt.Errorf("input is %d tokens, more than the %d allowed", n, e.maxTokens)
	}
	return nil
}

func addToken(batch llama.BatchExt, t llama.Token, pos llama.Pos) error {
	idx, err := llama.BatchExtAddToken(batch, 0, t)
	if err != nil {
		return err
	}
	return finishEntry(batch, idx, pos)
}

func finishEntry(batch llama.BatchExt, idx int32, pos llama.Pos) error {
	if err := llama.BatchExtSetPos(batch, idx, pos); err != nil {
		return err
	}
	return llama.BatchExtSetOutputEmbd(batch, idx, true)
}

func (e *Engine) forward(batch llama.BatchExt) ([]float32, error) {
	ret, err := llama.Process(e.lctx, llama.ProcessTypeEncode, batch)
	if err != nil {
		return nil, err
	}
	if ret != 0 {
		return nil, fmt.Errorf("llama_process returned %d", ret)
	}
	raw, err := llama.GetEmbeddingsSeq(e.lctx, 0, e.nEmbdOut)
	if err != nil {
		return nil, err
	}
	if raw == nil {
		return nil, errors.New("no pooled embedding for sequence 0")
	}
	for _, x := range raw {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			return nil, errors.New("embedding has non-finite values")
		}
	}
	return append([]float32(nil), raw...), nil
}
