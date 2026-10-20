package assets

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
)

// Pinned to a commit, not "main": a re-upload under the same name would
// otherwise change the embeddings an existing index was built with.
const (
	ModelRepo     = "ggml-org/embeddinggemma-2-GGUF"
	ModelRevision = "58faf90940886df2d62be555c8765d00dac362c5"
)

// Weights selects the quantization of the model files.
type Weights string

const (
	Q8_0 Weights = "q8_0"
	// BF16 is for parity checks only: it matches the reference exactly on
	// text, which separates quantization error from preprocessing differences.
	// CPUs without native bf16 (Apple M1) encode media over ten times slower.
	BF16 Weights = "bf16"
)

// The F16 variants are deliberately absent: the model card warns that float16
// activations produce NaN or silently degraded embeddings.
var modelFiles = map[Weights][2]pinned{
	Q8_0: {
		modelFile("embeddinggemma-2-Q8_0.gguf", "2188ac1deca4b77dffefd603c2776a9d76d9d74ec01841392982ebb840b09135", 309855456),
		modelFile("mmproj-embeddinggemma-2-Q8_0.gguf", "c4a8a52691ecef40618438928bdf9e68379b854e24166f292592353db0aab64f", 554821024),
	},
	BF16: {
		modelFile("embeddinggemma-2-BF16.gguf", "68bae29d62fb8c7d23e98d21fd4662753ddd636e6b62b8a70dfe059a9844f216", 557950176),
		modelFile("mmproj-embeddinggemma-2-BF16.gguf", "d2033b3cd0223cfb2e7a2b50dc80a664c78c816d375f5b017ec3ef80ee0bc766", 982074784),
	},
}

func modelFile(name, sha string, size int64) pinned {
	return pinned{
		Name:   name,
		URL:    fmt.Sprintf("https://huggingface.co/%s/resolve/%s/%s", ModelRepo, ModelRevision, name),
		SHA256: sha,
		Size:   size,
	}
}

// ModelOptions controls how model files are obtained.
type ModelOptions struct {
	Weights  Weights
	Offline  bool     // fail instead of downloading
	SeedDir  string   // copy from here instead of downloading (still checked)
	Progress Progress // nil for silence
}

// TextModel returns the path of the text model, fetching it if needed.
func TextModel(ctx context.Context, d Dirs, o ModelOptions) (string, error) {
	return ensureModel(ctx, d, o, 0)
}

// MMProj returns the path of the vision and audio encoders, fetching them if needed.
func MMProj(ctx context.Context, d Dirs, o ModelOptions) (string, error) {
	return ensureModel(ctx, d, o, 1)
}

// ModelDir is where the model files of the pinned revision live.
func ModelDir(d Dirs) string {
	return filepath.Join(d.Models, filepath.FromSlash(ModelRepo), ModelRevision)
}

func ensureModel(ctx context.Context, d Dirs, o ModelOptions, which int) (string, error) {
	w := o.Weights
	if w == "" {
		w = Q8_0
	}
	files, ok := modelFiles[w]
	if !ok {
		return "", fmt.Errorf("unknown weights %q", w)
	}
	f := files[which]
	dir := ModelDir(d)
	dst := filepath.Join(dir, f.Name)
	// Only a verified file is ever renamed to this name, so its size is enough
	// to tell it apart from one truncated by something else.
	if fi, err := os.Stat(dst); err == nil && fi.Size() == f.Size {
		return dst, nil
	}
	switch {
	case o.SeedDir != "":
		return copyVerified(dir, filepath.Join(o.SeedDir, f.Name), f)
	case o.Offline:
		return "", fmt.Errorf("%s: %w", f.Name, ErrOffline)
	}
	return fetchTo(ctx, dir, f, o.Progress)
}
