package eg2

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// Parity with the reference implementation, which defines correct output.
//
//	uv run scripts/reference/ref_embed.py --out .poc/ref.json   # once, needs torch
//	EG2_PARITY_REF=.poc/ref.json go test -run TestParity -v
//
// Optional: EG2_PARITY_MEDIA (default .poc/media, see scripts/make-media.sh),
// EG2_DEVICE (auto, cpu, gpu), EG2_LIB_DIR, EG2_SEED_DIR, EG2_OFFLINE=1.
func TestParity(t *testing.T) {
	refPath := os.Getenv("EG2_PARITY_REF")
	if refPath == "" {
		t.Skip("set EG2_PARITY_REF to a reference embeddings file")
	}
	var ref map[string][]float32
	data, err := os.ReadFile(refPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &ref); err != nil {
		t.Fatal(err)
	}
	mediaDir := os.Getenv("EG2_PARITY_MEDIA")
	if mediaDir == "" {
		mediaDir = ".poc/media"
	}
	inputs := loadCases(t, "testdata/cases.json", mediaDir)

	e, err := Open(context.Background(), Options{
		Device:  map[string]Device{"cpu": DeviceCPU, "gpu": DeviceGPU}[os.Getenv("EG2_DEVICE")],
		LibDir:  os.Getenv("EG2_LIB_DIR"),
		SeedDir: os.Getenv("EG2_SEED_DIR"),
		Offline: os.Getenv("EG2_OFFLINE") != "",
	})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	t.Logf("device: %s", e.Info().Device)

	results, err := e.Embed(context.Background(), inputs...)
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range results {
		if r.Err != nil {
			t.Errorf("%s: %v", r.ID, r.Err)
			continue
		}
		want, ok := ref[r.ID]
		if !ok {
			// A case added without its reference is not checked at all.
			t.Errorf("%s: no reference vector; regenerate the reference file", r.ID)
			continue
		}
		cos := cosine(r.Embedding, want)
		min := minCosine(r.ID)
		t.Logf("%-22s %.5f  %4d tokens  %s", r.ID, cos, r.Tokens, r.Modality)
		if !(cos >= min) { // NaN must fail too
			t.Errorf("%s: cosine %.5f to the reference, want at least %.4f", r.ID, cos, min)
		}
	}
}

// minCosine is the parity each case must reach. Text, image and video reach
// 0.9995 or more with Q8_0 weights on CPU and Metal.
func minCosine(id string) float64 {
	switch id {
	case "aud:tone.wav":
		// llama.cpp computes the audio log-mel as log(max(mel, floor)) where
		// the reference computes log(mel + floor). Only near-silent bands
		// differ, which a pure tone is made of. Fixable only in llama.cpp.
		return 0.97
	case "aud:jfk.mp3":
		// The same speech as WAV reaches 0.9998. mtmd decodes and resamples MP3
		// with miniaudio, the reference with torchcodec (FFmpeg): 0.997. A
		// reference made without torchcodec decoded it differently again (0.994).
		return 0.995
	}
	return 0.999
}

// loadCases reads the cases file, resolving media paths against mediaDir.
func loadCases(t *testing.T, path, mediaDir string) []Input {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var file struct {
		Queries   []json.RawMessage `json:"queries"`
		Documents []json.RawMessage `json:"documents"`
	}
	if err := json.Unmarshal(data, &file); err != nil {
		t.Fatal(err)
	}
	var inputs []Input
	for _, raw := range append(file.Queries, file.Documents...) {
		in, err := ParseInput(raw, ParseOptions{AllowPaths: true})
		if err != nil {
			t.Fatalf("%s: %v", raw, err)
		}
		for i := range in.Media {
			in.Media[i].Path = filepath.Join(mediaDir, in.Media[i].Path)
		}
		inputs = append(inputs, in)
	}
	return inputs
}

func cosine(a, b []float32) float64 {
	if len(a) != len(b) {
		return math.NaN()
	}
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return dot / math.Sqrt(na*nb)
}
