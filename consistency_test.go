package eg2

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/meinside/embedding-gemma-2-runner/internal/media"
)

// TestConsistency checks the pipeline against itself with the real model,
// without the Python reference TestParity needs: the same picture stored
// sideways with an EXIF orientation embeds like the upright one, the same
// speech as WAV and MP3 embeds alike, and queries rank the matching media
// and documents first. It needs the model in the cache and the media from
// scripts/make-media.sh, and skips without them.
func TestConsistency(t *testing.T) {
	mediaDir := os.Getenv("EG2_PARITY_MEDIA")
	if mediaDir == "" {
		mediaDir = ".poc/media"
	}
	file := func(name string) string {
		p := filepath.Join(mediaDir, name)
		if _, err := os.Stat(p); err != nil {
			t.Skipf("%s missing: run scripts/make-media.sh", p)
		}
		return p
	}
	ctx := context.Background()
	e, err := Open(ctx, Options{
		Offline: true,
		Device:  map[string]Device{"cpu": DeviceCPU, "gpu": DeviceGPU}[os.Getenv("EG2_DEVICE")],
		LibDir:  os.Getenv("EG2_LIB_DIR"),
	})
	if err != nil {
		t.Skipf("model not in the cache (eg2 setup -media): %v", err)
	}
	defer e.Close()

	image := func(name string) Input { return Input{ID: name, Media: []Media{{Kind: MediaImage, Path: file(name)}}} }
	audio := func(name string) Input { return Input{ID: name, Media: []Media{{Kind: MediaAudio, Path: file(name)}}} }
	query := func(text string) Input { return Input{ID: "q:" + text, Task: TaskSearchQuery, Text: text} }
	doc := func(id, text string) Input { return Input{ID: id, Task: TaskDocument, Text: text} }
	inputs := []Input{
		image("llama.jpg"), image("llama-exif6.jpg"), image("red.png"),
		audio("jfk.wav"), audio("jfk.mp3"), audio("noise.wav"),
		query("a llama"), query("a red square"), query("a man giving a speech"),
		query("채소 키우는 방법"), query("when are taxes due"),
		doc("garden", "토마토 모종은 해가 잘 드는 곳에 심고, 흙이 마르면 아침에 물을 충분히 준다."),
		doc("tax", "Quarterly estimated tax payments are due in April, June, September and January."),
	}
	video := media.FFmpegAvailable()
	if video {
		inputs = append(inputs,
			Input{ID: "llama.mp4", Media: []Media{{Kind: MediaVideo, Path: file("llama.mp4")}}},
			Input{ID: "testsrc60.mp4", Media: []Media{{Kind: MediaVideo, Path: file("testsrc60.mp4")}}})
	}
	results, err := e.Embed(ctx, inputs...)
	if err != nil {
		t.Fatal(err)
	}
	vec := map[string][]float32{}
	for _, r := range results {
		if r.Err != nil {
			t.Fatalf("%s: %v", r.ID, r.Err)
		}
		vec[r.ID] = r.Embedding
	}
	cos := func(a, b string) float64 { return cosine(vec[a], vec[b]) }
	atLeast := func(a, b string, min float64) {
		t.Helper()
		if c := cos(a, b); !(c >= min) {
			t.Errorf("cosine(%s, %s) = %.4f, want at least %.3f", a, b, c, min)
		} else {
			t.Logf("cosine(%s, %s) = %.4f", a, b, c)
		}
	}
	closer := func(q, near, far string) {
		t.Helper()
		if cn, cf := cos(q, near), cos(q, far); !(cn > cf) {
			t.Errorf("%s: %s scores %.4f, %s scores %.4f; want %s first", q, near, cn, far, cf, near)
		} else {
			t.Logf("%s: %s %.4f > %s %.4f", q, near, cn, far, cf)
		}
	}

	atLeast("llama.jpg", "llama-exif6.jpg", 0.99)
	atLeast("jfk.wav", "jfk.mp3", 0.99)
	closer("q:a llama", "llama.jpg", "red.png")
	closer("q:a red square", "red.png", "llama.jpg")
	closer("q:a man giving a speech", "jfk.wav", "noise.wav")
	closer("q:채소 키우는 방법", "garden", "tax")
	closer("q:when are taxes due", "tax", "garden")
	if video {
		closer("q:a llama", "llama.mp4", "testsrc60.mp4")
	}
}
