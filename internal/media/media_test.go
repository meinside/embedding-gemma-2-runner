package media

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"image"
	"image/color"
	"image/png"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
)

// Expected sizes give the soft-token counts the reference processor reported
// for testdata/cases.json: 266 per image, 130 per 640x480 video frame.
func TestHFTargetSize(t *testing.T) {
	tests := []struct {
		h, w, tokens int
		wantH, wantW int
	}{
		{720, 960, imageSoftTokens, 672, 912},  // downscaled, 14x19 = 266 tokens
		{480, 640, imageSoftTokens, 672, 912},  // upscaled to the same budget
		{480, 640, videoSoftTokens, 480, 624},  // 10x13 = 130 tokens
		{10, 5000, imageSoftTokens, 48, 13440}, // height rounds to 0: one row, capped width
	}
	for _, tt := range tests {
		h, w, err := hfTargetSize(tt.h, tt.w, tt.tokens)
		if err != nil {
			t.Fatalf("hfTargetSize(%d, %d, %d): %v", tt.h, tt.w, tt.tokens, err)
		}
		if h != tt.wantH || w != tt.wantW {
			t.Errorf("hfTargetSize(%d, %d, %d) = %dx%d, want %dx%d", tt.h, tt.w, tt.tokens, h, w, tt.wantH, tt.wantW)
		}
	}
}

func TestSampleFrameIndices(t *testing.T) {
	tests := []struct {
		name string
		m    VideoMeta
		want []int
	}{
		{"one per second", VideoMeta{FPS: 25, Duration: 8, Frames: 200},
			[]int{0, 25, 50, 75, 100, 125, 150, 175}},
		{"shorter than a second", VideoMeta{FPS: 30, Duration: 0.4, Frames: 12}, []int{0}},
		// 60 sampled frames reduced to 32 by np.linspace(0, 59, 32, dtype=int)
		{"capped uniformly", VideoMeta{FPS: 25, Duration: 60, Frames: 1500}, []int{
			0, 25, 75, 125, 175, 225, 275, 325, 375, 425, 475, 500, 550, 600, 650, 700,
			750, 800, 850, 900, 950, 975, 1025, 1075, 1125, 1175, 1225, 1275, 1325, 1375, 1425, 1475}},
	}
	for _, tt := range tests {
		if got := sampleFrameIndices(tt.m); !slices.Equal(got, tt.want) {
			t.Errorf("%s: got %v, want %v", tt.name, got, tt.want)
		}
	}
}

func TestDropAlphaKeepsColor(t *testing.T) {
	img := image.NewNRGBA(image.Rect(0, 0, 1, 1))
	img.SetNRGBA(0, 0, color.NRGBA{10, 20, 30, 0})
	got := dropAlpha(img).At(0, 0).(color.NRGBA)
	if want := (color.NRGBA{10, 20, 30, 0xff}); got != want {
		t.Errorf("got %v, want %v", got, want)
	}
}

// An NRGBA image is made opaque in place, within its bounds only.
func TestDropAlphaInPlace(t *testing.T) {
	parent := image.NewNRGBA(image.Rect(0, 0, 4, 4))
	sub := parent.SubImage(image.Rect(1, 1, 3, 3)).(*image.NRGBA)
	if got := dropAlpha(sub); got != image.Image(sub) {
		t.Fatal("NRGBA image copied")
	}
	for y := range 4 {
		for x := range 4 {
			inside := x >= 1 && x < 3 && y >= 1 && y < 3
			if a := parent.NRGBAAt(x, y).A; (a == 0xff) != inside {
				t.Errorf("pixel %d,%d: alpha %d", x, y, a)
			}
		}
	}
}

func TestCheckPixels(t *testing.T) {
	if err := checkPixels(maxPixels, 1); err != nil {
		t.Errorf("at the limit: %v", err)
	}
	if err := checkPixels(maxPixels+1, 1); err == nil {
		t.Error("over the limit accepted")
	}
	if err := checkPixels(1<<20, 1<<20); err == nil { // the product overflows int32
		t.Error("huge size accepted")
	}
}

// A small PNG declaring a huge size is refused before its pixels are allocated.
func TestDecodeImageRefusesBomb(t *testing.T) {
	var b bytes.Buffer
	if err := png.Encode(&b, image.NewGray(image.Rect(0, 0, 1, 1))); err != nil {
		t.Fatal(err)
	}
	data := b.Bytes()
	// IHDR: length, "IHDR", width, height, ..., CRC over type and data
	binary.BigEndian.PutUint32(data[16:], 20000)
	binary.BigEndian.PutUint32(data[20:], 20000)
	binary.BigEndian.PutUint32(data[29:], crc32.ChecksumIEEE(data[12:29]))

	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	_, ok, err := DecodeImage(data)
	runtime.ReadMemStats(&after)
	if !ok || err == nil || !strings.Contains(err.Error(), "more than the") {
		t.Fatalf("got ok=%v err=%v, want the size refused", ok, err)
	}
	if n := after.TotalAlloc - before.TotalAlloc; n > 16<<20 {
		t.Errorf("allocated %d MB before refusing", n>>20)
	}
}

// A quarter-turned phone video is decoded upright: ffmpeg rotates the frames,
// so they must be read at the rotated size or every row is garbled.
func TestVideoFramesRotated(t *testing.T) {
	if !FFmpegAvailable() {
		t.Skip("needs ffmpeg and ffprobe in PATH")
	}
	dir := t.TempDir()
	plain, rotated := filepath.Join(dir, "plain.mp4"), filepath.Join(dir, "rotated.mp4")
	// red left half, blue right half, 96x48
	if out, err := exec.Command("ffmpeg", "-v", "error", "-y", "-filter_complex",
		"color=c=red:s=48x48:d=2:r=5[a];color=c=blue:s=48x48:d=2:r=5[b];[a][b]hstack",
		"-pix_fmt", "yuv420p", "-c:v", "mpeg4", "-q:v", "2", plain).CombinedOutput(); err != nil {
		t.Fatalf("ffmpeg: %v\n%s", err, out)
	}
	if out, err := exec.Command("ffmpeg", "-v", "error", "-y", "-display_rotation", "90",
		"-i", plain, "-c", "copy", rotated).CombinedOutput(); err != nil {
		t.Skipf("this ffmpeg cannot set -display_rotation (FFmpeg 6 or later can): %v\n%s", err, out)
	}
	ctx := context.Background()
	m, ok, err := ProbeVideo(ctx, rotated)
	if err != nil || !ok {
		t.Fatalf("ProbeVideo: %v %v", ok, err)
	}
	if m.W != 48 || m.H != 96 {
		t.Fatalf("probed %dx%d, want the upright 48x96", m.W, m.H)
	}
	frames, err := VideoFrames(ctx, rotated, m)
	if err != nil {
		t.Fatal(err)
	}
	f := frames[0]
	if f.H <= f.W {
		t.Fatalf("frame is %dx%d, want portrait", f.W, f.H)
	}
	// Upright, the halves are stacked: each row is one color, top and bottom differ.
	colorAt := func(x, y int) string {
		p := f.Pix[(y*f.W+x)*3:]
		switch {
		case p[0] > 150 && p[2] < 100:
			return "red"
		case p[2] > 150 && p[0] < 100:
			return "blue"
		}
		return fmt.Sprintf("rgb(%d,%d,%d)", p[0], p[1], p[2])
	}
	top, bottom := colorAt(f.W/2, 2), colorAt(f.W/2, f.H-3)
	if top == bottom || (top != "red" && top != "blue") || (bottom != "red" && bottom != "blue") {
		t.Errorf("top %s, bottom %s; want red and blue stacked", top, bottom)
	}
	for _, y := range []int{2, f.H - 3} {
		if l, r := colorAt(2, y), colorAt(f.W-3, y); l != r {
			t.Errorf("row %d: %s on the left, %s on the right", y, l, r)
		}
	}
}
