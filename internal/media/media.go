// Package media prepares images and video frames for the vision encoder the
// way the reference processors do.
package media

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/color"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"math"
	"os"
	"os/exec"
	"strconv"
	"strings"

	"golang.org/x/image/draw"
	_ "golang.org/x/image/webp"
)

// Vision preprocessing reproduced from transformers' Gemma4ImageProcessor and
// EmbeddingGemma2VideoProcessor (v5.20.2). mtmd's own loaders keep images near
// their original size, sample other video frames and prepend "Video:", which
// took video down to 0.94 cosine to the reference.
const (
	patchSize       = 16
	poolingKernel   = 3
	imageSoftTokens = 280 // processor_config.json image_processor.max_soft_tokens
	videoSoftTokens = 140 // processor_config.json video_processor.max_soft_tokens
	videoSampleFPS  = 1   // video_processor.fps
	videoMaxFrames  = 32  // video_processor.max_frames, overflow_strategy "uniform"

	// maxPixels is where PIL refuses an image as a decompression bomb (twice
	// Image.MAX_IMAGE_PIXELS), so the reference cannot embed a larger one.
	// It is checked before decoding: a 1 MB PNG can declare 12000x12000
	// pixels, which took about 2 GB to decode.
	maxPixels = 2 * (1024 * 1024 * 1024 / 4 / 3)
)

// hfTargetSize mirrors get_aspect_ratio_preserving_size: the largest size that
// keeps the aspect ratio, fits maxSoftTokens after 3x3 pooling, and has sides
// divisible by 48.
func hfTargetSize(h, w, maxSoftTokens int) (int, int, error) {
	maxPatches := maxSoftTokens * poolingKernel * poolingKernel
	targetPx := float64(maxPatches * patchSize * patchSize)
	factor := math.Sqrt(targetPx / float64(h*w))
	side := poolingKernel * patchSize
	th := int(math.Floor(factor*float64(h)/float64(side))) * side
	tw := int(math.Floor(factor*float64(w)/float64(side))) * side

	maxSide := (maxPatches / (poolingKernel * poolingKernel)) * side
	switch {
	case th == 0 && tw == 0:
		return 0, 0, fmt.Errorf("%dx%d is too small to resize", w, h)
	case th == 0:
		th = side
		tw = min(int(math.Floor(float64(w)/float64(h)))*side, maxSide)
	case tw == 0:
		tw = side
		th = min(int(math.Floor(float64(h)/float64(w)))*side, maxSide)
	}
	return th, tw, nil
}

// resizeRGB scales img to w x h and returns packed RGB bytes. Catmull-Rom is
// the a=-0.5 cubic that torch's antialiased bicubic and PIL use; x/image/draw
// widens it on downscaling the same way.
func resizeRGB(img image.Image, w, h int) []byte {
	b := img.Bounds()
	dst := image.NewRGBA(image.Rect(0, 0, w, h))
	if b.Dx() == w && b.Dy() == h {
		draw.Draw(dst, dst.Bounds(), img, b.Min, draw.Src)
	} else {
		draw.CatmullRom.Scale(dst, dst.Bounds(), img, b, draw.Src, nil)
	}
	rgb := make([]byte, 0, w*h*3)
	for i := 0; i < len(dst.Pix); i += 4 {
		rgb = append(rgb, dst.Pix[i], dst.Pix[i+1], dst.Pix[i+2])
	}
	return rgb
}

// Frame is packed 8-bit RGB at the size the vision encoder gets it.
type Frame struct {
	W, H int
	Pix  []byte
}

// LoadImage decodes path and resizes it the way the reference processor does.
// ok is false when path is not in an image format Go decodes, so the caller
// can try video and audio next.
func LoadImage(path string) (Frame, bool, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Frame{}, false, err
	}
	return DecodeImage(data)
}

// DecodeImage is LoadImage for the bytes of an image file.
func DecodeImage(data []byte) (Frame, bool, error) {
	cfg, _, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		return Frame{}, false, nil
	}
	if err := checkPixels(cfg.Width, cfg.Height); err != nil {
		return Frame{}, true, err
	}
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return Frame{}, true, fmt.Errorf("decode image: %w", err)
	}
	img = orient(dropAlpha(img), exifOrientation(data))
	b := img.Bounds()
	th, tw, err := hfTargetSize(b.Dy(), b.Dx(), imageSoftTokens)
	if err != nil {
		return Frame{}, true, err
	}
	return Frame{W: tw, H: th, Pix: resizeRGB(img, tw, th)}, true, nil
}

// checkPixels refuses images larger than maxPixels.
func checkPixels(w, h int) error {
	if px := int64(max(1, w)) * int64(max(1, h)); px > maxPixels {
		return fmt.Errorf("%dx%d is %d pixels, more than the %d allowed", w, h, px, maxPixels)
	}
	return nil
}

// dropAlpha makes img opaque while keeping each pixel's color, as PIL's
// convert("RGB") does. Drawing a translucent image directly would
// premultiply it, darkening the edges and blacking out transparent areas.
// An NRGBA image, which is what translucent PNG and WebP decode to, is
// changed in place rather than copied.
func dropAlpha(img image.Image) image.Image {
	if o, ok := img.(interface{ Opaque() bool }); ok && o.Opaque() {
		return img
	}
	if n, ok := img.(*image.NRGBA); ok {
		for y := n.Rect.Min.Y; y < n.Rect.Max.Y; y++ {
			row := n.Pix[n.PixOffset(n.Rect.Min.X, y):n.PixOffset(n.Rect.Max.X, y)]
			for i := 3; i < len(row); i += 4 {
				row[i] = 0xff
			}
		}
		return n
	}
	b := img.Bounds()
	out := image.NewNRGBA(b)
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			c := color.NRGBAModel.Convert(img.At(x, y)).(color.NRGBA)
			c.A = 0xff
			out.SetNRGBA(x, y, c)
		}
	}
	return out
}

// VideoMeta is what the reference processor reads from a video container.
type VideoMeta struct {
	FPS      float64
	Duration float64
	Frames   int
	W, H     int // as displayed: after the stream's rotation, which ffmpeg applies
}

// ProbeVideo reads what torchcodec's metadata gives the reference processor.
// ok is false when ffprobe finds no video stream (an image or audio file) or
// cannot run.
func ProbeVideo(ctx context.Context, path string) (VideoMeta, bool, error) {
	out, err := exec.CommandContext(ctx, "ffprobe", "-v", "error", "-select_streams", "v:0",
		"-show_entries", "stream=width,height,avg_frame_rate,nb_frames,duration:"+
			"stream_side_data=rotation:stream_tags=rotate:format=duration",
		"-of", "json", path).Output()
	if err != nil {
		return VideoMeta{}, false, nil
	}
	var p struct {
		Streams []struct {
			Width, Height int
			AvgFrameRate  string `json:"avg_frame_rate"`
			NbFrames      string `json:"nb_frames"`
			Duration      string `json:"duration"`
			SideData      []struct {
				Rotation *float64 `json:"rotation"`
			} `json:"side_data_list"`
			Tags struct {
				Rotate string `json:"rotate"` // what FFmpeg before 5 reported instead
			} `json:"tags"`
		}
		Format struct {
			Duration string `json:"duration"`
		}
	}
	if err := json.Unmarshal(out, &p); err != nil {
		return VideoMeta{}, false, err
	}
	if len(p.Streams) == 0 {
		return VideoMeta{}, false, nil
	}
	s := p.Streams[0]
	m := VideoMeta{W: s.Width, H: s.Height}
	// ffmpeg turns frames upright by the display matrix, as torchcodec does,
	// so a quarter turn swaps the frame's sides. Reading them at the stored
	// size instead takes the same number of bytes and garbles every frame.
	rotation, _ := strconv.ParseFloat(s.Tags.Rotate, 64)
	for _, sd := range s.SideData {
		if sd.Rotation != nil {
			rotation = *sd.Rotation
		}
	}
	if quarter := int(math.Round(rotation / 90)); quarter%2 != 0 {
		m.W, m.H = m.H, m.W
	}
	if num, den, ok := strings.Cut(s.AvgFrameRate, "/"); ok {
		n, _ := strconv.ParseFloat(num, 64)
		d, _ := strconv.ParseFloat(den, 64)
		if d > 0 {
			m.FPS = n / d
		}
	}
	m.Duration, _ = strconv.ParseFloat(s.Duration, 64)
	if m.Duration == 0 {
		m.Duration, _ = strconv.ParseFloat(p.Format.Duration, 64)
	}
	m.Frames, _ = strconv.Atoi(s.NbFrames)
	if m.Frames == 0 && m.FPS > 0 {
		m.Frames = int(math.Round(m.Duration * m.FPS))
	}
	if m.FPS <= 0 || m.Duration <= 0 || m.Frames < 2 || m.W <= 0 || m.H <= 0 {
		// single-frame streams are cover art in audio files, not video
		return VideoMeta{}, false, nil
	}
	return m, true, nil
}

// sampleFrameIndices mirrors EmbeddingGemma2VideoProcessor.sample_frames with
// fps=1, max_frames=32, overflow_strategy="uniform".
func sampleFrameIndices(m VideoMeta) []int {
	step := m.FPS / videoSampleFPS
	n := max(1, int(m.Duration*videoSampleFPS))
	idx := make([]int, n)
	for i := range idx {
		idx[i] = min(m.Frames-1, int(float64(i)*step))
	}
	if len(idx) <= videoMaxFrames {
		return idx
	}
	// np.linspace(0, len-1, max_frames, dtype=int): i*step truncated toward
	// zero, with the last element set to exactly len-1
	out := make([]int, videoMaxFrames)
	last := len(idx) - 1
	ls := float64(last) / float64(videoMaxFrames-1)
	for i := range out {
		out[i] = idx[int(float64(i)*ls)]
	}
	out[len(out)-1] = idx[last]
	return out
}

// VideoFrames decodes the sampled frames with ffmpeg and resizes each like
// the reference video processor.
func VideoFrames(ctx context.Context, path string, m VideoMeta) ([]Frame, error) {
	if err := checkPixels(m.W, m.H); err != nil {
		return nil, fmt.Errorf("video frame: %w", err)
	}
	idx := sampleFrameIndices(m)
	terms := make([]string, len(idx))
	for i, n := range idx {
		terms[i] = fmt.Sprintf("eq(n\\,%d)", n)
	}
	cmd := exec.CommandContext(ctx, "ffmpeg", "-v", "error", "-nostdin", "-i", path,
		"-vf", "select='"+strings.Join(terms, "+")+"'", "-fps_mode", "passthrough",
		"-pix_fmt", "rgb24", "-f", "rawvideo", "-")
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	if err := cmd.Start(); err != nil {
		return nil, err
	}

	th, tw, err := hfTargetSize(m.H, m.W, videoSoftTokens)
	if err != nil {
		cmd.Process.Kill()
		cmd.Wait()
		return nil, err
	}
	var frames []Frame
	buf := make([]byte, m.W*m.H*3)
	for {
		if _, err := io.ReadFull(stdout, buf); err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			cmd.Process.Kill()
			cmd.Wait()
			return nil, fmt.Errorf("read frame %d: %w", len(frames), err)
		}
		src := &image.RGBA{Pix: rgbToRGBA(buf), Stride: m.W * 4, Rect: image.Rect(0, 0, m.W, m.H)}
		frames = append(frames, Frame{W: tw, H: th, Pix: resizeRGB(src, tw, th)})
	}
	if err := cmd.Wait(); err != nil {
		return nil, fmt.Errorf("ffmpeg: %w: %s", err, strings.TrimSpace(stderr.String()))
	}
	if len(frames) != len(idx) {
		return nil, fmt.Errorf("ffmpeg returned %d frames, want %d", len(frames), len(idx))
	}
	return frames, nil
}

func rgbToRGBA(rgb []byte) []byte {
	out := make([]byte, len(rgb)/3*4)
	for i, j := 0, 0; i < len(rgb); i, j = i+3, j+4 {
		out[j], out[j+1], out[j+2], out[j+3] = rgb[i], rgb[i+1], rgb[i+2], 0xff
	}
	return out
}

// FFmpegAvailable reports whether video can be read: it needs both ffmpeg and
// ffprobe in PATH.
func FFmpegAvailable() bool {
	_, err1 := exec.LookPath("ffmpeg")
	_, err2 := exec.LookPath("ffprobe")
	return err1 == nil && err2 == nil
}
