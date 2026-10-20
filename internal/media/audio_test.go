package media

import (
	"encoding/binary"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// wavHead is a PCM WAV header for dataBytes of audio at byteRate bytes a
// second; the data itself is not included.
func wavHead(byteRate, dataBytes uint32) []byte {
	b := []byte("RIFF")
	b = binary.LittleEndian.AppendUint32(b, 36+dataBytes)
	b = append(b, "WAVEfmt "...)
	b = binary.LittleEndian.AppendUint32(b, 16)
	b = binary.LittleEndian.AppendUint16(b, 1) // PCM
	b = binary.LittleEndian.AppendUint16(b, 1) // mono
	b = binary.LittleEndian.AppendUint32(b, 16000)
	b = binary.LittleEndian.AppendUint32(b, byteRate)
	b = binary.LittleEndian.AppendUint16(b, 2)  // block align
	b = binary.LittleEndian.AppendUint16(b, 16) // bits
	b = append(b, "data"...)
	return binary.LittleEndian.AppendUint32(b, dataBytes)
}

// flacHead is a STREAMINFO block for mono 16-bit audio.
func flacHead(rate uint32, samples uint64) []byte {
	b := []byte("fLaC\x00\x00\x00\x22") // last block, STREAMINFO, 34 bytes
	info := make([]byte, 34)
	// 20 bits of rate, 3 of channels-1 (0), 5 of bits-1 (15), 36 of samples
	info[10], info[11], info[12] = byte(rate>>12), byte(rate>>4), byte(rate&0xf)<<4
	info[13] = 0xf<<4 | byte(samples>>32&0xf)
	binary.BigEndian.PutUint32(info[14:], uint32(samples))
	return append(b, info...)
}

// mp3Frames is n MPEG-1 layer III frames at 128 kbps, 44.1 kHz, stereo:
// 417 bytes each, which is what ffmpeg writes for jfk.mp3.
func mp3Frames(n int) []byte {
	var b []byte
	for range n {
		frame := make([]byte, 417)
		copy(frame, []byte{0xff, 0xfb, 0x90, 0x00})
		b = append(b, frame...)
	}
	return b
}

func TestAudioDuration(t *testing.T) {
	id3 := append([]byte("ID3\x04\x00\x00"), 0, 0, 0, 100) // 100-byte tag
	id3 = append(id3, make([]byte, 100)...)
	xing := mp3Frames(1)
	copy(xing[4+32:], "Xing")
	xing[4+32+7] = 1                                         // frames field present
	binary.BigEndian.PutUint32(xing[4+32+8:], 44100*60/1152) // one minute
	tests := []struct {
		name string
		head []byte
		size int64 // 0: len(head)
		want time.Duration
		ok   bool
	}{
		{"wav", wavHead(32000, 32000*90), 44 + 32000*90, 90 * time.Second, true},
		{"wav, data size unknown", wavHead(32000, 0), 44 + 32000*5, 5 * time.Second, true},
		{"wav, declared past the end", wavHead(32000, 32000*3600), 44 + 32000*5, 5 * time.Second, true},
		{"wav, no byte rate", wavHead(0, 100), 0, 0, false},
		{"flac", flacHead(48000, 48000*120), 0, 120 * time.Second, true},
		{"flac, length unknown", flacHead(48000, 0), 0, 0, false},
		{"mp3, constant bitrate", mp3Frames(3), 128000 / 8 * 100, 100 * time.Second, true},
		{"mp3 after an ID3 tag", append(id3, mp3Frames(3)...), 110 + 128000/8*100, 100 * time.Second, true},
		{"mp3, Xing frame count", append(xing, mp3Frames(2)...), 0, time.Duration(44100*60/1152) * 1152 * time.Second / 44100, true},
		{"mp3, no sync", []byte("hello world, this is not audio"), 0, 0, false},
		{"empty", nil, 0, 0, false},
		{"ogg", []byte("OggS\x00\x02"), 0, 0, false},
	}
	for _, tt := range tests {
		size := tt.size
		if size == 0 {
			size = int64(len(tt.head))
		}
		got, ok := AudioDuration(tt.head, size)
		if ok != tt.ok || math.Abs(float64(got-tt.want)) > float64(10*time.Millisecond) {
			t.Errorf("%s: %v %v, want %v %v", tt.name, got, ok, tt.want, tt.ok)
		}
	}
}

// The parity media, when present, agree with ffprobe to within a frame.
func TestAudioDurationFiles(t *testing.T) {
	if _, err := exec.LookPath("ffprobe"); err != nil {
		t.Skip("needs ffprobe")
	}
	files, _ := filepath.Glob("../../.poc/media/*.*")
	n := 0
	for _, f := range files {
		switch filepath.Ext(f) {
		case ".wav", ".mp3", ".flac":
		default:
			continue
		}
		data, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		out, err := exec.Command("ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", f).Output()
		if err != nil {
			t.Fatal(err)
		}
		want, _ := strconv.ParseFloat(strings.TrimSpace(string(out)), 64)
		got, ok := AudioDuration(data, int64(len(data)))
		if !ok || math.Abs(got.Seconds()-want) > 0.05 {
			t.Errorf("%s: %v %v, ffprobe says %.3fs", filepath.Base(f), got, ok, want)
		}
		n++
	}
	if n == 0 {
		t.Skip("no media in .poc/media (scripts/make-media.sh)")
	}
}
