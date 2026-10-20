package media

import (
	"bytes"
	"encoding/binary"
	"time"
)

// AudioDuration reads how long a WAV, FLAC or MP3 file plays from its
// headers, without decoding it: mtmd decodes a file whole, and a long one
// takes gigabytes before the token limit rejects it. head is the start of
// the file (all of it when it is in memory) and size the file's length. ok
// is false for other formats and when the headers do not say.
func AudioDuration(head []byte, size int64) (d time.Duration, ok bool) {
	switch {
	case len(head) >= 12 && string(head[:4]) == "RIFF" && string(head[8:12]) == "WAVE":
		return wavDuration(head, size)
	case bytes.HasPrefix(head, []byte("fLaC")):
		return flacDuration(head)
	}
	return mp3Duration(head, size)
}

// wavDuration divides the data chunk by the byte rate of the fmt chunk, so
// it holds for compressed WAV formats too.
func wavDuration(head []byte, size int64) (time.Duration, bool) {
	var byteRate uint32
	for off := 12; off+8 <= len(head); {
		id, n := string(head[off:off+4]), int64(binary.LittleEndian.Uint32(head[off+4:]))
		body := int64(off + 8)
		switch id {
		case "fmt ":
			if n < 16 || body+16 > int64(len(head)) {
				return 0, false
			}
			byteRate = binary.LittleEndian.Uint32(head[body+8:])
		case "data":
			if byteRate == 0 {
				return 0, false
			}
			// A size of 0 or all ones means "to the end" (written while
			// streaming); a size past the end is cut to it.
			remaining := size - body
			if n == 0 || n == 0xFFFFFFFF || n > remaining {
				n = remaining
			}
			return time.Duration(float64(n) / float64(byteRate) * float64(time.Second)), true
		}
		next := body + n + n%2 // chunks are padded to an even size
		if next > int64(len(head)) || next <= int64(off) {
			return 0, false
		}
		off = int(next)
	}
	return 0, false
}

// flacDuration reads the sample rate and total samples of STREAMINFO, the
// first metadata block.
func flacDuration(head []byte) (time.Duration, bool) {
	const streamInfo = 8 // after "fLaC" and the block header
	if len(head) < streamInfo+18 || head[4]&0x7f != 0 {
		return 0, false
	}
	b := head[streamInfo:]
	rate := uint32(b[10])<<12 | uint32(b[11])<<4 | uint32(b[12])>>4
	samples := uint64(b[13]&0x0f)<<32 | uint64(b[14])<<24 | uint64(b[15])<<16 | uint64(b[16])<<8 | uint64(b[17])
	if rate == 0 || samples == 0 {
		return 0, false
	}
	return time.Duration(float64(samples) / float64(rate) * float64(time.Second)), true
}

// mp3Header is one frame header.
type mp3Header struct {
	bitrate    int // bits per second
	sampleRate int
	samples    int // per frame
	length     int // bytes, this frame
	sideInfo   int // bytes after the header, before a Xing or VBRI tag
}

var (
	mp3BitratesV1 = [4][15]int{
		{}, // reserved layer
		{0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320},     // layer III
		{0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384},    // layer II
		{0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448}, // layer I
	}
	mp3BitratesV2 = [4][15]int{
		{},
		{0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160},
		{0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160},
		{0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256},
	}
	mp3SampleRates = map[byte][3]int{3: {44100, 48000, 32000}, 2: {22050, 24000, 16000}, 0: {11025, 12000, 8000}}
)

// parseMP3Header reads a frame header at the start of b; ok is false when
// there is none.
func parseMP3Header(b []byte) (h mp3Header, ok bool) {
	if len(b) < 4 || b[0] != 0xff || b[1]&0xe0 != 0xe0 {
		return h, false
	}
	version, layer := (b[1]>>3)&3, (b[1]>>1)&3
	bitrateIdx, rateIdx, padding, mono := b[2]>>4, (b[2]>>2)&3, int(b[2]>>1&1), b[3]>>6 == 3
	rates, ok := mp3SampleRates[version]
	if !ok || layer == 0 || bitrateIdx == 0 || bitrateIdx == 15 || rateIdx == 3 {
		return h, false // reserved values, or a free-format bitrate
	}
	h.sampleRate = rates[rateIdx]
	if version == 3 {
		h.bitrate = mp3BitratesV1[layer][bitrateIdx] * 1000
	} else {
		h.bitrate = mp3BitratesV2[layer][bitrateIdx] * 1000
	}
	switch layer {
	case 3: // layer I
		h.samples = 384
		h.length = (12*h.bitrate/h.sampleRate + padding) * 4
	case 2: // layer II
		h.samples = 1152
		h.length = 144*h.bitrate/h.sampleRate + padding
	default: // layer III
		h.samples = 1152
		if version != 3 {
			h.samples = 576
		}
		h.length = h.samples/8*h.bitrate/h.sampleRate + padding
	}
	switch {
	case version == 3 && !mono:
		h.sideInfo = 32
	case version == 3 || !mono:
		h.sideInfo = 17
	default:
		h.sideInfo = 9
	}
	return h, true
}

// mp3Duration finds the first frame after any ID3v2 tag. A Xing, Info or
// VBRI tag in it gives the frame count; otherwise the file is taken as
// constant bitrate at the first frame's rate. A variable-bitrate file
// without such a tag (rare: encoders write one) can come out off by about
// a factor of two; counting its frames would mean reading the whole file.
func mp3Duration(head []byte, size int64) (time.Duration, bool) {
	off := 0
	if len(head) >= 10 && string(head[:3]) == "ID3" {
		n := 0
		for _, c := range head[6:10] { // syncsafe: 7 bits a byte
			n = n<<7 | int(c&0x7f)
		}
		off = 10 + n
		if head[5]&0x10 != 0 {
			off += 10 // footer
		}
	}
	for ; off+4 <= len(head); off++ {
		h, ok := parseMP3Header(head[off:])
		if !ok {
			continue
		}
		// A sync pattern can occur in data; the next frame must sync too.
		if next := off + h.length; next+4 <= len(head) {
			if _, ok := parseMP3Header(head[next:]); !ok {
				continue
			}
		}
		frame := head[off:]
		if tag := h.sideInfo + 4; tag+16 <= len(frame) {
			if id := string(frame[tag : tag+4]); (id == "Xing" || id == "Info") && frame[tag+7]&1 != 0 {
				frames := binary.BigEndian.Uint32(frame[tag+8:])
				return time.Duration(float64(frames) * float64(h.samples) / float64(h.sampleRate) * float64(time.Second)), true
			}
		}
		if tag := 4 + 32; tag+18 <= len(frame) && string(frame[tag:tag+4]) == "VBRI" {
			frames := binary.BigEndian.Uint32(frame[tag+14:])
			return time.Duration(float64(frames) * float64(h.samples) / float64(h.sampleRate) * float64(time.Second)), true
		}
		return time.Duration(float64(size-int64(off)) * 8 / float64(h.bitrate) * float64(time.Second)), true
	}
	return 0, false
}
