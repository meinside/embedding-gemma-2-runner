package media

import (
	"bytes"
	"encoding/binary"
	"hash/crc32"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"strings"
	"testing"
)

// tiffOrient is a TIFF header whose first IFD holds only the orientation.
func tiffOrient(bigEndian bool, o int) []byte {
	var bo binary.AppendByteOrder = binary.LittleEndian
	b := []byte("II*\x00\x08\x00\x00\x00")
	if bigEndian {
		bo, b = binary.BigEndian, []byte("MM\x00*\x00\x00\x00\x08")
	}
	b = bo.AppendUint16(b, 1)      // entries
	b = bo.AppendUint16(b, 0x0112) // Orientation
	b = bo.AppendUint16(b, 3)      // SHORT
	b = bo.AppendUint32(b, 1)      // count
	b = bo.AppendUint16(b, uint16(o))
	b = append(b, 0, 0)
	return bo.AppendUint32(b, 0) // no next IFD
}

func xmpOrient(o int) []byte {
	return []byte(`<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF><rdf:Description tiff:Orientation="` +
		string(rune('0'+o)) + `"/></rdf:RDF></x:xmpmeta>`)
}

// withJPEGSegments puts APP1 segments right after the SOI marker.
func withJPEGSegments(jpg []byte, payloads ...[]byte) []byte {
	out := append([]byte(nil), jpg[:2]...)
	for _, p := range payloads {
		out = append(out, 0xff, 0xe1)
		out = binary.BigEndian.AppendUint16(out, uint16(len(p)+2))
		out = append(out, p...)
	}
	return append(out, jpg[2:]...)
}

// withPNGChunk puts a chunk right after IHDR.
func withPNGChunk(p []byte, typ string, data []byte) []byte {
	const afterIHDR = 8 + 8 + 13 + 4
	out := append([]byte(nil), p[:afterIHDR]...)
	out = binary.BigEndian.AppendUint32(out, uint32(len(data)))
	c := append([]byte(typ), data...)
	out = append(out, c...)
	out = binary.BigEndian.AppendUint32(out, crc32.ChecksumIEEE(c))
	return append(out, p[afterIHDR:]...)
}

// webpWith is an extended WebP container with the given chunks (and no
// image, which exifOrientation does not need).
func webpWith(chunks ...[]byte) []byte {
	body := []byte("WEBP")
	for i := 0; i+1 < len(chunks); i += 2 {
		body = append(body, chunks[i]...)
		body = binary.LittleEndian.AppendUint32(body, uint32(len(chunks[i+1])))
		body = append(body, chunks[i+1]...)
		if len(chunks[i+1])%2 == 1 {
			body = append(body, 0)
		}
	}
	out := append([]byte("RIFF"), binary.LittleEndian.AppendUint32(nil, uint32(len(body)))...)
	return append(out, body...)
}

func encode(t *testing.T, img image.Image, f func(*bytes.Buffer, image.Image) error) []byte {
	t.Helper()
	var b bytes.Buffer
	if err := f(&b, img); err != nil {
		t.Fatal(err)
	}
	return b.Bytes()
}

func TestExifOrientation(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 8, 8))
	jpg := encode(t, img, func(b *bytes.Buffer, i image.Image) error { return jpeg.Encode(b, i, nil) })
	pngData := encode(t, img, func(b *bytes.Buffer, i image.Image) error { return png.Encode(b, i) })
	exif := func(bigEndian bool, o int) []byte { return append([]byte("Exif\x00\x00"), tiffOrient(bigEndian, o)...) }
	xmpSeg := func(o int) []byte { return append([]byte("http://ns.adobe.com/xap/1.0/\x00"), xmpOrient(o)...) }
	itxt := func(o int) []byte { return append([]byte("XML:com.adobe.xmp\x00\x00\x00\x00\x00"), xmpOrient(o)...) }

	tests := []struct {
		name string
		data []byte
		want int
	}{
		{"jpeg without metadata", jpg, 1},
		{"jpeg exif, big-endian", withJPEGSegments(jpg, exif(true, 6)), 6},
		{"jpeg exif, little-endian", withJPEGSegments(jpg, exif(false, 8)), 8},
		{"jpeg: the first exif counts", withJPEGSegments(jpg, exif(true, 3), exif(true, 6)), 3},
		{"jpeg xmp only", withJPEGSegments(jpg, xmpSeg(5)), 5},
		{"jpeg: exif wins over xmp", withJPEGSegments(jpg, exif(true, 1), xmpSeg(6)), 1},
		{"jpeg: out of range is ignored", withJPEGSegments(jpg, exif(true, 9)), 1},
		{"png without metadata", pngData, 1},
		{"png eXIf", withPNGChunk(pngData, "eXIf", tiffOrient(true, 6)), 6},
		{"png eXIf with an Exif prefix", withPNGChunk(pngData, "eXIf", exif(false, 7)), 7},
		{"png xmp", withPNGChunk(pngData, "iTXt", itxt(8)), 8},
		{"webp exif", webpWith([]byte("VP8X"), make([]byte, 10), []byte("EXIF"), tiffOrient(false, 6)), 6},
		{"webp xmp", webpWith([]byte("VP8X"), make([]byte, 10), []byte("XMP "), xmpOrient(2)), 2},
		{"truncated", withJPEGSegments(jpg, exif(true, 6))[:20], 1},
		{"not an image", []byte("hello"), 1},
	}
	for _, tt := range tests {
		if got := exifOrientation(tt.data); got != tt.want {
			t.Errorf("%s: orientation %d, want %d", tt.name, got, tt.want)
		}
	}
}

// The layouts are what PIL's exif_transpose gives (checked against Pillow 13).
func TestOrientMatchesPIL(t *testing.T) {
	// a b c
	// d e f
	src := image.NewGray(image.Rect(0, 0, 3, 2))
	copy(src.Pix, "abcdef")
	tests := map[int]string{
		1: "abc/def",
		2: "cba/fed",  // FLIP_LEFT_RIGHT
		3: "fed/cba",  // ROTATE_180
		4: "def/abc",  // FLIP_TOP_BOTTOM
		5: "ad/be/cf", // TRANSPOSE
		6: "da/eb/fc", // ROTATE_270
		7: "fc/eb/da", // TRANSVERSE
		8: "cf/be/ad", // ROTATE_90
		// invalid values are left alone, as PIL does
		0: "abc/def", 9: "abc/def", -1: "abc/def",
	}
	for o, want := range tests {
		img := orient(src, o)
		b := img.Bounds()
		var rows []string
		for y := b.Min.Y; y < b.Max.Y; y++ {
			var row []byte
			for x := b.Min.X; x < b.Max.X; x++ {
				row = append(row, color.GrayModel.Convert(img.At(x, y)).(color.Gray).Y)
			}
			rows = append(rows, string(row))
		}
		if got := strings.Join(rows, "/"); got != want {
			t.Errorf("orientation %d: %s, want %s", o, got, want)
		}
	}
}

// A sideways PNG tagged with orientation 6 becomes the same frame, byte for
// byte, as the upright image: the remap feeds the resize the same values.
func TestDecodeImageUpright(t *testing.T) {
	up := image.NewRGBA(image.Rect(0, 0, 120, 80))
	for y := range 80 {
		for x := range 120 {
			up.Set(x, y, color.RGBA{uint8(2 * x), uint8(3 * y), uint8(x ^ y), 0xff})
		}
	}
	// stored a quarter turn counterclockwise; orientation 6 turns it back
	side := image.NewRGBA(image.Rect(0, 0, 80, 120))
	for y := range 120 {
		for x := range 80 {
			side.Set(x, y, up.At(119-y, x))
		}
	}
	enc := func(img image.Image) []byte {
		return encode(t, img, func(b *bytes.Buffer, i image.Image) error { return png.Encode(b, i) })
	}
	want, ok, err := DecodeImage(enc(up))
	if err != nil || !ok {
		t.Fatal(ok, err)
	}
	got, ok, err := DecodeImage(withPNGChunk(enc(side), "eXIf", tiffOrient(true, 6)))
	if err != nil || !ok {
		t.Fatal(ok, err)
	}
	if got.W != want.W || got.H != want.H || !bytes.Equal(got.Pix, want.Pix) {
		t.Errorf("tagged sideways image: %dx%d, upright: %dx%d; equal pixels: %v",
			got.W, got.H, want.W, want.H, bytes.Equal(got.Pix, want.Pix))
	}
}

// A 12-megapixel JPEG, upright and tagged sideways: the cost of orienting.
func BenchmarkDecodeImage(b *testing.B) {
	img := image.NewYCbCr(image.Rect(0, 0, 4000, 3000), image.YCbCrSubsampleRatio420)
	for i := range img.Y {
		img.Y[i] = uint8(i)
	}
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: 80}); err != nil {
		b.Fatal(err)
	}
	jpg := buf.Bytes()
	exif := append([]byte("Exif\x00\x00"), tiffOrient(true, 6)...)
	for _, c := range []struct {
		name string
		data []byte
	}{{"upright", jpg}, {"orientation6", withJPEGSegments(jpg, exif)}} {
		b.Run(c.name, func(b *testing.B) {
			for range b.N {
				if _, ok, err := DecodeImage(c.data); !ok || err != nil {
					b.Fatal(ok, err)
				}
			}
		})
	}
}
