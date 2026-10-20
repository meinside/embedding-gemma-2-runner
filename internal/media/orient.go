package media

import (
	"bytes"
	"encoding/binary"
	"image"
	"image/color"
	"regexp"
)

// Images are turned upright by their EXIF orientation before anything else,
// as the reference's load_image does with PIL.ImageOps.exif_transpose. Phone
// cameras store portrait photos sideways and set the tag instead.

// exifOrientation returns the orientation (1 to 8) a JPEG, PNG or WebP file
// declares, 1 when it declares none. Like PIL it reads the EXIF orientation
// and, only when there is none, XMP's tiff:Orientation. Not read, as in no
// common camera output: PNG's "Raw profile type exif" text and compressed XMP.
func exifOrientation(data []byte) int {
	var exif, xmp []byte
	switch {
	case bytes.HasPrefix(data, []byte("\xff\xd8")):
		exif, xmp = jpegMetadata(data)
	case bytes.HasPrefix(data, []byte("\x89PNG\r\n\x1a\n")):
		exif, xmp = pngMetadata(data)
	case len(data) >= 12 && string(data[:4]) == "RIFF" && string(data[8:12]) == "WEBP":
		exif, xmp = webpMetadata(data)
	}
	if o, ok := tiffOrientation(exif); ok {
		return valid(o)
	}
	if m := xmpOrientationRE.FindSubmatch(xmp); m != nil {
		return valid(int(m[2][0] - '0'))
	}
	return 1
}

// valid maps orientations PIL has no transposition for to 1, which it leaves alone.
func valid(o int) int {
	if o < 1 || o > 8 {
		return 1
	}
	return o
}

var xmpOrientationRE = regexp.MustCompile(`tiff:Orientation(="|>)([0-9])`)

// jpegMetadata returns the first Exif APP1 segment (PIL appends later ones to
// it, so the first one's IFD is what it reads) and the last XMP one.
func jpegMetadata(data []byte) (exif, xmp []byte) {
	const xmpNS = "http://ns.adobe.com/xap/1.0/\x00"
	for i := 2; i+4 <= len(data); {
		if data[i] != 0xff {
			return
		}
		marker := data[i+1]
		switch {
		case marker == 0xff: // fill byte
			i++
			continue
		case marker == 0xd8 || marker == 0x01 || (marker >= 0xd0 && marker <= 0xd7):
			i += 2 // no length
			continue
		case marker == 0xda || marker == 0xd9: // image data or end: no metadata after it
			return
		}
		n := int(binary.BigEndian.Uint16(data[i+2:]))
		if n < 2 || i+2+n > len(data) {
			return
		}
		seg := data[i+4 : i+2+n]
		if marker == 0xe1 {
			switch {
			case bytes.HasPrefix(seg, []byte("Exif\x00\x00")):
				if exif == nil {
					exif = seg[6:]
				}
			case bytes.HasPrefix(seg, []byte(xmpNS)):
				xmp = seg[len(xmpNS):]
			}
		}
		i += 2 + n
	}
	return
}

// pngMetadata returns the last eXIf chunk and the last uncompressed iTXt
// XML:com.adobe.xmp chunk, the ones PIL keeps.
func pngMetadata(data []byte) (exif, xmp []byte) {
	for i := 8; i+12 <= len(data); {
		n := int(binary.BigEndian.Uint32(data[i:]))
		if i+12+n > len(data) {
			return
		}
		typ, body := string(data[i+4:i+8]), data[i+8:i+8+n]
		switch typ {
		case "eXIf":
			exif = bytes.TrimPrefix(body, []byte("Exif\x00\x00"))
		case "iTXt":
			// keyword NUL, compression flag, method, language NUL, translated keyword NUL, text
			if key, rest, ok := bytes.Cut(body, []byte{0}); ok && string(key) == "XML:com.adobe.xmp" &&
				len(rest) >= 2 && rest[0] == 0 {
				xmp = rest[2:]
			}
		case "IEND":
			return
		}
		i += 12 + n
	}
	return
}

// webpMetadata returns the EXIF and XMP chunks of an extended WebP file.
func webpMetadata(data []byte) (exif, xmp []byte) {
	for i := 12; i+8 <= len(data); {
		n := int(binary.LittleEndian.Uint32(data[i+4:]))
		if i+8+n > len(data) {
			return
		}
		body := data[i+8 : i+8+n]
		switch string(data[i : i+4]) {
		case "EXIF":
			if exif == nil {
				exif = bytes.TrimPrefix(body, []byte("Exif\x00\x00"))
			}
		case "XMP ":
			if xmp == nil {
				xmp = body
			}
		}
		i += 8 + n + n%2 // chunks are padded to an even size
	}
	return
}

// tiffOrientation reads tag 0x0112 from the first IFD of a TIFF header.
func tiffOrientation(b []byte) (int, bool) {
	if len(b) < 8 {
		return 0, false
	}
	var bo binary.ByteOrder
	switch string(b[:2]) {
	case "II":
		bo = binary.LittleEndian
	case "MM":
		bo = binary.BigEndian
	default:
		return 0, false
	}
	if bo.Uint16(b[2:]) != 42 {
		return 0, false
	}
	ifd := int64(bo.Uint32(b[4:]))
	if ifd < 8 || ifd+2 > int64(len(b)) {
		return 0, false
	}
	n := int64(bo.Uint16(b[ifd:]))
	for e := ifd + 2; e+12 <= int64(len(b)) && e < ifd+2+12*n; e += 12 {
		if bo.Uint16(b[e:]) != 0x0112 {
			continue
		}
		switch bo.Uint16(b[e+2:]) { // type
		case 3: // SHORT
			return int(bo.Uint16(b[e+8:])), true
		case 4: // LONG
			return int(bo.Uint32(b[e+8:])), true
		}
		return 0, false
	}
	return 0, false
}

// orient returns img as orientation o says to display it, untouched for 1
// and values PIL has no transposition for. The pixels are copied once into
// an upright image the resize reads through its fast path: remapping them on
// read instead made the resize of a 12-megapixel photo three times slower.
// Sources with 16-bit channels keep them; the rest fit RGBA.
func orient(img image.Image, o int) image.Image {
	if o <= 1 || o > 8 {
		return img
	}
	src, ok := img.(image.RGBA64Image)
	if !ok {
		src = rgba64Adapter{img}
	}
	b := src.Bounds()
	w, h := b.Dx(), b.Dy()
	ow, oh := w, h
	if o >= 5 { // a quarter turn or a diagonal flip swaps the sides
		ow, oh = h, w
	}
	switch img.(type) {
	case *image.RGBA64, *image.NRGBA64, *image.Gray16:
		dst := image.NewRGBA64(image.Rect(0, 0, ow, oh))
		for y := range oh {
			for x := range ow {
				sx, sy := sourcePoint(o, w, h, x, y)
				dst.SetRGBA64(x, y, src.RGBA64At(b.Min.X+sx, b.Min.Y+sy))
			}
		}
		return dst
	}
	dst := image.NewRGBA(image.Rect(0, 0, ow, oh))
	for y := range oh {
		row := dst.Pix[dst.PixOffset(0, y):]
		for x := range ow {
			sx, sy := sourcePoint(o, w, h, x, y)
			c := src.RGBA64At(b.Min.X+sx, b.Min.Y+sy)
			row[4*x], row[4*x+1], row[4*x+2], row[4*x+3] = uint8(c.R>>8), uint8(c.G>>8), uint8(c.B>>8), uint8(c.A>>8)
		}
	}
	return dst
}

// sourcePoint maps a point of the oriented image to the w by h source, by
// the transposition PIL's exif_transpose applies for orientation o.
func sourcePoint(o, w, h, x, y int) (int, int) {
	switch o {
	case 2: // FLIP_LEFT_RIGHT
		return w - 1 - x, y
	case 3: // ROTATE_180
		return w - 1 - x, h - 1 - y
	case 4: // FLIP_TOP_BOTTOM
		return x, h - 1 - y
	case 5: // TRANSPOSE
		return y, x
	case 6: // ROTATE_270: a quarter turn clockwise
		return y, h - 1 - x
	case 7: // TRANSVERSE
		return w - 1 - y, h - 1 - x
	case 8: // ROTATE_90: a quarter turn counterclockwise
		return w - 1 - y, x
	}
	return x, y
}

// rgba64Adapter gives an image without RGBA64At one. Every decoder in the
// standard library returns an image that has it.
type rgba64Adapter struct{ image.Image }

func (a rgba64Adapter) RGBA64At(x, y int) color.RGBA64 {
	r, g, b, al := a.At(x, y).RGBA()
	return color.RGBA64{uint16(r), uint16(g), uint16(b), uint16(al)}
}
