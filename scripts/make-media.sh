#!/bin/sh
# Generates the media files named in testdata/cases.json.
# Usage: scripts/make-media.sh [out-dir]   (default: .poc/media)
# Needs go, FFmpeg 6 or later, and curl.
set -eu

out=${1:-.poc/media}
mkdir -p "$out"
ff="ffmpeg -hide_banner -loglevel error -y"

# The rotated video below needs -display_rotation, which FFmpeg 6 added.
if ! $ff -display_rotation 90 -f lavfi -i "color=c=black:s=16x16:d=0.1" -f null - 2>/dev/null; then
	echo "make-media.sh: needs FFmpeg 6 or later (no -display_rotation)" >&2
	exit 1
fi

yzma=$(go list -m -f '{{.Dir}}' github.com/hybridgroup/yzma)
cp "$yzma/images/domestic_llama.jpg" "$out/llama.jpg"
chmod 644 "$out/llama.jpg"

# Pinned to a commit and checked, so both sides of a comparison read the same speech.
curl -sSfL -o "$out/jfk.wav" \
	https://raw.githubusercontent.com/ggml-org/whisper.cpp/b0a11594aec50892a02cd8d129eee2dfe93a8bb8/samples/jfk.wav
echo "59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e  $out/jfk.wav" | shasum -a 256 -c -

$ff -f lavfi -i "color=c=red:s=640x480" -frames:v 1 "$out/red.png"
$ff -i "$out/jfk.wav" -ar 44100 -ac 2 -b:a 128k "$out/jfk.mp3"
$ff -f lavfi -i "sine=frequency=440:duration=5" -ar 16000 "$out/tone.wav"
$ff -f lavfi -i "anoisesrc=d=5:c=white:a=0.3:s=1" -ar 16000 "$out/noise.wav"
$ff -loop 1 -i "$out/llama.jpg" \
	-vf "scale=1280:-2,zoompan=z='min(zoom+0.002,1.3)':d=1:s=640x480:fps=25" \
	-t 8 -pix_fmt yuv420p -c:v libx264 "$out/llama.mp4"
$ff -f lavfi -i "testsrc=duration=60:size=640x480:rate=25" -pix_fmt yuv420p -c:v libx264 "$out/testsrc60.mp4"

# As a phone stores portrait shots: the video landscape with a display
# rotation, the photo sideways with EXIF orientation 6.
$ff -display_rotation 90 -i "$out/llama.mp4" -c copy "$out/llama-rot90.mp4"
$ff -i "$out/llama.jpg" -vf transpose=2 -q:v 2 "$out/llama-sideways.jpg"
# APP1: "Exif\0\0", a big-endian TIFF header, one IFD entry (0x0112 SHORT 6)
exif6='\377\341\000\042Exif\000\000MM\000\052\000\000\000\010\000\001\001\022\000\003\000\000\000\001\000\006\000\000\000\000\000\000'
{ head -c 2 "$out/llama-sideways.jpg"; printf "$exif6"; tail -c +3 "$out/llama-sideways.jpg"; } >"$out/llama-exif6.jpg"
rm "$out/llama-sideways.jpg"

ls -l "$out"
