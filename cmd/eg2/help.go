package main

import (
	"fmt"
	"io"
	"strings"

	eg2 "github.com/meinside/embedding-gemma-2-runner"
)

// inputsHelp is the one description of what eg2 accepts. `eg2 help inputs`
// prints it and `eg2 serve` answers GET / with it.
func inputsHelp() string {
	var tasks strings.Builder
	for _, t := range eg2.Tasks() {
		p, _ := t.Prompt()
		fmt.Fprintf(&tasks, "  %-20s %q\n", t, p)
	}
	return `INPUTS

One input becomes one 768-dimensional vector. An input is text, media
(image, audio, video), or text and media together.

Media
  image  JPEG, PNG, GIF, WebP, turned upright by their EXIF orientation;
         at most 178,956,970 pixels, the most the reference accepts
  audio  WAV, MP3, FLAC (16 kHz mono is what the model expects)
  video  anything ffmpeg reads; needs ffmpeg and ffprobe in PATH.
         Sampled at 1 frame per second, at most 32 frames.
  All of it shares an 8192-token budget per input: about 270 tokens per
  image, 130 per video frame, 25 per second of audio.

Text and media in one vector
  Mark where each media item goes with a placeholder in the text:
    <|image|>  <|audio|>  <|video|>
  Each placeholder takes the next item of its kind, and every item must be
  used. Without placeholders, items go after the text, in the order given.

  Command line (quote the text: < and | are special to the shell):
    eg2 embed -text "Trail shoe <|image|> grip test <|video|>" \
              -image shoe.jpg -video grip.mp4

  JSON (eg2 embed -input FILE.jsonl, or POST /v1/embeddings):
    {"id": "p1", "text": "Trail shoe <|image|> grip test <|video|>",
     "image": "shoe.jpg", "video": "grip.mp4"}
  Media fields take a string or an array. A server takes data: URIs
  ("data:image/jpeg;base64,...") and, only with -allow-local-files, paths.
  Without placeholders, JSON key order decides the order:
  {"image": ..., "text": ...} puts the image first.

  To embed text that contains a placeholder or a string like <bos> as
  written, use -literal (JSON: "literal": true); media then follow the text.

Tasks
  The model was trained with an instruction before the text. Give one with
  -task (JSON: "task"); without it nothing is added, which works with some
  loss of precision. For retrieval, embed queries with a query task and the
  documents with "document". Classification, clustering and similarity give
  every input the same task.

` + tasks.String() + `
  With "document", -title (JSON: "title") replaces "none". A file given to
  eg2 embed as text gets its file name as the title. -prompt sets a prefix
  of your own instead of a task. The flags add the prefix to inputs that have
  text; media alone get none, as the model card advises. A JSON input's own
  "task" always applies.

Dimensions
  -dim (JSON request: "dimensions") keeps the first 512, 256 or 128 values
  and normalizes them again. Compare vectors of the same size only.
`
}

func printInputsHelp(w io.Writer) { fmt.Fprint(w, inputsHelp()) }
