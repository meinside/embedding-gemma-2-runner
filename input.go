package eg2

import (
	"errors"
	"fmt"
	"regexp"
	"strings"
)

// Task selects the instruction prefix the model was trained with. The empty
// Task adds none; the model still works, with some loss of precision.
// Retrieval is asymmetric: embed queries with a query task and the corpus with
// TaskDocument. Classification, clustering and similarity give every input
// the same task.
type Task string

const (
	TaskNone              Task = ""
	TaskSearchQuery       Task = "search-query"
	TaskDocument          Task = "document"
	TaskQuestionAnswering Task = "question-answering"
	TaskFactChecking      Task = "fact-checking"
	TaskCodeRetrieval     Task = "code-retrieval"
	TaskClassification    Task = "classification"
	TaskClustering        Task = "clustering"
	TaskSimilarity        Task = "similarity"
)

// taskPrompts are the prompts in the model's config_sentence_transformers.json.
var taskPrompts = map[Task]string{
	TaskSearchQuery:       "task: search result | query: ",
	TaskDocument:          "title: none | text: ",
	TaskQuestionAnswering: "task: question answering | query: ",
	TaskFactChecking:      "task: fact checking | query: ",
	TaskCodeRetrieval:     "task: code retrieval | query: ",
	TaskClassification:    "task: classification | query: ",
	TaskClustering:        "task: clustering | query: ",
	TaskSimilarity:        "task: sentence similarity | query: ",
}

// Tasks lists the tasks in the order they are documented.
func Tasks() []Task {
	return []Task{TaskSearchQuery, TaskDocument, TaskQuestionAnswering, TaskFactChecking,
		TaskCodeRetrieval, TaskClassification, TaskClustering, TaskSimilarity}
}

// Prompt returns the prefix for t, "" for TaskNone.
func (t Task) Prompt() (string, error) {
	if t == TaskNone {
		return "", nil
	}
	p, ok := taskPrompts[t]
	if !ok {
		return "", fmt.Errorf("unknown task %q", t)
	}
	return p, nil
}

// MediaKind is the modality of a media item.
type MediaKind int

const (
	MediaAuto MediaKind = iota // decided from the content: image, then video, then audio
	MediaImage
	MediaAudio
	MediaVideo
)

func (k MediaKind) String() string {
	switch k {
	case MediaImage:
		return "image"
	case MediaAudio:
		return "audio"
	case MediaVideo:
		return "video"
	}
	return "auto"
}

// Media is one image, audio clip or video, given by Path or by Data.
//
// Images: anything Go decodes (JPEG, PNG, GIF, WebP), turned upright by
// their EXIF orientation first. Audio: WAV, MP3, FLAC. Video: anything
// ffmpeg reads; needs ffmpeg and ffprobe in PATH.
type Media struct {
	Kind MediaKind
	Path string
	Data []byte
}

// Input is one thing to embed: text, media, or both in one vector.
//
// Text may mark where media go with <|image|>, <|audio|> and <|video|>. Each
// placeholder takes the next Media item of its kind, and every item must be
// used. Without placeholders, Media follow Text in order. This is how the
// reference implementation reads its inputs.
type Input struct {
	ID    string
	Text  string
	Media []Media
	Task  Task
	// Title is the document title for TaskDocument ("title: none" without it).
	Title string
	// Prompt is a custom prefix, instead of Task.
	Prompt string
	// Literal embeds Text as written: placeholders and special-token strings
	// such as <bos> stay text, and Media follow Text.
	Literal bool
}

var placeholderRE = regexp.MustCompile(`<\|(image|audio|video)\|>`)

// HasPlaceholders reports whether text marks media positions with <|image|>,
// <|audio|> or <|video|>.
func HasPlaceholders(text string) bool { return placeholderRE.MatchString(text) }

// HasText reports whether the input has text besides placeholders.
func (in Input) HasText() bool {
	t := in.Text
	if !in.Literal {
		t = placeholderRE.ReplaceAllString(t, "")
	}
	return strings.TrimSpace(t) != ""
}

// segment is a piece of an input in order: text, or the index of a Media item.
type segment struct {
	text  string
	media int // -1 for text
}

// prefix returns the instruction text that goes before everything else.
func (in Input) prefix() (string, error) {
	switch {
	case in.Prompt != "" && in.Task != TaskNone:
		return "", errors.New("prompt and task are mutually exclusive")
	case in.Title != "" && in.Task != TaskDocument:
		return "", errors.New("title needs task document")
	case in.Prompt != "":
		return in.Prompt, nil
	case in.Title != "":
		return "title: " + in.Title + " | text: ", nil
	}
	return in.Task.Prompt()
}

// segments lays the input out in order. kinds holds the resolved kind of each
// Media item (MediaAuto resolved by the caller).
func (in Input) segments(kinds []MediaKind) ([]segment, error) {
	if in.Text == "" && len(in.Media) == 0 {
		return nil, errors.New("empty input")
	}
	appendAll := func(segs []segment) []segment {
		for i := range in.Media {
			segs = append(segs, segment{media: i})
		}
		return segs
	}
	text := []segment{{text: in.Text, media: -1}}
	if in.Text == "" {
		text = nil
	}
	if in.Literal {
		return appendAll(text), nil
	}
	locs := placeholderRE.FindAllStringSubmatchIndex(in.Text, -1)
	if len(locs) == 0 {
		return appendAll(text), nil
	}

	// queue of unused Media indexes per kind, in order
	queue := map[MediaKind][]int{}
	for i, k := range kinds {
		queue[k] = append(queue[k], i)
	}
	var segs []segment
	last := 0
	for _, loc := range locs {
		if loc[0] > last {
			segs = append(segs, segment{text: in.Text[last:loc[0]], media: -1})
		}
		kind := map[string]MediaKind{"image": MediaImage, "audio": MediaAudio, "video": MediaVideo}[in.Text[loc[2]:loc[3]]]
		if len(queue[kind]) == 0 {
			return nil, fmt.Errorf("text has more <|%s|> than %s media", kind, kind)
		}
		segs = append(segs, segment{media: queue[kind][0]})
		queue[kind] = queue[kind][1:]
		last = loc[1]
	}
	if last < len(in.Text) {
		segs = append(segs, segment{text: in.Text[last:], media: -1})
	}
	for _, k := range []MediaKind{MediaImage, MediaAudio, MediaVideo} {
		if n := len(queue[k]); n > 0 {
			return nil, fmt.Errorf("%d %s media without a <|%s|> placeholder in the text", n, k, k)
		}
	}
	return segs, nil
}

// modality names what the input holds: one media kind alone, text, or mixed.
func modality(segs []segment, kinds []MediaKind) string {
	seen := map[string]bool{}
	for _, s := range segs {
		if s.media < 0 {
			if strings.TrimSpace(s.text) != "" {
				seen["text"] = true
			}
			continue
		}
		seen[kinds[s.media].String()] = true
	}
	if len(seen) == 1 {
		for k := range seen {
			return k
		}
	}
	if len(seen) == 0 {
		return "text"
	}
	return "mixed"
}
