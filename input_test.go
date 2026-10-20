package eg2

import (
	"math"
	"reflect"
	"strings"
	"testing"
)

func TestPrefix(t *testing.T) {
	tests := []struct {
		in      Input
		want    string
		wantErr bool
	}{
		{in: Input{}, want: ""},
		{in: Input{Task: TaskSearchQuery}, want: "task: search result | query: "},
		{in: Input{Task: TaskDocument}, want: "title: none | text: "},
		{in: Input{Task: TaskDocument, Title: "Aurora"}, want: "title: Aurora | text: "},
		{in: Input{Prompt: "custom: "}, want: "custom: "},
		{in: Input{Task: TaskSearchQuery, Title: "x"}, wantErr: true},
		{in: Input{Task: TaskSearchQuery, Prompt: "x"}, wantErr: true},
		{in: Input{Task: "nope"}, wantErr: true},
	}
	for _, tt := range tests {
		got, err := tt.in.prefix()
		if (err != nil) != tt.wantErr || got != tt.want {
			t.Errorf("%+v: got %q, %v; want %q, error %v", tt.in, got, err, tt.want, tt.wantErr)
		}
	}
}

func TestSegments(t *testing.T) {
	img, aud, vid := MediaImage, MediaAudio, MediaVideo
	tests := []struct {
		name    string
		in      Input
		kinds   []MediaKind
		want    []segment
		wantErr string
	}{
		{name: "text only", in: Input{Text: "hi"},
			want: []segment{{text: "hi", media: -1}}},
		{name: "media follow text without placeholders", in: Input{Text: "hi", Media: make([]Media, 2)}, kinds: []MediaKind{img, aud},
			want: []segment{{text: "hi", media: -1}, {media: 0}, {media: 1}}},
		{name: "placeholders take media of their kind in order",
			in:    Input{Text: "a <|video|> b <|image|><|image|>", Media: make([]Media, 3)},
			kinds: []MediaKind{img, vid, img},
			want:  []segment{{text: "a ", media: -1}, {media: 1}, {text: " b ", media: -1}, {media: 0}, {media: 2}}},
		{name: "literal keeps placeholders as text", in: Input{Text: "<|image|>", Media: make([]Media, 1), Literal: true}, kinds: []MediaKind{img},
			want: []segment{{text: "<|image|>", media: -1}, {media: 0}}},
		{name: "media only", in: Input{Media: make([]Media, 1)}, kinds: []MediaKind{aud},
			want: []segment{{media: 0}}},
		{name: "placeholder without media", in: Input{Text: "<|audio|>"}, wantErr: "more <|audio|>"},
		{name: "media without placeholder", in: Input{Text: "<|image|>", Media: make([]Media, 2)}, kinds: []MediaKind{img, aud},
			wantErr: "1 audio media without"},
		{name: "empty", in: Input{}, wantErr: "empty"},
	}
	for _, tt := range tests {
		got, err := tt.in.segments(tt.kinds)
		if tt.wantErr != "" {
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Errorf("%s: error %v, want one containing %q", tt.name, err, tt.wantErr)
			}
			continue
		}
		if err != nil || !reflect.DeepEqual(got, tt.want) {
			t.Errorf("%s: got %+v, %v; want %+v", tt.name, got, err, tt.want)
		}
	}
}

func TestModality(t *testing.T) {
	kinds := []MediaKind{MediaImage, MediaImage, MediaAudio}
	tests := []struct {
		segs []segment
		want string
	}{
		{[]segment{{text: "x", media: -1}}, "text"},
		{[]segment{{media: 0}, {media: 1}}, "image"},
		{[]segment{{text: " ", media: -1}, {media: 2}}, "audio"},
		{[]segment{{text: "x", media: -1}, {media: 0}}, "mixed"},
		{[]segment{{media: 0}, {media: 2}}, "mixed"},
	}
	for _, tt := range tests {
		if got := modality(tt.segs, kinds); got != tt.want {
			t.Errorf("%+v: got %s, want %s", tt.segs, got, tt.want)
		}
	}
}

func TestParseInputKeyOrder(t *testing.T) {
	paths := ParseOptions{AllowPaths: true}
	tests := []struct {
		json     string
		wantText string
		wantN    int
	}{
		{`"plain text"`, "plain text", 0},
		{`{"text": "shoe", "image": "a.jpg"}`, "shoe<|image|>", 1},
		{`{"image": ["a.jpg", "b.jpg"], "text": "shoe"}`, "<|image|><|image|>shoe", 2},
		{`{"audio": "a.wav", "image": "a.jpg"}`, "<|audio|><|image|>", 2},
		{`{"text": "x <|image|> y", "video": "v.mp4", "image": "a.jpg"}`, "x <|image|> y", 2},
		{`{"literal": true, "image": "a.jpg", "text": "<|image|>"}`, "<|image|>", 1},
	}
	for _, tt := range tests {
		in, err := ParseInput([]byte(tt.json), paths)
		if err != nil {
			t.Errorf("%s: %v", tt.json, err)
			continue
		}
		if in.Text != tt.wantText || len(in.Media) != tt.wantN {
			t.Errorf("%s: text %q with %d media, want %q with %d", tt.json, in.Text, len(in.Media), tt.wantText, tt.wantN)
		}
	}
}

func TestParseInputFields(t *testing.T) {
	in, err := ParseInput([]byte(`{"id":"d1","task":"document","title":"T","text":"body","image":"data:image/png;base64,aGk="}`), ParseOptions{})
	if err != nil {
		t.Fatal(err)
	}
	want := Input{ID: "d1", Task: TaskDocument, Title: "T", Text: "body<|image|>",
		Media: []Media{{Kind: MediaImage, Data: []byte("hi")}}}
	if !reflect.DeepEqual(in, want) {
		t.Errorf("got %+v, want %+v", in, want)
	}
}

func TestParseInputRejects(t *testing.T) {
	for _, s := range []string{
		`{"image": "a.jpg"}`,             // paths not allowed by default
		`{"image": "data:image/png,hi"}`, // not base64
		`{"text": "a", "text": "b"}`,     // duplicate
		`{"colour": "red"}`,              // unknown field
		`{"task": "translate"}`,          // unknown task
		`{"text": "a"} {"text": "b"}`,    // trailing data
		`42`,
	} {
		if _, err := ParseInput([]byte(s), ParseOptions{}); err == nil {
			t.Errorf("%s: accepted", s)
		}
	}
}

func TestTruncate(t *testing.T) {
	v := make([]float32, Dims)
	v[0], v[1], v[200] = 3, 4, 12
	got, err := Truncate(v, 128)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 128 || math.Abs(float64(got[0])-0.6) > 1e-6 || math.Abs(float64(got[1])-0.8) > 1e-6 {
		t.Errorf("got %v..., want [0.6 0.8 0 ...] of length 128", got[:3])
	}
	if _, err := Truncate(v, 300); err == nil {
		t.Error("dimension 300 accepted")
	}
}

func TestHasText(t *testing.T) {
	tests := []struct {
		in   Input
		want bool
	}{
		{Input{Text: "<|image|><|audio|>"}, false},
		{Input{Text: " <|image|> "}, false},
		{Input{Text: "a <|image|>"}, true},
		{Input{Text: "<|image|>", Literal: true}, true},
		{Input{}, false},
	}
	for _, tt := range tests {
		if got := tt.in.HasText(); got != tt.want {
			t.Errorf("%+v: got %v", tt.in, got)
		}
	}
}
