package eg2

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
)

// ParseOptions controls ParseInput.
type ParseOptions struct {
	// AllowPaths lets media values be file paths. Otherwise only data: URIs
	// (base64) are accepted, as a server reachable by others needs.
	AllowPaths bool
}

// ParseInput reads one input from JSON: a string is text, an object has the
// fields
//
//	{"id": "...", "text": "...", "image": ..., "audio": ..., "video": ...,
//	 "task": "...", "title": "...", "prompt": "...", "literal": false}
//
// where each media field is a string or an array of strings: data: URIs, or
// paths with AllowPaths. Key order matters as in the reference implementation:
// without placeholders in text, media and text are embedded in the order of
// their keys, so {"image": ..., "text": ...} puts the image first.
func ParseInput(data []byte, o ParseOptions) (Input, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	tok, err := dec.Token()
	if err != nil {
		return Input{}, err
	}
	if s, ok := tok.(string); ok {
		if err := expectEnd(dec); err != nil {
			return Input{}, err
		}
		return Input{Text: s}, nil
	}
	if d, ok := tok.(json.Delim); !ok || d != '{' {
		return Input{}, errors.New("input must be a string or an object")
	}

	var (
		in      Input
		hasText bool
		order   []string // "text" and media kinds, in key order
		media   = map[string][]Media{}
		seen    = map[string]bool{}
	)
	for dec.More() {
		keyTok, err := dec.Token()
		if err != nil {
			return Input{}, err
		}
		key := keyTok.(string)
		if seen[key] {
			return Input{}, fmt.Errorf("duplicate field %q", key)
		}
		seen[key] = true
		switch key {
		case "id", "text", "task", "title", "prompt":
			var s string
			if err := dec.Decode(&s); err != nil {
				return Input{}, fmt.Errorf("%s: %w", key, err)
			}
			switch key {
			case "id":
				in.ID = s
			case "text":
				in.Text, hasText = s, true
				order = append(order, key)
			case "task":
				in.Task = Task(s)
				if _, err := in.Task.Prompt(); err != nil {
					return Input{}, err
				}
			case "title":
				in.Title = s
			case "prompt":
				in.Prompt = s
			}
		case "literal":
			if err := dec.Decode(&in.Literal); err != nil {
				return Input{}, fmt.Errorf("literal: %w", err)
			}
		case "image", "audio", "video":
			var raw json.RawMessage
			if err := dec.Decode(&raw); err != nil {
				return Input{}, err
			}
			items, err := mediaValues(raw, key, o)
			if err != nil {
				return Input{}, err
			}
			media[key] = items
			order = append(order, key)
		default:
			return Input{}, fmt.Errorf("unknown field %q", key)
		}
	}
	if _, err := dec.Token(); err != nil { // closing }
		return Input{}, err
	}
	if err := expectEnd(dec); err != nil {
		return Input{}, err
	}

	// Media keep key order, and within a key their array order, which is also
	// the order placeholders of one kind consume them.
	for _, k := range order {
		in.Media = append(in.Media, media[k]...)
	}
	if hasText && !in.Literal && placeholderRE.MatchString(in.Text) {
		return in, nil
	}
	if in.Literal {
		return in, nil // Literal puts media after the text whatever the key order
	}
	// Without placeholders, spell the key order out as placeholders.
	var b strings.Builder
	for _, k := range order {
		if k == "text" {
			b.WriteString(in.Text)
			continue
		}
		for range media[k] {
			b.WriteString("<|" + k + "|>")
		}
	}
	in.Text = b.String()
	return in, nil
}

func expectEnd(dec *json.Decoder) error {
	if _, err := dec.Token(); err != io.EOF {
		return errors.New("trailing data after the input")
	}
	return nil
}

func mediaValues(raw json.RawMessage, key string, o ParseOptions) ([]Media, error) {
	var list []string
	if err := json.Unmarshal(raw, &list); err != nil {
		var one string
		if err := json.Unmarshal(raw, &one); err != nil {
			return nil, fmt.Errorf("%s: want a string or an array of strings", key)
		}
		list = []string{one}
	}
	kind := map[string]MediaKind{"image": MediaImage, "audio": MediaAudio, "video": MediaVideo}[key]
	out := make([]Media, 0, len(list))
	for i, v := range list {
		m, err := mediaValue(v, kind, o)
		if err != nil {
			return nil, fmt.Errorf("%s[%d]: %w", key, i, err)
		}
		out = append(out, m)
	}
	return out, nil
}

// mediaValue reads a data: URI ("data:[<type>][;base64],<data>") or a path.
func mediaValue(v string, kind MediaKind, o ParseOptions) (Media, error) {
	if rest, ok := strings.CutPrefix(v, "data:"); ok {
		meta, payload, ok := strings.Cut(rest, ",")
		if !ok || !strings.HasSuffix(meta, ";base64") {
			return Media{}, errors.New("data URI must be base64 (data:<type>;base64,...)")
		}
		data, err := base64.StdEncoding.DecodeString(payload)
		if err != nil {
			return Media{}, fmt.Errorf("data URI: %w", err)
		}
		return Media{Kind: kind, Data: data}, nil
	}
	if !o.AllowPaths {
		return Media{}, errors.New("file paths are not allowed here; send a data: URI")
	}
	if v == "" {
		return Media{}, errors.New("empty path")
	}
	return Media{Kind: kind, Path: v}, nil
}
