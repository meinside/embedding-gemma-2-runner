package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"io"
	"strconv"
)

// record is one line of `eg2 embed` output.
type record struct {
	ID        string    `json:"id"`
	Source    string    `json:"source"`
	Modality  string    `json:"modality,omitempty"`
	Tokens    int       `json:"tokens,omitempty"`
	Dim       int       `json:"dim,omitempty"`
	Task      string    `json:"task,omitempty"`
	Model     string    `json:"model,omitempty"`
	Embedding floatList `json:"embedding,omitempty"`
	Error     string    `json:"error,omitempty"`
}

// floatList encodes as the shortest decimal that reads back as the same
// float32, about half the size of encoding/json's float64 rendering.
type floatList []float32

func (l floatList) MarshalJSON() ([]byte, error) {
	b := make([]byte, 0, len(l)*12+2)
	b = append(b, '[')
	for i, v := range l {
		if i > 0 {
			b = append(b, ',')
		}
		b = strconv.AppendFloat(b, float64(v), 'g', -1, 32)
	}
	return append(b, ']'), nil
}

// recordWriter streams records as JSON lines, or with format "json" writes a
// single record as an object and several as an array.
type recordWriter struct {
	w      *bufio.Writer
	format string
	held   *record // "json": the first record, until a second shows whether to use an array
	n      int
}

func newRecordWriter(w io.Writer, format string) *recordWriter {
	return &recordWriter{w: bufio.NewWriter(w), format: format}
}

func (rw *recordWriter) write(r record) error {
	rw.n++
	if rw.format == "jsonl" {
		if err := rw.encode(r); err != nil {
			return err
		}
		if err := rw.w.WriteByte('\n'); err != nil {
			return err
		}
		return rw.w.Flush() // a reader sees each result as soon as it exists
	}
	switch rw.n {
	case 1:
		rw.held = &r
		return nil
	case 2:
		rw.w.WriteString("[\n")
		if err := rw.encode(*rw.held); err != nil {
			return err
		}
		rw.held = nil
	}
	rw.w.WriteString(",\n")
	return rw.encode(r)
}

func (rw *recordWriter) close() error {
	if rw.format == "json" {
		switch {
		case rw.held != nil:
			if err := rw.encode(*rw.held); err != nil {
				return err
			}
			rw.w.WriteByte('\n')
		case rw.n > 1:
			rw.w.WriteString("\n]\n")
		case rw.n == 0:
			rw.w.WriteString("[]\n")
		}
	}
	return rw.w.Flush()
}

func (rw *recordWriter) encode(r record) error {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false) // error messages quote <|image|>
	if err := enc.Encode(r); err != nil {
		return err
	}
	_, err := rw.w.Write(bytes.TrimRight(buf.Bytes(), "\n"))
	return err
}
