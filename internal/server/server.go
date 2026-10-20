// Package server serves embeddings over HTTP in the shape of OpenAI's
// /v1/embeddings, extended with media inputs and task prefixes.
package server

import (
	"bytes"
	"context"
	"crypto/subtle"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net/http"
	"os"
	"time"

	eg2 "github.com/meinside/embedding-gemma-2-runner"
	"github.com/meinside/embedding-gemma-2-runner/internal/media"
)

// ModelID is the model name the API reports. Requests may name any model:
// there is only one.
const ModelID = "embeddinggemma-2"

// Embedder is what the server needs from *eg2.Embedder.
type Embedder interface {
	Embed(ctx context.Context, inputs ...eg2.Input) ([]eg2.Result, error)
	Info() eg2.Info
}

// Options configure New.
type Options struct {
	// AllowLocalFiles lets inputs name files on the server's disk. Off, media
	// must come as data: URIs, so a client cannot read the server's files.
	AllowLocalFiles bool
	// APIKey, if set, must come as "Authorization: Bearer <key>".
	APIKey    string
	MaxBody   int64 // bytes per request
	MaxInputs int   // inputs per request
	// MaxAudio, if set, rejects audio longer than this, read from its
	// headers before anything is decoded: mtmd decodes a file whole, and a
	// long low-bitrate MP3 takes gigabytes. The model itself takes about
	// five minutes at most (25 tokens a second, 8192 tokens).
	MaxAudio time.Duration
	Help     string
	Logger   *slog.Logger
}

type server struct {
	emb  Embedder
	opts Options
}

// New returns the HTTP handler.
func New(e Embedder, o Options) http.Handler {
	if o.MaxBody <= 0 {
		o.MaxBody = 64 << 20
	}
	if o.MaxInputs <= 0 {
		o.MaxInputs = 256
	}
	if o.Logger == nil {
		o.Logger = slog.New(slog.DiscardHandler)
	}
	s := &server{emb: e, opts: o}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/embeddings", s.auth(s.embeddings))
	mux.HandleFunc("GET /v1/models", s.auth(s.models))
	mux.HandleFunc("GET /health", s.health)
	mux.HandleFunc("GET /{$}", s.help)
	return s.logged(mux)
}

type request struct {
	Input          json.RawMessage `json:"input"`
	Model          string          `json:"model"`
	Dimensions     int             `json:"dimensions"`
	EncodingFormat string          `json:"encoding_format"`
	User           string          `json:"user"` // accepted for OpenAI clients, unused
	// extensions: defaults for inputs with text and no prefix of their own
	Task   string `json:"task"`
	Title  string `json:"title"`
	Prompt string `json:"prompt"`
}

type embedding struct {
	Object    string `json:"object"`
	Index     int    `json:"index"`
	Embedding any    `json:"embedding"` // []float32, or a base64 string
}

type response struct {
	Object string      `json:"object"`
	Data   []embedding `json:"data"`
	Model  string      `json:"model"`
	Usage  struct {
		PromptTokens int `json:"prompt_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage"`
}

func (s *server) embeddings(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, s.opts.MaxBody)
	var req request
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&req); err != nil {
		var tooBig *http.MaxBytesError
		if errors.As(err, &tooBig) {
			writeError(w, http.StatusRequestEntityTooLarge, fmt.Sprintf("request larger than %d bytes", tooBig.Limit), "")
			return
		}
		writeError(w, http.StatusBadRequest, "invalid JSON: "+err.Error(), "")
		return
	}
	dim := req.Dimensions
	if dim == 0 {
		dim = eg2.Dims
	}
	if err := eg2.CheckDim(dim); err != nil {
		writeError(w, http.StatusBadRequest, err.Error(), "dimensions")
		return
	}
	if req.EncodingFormat != "" && req.EncodingFormat != "float" && req.EncodingFormat != "base64" {
		writeError(w, http.StatusBadRequest, `encoding_format must be "float" or "base64"`, "encoding_format")
		return
	}
	if _, err := eg2.Task(req.Task).Prompt(); err != nil {
		writeError(w, http.StatusBadRequest, err.Error(), "task")
		return
	}

	inputs, err := s.parseInputs(req)
	if err == nil {
		err = s.checkAudio(inputs)
	}
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error(), "input")
		return
	}
	resp := response{Object: "list", Model: ModelID, Data: make([]embedding, len(inputs))}
	for i, in := range inputs {
		// One at a time: the request is all or nothing, as OpenAI's is, so
		// the inputs after a failing one are not embedded for nothing.
		results, err := s.emb.Embed(r.Context(), in)
		if err != nil {
			if r.Context().Err() != nil {
				return // client went away
			}
			writeError(w, http.StatusInternalServerError, err.Error(), "")
			return
		}
		res := results[0]
		if res.Err != nil {
			writeError(w, http.StatusBadRequest, fmt.Sprintf("input[%d]: %v", i, res.Err), "input")
			return
		}
		vec, err := eg2.Truncate(res.Embedding, dim)
		if err != nil {
			writeError(w, http.StatusInternalServerError, fmt.Sprintf("input[%d]: %v", i, err), "")
			return
		}
		var out any = vec
		if req.EncodingFormat == "base64" {
			out = encodeBase64(vec)
		}
		resp.Data[i] = embedding{Object: "embedding", Index: i, Embedding: out}
		resp.Usage.PromptTokens += res.Tokens
	}
	resp.Usage.TotalTokens = resp.Usage.PromptTokens
	writeJSON(w, http.StatusOK, resp)
}

// parseInputs accepts what OpenAI does for text (a string or an array of
// strings) plus eg2 input objects, alone or in the array.
func (s *server) parseInputs(req request) ([]eg2.Input, error) {
	raw := bytes.TrimSpace(req.Input)
	if len(raw) == 0 {
		return nil, errors.New("input is required")
	}
	var items []json.RawMessage
	if raw[0] == '[' {
		if err := json.Unmarshal(raw, &items); err != nil {
			return nil, err
		}
		if len(items) == 0 {
			return nil, errors.New("input is empty")
		}
	} else {
		items = []json.RawMessage{raw}
	}
	if len(items) > s.opts.MaxInputs {
		return nil, fmt.Errorf("%d inputs, more than the %d allowed per request", len(items), s.opts.MaxInputs)
	}
	po := eg2.ParseOptions{AllowPaths: s.opts.AllowLocalFiles}
	inputs := make([]eg2.Input, len(items))
	for i, item := range items {
		if t := bytes.TrimSpace(item); len(t) > 0 && t[0] == '[' {
			return nil, fmt.Errorf("input[%d]: token arrays are not supported; send text", i)
		}
		in, err := eg2.ParseInput(item, po)
		if err != nil {
			return nil, fmt.Errorf("input[%d]: %w", i, err)
		}
		if in.HasText() && in.Task == eg2.TaskNone && in.Prompt == "" && in.Title == "" {
			in.Task, in.Title, in.Prompt = eg2.Task(req.Task), req.Title, req.Prompt
		}
		inputs[i] = in
	}
	return inputs, nil
}

// audioHead is how much of an audio file on disk is read for its duration:
// room for an ID3 tag with cover art before the first MP3 frame.
const audioHead = 1 << 20

// checkAudio rejects audio inputs longer than MaxAudio.
func (s *server) checkAudio(inputs []eg2.Input) error {
	if s.opts.MaxAudio <= 0 {
		return nil
	}
	for i, in := range inputs {
		for j, m := range in.Media {
			if m.Kind != eg2.MediaAudio {
				continue
			}
			head, size := m.Data, int64(len(m.Data))
			if m.Data == nil {
				var err error
				if head, size, err = readHead(m.Path, audioHead); err != nil {
					return fmt.Errorf("input[%d]: %w", i, err)
				}
			}
			if d, ok := media.AudioDuration(head, size); ok && d > s.opts.MaxAudio {
				return fmt.Errorf("input[%d]: audio %d is %s long, more than the %s allowed",
					i, j+1, d.Round(time.Second), s.opts.MaxAudio)
			}
		}
	}
	return nil
}

func readHead(path string, n int64) ([]byte, int64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, err
	}
	defer f.Close()
	fi, err := f.Stat()
	if err != nil {
		return nil, 0, err
	}
	head, err := io.ReadAll(io.LimitReader(f, n))
	return head, fi.Size(), err
}

// encodeBase64 is OpenAI's base64 format: little-endian float32s.
func encodeBase64(v []float32) string {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(x))
	}
	return base64.StdEncoding.EncodeToString(b)
}

func (s *server) models(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"object": "list",
		"data":   []map[string]any{{"id": ModelID, "object": "model", "owned_by": "google"}},
	})
}

func (s *server) health(w http.ResponseWriter, _ *http.Request) {
	info := s.emb.Info()
	writeJSON(w, http.StatusOK, map[string]any{
		"status":       "ok",
		"model":        info.Model,
		"weights":      info.Weights,
		"device":       info.Device,
		"media_loaded": info.MediaLoaded,
		"media_device": info.MediaDevice,
		"video":        info.FFmpeg,
	})
}

func (s *server) help(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	io.WriteString(w, "eg2 embeddings server\n\n"+
		"POST /v1/embeddings   OpenAI-compatible; also takes the inputs below\n"+
		"GET  /v1/models\n"+
		"GET  /health\n\n"+s.opts.Help)
}

func (s *server) auth(next http.HandlerFunc) http.HandlerFunc {
	if s.opts.APIKey == "" {
		return next
	}
	want := []byte("Bearer " + s.opts.APIKey)
	return func(w http.ResponseWriter, r *http.Request) {
		if subtle.ConstantTimeCompare([]byte(r.Header.Get("Authorization")), want) != 1 {
			writeError(w, http.StatusUnauthorized, "missing or wrong API key", "")
			return
		}
		next(w, r)
	}
}

type statusWriter struct {
	http.ResponseWriter
	status int
}

func (w *statusWriter) WriteHeader(code int) {
	w.status = code
	w.ResponseWriter.WriteHeader(code)
}

func (s *server) logged(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t0 := time.Now()
		sw := &statusWriter{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(sw, r)
		s.opts.Logger.Info("request", "method", r.Method, "path", r.URL.Path,
			"status", sw.status, "duration", time.Since(t0).Round(time.Millisecond))
	})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	enc.Encode(v)
}

// writeError answers in OpenAI's error shape.
func writeError(w http.ResponseWriter, status int, msg, param string) {
	var p any
	if param != "" {
		p = param
	}
	typ := "invalid_request_error"
	if status >= 500 {
		typ = "server_error"
	}
	writeJSON(w, status, map[string]any{"error": map[string]any{
		"message": msg, "type": typ, "param": p, "code": nil,
	}})
}
