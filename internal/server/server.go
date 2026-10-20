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
	"time"

	eg2 "github.com/meinside/embedding-gemma-2-runner"
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
	// MaxAudio and MaxVideo, if set, reject longer media before anything is
	// decoded (eg2.MediaDuration). mtmd decodes audio whole, so a long
	// low-bitrate MP3 takes gigabytes; ffmpeg decodes a video whole to pick
	// its 32 frames. The model itself takes about five minutes of audio at
	// most (25 tokens a second, 8192 tokens).
	MaxAudio time.Duration
	MaxVideo time.Duration
	// CORSOrigin, if set, lets browser pages from this origin ("*" for any)
	// call the API: it answers preflight requests and names the origin in
	// the responses.
	CORSOrigin string
	// BatchTexts hands consecutive text-only inputs of a request to one
	// Embed call, for an Embedder opened with eg2.Options.BatchSeqs.
	BatchTexts bool
	Help       string
	Logger     *slog.Logger
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
	var h http.Handler = mux
	if o.CORSOrigin != "" {
		h = s.cors(h)
	}
	return s.logged(h)
}

// cors adds the headers a browser needs and answers its preflight requests,
// which carry no credentials and so come before auth.
func (s *server) cors(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		h := w.Header()
		h.Set("Access-Control-Allow-Origin", s.opts.CORSOrigin)
		if s.opts.CORSOrigin != "*" {
			h.Add("Vary", "Origin")
		}
		if r.Method == http.MethodOptions {
			h.Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
			h.Set("Access-Control-Allow-Headers", "Authorization, Content-Type")
			h.Set("Access-Control-Max-Age", "86400")
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
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
		err = s.checkMedia(r.Context(), inputs)
	}
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error(), "input")
		return
	}
	resp := response{Object: "list", Model: ModelID, Data: make([]embedding, len(inputs))}
	for i := 0; i < len(inputs); {
		// One at a time: the request is all or nothing, as OpenAI's is, so
		// the inputs after a failing one are not embedded for nothing. With
		// BatchTexts a run of text-only inputs goes together.
		j := i + 1
		if s.opts.BatchTexts && len(inputs[i].Media) == 0 {
			for j < len(inputs) && len(inputs[j].Media) == 0 {
				j++
			}
		}
		results, err := s.emb.Embed(r.Context(), inputs[i:j]...)
		if err != nil {
			if r.Context().Err() != nil {
				return // client went away
			}
			writeError(w, http.StatusInternalServerError, err.Error(), "")
			return
		}
		for k, res := range results {
			idx := i + k
			if res.Err != nil {
				writeError(w, http.StatusBadRequest, fmt.Sprintf("input[%d]: %v", idx, res.Err), "input")
				return
			}
			vec, err := eg2.Truncate(res.Embedding, dim)
			if err != nil {
				writeError(w, http.StatusInternalServerError, fmt.Sprintf("input[%d]: %v", idx, err), "")
				return
			}
			var out any = vec
			if req.EncodingFormat == "base64" {
				out = encodeBase64(vec)
			}
			resp.Data[idx] = embedding{Object: "embedding", Index: idx, Embedding: out}
			resp.Usage.PromptTokens += res.Tokens
		}
		i = j
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

// checkMedia rejects audio and video inputs longer than MaxAudio and MaxVideo.
func (s *server) checkMedia(ctx context.Context, inputs []eg2.Input) error {
	limits := map[eg2.MediaKind]time.Duration{eg2.MediaAudio: s.opts.MaxAudio, eg2.MediaVideo: s.opts.MaxVideo}
	for i, in := range inputs {
		for j, m := range in.Media {
			limit := limits[m.Kind]
			if limit <= 0 {
				continue
			}
			d, ok, err := eg2.MediaDuration(ctx, m)
			if err != nil {
				return fmt.Errorf("input[%d]: %s %d: %w", i, m.Kind, j+1, err)
			}
			if ok && d > limit {
				return fmt.Errorf("input[%d]: %s %d is %s long, more than the %s allowed",
					i, m.Kind, j+1, d.Round(time.Second), limit)
			}
		}
	}
	return nil
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
