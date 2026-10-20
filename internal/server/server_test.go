package server

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	eg2 "github.com/meinside/embedding-gemma-2-runner"
)

// fake embeds every input as a unit vector along axis 0, and fails inputs
// whose text is "fail".
type fake struct{ got []eg2.Input }

func (f *fake) Embed(_ context.Context, in ...eg2.Input) ([]eg2.Result, error) {
	f.got = append(f.got, in...)
	out := make([]eg2.Result, len(in))
	for i, x := range in {
		if x.Text == "fail" {
			out[i].Err = errors.New("cannot")
			continue
		}
		v := make([]float32, eg2.Dims)
		v[0] = 1
		out[i] = eg2.Result{Embedding: v, Tokens: 3}
	}
	return out, nil
}

func (f *fake) Info() eg2.Info { return eg2.Info{Model: "m", Device: "CPU"} }

func post(t *testing.T, h http.Handler, body string, header ...string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest("POST", "/v1/embeddings", strings.NewReader(body))
	for i := 0; i+1 < len(header); i += 2 {
		req.Header.Set(header[i], header[i+1])
	}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

func TestEmbeddingsOpenAIShape(t *testing.T) {
	f := &fake{}
	rec := post(t, New(f, Options{}), `{"model":"text-embedding-3-small","input":["a","b"],"dimensions":256}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	var resp struct {
		Object string `json:"object"`
		Data   []struct {
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
		Usage struct {
			TotalTokens int `json:"total_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.Object != "list" || len(resp.Data) != 2 || resp.Data[1].Index != 1 ||
		len(resp.Data[0].Embedding) != 256 || resp.Usage.TotalTokens != 6 {
		t.Errorf("unexpected response: %s", rec.Body)
	}
}

func TestEmbeddingsBase64(t *testing.T) {
	rec := post(t, New(&fake{}, Options{}), `{"input":"a","encoding_format":"base64","dimensions":128}`)
	var resp struct {
		Data []struct {
			Embedding string `json:"embedding"`
		} `json:"data"`
	}
	json.Unmarshal(rec.Body.Bytes(), &resp)
	b, err := base64.StdEncoding.DecodeString(resp.Data[0].Embedding)
	if err != nil || len(b) != 128*4 || math.Float32frombits(binary.LittleEndian.Uint32(b)) != 1 {
		t.Errorf("base64 embedding: %d bytes, %v", len(b), err)
	}
}

func TestEmbeddingsMediaAndDefaults(t *testing.T) {
	f := &fake{}
	body := `{"task":"document","input":[
		"plain text",
		{"text":"shoe <|image|>","image":"data:image/png;base64,aGk="},
		{"image":"data:image/png;base64,aGk="},
		{"task":"search-query","text":"q"}]}`
	if rec := post(t, New(f, Options{}), body); rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	want := []eg2.Task{eg2.TaskDocument, eg2.TaskDocument, eg2.TaskNone, eg2.TaskSearchQuery}
	for i, in := range f.got {
		if in.Task != want[i] {
			t.Errorf("input %d: task %q, want %q (media alone get no default)", i, in.Task, want[i])
		}
	}
}

func TestEmbeddingsRejects(t *testing.T) {
	tests := []struct {
		name, body string
		opts       Options
		status     int
		contains   string
	}{
		{"local path", `{"input":{"image":"/etc/passwd"}}`, Options{}, 400, "paths are not allowed"},
		{"one bad input fails the request", `{"input":["ok","fail"]}`, Options{}, 400, "input[1]"},
		{"dimensions", `{"input":"a","dimensions":300}`, Options{}, 400, "dimension 300"},
		{"token arrays", `{"input":[[1,2,3]]}`, Options{}, 400, "token arrays"},
		{"unknown task", `{"input":"a","task":"translate"}`, Options{}, 400, "unknown task"},
		{"too many inputs", `{"input":["a","b","c"]}`, Options{MaxInputs: 2}, 400, "more than the 2"},
		{"too big", `{"input":"` + strings.Repeat("x", 2048) + `"}`, Options{MaxBody: 1024}, 413, "larger than"},
		{"no key", `{"input":"a"}`, Options{APIKey: "k"}, 401, "API key"},
	}
	for _, tt := range tests {
		rec := post(t, New(&fake{}, tt.opts), tt.body)
		if rec.Code != tt.status || !strings.Contains(rec.Body.String(), tt.contains) {
			t.Errorf("%s: %d %s; want %d containing %q", tt.name, rec.Code, rec.Body, tt.status, tt.contains)
		}
	}
}

// A failing input ends the request: the ones after it are not embedded.
func TestEmbeddingsStopAtFailure(t *testing.T) {
	f := &fake{}
	rec := post(t, New(f, Options{}), `{"input":["a","fail","b","c"]}`)
	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "input[1]") {
		t.Fatalf("%d %s; want 400 for input[1]", rec.Code, rec.Body)
	}
	var texts []string
	for _, in := range f.got {
		texts = append(texts, in.Text)
	}
	if got := strings.Join(texts, ","); got != "a,fail" {
		t.Errorf("embedded %s, want a,fail", got)
	}
}

func TestAPIKey(t *testing.T) {
	h := New(&fake{}, Options{APIKey: "k"})
	if rec := post(t, h, `{"input":"a"}`, "Authorization", "Bearer k"); rec.Code != http.StatusOK {
		t.Errorf("with key: %d", rec.Code)
	}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("GET", "/health", nil))
	if rec.Code != http.StatusOK {
		t.Errorf("health needs no key, got %d", rec.Code)
	}
}

func TestHelpPage(t *testing.T) {
	rec := httptest.NewRecorder()
	New(&fake{}, Options{Help: "INPUTS GUIDE"}).ServeHTTP(rec, httptest.NewRequest("GET", "/", nil))
	if !strings.Contains(rec.Body.String(), "INPUTS GUIDE") || !strings.Contains(rec.Body.String(), "/v1/embeddings") {
		t.Errorf("help page: %s", rec.Body)
	}
}

// flacURI is a data: URI of a FLAC STREAMINFO block declaring seconds of
// 16 kHz audio, without the frames: the duration comes from the header alone.
func flacURI(seconds uint32) string {
	b := make([]byte, 8+34)
	copy(b, "fLaC\x00\x00\x00\x22")
	info := b[8:]
	info[10], info[11], info[12] = 16000>>12, 16000>>4&0xff, 16000&0xf<<4
	info[13] = 0xf0 // mono, 16 bits, samples < 2^32
	binary.BigEndian.PutUint32(info[14:], 16000*seconds)
	return "data:audio/flac;base64," + base64.StdEncoding.EncodeToString(b)
}

func TestMaxAudio(t *testing.T) {
	long := `{"input":{"audio":"` + flacURI(3600) + `"}}`
	short := `{"input":{"audio":"` + flacURI(5) + `"}}`
	h := New(&fake{}, Options{MaxAudio: 10 * time.Minute})
	if rec := post(t, h, long); rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "1h0m0s long") {
		t.Errorf("1 hour of audio: %d %s; want 400", rec.Code, rec.Body)
	}
	if rec := post(t, h, short); rec.Code != http.StatusOK {
		t.Errorf("5 s of audio: %d %s; want 200", rec.Code, rec.Body)
	}
	if rec := post(t, New(&fake{}, Options{}), long); rec.Code != http.StatusOK {
		t.Errorf("1 hour of audio without a limit: %d %s; want 200", rec.Code, rec.Body)
	}
}
