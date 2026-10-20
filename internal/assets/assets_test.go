package assets

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestCacheDirsPrefersXDG(t *testing.T) {
	xdg := filepath.Join(t.TempDir(), "xdg-cache")
	t.Setenv("XDG_CACHE_HOME", xdg)
	d, err := CacheDirs("")
	if err != nil {
		t.Fatal(err)
	}
	if want := filepath.Join(xdg, cacheDirName); d.Root != want {
		t.Errorf("root %q, want %q", d.Root, want)
	}
}

func TestCacheDirsIgnoresRelativeXDG(t *testing.T) {
	t.Setenv("XDG_CACHE_HOME", "relative/cache")
	d, err := CacheDirs("")
	if err != nil {
		t.Fatal(err)
	}
	if !filepath.IsAbs(d.Root) || strings.HasPrefix(d.Root, "relative") {
		t.Errorf("root %q, want an absolute default", d.Root)
	}
}

type entry struct {
	name, body, link string
}

func tarGz(t *testing.T, entries []entry) string {
	t.Helper()
	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	tw := tar.NewWriter(gz)
	for _, e := range entries {
		h := &tar.Header{Name: e.name, Mode: 0o755, Size: int64(len(e.body)), Typeflag: tar.TypeReg}
		if e.link != "" {
			h = &tar.Header{Name: e.name, Linkname: e.link, Typeflag: tar.TypeSymlink}
		}
		if err := tw.WriteHeader(h); err != nil {
			t.Fatal(err)
		}
		if e.link == "" {
			tw.Write([]byte(e.body))
		}
	}
	tw.Close()
	gz.Close()
	p := filepath.Join(t.TempDir(), "a.tar.gz")
	if err := os.WriteFile(p, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestUnpackKeepsLibrariesFlat(t *testing.T) {
	archive := tarGz(t, []entry{
		{name: "llama-b1/libllama.0.7.0.dylib", body: "lib"},
		{name: "llama-b1/libllama.dylib", link: "libllama.0.7.0.dylib"},
		{name: "llama-b1/libggml.so.0", body: "so"},
		{name: "llama-b1/LICENSE", body: "mit"},
		{name: "llama-b1/llama-cli", body: "exe"},
	})
	dir := t.TempDir()
	if err := unpack(archive, dir); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"libllama.0.7.0.dylib", "libllama.dylib", "libggml.so.0", "LICENSE"} {
		if _, err := os.Stat(filepath.Join(dir, name)); err != nil {
			t.Errorf("%s missing: %v", name, err)
		}
	}
	if _, err := os.Stat(filepath.Join(dir, "llama-cli")); err == nil {
		t.Error("executable extracted")
	}
}

func TestUnpackRejectsEscapingSymlink(t *testing.T) {
	archive := tarGz(t, []entry{{name: "d/libx.dylib", link: "../../etc/passwd"}})
	if err := unpack(archive, t.TempDir()); err == nil {
		t.Fatal("escaping symlink accepted")
	}
}

func TestFetchRejectsWrongDigest(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte("tampered"))
	}))
	defer srv.Close()
	dir := t.TempDir()
	sum := sha256.Sum256([]byte("original"))
	_, err := fetchTo(context.Background(), dir, pinned{Name: "m.gguf", URL: srv.URL, SHA256: hex.EncodeToString(sum[:])}, nil)
	if err == nil {
		t.Fatal("wrong digest accepted")
	}
	if left, _ := os.ReadDir(dir); len(left) != 0 {
		t.Errorf("left files behind: %v", left)
	}
}

func TestLibrariesOffline(t *testing.T) {
	d, _ := CacheDirs(t.TempDir())
	ps := Processors()
	if len(ps) == 0 {
		t.Skip("no llama.cpp build for this platform")
	}
	if _, _, err := Libraries(context.Background(), d, LibOptions{Processor: ps[0], Offline: true}); err == nil {
		t.Fatal("offline install of a missing build succeeded")
	}
}

// TestLibrariesInstall downloads a real llama.cpp build; set EG2_NETWORK_TESTS=1.
func TestLibrariesInstall(t *testing.T) {
	if os.Getenv("EG2_NETWORK_TESTS") == "" {
		t.Skip("set EG2_NETWORK_TESTS=1 to download llama.cpp")
	}
	d, _ := CacheDirs(t.TempDir())
	dir, p, err := Libraries(context.Background(), d, LibOptions{Processor: DefaultProcessor(false)})
	if err != nil {
		t.Fatal(err)
	}
	for _, lib := range []string{"llama", "mtmd", "ggml", "ggml-base"} {
		matches, _ := filepath.Glob(filepath.Join(dir, "*"+lib+".*"))
		if len(matches) == 0 {
			t.Errorf("%s build lacks %s", p, lib)
		}
	}
	again, _, err := Libraries(context.Background(), d, LibOptions{Processor: p, Offline: true})
	if err != nil || again != dir {
		t.Errorf("second call: %q, %v; want the installed %q", again, err, dir)
	}
}

// An interrupted download leaves a .part file that the next call resumes
// with a Range request.
func TestFetchResumes(t *testing.T) {
	content := bytes.Repeat([]byte("0123456789abcdef"), 4096)
	sum := sha256.Sum256(content)
	var ranges []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ranges = append(ranges, r.Header.Get("Range"))
		if len(ranges) == 1 {
			// 20000 bytes, then the connection drops
			w.Header().Set("Content-Length", strconv.Itoa(len(content)))
			w.Write(content[:20000])
			w.(http.Flusher).Flush()
			conn, _, err := w.(http.Hijacker).Hijack()
			if err == nil {
				conn.Close()
			}
			return
		}
		http.ServeContent(w, r, "f", time.Time{}, bytes.NewReader(content))
	}))
	defer srv.Close()
	dir := t.TempDir()
	f := pinned{Name: "m.gguf", URL: srv.URL, SHA256: hex.EncodeToString(sum[:]), Size: int64(len(content))}
	if _, err := fetchTo(context.Background(), dir, f, nil); err == nil {
		t.Fatal("interrupted download succeeded")
	}
	part := filepath.Join(dir, "m.gguf.part")
	if fi, err := os.Stat(part); err != nil || fi.Size() != 20000 {
		t.Fatalf("part file after the interruption: %v %v", fi, err)
	}
	dst, err := fetchTo(context.Background(), dir, f, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got, _ := os.ReadFile(dst); !bytes.Equal(got, content) {
		t.Error("resumed file differs from the content")
	}
	if len(ranges) != 2 || ranges[1] != "bytes=20000-" {
		t.Errorf("ranges asked for: %q, want the rest after 20000", ranges)
	}
	if _, err := os.Stat(part); !os.IsNotExist(err) {
		t.Error("part file left behind")
	}
}

// A server that ignores the range answers 200 with everything: the partial
// file is dropped rather than appended to.
func TestFetchRestartsWhenRangeIgnored(t *testing.T) {
	content := []byte("the whole file")
	sum := sha256.Sum256(content)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.Write(content) }))
	defer srv.Close()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "m.gguf.part"), []byte("the wh"), 0o644); err != nil {
		t.Fatal(err)
	}
	dst, err := fetchTo(context.Background(), dir, pinned{Name: "m.gguf", URL: srv.URL, SHA256: hex.EncodeToString(sum[:])}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got, _ := os.ReadFile(dst); !bytes.Equal(got, content) {
		t.Errorf("got %q", got)
	}
}

// A download that stops delivering data ends with an error instead of
// waiting forever.
func TestFetchStall(t *testing.T) {
	old := fetchStallTimeout
	fetchStallTimeout = 200 * time.Millisecond
	t.Cleanup(func() { fetchStallTimeout = old })
	done := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte("abc"))
		w.(http.Flusher).Flush()
		<-done
	}))
	t.Cleanup(srv.Close)
	t.Cleanup(func() { close(done) }) // before srv.Close, which waits for the handler
	_, err := fetchTo(context.Background(), t.TempDir(), pinned{Name: "m.gguf", URL: srv.URL, SHA256: "00"}, nil)
	if err == nil || !strings.Contains(err.Error(), "no data for") {
		t.Fatalf("got %v, want a stall error", err)
	}
}

func TestPrunable(t *testing.T) {
	d, _ := CacheDirs(t.TempDir())
	cur := LibDir(d, CPU)
	mk := func(parts ...string) string {
		p := filepath.Join(parts...)
		if err := os.MkdirAll(p, 0o755); err != nil {
			t.Fatal(err)
		}
		return p
	}
	mk(cur)
	mk(d.Libs, "v0.6.0", "darwin-arm64", "metal")
	staging := mk(filepath.Dir(cur), "cpu.123.part")
	mk(ModelDir(d))
	if err := os.WriteFile(filepath.Join(ModelDir(d), "x.gguf.part"), nil, 0o644); err != nil {
		t.Fatal(err)
	}
	oldRev := mk(filepath.Dir(ModelDir(d)), "0000000")
	otherRepo := mk(d.Models, "ggml-org", "other-GGUF")
	otherOwner := mk(d.Models, "someone")

	got, err := Prunable(d)
	if err != nil {
		t.Fatal(err)
	}
	// whole trees of other versions, nothing of the current pins
	want := []string{filepath.Join(d.Libs, "v0.6.0"), staging, oldRev, otherRepo, otherOwner}
	sort.Strings(want)
	if strings.Join(got, "\n") != strings.Join(want, "\n") {
		t.Errorf("got:\n%s\nwant:\n%s", strings.Join(got, "\n"), strings.Join(want, "\n"))
	}
}
