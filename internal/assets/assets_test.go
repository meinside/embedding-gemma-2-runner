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
	"strings"
	"testing"
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
