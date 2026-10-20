package assets

import (
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"testing"
)

// The llama.cpp release pinned here must be the one the yzma in go.mod was
// written against: yzma's bindings follow that release's C API. The pin
// lives in yzma's pkg/download/version.go, which is not imported (it pulls
// in go-getter and cloud SDKs), so read it from the module cache.
func TestLlamaPinMatchesYzma(t *testing.T) {
	goBin := filepath.Join(runtime.GOROOT(), "bin", "go")
	if _, err := os.Stat(goBin); err != nil {
		if goBin, err = exec.LookPath("go"); err != nil {
			t.Skip("no go binary to locate the yzma module")
		}
	}
	out, err := exec.Command(goBin, "list", "-m", "-f", "{{.Dir}}", "github.com/hybridgroup/yzma").Output()
	if err != nil {
		t.Skipf("go list: %v", err)
	}
	src, err := os.ReadFile(filepath.Join(strings.TrimSpace(string(out)), "pkg", "download", "version.go"))
	if err != nil {
		t.Fatal(err)
	}
	m := regexp.MustCompile(`DefaultVersion\s*=\s*"([^"@]+)@sha256:([0-9a-f]+)"`).FindSubmatch(src)
	if m == nil {
		t.Fatal("DefaultVersion not found in yzma's version.go; its layout changed")
	}
	if got := string(m[1]); got != LlamaVersion {
		t.Errorf("yzma installs llama.cpp %s, libAssets pin %s: update LlamaVersion and the digests (see libAssets)", got, LlamaVersion)
	}
	if got := string(m[2]); got != llamaManifestSHA256 {
		t.Errorf("yzma's manifest digest is %s, llamaManifestSHA256 is %s: check the manifest and copy the digests", got, llamaManifestSHA256)
	}
}
