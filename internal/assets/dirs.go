// Package assets locates and fetches what eg2 needs at run time: the pinned
// model files and llama.cpp shared libraries, both kept in a cache directory.
package assets

import (
	"os"
	"path/filepath"
	"runtime"
)

const cacheDirName = "eg2"

// Dirs are the cache locations eg2 uses.
type Dirs struct {
	Root   string // everything below is inside it, except Libffi
	Models string
	Libs   string
	// Libffi is where github.com/jupiterrider/ffi extracts its embedded libffi.
	// That package decides it in its own init, before main, from
	// os.UserCacheDir; $XDG_CACHE_HOME and -cache do not move it. Empty on
	// platforms where it loads the system libffi instead (Linux, Windows arm64).
	Libffi string
}

// CacheDirs returns the cache locations under root, or under the default cache
// directory when root is empty.
func CacheDirs(root string) (Dirs, error) {
	if root == "" {
		var err error
		if root, err = defaultRoot(); err != nil {
			return Dirs{}, err
		}
	}
	d := Dirs{
		Root:   root,
		Models: filepath.Join(root, "models"),
		Libs:   filepath.Join(root, "lib"),
	}
	if runtime.GOOS == "darwin" || (runtime.GOOS == "windows" && runtime.GOARCH == "amd64") {
		if uc, err := os.UserCacheDir(); err == nil {
			d.Libffi = filepath.Join(uc, "github.com", "jupiterrider", "ffi")
		}
	}
	return d, nil
}

// defaultRoot prefers $XDG_CACHE_HOME on every OS. os.UserCacheDir ignores it
// on macOS and Windows, and fails on Linux when it is relative, which the XDG
// spec says to ignore instead.
func defaultRoot() (string, error) {
	if dir := os.Getenv("XDG_CACHE_HOME"); filepath.IsAbs(dir) {
		return filepath.Join(dir, cacheDirName), nil
	}
	switch runtime.GOOS {
	case "darwin", "windows", "ios", "plan9":
		dir, err := os.UserCacheDir()
		if err != nil {
			return "", err
		}
		return filepath.Join(dir, cacheDirName), nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".cache", cacheDirName), nil
}
