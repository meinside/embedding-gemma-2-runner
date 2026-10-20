package assets

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"fmt"
	"io"
	"os"
	"path"
	"path/filepath"
	"strings"
)

// unpack extracts the shared libraries and license files of a llama.cpp
// release archive (.tar.gz or .zip) flat into dir. The executables in the
// archives are not needed.
func unpack(archive, dir string) error {
	if strings.HasSuffix(archive, ".zip") {
		return unzip(archive, dir)
	}
	return untar(archive, dir)
}

// keep reports whether an archive entry is worth extracting.
func keep(name string) bool {
	base := path.Base(name)
	return strings.HasPrefix(base, "LICENSE") ||
		strings.HasSuffix(base, ".dylib") || strings.HasSuffix(base, ".dll") ||
		strings.HasSuffix(base, ".so") || strings.Contains(base, ".so.")
}

// flatName maps an entry to a file name directly under dir. Archives keep the
// libraries in one directory (llama-b11669/ for most), so the base name is
// enough; anything that is not a plain name is refused.
func flatName(name string) (string, error) {
	base := path.Base(name)
	if base == "." || base == ".." || base == "/" || strings.ContainsAny(base, `/\`) {
		return "", fmt.Errorf("bad entry name %q", name)
	}
	return base, nil
}

func untar(archive, dir string) error {
	f, err := os.Open(archive)
	if err != nil {
		return err
	}
	defer f.Close()
	gz, err := gzip.NewReader(f)
	if err != nil {
		return err
	}
	tr := tar.NewReader(gz)
	for {
		h, err := tr.Next()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		if !keep(h.Name) {
			continue
		}
		name, err := flatName(h.Name)
		if err != nil {
			return err
		}
		dst := filepath.Join(dir, name)
		switch h.Typeflag {
		case tar.TypeReg:
			if err := writeFile(dst, tr, h.FileInfo().Mode()); err != nil {
				return err
			}
		case tar.TypeSymlink:
			// versioned library names (libllama.dylib -> libllama.0.dylib)
			if _, err := flatName(h.Linkname); err != nil || h.Linkname != path.Base(h.Linkname) {
				return fmt.Errorf("symlink %s points outside the archive directory: %q", h.Name, h.Linkname)
			}
			if err := os.Symlink(h.Linkname, dst); err != nil {
				return err
			}
		}
	}
}

func unzip(archive, dir string) error {
	zr, err := zip.OpenReader(archive)
	if err != nil {
		return err
	}
	defer zr.Close()
	for _, zf := range zr.File {
		if zf.FileInfo().IsDir() || !keep(zf.Name) {
			continue
		}
		name, err := flatName(zf.Name)
		if err != nil {
			return err
		}
		rc, err := zf.Open()
		if err != nil {
			return err
		}
		err = writeFile(filepath.Join(dir, name), rc, 0o755)
		rc.Close()
		if err != nil {
			return err
		}
	}
	return nil
}

func writeFile(dst string, r io.Reader, mode os.FileMode) error {
	out, err := os.OpenFile(dst, os.O_WRONLY|os.O_CREATE|os.O_EXCL, mode.Perm()|0o600)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, r); err != nil {
		out.Close()
		return err
	}
	return out.Close()
}
