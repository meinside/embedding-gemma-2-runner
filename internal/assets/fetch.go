package assets

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
)

// Progress reports a download: done of total bytes of name. total is -1 when
// the server does not say.
type Progress func(name string, done, total int64)

// ErrOffline is returned when a file is missing and downloading is not allowed.
var ErrOffline = errors.New("not in the cache and downloads are disabled")

// pinned is a file whose content is fixed by its SHA-256.
type pinned struct {
	Name   string // file name to store it under
	URL    string
	SHA256 string
	Size   int64 // 0 when unknown
}

// fetchTo downloads f into dir, checking its digest before it gets its final
// name, so a file at that name is always complete and verified.
func fetchTo(ctx context.Context, dir string, f pinned, progress Progress) (string, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	tmp, err := os.CreateTemp(dir, f.Name+".*.part")
	if err != nil {
		return "", err
	}
	defer os.Remove(tmp.Name())
	defer tmp.Close()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, f.URL, nil)
	if err != nil {
		return "", err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("download %s: %w", f.Name, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("download %s: %s", f.Name, resp.Status)
	}

	total := resp.ContentLength
	if total < 0 && f.Size > 0 {
		total = f.Size
	}
	h := sha256.New()
	var src io.Reader = resp.Body
	if progress != nil {
		src = &progressReader{r: src, name: f.Name, total: total, report: progress}
	}
	n, err := io.Copy(io.MultiWriter(tmp, h), src)
	if err != nil {
		return "", fmt.Errorf("download %s: %w", f.Name, err)
	}
	if f.Size > 0 && n != f.Size {
		return "", fmt.Errorf("%s: got %d bytes, want %d", f.Name, n, f.Size)
	}
	if got := hex.EncodeToString(h.Sum(nil)); got != f.SHA256 {
		return "", fmt.Errorf("%s: sha256 %s, want %s", f.Name, got, f.SHA256)
	}
	if err := tmp.Close(); err != nil {
		return "", err
	}
	if err := os.Chmod(tmp.Name(), 0o644); err != nil { // CreateTemp makes it 0600
		return "", err
	}
	dst := filepath.Join(dir, f.Name)
	if err := os.Rename(tmp.Name(), dst); err != nil {
		return "", err
	}
	return dst, nil
}

// copyVerified copies src into dir as f.Name, checking it against f like a download.
func copyVerified(dir, src string, f pinned) (string, error) {
	in, err := os.Open(src)
	if err != nil {
		return "", err
	}
	defer in.Close()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	tmp, err := os.CreateTemp(dir, f.Name+".*.part")
	if err != nil {
		return "", err
	}
	defer os.Remove(tmp.Name())
	defer tmp.Close()

	h := sha256.New()
	if _, err := io.Copy(io.MultiWriter(tmp, h), in); err != nil {
		return "", err
	}
	if got := hex.EncodeToString(h.Sum(nil)); got != f.SHA256 {
		return "", fmt.Errorf("%s: sha256 %s, want %s", src, got, f.SHA256)
	}
	if err := tmp.Close(); err != nil {
		return "", err
	}
	if err := os.Chmod(tmp.Name(), 0o644); err != nil {
		return "", err
	}
	dst := filepath.Join(dir, f.Name)
	return dst, os.Rename(tmp.Name(), dst)
}

type progressReader struct {
	r      io.Reader
	name   string
	done   int64
	total  int64
	last   int64
	report Progress
}

func (p *progressReader) Read(b []byte) (int, error) {
	n, err := p.r.Read(b)
	p.done += int64(n)
	// every 8 MiB and at the end, so a terminal display stays cheap
	if p.done-p.last >= 8<<20 || errors.Is(err, io.EOF) {
		p.last = p.done
		p.report(p.name, p.done, p.total)
	}
	return n, err
}
