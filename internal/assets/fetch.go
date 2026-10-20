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
	"time"
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

// fetchClient waits a bounded time for a server to answer; the body is
// guarded by fetchStallTimeout instead, as a 550 MB download has no useful
// overall deadline.
var fetchClient = func() *http.Client {
	t := http.DefaultTransport.(*http.Transport).Clone()
	t.ResponseHeaderTimeout = 30 * time.Second
	return &http.Client{Transport: t}
}()

// fetchStallTimeout ends a download that delivers nothing for this long. A
// variable so a test can shorten it.
var fetchStallTimeout = 60 * time.Second

// fetchTo downloads f into dir, checking its digest before it gets its final
// name, so a file at that name is always complete and verified. An
// interrupted download leaves <name>.part behind, and the next call asks the
// server for the rest of it.
func fetchTo(ctx context.Context, dir string, f pinned, progress Progress) (string, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	part := filepath.Join(dir, f.Name+".part")
	tmp, err := os.OpenFile(part, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return "", err
	}
	defer tmp.Close()
	fi, err := tmp.Stat()
	if err != nil {
		return "", err
	}
	have := fi.Size()

	ctx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, f.URL, nil)
	if err != nil {
		return "", err
	}
	if have > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", have))
	}
	resp, err := fetchClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("download %s: %w", f.Name, err)
	}
	defer resp.Body.Close()
	switch resp.StatusCode {
	case http.StatusPartialContent:
	case http.StatusOK:
		// The server ignored the range (or there was none): start over.
		if have > 0 {
			if err := tmp.Truncate(0); err != nil {
				return "", err
			}
			have = 0
		}
	default:
		return "", fmt.Errorf("download %s: %s", f.Name, resp.Status)
	}

	total := int64(-1)
	if resp.ContentLength >= 0 {
		total = have + resp.ContentLength
	} else if f.Size > 0 {
		total = f.Size
	}
	errStalled := fmt.Errorf("no data for %v", fetchStallTimeout)
	src := &stallReader{r: resp.Body, timer: time.AfterFunc(fetchStallTimeout, func() { cancel(errStalled) })}
	defer src.timer.Stop()
	var body io.Reader = src
	if progress != nil {
		body = &progressReader{r: body, name: f.Name, done: have, total: total, report: progress}
	}
	if _, err := io.Copy(tmp, body); err != nil {
		if cause := context.Cause(ctx); errors.Is(cause, errStalled) {
			err = cause
		}
		return "", fmt.Errorf("download %s: %w", f.Name, err)
	}
	if err := tmp.Close(); err != nil {
		return "", err
	}
	if err := verify(part, f); err != nil {
		os.Remove(part) // resuming it would not help
		return "", err
	}
	dst := filepath.Join(dir, f.Name)
	if err := os.Rename(part, dst); err != nil {
		return "", err
	}
	return dst, nil
}

// verify checks the size and digest of a downloaded file.
func verify(path string, f pinned) error {
	in, err := os.Open(path)
	if err != nil {
		return err
	}
	defer in.Close()
	h := sha256.New()
	n, err := io.Copy(h, in)
	if err != nil {
		return err
	}
	if f.Size > 0 && n != f.Size {
		return fmt.Errorf("%s: got %d bytes, want %d", f.Name, n, f.Size)
	}
	if got := hex.EncodeToString(h.Sum(nil)); got != f.SHA256 {
		return fmt.Errorf("%s: sha256 %s, want %s", f.Name, got, f.SHA256)
	}
	return nil
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
	tmp, err := os.CreateTemp(dir, f.Name+".*.copy")
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

// stallReader restarts a timer on every read that returns data.
type stallReader struct {
	r     io.Reader
	timer *time.Timer
}

func (s *stallReader) Read(b []byte) (int, error) {
	n, err := s.r.Read(b)
	if n > 0 {
		s.timer.Reset(fetchStallTimeout)
	}
	return n, err
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
