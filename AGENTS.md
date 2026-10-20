# Agent notes

Only what the code and README do not tell you.

## The reference defines correct

- Correct output is what transformers/sentence-transformers produce for
  `google/embeddinggemma-2`. Judge a change by `TestParity`, not by whether retrieval still
  ranks well: rankings on `cases.json` stayed right even when video was at 0.94 cosine.
- Preprocessing, prompts and the joining of text and media are written to match it, which is
  why images and video are prepared in Go instead of by mtmd (`internal/media`).
- Audio is the known exception: llama.cpp computes the log-mel as `log(max(mel, floor))`
  where the reference computes `log(mel + floor)`. Go cannot change that; it has not been
  reported upstream. Its effect is in `minCosine` in `parity_test.go`.
- Not yet run on Linux or Windows, nor with the CUDA or Vulkan builds.

## Agent sandbox on this machine

- Metal cannot initialize inside the sandbox, so `-device auto` lands on the CPU there;
  that says nothing about the GPU path. Ask the user to run GPU checks.
- Hugging Face CDN downloads fail (403); GitHub downloads work. The reference script and a
  fresh model download need the user. `EG2_SEED_DIR` / `Options.SeedDir` can copy the GGUFs
  from a snapshot under `~/.cache/huggingface/hub/models--ggml-org--embeddinggemma-2-GGUF/`.
- `$XDG_CACHE_HOME` is `~/.cache` here: the cache is `~/.cache/eg2/`. Under
  `~/Library/Caches` only `github.com/jupiterrider/ffi/` is writable, and in some sessions
  not even that: then every test binary panics loading libffi. Run with
  `HOME=/tmp/agents/home XDG_CACHE_HOME=~/.cache` so the ffi package unpacks its copy under
  the temporary home while the eg2 cache stays put. With `HOME` changed, asdf no longer
  finds `go`; call it by its full path (`asdf which go`).
- Processes started in the background cannot be signalled (`kill`, `pkill` are refused), so
  a server started with `&` stays up. Test servers in-process (`httptest`, or `run` with a
  cancelled context) instead.
- Moving files between those locations with `mv` copies instead of renaming and has lost
  files once. Copy, verify, then delete.
