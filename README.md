# eg2

Embeddings from [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) for text,
images, audio and video, alone or combined into one vector. A Go library and a CLI with an
OpenAI-compatible server, with no cgo and nothing to install: llama.cpp's shared libraries
and the model are downloaded into a cache on first use and loaded at run time through
[yzma](https://github.com/hybridgroup/yzma).

## Install

```sh
go install github.com/meinside/embedding-gemma-2-runner/cmd/eg2@latest
eg2 setup -media   # optional: download everything now (about 900 MB)
```

Video needs `ffmpeg` and `ffprobe` in `PATH`. On Linux, the system `libffi.so.8` is needed.

## Use

```sh
eg2 embed --task search-query --text "waterproof trail shoes"
eg2 embed --task document photo.jpg talk.mp3 clip.mp4 notes.md     # one vector per file
eg2 embed --text "Trail shoe <|image|> grip test <|video|>" --image shoe.jpg --video grip.mp4
eg2 embed --input inputs.jsonl > vectors.jsonl
eg2 serve                                                           # POST /v1/embeddings on 127.0.0.1:8080
```

`eg2 help inputs` describes inputs, placeholders and tasks; `eg2 COMMAND -h` lists flags and
shows where the cache is. `eg2 info` shows versions and what is installed.

From Go:

```go
e, err := eg2.Open(ctx, eg2.Options{})
res, err := e.Embed(ctx, eg2.Input{Task: eg2.TaskSearchQuery, Text: "waterproof trail shoes"})
```

## Layout

| Path | What |
|---|---|
| `*.go` (package `eg2`) | public API: inputs, tasks, placeholders, `Embed`, `Truncate` |
| `internal/engine` | llama.cpp / mtmd: tokenize, encode, pool; device fallback |
| `internal/media` | image and video preprocessing (decode, frame sampling, resize) |
| `internal/assets` | cache directories; pinned model files and llama.cpp builds |
| `internal/server` | HTTP API |
| `cmd/eg2` | CLI |
| `testdata/cases.json` | parity cases, in the JSON input format |
| `scripts/` | media for the parity cases; reference embeddings (Python) |

## Checks

```sh
gofmt -l . && go vet ./... && go test ./...
ruff check scripts && ruff format --check scripts
```

Tests that need the network or a reference file skip themselves and say how to run them;
`parity_test.go` describes the parity check against the reference implementation.
`.poc/` is ignored scratch space.

## Model license

The model is not part of this repository; `eg2` downloads it at run time. Its use is governed
by its own terms: the [Apache 2.0 license](https://ai.google.dev/gemma/docs/gemma_4_license)
and the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy), as
stated on the [model card](https://huggingface.co/google/embeddinggemma-2).

