# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "sentence-transformers[image,audio,video]>=6.1.0",
#     "transformers>=5.20.2",
#     "torch",
#     "torchcodec>=0.16.0",
#     "pillow",
# ]
# ///
"""Reference embeddings from the official sentence-transformers implementation.

Embeds the cases in testdata/cases.json (the eg2 JSON input schema) and writes
{id: vector} JSON, which the parity test in parity_test.go compares against.

    uv run scripts/reference/ref_embed.py --out .poc/ref.json
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

MODEL_ID = "google/embeddinggemma-2"
# Pinned for the same reason as the GGUF revision in internal/assets/models.go.
MODEL_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"

# What a missing decoder backend, an unreadable file or an unknown suffix raises.
# One bad input is reported and skipped so it does not hide the others.
INPUT_ERRORS = (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError)

MEDIA_KEYS = ("image", "audio", "video")

# eg2 task names to the prompt names in the model's config_sentence_transformers.json.
# Using the model's own prompts here checks the copy of them in input.go.
PROMPT_NAMES = {
    "search-query": "SearchQuery",
    "document": "Document",
    "question-answering": "QuestionAnswering",
    "fact-checking": "FactChecking",
    "code-retrieval": "CodeRetrieval",
    "classification": "Classification",
    "clustering": "Clustering",
    "similarity": "SentenceSimilarity",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="testdata/cases.json")
    ap.add_argument("--media", default=".poc/media")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--only", default="", help="comma-separated name prefixes (default: all)"
    )
    args = ap.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoConfig, AutoProcessor

    cases = json.loads(Path(args.cases).read_text())
    media = Path(args.media)

    t0 = time.monotonic()
    # CPU and float32 only, so the reference is the same wherever it is made.
    # (The model card forbids float16, which yields NaN or degraded vectors.)
    model = SentenceTransformer(
        MODEL_ID,
        revision=MODEL_REVISION,
        device="cpu",
        model_kwargs={"torch_dtype": torch.float32},
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    config = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    print(f"loaded in {time.monotonic() - t0:.1f}s")

    media_token = {
        "image": config.image_token_id,
        "audio": config.audio_token_id,
        "video": config.video_token_id,
    }

    vecs: dict[str, list[float]] = {}
    failed = 0
    prefixes = tuple(p for p in args.only.split(",") if p)
    for item in cases["queries"] + cases["documents"]:
        name = item["id"]
        if prefixes and not name.startswith(prefixes):
            continue
        t0 = time.monotonic()
        try:
            inp, kwargs = to_st_input(item, media)
            kinds = [k for k in item if k in MEDIA_KEYS]
            tokens = ""
            if (
                len(kinds) == 1
                and "text" not in item
                and isinstance(inp[kinds[0]], str)
            ):
                kind = kinds[0]
                tokens = count_tokens(
                    processor, kind, Path(inp[kind]), media_token[kind]
                )
            vec = model.encode(
                inp, normalize_embeddings=True, convert_to_numpy=True, **kwargs
            )
        except INPUT_ERRORS as e:
            failed += 1
            print(f"  {name:22} FAILED: {type(e).__name__}: {e}")
            continue
        v = [float(x) for x in vec.reshape(-1)]
        if not all(math.isfinite(x) for x in v):
            failed += 1
            print(f"  {name:22} FAILED: non-finite values")
            continue
        vecs[name] = v
        print(f"  {name:22} {time.monotonic() - t0:7.2f}s  dim={len(v)}  {tokens}")

    Path(args.out).write_text(json.dumps(vecs))
    print(
        f"wrote {len(vecs)} embeddings to {args.out}"
        + (f", {failed} failed" if failed else "")
    )
    return 1 if failed else 0


def to_st_input(item: dict, media: Path):
    """The sentence-transformers input and encode() keywords for one case.

    Text-only cases become a string; others a dict in the case's key order,
    which sentence-transformers keeps when there are no placeholders.
    """
    if item.get("literal"):
        raise ValueError("literal has no sentence-transformers equivalent")
    kwargs = {}
    if "prompt" in item:
        kwargs["prompt"] = item["prompt"]
    elif "title" in item:
        kwargs["prompt"] = f"title: {item['title']} | text: "
    elif "task" in item:
        kwargs["prompt_name"] = PROMPT_NAMES[item["task"]]

    if not any(k in item for k in MEDIA_KEYS):
        return item["text"], kwargs
    inp = {}
    for key, value in item.items():
        if key == "text":
            inp["text"] = value
        elif key in MEDIA_KEYS:
            if isinstance(value, list):
                inp[key] = [str(media / v) for v in value]
            else:
                inp[key] = str(media / value)
    return inp, kwargs


def count_tokens(processor, kind: str, path: Path, token_id: int) -> str:
    """Sequence length and media soft-token count, to diagnose a parity gap."""
    try:
        if kind == "image":
            from PIL import Image

            batch = processor(
                images=[Image.open(path).convert("RGB")], return_tensors="pt"
            )
        elif kind == "audio":
            batch = processor(audio=[str(path)], return_tensors="pt")
        else:
            batch = processor(videos=[str(path)], return_tensors="pt")
        ids = batch["input_ids"][0]
        return f"seq={len(ids)} {kind}_tokens={int((ids == token_id).sum())}"
    except INPUT_ERRORS as e:
        return f"tokens=? ({type(e).__name__}: {e})"


if __name__ == "__main__":
    sys.exit(main())
