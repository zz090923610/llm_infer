"""Prefill / decode loop with streaming and optional padded batching."""

from __future__ import annotations

import sys
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from .cache import KVCache
from .model import LlamaModel
from .sampler import sample_token
from .tokenizer import StreamDecoder, Tokenizer, apply_chat_template


def _as_batch(token_lists: list[list[int]], pad_id: int) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.array([len(t) for t in token_lists], dtype=np.int32)
    slen = int(lengths.max()) if len(lengths) else 0
    batch = np.full((len(token_lists), slen), pad_id, dtype=np.int32)
    for i, ids in enumerate(token_lists):
        batch[i, : len(ids)] = np.asarray(ids, dtype=np.int32)
    return batch, lengths


def generate(
    model: LlamaModel,
    tokenizer: Tokenizer,
    prompt_ids: list[int],
    max_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 0,
    top_p: float = 0.9,
    cache: KVCache | None = None,
    rng: np.random.Generator | None = None,
    stop_ids: set[int] | None = None,
) -> Iterator[int]:
    """Yield generated token ids. Stops on EOS / END without yielding them."""
    if cache is None:
        cache = model.new_cache(batch=1)
    if stop_ids is None:
        stop_ids = tokenizer.stop_ids
    prompt = np.asarray(prompt_ids, dtype=np.int32)[None, :]
    logits = model.forward(prompt, cache)
    for _ in range(max_tokens):
        tid = sample_token(logits[0], temperature, top_k, top_p, rng)
        if tid in stop_ids:
            return
        yield tid
        logits = model.forward(np.array([[tid]], dtype=np.int32), cache)


def generate_text(
    model: LlamaModel,
    tokenizer: Tokenizer,
    prompt: str,
    max_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 0,
    top_p: float = 0.9,
    parse_special: bool = True,
    stream: bool = True,
    cache: KVCache | None = None,
) -> str:
    ids = tokenizer.encode(prompt, parse_special=parse_special)
    dec = StreamDecoder(tokenizer, skip_special=True)
    pieces: list[str] = []
    t0 = time.perf_counter()
    n = 0
    for tid in generate(model, tokenizer, ids, max_tokens, temperature, top_k, top_p, cache):
        n += 1
        chunk = dec.push(tid)
        pieces.append(chunk)
        if stream and chunk:
            sys.stdout.write(chunk)
            sys.stdout.flush()
    tail = dec.flush()
    pieces.append(tail)
    if stream and tail:
        sys.stdout.write(tail)
        sys.stdout.flush()
    elapsed = time.perf_counter() - t0
    if stream:
        tps = n / elapsed if elapsed > 0 else 0.0
        sys.stdout.write(f"\n[{n} tokens, {tps:.2f} tok/s]\n")
        sys.stdout.flush()
    return "".join(pieces)


def generate_batch(
    model: LlamaModel,
    tokenizer: Tokenizer,
    prompts: list[str],
    max_tokens: int = 32,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    parse_special: bool = True,
) -> list[list[int]]:
    """Padded prefill then per-row decode. Finished rows keep emitting EOS."""
    hp = model.hparams
    id_lists = [tokenizer.encode(p, parse_special=parse_special) for p in prompts]
    pad_id = hp.pad_id
    batch, lengths = _as_batch(id_lists, pad_id)
    bsz = batch.shape[0]
    cache = model.new_cache(batch=bsz)
    logits = model.forward(batch, cache, valid_len=lengths)
    out: list[list[int]] = [[] for _ in range(bsz)]
    done = np.zeros((bsz,), dtype=bool)
    rng = np.random.default_rng()
    for _ in range(max_tokens):
        nxt = np.zeros((bsz, 1), dtype=np.int32)
        for b in range(bsz):
            if done[b]:
                nxt[b, 0] = hp.eos_id
                continue
            tid = sample_token(logits[b], temperature, top_k, top_p, rng)
            if tid in tokenizer.stop_ids:
                nxt[b, 0] = hp.eos_id
                done[b] = True
                continue
            nxt[b, 0] = tid
            out[b].append(tid)
            if tid == hp.eos_id:
                done[b] = True
        if done.all():
            break
        # still-running rows get a real token; finished rows feed EOS (masked by valid_len=0)
        valid = np.where(done, 0, 1).astype(np.int32)
        # rows that are done should not grow the cache
        logits = model.forward(nxt, cache, valid_len=valid)
    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run a completion or a tiny batched demo")
    parser.add_argument("--model", default=str(Path(__file__).resolve().parents[1] / "nanogpt-chat-q8_0.gguf"))
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--batch-demo", action="store_true")
    args = parser.parse_args()

    print(f"loading {args.model}", flush=True)
    from .gguf_loader import load_model

    loaded = load_model(args.model)
    model = LlamaModel(loaded)
    tok = Tokenizer.from_gguf(loaded.gguf, loaded.hparams)
    if args.batch_demo:
        prompts = [
            apply_chat_template([{"role": "user", "content": "Say hi in one word."}]),
            apply_chat_template([{"role": "user", "content": "2+2="}]),
        ]
        print("batched greedy decode:", flush=True)
        outs = generate_batch(model, tok, prompts, max_tokens=args.max_tokens, temperature=0.0)
        for p, ids in zip(prompts, outs):
            print("---")
            print(tok.decode(ids, skip_special=True))
        return
    text = apply_chat_template([{"role": "user", "content": args.prompt}])
    generate_text(model, tok, text, args.max_tokens, args.temp, args.top_k, args.top_p)


if __name__ == "__main__":
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        __package__ = "py"
    main()
