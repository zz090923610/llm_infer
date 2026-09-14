"""Interactive ChatML loop."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .cache import KVCache
from .generate import generate
from .gguf_loader import load_model
from .model import LlamaModel
from .tokenizer import StreamDecoder, Tokenizer, apply_chat_template


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with SmolLM2-360M-Instruct (from-scratch NumPy)")
    parser.add_argument(
        "--model",
        default=str(Path(__file__).resolve().parents[1] / "models" / "smollm2-360m-instruct-q8_0.gguf"),
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--ctx", type=int, default=2048)
    args = parser.parse_args()

    print(f"loading {args.model} (Q8_0 → f32, ~1.4 GiB)", flush=True)
    loaded = load_model(args.model)
    model = LlamaModel(loaded)
    tok = Tokenizer.from_gguf(loaded.gguf, loaded.hparams)
    cache = model.new_cache(batch=1, max_seq=args.ctx)
    history: list[dict[str, str]] = []
    cached_ids: list[int] = []

    print("Chat ready. Empty line or /exit to quit, /reset to clear history.\n")
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user or user in {"/exit", "/quit"}:
            break
        if user == "/reset":
            history.clear()
            cached_ids = []
            cache.reset()
            print("(context cleared)")
            continue

        history.append({"role": "user", "content": user})
        full = apply_chat_template(history, add_generation_prompt=True, tokenizer=tok)
        ids = tok.encode(full, parse_special=True)
        # Reuse KV cache when the new prompt is a prefix continuation of what we fed.
        if cached_ids and ids[: len(cached_ids)] == cached_ids:
            prompt_ids = ids[len(cached_ids) :]
        else:
            cache.reset()
            prompt_ids = ids
            cached_ids = []
        if not prompt_ids:
            continue

        sys.stdout.write("Assistant: ")
        sys.stdout.flush()
        dec = StreamDecoder(tok, skip_special=True)
        gen_ids: list[int] = []
        t0 = time.perf_counter()
        for tid in generate(
            model,
            tok,
            prompt_ids,
            max_tokens=args.max_tokens,
            temperature=args.temp,
            top_k=args.top_k,
            top_p=args.top_p,
            cache=cache,
        ):
            gen_ids.append(tid)
            chunk = dec.push(tid)
            if chunk:
                sys.stdout.write(chunk)
                sys.stdout.flush()
        tail = dec.flush()
        if tail:
            sys.stdout.write(tail)
        elapsed = time.perf_counter() - t0
        n = len(gen_ids)
        tps = n / elapsed if elapsed > 0 else 0.0
        sys.stdout.write(f"\n[{n} tokens, {tps:.2f} tok/s]\n")
        sys.stdout.flush()
        reply = tok.decode(gen_ids, skip_special=True).strip()
        for marker in (" END", "END"):
            if reply.endswith(marker):
                reply = reply[: -len(marker)].rstrip()
        history.append({"role": "assistant", "content": reply})
        cached_ids = ids + gen_ids


if __name__ == "__main__":
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        __package__ = "py"
    main()
