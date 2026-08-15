"""Llama / SmolLM2 forward pass: RMSNorm, GQA, consecutive-pair RoPE, SwiGLU.

GQA: 15 query heads share 5 KV heads (repeat ×3). Tied embeddings: there is no
`output.weight`, so the lm_head is `token_embd.weight`. Logits are computed
only for the last valid token of each sequence (same as llama.cpp `inp_out_ids`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cache import KVCache
from .gguf_loader import LlamaHParams, LoadedModel, load_model
from .rope import apply_rope, build_rope_cache


def rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    ms = np.mean(np.square(x), axis=-1, keepdims=True)
    return x * (1.0 / np.sqrt(ms + eps)) * weight


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.where(np.isfinite(x), x, np.float32(-1e9))
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


@dataclass
class LayerWeights:
    attn_norm: np.ndarray
    wq: np.ndarray
    wk: np.ndarray
    wv: np.ndarray
    wo: np.ndarray
    ffn_norm: np.ndarray
    gate: np.ndarray
    up: np.ndarray
    down: np.ndarray


class LlamaModel:
    def __init__(self, loaded: LoadedModel):
        self.hparams = loaded.hparams
        w = loaded.weights
        hp = loaded.hparams
        self.tok_embd = w["token_embd.weight"]  # (vocab, n_embd)
        self.output_norm = w["output_norm.weight"]
        self.layers = []
        for i in range(hp.n_layer):
            p = f"blk.{i}."
            self.layers.append(
                LayerWeights(
                    attn_norm=w[p + "attn_norm.weight"],
                    wq=w[p + "attn_q.weight"],
                    wk=w[p + "attn_k.weight"],
                    wv=w[p + "attn_v.weight"],
                    wo=w[p + "attn_output.weight"],
                    ffn_norm=w[p + "ffn_norm.weight"],
                    gate=w[p + "ffn_gate.weight"],
                    up=w[p + "ffn_up.weight"],
                    down=w[p + "ffn_down.weight"],
                )
            )
        self._rope_len = 0
        self._cos = None
        self._sin = None

    @classmethod
    def from_file(cls, path: str, progress: bool = True) -> LlamaModel:
        loaded = load_model(path, dequant=True, progress=progress)
        return cls(loaded)

    def _rope_tables(self, need: int) -> tuple[np.ndarray, np.ndarray]:
        hp = self.hparams
        if self._cos is None or need > self._rope_len:
            n = max(need, hp.n_ctx)
            self._cos, self._sin = build_rope_cache(n, hp.head_dim, hp.rope_theta)
            self._rope_len = n
        return self._cos, self._sin

    def new_cache(self, batch: int = 1, max_seq: int | None = None) -> KVCache:
        return KVCache.create(self.hparams, batch, max_seq or self.hparams.n_ctx)

    def forward(
        self,
        tokens: np.ndarray,
        cache: KVCache,
        valid_len: np.ndarray | None = None,
    ) -> np.ndarray:
        """Run a prefill or decode step.

        tokens: (B, S) int32. Right-padded when `valid_len` is set.
        Returns logits (B, vocab) for the last valid token of each row.
        """
        hp = self.hparams
        tokens = np.asarray(tokens, dtype=np.int32)
        if tokens.ndim == 1:
            tokens = tokens[None, :]
        bsz, slen = tokens.shape
        if valid_len is None:
            valid_len = np.full((bsz,), slen, dtype=np.int32)
        else:
            valid_len = np.asarray(valid_len, dtype=np.int32)

        starts = cache.n_seq.copy()
        max_end = int(np.max(starts + valid_len))
        if max_end > cache.max_seq:
            raise ValueError(f"cache overflow: {max_end} > {cache.max_seq}")

        cos, sin = self._rope_tables(max_end)
        x = self.tok_embd[tokens]  # (B, S, D)

        n_head, n_kv, d = hp.n_head, hp.n_head_kv, hp.head_dim
        n_rep = n_head // n_kv
        scale = 1.0 / np.sqrt(d)

        # positions: (B, S) absolute RoPE / cache index; pads keep dummy 0
        positions = np.zeros((bsz, slen), dtype=np.int32)
        key_len = np.zeros((bsz,), dtype=np.int32)
        for b in range(bsz):
            n = int(valid_len[b])
            positions[b, :n] = np.arange(int(starts[b]), int(starts[b]) + n)
            key_len[b] = int(starts[b]) + n

        q_len_mask = np.arange(slen)[None, :] < valid_len[:, None]  # (B, S)

        for li, layer in enumerate(self.layers):
            h = rmsnorm(x, layer.attn_norm, hp.rms_eps)
            q = h @ layer.wq.T
            k = h @ layer.wk.T
            v = h @ layer.wv.T
            q = q.reshape(bsz, slen, n_head, d).transpose(0, 2, 1, 3)
            k = k.reshape(bsz, slen, n_kv, d).transpose(0, 2, 1, 3)
            v = v.reshape(bsz, slen, n_kv, d).transpose(0, 2, 1, 3)
            q = apply_rope(q, cos, sin, positions)
            k = apply_rope(k, cos, sin, positions)

            for b in range(bsz):
                n = int(valid_len[b])
                s0 = int(starts[b])
                cache.k[li, b, :, s0 : s0 + n, :] = k[b, :, :n, :]
                cache.v[li, b, :, s0 : s0 + n, :] = v[b, :, :n, :]

            max_k = int(np.max(key_len))
            k_all = cache.k[li, :, :, :max_k, :]  # (B, n_kv, K, d)
            v_all = cache.v[li, :, :, :max_k, :]
            if n_rep > 1:
                k_all = np.repeat(k_all, n_rep, axis=1)
                v_all = np.repeat(v_all, n_rep, axis=1)

            attn = (q @ k_all.transpose(0, 1, 3, 2)) * scale  # (B, H, S, K)
            # causal: query pos >= key pos, and drop pads on both sides
            qpos = positions[:, None, :, None]  # (B, 1, S, 1)
            kpos = np.arange(max_k)[None, None, None, :]  # (1,1,1,K)
            causal = kpos <= qpos
            k_valid = kpos < key_len[:, None, None, None]
            q_valid = q_len_mask[:, None, :, None]
            attn = np.where(causal & k_valid & q_valid, attn, np.float32(-np.inf))
            probs = softmax(attn, axis=-1)
            y = probs @ v_all  # (B, H, S, d)
            y = y.transpose(0, 2, 1, 3).reshape(bsz, slen, hp.n_embd)
            x = x + y @ layer.wo.T

            h = rmsnorm(x, layer.ffn_norm, hp.rms_eps)
            x = x + (silu(h @ layer.gate.T) * (h @ layer.up.T)) @ layer.down.T

        x = rmsnorm(x, self.output_norm, hp.rms_eps)
        last = np.clip(valid_len - 1, 0, slen - 1)
        h_last = x[np.arange(bsz), last]
        logits = h_last @ self.tok_embd.T
        cache.n_seq = key_len.astype(np.int32)
        return logits
