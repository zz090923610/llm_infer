"""Llama-GGUF RoPE: rotate consecutive pairs (ggml NORM), not HF rotate_half.

HuggingFace Llama uses split-half (NeoX) RoPE. llama.cpp permutes Q/K when
writing GGUF so the cheaper consecutive-pair kernel is equivalent. Applying
rotate_half on these weights would scramble the head dimensions.
"""

from __future__ import annotations

import numpy as np


def build_rope_cache(seq_len: int, head_dim: int, theta: float, dtype=np.float32) -> tuple[np.ndarray, np.ndarray]:
    """cos/sin tables of shape (seq_len, head_dim) for positions 0..seq_len-1."""
    half = head_dim // 2
    inv_freq = (theta ** (-np.arange(0, half, dtype=np.float64) / half)).astype(dtype)
    t = np.arange(seq_len, dtype=dtype)
    freqs = np.outer(t, inv_freq)  # (S, head_dim/2)
    cos = np.cos(freqs).astype(dtype)
    sin = np.sin(freqs).astype(dtype)
    # interleave so each pair (x[2i], x[2i+1]) shares cos/sin[i]
    cos = np.repeat(cos, 2, axis=-1)
    sin = np.repeat(sin, 2, axis=-1)
    return cos, sin


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Rotate consecutive pairs.

    x: (B, n_head, S, head_dim)
    positions: (S,) absolute positions into the cos/sin tables
    """
    # pair rotate: (x0, x1) -> (x0 c - x1 s, x0 s + x1 c)
    x0 = x[..., 0::2]
    x1 = x[..., 1::2]
    c = cos[positions][..., 0::2]
    s = sin[positions][..., 0::2]
    # x: (B, H, S, D/2). positions (S,) → (S, D/2); (B, S) → (B, S, D/2)
    if c.ndim == 2:
        c = c[None, None, :, :]
        s = s[None, None, :, :]
    else:
        c = c[:, None, :, :]
        s = s[:, None, :, :]
    rot0 = x0 * c - x1 * s
    rot1 = x0 * s + x1 * c
    out = np.empty_like(x)
    out[..., 0::2] = rot0
    out[..., 1::2] = rot1
    return out
