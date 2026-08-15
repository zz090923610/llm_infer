"""Per-layer K/V cache. Shape is (batch, n_kv_head, max_seq, head_dim)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gguf_loader import LlamaHParams


@dataclass
class KVCache:
    k: np.ndarray  # (n_layer, batch, n_kv_head, max_seq, head_dim)
    v: np.ndarray
    n_seq: np.ndarray  # (batch,) tokens currently stored
    max_seq: int

    @classmethod
    def create(cls, hparams: LlamaHParams, batch: int, max_seq: int, dtype=np.float32) -> KVCache:
        shape = (hparams.n_layer, batch, hparams.n_head_kv, max_seq, hparams.head_dim)
        return cls(
            k=np.zeros(shape, dtype=dtype),
            v=np.zeros(shape, dtype=dtype),
            n_seq=np.zeros((batch,), dtype=np.int32),
            max_seq=max_seq,
        )

    @property
    def batch(self) -> int:
        return int(self.k.shape[1])

    def reset(self) -> None:
        self.n_seq[:] = 0

    def clone_empty(self) -> KVCache:
        return KVCache(
            k=np.zeros_like(self.k),
            v=np.zeros_like(self.v),
            n_seq=np.zeros_like(self.n_seq),
            max_seq=self.max_seq,
        )
