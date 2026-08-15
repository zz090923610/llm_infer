"""GGML Q8_0 blocks: one fp16 scale and 32 int8 values (34 bytes).

Dequant is y[i] = float(d) * float(qs[i]). Blocks tile along ggml's innermost
dimension (ne[0]), which is why every weight dim we care about is a multiple of 32.
"""

from __future__ import annotations

import numpy as np

QK8_0 = 32
BLOCK_Q8_0 = 34  # 2-byte scale + 32 int8s


def dequantize_q8_0(data: bytes | memoryview | np.ndarray, n_elements: int) -> np.ndarray:
    """Dequantize packed Q8_0 bytes into a flat float32 vector of `n_elements`."""
    if n_elements % QK8_0 != 0:
        raise ValueError(f"Q8_0 length {n_elements} is not a multiple of {QK8_0}")
    n_blocks = n_elements // QK8_0
    raw = np.frombuffer(data, dtype=np.uint8, count=n_blocks * BLOCK_Q8_0)
    if raw.size != n_blocks * BLOCK_Q8_0:
        raise ValueError(
            f"need {n_blocks * BLOCK_Q8_0} bytes for {n_elements} Q8_0 values, got {raw.size}"
        )
    blocks = raw.reshape(n_blocks, BLOCK_Q8_0)
    # copy so the fp16 view is aligned regardless of mmap offset
    scales = np.ascontiguousarray(blocks[:, :2]).view(np.float16).astype(np.float32).reshape(n_blocks, 1)
    qs = blocks[:, 2:].view(np.int8).astype(np.float32)
    return (qs * scales).reshape(n_elements)


def dequantize_f32(data: bytes | memoryview | np.ndarray, n_elements: int) -> np.ndarray:
    """Read a tightly packed float32 tensor."""
    arr = np.frombuffer(data, dtype=np.float32, count=n_elements)
    if arr.size != n_elements:
        raise ValueError(f"need {n_elements} f32 values, got {arr.size}")
    return np.array(arr, dtype=np.float32, copy=True)
