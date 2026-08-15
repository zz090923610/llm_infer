"""Token sampling: greedy, temperature, top-k, top-p."""

from __future__ import annotations

import numpy as np


def softmax_last(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float64)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def sample_token(
    logits: np.ndarray,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    rng: np.random.Generator | None = None,
) -> int:
    """`temperature <= 0` (or top_k == 1) is greedy argmax."""
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    if temperature is None or temperature <= 0 or top_k == 1:
        return int(np.argmax(logits))

    logits = logits / temperature
    if top_k and top_k < logits.size:
        thresh = np.partition(logits, -top_k)[-top_k]
        logits = np.where(logits >= thresh, logits, -np.inf)

    probs = softmax_last(logits)
    if top_p < 1.0:
        order = np.argsort(probs)[::-1]
        cdf = np.cumsum(probs[order])
        keep = cdf <= top_p
        keep[0] = True  # always keep the top token
        mask = np.zeros_like(probs, dtype=bool)
        mask[order[keep]] = True
        # also keep the token that just crosses top_p
        if not keep.all():
            first_out = order[int(np.count_nonzero(keep))]
            mask[first_out] = True
        probs = np.where(mask, probs, 0.0)
        z = probs.sum()
        if z <= 0:
            return int(np.argmax(logits))
        probs = probs / z

    rng = rng or np.random.default_rng()
    return int(rng.choice(probs.size, p=probs))
