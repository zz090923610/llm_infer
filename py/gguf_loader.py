"""Minimal GGUF v3 reader.

GGUF stores metadata as typed key/value pairs, then a tensor index, then a
32-byte-aligned blob. Tensor dimensions are ggml order: dims[0] == ne[0] is
contiguous, so a weight listed as [n_in, n_out] becomes NumPy (n_out, n_in)
after dequant — the same layout as `torch.nn.Linear.weight`.
"""

from __future__ import annotations

import mmap
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .quant import dequantize_f32, dequantize_q8_0

GGUF_MAGIC = b"GGUF"
GGUF_DEFAULT_ALIGNMENT = 32

# ggml_type
GGML_F32 = 0
GGML_Q8_0 = 8

# gguf_type
_U8, _I8, _U16, _I16, _U32, _I32, _F32, _BOOL, _STR, _ARR, _U64, _I64, _F64 = range(13)


@dataclass
class TensorInfo:
    name: str
    dims: tuple[int, ...]  # ggml order, ne[0] first
    ggml_type: int
    offset: int  # relative to the tensor data blob


@dataclass
class GGUFFile:
    path: Path
    version: int
    kv: dict[str, Any]
    tensors: dict[str, TensorInfo]
    data_offset: int
    mmap: mmap.mmap
    _fd: int

    def close(self) -> None:
        self.mmap.close()
        os.close(self._fd)


@dataclass
class LlamaHParams:
    n_layer: int
    n_embd: int
    n_ff: int
    n_head: int
    n_head_kv: int
    n_rot: int
    head_dim: int
    n_vocab: int
    n_ctx: int
    rms_eps: float
    rope_theta: float
    bos_id: int
    eos_id: int
    unk_id: int
    pad_id: int
    add_space_prefix: bool
    chat_template: str
    tokenizer_pre: str


@dataclass
class LoadedModel:
    gguf: GGUFFile
    hparams: LlamaHParams
    weights: dict[str, np.ndarray] = field(default_factory=dict)


def _read_str(buf: memoryview, pos: int) -> tuple[str, int]:
    (n,) = struct.unpack_from("<Q", buf, pos)
    pos += 8
    s = bytes(buf[pos : pos + n]).decode("utf-8")
    return s, pos + n


def _read_value(buf: memoryview, pos: int, vtype: int) -> tuple[Any, int]:
    if vtype == _U8:
        return buf[pos], pos + 1
    if vtype == _I8:
        return struct.unpack_from("<b", buf, pos)[0], pos + 1
    if vtype == _U16:
        return struct.unpack_from("<H", buf, pos)[0], pos + 2
    if vtype == _I16:
        return struct.unpack_from("<h", buf, pos)[0], pos + 2
    if vtype == _U32:
        return struct.unpack_from("<I", buf, pos)[0], pos + 4
    if vtype == _I32:
        return struct.unpack_from("<i", buf, pos)[0], pos + 4
    if vtype == _F32:
        return struct.unpack_from("<f", buf, pos)[0], pos + 4
    if vtype == _BOOL:
        return bool(buf[pos]), pos + 1
    if vtype == _STR:
        return _read_str(buf, pos)
    if vtype == _U64:
        return struct.unpack_from("<Q", buf, pos)[0], pos + 8
    if vtype == _I64:
        return struct.unpack_from("<q", buf, pos)[0], pos + 8
    if vtype == _F64:
        return struct.unpack_from("<d", buf, pos)[0], pos + 8
    if vtype == _ARR:
        (atype,) = struct.unpack_from("<I", buf, pos)
        pos += 4
        (n,) = struct.unpack_from("<Q", buf, pos)
        pos += 8
        out = []
        for _ in range(n):
            v, pos = _read_value(buf, pos, atype)
            out.append(v)
        return out, pos
    raise ValueError(f"unknown GGUF type {vtype}")


def open_gguf(path: str | Path) -> GGUFFile:
    path = Path(path)
    fd = os.open(path, os.O_RDONLY)
    mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
    buf = memoryview(mm)

    if bytes(buf[:4]) != GGUF_MAGIC:
        raise ValueError(f"{path} is not a GGUF file")
    (version,) = struct.unpack_from("<I", buf, 4)
    if version != 3:
        raise ValueError(f"only GGUF v3 is supported, got {version}")
    n_tensors, n_kv = struct.unpack_from("<QQ", buf, 8)
    pos = 24

    kv: dict[str, Any] = {}
    for _ in range(n_kv):
        key, pos = _read_str(buf, pos)
        (vtype,) = struct.unpack_from("<I", buf, pos)
        pos += 4
        val, pos = _read_value(buf, pos, vtype)
        kv[key] = val

    tensors: dict[str, TensorInfo] = {}
    for _ in range(n_tensors):
        name, pos = _read_str(buf, pos)
        (n_dims,) = struct.unpack_from("<I", buf, pos)
        pos += 4
        dims = []
        for _d in range(n_dims):
            (ne,) = struct.unpack_from("<Q", buf, pos)
            pos += 8
            dims.append(int(ne))
        ggml_type, offset = struct.unpack_from("<IQ", buf, pos)
        pos += 12
        tensors[name] = TensorInfo(name, tuple(dims), int(ggml_type), int(offset))

    alignment = int(kv.get("general.alignment", GGUF_DEFAULT_ALIGNMENT))
    data_offset = (pos + alignment - 1) // alignment * alignment
    return GGUFFile(path, version, kv, tensors, data_offset, mm, fd)


def _hparams_from_kv(kv: dict[str, Any]) -> LlamaHParams:
    arch = kv.get("general.architecture")
    if arch != "llama":
        raise ValueError(f"expected llama architecture, got {arch!r}")
    n_embd = int(kv["llama.embedding_length"])
    head_dim = int(kv.get("llama.attention.key_length", n_embd // int(kv["llama.attention.head_count"])))
    return LlamaHParams(
        n_layer=int(kv["llama.block_count"]),
        n_embd=n_embd,
        n_ff=int(kv["llama.feed_forward_length"]),
        n_head=int(kv["llama.attention.head_count"]),
        n_head_kv=int(kv["llama.attention.head_count_kv"]),
        n_rot=int(kv.get("llama.rope.dimension_count", head_dim)),
        head_dim=head_dim,
        n_vocab=int(kv["llama.vocab_size"]),
        n_ctx=int(kv["llama.context_length"]),
        rms_eps=float(kv["llama.attention.layer_norm_rms_epsilon"]),
        rope_theta=float(kv["llama.rope.freq_base"]),
        bos_id=int(kv.get("tokenizer.ggml.bos_token_id", 1)),
        eos_id=int(kv.get("tokenizer.ggml.eos_token_id", 2)),
        unk_id=int(kv.get("tokenizer.ggml.unknown_token_id", 0)),
        pad_id=int(kv.get("tokenizer.ggml.padding_token_id", 2)),
        add_space_prefix=bool(kv.get("tokenizer.ggml.add_space_prefix", False)),
        chat_template=str(kv.get("tokenizer.chat_template", "")),
        tokenizer_pre=str(kv.get("tokenizer.ggml.pre", "")),
    )


def _n_elements(dims: tuple[int, ...]) -> int:
    n = 1
    for d in dims:
        n *= d
    return n


def _dequant_tensor(gguf: GGUFFile, info: TensorInfo) -> np.ndarray:
    start = gguf.data_offset + info.offset
    n = _n_elements(info.dims)
    blob = gguf.mmap
    if info.ggml_type == GGML_F32:
        flat = dequantize_f32(blob[start:], n)
    elif info.ggml_type == GGML_Q8_0:
        flat = dequantize_q8_0(blob[start:], n)
    else:
        raise ValueError(f"unsupported ggml type {info.ggml_type} for {info.name}")
    # ggml ne[0] is innermost → NumPy shape is reversed
    return np.ascontiguousarray(flat.reshape(tuple(reversed(info.dims))))


def load_model(path: str | Path, dequant: bool = True, progress: bool = True) -> LoadedModel:
    """Load GGUF metadata and (by default) dequantize every tensor to float32."""
    gguf = open_gguf(path)
    hparams = _hparams_from_kv(gguf.kv)
    loaded = LoadedModel(gguf=gguf, hparams=hparams)
    if not dequant:
        return loaded
    names = list(gguf.tensors)
    for i, name in enumerate(names):
        loaded.weights[name] = _dequant_tensor(gguf, gguf.tensors[name])
        if progress and (i + 1 == len(names) or (i + 1) % 40 == 0):
            print(f"  dequant {i + 1}/{len(names)} tensors", flush=True)
    return loaded
