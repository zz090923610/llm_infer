"""GPT-2 byte-level BPE with the SmolLM pre-tokenizer.

Specials (`<|im_start|>`, `<|im_end|>`, …) are matched as whole tokens before
BPE. Digits are split one-by-one (`\\p{N}`), which is the smollm/starcoder
pre-tokenizer llama.cpp uses for this GGUF.
"""

from __future__ import annotations

import heapq
import unicodedata
from dataclasses import dataclass

from .gguf_loader import GGUFFile, LlamaHParams, LoadedModel, open_gguf

_TYPE_UNKNOWN = 2
_TYPE_CONTROL = 3
_TYPE_USER_DEFINED = 4


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2 printable-byte map: every byte 0..255 ↔ a unique unicode char."""
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


_BYTE_ENCODER = _bytes_to_unicode()
_BYTE_DECODER = {v: k for k, v in _BYTE_ENCODER.items()}


def _is_letter(ch: str) -> bool:
    return unicodedata.category(ch).startswith("L")


def _is_number(ch: str) -> bool:
    return unicodedata.category(ch).startswith("N")


def _is_space(ch: str) -> bool:
    return ch.isspace()


def _gpt2_split(text: str) -> list[str]:
    """Port of llama.cpp unicode_regex_split_custom_gpt2 (codepoint walker)."""
    chars = list(text)
    n = len(chars)
    out: list[str] = []
    pos = 0
    while pos < n:
        ch = chars[pos]
        if ch == "'" and pos + 1 < n:
            nxt = chars[pos + 1]
            if nxt in "stmd":
                out.append("".join(chars[pos : pos + 2]))
                pos += 2
                continue
            if pos + 2 < n:
                two = nxt + chars[pos + 2]
                if two in ("re", "ve", "ll"):
                    out.append("".join(chars[pos : pos + 3]))
                    pos += 3
                    continue
        look_i = pos + 1 if ch == " " and pos + 1 < n else pos
        look = chars[look_i] if look_i < n else ""
        if look and _is_letter(look):
            start = pos
            pos = look_i
            while pos < n and _is_letter(chars[pos]):
                pos += 1
            out.append("".join(chars[start:pos]))
            continue
        if look and _is_number(look):
            start = pos
            pos = look_i
            while pos < n and _is_number(chars[pos]):
                pos += 1
            out.append("".join(chars[start:pos]))
            continue
        if look and not _is_space(look) and not _is_letter(look) and not _is_number(look):
            start = pos
            pos = look_i
            while pos < n and not _is_space(chars[pos]) and not _is_letter(chars[pos]) and not _is_number(chars[pos]):
                pos += 1
            out.append("".join(chars[start:pos]))
            continue
        ws = 0
        while pos + ws < n and _is_space(chars[pos + ws]):
            ws += 1
        if ws > 1 and pos + ws < n:
            out.append("".join(chars[pos : pos + ws - 1]))
            pos += ws - 1
            continue
        if ws > 0:
            out.append("".join(chars[pos : pos + ws]))
            pos += ws
            continue
        out.append(ch)
        pos += 1
    return out


def pretokenize_smollm(text: str) -> list[str]:
    """Two-pass smollm split: each digit, then GPT-2 words on the rest."""
    words: list[str] = []
    buf: list[str] = []
    for ch in text:
        if _is_number(ch):
            if buf:
                words.extend(_gpt2_split("".join(buf)))
                buf = []
            words.append(ch)
        else:
            buf.append(ch)
    if buf:
        words.extend(_gpt2_split("".join(buf)))
    return words


def apply_chat_template(messages: list[dict[str, str]], add_generation_prompt: bool = True) -> str:
    """ChatML, matching tokenizer.chat_template in the GGUF. No extra BOS."""
    parts: list[str] = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
    return "".join(parts)


@dataclass
class Tokenizer:
    vocab: list[str]
    token_to_id: dict[str, int]
    merges_rank: dict[tuple[str, str], int]
    specials: list[str]
    special_set: set[str]
    hparams: LlamaHParams

    @property
    def stop_ids(self) -> set[int]:
        """EOS plus this checkpoint's trained end marker (`END` / ` END`).

        NanoGPT-Chat often emits the word END instead of `<|im_end|>`, then
        loops on it until max_tokens. Treat those pieces as stop tokens.
        """
        ids = {self.hparams.eos_id}
        for piece in ("END", "ĠEND"):
            tid = self.token_to_id.get(piece)
            if tid is not None:
                ids.add(tid)
        return ids

    @classmethod
    def from_gguf(cls, gguf: GGUFFile, hparams: LlamaHParams | None = None) -> Tokenizer:
        kv = gguf.kv
        if kv.get("tokenizer.ggml.model") != "gpt2":
            raise ValueError(f"unsupported tokenizer {kv.get('tokenizer.ggml.model')!r}")
        vocab: list[str] = list(kv["tokenizer.ggml.tokens"])
        token_to_id = {t: i for i, t in enumerate(vocab)}
        ranks: dict[tuple[str, str], int] = {}
        for i, merge in enumerate(kv["tokenizer.ggml.merges"]):
            left, right = merge.split(" ", 1)
            ranks[(left, right)] = i
        types = kv.get("tokenizer.ggml.token_type") or []
        specials: list[tuple[int, str]] = []
        for i, tok in enumerate(vocab):
            ttype = types[i] if i < len(types) else 1
            if ttype in (_TYPE_UNKNOWN, _TYPE_CONTROL, _TYPE_USER_DEFINED) or (
                tok.startswith("<|") and tok.endswith("|>")
            ):
                specials.append((i, tok))
        specials.sort(key=lambda it: (-len(it[1]), it[0]))
        special_list = [s for _, s in specials]
        if hparams is None:
            from .gguf_loader import _hparams_from_kv

            hparams = _hparams_from_kv(kv)
        return cls(
            vocab=vocab,
            token_to_id=token_to_id,
            merges_rank=ranks,
            specials=special_list,
            special_set=set(special_list),
            hparams=hparams,
        )

    @classmethod
    def from_file(cls, path: str) -> Tokenizer:
        gguf = open_gguf(path)
        try:
            return cls.from_gguf(gguf)
        finally:
            gguf.close()

    def encode(self, text: str, parse_special: bool = True) -> list[int]:
        ids: list[int] = []
        for kind, piece in self._partition(text, parse_special):
            if kind == "special":
                ids.append(self.token_to_id[piece])
            else:
                ids.extend(self._encode_text(piece))
        return ids

    def decode(self, ids: list[int], skip_special: bool = False) -> str:
        chunks: list[str] = []
        for i in ids:
            piece = self.vocab[i]
            if skip_special and piece in self.special_set:
                continue
            chunks.append(piece)
        return self._decode_pieces(chunks)

    def decode_bytes(self, ids: list[int], skip_special: bool = False) -> bytes:
        raw = bytearray()
        for i in ids:
            piece = self.vocab[i]
            if skip_special and piece in self.special_set:
                continue
            if piece in self.special_set:
                raw.extend(piece.encode("utf-8"))
                continue
            for ch in piece:
                raw.append(_BYTE_DECODER[ch])
        return bytes(raw)

    def _partition(self, text: str, parse_special: bool) -> list[tuple[str, str]]:
        if not parse_special or not self.specials:
            return [("text", text)] if text else []
        fragments: list[tuple[str, str]] = [("text", text)]
        for spec in self.specials:
            nxt: list[tuple[str, str]] = []
            for kind, piece in fragments:
                if kind != "text" or spec not in piece:
                    nxt.append((kind, piece))
                    continue
                start = 0
                while True:
                    i = piece.find(spec, start)
                    if i < 0:
                        if start < len(piece):
                            nxt.append(("text", piece[start:]))
                        break
                    if i > start:
                        nxt.append(("text", piece[start:i]))
                    nxt.append(("special", spec))
                    start = i + len(spec)
            fragments = nxt
        return [(k, p) for k, p in fragments if p]

    def _encode_text(self, text: str) -> list[int]:
        ids: list[int] = []
        for word in pretokenize_smollm(text):
            encoded = "".join(_BYTE_ENCODER[b] for b in word.encode("utf-8"))
            ids.extend(self._bpe(encoded))
        return ids

    def _bpe(self, token: str) -> list[int]:
        if token in self.token_to_id:
            return [self.token_to_id[token]]
        word = list(token)
        if not word:
            return []
        if len(word) == 1:
            tid = self.token_to_id.get(word[0])
            return [tid] if tid is not None else []

        ranks = self.merges_rank
        heap: list[tuple[int, int, int]] = []
        for i in range(len(word) - 1):
            r = ranks.get((word[i], word[i + 1]))
            if r is not None:
                heapq.heappush(heap, (r, i, i + 1))

        next_idx = list(range(1, len(word))) + [-1]
        prev_idx = [-1] + list(range(len(word) - 1))
        alive = [True] * len(word)

        while heap:
            rank, left, right = heapq.heappop(heap)
            if not alive[left] or not alive[right]:
                continue
            if next_idx[left] != right:
                continue
            if ranks.get((word[left], word[right])) != rank:
                continue
            word[left] = word[left] + word[right]
            alive[right] = False
            nxt = next_idx[right]
            next_idx[left] = nxt
            if nxt != -1:
                prev_idx[nxt] = left
            prv = prev_idx[left]
            if prv != -1:
                r = ranks.get((word[prv], word[left]))
                if r is not None:
                    heapq.heappush(heap, (r, prv, left))
            if nxt != -1:
                r = ranks.get((word[left], word[nxt]))
                if r is not None:
                    heapq.heappush(heap, (r, left, nxt))

        ids: list[int] = []
        i = 0
        while i != -1:
            if alive[i]:
                piece = word[i]
                tid = self.token_to_id.get(piece)
                if tid is None:
                    for ch in piece:
                        ids.append(self.token_to_id[ch])
                else:
                    ids.append(tid)
            i = next_idx[i]
        return ids

    def _decode_pieces(self, pieces: list[str]) -> str:
        raw = bytearray()
        for piece in pieces:
            if piece in self.special_set:
                raw.extend(piece.encode("utf-8"))
                continue
            for ch in piece:
                raw.append(_BYTE_DECODER[ch])
        return raw.decode("utf-8", errors="replace")


class StreamDecoder:
    """Incrementally decode tokens, holding back incomplete UTF-8 sequences."""

    def __init__(self, tokenizer: Tokenizer, skip_special: bool = True):
        self.tokenizer = tokenizer
        self.skip_special = skip_special
        self._buf = bytearray()

    def push(self, token_id: int) -> str:
        piece = self.tokenizer.vocab[token_id]
        if self.skip_special and piece in self.tokenizer.special_set:
            return ""
        if piece in self.tokenizer.special_set:
            self._buf.extend(piece.encode("utf-8"))
        else:
            for ch in piece:
                self._buf.append(_BYTE_DECODER[ch])
        try:
            text = self._buf.decode("utf-8")
            self._buf.clear()
            return text
        except UnicodeDecodeError:
            for n in range(1, 4):
                try:
                    text = bytes(self._buf[:-n]).decode("utf-8")
                    self._buf[:] = self._buf[-n:]
                    return text
                except UnicodeDecodeError:
                    continue
            return ""

    def flush(self) -> str:
        text = self._buf.decode("utf-8", errors="replace")
        self._buf.clear()
        return text


def tokenizer_from_loaded(loaded: LoadedModel) -> Tokenizer:
    return Tokenizer.from_gguf(loaded.gguf, loaded.hparams)
