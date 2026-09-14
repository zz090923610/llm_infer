"""Unit tests. Run: python3 -m unittest py.tests.test_quant py.tests.test_tokenizer py.tests.test_generate"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODEL = REPO / "models" / "smollm2-360m-instruct-q8_0.gguf"
LLAMA_BIN = REPO / "3rd" / "llama.cpp" / "build" / "bin"
