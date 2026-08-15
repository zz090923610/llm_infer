# From-scratch NumPy inference for `nanogpt-chat-q8_0.gguf`

This is a **Llama / SmolLM2-360M** checkpoint (Q8_0 GGUF), not Karpathy GPT-2.
The runtime parses GGUF, dequantizes Q8_0, and runs RMSNorm, GQA, consecutive-pair
RoPE, and SwiGLU in NumPy (stdlib `unicodedata` for pretok). No `llama-cpp-python`,
no Transformers, no PyTorch.

```
pip install -r py/requirements.txt
python3 -m py.chat --model nanogpt-chat-q8_0.gguf
python3 -m py.generate --prompt "Hello" --temp 0
python3 -m py.generate --batch-demo
python3 -m unittest py.tests.test_quant py.tests.test_tokenizer py.tests.test_generate
```

| File | Role |
| --- | --- |
| `gguf_loader.py` | GGUF v3 + hparams |
| `quant.py` | Q8_0 block dequant (fp16 scale × 32 int8s) |
| `tokenizer.py` | smollm pretok + GPT-2 BPE + ChatML |
| `rope.py` | llama.cpp NORM RoPE (consecutive pairs) |
| `model.py` | transformer forward + last-token logits |
| `cache.py` | KV cache |
| `sampler.py` | greedy / temp / top-k / top-p |
| `generate.py` | stream + padded batch |
| `chat.py` | interactive ChatML |

Weights are dequantized to float32 at load (~1.4 GiB). llama.cpp under `3rd/` is
used only as a test oracle (`llama-tokenize`, `llama-cli`).
