# llm_infer

From-scratch **Llama / SmolLM2-360M Instruct** inference in C (`c/`), with a NumPy twin in `py/`. Default checkpoint: `models/smollm2-360m-instruct-q8_0.gguf` ([HuggingFaceTB](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF)).

Walkthrough of the model and code: [LEARN.md](LEARN.md).

## Build (host, no gem5)

One configure from the repo root fills the top-level `build/` dir with every backend this machine can compile:

```bash
cmake -B build
cmake --build build -j
```

Default `LLM_BACKEND=all` suffixes binaries (`generate-plain-cpu`, `chat-x86_64-simd`, `test_linear-pim`, …).

| `-DLLM_BACKEND=` | Kernels | Binary names |
| --- | --- | --- |
| `all` (default) | all available | `chat-<be>`, `generate-<be>`, `test_*-<be>` |
| `plain-cpu` | scalar C (`c/backends/host/pc/plain-cpu`) | unsuffixed `chat`, `generate`, … |
| `x86_64-simd` | AVX2/FMA + pthread pool (`c/backends/host/pc/x86_64-simd`) | unsuffixed |
| `pim` | decode GEMV via `pim_func` (attn/RoPE/prefill still x86 kernels on this host; `c/backends/pim`) | unsuffixed |
| `gpu` | Vulkan (`c/backends/host/android/gpu`) | unsuffixed |
| `aarch64-simd` | NEON (`c/backends/host/android/aarch64-simd`; this host or Android NDK) | unsuffixed |

Host builds bake `LLM_DEFAULT_MODEL` to `models/smollm2-360m-instruct-q8_0.gguf`. Cross/Android must pass `--model`.

## Run without gem5

```bash
# scalar CPU
./build/generate-plain-cpu --prompt "Hello" --temp 0 --max-tokens 16
./build/chat-plain-cpu --temp 0 --max-tokens 64

# AVX2
./build/generate-x86_64-simd --prompt "Hello" --temp 0 --max-tokens 16

# PIM Device in-process (default PIM_ISSUE=host)
./build/generate-pim --prompt "Hello" --temp 0 --max-tokens 16
./build/chat-pim --temp 0 --max-tokens 64
```

`chat`: empty line or `/exit` quits; `/reset` clears history.

Single-backend configure (`-DLLM_BACKEND=plain-cpu` or `pim`) uses the unsuffixed names from LEARN.md. Reconfigure the same `build/` dir, or use a second tree:

```bash
cmake -B build -DLLM_BACKEND=plain-cpu && cmake --build build -j
./build/generate --prompt "Hello" --temp 0 --max-tokens 16
```

### PIM issue mode (native vs gem5)

| `PIM_ISSUE` | Meaning |
| --- | --- |
| `host` (default in `./build` and `x86_64-pim`) | In-process `pim_func` Device. No gem5, no MMIO. |
| `mmio` | Guest AXI mailbox (gem5 `aarch64-pim`; `run_llm_infer.py` sets this). |
| `shm` / `hybrid` | Shared arena + libramulator MAC pool (`LLM_INFER_HYBRID=1`). |

Do not export `PIM_ISSUE=mmio` or `shm` for a normal desktop run. Override if needed:

```bash
PIM_ISSUE=host ./build/generate-pim --prompt Hello --temp 0 --max-tokens 2
```

`LLM_PIM_OPS=all` (default) offloads decode GEMVs in attn/FFN/output. Use `output`, `ffn`, or `attn` to narrow the set. Prefill (`n_tok > 1`) stays on CPU.

## Tests

```bash
ctest --test-dir build --output-on-failure
# or
./build/test_linear-plain-cpu
./build/test_linear-pim
./build/test_quant-plain-cpu
./build/test_tokenizer-plain-cpu
./build/test_prompt-plain-cpu
./build/test_planner-pim
```

## Python twin

See [py/README.md](py/README.md).

## gem5 / hybrid

Guest and hybrid launchers live under `develop/`, not this folder:

- CPU or PIM guest: `develop/working/run_llm_infer.py` (`LLM_INFER_BIN=generate|chat|test_linear`)
- Hybrid (native process A + gem5 stub ISRs): `LLM_INFER_HYBRID=1` — see `develop/docs/PIM_FUNC_MODE.md` and `pim_func/docs/llm_infer_ops.md`

## Recent backend notes

- Hybrid code (`pim_func` arena, `PIM_ISSUE=shm`) is linked into the PIM backend but is idle unless `PIM_ISSUE` is `shm`/`hybrid`. CPU and host-PIM paths do not attach shm or redirect `xmalloc`.
- Host `x86_64-pim` (under `develop/workloads/llm_infer`) defaults to `PIM_ISSUE=host` so `generate.elf` / `test_linear.elf` run on the desktop. gem5 still injects `mmio` or `shm`.
- `llm_set_arena_alloc` is only for a future bump allocator; interned PIM banks use `pim_hybrid_alloc` when the hybrid arena is already mapped.
