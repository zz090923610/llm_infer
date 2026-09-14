# llm_infer operators vs PIM (nanogpt)

First model: `nanogpt-chat-q8_0.gguf` — Llama-like, `n_embd D=960`, `n_ff F=2560`, `n_vocab=49153`, 32 layers. All 2D weights are GGUF Q8_0 except small f32 norms/biases.

Source: `llm_infer/c/model.c` (`full_attn_layer`, FFN block, output proj).

## Rule

- `n_tok == 1` (decode, `S==1`) is GEMV — PIM-suitable.
- `n_tok > 1` (prefill) is GEMM — stay on CPU.
- Output projection is always `n_tok==1` (last hidden state).

## Decode GEMV (offload set)

| Op class | Call | Typical shape (n_out, n_in) | Notes |
|----------|------|-----------------------------|--------|
| attn | `linear_rows_wt(wq)` | `(H*d or H*2*d, D)` ~ `(960, 960)` | then CPU bias / RoPE / GQA |
| attn | `linear_rows_wt(wk)` | `(KV*d, D)` | |
| attn | `linear_rows_wt(wv)` | `(KV*d, D)` | |
| attn | `linear_rows_add_wt(wo)` | `(D, H*d)` ~ `(960, 960)` | residual: PIM y then CPU add, or acc=1 |
| ffn | `linear_rows_wt(gate)` | `(F, D)` = `(2560, 960)` | |
| ffn | `linear_rows_wt(up)` | `(2560, 960)` | |
| ffn | `linear_rows_add_wt(down)` | `(960, 2560)` | residual |
| output | `linear_rows_wt(output)` | `(49153, 960)` | pad n_out to 49168 (16-bank) |

Gate with `LLM_PIM_OPS=output,ffn,attn` (default `all`). Enable one class at a time when debugging tokens.

## Always CPU

RMSNorm, SiLU, softmax/GQA, RoPE, embed, tokenizer, sampler, residuals around attn/FFN except `wo`/`down` add, all `n_tok>1` linears, Qwen GDN extras (later).

## Dual layout

Host keeps f32 row-major activations. PIM interned weights live as FP16 bank bursts. Each GEMV packs `x` into GPR/GB, unpacks `y` to host f32. GB/MAC are not live across llm_infer ops.

## Qwen (later)

Same decode vs prefill split. Extra GDN `linear_wt` calls in `gdn.c` can join `attn`/`ffn` once nanogpt tokens match.
