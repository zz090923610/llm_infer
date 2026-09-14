# pim_func

Functional PIM data-plane. gem5+Ramulator2 stays the timing path.

Layers:
- isa: extensible command registry (add/remove/deprecate fields such as row_start/row_end)
- layout: logical tensors + allocation policy (research hook)
- mem / gb / pu: banked DRAM (dense GEMV slabs + sparse fallback), Global Buffer, MAC accumulators
- hook: `on_retired_aim()` gated by `PIM_FUNC_MODE`; GPR/intern live on `global_device()`
- C API (`pim_func/c_api.h`): intern, prepare banks, issue one ISR / unpack a wave — used by `llm_infer` `-DLLM_BACKEND=pim`

The GEMV **planner** (op + shapes + hardware → `WR_GB` / `WR_BIAS` / `MAC_ABK` / `RD_MAC`) lives in `llm_infer/c/backends/pim/`, not here. Guest `pim_backend_gemv` calls `pim_execute`; the simulator Device only runs retired micros.

## Modes

`llm_infer` is the only token-generating program. `pim_replay` is not used.

| Knob | Where | Values |
|------|--------|--------|
| `PIM_FUNC_MODE` | gem5 / libramulator host env | `timing` (default): skip MAC at retire. `token`: execute `pim_func` on retire so `RD_MAC` returns real `y`. |
| `PIM_ISSUE` | guest `llm_infer` | `host` (in-process `Device::execute`, unit tests). `mmio`: planner cmds over AXI ISR mailbox. |

Gem5 PIM binaries (`cmake --preset aarch64-pim`) default to `PIM_ISSUE=mmio`. Weight intern is an untimed FUNC mailbox (`AIM_FUNC_*`); GEMV ISRs are fully timed. `LLM_INFER_BIN=test_linear` under `LLM_INFER_PIM=1` runs the 16×64 intern+GEMV smoke (`LLM_TEST_LINEAR_SMOKE=1`) against `llm_infer/c/tests/golden/linear_16x64.txt`.

Hardware for MMIO: 8 channels, functional `pim_chmask=0x0F` (HW ch 0–3). Guest ISR mask is `AIM_PIM_CHMASK=0xF0` (Ramulator bit reversal).

Host GEMV is **command-stepped** (`WR_GB` / `WR_BIAS` / `MAC_ABK` / `RD_MAC`). Fast path accelerates opcode handlers (dense slabs packed `[row][col][bank]`). Set `PIM_GEMV_SCALAR_STEPS=1` to force per-lane MicroStep expand.

Correctness: compare against `llm_infer` goldens in `llm_infer/c/tests/golden/`. If those look wrong, fix `llm_infer` first.
