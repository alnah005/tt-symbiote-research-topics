# Configuration Recommendations

This document provides ready-to-use TTNN configuration recipes for three production MoE inference scenarios, explains how to set `fused_activation` in sharded matmul program configs for large hidden dimensions, and gives a verification checklist using the Tracy profiler CSV to confirm that fusion is active.

---

## Three Production Scenarios

### Scenario 1: Decode (1–8 Tokens), BF16, Standard MoE FFN

**Context:** Single-user or small-batch autoregressive decode. `batch_size × top_k ≤ 8`. Tensors fit in L1 interleaved layout. No sharding required. BF16 throughout for maximum numerical fidelity.

**Recommendation:** Use Pattern A with the top-level `activation` parameter. *Pattern A is defined in Chapter 5, `swiglu_fusion_pattern.md`.*

```python
import ttnn

# Scenario 1: decode BF16, interleaved tensors
# x: [1, 1, num_tokens, d_model] in DRAM or L1 interleaved, BF16
# w1 (gate_proj weight): [1, 1, d_model, d_ff], BF16
# w3 (up_proj weight):   [1, 1, d_model, d_ff], BF16

# Pattern A: gate_proj + SiLU in one kernel dispatch
gate = ttnn.matmul(
    x,
    w1,
    activation="silu",          # fuses SiLU into the matmul kernel
    dtype=ttnn.bfloat16,
    memory_config=ttnn.L1_MEMORY_CONFIG,
)

# up_proj: no activation, separate dispatch
up = ttnn.matmul(
    x,
    w3,
    dtype=ttnn.bfloat16,
    memory_config=ttnn.L1_MEMORY_CONFIG,
)

# Element-wise multiply: silu(gate_proj(x)) * up_proj(x)
hidden = ttnn.mul(gate, up)
```

**Expected dispatch sequence in Tracy CSV:**

| Op name (Tracy CSV `OP TYPE` column) | Count |
|---|---|
| `MatmulMultiCoreReuse*` (gate+silu) | 1 |
| `MatmulMultiCoreReuse*` (up) | 1 |
| `EltwiseBinaryOp` or `BinaryOp` (mul) | 1 |

No `silu` row should appear. If a `silu` row appears, `activation="silu"` was not applied or was overridden by a program config that omits `fused_activation`.

---

### Scenario 2: Decode (1–8 Tokens), BFP8 Inference

**Context:** Quantized inference with BFP8_B weights and activations. Goal is to reduce L1 footprint of intermediate tensors and increase throughput on memory-bound decode workloads. The fused SiLU output is written in `bfloat8_b` format, halving the L1 footprint of the gate activation tensor compared to BF16.

**Recommendation:** Use Pattern A with `fused_activation` in a program config and set `activation_dtype=ttnn.bfloat8_b`. Validate PCC against a BF16 reference before deploying.

> **Warning:** The `activation` top-level parameter on `ttnn.matmul` does not control `activation_dtype`. To set both `fused_activation` and `activation_dtype`, you must use a program config. Omitting the program config and relying solely on `activation="silu"` will produce a BF16 fused output regardless of the surrounding dtype context.

```python
import ttnn

# Scenario 2: decode BFP8 inference with program config path

num_tokens = 8      # batch_size × top_k
d_model    = 4096   # example: Qwen-MoE / Mixtral d_model
d_ff       = 2048   # per-expert FFN hidden dim

program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
    compute_with_storage_grid_size=(8, 1),
    in0_block_w=2,
    out_subblock_h=1,
    out_subblock_w=4,
    per_core_M=1,               # num_tokens / num_cores_M; ceil to nearest tile
    per_core_N=d_ff // 8 // 32, # d_ff per core / tile_width
    fuse_batch=True,
    fused_activation=(ttnn.UnaryOpType.SILU, True),
    mcast_in0=False,
)

gate = ttnn.matmul(
    x,
    w1,
    program_config=program_config,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    dtype=ttnn.bfloat8_b,       # output dtype of the fused activation
)

up = ttnn.matmul(
    x,
    w3,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    dtype=ttnn.bfloat8_b,
)

hidden = ttnn.mul(gate, up)
```

**PCC validation requirement:** After enabling BFP8 activation dtype, run the full expert FFN forward pass on a representative input and compare the output of `hidden` against a BF16 reference using Pearson Correlation Coefficient (PCC). The threshold for acceptance is PCC > 0.999 measured against BF16 output.

```python
import torch

def pcc(a, b):
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    return torch.corrcoef(torch.stack([a_flat, b_flat]))[0, 1].item()

score = pcc(hidden_bf16_ref, hidden_bfp8)
assert score > 0.999, f"PCC {score:.6f} below threshold — BFP8 activation dtype degrades accuracy"
```

> **Tip:** Run PCC validation on at least 10 representative expert inputs drawn from the target benchmark (e.g., MMLU, HellaSwag) rather than random tensors. Random inputs may pass PCC thresholds that real model inputs fail, because the SiLU negative region (`x < 0`, output approaches zero as `x → −∞`, minimum ≈ −0.278 near `x ≈ −1.28`) is underrepresented in uniform random distributions.

---

### Scenario 3: Prefill (128+ Tokens), BF16

**Context:** Prompt processing for sequences of 128 or more tokens. The gate_proj matmul is partially compute-bound; SiLU latency is less than 5% of gate_proj matmul time. Sharding across the full 8×10 Tensix grid is typical for large hidden dims.

**Recommendation:** Fusion is optional. Standalone `ttnn.silu` (Pattern B) is acceptable. If a sharded program config is already required for the matmul, add `fused_activation` at no additional engineering cost. Do not re-architect the tensor layout solely to enable fusion at prefill token counts.

**Priority order for prefill optimization:**

1. Matmul program config block size and core grid tuning.
2. Sharding strategy: height-sharded for large M, width-sharded for large N.
3. Reducing down_proj latency.
4. (Optional, low priority) Adding `fused_activation` to the existing program config.

```python
import ttnn

# Scenario 3: prefill BF16, sharded program config
# Fusion added at no extra cost since program config is already required

num_tokens = 512    # prefill token count
d_model    = 4096
d_ff       = 8192   # large hidden dim requiring sharding

program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=(8, 4),   # 32 cores
    in0_block_w=4,
    out_subblock_h=4,
    out_subblock_w=2,
    per_core_M=num_tokens // 16,
    per_core_N=d_ff // 32 // 8,
    transpose_mcast=False,
    fused_activation=(ttnn.UnaryOpType.SILU, True),  # free to add; < 5% gain
)

gate = ttnn.matmul(
    x,
    w1,
    program_config=program_config,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    dtype=ttnn.bfloat16,
)

up = ttnn.matmul(x, w3, dtype=ttnn.bfloat16)
hidden = ttnn.mul(gate, up)
```

> **Tip:** At prefill, the primary bottleneck is the down_proj matmul (it has the same shape as gate_proj and up_proj combined output). Profile with Tracy before assuming gate_proj SiLU fusion is the limiting factor.

---

## Fused Activation in Sharded Matmul Program Configs

When `d_ff` is large (8192 or above) or when `num_tokens` is large enough to require distribution across multiple cores, the gate_proj matmul output must be sharded. In this case, the top-level `activation="silu"` parameter is ignored if a program config is also provided; `fused_activation` must be set inside the program config.

**Rule:** If you pass any `program_config` to `ttnn.matmul`, set `fused_activation` inside that config — not via `activation=` at the call site. *See Chapter 5, `ttnn_fused_activation_api.md` for the precedence rule.*

The `fused_activation` field accepts a tuple `(ttnn.UnaryOpType.SILU, True)`. The boolean flag enables the operation.

```python
# Correct: fused_activation inside program config
program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    ...,
    fused_activation=(ttnn.UnaryOpType.SILU, True),
)
gate = ttnn.matmul(x, w1, program_config=program_config)

# Incorrect: activation= at top level with a program config present
# The activation= argument is silently ignored when program_config is set.
gate = ttnn.matmul(x, w1, program_config=program_config, activation="silu")  # WRONG
```

> **Warning:** TTNN does not raise an error when `activation="silu"` is specified alongside a program config that does not include `fused_activation`. The `activation=` argument is silently ignored. Always verify fusion is active via the Tracy profiler CSV (see checklist below).

---

## Verification Checklist: Tracy Profiler CSV

Use this checklist after implementing Pattern A to confirm that SiLU fusion is active at runtime.

**Step 1: Capture a Tracy profile**

```bash
# Set env var before running your inference script
TT_METAL_ENABLE_PROFILER=1 python your_moe_inference_script.py
```

The Tracy CSV is written to `generated/profiler/reports/` by default, with a filename based on the run timestamp.

**Step 2: Open the CSV and filter for the relevant op window**

The Tracy profiler CSV contains a `DEVICE KERNEL DURATION [ns]` column and an `OP TYPE` (or `OP NAME`) column. Filter rows corresponding to the SwiGLU FFN block.

**Step 3: Check for the presence or absence of a standalone `silu` row**

| Observation | Interpretation |
|---|---|
| Two `MatmulMultiCoreReuse*` rows followed by one `BinaryOp`/`mul` row (no `silu` row) | Fusion is active. Pattern A is running correctly. |
| `MatmulMultiCoreReuse*` + standalone `silu` row + `MatmulMultiCoreReuse*` + `BinaryOp` | Fusion is NOT active. Pattern B is running. Check that `fused_activation` is set in the program config, or that `activation="silu"` is set and no conflicting program config is overriding it. |
| Unexpected `ttnn_interleaved_to_sharded` or `ttnn_sharded_to_interleaved` rows between matmul and mul | Memory config mismatch between `gate` and `up` tensors. `ttnn.mul` is triggering an implicit format conversion. Align shard specs. |

**Step 4: Confirm `DEVICE KERNEL DURATION` values are in expected range**

For the fused gate_proj+SiLU kernel (Pattern A), the kernel duration should be slightly longer than the unfused gate_proj matmul alone (because the SFPU SiLU pass is appended to the FPU pass), but shorter than the sum of the unfused gate_proj matmul + standalone SiLU kernel.

| Metric | Expected range (BF16, `num_tokens=8`, `d_ff=2048`) |
|---|---|
| Fused gate_proj+SiLU (Pattern A) | [MEASURED: target ~gate_proj + 10–20% overhead] |
| Standalone gate_proj (Pattern B) | [MEASURED: baseline] |
| Standalone SiLU separately (Pattern B) | [MEASURED: 2–5 µs expected] |
| Pattern A saving vs. Pattern B | [MEASURED: should be approximately standalone SiLU latency] |

*Replace `[MEASURED: ...]` placeholders with values from your Chapter 3 benchmark run. See Chapter 3, `measurement_methodology.md` for the procedure.*

**Step 5: Confirm no `silu` op appears in the T3K multi-chip trace**

On T3K, Tracy captures per-chip device traces. Check each chip's trace independently — expert parallelism means each chip runs a subset of the experts, and each chip's FFN kernel sequence should show the fused Pattern A signature.

---

## Next Steps

Continue to [`measurement_summary_and_next_steps.md`](measurement_summary_and_next_steps.md) for a summary of expected absolute latency numbers from the Chapter 4 benchmark, interpretation guidance, open research questions, and pointers to related guides.
