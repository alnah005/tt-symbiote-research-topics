# Math Fidelity Evaluation

## Context

This file addresses **Q4**: Is `HiFi2` sufficient for expert matmuls, or would `LoFi` improve throughput without meaningful accuracy loss?

Source ranges: `moe.py:L1152–L1157` (`_expert_compute_cfg` construction in `move_weights_to_device_impl`), `moe.py:L1449–L1454` (gate linear compute config in `TTNNMoE.forward`).

---

## The Three Math Fidelity Levels

`ttnn.MathFidelity` maps to Wormhole's Tensix FPU mantissa accumulation mode. Lower fidelity reduces the number of partial products computed per bf16 multiply-accumulate, trading numerical precision for FPU throughput.

| Level | Mantissa bits used | Relative throughput | Current use |
|---|---|---|---|
| `HiFi4` | Full bf16 mantissa (7 bits effective) | 1.0× (baseline) | Gate linear (`moe.py:L1449–L1454`) |
| `HiFi2` | Reduced mantissa (~4 bits effective) | ~1.3–1.5× | Expert matmuls (`moe.py:L1152–L1157`) |
| `LoFi` | Minimal mantissa (~2 bits effective) | ~1.5–2.0× | Not currently used in MoE path |

The throughput multipliers are approximate and regime-dependent. At batch=1 decode, the sparse matmul is largely memory-bound: the FPU is not the bottleneck, so the raw FPU throughput ratio does not translate directly to end-to-end speedup. The fidelity benefit is most visible in compute-bound conditions (large batch, large M).

### Why `HiFi4` is held fixed for the gate linear

The gate linear (`moe.py:L1449–L1454`) computes `router_logits = x_f32 @ gate_weight`, whose output drives the topk routing decision. A routing error — two tokens being assigned to different experts due to numerical noise — causes incorrect expert computation for the entire sequence. This is a correctness issue, not merely a quality degradation. `HiFi4` is therefore non-negotiable for the gate linear and is not evaluated in this chapter.

```python
# moe.py:L1449–L1454 — gate linear, held fixed at HiFi4
compute_kernel_config=ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
),
```

### Current expert compute config

```python
# moe.py:L1152–L1157 — expert matmuls, currently HiFi2
self._expert_compute_cfg = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)
```

`fp32_dest_acc_en=True` means the output accumulation register is 32-bit even though the inputs are bf16. This partially compensates for the reduced mantissa product precision: the partial sums accumulate with full fp32 precision, and only the multiply step uses reduced precision. This combination (`HiFi2` + `fp32_dest_acc_en`) is more precise than bare `HiFi2` without fp32 accumulation.

---

## Accuracy Metrics

Two complementary metrics measure the numerical impact of fidelity on expert outputs. Both compare against a reference computed at higher precision.

### Metric 1: Cosine similarity

Cosine similarity between the output of the tested fidelity and a reference fp32 computation:

```
cosine_sim(ref, out) = dot(ref.flat, out.flat) / (||ref.flat|| × ||out.flat||)
```

Cosine similarity of 1.0 is exact agreement; values approaching 0 indicate orthogonal outputs.

**Interpretation for MoE expert outputs:**

| Cosine similarity | Interpretation | Action |
|---|---|---|
| > 0.9999 | Numerically negligible; fidelity is safe | Proceed to latency validation |
| 0.999 – 0.9999 | Small but measurable deviation | Run end-to-end accuracy test (MMLU, GSM8K) before committing |
| 0.99 – 0.999 | Meaningful precision loss | HiFi2 required; do not adopt LoFi |
| < 0.99 | Severe precision degradation | LoFi is unsuitable; escalate if even HiFi2 fails |

### Metric 2: Max absolute error

```
max_abs_error = max over all output elements of |ref[i] - out[i]|
```

This catches large pointwise errors that cosine similarity can average away.

**Threshold:** For bf16 outputs with values in the range `[-10, 10]` (typical for intermediate MoE activations), a max absolute error below `0.1` is acceptable for `HiFi2`. For `LoFi`, the same threshold applies — if LoFi's max error exceeds `0.1` on a statistically representative batch, it fails the go/no-go criterion.

### Metric 3: Routing-agreement rate (end-to-end gate check)

Even though the gate linear uses `HiFi4`, downstream expert precision errors can in principle affect residual stream quality. Run this check only if cosine similarity is borderline (0.999–0.9999):

- Feed the same input through the full `TTNNMoE.forward` with `HiFi2` and `LoFi` expert configs.
- Compare `topk_indices` (routing decisions) — they must be identical because the gate is unchanged.
- Compare the final `ttnn.add(routed_output, shared)` residual add output using cosine similarity.

If routing decisions are identical (they always should be, since the gate is not changed) but the final residual output cosine similarity falls below 0.9999, the precision loss in expert outputs is propagating.

---

## Measurement Protocol

### Step 1: Single-matmul comparison (isolated)

Run each fidelity setting on a synthetic input matching the GLM-4-MoE expert matmul shapes. Use fp32 host computation as the reference. The expert MLP uses a gated linear unit (GLU) with three weight matrices: `w1` (gate projection), `w3` (up projection), and `w2` (down projection). The correct computation is `gate = x @ w1; up = x @ w3; intermediate = silu(gate) * up; out = intermediate @ w2`.

```python
import ttnn
import torch
import torch.nn.functional as F

TILE = 32
hidden_size       = 4096
intermediate_size = 1408
padded_tokens     = 32   # SPARSITY_BLOCK_SIZE


def make_expert_compute_cfg(fidelity):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


# Use the make_config helper defined in program_config_tuning.md to construct the
# program config. Import or copy it from that shared location; do not re-implement it here.
# make_config(device, out_features, in0_block_w=4) — see program_config_tuning.md


def run_matmul(x_torch, w_torch, out_features, fidelity, device):
    x_tt = ttnn.from_torch(x_torch.bfloat16().unsqueeze(0).unsqueeze(0),
                           layout=ttnn.TILE_LAYOUT, device=device,
                           memory_config=ttnn.L1_MEMORY_CONFIG)
    w_tt = ttnn.from_torch(w_torch.bfloat16(),
                           layout=ttnn.TILE_LAYOUT, device=device,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = ttnn.matmul(x_tt, w_tt,
                      program_config=make_config(device, out_features),  # make_config from program_config_tuning.md
                      compute_kernel_config=make_expert_compute_cfg(fidelity))
    result = ttnn.to_torch(out).squeeze(0).squeeze(0).float()
    ttnn.deallocate(out); ttnn.deallocate(x_tt); ttnn.deallocate(w_tt)
    return result


device = ttnn.open_device(device_id=0)

torch.manual_seed(42)
x_f32 = torch.randn(padded_tokens, hidden_size)
w1_f32 = torch.randn(hidden_size, intermediate_size)   # gate projection
w3_f32 = torch.randn(hidden_size, intermediate_size)   # up projection
w2_f32 = torch.randn(intermediate_size, hidden_size)   # down projection

# fp32 reference (host) — correct GLU formula (gate * up, three weights)
ref_gate  = x_f32 @ w1_f32                                      # gate projection
ref_up    = x_f32 @ w3_f32                                      # up projection
ref_inter = F.silu(ref_gate) * ref_up                           # GLU activation
ref_out   = ref_inter @ w2_f32                                  # down projection


def evaluate(fidelity, label):
    out_gate  = run_matmul(x_f32, w1_f32, intermediate_size, fidelity, device)
    out_up    = run_matmul(x_f32, w3_f32, intermediate_size, fidelity, device)
    out_inter = F.silu(out_gate.bfloat16().float()) * out_up.bfloat16().float()
    out_dn    = run_matmul(out_inter, w2_f32, hidden_size, fidelity, device)

    cos_inter = F.cosine_similarity(ref_inter.flatten(),
                                    out_inter.flatten(), dim=0).item()
    cos_dn    = F.cosine_similarity(ref_out.flatten(),
                                    out_dn.flatten(), dim=0).item()
    mae_inter = (ref_inter - out_inter).abs().max().item()
    mae_dn    = (ref_out   - out_dn).abs().max().item()

    print(f"\n{label}")
    print(f"  gate/up (inter) cosine similarity : {cos_inter:.6f}   max abs error: {mae_inter:.4f}")
    print(f"  down            cosine similarity : {cos_dn:.6f}   max abs error: {mae_dn:.4f}")
    return {"cos_inter": cos_inter, "cos_dn": cos_dn, "mae_inter": mae_inter, "mae_dn": mae_dn}


results = {}
results["HiFi2"] = evaluate(ttnn.MathFidelity.HiFi2, "HiFi2 (current)")
results["LoFi"]  = evaluate(ttnn.MathFidelity.LoFi,  "LoFi  (candidate)")

ttnn.close_device(device)
```

### Step 2: Latency comparison

Extend the harness from `program_config_tuning.md` with a fidelity axis. For each `{in0_block_w, fidelity}` combination, record the mean latency over 100 timed iterations (20 warmup). This produces a 2D grid:

**Gate/up latency grid (w1 or w3 individually): in0=32×4096, out=32×1408**

| `in0_block_w` | `HiFi2` latency (µs) | `LoFi` latency (µs) | LoFi speedup |
|---|---|---|---|
| 4 | ___ | ___ | ___ |
| 8 | ___ | ___ | ___ |

**Down latency grid (w2): in0=32×1408, out=32×4096**

| `in0_block_w` | `HiFi2` latency (µs) | `LoFi` latency (µs) | LoFi speedup |
|---|---|---|---|
| 4 | ___ | ___ | ___ |
| 11 | ___ | ___ | ___ |

### Step 3: End-to-end accuracy check (conditional)

Run only if LoFi passes the single-matmul cosine similarity threshold (> 0.9999):

1. Run 50 decode steps of GLM-4-MoE on a fixed prompt with `HiFi2` expert config; record final logits.
2. Run the same 50 decode steps with `LoFi` expert config; record final logits.
3. Compute token-level agreement rate: fraction of positions where `argmax(logits_LoFi) == argmax(logits_HiFi2)`.
4. Compute perplexity on a held-out set (e.g., WikiText-103, 500 tokens) for each config.

**Accept LoFi only if:** token agreement rate ≥ 99.5% AND perplexity delta ≤ 0.5%.

---

## Go / No-Go Criterion for LoFi

Apply the following decision tree after completing Steps 1–3:

```
1. Single-matmul cosine similarity (Step 1):
   ├─ gate/up (inter) cos_sim < 0.9999  →  NO-GO. LoFi introduces unacceptable
   │                                        precision loss in GLU intermediate output.
   │                                        Retain HiFi2.
   └─ gate/up (inter) cos_sim ≥ 0.9999
       └─ down cos_sim < 0.9999  →  NO-GO. Down projection precision loss
       │                             is too large. Retain HiFi2.
       └─ both cos_sim ≥ 0.9999
           └─ any max_abs_error > 0.1  →  NO-GO. Large pointwise errors
           │                              present. Retain HiFi2.
           └─ all max_abs_error ≤ 0.1
               2. Latency measurement (Step 2):
               └─ LoFi speedup < 5% on Stages 4+5  →  NO-GO. Speedup below
               │                                        noise floor. Retain HiFi2.
               └─ LoFi speedup ≥ 5%
                   3. End-to-end check (Step 3):
                   ├─ token agreement < 99.5%  →  NO-GO. Routing-visible
                   │                               degradation.
                   ├─ perplexity delta > 0.5%  →  NO-GO. Quality regression.
                   └─ both pass  →  GO. Adopt LoFi for expert matmuls.
                                     Document latency gain and accuracy
                                     delta in ch7 optimization matrix.
```

### Expected outcome at batch=1 decode

(See introduction: batch=1 is memory-bound; LoFi's compute savings are irrelevant until batch≥8.)

- LoFi's throughput advantage on the FPU is unlikely to produce a ≥ 5% end-to-end speedup at batch=1; the go/no-go criterion will likely result in NO-GO at Step 2.
- At batch ≥ 8 (compute-bound), LoFi is more likely to produce a measurable speedup and the end-to-end check (Step 3) becomes the binding constraint.

If the batch=1 result is NO-GO on speedup, record the exact LoFi speedup value in the latency grid and note the batch threshold at which LoFi becomes performance-relevant. This informs the Chapter 7 optimization priority matrix.

---

## Measurement Result Tables

### Accuracy results (fill in after running Step 1)

| Fidelity | Inter (gate*up) cos_sim | Down cos_sim | Inter max_abs | Down max_abs | Pass? |
|---|---|---|---|---|---|
| HiFi2 vs fp32 | ___ | ___ | ___ | ___ | ___ |
| LoFi vs fp32 | ___ | ___ | ___ | ___ | ___ |

### Latency results (fill in after running Step 2)

| Fidelity | `in0_block_w` | w1 (µs) | w3 (µs) | w2 (µs) | Stage 4+5 total (µs) |
|---|---|---|---|---|---|
| HiFi2 | 4 | ___ | ___ | ___ | ___ |
| LoFi | 4 | ___ | ___ | ___ | ___ |
| LoFi speedup | — | ___ | ___ | ___ | ___ |

### End-to-end check (fill in after running Step 3, if reached)

| Metric | HiFi2 | LoFi | Delta | Pass? |
|---|---|---|---|---|
| Token agreement rate | — | ___ | ___ | ___ |
| WikiText-103 perplexity | ___ | ___ | ___ | ___ |

### Final go/no-go decision

| Step | Result | Notes |
|---|---|---|
| 1. Single-matmul accuracy | ___ (GO / NO-GO) | |
| 2. Latency speedup | ___ (GO / NO-GO) | |
| 3. End-to-end accuracy | ___ (GO / NO-GO) | |
| **Overall** | **___** | |

---

**Next:** [Chapter 5 — Profiling Methodology](../ch5_profiling_methodology/index.md)
