# Correctness Verification

## Overview

Resharding a weight tensor from interleaved DRAM to DRAM-sharded layout changes how the tensor's pages are addressed in hardware, but does not alter the tensor values. A correctly executed reshard produces bit-identical weight reads — every element in the sharded tensor is numerically the same as in the original interleaved tensor.

Despite this invariant, correctness failures can occur if the shard configuration violates tile-alignment rules (Chapter 5), causing partial-tile reads or incorrect shard boundary placement. This file explains how to detect such failures using Pearson Cross-Correlation (PCC) and how to distinguish layout-induced failures from floating-point precision differences.

---

## PCC: Definition and Interpretation

**Pearson Cross-Correlation (PCC)** measures the linear correlation between two output tensors:

```
PCC(A, B) = cov(A_flat, B_flat) / (std(A_flat) × std(B_flat))
```

where `A_flat` and `B_flat` are the flattened 1D views of the two tensors. PCC equals 1.0 for perfectly identical tensors and approaches 0 for uncorrelated noise.

In TTNN development, PCC is the standard numerical correctness metric because it is robust to small, uniform scaling differences (which indicate precision reduction) while being sensitive to structural errors (which indicate data corruption or layout bugs).

**Acceptable thresholds:**

| Dtype | PCC threshold | Interpretation |
|---|---|---|
| `bfloat16` | > 0.999 | Near-bit-identical; layout change only |
| `bfloat8_b` | > 0.999 | Acceptable precision reduction from quantization |
| Any | < 0.99 | Likely data corruption; investigate shard config |
| Any | < 0.9 | Severe misalignment; shard boundary or dtype mismatch |

A PCC between 0.99 and 0.999 for bfloat16 output suggests a partial-tile read: some weight elements are being fetched from the wrong shard boundary, contaminating the matmul accumulation with incorrect values. See the diagnosis section below.

---

## Step-by-Step Verification Workflow

The workflow compares two TTNN inference runs — one with interleaved weights (the reference), one with DRAM-sharded weights (the candidate) — using the same random activation input. Since only the weight addressing changes, the outputs should be numerically identical up to floating-point rounding in the matmul accumulator.

```python
import ttnn
import torch
import numpy as np

# -----------------------------------------------------------------------
# Helper: compute PCC between two TTNN tensors by converting to CPU.
# -----------------------------------------------------------------------
def compute_pcc(ttnn_tensor_a: ttnn.Tensor, ttnn_tensor_b: ttnn.Tensor) -> float:
    """
    Return Pearson Cross-Correlation between two TTNN tensors.
    Both tensors are moved to CPU for the comparison.
    """
    a = ttnn.to_torch(ttnn_tensor_a).float().flatten()
    b = ttnn.to_torch(ttnn_tensor_b).float().flatten()
    # Use numpy corrcoef for numerical stability.
    pcc = float(np.corrcoef(a.numpy(), b.numpy())[0, 1])
    return pcc


# -----------------------------------------------------------------------
# Step 1: Prepare reference output using interleaved DRAM weights.
# -----------------------------------------------------------------------
# Use a fixed random activation for reproducibility.
torch.manual_seed(42)
cpu_activation = torch.randn(1, 1, 4096, dtype=torch.bfloat16)  # Decode: batch=1, seq=1.
activation = ttnn.from_torch(
    cpu_activation,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.L1_MEMORY_CONFIG,
)

# Run with interleaved weights (gate_weights, up_weights, down_weights from Step 1
# of code_patterns.md — before resharding).
ref_output = expert_ffn_forward(
    activation,
    gate_weights[0],   # Interleaved DRAM.
    up_weights[0],
    down_weights[0],
    lofi_config,
)
# Move to CPU for storage.
ref_cpu = ttnn.to_torch(ref_output)
ttnn.deallocate(ref_output)


# -----------------------------------------------------------------------
# Step 2: Reshard weights (if not already done) and run candidate.
# -----------------------------------------------------------------------
# sharded_gate, sharded_up, sharded_down are produced by reshard_expert_weights()
# in code_patterns.md.

candidate_output = expert_ffn_forward(
    activation,
    sharded_gate[0],   # DRAM-sharded.
    sharded_up[0],
    sharded_down[0],
    lofi_config,
)
candidate_cpu = ttnn.to_torch(candidate_output)
ttnn.deallocate(candidate_output)
ttnn.deallocate(activation)


# -----------------------------------------------------------------------
# Step 3: Compute and check PCC.
# -----------------------------------------------------------------------
ref_tt = ttnn.from_torch(ref_cpu, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                          device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
cand_tt = ttnn.from_torch(candidate_cpu, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.L1_MEMORY_CONFIG)

pcc = compute_pcc(ref_tt, cand_tt)
print(f"PCC (interleaved vs DRAM-sharded): {pcc:.6f}")
assert pcc > 0.999, f"PCC {pcc:.6f} below bfloat16 threshold 0.999 — check shard config."
```

> **Tip:** Run the PCC check across at least 5 independent random activation inputs before declaring the configuration correct. A shard misalignment that only affects certain tile boundaries may not manifest with every input. Use `torch.manual_seed(i)` in a loop over `i in range(5)`.

---

## Multi-Trial Verification Loop

```python
PASS_THRESHOLD_BF16 = 0.999
NUM_TRIALS = 5

all_pcc = []
for trial in range(NUM_TRIALS):
    torch.manual_seed(trial)
    cpu_act = torch.randn(1, 1, 4096, dtype=torch.bfloat16)
    act = ttnn.from_torch(cpu_act, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.L1_MEMORY_CONFIG)

    ref  = expert_ffn_forward(act, gate_weights[0],  up_weights[0],  down_weights[0],  lofi_config)
    cand = expert_ffn_forward(act, sharded_gate[0], sharded_up[0], sharded_down[0], lofi_config)

    pcc_val = compute_pcc(ref, cand)
    all_pcc.append(pcc_val)
    print(f"Trial {trial}: PCC = {pcc_val:.6f}")

    ttnn.deallocate(ref)
    ttnn.deallocate(cand)
    ttnn.deallocate(act)

min_pcc = min(all_pcc)
print(f"\nMin PCC across {NUM_TRIALS} trials: {min_pcc:.6f}")
assert min_pcc > PASS_THRESHOLD_BF16, (
    f"PCC verification FAILED. Min PCC = {min_pcc:.6f}, threshold = {PASS_THRESHOLD_BF16}."
)
print("Correctness verification PASSED.")
```

---

## Verifying Individual Projections

When a full FFN forward pass fails PCC, isolate the failure to a specific projection by testing gate, up, and down matmuls independently.

```python
def verify_single_projection(weight_interleaved, weight_sharded, d_in, d_out, label):
    """Test a single matmul projection in isolation."""
    cpu_x = torch.randn(1, 1, d_in, dtype=torch.bfloat16)
    x = ttnn.from_torch(cpu_x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                         device=device, memory_config=ttnn.L1_MEMORY_CONFIG)

    ref  = ttnn.matmul(x, weight_interleaved, memory_config=ttnn.L1_MEMORY_CONFIG,
                        compute_kernel_config=lofi_config, dtype=ttnn.bfloat16)
    cand = ttnn.matmul(x, weight_sharded,     memory_config=ttnn.L1_MEMORY_CONFIG,
                        compute_kernel_config=lofi_config, dtype=ttnn.bfloat16)

    pcc_val = compute_pcc(ref, cand)
    status = "PASS" if pcc_val > 0.999 else "FAIL"
    print(f"[{status}] {label}: PCC = {pcc_val:.6f}")

    ttnn.deallocate(ref)
    ttnn.deallocate(cand)
    ttnn.deallocate(x)
    return pcc_val


verify_single_projection(gate_weights[0], sharded_gate[0], d_model=4096, d_out=14336, label="gate")
verify_single_projection(up_weights[0],   sharded_up[0],   d_model=4096, d_out=14336, label="up")
verify_single_projection(down_weights[0], sharded_down[0], d_model=14336, d_out=4096, label="down")
```

---

## Diagnosing PCC Failures

### Failure pattern 1: PCC between 0.99 and 0.999 (partial-tile contamination)

This pattern indicates that some weight elements are being read from incorrect positions due to a shard boundary misalignment. The error is not random noise but structured — specific output positions are systematically wrong.

**Diagnostic steps:**

1. Print `weight_sharded.shard_spec().shape` and verify it satisfies all five Rules from Chapter 5.
2. Check that `d_ff % num_banks == 0`: if not, the final shard is narrower than the others, causing the matmul to read zero-padded tiles for the last shard.
3. Verify that `shard_shape[0] % 32 == 0` and `shard_shape[1] % 32 == 0` (Rules 1 and 2).
4. Check that `ShardSpec.shape` is in elements, not tiles (Chapter 5 Pitfall 2). A common error is passing `[d_model // 32, d_ff // 32]` (tile counts) instead of `[d_model, d_ff // num_banks]` (element counts).

### Failure pattern 2: PCC below 0.9 (severe data corruption)

Typically caused by a `buffer_type` or `memory_layout` mismatch: the `MemoryConfig` specifies `L1` but the tensor is in DRAM, or `HEIGHT_SHARDED` is used with a `CoreRangeSet` sized for `WIDTH_SHARDED`.

**Diagnostic steps:**

1. Inspect `weight_sharded.memory_config()` and confirm `buffer_type=DRAM` and `memory_layout` matches the intended strategy.
2. Print the `CoreRangeSet`: the number of cores must equal `d_ff // shard_w` for WIDTH_SHARDED or `d_ff // shard_h` for HEIGHT_SHARDED.
3. Ensure the weight tensor is in `TILE_LAYOUT` before resharding (Chapter 5 Pitfall 1). Use `weight.layout()` to verify.

### Failure pattern 3: PCC is exactly 1.0 but output dtype is wrong

This can happen if the matmul output is silently cast to a lower-precision dtype (e.g., `bfloat8_b` instead of `bfloat16`). PCC of 1.0 with unexpected output magnitude is normal for bfloat8_b — the values are quantized but correlated.

**Diagnostic step:** Check `output.dtype()` after the matmul call and ensure it matches the `dtype=` argument passed to `ttnn.matmul`.

> **Warning:** A PCC of 1.0 between two TTNN tensors does not guarantee bit-exact equality. PCC measures correlation, not identity. If bit-exact verification is required (e.g., for determinism testing), compare `ttnn.to_torch(a) == ttnn.to_torch(b)` elementwise and check that the fraction of equal elements exceeds 99.9%.

---

## Expected PCC Ranges by Configuration

| Comparison | Expected PCC | Reason |
|---|---|---|
| BF16 interleaved vs BF16 DRAM-sharded | > 0.999 | Layout change only; values identical |
| BF16 vs bfloat8_b (same weights) | 0.999–0.999 | Quantization error from reduced mantissa bits |
| LOFI vs HIFI2 compute kernel | 0.999–0.999 | Reduced math fidelity in LOFI |
| BF16 TTNN vs PyTorch CPU reference | > 0.999 | TTNN BF16 matmul matches PyTorch BF16 |
| bfloat8_b TTNN vs PyTorch CPU reference | > 0.999 | Acceptable; use HIFI2 if threshold is marginal |

---

## Next Steps

Once PCC verification passes across all projections and multiple trials, proceed to `benchmark_methodology.md` to measure the actual latency and bandwidth improvement from DRAM-sharded weights versus interleaved, using a reproducible benchmark harness with Tracy profiler support.
