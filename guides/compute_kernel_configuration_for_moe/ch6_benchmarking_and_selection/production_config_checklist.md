# Production Config Checklist

## Purpose

This checklist is the last gate before a `WormholeComputeKernelConfig` change ships to a production MoE deployment. Work through every item in order. Items marked **[BLOCKING]** must pass before the change is merged; items marked **[RECOMMENDED]** are strongly advised but can be deferred to a follow-up if timeline requires.

---

## Pre-Deployment Checklist

### 1. PCC Threshold Verified [BLOCKING]

- [ ] PCC measured against a float32 PyTorch reference for each projection type (gate, up, down) independently.
- [ ] PCC measured at layer level: full expert FFN (gate + up + SiLU + element-mul + down) as a unit.
- [ ] Both token-level and layer-level PCC reported; layer-level PCC is the binding threshold.
- [ ] Measured at seq >= 128 (prefill) or batch >= 32 (decode) — not on a single decode token.
- [ ] PCC >= 0.999 for all projections, or a lower threshold is explicitly documented and approved.
- [ ] For models with top-K >= 4: layer-level PCC measured with actual model weights (not just synthetic random tensors).

### 2. Latency Measured at Both Regimes [BLOCKING]

- [ ] Per-op on-device latency measured in prefill regime: seq = 512 (or the largest expected sequence length).
- [ ] Per-op on-device latency measured in decode regime: batch = 1 and batch = 32 (or max production batch size).
- [ ] Latency reported in microseconds from `ttnn.device.profiler` or Tracy — not wall-clock Python time.
- [ ] Latency compared against the previous config (or TTNN default) to confirm improvement or document any regression.
- [ ] Warm-up iterations (>= 3) excluded from all reported latency measurements.

### 3. L1 Budget Confirmed [BLOCKING]

- [ ] A dry-run dispatch (no profiling) executed at the maximum expected batch size with the target config.
- [ ] No `ttnn.exceptions.TTNNException` or allocation error raised at dispatch time.
- [ ] If `fp32_dest_acc_en=True` is used alongside `packer_l1_acc=True`: L1 budget explicitly verified because the fp32 accumulation buffer is 2x the size of the BF16 buffer and can overflow the 1.5 MB per-core L1 on Wormhole B0.
- [ ] If any L1 allocation error is observed: reduce `per_core_N` or `out_subblock_w` in the associated `MatmulMultiCoreReuseMultiCastProgramConfig` before disabling `packer_l1_acc`.

> **Warning:** The most common L1 overflow pattern is `packer_l1_acc=True` combined with `fp32_dest_acc_en=True` at large `per_core_N`. The accumulation buffer for a single output tile is `(per_core_N * out_subblock_h * 32 * 32 * 4 bytes)` in fp32. For `per_core_N=8` and `out_subblock_h=4`, that is 8 × 4 × 1024 × 4 = 131,072 bytes = 128 KB per core — a significant fraction of the 1.5 MB L1 budget before accounting for input and weight tile buffers.

### 4. Configs Stored in a Shared `config_helpers.py` [RECOMMENDED]

- [ ] `COMPUTE_KERNEL_CONFIG_LOFI` and `COMPUTE_KERNEL_CONFIG_HIFI2` defined once in a shared helper module, not inline at each call site.
- [ ] Any HiFi4 or custom configs also defined in the shared module with a descriptive name and inline comment explaining when to use them.
- [ ] All expert matmul call sites import from the shared module rather than constructing `WormholeComputeKernelConfig` directly.

The reference pattern is `models/demos/deepseek_v3/utils/config_helpers.py`. Define your model's config helpers in an analogous location, e.g., `models/demos/<your_model>/utils/config_helpers.py`. For the config constructor values, see Chapter 1, `ch1_kernel_config_fundamentals/wormhole_compute_kernel_config_api.md` §Canonical Production Configs. Each config constant should include an inline comment stating the projection type and hardware constraint it is intended for (e.g., "Safe for any SwiGLU/SiLU MoE gate/up projection on Wormhole B0" and "Required for down projections that accumulate into the residual stream").

### 5. Firmware Version Pinned [RECOMMENDED]

- [ ] The tt-metal firmware version used during benchmarking is recorded alongside the config values in the model's documentation or a comment in `config_helpers.py`.
- [ ] If the model is deployed against a pinned tt-metal release, that release hash or version string is noted in the `config_helpers.py` header.

> **Warning:** `WormholeComputeKernelConfig` parameters can change behavior across tt-metal firmware releases. A kernel config that achieves PCC = 0.9998 on one firmware version may behave differently after a firmware update if the underlying microcode for a fidelity level is revised. Re-run PCC and latency validation after any tt-metal upgrade.

---

## Handling Heterogeneous Expert Sizes

Some MoE models include more than one class of expert with different FFN dimensions. DeepSeek-V3 is the canonical example: it has both **routed experts** (256 active from 256 total, d_ff = 2048) and **shared experts** (always active, dense FFN, larger d_ff).

### Routed Experts (Standard MoE FFN)

Apply the two-config pattern from Chapter 5 directly: `COMPUTE_KERNEL_CONFIG_LOFI` for gate and up, `COMPUTE_KERNEL_CONFIG_HIFI2` for down.

### Shared Experts (Dense FFN, Always Active)

Shared experts are not gated — they do not use softmax routing and their output is always added to the residual stream. This changes two properties:

1. **There is no softmax routing to absorb gate rounding bias.** The shared expert gate projection output feeds directly into an activation; the reasoning for LoFi tolerance still holds structurally if the activation is SiLU.
2. **The shared expert down projection is always active**, contributing to every token's residual stream. There is no routing-weight averaging of rounding errors across multiple experts. The rounding bias is deterministic and unreduced.

**Recommendation for shared experts:** Start with the same two-config pattern, but benchmark the down projection at both HiFi2 and HiFi4. If the shared expert has a larger d_ff than the routed experts (common in models that use shared experts for global reasoning capacity), the larger K_t amplifies rounding accumulation. Use the d_ff/d_model guidance from `config_decision_matrix.md`.

```python
# Example: separate configs for routed and shared experts
# Routed experts: standard two-config pattern
COMPUTE_KERNEL_CONFIG_ROUTED_GATE_UP = COMPUTE_KERNEL_CONFIG_LOFI
COMPUTE_KERNEL_CONFIG_ROUTED_DOWN    = COMPUTE_KERNEL_CONFIG_HIFI2

# Shared experts: benchmark to determine; start with HIFI2 for down
# and elevate to HiFi4 if layer-level PCC is below threshold
COMPUTE_KERNEL_CONFIG_SHARED_GATE_UP = COMPUTE_KERNEL_CONFIG_LOFI
COMPUTE_KERNEL_CONFIG_SHARED_DOWN    = COMPUTE_KERNEL_CONFIG_HIFI2  # or HiFi4 after benchmarking
```

---

## Common Mistake: L1 Budget Not Verified

The most frequent error when enabling `packer_l1_acc=True` combined with `fp32_dest_acc_en=True` is skipping the L1 verification step. The failure mode is an allocation error at op dispatch time:

```
ttnn.exceptions.TTNNException: ... L1 allocation failed ...
```

This error occurs at runtime, not at model initialization, and only manifests at the batch size or sequence length that triggers the specific program config tile layout that overflows L1. It may not appear during a quick smoke test at small batch size but will crash in production at max batch size.

**Prevention procedure:**

See Chapter 3, `packer_l1_acc_constraints.md` for the L1 accumulation buffer arithmetic and overflow diagnosis procedure.

Run a dry dispatch at max batch size with the config before running the profiler.

> **Tip:** If you hit an L1 allocation error, the correct fix is to reduce `per_core_N` or `out_subblock_w` in the `MatmulMultiCoreReuseMultiCastProgramConfig` — not to disable `packer_l1_acc`. Disabling `packer_l1_acc` to avoid the overflow trades a configuration problem for a performance regression, and it is always possible to find a tiling that fits.

---

## Checklist Summary Card

Copy and annotate this card when starting a new model deployment:

```
MoE Compute Kernel Config Pre-Deployment Checklist
Model: ___________________  Firmware: ___________________

PCC Verification
[ ] Gate projection PCC (token-level):   ______   >= 0.999?
[ ] Up projection PCC (token-level):     ______   >= 0.999?
[ ] Down projection PCC (token-level):   ______   >= 0.999?
[ ] Layer-level PCC (full expert FFN):   ______   >= 0.999?
[ ] Measured at seq >= 128 or batch >= 32: yes/no
[ ] Measured with actual model weights:    yes/no

Latency Verification
[ ] Gate projection latency (prefill, seq=512):  ______ µs
[ ] Gate projection latency (decode, batch=32):  ______ µs
[ ] Down projection latency (prefill, seq=512):  ______ µs
[ ] Down projection latency (decode, batch=32):  ______ µs
[ ] Improvement vs. baseline:  ______%

L1 Budget
[ ] Dry dispatch at max batch size: pass/fail
[ ] fp32_dest_acc_en=True used anywhere: yes/no
    If yes: accumulation buffer size verified: ______ KB

Config Storage
[ ] Configs defined in config_helpers.py:  yes/no
[ ] Firmware version noted in comment:     yes/no
```

---

## Next Steps

This is the final file in Chapter 6. You have now completed the full guide:

- **Chapter 1–4**: Parameter fundamentals — what each field controls at the hardware level.
- **Chapter 5**: Production config pattern — LOFI for gate/up, HIFI2 for down, validated on DeepSeek-V3 and applicable to Qwen MoE.
- **Chapter 6**: Benchmarking methodology, decision matrix, and production checklist.

To apply this guide to a new model, start with `config_decision_matrix.md`, run the benchmarks from `benchmarking_methodology.md`, and complete this checklist before merging.
