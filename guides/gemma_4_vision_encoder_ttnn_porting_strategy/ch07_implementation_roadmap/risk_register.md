# Risk Register

This file documents the six primary risks identified for the Gemma 4 vision encoder TTNN porting effort. Each risk is rated by severity (High / Medium / Low), assigned to a phase, and paired with a concrete mitigation strategy. The risks are drawn from the analysis in prior chapters and the implementation plan in [`phased_plan.md`](./phased_plan.md).

## Risk Summary Table

| # | Risk | Severity | Phase | Mitigation |
|---|------|----------|-------|------------|
| 1 | Variable input shapes prevent tracing | High | 3-4 | Pre-trace all five token budgets; pad to nearest budget at runtime |
| 2 | 2D RoPE numerical divergence | Medium-High | 2 | Float32 frequency tables; tight PCC thresholds; layer-by-layer validation |
| 3 | No direct TTNN `avg_pool2d` for adaptive pooling | Medium | 2 | Implement as reshape + `ttnn.mean`; CPU fallback if blocked |
| 4 | `patch_size=16` does not align to tile size 32 | Medium | 2 | Pad to tile boundary; profile overhead |
| 5 | Vision encoder latency does not justify porting effort | Medium | 1 | Run CPU profiling first; abort if latency < 5% of total |
| 6 | Weight format differences between Gemma 3 and Gemma 4 | Low | 2 | Update key mappings early; validate shape expectations |

## Risk 1: Variable Input Shapes Prevent Tracing

> **Risk (High):** The Gemma 4 vision encoder accepts variable aspect ratios, producing different patch grid dimensions per image. TTNN's tracing mechanism requires fixed tensor shapes. If every new image resolution triggers a re-trace or recompilation, the latency benefit of TTNN is negated by compilation overhead.

**Phase affected:** Phase 3 (tracing) and Phase 4 (variable resolution)

**Impact:** Without tracing, each inference call incurs dispatch overhead for all ops across 27 encoder layers. This can add hundreds of microseconds to milliseconds of overhead, potentially erasing the speedup over CPU for small token budgets.

**Likelihood:** High. Variable shapes are inherent to Gemma 4's design. Unlike Gemma 3, which always uses a fixed 896x896 input, Gemma 4 has no single canonical input shape.

**Mitigation:**

1. **Pre-trace the five standard token budgets.** Gemma 4 supports five discrete budgets: 70, 140, 280, 560, and 1120. Each budget constrains the total patch count. Pre-trace the encoder for each budget during model initialization.

2. **Pad to the nearest budget at runtime.** When an image arrives, compute its natural token budget and pad the patch sequence to the nearest standard budget. This ensures the pre-traced program applies.

3. **Group batches by budget.** For batch inference, group images by their token budget so all images in a batch use the same traced program. This avoids mixing budgets within a single batch execution.

4. **Accept per-budget program cache entries.** If full tracing is not feasible (e.g., due to dynamic control flow in the pooler), fall back to TTNN's program cache. Five cached programs (one per budget) still eliminate recompilation for the vast majority of inputs.

> **Tip:** Start with tracing at the 280-token budget only (Phase 3). Extend to all five budgets in Phase 4. This front-loads risk discovery without delaying Phase 3 deliverables.

**Residual risk after mitigation:** Low. The five-budget quantization covers all practical image sizes. Images whose natural budget falls between two standard budgets are padded to the next larger one, wasting some compute but avoiding recompilation.

## Risk 2: 2D RoPE Implementation Has Numerical Divergence

> **Risk (Medium-High):** The 2D factored RoPE applies trigonometric rotations to query and key tensors at every attention layer. In BF16, the limited mantissa precision (7 bits) can cause cos/sin values to diverge from the float32 reference, especially for larger position coordinates. This error accumulates through 27 encoder layers and may degrade the final output quality.

**Phase affected:** Phase 2 (module-level port)

**Impact:** If the PCC between the TTNN attention output and the CPU reference drops below 0.999 at a single layer, the cumulative error across 27 layers may push the end-to-end PCC below the 0.998 acceptance threshold.

**Likelihood:** Medium. Standard 1D RoPE in TTNN language models handles this well because the frequency tables are precomputed in float32. The risk is slightly higher here because the 2D split and concatenation introduces additional operations that may interact poorly with BF16 rounding.

**Mitigation:**

1. **Compute all frequency tables and cos/sin values in float32 on the CPU.** Convert to BF16 only when transferring to device. This is the recommended Strategy 1 from [Chapter 3](../ch03_2d_factored_rope/ttnn_rope_gap_analysis.md).

2. **Validate per-layer, not just end-to-end.** In Phase 2, validate the attention module output at each of the 27 layers independently. If a specific layer shows PCC degradation, investigate whether the issue is in the RoPE application or in other attention operations.

3. **Compare rotation angles directly.** Before validating the full attention output, compare the cos/sin tables transferred to the device against the float32 reference. Any divergence at this stage indicates a precision issue in the table computation itself, not in the application.

4. **Set a strict threshold: PCC > 0.9995 for the RoPE module in isolation.** The RoPE module alone (before Q*K^T matmul) should be nearly exact because it consists only of element-wise multiplications and additions with precomputed constants.

**Residual risk after mitigation:** Low. CPU-precomputed float32 tables converted to BF16 are the standard approach for TTNN RoPE and have been validated in production for language models.

## Risk 3: Adaptive Pooling Has No Direct TTNN Op

> **Risk (Medium):** The Gemma 4 vision pooler performs 2D average pooling with `kernel_size=3` and `stride=3` over the patch grid. TTNN's `ttnn.avg_pool2d` may not support the exact combination of kernel size, stride, and padding required, or may not handle the variable spatial dimensions that arise from different token budgets.

**Phase affected:** Phase 2 (module-level port)

**Impact:** If the pooler cannot be implemented efficiently on device, it must fall back to CPU. The pooler runs once per image (after the 27 encoder layers) so the latency impact is bounded, but the host-device data transfer adds overhead.

**Likelihood:** Medium. TTNN's `avg_pool2d` is primarily optimized for common CNN pooling configurations. A 3x3 kernel with stride 3 is a valid configuration but may not be on the optimized fast path.

**Mitigation:**

1. **Test `ttnn.avg_pool2d` first.** Attempt the pooling with the exact kernel/stride/padding parameters. If it works and meets performance requirements, use it directly.

2. **Fallback: reshape + `ttnn.mean`.** Reshape the `[batch, height_patches, width_patches, hidden_size]` tensor so that each 3x3 neighborhood is grouped along a new dimension, then apply `ttnn.mean` along that dimension:

```python
# Reshape to group 3x3 neighborhoods
# [batch, H_patches, W_patches, 1152] -> [batch, H_pool, 3, W_pool, 3, 1152]
x = ttnn.reshape(x, [batch, h_pool, 3, w_pool, 3, hidden_size])
# Mean over the 3x3 kernel dimensions
x = ttnn.mean(x, dim=[2, 4])  # [batch, H_pool, W_pool, 1152]
```

3. **Fallback: CPU pooling.** If neither TTNN approach works, perform the pooling on CPU. Transfer the encoder output (approximately 2520 x 1152 x 2 bytes = 5.8 MB at the 280-token budget) to the host, pool, and transfer the result (280 x 1152 x 2 bytes = 645 KB) back. At PCIe Gen4 bandwidth, the round trip adds approximately 0.5 ms — acceptable if it unblocks the rest of the pipeline.

**Residual risk after mitigation:** Low. At least one of the three approaches will work. CPU fallback is the guaranteed escape hatch with bounded latency impact.

## Risk 4: `patch_size=16` Does Not Align to Tile Size 32

> **Risk (Medium):** Tenstorrent Wormhole hardware operates on 32x32 tiles as the atomic compute unit. The patch embedding produces tensors with a patch dimension that is a multiple of the patch grid size, not necessarily a multiple of 32. Additionally, the patch flattening produces vectors of length 768 (16x16x3), and the head dimension is 72 — neither of which is a multiple of 32.

**Phase affected:** Phase 2 (module-level port)

**Impact:** Non-tile-aligned dimensions require padding, which wastes compute and memory bandwidth. For the head dimension (72, padded to 96 or 128), this represents 33-78% overhead on per-head operations. For the patch dimension, the overhead depends on the specific grid size.

**Likelihood:** High that padding is needed. The question is whether the overhead is acceptable.

**Mitigation:**

1. **Accept tile padding for head_dim=72.** Pad to 96 (nearest multiple of 32) for a 33% overhead. This is consistent with how other non-power-of-2 head dimensions are handled in TTNN language models. The overhead applies only to per-head operations (RoPE application, Q*K^T, attn*V), not to the full hidden_size=1152 matmuls.

2. **Profile to quantify the overhead.** In Phase 3, compare the actual execution time against the theoretical minimum (no padding). If padding overhead exceeds 20% of total latency, investigate:
   - Using `width_sharded` memory configs to avoid padding along the sequence dimension.
   - Packing multiple heads into a single tile-aligned block before computing attention.

3. **For the patch dimension:** The number of patches depends on the image resolution. At the 280-token budget, pre-pooling patch counts of ~2520 are not multiples of 32 (2520 = 32 x 78 + 24). TTNN will pad the sequence dimension to the next multiple of 32 (2528). This is only 0.3% overhead — negligible.

4. **For the flattened patch vector (768):** This is 24 x 32, a perfect multiple of 32. No padding needed here.

**Residual risk after mitigation:** Low. The head_dim padding is the primary concern and the 33% overhead on per-head operations is within acceptable bounds for initial bringup. Optimization can reduce this in Phase 3 if profiling shows it matters.

## Risk 5: Vision Encoder Latency Does Not Justify Porting Effort

> **Risk (Medium):** The vision encoder has approximately 570M parameters — roughly 2% of the 31B total model. If the deployment scenario is dominated by language model decode latency (which is the case for long-form text generation from a single image), the vision encoder's contribution to total latency may be negligible. In that case, 5-7 weeks of engineering effort for the TTNN port may not be the best use of resources.

**Phase affected:** Phase 1 (CPU profiling determines go/no-go)

**Impact:** If the port is not justified, the engineering effort is wasted. If the port is prematurely abandoned, the team loses the potential latency improvement for high-throughput scenarios.

**Likelihood:** Medium. The justification depends heavily on the deployment scenario:
- Single-image, long-generation: vision encoder runs once, language model generates hundreds of tokens. Vision encoder latency is amortized and may be < 5% of total.
- Batch inference with many images: vision encoder latency scales linearly with batch size and becomes a meaningful fraction. Port is justified.
- Continuous batching with mixed modalities: CPU vision encoder can stall the pipeline. Port is strongly justified.

**Mitigation:**

1. **Run Phase 1 before committing to Phases 2-4.** The CPU profiling in Phase 1 provides the data needed for the go/no-go decision. This is a 1-week investment that protects against 4-6 weeks of potentially unjustified work.

2. **Use the decision matrix from [Chapter 5](../ch05_cpu_vs_ttnn_latency/decision_matrix.md).** The matrix provides concrete thresholds for when the port is justified based on deployment scenario, token budget, and batch size.

3. **Consider a partial port.** If the full port is not justified but the team wants to reduce vision encoder latency, port only the attention and MLP matmuls (the dominant cost, ~85-90% of vision encoder compute) and leave the patch embedding and pooler on CPU.

**Residual risk after mitigation:** Low. Phase 1 acts as a circuit breaker that prevents wasted effort.

## Risk 6: Weight Format Differences Between Gemma 3 and Gemma 4 Checkpoints

> **Risk (Low):** The Gemma 4 checkpoint may use different key names, weight shapes, or storage formats compared to Gemma 3. The existing `load_checkpoints.py` infrastructure expects Gemma 3 key conventions. If the key mapping is not updated correctly, modules will load incorrect weights and produce garbage outputs.

**Phase affected:** Phase 2 (weight loading)

**Impact:** Incorrect weight loading produces completely wrong outputs (PCC near 0). However, this is easy to detect and fix — the failure mode is obvious, not subtle.

**Likelihood:** High that key names differ (Gemma 4 modules have different class names). Low that the data format itself (BF16 tensor storage) changes.

**Mitigation:**

1. **Map all weight keys early in Phase 2.** Before porting any module, dump the full list of keys from the Gemma 4 checkpoint and create a key mapping table from Gemma 4 keys to the TTNN module parameter names.

2. **Validate weight shapes at load time.** Add assertions in `gemma4_load_checkpoints.py` that verify each loaded tensor has the expected shape:

```python
def load_weight(state_dict, key, expected_shape):
    tensor = state_dict[key]
    assert tensor.shape == expected_shape, (
        f"Weight {key}: expected shape {expected_shape}, got {tensor.shape}"
    )
    return tensor
```

3. **Cross-reference against the config.** The `Gemma4VisionConfig` specifies all dimension parameters. Use these to compute expected weight shapes programmatically rather than hard-coding them.

4. **Test weight loading independently.** Before running any TTNN inference, load all weights into a PyTorch model with the same architecture and verify that the PyTorch model produces the same output as the HuggingFace reference. This isolates weight loading bugs from TTNN implementation bugs.

**Residual risk after mitigation:** Negligible. Weight loading issues are caught immediately by PCC validation and are straightforward to fix.

## Risk Interaction Matrix

Some risks compound each other:

| Risk Pair | Interaction |
|-----------|-------------|
| Risk 1 (variable shapes) + Risk 4 (tile alignment) | Variable shapes mean variable padding overhead. The worst case is a patch count that is 1 more than a multiple of 32, requiring 31 elements of padding. |
| Risk 2 (RoPE divergence) + Risk 4 (tile alignment) | Head_dim=72 padded to 96 means the RoPE cos/sin tables must also be padded. Ensure padding values are 1.0 for cos and 0.0 for sin (identity rotation) to avoid corrupting the padded dimensions. |
| Risk 1 (variable shapes) + Risk 5 (latency justification) | If tracing fails for variable shapes and the fallback (per-budget program cache) has high dispatch overhead, the TTNN speedup may be insufficient to justify the port. Phase 1 profiling and Phase 3 tracing attempts together determine viability. |

## Monitoring During Implementation

Track these metrics throughout the implementation to detect risks early:

| Metric | Threshold | Check Frequency | Action if Exceeded |
|--------|-----------|----------------|-------------------|
| Module PCC vs. reference | < 0.999 | Every module (Phase 2) | Investigate precision; consider float32 accumulation |
| End-to-end PCC | < 0.998 | Phase 3 integration | Bisect layers to find divergence source |
| TTNN latency vs. CPU | < 2x speedup | Phase 3 profiling | Profile individual ops; optimize memory configs |
| Compilation time per budget | > 60 seconds | Phase 4 | Ensure programs are cached; investigate what triggers recompilation |
| Device memory utilization | > 90% of DRAM | Phase 3 | Review sharding; consider offloading position embedding table |

---

**End of guide.** Return to [Guide Index](../index.md)
