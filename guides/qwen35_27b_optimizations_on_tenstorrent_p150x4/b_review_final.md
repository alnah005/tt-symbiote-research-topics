# Final Cross-Chapter Consistency Review

## Issue 1: Source filename inconsistency -- `gated_delta_net.py` vs `gdn.py`

The top-level `index.md` (line 72) lists the GDN implementation file as `gated_delta_net.py`. Every other chapter that references this file -- Ch3 index (line 5), `gdn_decode_flow.md` (line 3), `batched_projections.md` (line 42), `gdn_prefill_strategy.md` (lines 5, 9), `state_replication.md` (line 9) -- calls it `gdn.py`. One of these names is wrong. Pick one and fix the other.

**Fix:** Update the `index.md` source code tree to use `gdn.py` (or vice versa), matching the actual filename in the repository.

## Issue 2: Contradictory total state across 48 GDN layers -- 576 MB vs 605 MB

- Ch1 `hybrid_architecture.md` (line 59): "576 MB across 48 GDN layers per device"
- Ch3 `recurrence_math.md` (line 169): "48 * 12 MB ~= 576 MB per device"
- Ch3 `conv1d_shift_register.md` (line 101): "~605 MB per device for 48 layers at 12.6 MB each"
- Ch6 `l1_state_design.md` (line 3): "The total state footprint of 605 MB"

The 576 MB figure comes from 48 * 12.0 MB (raw bytes). The 605 MB figure comes from 48 * 12.6 MB (tile-aligned bytes). Both calculations are internally valid, but using two different numbers for the same quantity across chapters is confusing.

**Fix:** Pick one consistent number with an explicit basis. Recommended: use 12.0 MB / 576 MB everywhere for raw state size, and note tile overhead parenthetically in the one place it matters (Ch7 `performance_summary.md` where the tile arithmetic is derived). Alternatively, use 12.6 MB / 605 MB everywhere and note it includes tile alignment.

## Issue 3: Per-layer state size stated as both "12 MB" and "12.6 MB"

Related to Issue 2 but appearing at the per-layer level:

- Ch3 `recurrence_math.md` (line 168): "12,288 KB = 12 MB (no tile padding -- both dimensions are exact multiples of 32)"
- Ch7 `performance_summary.md` (line 38): "12,582,912 bytes = 12.6 MB including tile alignment overhead"

Ch3 explicitly states there is no tile padding (128 and 128 are multiples of 32), yet Ch7 computes 12.6 MB "including tile alignment overhead" from the same dimensions. If both dimensions are exact multiples of tile size, there should be no alignment overhead, and both numbers should be 12.0 MB. The Ch7 calculation `384 * 16 * 2048 = 12,582,912` is correct arithmetic but equals 12.0 MiB, not 12.6 MB -- the discrepancy arises from mixing MiB and MB units without stating so.

**Fix:** In Ch7 `performance_summary.md`, note that `12,582,912 bytes = 12.0 MiB = 12.6 MB (decimal)` to resolve the apparent contradiction, or simply use 12 MB consistently since the tile padding claim is unfounded for these dimensions.

No other cross-chapter factual or consistency errors found. Navigation footer chains are complete and correct across all content files. All cross-chapter references resolve to valid files.

## Pass 2

### Pass 1 Fix Verification

All three Pass 1 fixes are confirmed in place:

1. `index.md` source tree (line 72): now lists `gdn.py`. Fixed.
2. Total 48-layer state: `conv1d_shift_register.md` (line 94) and `l1_state_design.md` (line 3) both read 576 MB. Fixed.
3. Per-layer state: `l1_state_design.md` (line 3) reads 12.0 MB; `performance_summary.md` (line 35) reads 12.0 MB; `recurrence_math.md` (line 147) reads 12 MB. Fixed.

### Remaining Issue

**Issue 4: HiFi2 vs HiFi4 for GDN recurrence — cross-chapter contradiction**

`ch3_gdn_layer_decode_pipeline/recurrence_math.md` (lines 171–188) states that all matmul operations in `TtGatedDeltaNet` use `args.compute_kernel_config_hifi2` (HiFi2), cites `gdn.py` line 116, prints the `COMPUTE_HIFI2` struct, and explicitly concludes: "HiFi2 mode truncates one mantissa operand for higher throughput … This trade-off is accepted for the GDN recurrence as a performance optimization."

Every other chapter that addresses this point states the opposite:

- `ch1_architecture_and_hardware_mapping/tp_sharding_strategy.md` (line 84): "`COMPUTE_HIFI4` … Used for the GDN recurrence computation where numerical precision is critical."
- `ch4_custom_fused_gdn_kernel/kernel_dispatch.md` (line 116): "specifying `MathFidelity.HiFi4`, `fp32_dest_acc_en=True`, and `math_approx_mode=False` for maximum numerical precision during the recurrence."
- `ch4_custom_fused_gdn_kernel/compute_kernel.md` (line 5): "The kernel runs with `fp32_dest_acc_en=True` and `MathFidelity.HiFi4` (verified at `gdn_kernel_op.py` lines 485–490)."
- `ch7_performance_analysis/bottleneck_analysis.md` (lines 83, 86): "HiFi4 for the fused recurrence kernel … The recurrence uses HiFi4 because the iterative state update accumulates numerical error across tokens."

The `gdn.py` line 116 reference in Ch3 is correct for the unfused projection matmuls (`self.compute_cfg`), but the fused kernel uses a separate `ComputeConfigDescriptor` with HiFi4 (verified at `gdn_kernel_op.py` lines 485–490 per Ch4). Ch3 conflates the two and incorrectly applies `COMPUTE_HIFI2` to the recurrence itself.

**Fix:** In `recurrence_math.md`, the Numerical Precision section should distinguish between `self.compute_cfg` (HiFi2, used for the unfused path's weight projections) and the fused kernel's HiFi4 config. The concluding sentence claiming HiFi2 is accepted for the GDN recurrence should be corrected to note that the fused kernel uses HiFi4; `COMPUTE_HIFI2` applies only to the projection matmuls in the unfused path.

No further cross-chapter inconsistencies found. Navigation footer chains are complete on all spot-checked content files. All index links are clickable (relative paths, confirmed present).

## Pass 3

### Pass 2 Fix Verification

The Pass 2 fix is confirmed in place. `ch3_gdn_layer_decode_pipeline/recurrence_math.md` lines 171–186 now correctly distinguishes the two precision paths:

- Unfused path (projection matmuls): `self.compute_cfg = args.compute_kernel_config_hifi2` at `gdn.py:116` — verified correct against source.
- Fused kernel (recurrence core): HiFi4 config at `gdn_kernel_op.py:485–490` — verified correct: source reads `math_fidelity=ttnn.MathFidelity.HiFi4`, `math_approx_mode=False`, `fp32_dest_acc_en=True`.

Cross-chapter consistency on this point is now clean: `kernel_dispatch.md` (line 116) and `compute_kernel.md` (line 5) both state HiFi4 for the fused kernel, matching the fixed Ch3 text.

### Remaining Issue

**Issue 5: Code block in `recurrence_math.md` shows wrong class name and wrong field for the HiFi4 config struct**

`recurrence_math.md` lines 177–184 renders the fused kernel's compute config as:

```python
COMPUTE_HIFI4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)
```

The actual source at `gdn_kernel_op.py:485–490` is:

```python
config=ttnn.ComputeConfigDescriptor(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    dst_full_sync_en=False,
)
```

Two concrete errors:

1. **Wrong class name.** The guide shows `ttnn.WormholeComputeKernelConfig`; the source uses `ttnn.ComputeConfigDescriptor`. These are different API classes.
2. **Wrong field.** The guide shows `packer_l1_acc=True`; the source has `dst_full_sync_en=False`. The `packer_l1_acc` field does not appear in the actual config call at these lines.

Both errors appear only in the illustrative code block; the surrounding prose correctly states HiFi4 and `math_approx_mode=False`. Nevertheless the code block is cited as source-verified evidence of the fix, making accuracy there load-bearing.

**Fix:** Update the code block in `recurrence_math.md` to use `ttnn.ComputeConfigDescriptor` and replace `packer_l1_acc=True` with `dst_full_sync_en=False`, matching `gdn_kernel_op.py:485–490` exactly. Remove the variable-assignment wrapper (`COMPUTE_HIFI4 = ...`) since the source passes the config inline as a keyword argument.

No further cross-chapter factual errors, formula errors, or missing navigation footers found. All other cross-chapter references to HiFi4/HiFi2 are now internally consistent.

## Pass 4

### Pass 3 Fix Verification

The Pass 3 fix is confirmed in place. `ch3_gdn_layer_decode_pipeline/recurrence_math.md` lines 178–183 now render the fused kernel config as:

```python
config=ttnn.ComputeConfigDescriptor(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    dst_full_sync_en=False,
)
```

This matches `gdn_kernel_op.py:485–490` exactly on all four points: class name (`ttnn.ComputeConfigDescriptor`), all three shared fields (`math_fidelity`, `math_approx_mode`, `fp32_dest_acc_en`), and the corrected fourth field (`dst_full_sync_en=False`). The erroneous `ttnn.WormholeComputeKernelConfig` class name and `packer_l1_acc=True` field from Pass 3 are gone.

No feedback — guide approved.
