# Agent B Review — Chapter 6: Weight Precision — Pass 1

## Item 1 — Wrong: `in_proj_all` fused output dimension formula (`dtype_choices.md`, line 60)

**Claimed:**
> the fused `in_proj_all` has output dimension $2 \times K\_{dim} + V\_{dim} + H\_v + H\_v = 2 \times 2048 + 6144 + 48 + 48 = 10336$

**Actual:**
`gated_deltanet.py` lines 64–68 concatenate four separate projections:

```python
w_all = torch.cat(
    [load_weight("in_proj_qkv"), load_weight("in_proj_z"), load_weight("in_proj_b"), load_weight("in_proj_a")],
    dim=-1,
)
self._proj_splits = [self.conv_dim, self.value_dim, self.num_v_heads, self.num_v_heads]
```

`in_proj_qkv` maps to `conv_dim = key_dim * 2 + value_dim = 2048*2 + 6144 = 10240`, not to `2 × K_dim = 4096`. The guide strips out the `value_dim` component of `conv_dim` and accounts for it separately as if `in_proj_z` handles it, but `in_proj_z` produces a second, independent `value_dim = 6144` projection (the gate `z`). The correct total is:

```
conv_dim + value_dim + num_v_heads + num_v_heads
= 10240  + 6144      + 48          + 48
= 16480
```

The guide's formula `2 × K_dim + V_dim + H_v + H_v = 10336` is wrong by a factor; it undersizes the fused weight by 6144 output rows (it omits that `in_proj_qkv` includes the `value_dim` span inside `conv_dim`).

---

## Item 2 — Wrong: A3B `q_proj` shape example (`dtype_choices.md`, line 59; repeated in `hf_to_meta_conversion.md`, line 59)

**Claimed:**
> `[16 * 256 * 2, 2048] = [8192, 2048]` for A3B

**Actual:**
A3B has `hidden_size = 2048` and `n_heads = 16` (confirmed in `README.md` and `test_a3b_pcc.py`). A `head_dim` of 256 would give `n_heads × head_dim = 4096`, which is larger than `hidden_size = 2048`. That is architecturally impossible: the `q_proj` output dimension `n_heads × head_dim` cannot exceed `hidden_size` in a standard square or rectangular projection without an explicit expansion design.

The correct `head_dim` for A3B is 128 (`= hidden_size / n_heads = 2048 / 16`), so:

```
q_proj shape = [16 * 128 * 2, 2048] = [4096, 2048]
```

The same wrong shape `[8192, 2048]` is copied verbatim into `hf_to_meta_conversion.md` (Step 2, shape example table). Both occurrences must be corrected.

---

## Item 3 — Wrong: `split_hf_keys` split position (`moe_key_protection.md`, lines 26–27)

**Claimed:**
> attempts to split `[256, 1024, 2048]` at `dim=0` position 512 (half of `intermediate`)

**Two errors in this sentence:**

1. Dimension 0 of the tensor is 256 (the expert batch axis). Position 512 in a dimension of size 256 is out of bounds. The actual split attempt by `split_hf_keys` targets the output-feature dimension, which for a standard 2D `gate_up_proj [2*I, H]` would be dim 0. For the 3D expert tensor, if `split_hf_keys` treats dim 0 naively as the output-feature axis, the split position it would compute is `intermediate = 512` rows — which is exactly dimension 1's half (1024 / 2 = 512), not dimension 0.

2. Even granting the intent (split along the fused-output axis), "position 512 (half of intermediate)" is only correct if the split uses `intermediate = 512`; `half of intermediate = 256`. The guide says "position 512" but calls it "half of intermediate" in the same clause. With `MOE_INTERMEDIATE = 512` (from `test_a3b_pcc.py` line 35), the split position for a standard 2D tensor would be 512 (full intermediate, not half), because `gate_up_proj` fuses `[intermediate, hidden]` + `[intermediate, hidden]` so the split point is at row `intermediate = 512`, not `intermediate/2 = 256`. The parenthetical "half of intermediate" is numerically incorrect; the correct label is "at row `intermediate`" or equivalently "at row 512 = half of the fused first dimension 1024".

The net effect: the two-sentence description conflates the axis being split (should be dim 1, not dim 0) and mislabels the split value. An implementer reading this could derive the wrong conclusion about how `split_hf_keys` fails.

---

No further correctness issues found. Items 1 and 2 are definitive wrong numbers that would cause an incorrect implementation (wrong weight-loading code or wrong size assumptions). Item 3 is a wrong axis and wrong label in the failure-mode description.

---

# Agent B Review — Chapter 6: Weight Precision — Pass 2

Pass 1 Items 1 and 2 are confirmed fixed in the current chapter files. Pass 1 Item 3 was revised but the fix introduced a new wrong value — see Item 1 below.

## Item 1 — Still Wrong: `split_hf_keys` split axis and chunk size (`moe_key_protection.md`, line 27)

**Claimed (current text):**
> attempts to split `[256, 1024, 2048]` at `dim=1` position 512 (equal to `intermediate_size = 512`)

**Actual (source: `load_checkpoints.py` line 400):**
```python
gate_tensor, up_tensor = torch.split(tensor, tensor.shape[0] // 2, dim=0)
```

The split is along **dim=0**, not dim=1. The chunk size is `tensor.shape[0] // 2 = 256 // 2 = 128`, not 512. The call produces two tensors of shape `[128, 1024, 2048]`, splitting the expert-batch dimension in half — not the fused-projection dimension. The current text corrected the Pass 1 error (which said dim=0 was right for a standard 2D tensor) but swung to the wrong answer: dim=1 is not what the code uses.

Correct description: `split_hf_keys` calls `torch.split(tensor, 128, dim=0)` on the `[256, 1024, 2048]` tensor, yielding two `[128, 1024, 2048]` tensors — slicing the expert batch axis, not the projection axis.

---

No additional correctness issues found in `dtype_choices.md` or `hf_to_meta_conversion.md` for Pass 2. All numeric values in DRAM tables, expert shapes, router weight shapes, and recurrent state dimensions are consistent with source.

---

# Agent B Review — Chapter 6: Weight Precision — Pass 3

Pass 2 item is confirmed fixed in the current `moe_key_protection.md`. One new correctness issue found in `dtype_choices.md`.

## Item 1 — Wrong: Recurrent state example shape and DRAM cost (`dtype_choices.md`, lines 127–136)

**Claimed:**

```python
self._dev_state  # shape: [batch_size, H, head_k_dim, head_v_dim]
                 #        e.g. [32, 32, 128, 128] for A3B
```

> The fp32 state does not carry a large DRAM cost: at A3B dimensions $[32, 32, 128, 128]$ the state is only $32 \times 32 \times 128 \times 128 \times 4 = 67$ MB per DeltaNet layer.

**Actual (`gated_deltanet.py`, lines 132–153):**

```python
def initialize_states(self, batch_size=1, B_pad=32):
    ...
    self._dev_state = ttnn.from_torch(
        torch.zeros(batch_size, H, self.head_k_dim, D),
        dtype=ttnn.float32, ...
    )
```

The default `batch_size` argument is `1`, not `32`. `B_pad=32` is a separate tile-padding parameter used for conv state rows and is not the batch dimension of `_dev_state`. The canonical A3B recurrent state shape with default arguments is `[1, 32, 128, 128]`, not `[32, 32, 128, 128]`.

The DRAM cost therefore follows from the correct shape: `1 × 32 × 128 × 128 × 4 = 2,097,152 bytes ≈ 2 MB` per DeltaNet layer, not 67 MB. The guide overstates the per-layer state footprint by a factor of 32.

---

No further correctness issues found. `moe_key_protection.md` and `hf_to_meta_conversion.md` are consistent with source for this pass.

---

# Agent B Review — Chapter 6: Weight Precision — Pass 4

Pass 3 item is confirmed fixed in the current `dtype_choices.md` (recurrent state shape is now `[1, 32, 128, 128]` and DRAM cost is `2 MB`). One new correctness issue found in `dtype_choices.md`.

## Item 1 — Wrong: `WormholeComputeKernelConfig` parameter name (`dtype_choices.md`, lines 145–151)

**Claimed:**

```python
WormholeComputeKernelConfig(
    math_fidelity=MathFidelity.HiFi2,
    fp32_dest_acc=False,
    packer_l1_acc=True,
)
```

**Actual (`gated_deltanet.py`, lines 40–45):**

```python
self.proj_compute_config = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)
```

The parameter name is `fp32_dest_acc_en`, not `fp32_dest_acc`. These are different keyword argument names. Code copied verbatim from the guide would either raise a `TypeError` (unexpected keyword argument) or pass the flag silently under the wrong name depending on the TTNN version, leaving `fp32_dest_acc_en` at its default value. The guide also omits `math_approx_mode=False`, which is present in the source.

---

No further correctness issues found. All numeric values in DRAM tables, expert shapes, split logic, and recurrent state dimensions are consistent with source for this pass.

---

# Agent B Review — Chapter 6: Weight Precision — Pass 5

Pass 4 item is confirmed fixed in the current `dtype_choices.md` (`fp32_dest_acc_en` and `math_approx_mode=False` are both present and correct). One new correctness issue found.

## Item 1 — Wrong: Expert weight DRAM estimates at bfp8 and bfp4 (`dtype_choices.md`, line 31)

**Claimed:**
> At bfp8, 256 experts across 40 layers would require approximately 25.6 GB for expert weights alone [...] Halving to bfp4 brings expert weights to 12.8 GB

**Actual:**

The two expert tensors per layer have the following element counts:

- `gate_up_proj` `[256, 1024, 2048]`: `256 × 1024 × 2048 = 536,870,912` elements
- `down_proj` `[256, 2048, 512]`: `256 × 2048 × 512 = 268,435,456` elements
- Combined per layer: `805,306,368` elements

At bfp8 (1 byte/element) across 40 layers:

```
805,306,368 × 40 = 32,212,254,720 bytes ≈ 32.2 GB
```

At bfp4 (0.5 byte/element) across 40 layers:

```
805,306,368 × 40 × 0.5 = 16,106,127,360 bytes ≈ 16.1 GB
```

The guide's figures of 25.6 GB (bfp8) and 12.8 GB (bfp4) are each approximately 20% too low. The 12.8 GB figure is exactly half of 25.6 GB, confirming the author applied the bfp4 halving correctly relative to their bfp8 figure — but the bfp8 base is wrong. The correct pair is ~32.2 GB (bfp8) and ~16.1 GB (bfp4).

The qualitative conclusion (expert weights at bfp8 are prohibitive; bfp4 is necessary) remains valid, but the specific numbers cited for this justification are wrong and would mislead any implementer computing DRAM budgets.

---

No further correctness issues found. All other numeric values in DRAM tables, expert shapes, router shapes, recurrent state dimensions, compute kernel config, and conversion pipeline logic are consistent with source for this pass.

---

# Agent B Review — Chapter 6: Weight Precision — Pass 6

Pass 5 item is confirmed partially addressed: the prose on line 31 was updated to `~30.0 GiB` (bfp8) and `~15.0 GiB` (bfp4), which are numerically correct in GiB units. However, the corresponding table entry was not updated and retains the old wrong value.

## Item 1 — Wrong: Expert weight bfp4 table entry inconsistent with prose (`dtype_choices.md`, line 113)

**Claimed (table):**
> Expert weights (256 × gate+up+down, 40 layers) | bfp4 | 12.8 GB

**Claimed (prose, line 31):**
> Halving to bfp4 brings expert weights to ~15.0 GiB

**Actual:**

The two expert tensors per layer:

- `gate_up_proj` `[256, 1024, 2048]`: `256 × 1024 × 2048 = 536,870,912` elements
- `down_proj` `[256, 2048, 512]`: `256 × 2048 × 512 = 268,435,456` elements
- Total per layer: `805,306,368` elements

At bfp4 (0.5 byte/element) across 40 layers:

```
805,306,368 × 40 × 0.5 = 16,106,127,360 bytes
  = 16.1 GB  (using 1 GB = 10^9 bytes)
  = 15.0 GiB (using 1 GiB = 2^30 bytes)
```

The prose (`~15.0 GiB`) is correct. The table entry (`12.8 GB`) is the stale value from before the Pass 5 correction was applied to the prose. The two figures are inconsistent: `15.0 GiB ≈ 16.1 GB`, not `12.8 GB`. An implementer reading only the table would underestimate bfp4 expert DRAM by ~3.3 GB (~21%), which affects DRAM budget planning.

The table entry must be updated to `~16.1 GB` (or `~15.0 GiB`) to match the corrected prose and the arithmetic.

---

No further correctness issues found. All other numeric values, shapes, formulas, kernel config parameters, and conversion pipeline logic are consistent with source for this pass.
