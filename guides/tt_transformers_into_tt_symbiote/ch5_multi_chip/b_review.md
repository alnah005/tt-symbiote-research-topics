# Agent B Review — Chapter 5: Multi-Chip Parallelism — Pass 1

## Issues

### Issue 1 — `TTNNLinearIColShardedWRowSharded` weight sharding described as column-sharded instead of row-sharded (Severity: Medium)

**File:** symbiote_distributed_primitives.md line 13
**What guide said:** "the weight is column-sharded (split along the output dimension, i.e. the last dim of the pre-transposed weight, which becomes `dim=-2` after preprocessing)"
**What source says:** The class is named `IColShardedWRowSharded` — "W**Row**Sharded". `weight_dim=-2` is applied via `ttnn.shard_tensor_to_mesh_mapper(self.device, dim=self.weight_dim)` on the preprocessed weight tensor. `preprocess_linear_weight` transposes the original `[out, in]` weight to `[in, out]`, so `dim=-2` of the stored tensor is the `in` (input/row) dimension — i.e., the weight is sharded along its **input dimension** (row-parallel). Source: `linear.py` lines 101, 113, 126-130.
**Fix:** Change "weight is column-sharded (split along the output dimension)" to "weight is row-sharded (split along the input dimension, i.e. `dim=-2` of the transposed `[in, out]` weight)".

---

## Verdict

Approved with fixes.

## Change Log — Pass 1 fixes applied

- **symbiote_distributed_primitives.md line 13:** Corrected `TTNNLinearIColShardedWRowSharded` weight sharding description from "weight is column-sharded (split along the output dimension, i.e. the last dim of the pre-transposed weight, which becomes `dim=-2` after preprocessing)" to "weight is row-sharded (split along the input dimension, i.e. `dim=-2` of the transposed `[in, out]` weight)". The class name `WRowSharded` and the `weight_dim=-2` applied to a `[in, out]` preprocessed weight both confirm row (input-dimension) sharding.

---

# Agent B Review — Chapter 5: Multi-Chip Parallelism — Pass 2

## Pass 1 Fix Verification

The Pass 1 fix is correctly applied. `symbiote_distributed_primitives.md` line 13 now reads: "the weight is row-sharded (split along the input dimension, i.e. `dim=-2` of the transposed `[in, out]` weight)", which matches the class name `WRowSharded` and the `weight_dim=-2` applied to the transposed `[in, out]` weight tensor. Confirmed against `linear.py` lines 126-130.

## New issues found

### Issue 1 — Residual contradictory dimension label in `move_weights_to_device_impl` description (Severity: Medium)

**File:** symbiote_distributed_primitives.md line 34
**What guide said:** "The weight is sharded along `dim=-2` (the row dimension of the transposed weight, i.e. the output dimension of the linear). The bias, if present, is sharded along `dim=-1` (input dimension)."
**What source says:** After `preprocess_linear_weight` transposes the original `[out, in]` PyTorch weight to `[in, out]`, `dim=-2` of the stored tensor is the `in` (input) dimension and `dim=-1` is the `out` (output) dimension. The parenthetical "output dimension of the linear" is the inverse of what `dim=-2` represents, and the bias label "input dimension" for `dim=-1` is also inverted. Source: `linear.py` lines 107-123; the same reasoning that drove the Pass 1 fix at line 13 applies here identically.
**Fix:** Change "i.e. the output dimension of the linear" to "i.e. the input dimension of the linear" and change the bias label from "(input dimension)" to "(output dimension)".

## Verdict

Issues found — Pass 2 fix applied; no further pass required (single residual error corrected).

## Change Log — Pass 2 fixes applied

- **symbiote_distributed_primitives.md line 34:** Corrected the dimension labels in the `move_weights_to_device_impl` prose. "the output dimension of the linear" changed to "the input dimension of the linear" (weight `dim=-2` of the transposed `[in, out]` tensor is the input/row dimension). Bias label corrected from "(input dimension)" to "(output dimension)" (`dim=-1` of the transposed weight is the output/column dimension). This is consistent with the Pass 1 fix at line 13 and with the class name `WRowSharded`.

---

# Agent B Review — Chapter 5: Multi-Chip Parallelism — Post-Compression Review

## Issues found

### Issue 1 — Forward-reference sentence in Path B overstates what Section 6 covers (Severity: Medium)

**File:** `tt_transformers_parallelism.md`, Path B section (was line 125 prior to fix)
**What the guide said:** "For a full enumeration of all Galaxy-specific call sites (shard dim swaps, intermediate MLP CCL, `slice_mat` batch selection, decode gather axis, and KV cache differences), see Section 6 below."
**What is actually true:** Section 6 ("All-Reduce Placement in the MLP Forward Pass") covers only the MLP CCL call sites — the intermediate reduce-scatter/all-reduce on w1 and w3, and the all-gather on w2_in, plus the final w2_out all-reduce. It does not cover shard dim swaps (documented in Section 4 of the same file), `slice_mat` batch selection (referenced in `index.md`), decode gather axis, or KV cache differences. Claiming Section 6 provides "a full enumeration of all Galaxy-specific call sites" including those items is factually wrong. The sentence was added by the compression pass as a forward-reference replacement for the removed Galaxy topology table.
**Verification:** `tt_transformers_parallelism.md` Section 6 (lines 198–213); shard dim swaps are at lines 147–155 (Section 4); `slice_mat` is described in `index.md` line 36 with reference to `attention.py` lines 64-88, 526-533, not Section 6.
**Fix applied:** Replaced the sentence with: "For a full enumeration of the Galaxy-specific MLP CCL call sites (intermediate reduce-scatter, all-reduce, and all-gather around the element-wise multiply), see Section 6 below. Shard dim swaps are documented directly in Section 4."

## Additional checks (no issues found)

- **`index.md` remaining content:** After removal of the "Why Multi-Chip Parallelism Matters" section, the Overview, Supported Topologies, and Chapter Navigation sections are internally coherent. No broken links or dangling references. Topology table values (N300: 1x2, T3K: 1x8, TG: 4x8; link counts 1/1 and 4/3) verified against `ccl.py` `link_dict` lines 38-44. Galaxy-specific claims in the TG section (dimension swaps, intermediate CCL, `slice_mat`) verified against `mlp.py` lines 79-80, 181-266, and `attention.py` references.
- **`symbiote_distributed_primitives.md` trailing sentence removal:** The `@run_on_devices` section (Section 7, lines 218-247) reads coherently after the removal. The concluding sentence "No TG / Galaxy variant exists yet in Symbiote." still follows naturally from the list of T3K-only decorators and is accurate per `linear.py`.
- **Galaxy/TG topology claims in remaining content:** All numeric claims verified. `link_dict` in `ccl.py` confirms TG/BHGLX: axis-0 = 4 links, axis-1 = 3 links. `mlp.py` line 79 confirms `w1_dims = (-1, -2)` for Galaxy. `mlp.py` line 302 confirms `use_composite = True` when `self.dim == 8192`. `mlp.py` line 295 confirms `dim=0` for TG when `self.dim < 8192`. All match the guide's claims in `index.md` and `tt_transformers_parallelism.md`.
- **`integration_strategy.md`:** Not touched by any compression edit; verified unchanged and internally consistent.

## Verdict

Issues found — 1 fix applied.

## Change Log

- **`tt_transformers_parallelism.md` Path B forward-reference sentence:** Narrowed the over-broad claim "full enumeration of all Galaxy-specific call sites (shard dim swaps, intermediate MLP CCL, `slice_mat` batch selection, decode gather axis, and KV cache differences)" to accurately describe what Section 6 actually contains: "the Galaxy-specific MLP CCL call sites (intermediate reduce-scatter, all-reduce, and all-gather around the element-wise multiply)." Added a clarification that shard dim swaps are in Section 4.
