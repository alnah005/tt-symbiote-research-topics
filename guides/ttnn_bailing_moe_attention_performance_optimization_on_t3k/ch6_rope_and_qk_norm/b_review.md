# Agent B Review — Chapter 6 — Pass 1

## Issue 1 — Broken code: undefined variable `k_norm_in_raw` in `qk_norm_latency.md` Step 3

**File:** `qk_norm_latency.md`, Step 2 (line ~44) and Step 3 (line ~61).

Step 2 assigns the K DRAM→L1 result to the variable `q_norm_in_raw_k`:

```python
q_norm_in_raw_k = ttnn.to_memory_config(k_post_rope, norm_l1_config_q)
```

Step 3 then references `k_norm_in_raw` in the reshape call:

```python
k_norm_2d = ttnn.reshape(k_norm_in_raw, (4, 128))
```

`k_norm_in_raw` is never assigned anywhere. The code as written would raise a `NameError` at runtime. The Step 2 assignment must use `k_norm_in_raw` (not `q_norm_in_raw_k`) to match the Step 3 reference.

---

## Issue 2 — Arithmetic error: Q FLOPs count in `qk_norm_latency.md` Step 4

**File:** `qk_norm_latency.md`, Step 4 (line ~93).

The text states:

> "16 × (128 + 1 + 128) = 4,128 floating-point operations for Q"

16 × (128 + 1 + 128) = 16 × 257 = **4,112**, not 4,128. The K figure (4 × 257 = 1,028) is computed correctly in the same sentence, confirming the per-head factor of 257 is intended. The Q result is off by 16.

---

# Agent B Review — Chapter 6 — Pass 2

## Issue 1 — Wrong shape in code comment: `cos_table`/`sin_table` in `partial_rotary_rope.md`

**File:** `partial_rotary_rope.md`, `_build_rope_tables` implementation sketch, lines ~263–264 (the comment inside the function).

The code computes `cos_full` and `sin_full` by concatenating partial cos/sin of width `rotary_dim // 2 = 32` with `pad_cols = head_dim // 2 - rotary_dim // 2 = 32` padding columns, giving shape `(max_seq_len, 64)` = `(max_seq_len, head_dim // 2)`. After `unsqueeze(0).unsqueeze(0)`, the resulting `cos_table` and `sin_table` have shape `(1, 1, max_seq_len, head_dim // 2)` = `(1, 1, max_seq_len, 64)`.

The inline comment immediately above the `ttnn.from_torch` call states:

> `# Interleave format expected by distributed kernel: (1, 1, max_seq_len, head_dim)`

This is wrong. The actual shape is `(1, 1, max_seq_len, head_dim // 2)`, not `(1, 1, max_seq_len, head_dim)`. A reader using this comment as a reference to understand the tensor passed to `TTNNDistributedRotaryPositionEmbedding` will have an incorrect mental model of the table shape. If the distributed kernel actually requires width `head_dim` (not `head_dim // 2`), the sketch as written also has a structural error beyond the comment.

---

## Issue 2 — Memory size formula doubles the true value in `partial_rotary_rope.md`

**File:** `partial_rotary_rope.md`, "Memory Cost of Padding" section, lines ~167–171.

The formula states:

```
Padded table size: S_max × head_dim × 2 bytes (BF16) × 2 tables (cos + sin)
                 = S_max × 128 × 2 × 2 = S_max × 512 bytes
```

For S_max = 4096: 4096 × 512 = **2 MB**.

But `cos_full` (and `sin_full`) each have shape `(max_seq_len, head_dim // 2)` = `(S_max, 64)`, not `(S_max, 128)`. The actual size per table is `S_max × 64 × 2 bytes`, and both tables together are `S_max × 64 × 2 × 2 = S_max × 256 bytes`. For S_max = 4096: 4096 × 256 = **1 MB**, not 2 MB.

The formula uses `head_dim = 128` where it should use `head_dim // 2 = 64`. The "2× growth vs. original" ratio is still correct (32 → 64 columns), but the absolute size stated is 2× too large. The S_max = 4096 example figure of "2 MB" should be "1 MB".

# Agent B Review — Chapter 6 — Pass 3

## Issue 1 — Claimed mathematical equivalence of padded cos/sin strategy is unverified and likely incorrect

**File:** `partial_rotary_rope.md`, "The Padding Strategy" section (lines ~115–127) and "Implementation Requirements" section.

The chapter claims that padding the cos/sin tables with identity values (`cos=1.0`, `sin=0.0`) for indices `[rotary_dim : head_dim]` and then invoking `TTNNDistributedRotaryPositionEmbedding` on the full `head_dim=128` is "mathematically equivalent" to partial-rotary with `rotary_dim=64`. This claim depends critically on both kernels using the same element-pairing convention — and the chapter never establishes that they do.

The non-distributed kernel (`TTNNRotaryPositionEmbedding`) applies rotation with offset `rotary_dim/2 = 32`, pairing elements `(q[i], q[i+32])` for `i in [0, 32)`. This covers elements `q[0:64]` and leaves `q[64:128]` unchanged.

The distributed kernel (`TTNNDistributedRotaryPositionEmbedding`) is designed for full `head_dim=128` coverage and uses offset `H/2 = 64`, pairing elements `(q[i], q[i+64])` for `i in [0, 64)`. With the padded tables, the pairs processed are:

- `i in [0, 32)`: real rotation values at `cos[i]`, `sin[i]` applied to pair `(q[i], q[i+64])`. This rotates `q[0:32]` together with `q[64:96]` — the pass-through region.
- `i in [32, 64)`: identity values (`cos=1`, `sin=0`) applied to pair `(q[i], q[i+64])`, i.e., pair `(q[32:64], q[96:128])`. These produce identity output — no rotation.

The result is that real rotation is applied to pairs `(q[0:32], q[64:96])`, not to the pairs `(q[0:32], q[32:64])` that the non-distributed kernel processes. Elements `q[32:64]` are left unchanged by the distributed path but are actively involved in the rotation by the non-distributed path. The two paths produce different embeddings for the same input.

The equivalence proof shown in the chapter (lines ~122–126) correctly verifies the identity-pad formula but implicitly assumes the same `i + rotary_dim/2` offset structure for both kernels. If `TTNNDistributedRotaryPositionEmbedding` uses offset `H/2=64` (as required for full-head operation), the equivalence does not hold. The chapter must either (a) verify that the distributed kernel can be configured with offset `rotary_dim/2=32` even when `head_dim=128`, or (b) correct the claim by replacing "mathematically equivalent" with "equivalent only if both kernels use the same pairing offset."

---

## Issue 2 — `_apply_rope` decode-mode cos/sin slice is a no-op and passes wrong table width

**File:** `partial_rotary_rope.md`, `_apply_rope` implementation sketch (lines ~292–306).

In `decode` mode, the sketch slices the cos table as `cos_slice[:, :, :, :self.rotary_dim]`. `cos_slice` has shape `(1, 1, seq_len, head_dim//2)` = `(1, 1, seq_len, 64)`. `self.rotary_dim = 64`. The slice `[:64]` on a dimension of size 64 is a no-op — it returns all 64 columns unchanged, including the 32 identity-padding columns at positions `[32:64]`.

The non-distributed kernel `TTNNRotaryPositionEmbedding` with `rotary_dim=64` expects a cos/sin table of width `rotary_dim//2 = 32` (one value per pair), not 64. Passing a 64-column table to a kernel expecting 32 will either silently use only the first 32 columns (discarding the padding), use all 64 and produce wrong rotation, or raise an error — none of which match the intended behavior. The correct slice should be `cos_slice[:, :, :, :self.rotary_dim // 2]` (i.e., `[:32]`), which extracts only the 32 real rotation values for the 32 pairs processed by the non-distributed kernel.

This error is directly related to Issue 1: both stem from the chapter conflating `rotary_dim` (number of elements that receive rotation = 64) with `rotary_dim // 2` (number of cos/sin values needed = 32, one per complex pair).

# Agent B Review — Chapter 6 — Pass 4

## Issue 1 — Options A+C combined saving (96 µs) exceeds the stated total norm overhead (74–92 µs)

**File:** `qk_norm_latency.md`, end of "Option C" section (line ~274).

The text states:

> "The combined saving of Options A+C is ≈ 96 µs [ESTIMATE] per step, which would recover over 60% of the total estimated transition overhead."

The 96 µs figure is derived by summing Option A saving (T2a + T2b + T_norm_in = 32 + 32 = 64 µs) and Option C saving (T_norm_out = 32 µs). However, T2a and T2b are post-RoPE DRAM evictions from the RoPE kernel path — they are not part of the QK norm path costs tabulated in this file. The "Complete QK norm overhead" table in this same file totals 74–92 µs and does not include T2a/T2b (those are charged to the RoPE path in Chapter 4). Claiming a 96 µs saving against a 74–92 µs process implies a saving that exceeds 100% of the stated cost, which is self-contradictory.

Additionally, if the "total estimated transition overhead" refers to the 74–92 µs norm table, then 96/74 > 100% — not "over 60%". If it refers to a combined RoPE+norm total not defined in this file, that total is never stated, making "60%" unverifiable.

**Required correction:** Either (a) clarify that the 96 µs saving spans both the QK norm path costs (T_norm_in + T_norm_out = 64 µs) and the separately-accounted RoPE-path eviction costs (T2a + T2b = 32 µs), and state the combined RoPE+norm overhead against which the percentage is computed; or (b) recompute the saving to cover only the norm-path components (T_norm_in + T_norm_out = 64 µs, not 96 µs) and revise the percentage accordingly.

---

## Issue 2 — Latency total inconsistency between the two tables in `qk_norm_latency.md`

**File:** `qk_norm_latency.md`, "Component Costs" table (~line 163) and "Complete QK norm overhead" table (~line 200).

The first table (Component Costs) gives a combined total of **≈ 70–80 µs**. The second table (Complete QK norm overhead) sums the same path to **≈ 74–92 µs**. These ranges diverge: the lower bound shifts from 70 to 74, and the upper bound shifts from 80 to 92. The difference is the reshape dispatch overhead (≈ 4–12 µs) included in the second table but absent from the first. The first table's row for reshape calls says "< 1 µs" per call, which appears to exclude the Python dispatch overhead that the second table accounts for separately.

A reader using the first table to size the optimization opportunity will understate the upper bound by 12 µs (80 vs. 92). The two tables should either agree on a single range or the first table should explicitly state it excludes reshape dispatch and reference the second table for the complete figure.

---

## Issue 3 — Wrong shape comment on `cos_slice` in `_apply_rope` sketch

**File:** `partial_rotary_rope.md`, `_apply_rope` implementation sketch (~line 296).

The line:

```python
cos_slice = self.cos_cached[:, :, position_ids, :]   # (1, 1, seq_len, head_dim)
```

has an incorrect comment. `self.cos_cached` was constructed with shape `(1, 1, max_seq_len, head_dim // 2)` = `(1, 1, max_seq_len, 64)`. The slice `[:, :, position_ids, :]` preserves all dimensions and produces shape `(1, 1, seq_len, 64)` = `(1, 1, seq_len, head_dim // 2)`, not `(1, 1, seq_len, head_dim)`. The same incorrect comment appears on the `sin_slice` line.

A developer implementing the prefill path from this sketch will construct or expect a `head_dim`-wide table (128 columns) where only `head_dim // 2` (64 columns) are present. If the distributed kernel requires a 128-column table, the sketch will fail; if it requires 64 columns, the comment misleads about what is being passed. The comment should read `# (1, 1, seq_len, head_dim // 2)` on both lines.

# Agent B Review — Chapter 6 — Pass 5

No feedback — chapter approved.
