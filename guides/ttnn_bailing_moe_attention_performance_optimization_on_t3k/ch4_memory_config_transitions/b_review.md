# Agent B Review — Chapter 4 — Pass 1

## Item 1 — Factual shape error: `v_raw` tile-padded shape is wrong in `transition_cost_model.md`

**File:** `transition_cost_model.md`, Tensor Size Reference table (line 69)

The table row for `v_raw` / `v_update_in` states the tile-padded shape as `(1, 4, 32, 512)`. This is inconsistent with the element count (16,384) stated in the same row: `1 × 4 × 32 × 512 = 65,536`, not 16,384. The correct tile-padded shape after the V slice (which extracts 512 columns from a `(1, 1, 32, 3072)` source) is `(1, 1, 32, 512)`, which gives `1 × 1 × 32 × 512 = 16,384` elements and 32,768 BF16 bytes. The shape `(1, 4, 32, 512)` conflates V's element count with Q's four-times-larger shape.

The element count, byte count, and all cost arithmetic derived from them are correct; only the shape label is wrong.

---

## Item 2 — Same shape error propagated to `decode_tensor_lifecycle.md` transition table

**File:** `decode_tensor_lifecycle.md`, Complete Transition Map table (line 282)

The T3b row lists the V source shape as `DRAM/ITVL (1, 4, 32, 512) → reshaped`. This repeats the same error from Item 1. `v_raw` exits the slice as `(1, 1, 32, 512)` (as correctly shown in Stage 1 of the same file, line 61), not `(1, 4, 32, 512)`. The reshape to per-head view produces `(1, 4, 32, 128)`, which is the correct destination shape, but the source shape label in this table row is wrong.

---

## Item 3 — Coherence error in `optimization_opportunities.md` Strategy A code-change annotation

**File:** `optimization_opportunities.md`, T_norm section, code location block (line 149)

The change annotation reads:

```
Change (Strategy A): align the V-slice memory config to norm input requirements.
```

Strategy A is defined two paragraphs earlier as producing Q in the 2D L1 format required by `TTNNRMSNorm` at the point of the initial QKV split. V is never passed through QK norm; this annotation should reference the Q-slice, not the V-slice. As written, a reader following this instruction would modify the wrong slice call and leave the intended optimisation unmade.

---

# Agent B Review — Chapter 4 — Pass 2

## Item 1 — Critical structural gap: V reshape missing before T3b in `decode_tensor_lifecycle.md`

**File:** `decode_tensor_lifecycle.md`, Stage 6, code block (lines 203–208)

The Stage 6 code block applies `ttnn.to_memory_config(v_raw, kv_update_mem)` directly on `v_raw`. However `v_raw` carries shape `(1,1,32,512)` (one head dimension, 512-column flat layout), whereas `kv_update_mem` specifies shard shape `(32,128)` on 4 cores — a layout that corresponds to a `(1,4,32,128)` per-head view. `ttnn.to_memory_config` is a pure data-movement call; it does not change tensor shape. A reshape from `(1,1,32,512)` to `(1,4,32,128)` is a prerequisite for the T3b transition to be valid.

The summary table row for T3b (line 282) acknowledges this with the footnote `→ reshaped`, but that acknowledgement appears only in the table, not in the stage body where the code lives. A practitioner copying the Stage 6 code block verbatim would produce an incorrect or runtime-failing call. The lifecycle file must show an explicit `v_heads = ttnn.reshape(v_raw, (1, N_kv, 1, H))` step before the `to_memory_config` call, matching the pattern already shown for Q and K in Stage 2.

---

## Item 2 — Consequent factual error in `transition_cost_model.md`: no-re-tiling claim is not universally true for T3b

**File:** `transition_cost_model.md`, Re-Tiling Cost section (line 57)

The file states: "For the transitions in the Ling decode path, the shard shape `(32,128)` is already tile-aligned (32 = TILE_SIZE, 128 = 4×TILE_SIZE), so re-tiling is not required and adds no cost." This claim is presented as applying to all transitions in the chapter.

The claim is only true if `v_raw` has already been reshaped to `(1,4,32,128)` before T3b. In the current lifecycle description (Item 1 above), that reshape is absent. Without it, the source tensor's tile layout at `(1,1,32,512)` — where the innermost dimension spans 512/32 = 16 tile-columns — does not match the destination shard shape of `(32,128)` (4 tile-columns per shard). A re-tiling step would be required by the transition kernel in that case, contradicting the no-re-tiling assertion and invalidating T3b's cost estimate (which uses the plain bandwidth formula with no re-tiling term). The cost model's claim should be conditioned on the reshape being present, or the lifecycle must be fixed so it is unambiguously present.

---

## Item 3 — Coherence gap in `optimization_opportunities.md` T3b optimisation description

**File:** `optimization_opportunities.md`, T3a/T3b section (lines 88–97)

The T3b optimisation proposes:

```python
v_update_in = ttnn.slice(qkv_replicated, ..., memory_config=kv_update_mem)
```

`kv_update_mem` specifies shard shape `(32,128)` on 4 cores. A slice from `qkv_replicated` — which has a flat 3072-column layout — produces a tensor with shape `(1,1,32,512)`, not `(1,4,32,128)`. A single `ttnn.slice` call with `memory_config=kv_update_mem` cannot simultaneously change the tensor shape and place it in the correct shard layout; an intermediate reshape remains necessary. The optimisation description omits this constraint and presents the change as a straightforward one-liner, which it is not. A reader implementing Priority 4 (line 211) from this description would produce a misshapen shard assignment.

---

# Agent B Review — Chapter 4 — Pass 3

## Item 1 — Factual error: V shape stated as `(1, 4, 32, 512)` in `decode_tensor_lifecycle.md` Stage 4

**File:** `decode_tensor_lifecycle.md`, Stage 4 prose (line 152)

The sentence reads:

> "V is not passed through RoPE. It remains in `DRAM/ITVL` at `(1, 4, 32, 512)` from Stage 2 (or a pre-split form)."

Stage 2 (Reshape for Per-Head View) reshapes only Q and K. V is never reshaped in that stage — it exits the Stage 1 slice as `(1, 1, 32, 512)` and that shape is unchanged until the explicit `ttnn.reshape` added before T3b in Stage 6. The shape `(1, 4, 32, 512)` does not correspond to any valid state of V in this lifecycle; it would imply 1×4×32×512 = 65,536 elements, which contradicts V's established 32 KB / 16,384-element size. The correct shape at this point in the lifecycle is `(1, 1, 32, 512)`.

---

## Item 2 — Arithmetic inconsistency in `optimization_opportunities.md`: combined saving stated as ~65 µs but computed as 62 µs

**File:** `optimization_opportunities.md`, "Combined potential saving" block (lines 233–236)

The code block on line 233 computes:

```
T1a + T1b + T2b + T3a + T3b(partial) = 21 + 11 + 11 + 11 + 8 = 62 µs
```

The immediately following prose (line 236) states: "Approximately **65 µs** can be recovered per decode step." The two figures are inconsistent — 62 µs ≠ 65 µs, and there is no stated rounding or adjustment that accounts for the 3 µs discrepancy. A reader using the prose figure for budgeting would overstate the saving by ~5%. The prose should read "approximately **62 µs**" to match the arithmetic above it.

---

## Item 3 — Erroneous `×4` notation in `transition_cost_model.md` tensor size table

**File:** `transition_cost_model.md`, Tensor Size Reference table (line 71)

The V row reads:

```
`(1, 1, 32, 512)` → reshaped `(1,4,32,128)×4`
```

The `×4` suffix on `(1,4,32,128)` is unexplained and incorrect. There is no operation that produces a tensor four times the size of `(1,4,32,128)`. The element count in the same row (16,384) equals exactly 1×4×32×128, confirming the reshaped shape is `(1,4,32,128)` without any multiplier. The `×4` appears to be a copy-paste artefact and would mislead a reader into thinking the V tensor is four times larger than it is, which would also contradict the stated byte count of 32,768.

# Agent B Review — Chapter 4 — Pass 4

## Item 1 — Arithmetic error: combined saving stated as ~129 µs but computable total is 126 µs

**File:** `optimization_opportunities.md`, combined saving paragraph (line 236)

The paragraph states:

> "Combined with Priority 1 (if `TTNNRMSNorm` can accept HEIGHT_SHARDED input), the total recoverable overhead approaches **129 µs** [ESTIMATE] out of the estimated **150–171 µs** total transition cost."

The document's own figures do not support 129 µs. Priority 1 is explicitly estimated at "approximately 50–64 µs" (line 187), with the upper bound of 64 µs matching T_norm exactly. Priorities 2–4 combined are computed as 62 µs in the immediately preceding code block (line 233). Taking the highest stated value for Priority 1 (64 µs) and adding Priorities 2–4 (62 µs) gives 64 + 62 = **126 µs**, not 129 µs. There is no stated rounding adjustment, partial-overlap term, or additional saving that bridges the 3 µs gap. A reader using 129 µs to estimate residual transition overhead after all optimisations would understate the remaining cost by 3 µs per decode step (approximately 3 ms per 1000 tokens). The prose should read "approaches **126 µs**" to be consistent with the arithmetic presented in the same section.

# Agent B Review — Chapter 4 — Pass 5

## Item 1 — Factual error: `__init__` core-grid pattern exceeds T3K column count for Q

**File:** `decode_tensor_lifecycle.md`, Source Code Locations section (lines 313–323)

The `__init__` pattern for `rope_shard_mem_q` uses:

```python
ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.num_heads - 1, 0))
```

With `num_heads=16`, this evaluates to `CoreCoord(0, 0)` to `CoreCoord(15, 0)` — a single-row range spanning 16 columns. The T3K Wormhole chip has an 8-column × 10-row Tensix grid (columns 0–7, rows 0–9), as stated in Chapter 1 and consistent with `index.md` line 17 ("80 Tensix cores arranged in a grid"). Column index 15 does not exist; this core range is out of bounds and would cause a runtime failure for any model with more than 8 Q heads, including Ling's `num_heads=16`.

A valid 16-core grid must span at least two rows, for example:

```python
# Option A: two consecutive rows of 8 cores each
ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 1))   # 8×2 = 16 cores
# Option B: explicit CoreRangeSet of two single-row ranges
ttnn.CoreRangeSet({
    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0)),  # 8 cores
    ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(7, 1)),  # 8 cores
})
```

The 4-core K variant (`CoreCoord(0,0)` to `CoreCoord(3,0)`) is within bounds and is not affected by this error.

# Agent B Review — Chapter 4 — Pass 6

## Item 1 — Factual arithmetic error: T2b double-counted between Priority 1 and Priority 3 inflates combined saving from 115 µs to 126 µs

**File:** `optimization_opportunities.md`, "Combined potential saving" block (lines 230–236)

The combined saving of **126 µs** is computed as Priority 1 (64 µs) + Priorities 2–4 (62 µs). However, T2b (K L1→DRAM eviction, ~11 µs) is included in both groups:

- **Priority 1** (lines 187–193) explicitly lists T2b as one of the eliminated transitions: "eliminating T2a, T2b, T_norm_in for both Q and K."
- **Priority 3** (lines 211–213) also claims T2b as part of its ~22 µs saving: "K can be passed directly from the RoPE output to `paged_update_on_device` without evicting to DRAM (T2b) and reloading (T3a)."
- The Priorities 2–4 code block (line 233) includes T2b explicitly: `T1a + T1b + T2b + T3a + T3b(partial) = 21 + 11 + 11 + 11 + 8 = 62 µs`.

T2b(11 µs) is therefore counted once in the Priority 1 subtotal and again in the Priorities 2–4 subtotal. The non-overlapping total is 64 + (62 − 11) = **115 µs**, not 126 µs. The stated 126 µs overstates the combined recoverable overhead by 11 µs per decode step — approximately 11 ms per 1,000 generated tokens. The prose on line 236 should read "approaches **115 µs**" and the Priorities 2–4 code block should remove T2b from its sum (since Priority 1 already claims it), giving `T1a + T1b + T3a + T3b(partial) = 21 + 11 + 11 + 8 = 51 µs`.

---

## Item 2 — Factual coherence error: Stage 5 in `decode_tensor_lifecycle.md` names the wrong driver for T2a

**File:** `decode_tensor_lifecycle.md`, Stage 5 introduction (lines 173–174)

The opening sentence of Stage 5 states:

> "After RoPE, the Q tensor needs to be available in a format suitable for `paged_sdpa_decode`, and K must be re-formatted for `paged_update_on_device`."

This attributes the T2a eviction of Q (L1→DRAM) to the requirements of `paged_sdpa_decode`. However, Q does not flow directly from T2a into `paged_sdpa_decode`. When `use_qk_norm=True` (which is the stated configuration for Ling throughout this chapter), Q passes through `TTNNRMSNorm` between T2a and SDPA. The Note on line 192 of the same file acknowledges this: "When `use_qk_norm=True`... additional transitions occur around `TTNNRMSNorm`... treated as occurring between T2 and T3." The immediate downstream consumer driving T2a is therefore `TTNNRMSNorm`, not `paged_sdpa_decode`.

This misidentification has a direct consequence for optimization reasoning: `optimization_opportunities.md` (lines 68–69) correctly notes that T2a cannot be eliminated if `TTNNRMSNorm` requires DRAM INTERLEAVED input — but a reader following the Stage 5 narrative would look for SDPA's Q input requirements when trying to eliminate T2a, not the norm kernel's input requirements. The wrong source of constraint is identified, which would lead an engineer to the wrong code location. The Stage 5 intro should state that Q must be evicted to DRAM because `TTNNRMSNorm` (when `use_qk_norm=True`) requires DRAM INTERLEAVED input, with SDPA as a secondary consumer after the norm path.

---

# Agent B Review — Chapter 4 — Pass 7

## Item 1 — Factual arithmetic error: T3b saving stated as ~8 µs but should be ~11 µs given reshape is zero-copy

**File:** `optimization_opportunities.md`, T3b section (line 100) and Priority 4 description (lines 215–217)

The document justifies a saving of only ~8 µs for eliminating T3b by arguing that the reshape "incurs a small metadata operation cost" that absorbs the 3.3 µs transfer component of T3b's 11 µs total:

> "The estimated saving therefore reflects the removal of the `to_memory_config` kernel dispatch overhead only (~8 µs [ESTIMATE]), not the full T3b cost (~11 µs), since the reshape itself incurs a small metadata operation cost."

However, the document defines reshape as a **zero-copy metadata operation** throughout. `decode_tensor_lifecycle.md` Stage 2 (line 80) states explicitly: "Reshape in TTNN is a zero-copy metadata operation when the source tensor is contiguous and the new shape is compatible with the existing tile boundaries." The same zero-copy condition holds for the V reshape from `(1,1,32,512)` to `(1,4,32,128)` — the element count is unchanged and tile boundaries are compatible.

A zero-copy reshape has no data-transfer cost. It does not absorb the 3.3 µs bandwidth component of T3b. Eliminating the `to_memory_config` call saves the full ~11 µs (8 µs dispatch overhead + 3.3 µs transfer), not just 8 µs. The 3.3 µs transfer component is not redirected to the reshape — it disappears entirely when the explicit copy is removed.

**Downstream arithmetic error:** If the T3b saving is ~11 µs (not ~8 µs), the Priorities 2–4 combined saving is `21 + 11 + 11 + 11 = 54 µs`, not 51 µs (line 233). The total recoverable overhead with Priority 1 becomes `64 + 54 = 118 µs`, not 115 µs (line 238). Both figures need correction.

---

# Agent B Review — Chapter 4 — Pass 8

## Item 1 — Factual error: proposed Priority 4 code applies reshape to a HEIGHT_SHARDED L1 tensor, invalidating the zero-copy claim

**File:** `optimization_opportunities.md`, T3b / Priority 4 section (lines 220–227)

The proposed optimisation for Priority 4 is:

```python
v_reshaped = ttnn.reshape(
    ttnn.slice(qkv_replicated, ..., memory_config=kv_update_mem),
    (1, 4, 32, 128)
)
```

The inner `ttnn.slice(..., memory_config=kv_update_mem)` produces a tensor placed directly into `kv_update_mem` — a HEIGHT_SHARDED L1 buffer with shard shape `(32, 128)` on 4 cores. The outer `ttnn.reshape(…, (1, 4, 32, 128))` is then applied to that HEIGHT_SHARDED L1 tensor.

The document's zero-copy reshape guarantee (stated in `decode_tensor_lifecycle.md` Stage 2 and repeated at `optimization_opportunities.md` line 217) applies to contiguous DRAM INTERLEAVED tensors where the shape change is compatible with existing tile boundaries. A HEIGHT_SHARDED L1 tensor is not contiguous in the same sense: each shard occupies a separate L1 bank on a separate core. Reshaping from the slice output shape `(1, 1, 32, 512)` — one logical tile block of 512 columns — to `(1, 4, 32, 128)` changes both the logical shape and the required shard assignment (from one shard to four shards across four cores). This cannot be performed as a zero-copy metadata update; it requires data movement to redistribute tiles across four separate L1 banks.

The correct order to preserve the zero-copy property is to keep the slice output in DRAM INTERLEAVED format, apply the reshape while the tensor is in contiguous DRAM, and then call `to_memory_config` once to move the correctly shaped tensor into the HEIGHT_SHARDED L1 layout:

```python
# Correct order: reshape first (zero-copy, contiguous DRAM), then single DRAM→L1 copy
v_reshaped = ttnn.reshape(ttnn.slice(qkv_replicated, ...), (1, 4, 32, 128))
v_update_in = ttnn.to_memory_config(v_reshaped, kv_update_mem)
```

The proposed Priority 4 code inverts this order. As written, it replaces one explicit `to_memory_config` call with a reshape applied to a HEIGHT_SHARDED tensor — an operation that itself requires data movement and is not zero-copy. The `to_memory_config` elimination and the ~11 µs saving estimate are not valid for the sequence as written.

# Agent B Review — Chapter 4 — Pass 9

## Item 1 — Coherence error: T3b summary table row contradicts Priority 4 prose on eliminability and saving

**Files:** `optimization_opportunities.md`, Summary Table row for T3b (line 177) vs. Priority 4 section (lines 215–242)

The T3b row in the Summary Table states:

> "Partially eliminable — V slice can target `kv_update_mem` directly, but reshape from `(1,1,32,512)` to `(1,4,32,128)` remains required. Strategy: change V slice output `memory_config`. Estimated saving: ~11 µs."

Priority 4 (the authoritative prose for T3b) explicitly refutes this strategy. Lines 225–229 state:

> "passing `memory_config=kv_update_mem` directly to `ttnn.slice` would produce a HEIGHT_SHARDED L1 tensor immediately after the slice. Calling `ttnn.reshape` on that sharded tensor is not a zero-copy operation… That sequence would therefore not be functionally equivalent."

Priority 4 then defines the optimisation as eliminating a *redundant intermediate* `to_memory_config` call (a DRAM staging step) that existed in the baseline between the reshape and the final shard copy. The final `to_memory_config` to `kv_update_mem` — which is the T3b transition as catalogued in `decode_tensor_lifecycle.md` — is **explicitly not eliminated**: "the final `to_memory_config` to `kv_update_mem` still executes — it is not eliminated" (line 227).

This produces three separate contradictions:

1. **Eliminability:** The table says T3b (the DRAM→L1 shard copy to `kv_update_mem`) is partially eliminable; Priority 4 says it is not eliminated.
2. **Strategy:** The table says the strategy is "change V slice output `memory_config`"; Priority 4 says that strategy is incorrect because it makes the reshape non-zero-copy.
3. **Combined saving arithmetic:** Line 247 labels the fourth term in `T1a + T1b + T3a + T3b = 21 + 11 + 11 + 11 = 54 µs` as "T3b". But Priority 4 says T3b is not eliminated; what Priority 4 removes is the intermediate redundant staging call (~8–11 µs). Labeling that saving as "T3b" is incorrect — T3b still executes. The label should identify the intermediate staging call, and the saving figure should reflect the ~8–11 µs range cited in Priority 4, not T3b's full ~11 µs from the cost model.

A practitioner reading the summary table to select an optimisation strategy would attempt to pass `memory_config=kv_update_mem` to the V slice — precisely the sequence Priority 4 warns against. The table must be updated to reflect Priority 4's correct description: T3b is not eliminated; only a redundant intermediate call is removed; and the strategy is to remove that intermediate call, not to redirect the slice output.

# Agent B Review — Chapter 4 — Pass 10

No feedback — chapter approved.

# Agent B Review — Chapter 4 — Pass 11

## Item 1 — Critical coherence block: T3a/T3b section contradicts Priority 4 on strategy, eliminability, zero-copy property, and saving amount

**File:** `optimization_opportunities.md`, T3a/T3b section (lines 84–109) vs. Priority 4 section (lines 215–242)

The T3a/T3b section proposes the following optimisation for V:

```python
v_slice = ttnn.slice(qkv_replicated, ..., memory_config=kv_update_mem)
v_update_in = ttnn.reshape(v_slice, (1, 4, 32, 128))
# The explicit to_memory_config (T3b) is eliminated, but the reshape remains.
```

This code passes `memory_config=kv_update_mem` to `ttnn.slice`, producing a HEIGHT_SHARDED L1 tensor, and then calls `ttnn.reshape` on that HEIGHT_SHARDED L1 result. Priority 4 (lines 225–229) explicitly identifies this exact sequence as incorrect:

> "passing `memory_config=kv_update_mem` directly to `ttnn.slice` would produce a HEIGHT_SHARDED L1 tensor immediately after the slice. Calling `ttnn.reshape` on that sharded tensor is not a zero-copy operation… That sequence would therefore not be functionally equivalent."

Priority 4 then provides the correct three-step sequence — slice to DRAM INTERLEAVED, reshape on contiguous DRAM (zero-copy), single `to_memory_config` to L1 — and explicitly states that T3b (the final `to_memory_config` to `kv_update_mem`) is **not** eliminated by Priority 4.

The T3a/T3b section contradicts Priority 4 on four independent points:

1. **Strategy:** The section says the strategy is to pass `memory_config=kv_update_mem` to `ttnn.slice`; Priority 4 says that strategy is wrong.
2. **Zero-copy claim:** Line 100 asserts the reshape is zero-copy in this sequence; Priority 4 says it is not zero-copy when applied to a HEIGHT_SHARDED L1 tensor.
3. **Eliminability:** Line 97 says "The explicit `to_memory_config` (T3b) is eliminated"; Priority 4 says T3b still executes.
4. **Saving amount:** Line 100 states "Potential saving for T3b: ~11 µs"; Priority 4's saving is ~8–11 µs for the intermediate staging call only — T3b's own cost is not recovered.

The "Code location" block at lines 103–109 compounds the error by instructing the reader to "add `memory_config=self.kv_update_mem_v` to the V slice" — the precise change Priority 4 warns against. A practitioner following the T3a/T3b section would produce incorrect code and would not achieve the saving claimed. The T3a/T3b section must be revised to match Priority 4's analysis.

# Agent B Review — Chapter 4 — Pass 12

## Item 1 — Factual error: T1a/T1b and Priority 2 code omits mandatory reshape, making the proposed `rope_shard_mem_q` slice incompatible with Q's flat post-slice shape

**Files:** `optimization_opportunities.md`, T1a/T1b section (lines 19–27) and Priority 2 section (lines 208–216)

The "Current pattern" code in both locations shows `q_raw` (output of `ttnn.slice`) being passed directly to `ttnn.to_memory_config(..., rope_shard_mem_q)`. The "Proposed pattern" replaces that pair with `ttnn.slice(..., memory_config=rope_shard_mem_q)`.

Neither version shows the reshape that the lifecycle file explicitly includes between the slice and the `to_memory_config` call. From `decode_tensor_lifecycle.md` Stage 2 (lines 71–80), `q_raw` exits the slice as `(1, 1, 32, 2048)` — a flat layout with only 32 tile-rows — and is reshaped to `(1, 16, 32, 128)` (per-head view, 512 tile-rows) before `to_memory_config` is called.

`rope_shard_mem_q` specifies HEIGHT_SHARDED with shard shape `(32, 128)` on 16 cores. HEIGHT_SHARDED distributes contiguous row groups across cores: 16 cores × 32 rows per shard = 512 total rows required. The flat slice output `(1, 1, 32, 2048)` provides only 32 rows. The shard spec is geometrically incompatible with the pre-reshape shape; the transition (or equivalent direct slice) cannot succeed without the reshape being applied first.

The "Proposed pattern" in both sections therefore produces an invalid `ttnn.slice` call whose output shape is incompatible with `rope_shard_mem_q`. A practitioner implementing Priority 2 from this code would get a runtime shape-mismatch error. The correct sequence — regardless of whether the slice or a subsequent `to_memory_config` is used — requires the reshape to `(1, 16, 32, 128)` before the HEIGHT_SHARDED config can be applied. The T1a/T1b "Current pattern" and "Before" code in Priority 2 should reflect the reshape step that is already documented in the lifecycle, and the optimization should be described in terms of (slice → reshape → target L1 layout) versus (slice → reshape → DRAM → to_memory_config).

# Agent B Review — Chapter 4 — Pass 13

## Item 1 — Structural defect: duplicate item number "2" in "Non-Eliminable Transitions" list

**File:** `optimization_opportunities.md`, "Non-Eliminable Transitions" section (lines 281–287)

The list numbers its entries: **1**, **2**, **2**, **3** — the T4 item and the `T_norm_out` item both carry the number 2. The list therefore has four entries but only counts to three. A reader cross-referencing the list by index would conflate T_norm_out and T4 into a single slot and could overlook T4 as a distinct non-eliminable transition when auditing residual cost. The T4 item should be numbered 3 and the `paged_update_on_device` internal DRAM item should be numbered 4.

---

## Item 2 — Coherence gap: Summary Table T2b row omits the "subsumed by Priority 1" caveat, enabling re-introduction of double-counting

**File:** `optimization_opportunities.md`, Summary Table row for T2b (line 191)

The T2b row shows "~11 µs" as an independent estimated saving with strategy "Reuse `k_rope_out` directly for paged update." No note in the table cell indicates that this ~11 µs is already included within Priority 1's 64 µs figure. The Priority 3 prose (line 236) explicitly states that T2b is not counted again there precisely to avoid double-counting, but the Summary Table — which a practitioner is most likely to consult as a quick reference — carries no corresponding flag. A reader summing the "Estimated saving" column across rows T2b and T_norm would add T2b's 11 µs on top of Priority 1's total and re-introduce the double-counting that Pass 6 corrected in the Priority 3 prose. The T2b row should note that its saving is subsumed within Priority 1 and should not be added independently.

# Agent B Review — Chapter 4 — Pass 14

## Item 1 — Arithmetic error: combined recoverable overhead lower bound stated as 83 µs but correct lower bound is 69 µs

**File:** `optimization_opportunities.md`, final combined saving paragraph (line 275)

The paragraph states:

> "Combined with Priority 1 (if `TTNNRMSNorm` can accept HEIGHT_SHARDED input), the total recoverable overhead approaches **83–86 µs** [ESTIMATE] out of the estimated **150–171 µs** total transition cost."

The Priority 1 saving is stated as "approximately 50–64 µs" (line 203). The Priorities 3–4 combined saving is stated as "~19–22 µs" (line 270). Proper interval addition pairs lower bounds with lower bounds and upper bounds with upper bounds:

```
Lower bound: 50 + 19 = 69 µs
Upper bound: 64 + 22 = 86 µs
Correct range: 69–86 µs
```

The stated lower bound of 83 µs is produced by adding the Priority 1 **upper** bound (64) to the Priority 3–4 **lower** bound (19): 64 + 19 = 83. This mixes bounds from the two ranges, which is arithmetically invalid and overstates the lower end of the achievable saving by 14 µs. A practitioner using 83 µs as the pessimistic case would underestimate residual transition cost by 14 µs per decode step (approximately 14 ms per 1,000 tokens). The range should read "**69–86 µs**".

# Agent B Review — Chapter 4 — Pass 16

No feedback — chapter approved.

# Agent B Review — Chapter 4 — Pass 15

## Item 1 — Factual inconsistency: Priority 1 body states "50–64 µs" saving but the heading, the enumerated eliminated transitions, and the arithmetic all indicate exactly 64 µs with no basis for a lower bound of 50 µs

**File:** `optimization_opportunities.md`, Priority 1 section (lines 201 and 203)

The Priority 1 heading reads:

> "**Priority 1 — Investigate and remove QK norm round-trips (~64 µs potential saving):**"

The body of the same paragraph states:

> "If yes, Q and K can remain in L1 from the RoPE output through the norm computation, eliminating T2a, T2b, T_norm_in for both Q and K, and saving approximately **50–64 µs** [ESTIMATE]."

The code comment immediately below (line 209) confirms the eliminated set: "T2a, T2b, T_norm_in (Q), T_norm_in (K) all eliminated." From the cost model: T2a = 21 µs, T2b = 11 µs, T_norm_in_Q = 21 µs, T_norm_in_K = 11 µs. The sum is exactly **64 µs** with no stated partial-application scenario that would yield a lower value. No mechanism for a 50 µs outcome is described anywhere in the paragraph or in the surrounding text.

The 50 µs lower bound is therefore unsupported and inconsistent with both the heading (~64 µs) and the enumerated transitions (which sum to 64 µs in all cases). The body should read "saving approximately **64 µs**" to match the heading and the arithmetic. As written, the 50 µs figure propagates into the combined 69–86 µs range (where 69 = 50 + 19), making the combined lower bound 14 µs lower than is derivable from the document's own cost model. A practitioner using the pessimistic combined figure of 69 µs would underestimate achievable savings by 14 µs per decode step (approximately 14 ms per 1,000 generated tokens).

# Agent B Review — Chapter 4 — Pass 17

## Item 1 — Coherence gap: Priority 1 header label does not match the transitions it actually subsumes

**File:** `optimization_opportunities.md`, Priority 1 header (line 201) and summary table T2b row (line 191)

The Priority 1 heading reads "Investigate and remove QK norm round-trips (~64 µs potential saving)". The 64 µs figure is correct per the cost model. However, the transitions that compose the 64 µs are T2a (Q L1→DRAM post-RoPE, 21 µs) + T2b (K L1→DRAM post-RoPE, 11 µs) + T_norm_in_Q (DRAM→L1 for norm, 21 µs) + T_norm_in_K (DRAM→L1 for norm, 11 µs) = 64 µs. T2a and T2b are RoPE evictions, not norm transitions. They are subsumed into Priority 1 only because Strategy B (fusing RoPE and norm in L1) eliminates the need to evict from L1 before norm — a consequence, not a direct norm round-trip.

The summary table row for T2b (line 191) reinforces this confusion by labelling T2b as "included in Priority 1" without explaining the dependency. A reader counting Priority 1's scope from the heading alone — "QK norm round-trips" — would identify only T_norm_in_Q + T_norm_in_K = 32 µs, not 64 µs, and would be unable to reproduce the 64 µs figure.

The 83–86 µs combined saving in the final paragraph is arithmetically correct (64 + 19–22 = 83–86) but rests on this conflated grouping. The inaccuracy is in the label, not the arithmetic, but a reader using Priority 1 as a code-change checklist would miss T2a and T2b unless they read the prose carefully.

**Correction needed:** Either retitle Priority 1 to reflect its full scope (e.g., "Remove RoPE evictions and QK norm reloads via L1-resident fused path"), or add a sentence listing the four subsumed transitions (T2a, T2b, T_norm_in_Q, T_norm_in_K) immediately below the heading so the 64 µs is traceable without reading through the T_norm section.

# Agent B Review — Chapter 4 — Pass 18

No feedback — chapter approved.

# Agent B Review — Chapter 4 — Pass 19

## Item 1 — Critical coherence block: T_norm row in cost model bundles T_norm_out into 64 µs, but Priority 1 eliminates T2a+T2b+T_norm_in (not T_norm_out), and Non-Eliminable item 2 says T_norm_out cannot be removed

**Files:** `transition_cost_model.md`, T_norm row in both tables; `optimization_opportunities.md`, Priority 1 section (lines 201–213) and Non-Eliminable Transitions item 2

The T_norm row in both the Per-Transition Estimates table and the Aggregated Latency table carries the label "QK norm transitions (Q+K, in+out)" and the note "bundles four DRAM↔L1 moves (Q-in, Q-out, K-in, K-out)" for a total of 64 µs. Breaking down by the cost model's own per-transition figures: T_norm_in_Q(21 µs) + T_norm_out_Q(21 µs) + T_norm_in_K(11 µs) + T_norm_out_K(11 µs) = 64 µs.

Priority 1 in `optimization_opportunities.md` claims a saving of "approximately **64 µs**" by "eliminating T2a, T2b, T_norm_in for both Q and K." The four transitions it names are T2a(21 µs) + T2b(11 µs) + T_norm_in_Q(21 µs) + T_norm_in_K(11 µs) = 64 µs. T_norm_out_Q and T_norm_out_K are **not** listed as eliminated by Priority 1.

Non-Eliminable Transitions item 2 confirms this: "T_norm_out (Q and K, L1→DRAM after norm): As long as the norm kernel writes its output to DRAM INTERLEAVED… Removing this requires modifying TTNNRMSNorm to support a configurable output memory config." T_norm_out is explicitly not eliminated.

These three claims are mutually inconsistent on what composes the 64 µs figure:

1. The cost model says T_norm = in+out = 64 µs (T_norm_out is inside this 64 µs).
2. Priority 1 saves 64 µs from T2a+T2b+T_norm_in — a different set of four transitions whose 64 µs sum coincidentally matches (T2a replaces T_norm_out_Q, T2b replaces T_norm_out_K in the arithmetic).
3. Non-Eliminable item 2 says T_norm_out remains.

The 64 µs numerical coincidence masks the inconsistency. If the cost model's T_norm row includes T_norm_out (21+11 = 32 µs), then T_norm_out must appear somewhere in the total per-step cost. It does not appear as a separate row. After Priority 1 is applied, T_norm_out_Q (~21 µs) and T_norm_out_K (~11 µs) are still incurred per step — 32 µs of uneliminated cost that is simultaneously included in the T_norm "in+out" row and excluded from Priority 1's savings. The total remaining transition cost after all optimisations is therefore understated by approximately 32 µs wherever it is derived from the post-optimisation saving figures.

**Correction needed:** Either (a) split the T_norm row in both cost-model tables into T_norm_in and T_norm_out sub-rows (so that the in-transitions and out-transitions are separately tracked and it is clear which are eliminated by Priority 1 and which remain), or (b) redefine the T_norm row to cover only T_norm_in (Q+K = 32 µs) and add T_norm_out as a separate row (~32 µs), and correct Priority 1's saving to 21+11+21+11 = 64 µs but sourced from T2a+T2b+T_norm_in_Q+T_norm_in_K with T_norm_out explicitly shown as residual. Either way the total per-step cost and the post-optimisation residual must be made consistent with what Non-Eliminable item 2 states about T_norm_out.

# Agent B Review — Chapter 4 — Pass 20

No feedback — chapter approved.
