# B Review — Chapter 2 — Pass 1

## Item 1 — Wrong shape for GLA write term (`gated_delta_rule_formulation.md` §5)

The GLA recurrence is written as:

```
S_t  = (1 α_t^T) ⊙ S_{t-1}  +  v_t k̃_t^T    [GLA]
```

`v_t ∈ R^{d_v}` and `k̃_t ∈ R^{d_k}`, so `v_t k̃_t^T ∈ R^{d_v × d_k}`. But throughout this chapter `S ∈ R^{d_k × d_v}`. The outer product is transposed relative to the state shape, making the addition dimensionally inconsistent. The correct GLA write is `k̃_t v_t^T ∈ R^{d_k × d_v}`. A reader implementing GLA from this formula would produce a write matrix of the wrong shape.

---

## Item 2 — Wrong byte count for k̃, q̃, v read (`parallelism_and_scan.md` §3)

The decode memory table states:

```
Read  k̃, q̃, v:  (d_k + d_k + d_v) × 2 = 512 + 512 + 512 = ~1.5 KB
```

With d_k = d_v = 128 and BF16 = 2 bytes per element:

```
(128 + 128 + 128) × 2 = 768 bytes  ≈  0.75 KB
```

Each `512` in the addition implies a dimension of 256, not 128. The total should be ~0.75 KB, not ~1.5 KB. This does not affect the dominant cost conclusion (which is the 64 KB state I/O), but a reader reproducing the arithmetic intensity calculation from raw numbers would get a wrong intermediate byte figure.

---

## Item 3 — Arithmetic intensity figure is unaffected but derivation path is broken (`parallelism_and_scan.md` §3)

Following directly from Item 2: the arithmetic intensity is stated as `~0.08 MFLOPs / 64 KB ≈ 1.25 FLOPs/byte`, which uses only the dominant state I/O term (64 KB) in the denominator and is therefore numerically correct despite the wrong k̃/q̃/v byte count. This item is noted for completeness but does not produce a wrong final answer on its own.

**Withdraw — not a qualifying issue under scope (c) given the dominant term is correct. Replacing with Item 4 below.**

---

## Item 4 — Crossover upper bound "1,098 tokens" does not correspond to any calculation (`state_vs_kv_cache_memory.md` §3)

The summary sentence reads:

> "for any sequence longer than ~512–1,098 tokens, the Gated Attention KV cache is more expensive per layer than the DeltaNet recurrent state."

The three crossover values derived in the same section are 512, 544, and 1,024 tokens. No calculation produces 1,098. A reader using this range to set a design threshold would use a wrong upper bound. The correct upper bound from the section's own arithmetic is 1,024 tokens (K-only crossover).

---

## Item 5 — Scan workspace memory overstatement may mislead capacity planning (`parallelism_and_scan.md` §2.3)

The scan workspace is estimated as `~2 × (T/C)` state matrices with the parenthetical "factor 2 for workspace." A standard parallel prefix scan over N elements requires N − 1 ≈ N internal nodes, not 2N. The factor-of-2 multiplier is unexplained and inflates the stated workspace from ~125 MB to ~250 MB per head per layer. With 32 heads the chapter concludes "~8 GB of scan workspace per layer." If the factor of 2 is unjustified, the true figure is ~4 GB — still prohibitive, so the qualitative conclusion (sequential scan preferred) is unchanged, but the stated ~8 GB figure is numerically incorrect and a reader doing capacity planning against a specific DRAM budget would use a wrong number.

---

# B Review — Chapter 2 — Pass 2

## Item 1 — `g_t` range is contradicted between symbol table and §6 derivation (`gated_delta_rule_formulation.md`)

The symbol table in §2 defines `g_t ∈ (0, 1]` (closed bracket at 1), and the recurrence box repeats this. Section 6, step 4 concludes: "g_t = exp(α_t) ∈ (0, 1) for any finite input. At α_t = 0 (which cannot be reached), g_t = 1." This makes the range open at 1 — contradicting the closed bracket in the table.

A reader implementing the gate and checking boundary conditions will face a direct conflict: is `g_t = 1` a valid run-time value or not? The §6 derivation is correct (α_t < 0 strictly, so g_t < 1 strictly at all finite inputs), but the table header `(0, 1]` is wrong. An implementer who clips or saturates to `(0, 1]` instead of `(0, 1)` will write code that is technically harmless in the open-set case but conceptually wrong, and may suppress valid error checks.

---

## Item 2 — "256K" is used as 256,000 in §2.3 and as 262,144 in §4.1, producing inconsistent numbers (`parallelism_and_scan.md` §2.3 and `state_vs_kv_cache_memory.md` §4.1)

In `parallelism_and_scan.md` §2.3 the chunk-count calculation reads:

```
Number of chunks:  T / C  =  256,000 / 64  =  4,000
```

This treats "T = 256K" as 256,000 (decimal kilo). In `state_vs_kv_cache_memory.md` §4.1 the KV cache computation for "T = 256K" yields:

```
B × 2 × 256,000 × 256 × 4  =  524,288,000 bytes  ≈  524 MB
```

but the document rounds this to "≈ B × 512 MB," which is only exact when 256K = 2^18 = 262,144 (binary Ki):

```
2 × 262,144 × 256 × 4  =  536,870,912 bytes  =  512 MiB
```

A reader computing either the chunk count or the KV footprint from the given numbers and expecting consistency will get a ~2.3 % error in the byte count and a mismatched chunk count (4,000 vs. 4,096). Any downstream capacity-planning arithmetic that chains both figures will accumulate this inconsistency.

---

**No further qualifying issues found.**

---

## Agent A Change Log — Pass 1

All 4 feedback items from Agent B have been applied:

**Item 1** (`gated_delta_rule_formulation.md`, §5 GLA comparison):
- Fixed the GLA write term in the code block from `v_t k̃_t^T` to `k̃_t v_t^T` to match the correct `[d_k, d_v]` shape of S.
- Fixed the prose description in the "Key differences" list (item 2) from `v_t k̃_t^T` to `k̃_t v_t^T`.

**Item 2** (`parallelism_and_scan.md`, §3 arithmetic intensity):
- Corrected the byte count for reading k̃, q̃, v per head per step from `512 + 512 + 512 = ~1.5 KB` to `256 + 256 + 256 = 768 bytes` (each 128-element BF16 vector = 256 bytes, not 512).
- Updated the arithmetic intensity calculation from `~0.08 MFLOPs / 64 KB ≈ 1.25 FLOPs/byte` to `~82K FLOPs / (768 + 65,536) bytes ≈ 1.24 FLOPs/byte`. Conclusion (heavily memory-bandwidth-bound) is unchanged.

**Item 3** (`state_vs_kv_cache_memory.md`, §3 crossover):
- Removed the fabricated "~1,098 tokens" upper bound from the crossover summary sentence.
- Replaced with the values actually derived in the section: `~512–544 tokens` (combined KV crossover: ~512 tokens; including conv state: ~544 tokens). The K-only crossover of 1,024 tokens is now referenced inline.

**Item 4** (`parallelism_and_scan.md`, §2.3 associative scan workspace):
- Removed the unexplained `× 2` multiplier (`~2 × (T/C)` state matrices).
- Corrected the workspace calculation: 4,000 chunks × 32 KB = ~128 MB per head per layer (was ~250 MB).
- Corrected the per-layer total: 32 heads × 128 MB = 4,096 MB ≈ 4 GB (was ~8 GB). Qualitative conclusion (sequential scan preferred, parallel scan prohibitive) is unchanged.

---

## Agent A Change Log — Pass 2

All 2 feedback items from Agent B Pass 2 have been applied.

**Item 1** (`gated_delta_rule_formulation.md`, symbol table and prose — g_t interval):

Four occurrences of `g_t ∈ (0, 1]` (closed at 1) were fixed to `g_t ∈ (0, 1)` (open at 1):

1. Symbol table row (§2): `scalar ∈ (0, 1]` → `scalar ∈ (0, 1)`.
2. Recurrence code block (§2): `α_t < 0 → g_t ∈ (0, 1]` → `α_t < 0 → g_t ∈ (0, 1)`.
3. §3.1 prose: `g_t ∈ (0, 1]` → `g_t ∈ (0, 1)`; also removed the clause "when `g_t = 1` no forgetting occurs" (which is unreachable) and replaced with "when `g_t` is close to 1 forgetting is minimal" to avoid implying `g_t = 1` is attainable.
4. §6 opening sentence: `to keep g_t ∈ (0, 1]` → `to keep g_t ∈ (0, 1)`.

Rationale: §6 step 4 already established that α_t < 0 strictly at all finite inputs, so `g_t = exp(α_t) < 1` strictly. The open interval `(0, 1)` is correct; the closed bracket at 1 was a contradiction.

**Item 2** (`parallelism_and_scan.md` and `state_vs_kv_cache_memory.md` — 256K standardized to 262,144 = 2^18):

`parallelism_and_scan.md` (§2.3):
- Context length changed from `T = 256K` to `T = 262,144 (2^18)` in the scan workspace block.
- Chunk count corrected: `256,000 / 64 = 4,000` → `262,144 / 64 = 4,096`.
- Scan workspace corrected: `4,000 chunks × 32 KB = 128,000,000 bytes ≈ 128 MB` → `4,096 chunks × 32 KB = 131,072 KB = 128 MB` (exact binary: 128 MiB, the MB figure is coincidentally the same).
- Per-layer total: already correct at `~4,096 MB ≈ 4 GB` from Pass 1; prose now explicitly shows `32 × 128 MB = 4,096 MB`.
- §2.4 FLOP count at T=256K: updated label to `T = 262,144 (2^18)` and corrected figure from `~333 GFLOPs` to `~341 GFLOPs` (262,144 × 1.3 M ≈ 341 GFLOPs).

`state_vs_kv_cache_memory.md`:
- §4 heading changed from `T = 256K` to `T = 262,144 (2^18, "256K" Max Context)`.
- §4.1 calculation replaced: uses `n_kv_h=2, T=262,144, d_h=256, 2 bytes`: `2 × 262,144 × 256 × 2 = 268,435,456 bytes = 256 MB` per Gated Attention layer (was `≈ 512 MB` based on T=256,000 with combined K+V doubling).
- §4.3 ratio corrected: `512 MB / 1.125 MB ≈ 455×` → `256 MB / 1.125 MB ≈ 228×`; text updated accordingly.
- §5.2 corrected: per-layer figure `B × 512 MB` → `B × 256 MB`; 10-layer total `B × 5,120 MB = B × 5 GB` → `B × 2,560 MB ≈ B × 2.5 GB`; batch scaling examples updated (B=1: 2.5 GB, B=4: 10 GB, B=8: 20 GB).
- §5.3 DRAM budget examples updated to use `T=262,144` and the corrected 2.5 GB per-layer figure; effective batch size ceiling revised from "B=2–4" to "B=8–10".
- §7 summary table: per-layer row updated to `~256 MB`, 10-layer row to `~2.5 GB`; shape column updated from `[1,2,256K,256]×2` to `[1,2,262144,256]`.

# B Review — Chapter 2 — Pass 3

## Item 1 — §4.1 of `state_vs_kv_cache_memory.md` computes only one cache (K or V alone) but labels the result "KV bytes"

The computation in §4.1 is:

```
KV bytes per batch element:  n_kv_h × T × d_h × 2
  =  2 × 262,144 × 256 × 2
  =  268,435,456 bytes
  =  256 MB per Gated Attention layer
```

This formula — `n_kv_h × T × d_h × 2 bytes` — is the byte count for **one** cache tensor (K alone, or V alone). The combined KV is twice this: `2 × 268,435,456 = 536,870,912 bytes = 512 MB`. The label "KV bytes" and the assignment to "per Gated Attention layer" both imply K+V combined, but the arithmetic only counts one.

This causes a cascading error. §4.3 divides the DeltaNet state (1.125 MB) by 256 MB to report the ratio as ~228×. The correct combined-KV figure is 512 MB, giving a ratio of ~455×. §5.2 compounds the error: the 10-layer KV total is stated as 2.5 GB (based on 256 MB × 10), but the correct combined-KV total is 5.12 GB. The batch-size headroom conclusions in §5.3 are therefore wrong: at B=4, T=262,144, combined KV is ~20 GB, not ~10 GB; at B=8 it is ~40 GB, which would exceed the ~26 GB DRAM headroom entirely. A reader doing capacity planning from these figures would underestimate KV cache pressure by a factor of 2.

Cross-check: §3 correctly uses `B × 2,048 × T bytes` as the combined KV in the crossover formula (which equals 2 × 1,024 × T, i.e., K+V combined), confirming that §4.1 is internally inconsistent with §3.

---

## Item 2 — §2.3 of `parallelism_and_scan.md` states scan workspace as 128 MB per head but sums to 4 GB for 32 heads using an inconsistent multiplier

The workspace per head is computed as `4,096 chunks × 32 KB = 131,072 KB = 128 MB`. The per-layer total is then stated as `32 × 128 MB = 4,096 MB ≈ 4 GB`. This arithmetic is internally consistent.

However the qualitative conclusion is stated as "completely prohibitive," which is the correct takeaway. The issue is that the scan tree for N leaves requires N−1 internal nodes, not N nodes. With 4,096 chunks the internal node count is 4,095, which is immaterially different. This is not a qualifying issue.

**Withdraw — not a qualifying issue. No replacement item found.**

---

**No further qualifying issues found.**

---

## Agent A Change Log — Pass 3

**Item 1** (`state_vs_kv_cache_memory.md` — KV cache factor-of-2 fix, cascading corrections):

**§4.1** — The per-layer KV cache calculation previously computed only a single cache tensor (K alone, or V alone). The label "KV bytes" was therefore wrong by a factor of 2. The step-by-step arithmetic has been corrected to include the factor of 2 for K AND V combined:

```
Combined K+V bytes per batch element:  2 (K and V) × n_kv_h × T × d_h × 2
  =  2 × 2 × 262,144 × 256 × 2
  =  536,870,912 bytes
  =  512 MB per Gated Attention layer
```

**§4.3** — Ratio corrected from `256 MB / 1.125 MB ≈ 228×` to `512 MB / 1.125 MB ≈ 455×`. Summary sentence updated from "228 times smaller" to "455 times smaller."

**§5.2** — 10-layer KV aggregate corrected:
- Per-layer figure: `B × 256 MB` → `B × 512 MB`
- 10-layer total: `B × 2,560 MB ≈ B × 2.5 GB` → `B × 5,120 MB ≈ B × 5 GB`
- Batch scaling examples updated: B=1: 5 GB; B=4: 20 GB; B=8: 40 GB (infeasible).

**§5.3** — DRAM budget analysis revised using the corrected 5 GB per-batch-element figure:
- B=1: ~5 GB (fits within ~26 GB headroom)
- B=4: ~20 GB (tight; ~6 GB remaining for activations)
- B=8: ~40 GB (exceeds headroom; not feasible)
- Effective batch-size ceiling revised from "B = 8–10" to "B = 4–5."

**§7 summary table** — Per-layer row updated to `~512 MB`; shape column updated to `[1, 2, 262144, 256] × 2 (K+V)`; 10-layer row updated to `~5 GB`.

# B Review — Chapter 2 — Pass 4

## Item 1 — Wrong transpose placement in inter-chunk cross term formula (`parallelism_and_scan.md` §2.1, Step 3)

The formula for the inter-chunk contribution is written as:

```
O_cross = (Q̃_chunk)^T S_0^{(c)}
```

`Q̃_chunk` has shape `[C, d_k]`, so `(Q̃_chunk)^T` has shape `[d_k, C]`. `S_0^{(c)}` has shape `[d_k, d_v]`. The product `[d_k, C] × [d_k, d_v]` is dimensionally undefined (inner dimensions C vs. d_k do not match), and even if transpositions were rearranged it would not yield the required output shape `[C, d_v]`.

The correct operation is `Q̃_chunk @ S_0^{(c)}`: `[C, d_k] × [d_k, d_v] → [C, d_v]`, which computes `S_0^{(c)T} q̃_t` for each of the C query positions simultaneously. Equivalently written per-position: `o_cross_t = S_0^{(c)T} q̃_t ∈ R^{d_v}`, stacked over t into `[C, d_v]`.

A reader implementing the chunkwise kernel from this formula will apply the transpose in the wrong place and produce a runtime shape error or silently compute a wrong result if dimensions happen to be equal (d_k = C = 64 would mask the bug numerically while being conceptually wrong).

---

## Item 2 — Rounded per-layer DeltaNet total of 1.125 MB is inconsistent with the computed 1.0625 MB, producing a wrong ratio in §4.3 (`state_vs_kv_cache_memory.md` §1.3 and §4.3)

Section 1.3 derives the exact total as:

```
B × 1,114,112 bytes  ≈  B × 1,088 KB  ≈  B × 1.0625 MB
```

The next line then states: "Rounded: B × 1.125 MB per Gated Delta Net layer (using the plan's stated aggregate)." The jump from 1.0625 MB to 1.125 MB is a ~5.9% upward rounding with no arithmetic justification (1.125 MB = 1,152 KB, not 1,088 KB).

This propagates directly into §4.3:

```
KV cache / DeltaNet state  =  512 MB / 1.125 MB  ≈  455×
```

Using the correct 1.0625 MB figure: `512 / 1.0625 ≈ 482×`. A reader computing the ratio from the numbers in §1.3 will get ~482×, not ~455×, a discrepancy of ~6%. The stated 455× is numerically wrong relative to the chapter's own derived figures. The correct divisor is 1.0625 MB (or equivalently 1,088 KB = 1,114,112 bytes).

---

## Agent A Change Log — Pass 4

**Item 1** (`parallelism_and_scan.md`, §2.1 Step 3 — O_cross transpose fix):

- Changed `O_cross = (Q̃_chunk)^T S_0^{(c)}` to `O_cross = Q̃_chunk @ S_0^{(c)}`.
- The previous formula placed the transpose on `Q̃_chunk`, yielding shape `[d_k, C] × [d_k, d_v]` — dimensionally invalid. The corrected formula `Q̃_chunk @ S_0^{(c)}` gives `[C, d_k] × [d_k, d_v] = [C, d_v]`, which is the correct output shape for all C query positions.

**Item 2** (`state_vs_kv_cache_memory.md` — 1.125 MB rounding error corrected throughout):

- **§1.3**: Removed the incorrect "Rounded: B × 1.125 MB" sentence. The correctly derived value 1.0625 MB is now stated directly as the total per Gated Delta Net layer.
- **§4.2**: Removed the `≈ B × 1.125 MB (plan rounding)` secondary approximation, leaving only the correct `≈ B × 1.0625 MB`.
- **§4.3**: Corrected ratio from `512 MB / 1.125 MB ≈ 455×` to `512 MB / 1.0625 MB ≈ 482×`. Summary sentence updated from "455 times smaller" to "482 times smaller."
- **§5.1**: Removed the incorrect `≈ B × 33.75 MB (plan rounding)` secondary figure (33.75 MB = 30 × 1.125 MB, which was wrong). Replaced with `≈ B × 31.875 MB ≈ B × 31.9 MB` (30 × 1.0625 MB = 31.875 MB), consistent with the exact byte count already shown.

# B Review — Chapter 2 — Pass 5

## Item 1 — L1 per-core state figure is wrong by a factor of 4 (`state_vs_kv_cache_memory.md` §6.1)

The distributed-across-cores bullet states:

> "4 heads × 128 KB = 512 KB per core if 8 cores share the layer"

Per-head state = `d_k × d_v × 2 bytes = 128 × 128 × 2 = 32,768 bytes = 32 KB`, not 128 KB. With 32 heads across 8 cores, each core holds 4 heads: `4 × 32 KB = 128 KB`, not 512 KB. Both the per-head figure (128 KB vs. correct 32 KB) and the per-core total (512 KB vs. correct 128 KB) are off by a factor of 4. A reader using 512 KB as the per-core footprint would incorrectly conclude that head-sharding across 8 cores barely fits in 1.5 MB L1, when in fact 128 KB per core leaves ~1.37 MB free — a qualitatively different capacity picture.

---

## Item 2 — Bolded "Simplified" line in §2 states 1,024×T as the per-layer KV figure, directly contradicting the code block above it (`state_vs_kv_cache_memory.md` §2)

The code block immediately above line 72 derives:

```
Combined K+V bytes per batch element:  2 × 1,024 × T  =  2,048 × T bytes
```

The very next line then states:

> **B × 1,024 × T bytes per Gated Attention layer** (K+V combined: 2 × 1,024 × T, but plan uses 1,024 × T as the single-cache unit; combined KV = 2,048 × T).

The bolded figure 1,024×T is half the combined KV value. A reader who reads the bolded summary line as the per-layer combined KV will use a figure 2× too small in any downstream calculation. This contradiction is not resolved by the "Wait — let us be precise" block below, which correctly lands on 2,048×T but leaves the wrong bolded line intact above it.

---

## Item 3 — `Γ_C` range uses a closed bracket at 1, inconsistent with the now-corrected `g_t ∈ (0, 1)` (`parallelism_and_scan.md` §2.1)

The WY-decomposition section defines:

> `Γ_C = ∏_{t=1}^{C} g_t ∈ (0, 1]`

After Pass 2 corrections, `g_t ∈ (0, 1)` strictly (open at 1) in `gated_delta_rule_formulation.md`. Since every factor g_t is strictly less than 1 and strictly greater than 0, their product Γ_C is also strictly in (0, 1). The closed bracket `(0, 1]` is therefore wrong and inconsistent with the corrected gate range. An implementer checking boundary conditions for the chunk decay accumulation (e.g., whether to special-case Γ_C = 1) would be misled.

---

## Item 4 — GLA gate formula produces wrong shape for elementwise multiplication with S, and the descriptor "per-column (per-key-dimension)" is self-contradictory (`gated_delta_rule_formulation.md` §5)

The GLA gate is written as `G_t = 1 α_t^T` where `α_t ∈ R^{d_k}`. For this outer product to match S ∈ R^{d_k × d_v} under elementwise multiplication, `1` must be in R^{d_k}, giving `1 α_t^T ∈ R^{d_k × d_k]` — which is square and does not match the `[d_k, d_v]` shape of S when d_k ≠ d_v. Alternatively, if `1 ∈ R^{d_v}`, the outer product is `[d_v, d_k]` — the transpose of S's shape, also incompatible. Either reading makes the elementwise multiply `(1 α_t^T) ⊙ S_{t-1}` dimensionally undefined.

Additionally, the text labels the gate "per-column (i.e., per-key-dimension)." Columns of S ∈ R^{d_k × d_v} are indexed by the d_v (value) axis, not the d_k (key) axis. Per-column gating is per-value-dimension; per-row gating is per-key-dimension. The two parenthetical descriptions name opposite dimensions. A reader implementing GLA from this section would either produce a shape error or gate the wrong axis.

---

## Item 5 — Decode FLOPs denominator includes only q̃ read in the bytes row but the intensity formula implicitly uses the correct total (`parallelism_and_scan.md` §3)

The memory read/write table labels one line as:

```
Read  k̃, q̃, v:  (d_k + d_k + d_v) × 2 = 256 + 256 + 256 = 768 bytes
```

The three summands each show 256, which is `128 × 2 = 256 bytes` ✓. However the labels say `d_k + d_k + d_v` for `k̃, q̃, v` respectively. With d_k = d_v = 128, these are all equal, so the arithmetic is correct for this specific configuration. But the label uses `d_k` for the v-vector dimension, which is `d_v` in general. For any configuration where d_k ≠ d_v, a reader applying this formula template directly would compute the wrong byte count for v (substituting d_k instead of d_v). Since the Qwen3.5-35B-A3B head dimensions are d_k = d_v = 128, no numerical error occurs here, but the formula template is wrong in general and would mislead an implementer targeting a different head-dimension configuration.

**Withdraw — the numeric values are correct for the documented configuration (d_k = d_v = 128), and the scope threshold requires a wrong numerical answer or implementation error that is not merely latent. Replacing with the following:**

The FLOPs-per-head decode total is listed as "~82,000 FLOPs (~0.08 MFLOPs)." Summing the table rows: 16,384 (scale) + 16,384 (matvec retrieval) + 128 (sub+scale) + 16,384 (outer product) + 16,384 (add) + 16,384 (matvec output) = 82,048 ≈ 82,000 ✓. The per-layer total is `32 × 82,000 = 2,624,000 ≈ 2.6 MFLOPs` ✓. No error here.

**This item is withdrawn. Items 1–4 constitute the Pass 5 findings.**

## Agent A Change Log — Pass 5

All 4 feedback items from Agent B Pass 5 have been applied.

**Item 1** (`state_vs_kv_cache_memory.md`, §6.1 — L1 per-core state):
- Corrected the distributed-across-cores bullet: `4 heads × 128 KB = 512 KB per core` → `4 heads × 32 KB = 128 KB per core`.
- Per-head state is `128 × 128 × 2 = 32,768 bytes = 32 KB`; with 4 heads per core (32 heads across 8 cores) the per-core footprint is `4 × 32 KB = 128 KB`, not 512 KB. The previous figure was off by a factor of 4.

**Item 2** (`state_vs_kv_cache_memory.md`, §2 — bolded summary line):
- Changed the bolded summary from `B × 1,024 × T bytes per Gated Attention layer` to `B × 2,048 × T bytes per Gated Attention layer`.
- The code block and the "Wait — let us be precise" block both correctly derive 2,048 × T as the combined K+V figure; the bolded line now agrees with that derivation.

**Item 3** (`parallelism_and_scan.md`, §2.1 — Γ_C range):
- Changed `Γ_C = ∏_{t=1}^{C} g_t ∈ (0, 1]` to `Γ_C = ∏_{t=1}^{C} g_t ∈ (0, 1)` (open at 1).
- Since each g_t ∈ (0, 1) strictly (established in Pass 2 for `gated_delta_rule_formulation.md`), their product is also strictly less than 1. The closed bracket was inconsistent with the corrected gate range.

**Item 4** (`gated_delta_rule_formulation.md`, §5 — GLA gate description):
- Rewrote the GLA gate as `G_t = α_t 1^T` (column vector α_t ∈ R^{d_k} times all-ones row vector 1 ∈ R^{d_v}), giving `G_t ∈ R^{d_k × d_v}` — the correct shape for elementwise multiplication with S ∈ R^{d_k × d_v}.
- Updated the code block from `(1 α_t^T) ⊙ S_{t-1}` to `(α_t 1^T) ⊙ S_{t-1}`.
- Changed the gating description from "per-column (i.e., per-key-dimension)" to "row-wise (per key-dimension) — row i of S scaled by α_t[i]", which correctly identifies that rows of S are indexed by d_k and that α_t[i] scales row i.
- Clarified that α_t ∈ R^{d_k} is a data-dependent column vector from the input projection.

# B Review — Chapter 2 — Pass 6

## Item 1 — Cross-chunk contribution formula omits per-position intra-chunk decay, producing wrong output values (`parallelism_and_scan.md` §2.1, Step 3)

Step 3 of the chunkwise computation states:

```
O_cross = Q̃_chunk @ S_0^{(c)}    [C, d_k] × [d_k, d_v] → [C, d_v]
```

This formula multiplies the raw chunk-initial state `S_0^{(c)}` against all C query vectors without applying the cumulative intra-chunk decay that `S_0^{(c)}` must undergo by the time each query position retrieves from it. The correct contribution to output token at intra-chunk offset τ ∈ {1,...,C} is:

```
o_cross[τ]  =  (∏_{t=1}^{τ} g_t) · S_0^{(c)T} q̃_τ
```

That is, `S_0^{(c)}` must be scaled by the cumulative gate prefix product up to position τ before being queried by `q̃_τ`. Equivalently, each row τ of `O_cross` must be multiplied by the corresponding scalar `Γ_{0..τ}`. The formula `Q̃_chunk @ S_0^{(c)}` treats every query as seeing the fully undecayed carry-in state, which is only correct if all intra-chunk `g_t = 1` (never true given `g_t ∈ (0, 1)` strictly).

A reader implementing the chunkwise kernel directly from this formula will compute `O_cross` without the per-row decay prefix scaling, producing systematically wrong output token values for every position except where the cumulative decay happens to be negligible. The intra-chunk attention term `O_intra` already bakes the decay products into the causal attention mask, so the omission is asymmetric: the within-chunk path is handled correctly but the carry-in state path is not. The corrected Step 3 requires a per-row elementwise scale: multiply row τ of `Q̃_chunk @ S_0^{(c)}` by `∏_{t=1}^{τ} g_t` before adding to `O_intra`.

---

No further qualifying issues found.

---

## Agent A Change Log — Pass 6

**Item 1** (`parallelism_and_scan.md`, §2.1 Step 3 — O_cross diagonal prefix-product scaling):

- Replaced the single-line formula `O_cross = Q̃_chunk @ S_0^{(c)}` with the corrected formulation that applies per-position intra-chunk decay.
- Added the per-position scalar form: `o_cross[τ] = Γ_τ · S_0^{(c)T} q̃_τ` where `Γ_τ = ∏_{t=1}^{τ} g_t`.
- Added the matrix form: `O_cross = D · (Q̃_chunk @ S_0^{(c)})` where `D ∈ R^{C×C}` is a diagonal matrix with `D[τ,τ] = Γ_τ`.
- Explained what D represents (diagonal matrix of intra-chunk prefix products) and why it is necessary (without D, every query position incorrectly retrieves from the fully undecayed carry-in state, over-weighting the cross-chunk contribution at every position τ > 1).
- The previous formula was correct only in the degenerate case where all g_t = 1, which cannot occur since g_t ∈ (0, 1) strictly.

# B Review — Chapter 2 — Pass 7

No feedback — chapter approved.

# B Review — Chapter 2 — Pass 8

No feedback — chapter approved.
