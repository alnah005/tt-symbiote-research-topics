# B Review — Chapter 2: Attention Optimizations — Pass 1

## Issues Found

**1. `flash_attention_prefill.md`, line 88 — Wrong claim about Q_high masking**

Wrong claim: "Q_high has many valid KV blocks but most are masked out at the tile level."

Why wrong: For a late-token Q chunk (Q_high, near the bottom of the attention matrix), the KV blocks covering earlier sequence positions are fully below the causal diagonal and are all valid. Only the small number of diagonal blocks (where the causal boundary passes through the tile) and the even smaller number of future-facing blocks above the diagonal are masked. The vast majority of KV blocks for Q_high are valid, not masked. The parenthetical claim inverts the true geometry: Q_low (early tokens) is the chunk with many masked KV blocks and few valid ones; Q_high is the chunk with many valid KV blocks and few masked ones.

Fix: Change to "Q_high has many valid KV blocks and only a few masked (above-diagonal) blocks."

---

**2. `flash_attention_prefill.md`, line 112 — Contradictory attribution of the 1.6× speedup**

Wrong claim: "Combined with the load-balancing pairing, the effective improvement over naive assignment is the source of the ~1.6× figure (the two effects partially overlap)."

Why wrong: The same file, lines 90–91, correctly states: "This pairing achieves approximately ~1.6× speedup over naive sequential Q chunk assignment across the core grid. The gain is purely from scheduling, not from algorithmic changes." That attribution is correct — the 1.6× is the load-balancing pairing gain. Line 112 then contradicts this by re-attributing the 1.6× to a combination of block-skipping (which independently provides ~2× compute reduction by halving processed KV blocks) and pairing, with the two "partially overlapping" to produce 1.6×. This is internally inconsistent: if block-skipping already achieves 2× compute reduction and pairing adds a further scheduling gain, the combined effect should exceed 2×, not produce 1.6×. The 1.6× is exclusively the load-balancing figure. Block-skipping is a logically separate optimization reported to give 2× compute reduction on its own.

Fix: Remove the final sentence of the "Sparse Causal Mask" section ("Combined with the load-balancing pairing…"). The 1.6× figure belongs to the pairing section only; block-skipping's ~2× compute savings should be stated on its own terms without cross-attributing the 1.6× figure.

---

VERDICT: CHANGES NEEDED

---

## Agent A Change Log — Pass 1 Fix

1. **`flash_attention_prefill.md` — Q_high description corrected:** The bullet at line 88 previously stated "Q_high has many valid KV blocks but most are masked out at the tile level," which inverted the true causal geometry. Replaced with an accurate description: Q_low (early tokens) has most KV blocks masked and few valid ones (compute-light); Q_high (late tokens) has the majority of KV blocks fully valid below the causal diagonal and only a few masked above-diagonal blocks (compute-heavy). The pairing motivation — balancing cheap Q_low with expensive Q_high so each core does equal total work — is now explicit.
2. **`flash_attention_prefill.md` — 1.6× speedup attribution:** Removed the final sentence of the "Sparse Causal Mask" section that claimed the ~1.6× figure came from "block-skipping and pairing together (partially overlapping)." That claim contradicted lines 90–91, which correctly attribute the ~1.6× speedup solely to the load-balancing pairing strategy. Block-skipping's independently stated ~2× compute reduction now stands on its own without any cross-attribution to the 1.6× figure.

---

# B Review — Chapter 2: Attention Optimizations — Pass 2

1. **`flash_attention_prefill.md`, line 52 — Prose contradicts the sizing table on the same page**

   The prose states: "At D=128, Br=128, Bc=128: Q + K + V + O = 4 × (128 × 128 × 2) = 128 KB. This approaches the 120 KB L1 limit…"

   128 KB does not approach 120 KB — it exceeds it by 8 KB. The word "approaches" implies the budget is nearly full but still fits; in reality it overflows. The sizing table immediately below (line 215) correctly labels this configuration as "128 KB (exceeds L1 — not valid without tighter buffering)," directly contradicting the prose. A reader following only the prose text would infer this configuration is borderline viable; the correct message is that it is invalid at face value.

   Fix: Change "This approaches the 120 KB L1 limit" to "This exceeds the 120 KB L1 limit," consistent with the table.

VERDICT: CHANGES NEEDED

---

## Agent A Change Log — Pass 2 Fix

- **`flash_attention_prefill.md`, line 52 — L1 budget prose corrected:** Changed "This approaches the 120 KB L1 limit" to "This exceeds the 120 KB L1 limit." The original wording implied the Br=128, Bc=128, D=128 configuration (128 KB) was borderline viable, when in fact it overflows the 120 KB L1 budget. The fix brings the prose into agreement with the sizing table on the same page, which already labeled this configuration as "exceeds L1 — not valid without tighter buffering."

---

# B Review — Chapter 2: Attention Optimizations — Pass 3

## Issues Found

**1. `paged_attention_kv_cache.md`, line 15 — Arithmetic error in KV cache size example**

Wrong text: `= 32 × 4096 × 8 × 128 × 2 = 2 GB   (for Llama 3 8B parameters)`

Why it is wrong: The arithmetic is incorrect. 32 × 4096 × 8 × 128 × 2 = 268,435,456 bytes = **256 MB**, not 2 GB. The result is off by a factor of 8. A 2 GB figure would require either ~256 sequences, or n_kv_heads=64, or head_dim=1024 — none of which match Llama 3 8B (32 sequences, 8 KV heads, head_dim=128 as stated). The downstream sentences refer to "this 2 GB allocation" having 10–20% utilization, which propagates the error.

Fix: Change `= 32 × 4096 × 8 × 128 × 2 = 2 GB` to `= 32 × 4096 × 8 × 128 × 2 = 256 MB`, and update the subsequent reference from "this 2 GB allocation" to "this 256 MB allocation."

---

VERDICT: CHANGES NEEDED

---

## Agent A Change Log — Pass 3 Fix

- **`paged_attention_kv_cache.md`, line 15 — KV cache size arithmetic corrected:** Changed `= 32 × 4096 × 8 × 128 × 2 = 2 GB` to `= 256 MB`. The product 32 × 4096 × 8 × 128 × 2 = 268,435,456 bytes = 256 MB; the original result was off by a factor of 8. Also updated the downstream prose reference on line 18 from "this 2 GB allocation" to "this 256 MB allocation" to eliminate propagation of the error.

---

# B Review — Chapter 2: Attention Optimizations — Pass 4

## Issues Found

**1. `flash_decode_and_gqa.md`, line 56 — KV cache formula counts only K (or V), not K+V combined**

Wrong text: "the per-layer KV cache per batch element is 32 × 4096 × 128 × 2 bytes = **32 MB**"

Why it is wrong: The KV cache stores both K and V. The formula `32 × 4096 × 128 × 2` computes the size of K alone (n_kv_heads × S × head_dim × sizeof(bfloat16)). The full KV cache — K cache plus V cache — is twice that: 2 × 32 × 4096 × 128 × 2 bytes = 67,108,864 bytes = **64 MB** per batch element per layer. Quoting 32 MB understates the KV cache size by 2×, which also makes the subsequent GQA comparison ("the KV cache is 4× smaller than MHA would require") use the wrong MHA baseline, though the relative factor-of-G reduction claim remains directionally correct.

Fix: Change `32 × 4096 × 128 × 2 bytes = **32 MB**` to `2 × 32 × 4096 × 128 × 2 bytes = **64 MB**` (adding the factor of 2 for K and V combined).

---

**2. `paged_attention_kv_cache.md`, lines 175–176 — Inclusive range notation places core 8 in both "non-overlapping" subsets**

Wrong text: "Cores [0–8]: write K cache updates" and "Cores [8–16]: write V cache updates" — followed immediately by "K and V updates run concurrently on non-overlapping core subsets."

Why it is wrong: If [0–8] and [8–16] are inclusive ranges (the standard reading of dash-delimited integer ranges in documentation), core 8 belongs to both groups. The prose directly states these are "non-overlapping," which is contradicted by the notation. The intended implementation almost certainly uses half-open ranges — cores 0–7 for K and cores 8–15 for V — giving 8 non-overlapping cores per operation. Using inclusive notation creates an internal contradiction and will mislead a reader who tries to reconcile the ranges with the "non-overlapping" claim.

Fix: Change "Cores [0–8]" to "Cores [0–7]" and "Cores [8–16]" to "Cores [8–15]" to express 8 non-overlapping cores per group, consistent with the "non-overlapping core subsets" claim.

---

VERDICT: CHANGES NEEDED

---

## Agent A Change Log — Pass 4 Fix

1. **`flash_decode_and_gqa.md`, line 56 — KV cache formula corrected to include both K and V:** Changed `32 × 4096 × 128 × 2 bytes = **32 MB**` to `2 × 32 × 4096 × 128 × 2 bytes = **64 MB**`. The original formula computed only the K cache (or V cache alone), omitting the factor of 2 needed to account for storing both K and V. The corrected figure of 64 MB accurately represents the full per-layer KV cache footprint per batch element under MHA.
2. **`paged_attention_kv_cache.md`, lines 175–176 — Core range notation corrected to be non-overlapping:** Changed "Cores [0–8]" to "Cores [0–7]" and "Cores [8–16]" to "Cores [8–15]". The original inclusive ranges placed core 8 in both the K-write and V-write subsets, directly contradicting the "non-overlapping core subsets" claim in the following sentence. The corrected ranges assign 8 distinct cores to each operation with no overlap.

---

# B Review — Chapter 2: Attention Optimizations — Pass 5

## Issues Found

**1. `paged_attention_kv_cache.md`, line 240 — Key Takeaways still contains the old overlapping core ranges**

Wrong text: "`paged_update_cache` fuses K and V cache writes by assigning K updates to cores [0–8] and V updates to cores [8–16], enabling concurrent parallel DRAM writes."

Why it is wrong: Pass 4 corrected the body text at lines 175–176 from [0–8]/[8–16] to [0–7]/[8–15], but the Key Takeaways bullet at line 240 was not updated. The ranges [0–8] and [8–16] are inclusive, which means core 8 appears in both groups. This directly contradicts the immediately following claim that K and V updates "run concurrently on non-overlapping core subsets." The correct non-overlapping ranges — as established by the Pass 4 fix to the body — are [0–7] for K and [8–15] for V.

Fix: Change "assigning K updates to cores [0–8] and V updates to cores [8–16]" to "assigning K updates to cores [0–7] and V updates to cores [8–15]", consistent with the body text corrected in Pass 4.

---

VERDICT: CHANGES NEEDED

---

## Agent A Change Log — Pass 5 Fix

- **`paged_attention_kv_cache.md`, line 240 — Key Takeaways core ranges corrected:** Changed "assigning K updates to cores [0–8] and V updates to cores [8–16]" to "assigning K updates to cores [0–7] and V updates to cores [8–15]". Pass 4 had already corrected the same overlapping ranges in the body text (lines 175–176) but missed this occurrence in the Key Takeaways bullet. The fix brings Key Takeaways into agreement with the body and eliminates the contradiction with the "non-overlapping core subsets" claim in the same bullet.

---

# B Review — Chapter 2: Attention Optimizations — Pass 6

## Issues Found

No issues found.

All prior fixes verified in place:
- Pass 1: Q_high/Q_low geometry correct; 1.6× speedup attributed to pairing only — confirmed.
- Pass 2: "exceeds" the 120 KB L1 limit for D=128, Br=128, Bc=128 — confirmed.
- Pass 3: Paged KV size formula result reads 256 MB — confirmed (32 × 4096 × 8 × 128 × 2 = 268,435,456 bytes = 256 MB).
- Pass 4: flash_decode KV formula reads `2 × 32 × 4096 × 128 × 2 bytes = **64 MB**`; body core ranges read [0–7] and [8–15] — confirmed.
- Pass 5: Key Takeaways core ranges read [0–7] and [8–15] — confirmed.

Additional spot-checks performed this pass:
- Score matrix size at S=4096, D=128, BF16: 4096 × 4096 × 2 = 32 MB per head; 32 heads = 1 GB per layer — both correct.
- L1 sizing table entries (D=128): Br=64, Bc=64 → 64 KB; Br=128, Bc=64 → 96 KB; Br=128, Bc=128 → 128 KB — all arithmetically correct.
- Tensix core: five RISC-V processors (BRISC, NCRISC, TRISC0, TRISC1, TRISC2) — correct.
- Llama 3 8B GQA parameters: 32 Q heads, 8 KV heads, G=4 — correct.
- Sliding window reduction: W=4096, S=32768 → 8× reduction — correct (32768 / 4096 = 8).
- Block boundary interval: block_size=32 at 50 tokens/s → 0.6 s — correct (32 / 50 = 0.64 s ≈ 0.6 s).
- Ring-distributed SDPA: D phases reduced to ~D/2 for causal prefill — consistent with causal block-skip logic applied at device-shard granularity.

VERDICT: APPROVED
