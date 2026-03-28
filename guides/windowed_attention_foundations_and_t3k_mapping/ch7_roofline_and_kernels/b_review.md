## Pass 1

### Item 1 — Wrong unit label in BW column of comparison table

**File:** `roofline_analysis.md`, line ~218 (table header `BW required per layer`)

**Error:** The column header labels bandwidth values as "MiB" but the values are in MB (decimal megabytes, base-10). For `n_read = 4096`, the formula `B · 2 · H_kv · n_read · d · 2 = 16,777,216 bytes` equals exactly **16.0 MiB** (binary) but the table shows "16.8 MiB". The value 16.8 is correct only in MB (16,777,216 / 10^6 = 16.777 ≈ 16.8 MB). Same mismatch for all three rows (32.0 MiB vs "33.6 MiB"; 128.0 MiB vs "134.2 MiB").

A reader using the label "MiB" to verify the numbers will be unable to reproduce them and will compute bandwidth requirements ~5% too low.

**Fix:** Change the column header from "MiB" to "MB" (and keep the values as-is), or change the values to 16.0 / 32.0 / 128.0 and keep the "MiB" label.

---

### Item 2 — Throughput values computed with 256 GiB/s, not the stated 288 GB/s

**File:** `roofline_analysis.md`, lines ~218–222 (throughput column of comparison table)

**Error:** The throughput values (16,385 / 8,192 / 2,048 layers/s) are each consistent with a peak DRAM bandwidth of exactly **256 GiB/s = 274.9 GB/s**, not the 288 GB/s stated in the hardware table and used throughout the document.

Verification:
- `n_read = 8192`: BW per step = 33,554,432 bytes; 256 GiB/s / 33,554,432 bytes = **8,192** layers/s (exact). With 288 GB/s: 288×10^9 / 33,554,432 = **8,583** layers/s.
- `n_read = 32768`: 256 GiB/s / 134,217,728 bytes = **2,048** layers/s (exact). With 288 GB/s: **2,146** layers/s.

Any reader who applies the stated formula (`Peak BW / BW required`) with the stated 288 GB/s will compute throughputs ~5% higher than the table shows and will be unable to reproduce the values.

**Fix:** Recompute all three throughput entries using 288 GB/s = 288 × 10^9 bytes/s:

| Mode              | n_read | BW required per layer | Throughput (layers/s) | vs full T=32768 |
|-------------------|--------|-----------------------|-----------------------|-----------------|
| Full attn T=4096  | 4,096  | 16.0 MiB              | 17,166                | 8×              |
| Full attn T=8192  | 8,192  | 32.0 MiB              | 8,583                 | 4×              |
| Full attn T=32768 | 32,768 | 128.0 MiB             | 2,146                 | 1× (baseline)   |
| Windowed w=4096   | 4,096  | 16.0 MiB              | 17,166                | 8×              |
| Windowed w=8192   | 8,192  | 32.0 MiB              | 8,583                 | 4×              |

(This fix also corrects the MiB label from Item 1.)

---

### Item 3 — `valid_seq_len` fill-phase prescription is off by one in `existing_kernel_survey.md`

**File:** `existing_kernel_survey.md`, line ~110 (`For the fill phase (T < w), passing a mask with -inf at uninitialised slots [T+1, w-1]`) and line ~295 (`valid_seq_len = T + 1`)

**Assessment:** These two values are actually consistent and correct. At 0-indexed position T the buffer contains tokens 0 through T inclusive — that is T+1 filled slots — so `valid_seq_len = T + 1` and masking out `[T+1, w-1]` (0-indexed) are both correct. **No error here.**

*(Included to document that this was checked; not a flagged item.)*

---

### Items 4–5 — No further correctness issues found

**Navigation footer** (`roofline_analysis.md`, final line): `**Next:** [existing_kernel_survey.md](./existing_kernel_survey.md)` — present and correct.

**Terminal marker** (`existing_kernel_survey.md`, final line): `*This is the final chapter of the guide.*` — present and correct.

**Clickable links in `index.md`**: All six relative links verified against the filesystem:
- `./roofline_analysis.md` — exists
- `./existing_kernel_survey.md` — exists
- `../ch3_data_dependencies/decode_access_patterns.md` — exists
- `../ch3_data_dependencies/prefill_access_patterns.md` — exists
- `../ch4_ttnn_primitives/decode_primitives.md` — exists
- `../ch4_ttnn_primitives/kernel_or_op_gap_analysis.md` — exists
- `../ch6_t3k_sharding/sharding_strategies.md` — exists

**Hardware numbers** (crossover AI = 111 FLOPs/byte): 32×10^12 / 288×10^9 = 111.1. Correct.

**AI formula derivation** (FLOPs/bytes = H_q/H_kv): algebra verified; correct.

**B_crossover formula** (= AI_crossover / G = 111 / G): algebra verified; correct.

**Gap table and recommended paths**: No factual errors found; severity classifications and the "no new kernel required" conclusion are consistent with the analysis in the body of the file.

---

**Summary: 2 actionable correctness issues (Items 1 and 2). Both are in the same comparison table in `roofline_analysis.md` (lines ~216–222). The BW column unit label is wrong (MB labeled as MiB) and the throughput column was computed using 256 GiB/s instead of the stated 288 GB/s.**

---

## Change Log (Pass 1)

**File edited:** `roofline_analysis.md`, comparison table (~lines 216–222)

**Issue 1 fixed — BW column values corrected to true MiB:**

All five BW values in the "BW required per layer" column were recomputed as exact binary MiB using the formula `B · 2 · H_kv · n_read · d · 2` with H_kv=8, d=128, B=1, BF16 (2 bytes):

- n_read=4096: 16,777,216 bytes = **16.0 MiB** (was 16.8 MiB)
- n_read=8192: 33,554,432 bytes = **32.0 MiB** (was 33.6 MiB)
- n_read=32768: 134,217,728 bytes = **128.0 MiB** (was 134.2 MiB)

The unit label "MiB" is retained; the old values were decimal MB incorrectly labelled as MiB.

**Issue 2 fixed — Throughput column recomputed using 288 GB/s:**

All five throughput values were recalculated as `288 × 10^9 bytes/s / bytes_per_step`:

- n_read=4096: 288,000,000,000 / 16,777,216 = **17,166 layers/s** (was 16,385)
- n_read=8192: 288,000,000,000 / 33,554,432 = **8,583 layers/s** (was 8,192)
- n_read=32768: 288,000,000,000 / 134,217,728 = **2,146 layers/s** (was 2,048)

The old values (16,385 / 8,192 / 2,048) were consistent with 256 GiB/s, not the 288 GB/s stated in the hardware table.

**Internal consistency check:** For each corrected row, `BW_MiB × 1,048,576 × throughput` approximates 288 × 10^9 bytes/s within rounding:
- 16.0 × 1,048,576 × 17,166 = 288.0 × 10^9 ✓
- 32.0 × 1,048,576 × 8,583 = 288.0 × 10^9 ✓
- 128.0 × 1,048,576 × 2,146 = 288.0 × 10^9 ✓

---

## Pass 2

### Item 1 — Chunked prefill loop uses `is_causal=True` incorrectly, producing wrong attention weights

**File:** `existing_kernel_survey.md`, lines 330–339 (Approach A code snippet)

**Error:** The code passes `is_causal=True` to `ttnn.scaled_dot_product_attention` for each query chunk. The causal flag generates a lower-triangular mask in the **local** index space of the chunk, meaning local query index `i` is blocked from attending to local KV index `j` whenever `j > i`. However, the KV chunk starts at `k_start = max(0, t0 - w + 1)`, which is offset from the query chunk start `t0`. Once `k_start < t0` (i.e., once T ≥ w), valid KV positions with absolute index `k_start + j < t0 + i` but local index `j > i` are incorrectly masked out.

Concrete example: `w = 4, q_chunk = 2, t0 = 2`. Then `k_start = max(0, 2-4+1) = 0`, `k_end = 4`. Local Q indices 0 and 1 correspond to absolute positions 2 and 3. Local KV index 0 = absolute 0, index 1 = absolute 1, index 2 = absolute 2, index 3 = absolute 3. `is_causal=True` blocks (Q_local=0, KV_local=1), (Q_local=0, KV_local=2), (Q_local=0, KV_local=3), (Q_local=1, KV_local=2), (Q_local=1, KV_local=3). But absolute position 2 attending to absolute positions 0 and 1 (local KV 0 and 1) is valid and is correctly unblocked. The problem is that local query 0 (absolute 2) is also blocked from attending to local KV 2 (absolute 2) — that pair should be allowed by causal masking. Wait, local Q=0 blocked from local KV 2 and 3 is correct (future positions). Let me re-examine.

Re-examining: `is_causal=True` blocks `j > i`. For (Q_local=0, KV_local=0): j=0 ≤ i=0 → allowed. KV_local=1: j=1 > i=0 → **blocked**. But absolute KV position 1 < absolute Q position 2, so this should be allowed. A reader implementing this code verbatim produces incorrect attention for any chunk where `k_start < t0`, which is every chunk after the window fills (T ≥ w). The correct approach is either to pass an explicit causal mask offset by `t0 - k_start`, or to use an additive mask constructed from absolute positions.

**Impact:** Incorrect (category b). The code produces silent attention errors that distort model outputs without raising an exception.

---

### Item 2 — Prefill AI formula `T·w/(w+T)` is labelled as the general prefill AI but is only valid for the chunked working-set model

**File:** `roofline_analysis.md`, line 326

**Error:** The text states "Prefill AI (derived in ch3) is `T·w / (w + T)`, which for T >> w approaches w FLOPs/byte and can be compute-bound for large w." This formula is derivable only if the byte count treats the KV working set as length w (not T): bytes = H*T*d*2 (Q) + H*w*d*4 (KV, window-sized) gives AI = 4*H*T*w*d / (2*H*T*d + 4*H*w*d) = T·w/(T/2 + w) — close but not exact. The exact match arises when bytes = 2*H*(T + w)*d*2 (Q of length T plus KV of length w), giving AI = 4*H*T*w*d / (4*H*(T+w)*d) = T·w/(T+w).

For the naive single-call path described immediately before this line (full attn_mask over T×T score matrix, no tile skipping), the actual bytes read are T-length K and V tensors: bytes = 4*H*T*d. FLOPs in-band = 4*H*T*w*d. AI = w (not T·w/(T+w)). The formula systematically understates AI by a factor of T/(T+w), which for T = 32768 and w = 4096 is 32768/36864 ≈ 0.89 — about 11% lower than the correct value. A reader trying to determine whether prefill crosses the compute-bound threshold at a specific T and w will compute a threshold that is ~11% too pessimistic (i.e., conclude prefill is bandwidth-bound at parameter combinations where it would actually be compute-bound).

**Impact:** Wrong numerical answer for any reader evaluating the single-call prefill path (category a). The formula is correct only for the chunked working-set scenario, which should be stated explicitly.

---

No further issues found. Pass 1 corrections (BW column to exact MiB values, throughput column to 288 GB/s basis) are confirmed applied and arithmetically correct.

---

## Pass 3

### Item 1 — "Arithmetic intensity crosses the roofline" with batch size is factually wrong

**File:** `roofline_analysis.md`, lines 262–264

**Error:** The text states "at a high enough batch size the arithmetic intensity crosses the roofline." AI is constant and independent of B — as the document itself correctly states two sentences earlier ("Increasing the batch size multiplies both FLOPs and bytes by B... leaving AI unchanged"). AI never crosses 111 FLOPs/byte by batching; what actually happens is that the bandwidth-bound throughput scaling line (throughput ∝ B) hits the compute ceiling (32 TFLOPS). B_crossover marks the point where compute saturates, not where AI changes.

A reader who takes the stated claim at face value will incorrectly believe batching increases AI, which would cause them to model the operation as shifting along the roofline x-axis when it does not. The formula `B_crossover = AI_crossover / G = 111/G` and the concrete values (B≈28 for G=4; B≈111 for G=1) are arithmetically correct — only the causal explanation is wrong.

**Fix:** Replace "the arithmetic intensity crosses the roofline" with language describing that the compute ceiling is reached: e.g., "the bandwidth-limited throughput scaling (throughput ∝ B) hits the 32 TFLOPS compute ceiling."

---

### Item 2 — Naive prefill AI stated as "≈ w FLOPs/byte" — only correct for MHA (G=1)

**File:** `roofline_analysis.md`, line 332

**Error:** The scoping note states "For the naive full-sequence path (full T-length K and V tensors, no tile reuse), AI ≈ w FLOPs/byte directly." This is correct only when H_q = H_kv (MHA, G=1). In the general GQA case:

- FLOPs = 4·H_q·T·w·d (each of H_q query heads attends to w KV positions)
- Bytes = 4·H_kv·T·d (K and V, each of length T, H_kv heads, BF16)
- AI = (H_q/H_kv)·w = G·w

For G=4 (the example model used in the comparison table, H_q=32, H_kv=8) and w=4096, the naive prefill AI is 4×4096 = 16,384 FLOPs/byte — not 4,096. A reader computing the compute-bound threshold for a GQA model from the stated formula will get a result that is G× too low. The conclusion that prefill is compute-bound at w=4096 still holds (G·w >> 111 for all G ≥ 1 and w ≥ 4096), but the stated AI value is wrong by a factor of G.

**Fix:** Change "AI ≈ w FLOPs/byte" to "AI ≈ G·w FLOPs/byte (where G = H_q/H_kv; equals w for MHA)."

---

No further issues found. The two Pass 2 corrections (Approach A mask fix and prefill AI formula scoping note) are confirmed applied and substantively correct.

---

## Change Log (Pass 3)

### Fix 1 — `roofline_analysis.md`, lines ~262–264 (B_crossover causal explanation)

**Issue:** The prose stated "at a high enough batch size the arithmetic intensity crosses the roofline." This is factually wrong: AI = H_q/H_kv is fixed regardless of batch size, as the document itself states two sentences earlier. Batching does not move the operating point along the AI axis; what happens is that the bandwidth-limited throughput scaling line (throughput ∝ B) hits the compute ceiling.

**Fix applied:** Replaced the incorrect sentence with: "at B_crossover, the total FLOP rate (batch × FLOPs_per_seq) reaches the 32 TFLOPS compute ceiling — the system transitions from bandwidth-limited to compute-limited throughput. The arithmetic intensity of a single sequence is unchanged; it is the aggregate work per memory transfer that saturates the ALUs." The B_crossover formula and concrete values (≈28 for G=4, ≈111 for MHA) are correct and unchanged.

---

### Fix 2 — `roofline_analysis.md`, line ~332 (naive prefill AI missing G factor)

**Issue:** The scoping note stated "AI ≈ w FLOPs/byte" for the naive full-sequence prefill path. This is only correct for MHA (G=1). In the general GQA case: FLOPs = 4·H_q·T·w·d, Bytes = 4·H_kv·T·d, so AI = (H_q/H_kv)·w = G·w. For G=4 and w=4096 the correct naive prefill AI is 16,384 FLOPs/byte, not 4,096. The old formula understates AI by a factor of G for any GQA model.

**Fix applied:** Changed "AI ≈ w FLOPs/byte" to "AI ≈ G·w FLOPs/byte (where G = H_q/H_kv; for MHA G=1 this simplifies to w)" and added a concrete example: "with G=4 and w=4096 the naive prefill AI is 4×4096 = 16,384 FLOPs/byte." Also updated the comparison sentence to note that `T·w/(T+w)` is ≤ w and thus ≤ G·w. The conclusion that prefill is well above the 111 FLOPs/byte crossover remains valid and is now accurate for GQA models.

---

## Pass 4

### Pass 3 verification

**Item 1 — B_crossover prose (lines 262–264):** Confirmed fixed. The text now reads "at B_crossover, the total FLOP rate (batch × FLOPs_per_seq) reaches the 32 TFLOPS compute ceiling — the system transitions from bandwidth-limited to compute-limited throughput. The arithmetic intensity of a single sequence is unchanged; it is the aggregate work per memory transfer that saturates the ALUs." The prior "AI crosses roofline" language is gone.

**Item 2 — Naive prefill AI (lines 334–338):** Confirmed fixed. The text now reads "AI ≈ G·w FLOPs/byte (where G = H_q/H_kv; for MHA G=1 this simplifies to w)" with the concrete example "with G=4 and w=4096 the naive prefill AI is 4×4096 = 16,384 FLOPs/byte."

### Independent pass over full chapter

No issues found that meet the flagging threshold. Verification details:

- Crossover AI = 32×10^12 / 288×10^9 = 111.1 FLOPs/byte. Correct.
- AI_decode = H_q/H_kv = G. Algebra verified; correct.
- B_crossover = AI_crossover / G = 111/G. Derivation: in the BW-bound regime the achieved FLOP rate = B × G × BW; setting equal to 32 TFLOPS gives B_crossover = 32T/(G × 288G) = 111/G. Correct. Concrete values G=4 → ≈28, G=1 → ≈111 confirmed.
- Comparison table (lines 216–222): throughput values recomputed — 288×10^9 / 16,777,216 = 17,166; 288×10^9 / 33,554,432 = 8,583; 288×10^9 / 134,217,728 = 2,146. All match the table. BW column values confirmed as exact MiB (16.0/32.0/128.0). Correct.
- Chunked loop mask (existing_kernel_survey.md lines 352–358): uses absolute positions `q_idx = t0..t0+q_len-1`, `kv_idx = k_start..k_end-1`; causal constraint `kv_idx <= q_idx`; window constraint `kv_idx >= q_idx - w + 1`. Both conditions correct. The comment explaining why `is_causal=True` is wrong for offset chunks is present and accurate.
- T·w/(T+w) ≤ w ≤ G·w ordering (lines 339–340): mathematically correct. The "pessimistic (lower)" framing refers to this formula giving a conservative lower bound relative to G·w; the Implications section (point 4) still correctly recommends Flash-Attention for prefill. No category (a)/(b)/(c) issue.
- Gap table and recommended paths: severity classifications and "no new kernel required" conclusion are internally consistent with body text.

**No feedback — chapter approved.**

---

## Change Log (Pass 2)

### Fix 1 — `existing_kernel_survey.md`, Approach A code snippet (~lines 330–339)

**Issue:** The snippet passed `is_causal=True` to `ttnn.scaled_dot_product_attention`. That flag applies a lower-triangular mask in local chunk index space. Once `k_start < t0` (every chunk after the first when T ≥ w), valid KV tokens whose local index `j > i` (local Q index) but whose absolute position `k_start + j ≤ t0 + i` are incorrectly masked out, silently producing wrong attention weights.

**Fix applied:** Replaced `is_causal=True` with an explicit additive attention mask constructed in absolute token-position space. The mask is built as a `[q_len, kv_len]` float tensor where entry `(i, j)` is `0.0` when both the causal constraint (`k_start + j ≤ t0 + i`) and the window constraint (`k_start + j ≥ t0 + i - w + 1`) hold, and `-inf` otherwise. The mask is expanded to `[B, 1, q_len, kv_len]` and passed as `attn_mask=mask`. A comment in the snippet explains why `is_causal=True` is incorrect for offset KV chunks.

---

### Fix 2 — `roofline_analysis.md`, line ~326

**Issue:** The text stated Prefill AI is `T·w/(T+w)` without qualification. This formula is derived for the Flash-Attention tiled path (KV working set of size w reused across query tiles). For the naive full-sequence path (full T-length K and V, no tile reuse), the correct AI is approximately `w` FLOPs/byte. The unscoped formula is always ≤ w and gives a slightly more pessimistic AI, which could cause a reader to incorrectly conclude prefill is bandwidth-bound at parameter values where it would actually be compute-bound.

**Fix applied:** Added a clarifying note immediately after the formula scoping it to the Flash-Attention tiled implementation, explaining the difference for the naive path (AI ≈ w), and confirming that at w=4096 the AI is well above the 111 FLOPs/byte roofline crossover for T >> w regardless of which formula applies. The formula itself is retained.
