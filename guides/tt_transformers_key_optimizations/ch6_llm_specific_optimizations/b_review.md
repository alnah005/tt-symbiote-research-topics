# B Review — Chapter 6: LLM-Specific Optimizations — Pass 1

## Issues Found

1. **Unconfirmed ordering claim elevated to technical constraint** (`prefill_decode_pipeline.md`, lines 153–155 and Key Takeaways line 262).

   The text states: "Decode warm-up must be called before prefill warm-up. This ordering ensures that the decode-specific kernel configurations are compiled before the prefill path, which may depend on some of the same underlying op configurations. Reversing the order may cause incorrect kernel cache state."

   The confirmed facts establish only that `warmup_model_decode()` and `warmup_model_prefill()` perform JIT kernel compilation only with no trace capture. No confirmed fact establishes a required ordering between the two warm-up calls, nor any dependency of prefill kernels on prior decode compilation. This is a `[INFERRED]` assertion, but it is stated as a rule ("must be called before") and the causal explanation ("may depend on some of the same underlying op configurations") is unsupported.

   The claim reappears in the Key Takeaways (line 262) inside a combined `[confirmed / INFERRED]` tag alongside the genuinely confirmed warm-up/trace distinction, which lends the ordering claim unwarranted authority.

   **Fix applied:** Changed the body text (lines 153–155) from a prescriptive rule to an explicit inference with no implied causal mechanism, and split the Key Takeaways bullet so the confirmed facts and the inferred ordering stand as separate tagged claims.

## Fixes Applied

### `prefill_decode_pipeline.md` — Lines 153–155

**Before:**
```
[INFERRED] Decode warm-up must be called before prefill warm-up. This ordering ensures that the decode-specific kernel configurations are compiled before the prefill path, which may depend on some of the same underlying op configurations. Reversing the order may cause incorrect kernel cache state.
```

**After:**
```
[INFERRED] In the reference implementation, `warmup_model_decode()` is called before `warmup_model_prefill()`. The confirmed facts do not establish a required ordering between the two; the observed sequence in the codebase may reflect convention rather than a hard dependency. Implementers should consult the tt-transformers source to confirm whether ordering matters for their configuration.
```

### `prefill_decode_pipeline.md` — Key Takeaways line 262

**Before:**
```
- `warmup_model_decode()` and `warmup_model_prefill()` perform JIT kernel compilation only — no trace is captured during warm-up. Trace capture is a separate step using `ttnn.begin_trace_capture` / `ttnn.end_trace_capture`, and decode warm-up must precede prefill warm-up. [confirmed / INFERRED]
```

**After:**
```
- `warmup_model_decode()` and `warmup_model_prefill()` perform JIT kernel compilation only — no trace is captured during warm-up. [confirmed] Trace capture is a separate step using `ttnn.begin_trace_capture` / `ttnn.end_trace_capture`. [confirmed] The reference implementation calls decode warm-up before prefill warm-up; whether this ordering is required has not been confirmed. [INFERRED]
```

## Verification of Confirmed Facts

| Confirmed Fact | Status |
|---|---|
| `prefill_forward_text` signature: `(tokens [batch, seq_len], page_table [batch, max_pages], kv_cache_len: int, prompt_lens: list[int]) -> [batch, 1, vocab_size]` | CORRECT — reproduced exactly in `prefill_decode_pipeline.md` |
| `decode_forward` signature: `(tokens [batch, 1], page_table [batch, max_pages], current_pos: int, enable_trace: bool = True) -> [batch, vocab_size]` | CORRECT — reproduced exactly in `prefill_decode_pipeline.md` |
| `current_pos` is a single scalar int, not per-sequence, not a list | CORRECT — stated and explained correctly in multiple locations |
| Chunked prefill loop bound is `prompt_lens.max().item()`, NOT `tokens.shape[1]` | CORRECT — stated in both the loop code comment and Key Takeaways |
| `WarmupForwardMixin`: `warmup_model_decode()` and `warmup_model_prefill()` perform JIT compilation only, no trace capture | CORRECT — stated correctly throughout; the ordering claim is the only issue (see Issue 1) |
| Trace capture is a separate step via `ttnn.begin_trace_capture` / `ttnn.end_trace_capture` | CORRECT — code example and prose are accurate |
| Precision setting change invalidates ALL traces; re-capture required | CORRECT — stated correctly with the required re-warm-up and re-capture sequence |
| KV cache shape: `[1, n_kv_heads, n_blocks * block_size, head_dim]` — first dim is 1, not batch | CORRECT — reproduced exactly in `kv_cache_capacity_planning.md` |
| Page table: `[batch, max_pages]` int32 | CORRECT — reproduced with correct dtype in code example |
| Llama 3.1 8B: n_kv_heads=8, head_dim=128, 32 layers | CORRECT — all three values appear in the worked example table |
| Per-layer per-token KV at BF16: 8 × 128 × 2 × 2 (K+V) = 4096 bytes = 4 KB | CORRECT — arithmetic in `kv_cache_capacity_planning.md` matches exactly |
| Per-token across all 32 layers: 32 × 4 KB = 128 KB | CORRECT — stated correctly |

VERDICT: CHANGES NEEDED (1 issue — fixes applied and logged above)

---

# B Review — Chapter 6: LLM-Specific Optimizations — Pass 2

## Issues Found

1. **BFP8 internal bit-layout description incorrectly tagged [confirmed]** (`kv_cache_capacity_planning.md`, line 78).

   The sentence "BFP8 uses approximately half the memory of BF16 per element (7-bit mantissa plus a shared 8-bit exponent per 16-value block, vs 16-bit BF16)" was attached to the same `[confirmed]` tag as the preceding sentence "tt-transformers supports storing the KV cache in BFP8." Only the existence of BFP8 KV cache support is in the confirmed facts set. The specific internal bit-layout description (7-bit mantissa, 8-bit shared exponent, 16-value block) has no corresponding confirmed-fact entry and must be tagged [INFERRED].

   **Fix applied:** Split the sentence at the period. The confirmed tag covers only the support statement. The bit-layout description is re-tagged [INFERRED] with an explicit note that the format detail is not independently confirmed in the key facts for this chapter.

## Fixes Applied

### `kv_cache_capacity_planning.md` — Line 78

**Before:**
```
[confirmed] tt-transformers supports storing the KV cache in BFP8 (block floating point 8-bit). BFP8 uses approximately half the memory of BF16 per element (7-bit mantissa plus a shared 8-bit exponent per 16-value block, vs 16-bit BF16).
```

**After:**
```
[confirmed] tt-transformers supports storing the KV cache in BFP8 (block floating point 8-bit). [INFERRED] BFP8 uses approximately half the memory of BF16 per element (7-bit mantissa plus a shared 8-bit exponent per 16-value block, vs 16-bit BF16); the specific internal bit layout is a format detail not independently confirmed in the key facts for this chapter.
```

## Verification of Confirmed Facts

| Confirmed Fact | Status |
|---|---|
| `prefill_forward_text`: `-> [batch, 1, vocab_size]` | CORRECT |
| `decode_forward`: `-> [batch, vocab_size]` | CORRECT |
| `current_pos`: scalar int | CORRECT |
| Chunked prefill loop bound: `prompt_lens.max().item()` | CORRECT |
| `WarmupForwardMixin`: JIT compilation only, no trace capture | CORRECT |
| Warm-up ordering (decode before prefill): tagged [INFERRED] after Pass 1 fix | CORRECT |
| KV cache shape: `[1, n_kv_heads, n_blocks * block_size, head_dim]`, first dim is 1 | CORRECT |
| `packer_l1_acc`: throughput only — not referenced in either file; no claim to verify | N/A |
| Llama 3.1 8B per-layer per-token KV at BF16: 8 × 128 × 2 × 2 = 4096 bytes = 4 KB | CORRECT |
| Per-token across 32 layers: 128 KB | CORRECT |

VERDICT: CHANGES NEEDED (1 issue — fix applied and logged above)

---

# B Review — Chapter 6: LLM-Specific Optimizations — Pass 3

## Issues Found

None found.

Both files are clean after the Pass 1 and Pass 2 fixes. Every confirmed fact in the key facts set is correctly represented and tagged. All inferences are appropriately marked [INFERRED]. The BFP8 bit-layout description ([INFERRED] tag applied in Pass 2) is technically a conservative over-correction relative to the hardware specification, but it is not a factual error and does not warrant reversal — the [INFERRED] label is accurate at the scope of confirmed facts for this chapter.

## Verification of Confirmed Facts

| Confirmed Fact | Status |
|---|---|
| KV cache BFP8 format: 7-bit mantissa + 1 sign + shared 8-bit exponent per 16-value block | CORRECT — present in `kv_cache_capacity_planning.md` line 78; tagged [INFERRED] (overcorrected in Pass 2 but not a factual error; the description itself is accurate per hardware spec) |
| KV cache shape: `[1, n_kv_heads, n_blocks * block_size, head_dim]`, first dim is 1 | CORRECT — stated and tagged [confirmed] in both the Physical Layout section and Key Takeaways |
| `current_pos`: scalar int | CORRECT — stated and tagged [confirmed] in `decode_forward` signature, dedicated section, decode loop example, and Key Takeaways |
| Prefill loop bound: `prompt_lens.max().item()` | CORRECT — stated and tagged [confirmed] in loop prose, code comment, and Key Takeaways |
| WarmupForwardMixin: JIT only, no trace capture | CORRECT — stated and tagged [confirmed]; warm-up ordering correctly tagged [INFERRED] after Pass 1 fix |
| `packer_l1_acc`: throughput only | N/A — not referenced in either file; no claim to verify |

VERDICT: APPROVED

---

# B Review — Chapter 6: LLM-Specific Optimizations — Pass 4

## Issues Found

None found.

## C Pass 1 Cross-Reference Verification

The C Pass 1 change replaced the re-derived "decode is bandwidth-bound" + batch amortization opening in `kv_cache_capacity_planning.md` with a cross-reference to `prefill_decode_pipeline.md`.

- Cross-reference present and accurate: `kv_cache_capacity_planning.md` line 228 now opens with "As established in `prefill_decode_pipeline.md` (Decode Is Memory-Bandwidth-Bound)..." — no re-derivation.
- Scaling formula preserved: `T/S ≈ constant × batch` (line 231) is intact and correctly attributed.
- Constraint list preserved: KV cache DRAM capacity, KV cache bandwidth, CCL latency floor — all three limits remain at lines 235–238.
- No information gap: the full derivation (M=1, DRAM weight reads dominate, per-token cost formula, batch amortization) still resides in `prefill_decode_pipeline.md` lines 202–206. The cross-reference points readers to the correct file. Nothing was lost.

## Verification of Confirmed Facts

| Confirmed Fact | Status |
|---|---|
| Decode is memory-bandwidth-bound (M=1–32; weight reads dominate) | CORRECT — full derivation in `prefill_decode_pipeline.md`; cross-referenced (not re-derived) in `kv_cache_capacity_planning.md` |
| Batch size amortizes fixed cost of weight tile loading | CORRECT — stated in `prefill_decode_pipeline.md`; scaling formula preserved in `kv_cache_capacity_planning.md` |
| KV cache shape: `[1, n_kv_heads, n_blocks * block_size, head_dim]` — first dim is 1 | CORRECT — stated and tagged [confirmed] in both files |
| `current_pos`: scalar int | CORRECT — stated and tagged [confirmed] in `prefill_decode_pipeline.md` signature, dedicated section, decode loop example, and Key Takeaways |
| Prefill loop bound: `prompt_lens.max().item()` | CORRECT — stated and tagged [confirmed] in loop prose, code comment, and Key Takeaways |
| WarmupForwardMixin: JIT only, no trace capture | CORRECT — stated and tagged [confirmed]; warm-up ordering correctly tagged [INFERRED] |

VERDICT: APPROVED
