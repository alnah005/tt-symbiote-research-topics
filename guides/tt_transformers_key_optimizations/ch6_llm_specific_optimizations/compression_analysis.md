# Compression Analysis — Chapter 6: LLM-Specific Optimizations
Reviewer: Agent C (compression reviewer)
Date: 2026-03-24
Files reviewed:
- index.md
- prefill_decode_pipeline.md
- kv_cache_capacity_planning.md

## Summary

The three files are well-scoped and largely non-redundant. `index.md` serves as a navigation layer (Key Concepts table, reading order, prerequisites), so its brief restatements of facts owned by the two content files are expected and appropriate — they are pointers, not re-derivations, and should not be cut. The only substantive cross-file redundancy is a near-verbatim block about decode being memory-bandwidth-bound: `prefill_decode_pipeline.md` explains the mechanism in full (the M=1 arithmetic intensity argument, the amortization logic), and `kv_cache_capacity_planning.md` independently re-derives the same mechanism in the Capacity vs Throughput section rather than simply citing it. That duplication is the one case where a cut-and-cross-reference is warranted. All other cross-file overlaps are either summary pointers in `index.md` (correct by design) or single-sentence restatements that orient the reader without duplicating prose.

---

## CRUCIAL Suggestions

1. **Decode-is-memory-bandwidth-bound prose duplicated across `prefill_decode_pipeline.md` and `kv_cache_capacity_planning.md`.**

   `prefill_decode_pipeline.md` lines 204–206 (the "Decode Is Memory-Bandwidth-Bound" subsection) contains the full, confirmed explanation: M=1 arithmetic intensity, DRAM weight-load cost, and the amortization argument. `kv_cache_capacity_planning.md` lines 228–234 (the "Throughput Scaling with Batch Size" subsection opening) re-derives the same mechanism nearly verbatim — "Decode is memory-bandwidth-bound: the dominant cost per step is loading weight tiles from DRAM, not compute. Each decode step loads the same weight tiles regardless of batch size."

   **Fix:** In `kv_cache_capacity_planning.md`, replace the re-derived opening sentence(s) of "Throughput Scaling with Batch Size" with a one-sentence back-reference to `prefill_decode_pipeline.md`, then continue directly with the scaling formula and the secondary constraint list. The confirmed fact and its explanation live in `prefill_decode_pipeline.md`; `kv_cache_capacity_planning.md` only needs to invoke the conclusion.

---

## MINOR Suggestions

1. **`index.md` Key Concepts table repeats warm-up wording verbatim from `prefill_decode_pipeline.md`.** The table entries for "Warm-up (JIT compilation)" and "Trace capture" (index.md lines 54–55) use phrasing identical to the confirmed-fact sentences in `prefill_decode_pipeline.md` lines 143 and 159. This is expected for a navigation index and does not need cutting, but if the index ever diverges from the source file the duplication becomes a maintenance hazard. No action required now; noted for awareness.

2. **`index.md` "What You Will Be Able to Do" learning objective (line 32) restates the `kv_cache_capacity_planning.md` block pool formula** ("Size a paged block pool for a target batch size, context length, and block size"). This is a legitimate learning-objective summary and should not be cut. Noted only because it is a near-verbatim echo of the section heading in `kv_cache_capacity_planning.md`.

---

## Load-Bearing Evidence

The following content must NOT be cut regardless of compression decisions:

- `prefill_decode_pipeline.md` — the full `prefill_forward_text` and `decode_forward` confirmed signatures (lines 17–23 and 37–43). These are the only confirmed API references in the chapter.
- `prefill_decode_pipeline.md` — the chunked prefill loop code block (lines 68–90) and the confirmation that the loop bound is `prompt_lens.max().item()`, not `tokens.shape[1]` (line 64). This is a correctness-critical distinction.
- `prefill_decode_pipeline.md` — the confirmation that `current_pos` is a scalar int (lines 129–133). This has direct implications for static vs continuous batching compatibility.
- `prefill_decode_pipeline.md` — the initialization sequence table (lines 192–197) and the precision-change invalidation rule (lines 179–187). These are operational requirements.
- `kv_cache_capacity_planning.md` — the per-token KV footprint derivation table and the 128 KB/token confirmed result for Llama 3.1 8B BF16 (lines 34–66). This is the anchor calculation for all sizing decisions.
- `kv_cache_capacity_planning.md` — the confirmed page table shape `[batch, max_pages]` int32 and the constant-shape requirement with recompilation warning (lines 171–202). This is a confirmed correctness constraint.
- `kv_cache_capacity_planning.md` — the `compute_kv_pool_bytes` reference function and worked example (lines 251–276). This is the canonical sizing procedure for the chapter.

---

## VERDICT
- Crucial updates: yes — applied. In `kv_cache_capacity_planning.md`, the "Throughput Scaling with Batch Size" subsection opening was revised: the re-derived decode-is-memory-bandwidth-bound prose (lines 228–229) was replaced with a single back-reference sentence to `prefill_decode_pipeline.md`, eliminating the cross-file duplication while preserving the scaling formula and secondary constraint list that are unique to the capacity planning context.

---

# Compression Pass 2 — Chapter 6: LLM-Specific Optimizations
Date: 2026-03-24

## CRUCIAL Suggestions

None found.

## MINOR Suggestions

1. **`prefill_decode_pipeline.md` restates `current_pos`-is-a-scalar in three places within the same file.** The API signature section (lines 48–49), the dedicated "current_pos Is a Scalar" subsection (lines 129–133), and the Key Takeaways (line 261) all state the same confirmed fact. This is within-file repetition only (not cross-file), and Key Takeaways restatement is standard structure. No action required; noted for awareness.

2. **`index.md` Key Concepts table block pool formula (line 60) is arithmetically equivalent to `kv_cache_capacity_planning.md` lines 126–127.** The index renders it as a single collapsed expression (`ceil(batch * max_seq_len / block_size)`); the capacity file derives it in two steps. These serve different purposes (navigation pointer vs derivation) and the minor phrasing difference is not a maintenance hazard. No action required.

## VERDICT
- Crucial updates: no
