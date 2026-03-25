# Compression Analysis — Chapter 7: Llama3 Walkthrough
Reviewer: Agent C (compression reviewer)
Date: 2026-03-24
Files reviewed:
- index.md
- single_device_decode.md
- multi_device_decode.md

## Summary

The three files are generally well-scoped: `index.md` is a navigation layer, `single_device_decode.md` is the single-device reference, and `multi_device_decode.md` extends it for TP=8. Two crucial cross-section redundancies were found and fixed. First, `single_device_decode.md` carried a full re-statement of the LM-head-never-quantized rationale in Step 13 that was already given verbatim in the dedicated subsection "LM Head and Embedding: Always BF16" forty lines earlier. Second, `multi_device_decode.md` Step 11 repeated the `reduce_scatter`-vs-`all_reduce` justification in full even though Step 5, just above, already explained it identically. Several minor repetitions were also found — notably the `current_pos` scalar explanation appearing four times across files and the KV-cache first-dimension note appearing twice in matching language — but these are tolerable given their role as inline cross-references and were left in place.

## CRUCIAL Suggestions

1. **LM head rationale duplicated inside `single_device_decode.md` (Step 13 vs dedicated subsection).** The paragraph beginning "The LM head is never quantized to BFP8 or BFP4" in Step 13 was near-verbatim identical to the paragraph beginning "The language model head ... and the token embedding table are always kept in BF16" in the "LM Head and Embedding: Always BF16" subsection. Step 13 was the secondary occurrence. **Fixed:** Step 13 now reads "See 'LM Head and Embedding: Always BF16' above for the rationale."

2. **`reduce_scatter`-vs-`all_reduce` rationale duplicated inside `multi_device_decode.md` (Step 11 vs Step 5).** Step 5 carries the authoritative explanation: "`reduce_scatter` is preferred here because the result is immediately used by the next all_gather (or by a subsequent sharded operation), making the scatter distribution the efficient choice rather than replicating the full result on every device." Step 11 then restated this in nearly identical language: "The row-parallel output projection always uses `reduce_scatter` to combine partial sums while also sharding the result for downstream use." **Fixed:** Step 11 now reads "`ttnn.reduce_scatter` is used, not `ttnn.all_reduce` — same rationale as Step 5."

## MINOR Suggestions

1. **`current_pos` scalar explanation stated four times.** `single_device_decode.md` Step 3 (lines 145–147) gives the full explanation; `multi_device_decode.md` Step 3 (line 119) and Key Takeaways (line 346) restate it, and the Further Reading section points to it again. The Step 3 instances are acceptable as inline reminders in context. The Key Takeaways restatement in `multi_device_decode.md` ("just as in the single-device case") is a near-echo that could be trimmed to a cross-reference, but the redundancy is mild.

2. **KV cache first-dimension note duplicated.** `single_device_decode.md` Step 4 and `multi_device_decode.md` Step 4 both carry the identical parenthetical "The first dimension is 1, not batch; the paged block pool is shared across sequences." In `multi_device_decode.md` this is appropriate context because the KV cache shape changes (from `[1, 8, ...]` to `[1, 1, ...]`), so leaving both in place is defensible.

3. **TP=8 = n_kv_heads=8 favorable-property paragraph tripled in `multi_device_decode.md`.** The observation that TP equals n_kv_heads making the split clean appears in the "GQA at TP=8" section, again in Step 4, and again in Key Takeaways. The Step 4 instance is a brief single-sentence reminder; the GQA section and Key Takeaways both carry the fuller explanation. The GQA section is the right home; the Key Takeaways sentence is compact enough to keep; the Step 4 inline note is the one that could be shortened to a pointer, but the overlap is minor.

## Load-Bearing Evidence

The following content must NOT be cut:

- `single_device_decode.md` — "LM Head and Embedding: Always BF16" subsection (the authoritative rationale for not quantizing the LM head; the Step 13 occurrence was correctly reduced to a pointer to this section)
- `single_device_decode.md` — "Why BFP4 MLP Gives ~22% Higher Throughput" section with the 3.56x / 1.88x / 1.89x figures and the ~22% / 28 t/s/u / 23 t/s/u numbers
- `multi_device_decode.md` Step 5 — `ttnn.reduce_scatter` rationale paragraph ("preferred here because the result is immediately used by the next all_gather..."); Step 11 correctly now defers to this
- `multi_device_decode.md` Step 13 — the `ttnn.all_reduce` explanation distinguishing it from `ttnn.all_gather` (every device needs the full logit vector for next-token sampling)
- `multi_device_decode.md` — "Collective Communication Summary" table; it is the only place all five collective sites are listed in tabular form
- `index.md` — "Key Concepts" table and "Relationship to Prior Chapters" section; these are navigation aids, not body repetition

## VERDICT
- Crucial updates: yes

# Compression Pass 2 — Chapter 7: Llama3 Walkthrough
Date: 2026-03-24

## CRUCIAL Suggestions
None found.

## MINOR Suggestions
1. `multi_device_decode.md` Step 5 contains a full inline rationale for why `ttnn.reduce_scatter` is preferred over `ttnn.all_reduce` (lines ~161–162). The Key Takeaways section (line ~342) restates the same conclusion in one sentence. Both are within the same file, not across files, and the takeaway is brief — borderline minor. No cross-file duplication remains.

2. `index.md` Key Concepts table repeats the throughput figures (~23 t/s/u accuracy mode, ~28 t/s/u performance mode) and the BFP4 bandwidth multiplier (3.56x, ~22% higher throughput) that are fully explained in `single_device_decode.md`. The index entries are single-cell summaries pointing to the source file, which is the intended role of an index — not actionable.

## VERDICT
- Crucial updates: no
