# Compression Analysis -- Chapter 7: Model Assembly

## File-by-File Assessment

---

### 1. `index.md`

**Crucial updates: no**

**Load-Bearing Evidence:** The module hierarchy diagram (lines 31-73) is genuinely useful as a standalone reference artifact -- it gives the reader the complete module tree in one place and is not duplicated verbatim elsewhere.

**MINOR suggestions:**

- **"After reading this chapter you will know" list (lines 12-23) restates the Reading Order section (lines 76-87).** Both enumerate the same four topics (decoder layer, FFN, PLE, full model) with nearly identical descriptions. Remove the bullet list and let the Reading Order section serve as the chapter map. Saves ~12 lines.

- **Key Constants table (lines 108-122) repeats Chapter 1.** The table is a verbatim subset of config parameters already defined in the Architecture Overview. A single sentence linking to Chapter 1 would suffice; readers who have reached Chapter 7 do not need the constants re-listed.

- **Prerequisites section (lines 89-105) lists every prior chapter with one-line summaries.** These summaries add little value this deep in the guide. Replace with a single line: "This chapter builds on Chapters 1-6; see the [guide index](../index.md) for links." Saves ~16 lines.

---

### 2. `decoder_layer_module.md`

**Crucial updates: no**

**Load-Bearing Evidence:** The forward method pseudocode (lines 140-198) is the authoritative reference for the decoder layer's operation ordering (PLE -> norm -> attn -> residual -> norm -> FFN -> norm -> residual -> scalar), and it is not reproduced in this exact form anywhere else.

**MINOR suggestions:**

- **Layer Type Dispatch table (lines 107-117) is unnecessary.** The rule `layer_idx % 6 == 5` is stated on line 104. The 10-row table walking through indices 0-7 plus 59 to illustrate modular arithmetic is padding. One sentence and the formula are sufficient. Saves ~12 lines.

- **Attention Module Types table (lines 129-132) duplicates Chapter 5 and the index.md hierarchy diagram.** The table restates KV head counts, head_dim, RoPE type, and window size -- all of which are defined in earlier chapters and shown in the index diagram. A cross-reference sentence is enough.

- **Tensor Shapes Through the Layer table (lines 245-255) adds minimal value.** Every row shows the same shape `[1, 1, 5376]` with status "Replicated." The table has 9 rows that all say the same thing. A single sentence ("The hidden_states tensor remains `[1, 1, 5376]` replicated throughout the layer; sharding occurs only inside the attention and FFN submodules") replaces the entire table. Saves ~12 lines.

- **Memory Lifetime of Residual Tensors section (lines 261-274) belabors a trivial point.** Two paragraphs to say a 10 KB tensor fits in L1 and can be freed after the add. Condense to 2 sentences.

---

### 3. `ffn_module.md`

**Crucial updates: no**

**Load-Bearing Evidence:** The two option pseudocode blocks (Option 1: lines 58-93, Option 2: lines 100-132) with the trade-off table (lines 136-143) provide concrete implementation alternatives that are unique to this file and directly actionable.

**MINOR suggestions:**

- **Weight Shapes and Sharding section (lines 173-213) repeats Chapter 6.** The per-device shapes, byte calculations, and column/row parallel explanations are already covered in the TP Sharding chapter. The "Per-Device FFN Weight Total" sub-table (lines 199-208) and the cross-layer calculation (line 208-209) are direct restates. Replace with a brief summary and a link. Saves ~35 lines.

- **"The output dim 21504 divides cleanly by 8: 21504 / 8 = 2688" (line 185) is repeated arithmetic.** This division is stated in the pseudocode comments (line 66), the forward pass diagram (line 33), and again here. Once is enough.

- **Program Config Recommendations section (lines 216-259) repeats DRAM-sharded / L1 placement advice.** The gate/up subsection and the down subsection say nearly identical things: weight in DRAM-sharded, activation in L1, output in L1. Factor out the common pattern into one paragraph, then note only the differences (shape and the all-reduce for down). Saves ~15 lines.

---

### 4. `ple_module.md`

**Crucial updates: no**

**Load-Bearing Evidence:** The no-op implementation (lines 28-61) combined with the "PLE Status in 31B" table (lines 11-14) clearly establishes why PLE can be skipped in implementation, which is the key actionable takeaway for anyone building the 31B variant.

**MINOR suggestions:**

- **The file is ~216 lines for a feature that is disabled in 31B.** The "PLE Mechanism (When Enabled)" section (lines 63-132), "Host vs Device Decision" section (lines 134-163), "Multimodal Pad-Token Handling" section (lines 165-198), and "Weight Shapes (When PLE Is Active)" section (lines 199-213) together account for ~150 lines documenting a code path that does not execute. Consider moving all PLE-enabled content to an appendix or collapsible section, keeping only the no-op implementation and a forward reference. This would reduce the file to ~50 lines for the 31B-relevant content.

- **"For documentation purposes and to support future Gemma 4 variants" (line 64) is a hedge that could be a single-line note** rather than a lead-in to 70 lines of mechanism description.

---

### 5. `full_model_module.md`

**Crucial updates: no**

**Load-Bearing Evidence:** The `decode_step` pseudocode (lines 394-429) is the authoritative single-step inference pipeline -- embedding, RoPE slicing, decoder loop, final norm, LM head, softcapping -- in one place. This is unique and not assembled elsewhere.

**MINOR suggestions:**

- **Layer Sequence visualization (lines 167-171) repeats the same pattern shown in the index.md diagram, decoder_layer_module.md's dispatch table, and Chapter 1.** This is the fourth rendering of the 5:1 pattern in the guide. Remove it.

- **Logit Softcapping section (lines 239-260) is over-explained.** The math formula, the 3-line TTNN mapping, and then two paragraphs explaining what tanh does and that a fused kernel could help. The formula and code are sufficient; the prose adds ~6 lines of explanation a developer does not need.

- **Complete Per-Device Memory Summary table (lines 480-495) largely duplicates Chapter 6's KV cache sharding analysis and weight budget tables.** The numbers are reassembled from prior chapters. A summary line with a cross-reference would suffice. Saves ~20 lines.

- **The `generate` method (lines 456-477) is a generic autoregressive loop** with no Gemma 4 or TTNN-specific content. It is standard boilerplate (`for step in range(max_new_tokens): logits = decode_step(...)`) that any reader at this level already knows. Remove or reduce to a 2-line note.

---

## Cross-File Redundancy

| Repeated Content | Occurrences | Recommendation |
|-----------------|-------------|----------------|
| 5:1 sliding/global layer pattern with index list {5,11,17,...,59} | index.md, decoder_layer_module.md, full_model_module.md (also Ch1) | Define once in index.md; cross-reference elsewhere |
| Per-device FFN weight sizes (28.9 MB each, 86.7 MB total) | ffn_module.md, full_model_module.md (also Ch6) | Keep in ffn_module.md only; link from full_model_module.md |
| "PLE is disabled / no-op in 31B" | index.md (x2), decoder_layer_module.md (x3), ple_module.md (x4), full_model_module.md (x2) | State once prominently in index.md; mention once per file max |
| `[1, 1, 5376]` replicated shape annotation | decoder_layer_module.md (9x in table + diagram), ffn_module.md (diagram), full_model_module.md | The dataflow diagrams are fine; remove the redundant shapes table in decoder_layer_module.md |
| Column-parallel / row-parallel sharding explanation for FFN | ffn_module.md, Ch6 | Keep brief version in ffn_module.md; remove per-device byte calculations that duplicate Ch6 |

## Estimated Savings

| File | Current Lines | Estimated Reducible | Reduction |
|------|--------------|--------------------|-----------|
| index.md | 127 | ~30 | ~24% |
| decoder_layer_module.md | 325 | ~40 | ~12% |
| ffn_module.md | 274 | ~55 | ~20% |
| ple_module.md | 216 | ~100 | ~46% |
| full_model_module.md | 507 | ~50 | ~10% |
| **Total** | **1,449** | **~275** | **~19%** |
