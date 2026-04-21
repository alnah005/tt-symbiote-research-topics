# Critic Review — Chapter 3 (Pass 1)

## Issue 1 — Attention hyperparameters contradict the plan (config_diff.md, Section "Text Encoder: Core Dimensions")

`config_diff.md` states `num_attention_heads: 32` and `head_dim: 128` as values shared by both Qwen3.5 and Qwen3.6.

The plan (Ch1 architecture notes) states the Gated Attention configuration is `num_attention_heads=16, head_dim=256`.

An implementer reading Ch3 would allocate Q-projection buffers sized `[32 * 128, 2048] = [4096, 2048]` and KV-projection buffers sized `[4 * 128, 2048] = [512, 2048]`. If the correct values are `num_attention_heads=16, head_dim=256`, then the correct Q-projection is `[16 * 256, 2048] = [4096, 2048]` and KV is `[2 * 256, 2048] = [512, 2048]`. The Q-projection total width happens to be identical, but `num_key_value_heads` in the table is 4 while the plan implies 2 (Ch1: `n_kv=2`). This matters for GQA grouping logic and KV cache allocation.

The `head_dim` note is also wrong: it says `hidden_size / num_attention_heads * 2, due to GQA`. GQA does not alter head_dim. Head_dim is independently configured; the note will mislead anyone trying to derive it from other fields.

**Required fix:** Verify `num_attention_heads`, `num_key_value_heads`, and `head_dim` against the actual Qwen3.5/3.6 config.json files and correct whichever is wrong (either the config_diff.md table or the plan's Ch1 notes). Remove the incorrect GQA-based derivation formula.

---

## Issue 2 — DeltaNet head dimensions contradict the plan (config_diff.md, Section "DeltaNet Configuration")

`config_diff.md` states `deltanet_num_heads: 32` and `deltanet_head_dim: 64`.

The plan (Ch2 architecture notes) states `linear_num_key_heads=16, linear_num_value_heads=32, linear_key_head_dim=128, linear_value_head_dim=128`. The plan also defines the state matrix as `S in R^{128 x 128}` per head.

An implementer sizing the DeltaNet recurrent state would get `[B, 32, 64, 64]` from config_diff.md but `[B, 32, 128, 128]` (for V heads) from the plan — a 4x error in state memory. The projection inventory would also be wrong (e.g., out_proj input size differs by 2x).

**Required fix:** Reconcile `deltanet_num_heads` and `deltanet_head_dim` in config_diff.md against the actual config.json and the plan's Ch2 values before this chapter is used as a reference for implementation.

---

## Issue 3 — vocab_size is wrong and propagates to weight shapes (config_diff.md and post_training_differences.md)

`config_diff.md` states `vocab_size: 151936`. The plan (Ch1) states `vocab_size=248320`.

`post_training_differences.md` then lists token embedding weight shape as `[151936, 2048]` and LM head as `[151936, 2048]`, which would be wrong if the actual vocab size is 248320. These are the two largest weight tensors in the model; an implementer allocating DRAM buffers would be off by roughly 30%.

Additionally, `config_diff.md` separately lists `bos_token_id: 248044` as a new field added in Qwen3.6. A BOS token ID of 248044 exceeds a vocab_size of 151936, which is internally inconsistent and is a further signal that 151936 is wrong.

**Required fix:** Correct vocab_size and update the embedding/LM-head shapes in post_training_differences.md accordingly.

---

## Issue 4 — benchmark_comparison.md vision section names a model version that does not exist

`benchmark_comparison.md` (Vision Benchmarks section) states Qwen3.6-35B-A3B is competitive with "Claude Sonnet 4.5". As of the guide's writing context (Qwen3.6 released ~2025), the correct released model name is Claude Sonnet 3.5 or Claude Sonnet 3.7. "Claude Sonnet 4.5" is not a model version that existed at the time of these benchmarks.

A reader relying on this comparison to position Qwen3.6 against closed models would be citing a non-existent baseline.

**Required fix:** Verify the Qwen3.6 technical report or blog post for the exact model name used in the vision comparison table and correct accordingly.

---

## Issue 5 — Navigation footer in index.md points to wrong chapter title

`index.md` navigation table links to:

> Previous chapter: Chapter 2 — Hybrid Attention: DeltaNet and Full Attention

The plan names Chapter 2 "Gated DeltaNet Deep Dive" (directory `ch2_gated_deltanet/`). The linked path `../ch2_hybrid_attention_deltanet/index.md` references a directory `ch2_hybrid_attention_deltanet/` that does not exist per the plan. If that directory does not exist on disk, the link is broken and navigation fails.

**Required fix:** Change the previous-chapter link to `../ch2_gated_deltanet/index.md` with label "Chapter 2 — Gated DeltaNet Deep Dive" to match the plan's actual directory name and chapter title.

---

# Critic Review — Chapter 3 (Pass 2)

## Issue 1 — head_dim derivation note gives a false causal explanation (config_diff.md, Section "Text Encoder: Core Dimensions")

The table note for `head_dim` reads: `Per-head dimension (hidden_size / num_attention_heads · 2, due to GQA)`.

GQA (Grouped Query Attention) reduces the number of key/value heads relative to query heads — it does not expand `head_dim`. The factor of 2 in the formula has nothing to do with GQA. An implementer who reads this note and applies the same reasoning to another GQA model will compute `head_dim = (hidden_size / num_q_heads) * 2` and get a wrong value. `head_dim` is an independently configured field; in this model it happens to be 256 while `hidden_size / num_attention_heads = 128`, but the doubling is not caused by GQA and no causal explanation is given for why it is doubled.

**Required fix:** Remove the parenthetical formula and GQA attribution entirely. The note should state that `head_dim` is an explicit config field set to 256, independent of `hidden_size / num_attention_heads`.

---

## Issue 2 — `intermediate_size` is mislabeled as the shared expert FFN width (config_diff.md, Section "Text Encoder: Core Dimensions")

The identical-fields table lists `intermediate_size: 768` with the note "Shared expert FFN intermediate width."

In Qwen3 MoE configs, `intermediate_size` is the FFN intermediate width for dense (non-MoE) layers — the field that would apply to any layer listed in `mlp_only_layers`. Since `mlp_only_layers = []`, this field governs zero layers at inference. The actual shared expert width is controlled by the separate field `shared_expert_intermediate_size: 768`, which appears correctly in the MoE section below.

Both fields are 768 in this model, which masks the error, but the label is wrong. An implementer porting this to a variant where `intermediate_size` and `shared_expert_intermediate_size` diverge would assign the wrong width to the shared expert.

**Required fix:** Change the note for `intermediate_size` to "Dense/fallback FFN intermediate width (unused at inference — mlp_only_layers is empty)" to distinguish it from `shared_expert_intermediate_size`.

---

## Issue 3 — DeltaNet state matrix leading dimension is wrong (config_diff.md, Section "DeltaNet Configuration")

The document states: "The DeltaNet state matrix is shaped `[B, 32, 128, 128]` per layer (value heads × key head dim × value head dim)."

In DeltaNet, the recurrent state `S` is updated via the outer product `k_t^T v_t`, where each head maintains one state matrix of shape `[d_k, d_v]`. The number of independent state matrices equals the number of heads performing the outer-product update — which is determined by the key/query head count, not the value head count. The config specifies `linear_num_key_heads = 16` and `linear_num_value_heads = 32`. With 16 K heads, there are at most 16 independent recurrent states. The correct leading dimension for the state batch is 16, not 32.

Using 32 as the leading dimension would allocate 2× the required recurrent state memory (`[B, 32, 128, 128]` instead of `[B, 16, 128, 128]`) and would produce an incorrect outer-product update implementation. If the architecture genuinely has 32 V heads mapping to 16 K heads through a grouping mechanism, that mechanism must be documented here — without it, an implementer cannot correctly implement the state update rule.

**Required fix:** Either correct the state matrix shape to `[B, 16, 128, 128]` (if each K head has one state), or explicitly document the grouping mechanism by which 16 K heads map to 32 V heads and how that affects the per-state shape.

---

# Critic Review — Chapter 3 (Pass 3)

## Issue 1 — DeltaNet state matrix "GQA expansion" explanation is fabricated and still wrong (config_diff.md, Section "DeltaNet Configuration")

The Pass 2 fix was required to either correct the leading dimension to 16 or document the grouping mechanism. The current text attempts to document a mechanism but invents one that does not exist:

> "32 = V heads, after GQA expansion of 16 QK heads via `repeat_interleave(2)`; one state matrix R^{128×128} is maintained per expanded V head group, giving 32 states total"

GQA (Grouped Query Attention) means Q heads outnumber KV heads — KV heads are never expanded by repeating to match a larger V head count. There is no standard or documented operation in DeltaNet called "GQA expansion of QK heads via repeat_interleave" that produces additional V heads. The explanation is architecturally incoherent: if anything, GQA repeats K heads to broadcast over groups of Q heads, not to create additional V heads. An implementer reading this would implement a non-existent `repeat_interleave` expansion and allocate `[B, 32, 128, 128]` states instead of the correct shape.

**Required fix:** Remove the "GQA expansion via repeat_interleave" sentence. If the architecture genuinely uses 32 independent recurrent states (one per V head), state that directly: "32 independent state matrices are maintained, one per value head." If the correct count is 16 (one per K head), change the shape to `[B, 16, 128, 128]` and remove the expansion claim entirely.

---

## Issue 2 — "Claude Sonnet 4.5" comparison baseline was not corrected from Pass 1 (benchmark_comparison.md, Vision Benchmarks)

Pass 1 Issue 4 required verifying and correcting the model name "Claude Sonnet 4.5" used as a competitive comparison baseline in the Vision Benchmarks section. The file is unchanged: it still reads "Claude Sonnet 4.5." The fix was not applied. A reader citing this comparison to position Qwen3.6 against closed models will reference either a wrong model name or an unverified baseline, depending on whether "Claude Sonnet 4.5" existed at the time of the benchmarks.

**Required fix:** Apply the Pass 1 correction — verify the Qwen3.6 technical report for the exact closed-model name used in vision comparisons and update the text accordingly.

---

## Issue 3 — AIME 2026 benchmark scores contradict the release timeline (benchmark_comparison.md, General Reasoning Benchmarks)

The document lists AIME 2026 results (91.0 for Qwen3.5, 92.7 for Qwen3.6). AIME 2026 refers to contest problems from early 2026. The chapter's own framing states Qwen3.6 was released "~2025" (post_training_differences.md and the broader guide context). A model released in 2025 cannot have been benchmarked on AIME 2026 problems at release time.

Either (a) the benchmark year is wrong and should be AIME 2025 or earlier, or (b) these are post-release evaluation numbers obtained after the 2026 contest, in which case the text should state that explicitly. As written, a reader will conclude the model was trained or evaluated on 2026 contest problems, which would normally indicate data contamination concerns — a materially misleading implication for anyone interpreting the benchmark results.

**Required fix:** Correct the year to the actual contest year used in the benchmark, or add a note clarifying that this is a post-release evaluation on 2026 problems and the model's training cutoff predates those problems.

---

# Critic Review — Chapter 3 (Pass 4)

## Issue 1 — DeltaNet state matrix: K-to-V pairing mechanism still unexplained (config_diff.md, Section "DeltaNet Configuration")

The Pass 3 fix removed the fabricated "GQA expansion via repeat_interleave" language. The current text now reads:

> "32 = number of value heads; each of the 16 key heads is shared across 2 value heads, giving 32 (key, value) head pairs, each maintaining an independent R^{128×128} state matrix"

The GQA/repeat_interleave text is gone, which is correct. However the replacement still does not give an implementer enough information to write the state update. In standard DeltaNet the recurrent state update is `S_t = β_t S_{t-1} + k_t ⊗ v_t` where `k_t` and `v_t` are a matched pair from the same head. With `linear_num_key_heads=16` and `linear_num_value_heads=32`, the claim that each K head "is shared across 2 value heads" implies each K vector participates in two separate outer products with two different V vectors, producing two independent state matrices. That is a specific, non-default implementation choice — the text must state it explicitly: does the K projection produce 16 vectors that are each used twice (same K, different V), or are there actually 32 K projections derived from 16 shared parameters? Without this, an implementer cannot determine (a) whether to project K to shape `[B, T, 16, 128]` or `[B, T, 32, 128]`, and (b) how to index K when writing the 32 state update loops.

**Required fix:** State the pairing rule unambiguously. If each of the 16 K heads pairs with 2 distinct V heads to produce 2 independent states per K head (32 total), write: "K head `i` (for i=0..15) pairs with V heads `2i` and `2i+1`; the K projection output shape is `[B, T, 16, 128]` and each K vector is reused for two outer-product updates." If the correct count is 16 independent states (one per K head), correct the shape to `[B, 16, 128, 128]` and remove the 32-state claim.

---

# Critic Review — Chapter 3 (Pass 5)

## Issue 1 — DeltaNet Q retrieval against 32 state matrices is undocumented (config_diff.md, Section "DeltaNet Configuration")

The Pass 4 fix correctly specifies the state-update (write) side of DeltaNet: K is projected to `[B, T, 16, 128]`, expanded via `repeat_interleave(2)` to `[B, T, 32, 128]`, K head `i` pairs with V heads `2i` and `2i+1`, producing 32 independent R^{128×128} state matrices stored in `[B, 32, 128, 128]`.

The implementer note stops there. It never describes the retrieval (read) side: Q is projected to 16 heads of dimension 128 (`[B, T, 16, 128]`). With 32 state matrices and 16 Q heads, an implementer cannot write the output computation `O = Q · S` without knowing which state matrices each Q head reads from. The two most plausible rules — (a) Q head `i` reads only state matrix `2i` (ignoring `2i+1`), or (b) Q head `i` reads states `2i` and `2i+1` and concatenates or sums the results — produce different output shapes and different FLOP counts. Neither is implied by the current text.

Without this rule, the output projection shape also cannot be verified: if Q head `i` reads two states, the output before aggregation is `[B, T, 16, 2 * 128]` rather than `[B, T, 16, 128]`, changing the input width of the output linear.

**Required fix:** Add to the implementer note the Q-to-state indexing rule. For example, if the standard DeltaNet retrieval is "Q head `i` reads state `S[2i]` only, and the V-head doubling exists solely on the write side," state that explicitly. If Q head `i` retrieves from both `S[2i]` and `S[2i+1]`, document the aggregation and the resulting output shape before the output linear.

---

# Critic Review — Chapter 3 (Pass 6)

## Issue 1 — DeltaNet out_proj weight is absent from the weight shape table (post_training_differences.md, Section "Shapes and Dtypes: Identical")

The Pass 5 addition to `config_diff.md` correctly establishes that the 32 expanded head outputs are concatenated to `[B, T, 4096]` and then an `out_proj` maps `[B, T, 4096]` → `[B, T, 2048]`. This means each DeltaNet layer carries an out_proj weight tensor of shape `[4096, 2048]`.

The weight shape table in `post_training_differences.md` lists DeltaNet Q projection (`[16*128, 2048]`), K projection (`[16*128, 2048]`), and V projection (`[32*128, 2048]`), but has no row for the DeltaNet out_proj. The input width of 4096 for that weight is non-obvious — it follows from the 32-state concatenation documented in the Pass 5 implementer note, not from any single config field. An implementer building weight-loading code from this table would either omit the out_proj tensor entirely or be forced to cross-reference the implementer note in a separate file to determine the correct shape. Both outcomes produce incorrect or incomplete weight loading.

The full-attention out_proj is also absent, but its input width (`16 * 256 = 4096`) is directly derivable from the `num_attention_heads` and `head_dim` entries already in the table, so the omission there is recoverable. The DeltaNet out_proj is not recoverable from the table alone.

**Required fix:** Add a row to the weight table for "DeltaNet out_proj (per DeltaNet layer)" with shape `[32 * 128, 2048]` = `[4096, 2048]` and dtype `bfloat16`.

---

# Critic Review — Chapter 3 (Pass 7)

No feedback — chapter approved.
