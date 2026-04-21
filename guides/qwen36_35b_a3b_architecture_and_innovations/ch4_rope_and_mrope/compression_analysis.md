# Compression Analysis — Pass 1
## Chapter 4: partial_rotary_embedding.md + mrope_multimodal_positions.md
**Agent C — Compressor | Date: 2026-04-21**

---

## VERDICT

**Crucial updates: no**

Both files are tight, non-redundant, and technically load-bearing throughout. No passage meets the bar for removal or significant cutting.

---

## Load-Bearing Evidence

**partial_rotary_embedding.md**

- Lines 5–6 (Scope section): "Gated DeltaNet layers … carry no positional encoding whatsoever" — establishes a hard architectural boundary repeated nowhere else in the chapter; removing it would leave the reader unable to distinguish which layers even run RoPE.
- Lines 14–21 (Dimensions table + cos/sin cache shape): Every numeric value (n_q=16, n_kv=2, d_h=256, partial_rotary_factor=0.25, rotary_dim=64, rope_theta=10M, max_seq_len=262144, cache shape [262144,64]) is load-bearing config ground truth that downstream TTNN checklist items (lines 146–150) depend on directly.
- Lines 29–43 (Head vector decomposition + math block): The explicit split `[h_rot | h_pass] ←64→ ←192→` is the first and only place this decomposition is defined; the formula block is not a restatement but the primary definition. The MRoPE file explicitly defers to "the previous section" for the rotation math (line 7), so this block is canonical.
- Lines 64–79 (Frequency spectrum table + paragraph): The four-row table shows concrete computed values (1.0, 0.604, 5.24×10⁻⁴, 1.65×10⁻⁷) and the prose derives from them the "40 million position period vs 262K context window" anti-aliasing argument — a design rationale not duplicated anywhere.
- Lines 83–88 (Why Not Full RoPE): The three numbered points give the qualitative justification for partial RoPE in Qwen3.6 specifically; this is the only place these reasons are articulated and they directly motivate the architectural choice.
- Lines 93–113 (Q/K RMSNorm section): The distinction from Gated DeltaNet L2-norm (line 113) is a common confusion point called out explicitly; without it the TTNN checklist item "Gated DeltaNet layers do not invoke any rotary kernel" loses its motivating context.
- Lines 119–134 (cos/sin cache shape + indexing): The step-by-step inference walkthrough (gather → broadcast → rotate → concatenate) and the bug-guard note about mistakenly allocating [max_seq_len, 256] buffers are implementation-critical, not decorative.
- Lines 139–150 (TTNN Deployment + checklist): Four-item verification checklist is a direct engineering artifact; each item addresses a distinct failure mode.

**mrope_multimodal_positions.md**

- Lines 5–7 (What M-RoPE Is intro paragraph): Concisely scopes M-RoPE to Gated Attention only and states that "the rotation mathematics are identical" — this cross-file deference is not a restatement but a deliberate pointer that prevents duplicating the math block.
- Lines 16–24 (mrope_section table): The three-section split [11, 11, 10] and the dimension ranges (0–21, 22–43, 44–63) are the primary definition of M-RoPE's structural parameter; they appear nowhere else.
- Lines 31–42 (φᵢ formula): The piecewise angle formula is the formal definition of M-RoPE; it cannot be cut without removing the document's core mathematical content.
- Lines 58–64 (Text Tokens subsection): Although the scalar degeneracy was already stated in the previous section (line 44–50), this sub-section adds the concrete statement "No special handling is required in the attention kernel" — an implementation consequence not derivable from the math alone.
- Lines 68–82 (Vision Tokens subsection + inner-product decay formula): The (frame f, row r, col c) → (m_t, m_h, m_w) assignment and the inner-product decomposition formula are the only explicit statements of spatial/temporal attention decay bias; they justify M-RoPE's architectural motivation.
- Lines 88–104 (mrope_interleaved = true): The contrast between interleaved and contiguous buffer layouts, and the explanation of why interleaved matches the rotation kernel's dimension ordering, is unique diagnostic content; a transposition bug in buffer construction would be silent and this is the only guard against it.
- Lines 110–112 (Vision Encoder independence): "No coupling between the vision encoder's position encoding and M-RoPE" prevents a specific architectural misunderstanding (that ViT positions propagate into decoder M-RoPE); this is stated once and is not a repeat.
- Lines 118–134 (Text-Only Inference): The [B, 3, T] tensor layout with all-identical rows and the three-step simplification for TTNN text-only inference are engineering specifications, not redundant prose. The simplification is valid by a mathematical identity argument, which needs to be stated explicitly for an implementer to trust it.
- Lines 140–146 (TTNN Deployment Summary table): The three-row table (text-only / text+image / text+video) with Code delta column concisely maps inference modes to implementation effort; it is a decision aid not reducible to any single prior passage.

---

## MINOR Suggestions

1. **partial_rotary_embedding.md, lines 54–58 (complex-multiplication restatement):** The standard 2×2 rotation matrix on lines 49–52 is already fully self-explanatory. The immediately following sentence "which is equivalently written using complex multiplication" plus the Euler form on lines 57–58 adds a notation variant that is not used anywhere else in either file. Consider dropping lines 54–58 (the equivalence sentence and the complex-number formula) to save three lines without any information loss for the target audience (TTNN implementers who work with real-valued tensors, not complex arithmetic).

2. **mrope_multimodal_positions.md, lines 58–64 (Text Tokens subsection):** The content of this subsection is entirely derivable from the "For text tokens" paragraph on lines 44–50 of the same file, and the only addendum — "No special handling is required in the attention kernel" — could be appended as a single sentence to that earlier paragraph. Merging would eliminate a redundant sub-heading and shorten the document by ~5 lines.

3. **mrope_multimodal_positions.md, lines 93–94 (interleaved buffer example, repeated cos values):** The example line `[cos(phi_0), cos(phi_0), cos(phi_1), cos(phi_1), ...]` uses duplicate entries to represent real/imaginary pairing, which requires the parenthetical label `pair 0 real   pair 0 imag` below it to prevent confusion. Replacing the duplicated-value notation with a cleaner `[cos(φ₀)|real, cos(φ₀)|imag, cos(φ₁)|real, ...]` inline annotation would halve the visual noise in this block without changing meaning.
