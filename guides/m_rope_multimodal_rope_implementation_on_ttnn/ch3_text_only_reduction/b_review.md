## B Feedback — Pass 1

1. **`practical_implications_for_text_inference.md`, line 8–9 — Wrong tensor shape in the HuggingFace construction snippet**

   The code reads:
   ```python
   position_ids = torch.arange(S, dtype=torch.long).unsqueeze(0).repeat(3, 1)
   # shape [3, 1, S]
   ```
   `.unsqueeze(0)` on a 1-D `[S]` tensor produces `[1, S]`. `.repeat(3, 1)` then tiles along the two existing axes, producing `[3, S]` — not `[3, 1, S]`. The batch dimension is absent. The ground-truth shape is `[3, batch_size, seq_len]`; with batch=1 it must be `[3, 1, S]`. A reader who copies this snippet will get a 2-D tensor where the forward pass expects a 3-D tensor; indexing `position_ids[0]` yields `[S]` instead of `[1, S]`, breaking the gather broadcast.

   **Fix:** Replace the construction with one that includes the batch dimension:
   ```python
   position_ids = torch.arange(S, dtype=torch.long).unsqueeze(0).unsqueeze(0).repeat(3, 1, 1)
   # shape [3, 1, S]; all three rows identical — [0, 1, 2, ..., S-1]
   ```
   or equivalently `torch.arange(S).reshape(1, 1, S).expand(3, 1, S).contiguous()`.

## B Feedback Application Log — Pass 1

- Item 1: Fixed position ID tensor shape in HF snippet — changed `.unsqueeze(0).repeat(3, 1)` to `.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1)` to produce correct `[3, 1, S]` shape instead of `[3, S]`

## B Feedback — Pass 2

**Pass 1 fix verification:** The corrected snippet at `practical_implications_for_text_inference.md` line 8 now reads `.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1)`. Shape trace: `[S]` → `[1, S]` → `[1, 1, S]` → `[3, 1, S]`. Comment correctly states `# shape [3, 1, S]`. Fix is correct.

1. **`mathematical_equivalence_proof.md`, lines 58–88 — numerical snippet does not verify the full proposition it claims to confirm**

   The proposition in Section 1 (lines 16–19) states that `cos_assembled[b, s, :]` equals `cat([cos_table[t, :], cos_table[t, :]])` — a 64-wide vector. The sentence at line 58 says "The following Python snippet confirms this numerically." However, the snippet only asserts half-vector equality (`[B, S, 32]`) and never applies the duplication step `torch.cat([cos_mrope, cos_mrope], dim=-1)`. A reader who treats the snippet as a complete validation would miss that the final output must be `[B, S, 64]`, not `[B, S, 32]`. A developer implementing the gather step in isolation could copy this snippet and falsely conclude no duplication is needed.

   **Fix:** Add the duplication to the snippet and update the assertion to check the full 64-wide vector:
   ```python
   cos_mrope_full = torch.cat([cos_mrope, cos_mrope], dim=-1)  # [B, S, 64]
   cos_1d_full = torch.cat([cos_1d, cos_1d], dim=-1)           # [B, S, 64]
   assert torch.allclose(cos_mrope_full, cos_1d_full), "Must be identical for text-only inputs"
   ```
   The intermediate half-vector assertion can be retained as an additional check but should not be the only one.

## B Feedback Application Log — Pass 2

- Item 1: Extended Python validation snippet in `mathematical_equivalence_proof.md` to include the duplication step and 64-wide assertion, matching the full proposition in Section 1

## B Feedback — Pass 3

**Pass 2 fix verification:** The fix is present and correct. `mathematical_equivalence_proof.md` lines 90–93 now contain:
- `cos_1d_full = torch.cat([cos_1d, cos_1d], dim=-1)` — produces `[B, S, 64]`
- `cos_mrope_full = torch.cat([cos_mrope, cos_mrope], dim=-1)` — produces `[B, S, 64]`
- `assert torch.allclose(cos_mrope_full, cos_1d_full), "Full 64-wide cos must also be identical"`

The original 32-wide assertion at line 88 is retained as an intermediate check. Both assertions are present, correctly formed, and consistent with the proposition in Section 1.

**No feedback — chapter approved.**

## B Feedback — Pass 4

**Pass 3 approval re-verification after Agent C's structural edits:**

C made two edits: (1) removed a duplicate Key Finding blockquote from `mathematical_equivalence_proof.md`; (2) removed a redundant cross-reference sentence from `practical_implications_for_text_inference.md`.

Both edits were verified against all four files. No new correctness issues were introduced:

- `mathematical_equivalence_proof.md`: the only blockquote present is the `[SILENT FAILURE]` callout (line 120). The Key Finding blockquote correctly exists only in `practical_implications_for_text_inference.md`. All numbers, gather indices, and Python assertions are intact and correct.
- `practical_implications_for_text_inference.md`: the file ends cleanly with the standard next-file link. No broken references, no missing content, no dangling sentences. All numbers and the HF construction snippet remain correct.
- `index.md` and `mrope_section_always_active.md`: unchanged and consistent with ground truth.

**No feedback — chapter approved.**
