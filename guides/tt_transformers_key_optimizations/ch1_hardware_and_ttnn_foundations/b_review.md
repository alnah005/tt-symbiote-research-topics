# B Review — Chapter 1: Hardware and TTNN Foundations — Pass 1

1. **`math_fidelity_and_data_formats.md`, line 11** — Incorrect statement of multiplier output width. The file says "The hardware multiplier provides 5 × 7 = 35 bits of product per pass." A 5-bit × 7-bit integer multiplier produces a product that is at most 12 bits wide (5 + 7 = 12), not 35. The value "35" is the arithmetic product of 5 and 7 (i.e., 5 × 7 = 35 as a number), but "bits of product" for a multiplier refers to the output bit-width, which is 12. A reader trying to reason about precision coverage from this sentence will reach wrong conclusions. Fix: replace "5 × 7 = 35 bits of product per pass" with "5-bit × 7-bit operands, yielding a 12-bit product per pass."

2. **`tensix_architecture.md`, line 15** — Incorrect RISC-V processor count. The file states "Two lightweight RISC-V control processors (one handling data movement, one controlling compute sequencing)." A Tenstorrent Tensix core contains five RISC-V processors: one BRISC (data movement / host interface), one NCRISC (NoC DMA), and three TRISC cores (TRISC0, TRISC1, TRISC2 for the compute pipeline stages). Stating "two" gives a reader an incorrect mental model of kernel dispatch and pipelining, and directly conflicts with the TT-Metalium programming guide that assigns Reader, Writer, and Compute kernels to separate RISC-V cores. Fix: update to "five RISC-V processors (BRISC and NCRISC for data movement; TRISC0, TRISC1, TRISC2 for compute pipeline stages)."

3. **`ttnn_tensor_model.md`, line 41** — Inconsistent face-primitive description. The file says the face layout "matches the order in which the matrix FPU consumes them for its 8×16 × 16×16 sequential face multiplications." Earlier in `tensix_architecture.md` (lines 25–26), SrcA and SrcB faces are both described as 16×16. An "8×16 × 16×16" primitive is inconsistent with those descriptions: if the face is 16×16 and SrcB is 16×16 then the primitive is 16×16 × 16×16, not 8×16 × 16×16. A reader implementing a tiling scheme using the stated 8×16 dimension will pad or block incorrectly. Fix: reconcile the face primitive description across both files — if the hardware FPU array is physically 8 rows wide and processes a 16×16 face in two 8-row passes, state that explicitly; otherwise correct "8×16" to "16×16."

4. **`math_fidelity_and_data_formats.md`, line 25** — HiFi2 mantissa-bits-covered value is physically impossible. The table states HiFi2 covers "Upper ~10 mantissa bits." BF16 has only 7 mantissa bits. Covering 10 mantissa bits of a 7-bit mantissa field is not physically meaningful and will confuse a reader trying to reason about why HiFi2 is sufficient for BFP8 weights. The correct statement is that 2 passes of the 5-bit sub-multiplier cover the upper 5+5=10 product-contribution bits, but the source mantissa field is 7 bits, so HiFi2 covers all 7 meaningful mantissa bits of a BFP8 weight. Fix: replace "Upper ~10 mantissa bits" with a description consistent with the 7-bit BF16 mantissa — e.g., "Covers all meaningful bits of a 7-bit mantissa operand (BFP8/BF16); 2 passes of the 5-bit sub-multiplier suffice."

5. **`math_fidelity_and_data_formats.md`, line 11** — Companion error: "To cover the full 7-bit × 7-bit mantissa multiplication at full precision, the hardware must do four passes (feeding progressively lower-significance mantissa sub-groups in each pass)." If the sub-multiplier is 5-bit × 7-bit per pass, two passes (each covering different 5-bit windows of the 7-bit input mantissa) would be sufficient to cover all 7 bits — yet the file claims four passes are needed for HiFi4. The stated rationale for why four passes are required is not derived correctly from the 5×7 multiplier description. A reader following this reasoning to understand the fidelity system will reach an incorrect understanding of how mantissa coverage maps to pass count. Fix: either provide the correct derivation of why 4 passes are needed (e.g., both operands require multi-pass coverage, or the sub-multiplier width differs from what is stated), or correct the pass-count rationale to match the hardware spec.

---

# B Review — Chapter 1: Hardware and TTNN Foundations — Pass 2

1. **`math_fidelity_and_data_formats.md`, line 32** — LoFi precision description uses "each" incorrectly. The file states "only the top 5 bits of **each** BF16 mantissa (7 bits total) participate in the multiplication. The bottom 2 bits are dropped." The word "each" implies both operands are truncated to 5 bits (a 5×5 effective multiply). However, the hardware multiplier is defined throughout this file as **5-bit × 7-bit**: one operand contributes 5 bits and the other contributes 7 bits per pass. A LoFi pass is therefore 5×7, not 5×5. Stating "each mantissa" loses 2 bits from one operand that the hardware actually retains, giving the reader a precision model that is inaccurate in one direction. A reader designing a custom kernel or reasoning about LoFi error bounds for a 7-bit operand will reach wrong conclusions about which operand side is truncated and by how much.

2. **`math_fidelity_and_data_formats.md`, lines 24 and 176** — Internal contradiction between the HiFi2 table entry and the Key Pairing Rule for BF16 operands. The fidelity table (line 24) says HiFi2 "Covers all meaningful bits of a 7-bit mantissa operand (BFP8/BF16); 2 passes of the 5-bit sub-multiplier suffice." The Key Pairing Rule section (line 176) says "BF16 × BF16 (7-bit × 7-bit mantissa) → HiFi4 (4 passes) for full precision," and the decision-guide table lists HiFi4 for "Attention score QK^T (prefill)" which uses BF16 activations. If HiFi2 already covers all meaningful bits of a BF16 operand, there is no basis to require HiFi4 for BF16×BF16. A reader following the table will believe HiFi2 suffices for all BF16 work; a reader following the pairing rule will use HiFi4. The two sections give contradictory guidance for the same case. The underlying issue is that the HiFi2 cell's claim that 2 passes "suffice" for BF16 is too broad — it covers the bits of one operand but not the full cross-product of two 7-bit operands, which requires HiFi4.

3. **`tensix_architecture.md`, line 96** — Aggregate NoC bandwidth figure is implausibly large and will mislead bottleneck analysis. The file claims "Total intra-chip NoC bandwidth is in the range of ~**8–16 TB/s** aggregate (across all links simultaneously)." The same file states Wormhole DRAM aggregate bandwidth is ~256–300 GB/s. A 8–16 TB/s NoC vs ~300 GB/s DRAM would imply the NoC has 25–50× more bandwidth than DRAM, making it essentially impossible for any workload to be NoC-bound. Wormhole has a grid of ~80 cores; at 32 bytes/cycle per link at 1 GHz, and counting two bidirectional NoC meshes, peak theoretical aggregate is on the order of hundreds of GB/s to low single-digit TB/s — not 8–16 TB/s. A reader using the stated 8–16 TB/s figure to determine whether a kernel is NoC-bound or DRAM-bound will systematically under-attribute NoC cost.

4. **`ttnn_tensor_model.md`, line 229** — BFP8_B per-value bit layout description is self-contradictory. The file says "each value has a 7-bit mantissa (plus 1 sign bit = 8 bits total per value **after including the shared exponent overhead**)." The parenthetical is wrong: 7 mantissa bits + 1 sign bit = 8 bits per value *before* adding any shared exponent cost. The shared exponent (8 bits per block of 16 values = 0.5 bits/value overhead) is a separate field, not part of the per-value 8 bits. Saying "8 bits total after including the shared exponent overhead" incorrectly implies the shared exponent is somehow absorbed into the per-value 8-bit field, which contradicts the memory footprint table on line 250 that correctly shows BFP8_B as ~8.5 bits/element (8 per-value bits + 0.5 exponent overhead). A reader implementing BFP8_B serialization or computing buffer sizes from the narrative description will get the per-value and per-block structure wrong.

---

## Agent A Change Log — Pass 1 Fix

Applied all five feedback items from Agent B Pass 1 review. Changes made on 2026-03-24:

**`math_fidelity_and_data_formats.md`**

- **Item 1 (line ~11, product width):** Replaced "5 × 7 = 35 bits of product per pass" with the correct statement that a 5-bit × 7-bit multiplier yields a 12-bit product per pass (5 + 7 = 12).

- **Item 5 (line ~11, pass-count rationale):** Replaced the incorrect derivation ("four passes to cover 7-bit × 7-bit") with a qualitative but accurate description: the hardware performs multiple passes (LoFi=1, HiFi2=2, HiFi3=3, HiFi4=4), each contributing partial products at successively lower significance, as defined by Tenstorrent hardware. Avoided stating an unsupported specific derivation.

- **Item 4 (line ~25, HiFi2 mantissa coverage):** Replaced the physically impossible "Upper ~10 mantissa bits" for HiFi2 with "Covers all meaningful bits of a 7-bit mantissa operand (BFP8/BF16); 2 passes of the 5-bit sub-multiplier suffice." Updated LoFi to "~3–4 bits" and HiFi3 to "~5–6 bits with additional partial-product coverage" to be consistent with the 7-bit BF16 mantissa reality. Also fixed the Key Pairing Rule section which repeated the "~10 bits" error.

**`tensix_architecture.md`**

- **Item 2 (line ~15, RISC-V processor count):** Replaced "Two lightweight RISC-V control processors (one handling data movement, one controlling compute sequencing)" with the correct five-processor description: BRISC (Reader kernel), NCRISC (Writer kernel), TRISC0/TRISC1/TRISC2 (Compute pipeline stages). Also updated the Key Takeaways section which repeated the "two RISC-V control processors" error.

**`ttnn_tensor_model.md`**

- **Item 3 (line ~41, face primitive dimensions):** Replaced "8×16 × 16×16 sequential face multiplications" with "16×16 × 16×16 sequential face multiplications", consistent with the 16×16 face primitive described in both tensix_architecture.md (SrcA/SrcB both 16×16) and throughout ttnn_tensor_model.md itself.

---

## Agent A Change Log — Pass 2 Fix

Applied all four feedback items from Agent B Pass 2 review. Changes made on 2026-03-24:

**`math_fidelity_and_data_formats.md`**

- **Item 1 (line ~32, LoFi truncation):** Replaced "only the top 5 bits of each BF16 mantissa (7 bits total) participate in the multiplication. The bottom 2 bits are dropped." with an accurate description: LoFi is a single 5-bit × 7-bit pass — one operand contributes its top 5 mantissa bits (bottom 2 dropped), while the other retains its full 7-bit mantissa. Removed the incorrect implication of a 5×5 effective multiply.

- **Item 2a (line ~24, HiFi2 table entry):** Updated the HiFi2 "Mantissa bits covered" cell from "Covers all meaningful bits of a 7-bit mantissa operand (BFP8/BF16); 2 passes of the 5-bit sub-multiplier suffice" to a statement that clarifies HiFi2 is sufficient when one operand is BFP8/BFP4, but HiFi4 is needed when both operands are BF16 — resolving the internal contradiction with the Key Pairing Rule.

- **Item 2b (line ~176, Key Pairing Rule):** Expanded the BFP8 and BF16×BF16 bullet points to explicitly explain why: for BFP8 × BF16 only one operand drives the precision requirement (so HiFi2 suffices), whereas for BF16 × BF16 both operands carry full 7-bit mantissas and the complete cross-product requires HiFi4. Directly cited the attention QK^T prefill case as an example.

**`tensix_architecture.md`**

- **Item 3 (line ~96, NoC bandwidth):** Removed the implausible "~8–16 TB/s" aggregate NoC bandwidth claim (which implied a 25–50× ratio over the stated ~256–300 GB/s DRAM bandwidth). Replaced with a qualitative statement: aggregate intra-chip NoC bandwidth is significantly higher than DRAM bandwidth, making it practical to saturate DRAM without becoming NoC-bound; a precise figure is architecture-version-dependent and should come from official Tenstorrent documentation.

**`ttnn_tensor_model.md`**

- **Item 4 (line ~229, BFP8_B per-value bits):** Corrected the parenthetical from "8 bits total per value after including the shared exponent overhead" (wrong — the shared exponent is a separate field, not included in the per-value 8 bits) to "7-bit mantissa + 1 sign bit = 8 bits per value (before any shared exponent cost), with the shared 8-bit exponent adding ~0.5 bits/value overhead on top, giving ~8.5 bits/element total." This is now consistent with the memory footprint table showing BFP8_B as ~8.5 bits/element.

---

# B Review — Chapter 1: Hardware and TTNN Foundations — Pass 3

1. **`math_fidelity_and_data_formats.md`, Key Pairing Rule (line ~175)** — The stated rationale for why HiFi2 suffices for BFP8 × BF16 but not for BF16 × BF16 is logically unsound. The text claims: "the activation's 7-bit mantissa is covered because only one operand side drives the precision requirement when the other is already quantized." This is not a valid hardware argument: BFP8_B has exactly the same per-value mantissa width as BF16 — both are 7 bits. BFP8_B differs from BF16 in that it uses a shared block exponent, not in that it has fewer mantissa bits. A reader who accepts this explanation and applies the pairing rule to a different pair of types — e.g., a BF16 weight with a 7-bit mantissa multiplied by a BFP8 activation — will get the wrong answer, because the "one side drives precision" logic does not follow from a mantissa-width argument. The true reason HiFi2 is empirically sufficient for BFP8 × BF16 in LLM inference is that the effective dynamic range per block is already constrained by the shared exponent encoding (reducing worst-case rounding impact), not because one operand's mantissa is narrower. Fix: remove or replace the "only one operand side drives the precision requirement when the other is already quantized" sentence with either a correct hardware rationale or an explicit statement that the HiFi2 sufficiency for BFP8 × BF16 is empirically validated rather than derived from mantissa bit-width logic.

2. **`math_fidelity_and_data_formats.md`, LoFi description (line ~32)** — The LoFi section states: "one operand contributes its top 5 mantissa bits (the bottom 2 bits of its 7-bit mantissa are dropped), while the other operand retains its full 7-bit mantissa." This asymmetry is left unexplained and unattributed: the reader has no way to know which operand (the weight or the activation) is the one truncated to 5 bits in the hardware's 5×7 multiplier. For a reader trying to reason about which operand incurs precision loss under LoFi — which is critical for deciding whether to use LoFi when activations rather than weights are the precision-sensitive values — the missing identification of which operand maps to the 5-bit side versus the 7-bit side is a conceptual gap that can lead to wrong implementation decisions.

---

## Agent A Change Log — Pass 3 Fix

Applied both feedback items from Agent B Pass 3 review. Changes made on 2026-03-24:

**`math_fidelity_and_data_formats.md`**

- **Item 1 (line ~175, Key Pairing Rule — BFP8 × BF16 rationale):** Removed the factually wrong claim that "only one operand side drives the precision requirement when the other is already quantized." Replaced with a two-part correction: (a) explicitly noted that BFP8_B and BF16 share the same per-value mantissa width (7 bits each), so the distinction is not mantissa narrowness but the shared block exponent structure; (b) stated that the shared block exponent constrains effective dynamic range per block, reducing worst-case rounding impact and making HiFi4's extra passes offer diminishing returns. Acknowledged that the exact hardware mechanism is not publicly confirmed in full detail and directed readers to treat HiFi2 sufficiency for BFP8 × BF16 as an empirically validated guideline (per Tenstorrent's PERF.md) rather than a first-principles derivation.

- **Item 2 (line ~32, LoFi description — truncated operand not identified):** Added an explicit statement that the specific operand-side assignment (which of the two operands maps to the 5-bit side of the 5×7 pass) is not definitively confirmed in public documentation. Provided a practical guideline in place of the missing confirmation: LoFi is safe when the weight has low precision (BFP4, with only a 3-bit mantissa), regardless of activation precision, because the weight's quantization already constrains representable values far below what the 5-bit truncation would affect.

---

# B Review — Chapter 1: Hardware and TTNN Foundations — Pass 4

1. **`math_fidelity_and_data_formats.md`, "What Passes Mean for Accuracy" subsection (line ~36)** — The sentence "the weight itself only has 7-bit mantissa, so the extra two passes of HiFi4 are covering mantissa bits that do not exist in the operand" is factually wrong. BFP8_B has exactly 7 mantissa bits per value — identical to BF16. HiFi4's extra passes are not covering non-existent bits; the per-value mantissa width is the same for BFP8_B and BF16. The Pass 3 fix correctly corrected this logic in the Key Pairing Rule section (line ~175), but this sentence in the earlier "What Passes Mean" subsection was left uncorrected and still asserts the false premise. A reader using this sentence to understand why HiFi2 suffices for BFP8 will absorb an incorrect explanation that contradicts the corrected text later in the same file. Fix: replace the erroneous clause with the correct reasoning: HiFi2 suffices for BFP8 × BF16 because BFP8's shared block exponent constrains effective dynamic range per block (as stated in the corrected Key Pairing Rule section), not because BFP8 has fewer mantissa bits than BF16.

2. **`math_fidelity_and_data_formats.md`, fidelity table, HiFi2 row (line ~24)** — The "Mantissa bits covered" cell states HiFi2 is "sufficient when one operand is BFP8 or BFP4 (only one side needs full coverage)." For BFP4 (3-bit mantissa) this framing is defensible, but for BFP8 it applies the same "only one side needs full coverage" logic — which the Pass 3 fix explicitly corrected in the Key Pairing Rule section by stating that BFP8_B has the same 7-bit per-value mantissa as BF16. The table cell was not updated and now contradicts the corrected Key Pairing Rule text in the same file. A reader who reads the table before the Key Pairing Rule section will internalize the wrong explanation and not have it corrected unless they read to the end. Fix: update the HiFi2 table cell to remove "only one side needs full coverage" and instead reference the empirically validated guideline (consistent with the Key Pairing Rule fix), distinguishing BFP4 (genuinely fewer mantissa bits) from BFP8 (same per-value mantissa width as BF16, but shared block exponent constrains dynamic range).

3. **`ttnn_tensor_model.md`, Further Reading section** — The link given for "Tenstorrent block floating-point format specification" points to `tt_metal/include/tt_metal/third_party/cpp-tqdm/`. `cpp-tqdm` is a C++ progress bar library; it contains no BFP format specification. A reader following this reference to understand BFP8_B or BFP4_B encoding will find completely unrelated code. Fix: replace with the correct path to the BFP format documentation or the relevant header files (e.g., `tt_metal/include/tt_metal/third_party/tt_llk_new_api/` or the `tt_dtype.h` definitions), or remove the reference if no stable public path exists.

---

## Agent A Change Log — Pass 4 Fix

Applied all three feedback items from Agent B Pass 4 review. Changes made on 2026-03-24:

**`math_fidelity_and_data_formats.md`**

- **Item 1 (line ~36, "What Passes Mean for Accuracy" — false claim about BFP8 mantissa):** Replaced the sentence "the weight itself only has 7-bit mantissa, so the extra two passes of HiFi4 are covering mantissa bits that do not exist in the operand" — which falsely implied BFP8_B has fewer mantissa bits than BF16 — with the correct reasoning: BFP8_B has the same 7-bit per-value mantissa as BF16, and HiFi4's extra passes provide diminishing returns because BFP8_B's shared block exponent already constrains effective dynamic range per block, reducing worst-case rounding impact. This is now consistent with the Pass 3 fix to the Key Pairing Rule section.

- **Item 2 (line ~24, fidelity table HiFi2 row — "only one side needs full coverage"):** Removed the "only one side needs full coverage" framing that implied BFP8 has narrower mantissa bits than BF16. Replaced with a description that: (a) states HiFi2 is empirically validated as sufficient for BFP8 × BF16 workloads per Tenstorrent PERF.md; (b) notes that BFP8_B has the same 7-bit per-value mantissa as BF16, with the practical difference being the shared block exponent; (c) retains the accurate statement that BFP4 (3-bit mantissa) is even more clearly handled by HiFi2; (d) still correctly notes HiFi4 is required for BF16 × BF16. This is now consistent with the corrected Key Pairing Rule section.

**`ttnn_tensor_model.md`**

- **Item 3 (Further Reading — wrong link pointing to cpp-tqdm progress bar library):** Replaced the false reference to `tt_metal/include/tt_metal/third_party/cpp-tqdm/` (a C++ progress bar library with no BFP content) with a correct reference pointing to `tt_metal/include/tt_metal/common/bfloat16.hpp` and related headers as the authoritative source for BFP8_B and BFP4_B encoding, with a note that no single dedicated public specification document exists.

---

# B Review — Chapter 1: Hardware and TTNN Foundations — Pass 5

1. **`ttnn_tensor_model.md`, Further Reading (line 294)** — The Pass 4 fix replaced the broken `cpp-tqdm` link with `tt_metal/include/tt_metal/common/bfloat16.hpp`, described as the "authoritative source for BFP8_B and BFP4_B encoding." This is still wrong: `bfloat16.hpp` is a BF16 (standard Brain Float 16) helper — it handles the 16-bit IEEE-like float, not Tenstorrent's block floating-point packing format. BFP8_B and BFP4_B encoding (the block structure: 16 values sharing one 8-bit exponent, per-value mantissa packing) is defined in separate headers such as `bfloat8.hpp` and `bfloat4.hpp` (or equivalent tt-metal dtype infrastructure files), not in `bfloat16.hpp`. A reader following the cited reference to understand BFP8_B block serialization or to compute buffer sizes will find a completely unrelated file. Fix: replace `bfloat16.hpp` with the correct headers for block floating-point types (e.g., `tt_metal/include/tt_metal/common/bfloat8.hpp` and `tt_metal/include/tt_metal/common/bfloat4.hpp`), or if those paths are unstable, remove the specific filename claim and describe the reference as "BFP dtype headers in `tt_metal/include/tt_metal/common/`" to avoid pointing readers to a demonstrably wrong file.

---

## Agent A Change Log — Pass 5 Fix

Applied the single feedback item from Agent B Pass 5 review. Changes made on 2026-03-24:

**`ttnn_tensor_model.md`**

- **Item 1 (line 294, Further Reading — wrong header reference for BFP8_B/BFP4_B):** Replaced the incorrect reference to `tt_metal/include/tt_metal/common/bfloat16.hpp` (which is the standard BF16 helper, unrelated to block floating-point encoding) with the correct headers: `tt_metal/include/tt_metal/common/bfloat8.hpp` and `tt_metal/include/tt_metal/common/bfloat4.hpp`. Added an explicit clarification that `bfloat16.hpp` covers standard BF16 and does not contain the block floating-point format definitions. Also added a note that file names should be verified against the current repository, as paths may change across versions.

---

# B Review — Chapter 1: Hardware and TTNN Foundations — Pass 6

No feedback — chapter approved.

---

# B Review — Chapter 1: Hardware and TTNN Foundations — Pass 7

1. **`math_fidelity_and_data_formats.md`, fidelity table, HiFi3 row (line 25)** — The stated relative throughput for HiFi3 is "~1.33× vs HiFi4." This is internally inconsistent with the TOPS figures in `tensix_architecture.md` (lines 151–154), which list HiFi3 at ~111 TOPS and HiFi4 at ~74 TOPS. The ratio 111 / 74 ≈ 1.50×, not 1.33×. A reader using the 1.33× figure to estimate HiFi3 throughput from a known HiFi4 baseline will underestimate HiFi3 by roughly 12% of HiFi4 throughput. Fix: update the HiFi3 relative throughput cell from "~1.33×" to "~1.5×" to match the TOPS values already present in the chapter.

---

## Agent A Change Log — Pass 7 Fix

Applied the single feedback item from Agent B Pass 7 review. Changes made on 2026-03-24:

**`math_fidelity_and_data_formats.md`**

- **Item 1 (fidelity table, HiFi3 row — wrong relative throughput):** Changed the HiFi3 relative throughput from "~1.33× vs HiFi4" to "~1.5× vs HiFi4". The corrected value is consistent with the TOPS figures elsewhere in the chapter: HiFi3 ≈ 111 TOPS and HiFi4 ≈ 74 TOPS give a ratio of 111 / 74 ≈ 1.50×. The previous "~1.33×" value would have caused readers to underestimate HiFi3 throughput by approximately 12% of HiFi4 baseline.

---

# B Review — Chapter 1: Hardware and TTNN Foundations — Pass 8

1. **`math_fidelity_and_data_formats.md`, Decision Guide table, RMSNorm/LayerNorm row (line ~168)** — The recommended fidelity is listed as "HiFi4 (or HiFi2)", but the rationale in the same row states "Element-wise only (SFPU); `math_approx_mode=True` reduces rsqrt cost more than fidelity choice." Math fidelity governs the number of passes through the 5-bit × 7-bit matmul multiplier; it has no effect on SFPU operations (rsqrt, exp, silu, etc.). Because RMSNorm/LayerNorm are purely element-wise SFPU ops, the fidelity setting in the config is irrelevant to their throughput or accuracy. A reader following this table row will tune fidelity for norm layers expecting a throughput effect that cannot occur, and will also form a wrong mental model that fidelity controls SFPU throughput. Fix: replace the "HiFi4 (or HiFi2)" fidelity recommendation with a note such as "Any (fidelity does not affect SFPU ops)" and make explicit in the rationale that the only meaningful knob for RMSNorm performance is `math_approx_mode`, not `math_fidelity`.

2. **`math_fidelity_and_data_formats.md`, `packer_l1_acc` step-by-step description (lines ~86–92)** — The default-mode steps describe the Packer as overwriting and losing the previous partial sum on each K-block iteration (step 4: "Packer writes → L1 output buffer (overwrite, losing previous partial sum)"). This is factually wrong. Without `packer_l1_acc`, TTNN matmul kernels do not simply discard partial sums; the kernel accumulates across K-blocks using Dst and only writes to L1/DRAM at subblock boundaries without destroying the running total. The actual problem `packer_l1_acc` solves is avoiding DRAM round-trips for the partial sum when K-dimension tile blocking requires multiple Packer write events — the intermediate partial sum would otherwise be written to DRAM and read back, whereas `packer_l1_acc` keeps it in L1. As written, a reader who builds a custom kernel based on this description will believe that accumulating over K tiles without `packer_l1_acc` is impossible, which is incorrect. Fix: replace the step-by-step "losing previous partial sum" description with an accurate account: without `packer_l1_acc`, partial sums between K-block iterations are flushed to DRAM (slow); with `packer_l1_acc`, the Packer accumulates directly in L1 (L1 += Dst), keeping the running total in fast local memory and eliminating the DRAM round-trip.

---

## Agent A Change Log — Pass 8 Fix

Applied both feedback items from Agent B Pass 8 review. Changes made on 2026-03-24:

**`math_fidelity_and_data_formats.md`**

- **Item 1 (Decision Guide table, RMSNorm row — wrong fidelity recommendation):** Replaced the "HiFi4 (or HiFi2)" fidelity recommendation for the RMSNorm/LayerNorm row with "Any (fidelity does not affect SFPU ops)". Updated the rationale cell to explicitly state that math fidelity controls the matmul multiplier pass count and has no effect on SFPU operations (rsqrt, exp, etc.), and that the only meaningful performance knob for RMSNorm/LayerNorm is `math_approx_mode=True`. This prevents readers from incorrectly tuning fidelity for norm layers expecting a throughput effect that cannot occur.

- **Item 2 (packer_l1_acc step-by-step description — "losing previous partial sum" implies broken K-accumulation):** Corrected three locations where the incorrect "overwriting/losing partial sum" framing appeared:
  1. The `packer_l1_acc` field description (line ~77): replaced "overwriting the previous value" framing with an accurate account that without the flag, intermediate partial sums are flushed to DRAM (expensive round-trip), not lost. Explicitly labeled `packer_l1_acc` as a throughput optimization, not a correctness requirement.
  2. The "packer_l1_acc in Depth" step-by-step walkthrough (lines ~86–92): removed step 4 language "overwrite, losing previous partial sum" and replaced with an accurate description of the DRAM round-trip cost. Added an explicit note that accumulation across K-blocks is not broken without the flag — TTNN kernels correctly accumulate using the Dst register; the problem is the expensive DRAM flush-and-reload between write events.
  3. The Key Takeaways bullet for `packer_l1_acc`: updated to clarify that the flag eliminates DRAM round-trips (a throughput concern) and is not a correctness requirement.

---

# B Review — Chapter 1: Hardware and TTNN Foundations — Pass 9

1. **`math_fidelity_and_data_formats.md`, `packer_l1_acc` field description (line 77) vs. "When `packer_l1_acc` helps most" (line 108) — direct contradiction about output buffer location.** Line 77 states that without `packer_l1_acc` partial sums are "flushed to DRAM and must be read back." Line 108 then states the flag helps most when "The output buffer is in L1 (not DRAM)." These two statements are mutually contradictory: if the output buffer is in L1, there is no DRAM flush to eliminate. The Pass 8 fix introduced this contradiction by changing the packer description to say "flushed to DRAM" while line 108 continued to specify L1 as the prerequisite. A reader following line 77 will believe `packer_l1_acc` is for DRAM-backed output buffers; line 108 tells them the opposite. Fix: reconcile the two sections. The accurate description is that `packer_l1_acc` is used when the output buffer is in L1, and it enables the Packer to accumulate in-place (L1 += Dst) instead of reading the existing L1 value, adding Dst, and writing back — eliminating that read-modify-write overhead. Remove the "flushed to DRAM" framing from line 77, which only applies when the output buffer is in DRAM (a different scenario where `packer_l1_acc` does not apply).

2. **`math_fidelity_and_data_formats.md`, `packer_l1_acc` in Depth section (line 102) — wrong claim that `packer_l1_acc` enables larger `in0_block_w` without exceeding Dst capacity.** Line 102 states: "This enables larger `in0_block_w` values (the K-dimension tile block size in the program config) without exceeding Dst capacity." This is factually wrong on two counts. First, `in0_block_w` is constrained by L1 storage capacity for input tiles in the SrcA/SrcB circular buffers — not by Dst capacity. Dst capacity constrains `out_subblock_h × out_subblock_w` (the output subblock dimensions), which is a separate parameter. Second, `packer_l1_acc` affects Packer output accumulation behavior; it has no effect on Dst capacity or the constraint on `in0_block_w`. A reader acting on this statement will misattribute the `in0_block_w` limit to Dst and believe `packer_l1_acc` relaxes it, leading to incorrect program config tuning. Fix: remove the Dst-capacity claim from this sentence. The correct statement is that `packer_l1_acc` allows the output partial sum to be held in L1 across K-block iterations, reducing NoC traffic; any `in0_block_w` sizing benefit comes separately from L1 capacity management, not from Dst capacity.

3. **`math_fidelity_and_data_formats.md`, Fidelity × Throughput Tradeoff table (line 189) vs. explanatory text (line 192) — internal inconsistency in BFP8/HiFi2 throughput figure.** The table at line 189 shows "BFP8 weights, HiFi2 → ~21 t/s/u." The explanatory sentence at line 192 states "the jump from BFP8/HiFi2 to BFP4/LoFi on MLP weights accounts for approximately +22% on its own (23 → 28 t/s/u)," implying BFP8/HiFi2 gives ~23 t/s/u as the starting point. The same configuration (BFP8/HiFi2 on Llama 3.1 8B, N150) is cited as both ~21 and ~23 t/s/u within the same section. A reader calculating the BFP8→BFP4 throughput gain will get a different answer depending on which number they use (28/21 ≈ +33% vs 28/23 ≈ +22%). Fix: make the two figures consistent. The plan's terminology section and the `ttnn_tensor_model.md` cite "23 → 28 t/s/u" as the source data point; the table value of ~21 t/s/u should be corrected to ~23 t/s/u to match.

## Agent A Change Log — Pass 9 Fix

Applied all three feedback items from Agent B Pass 9 review.

**`math_fidelity_and_data_formats.md`**

- **Item 1 (packer_l1_acc contradiction — "flushed to DRAM" vs "output buffer in L1"):** Rewrote the `packer_l1_acc` field description (line ~77) to remove the "flushed to DRAM / read back" framing entirely. Replaced with an accurate L1-only account: the flag applies when the output buffer is in L1, and without it the Packer performs a read-modify-write (read existing L1 partial sum, add Dst, write back), incurring an extra L1 read per K-block write event; with the flag the Packer accumulates in-place (L1 += Dst), eliminating that overhead. Updated the corresponding paragraph, the step-by-step walkthrough note (line ~92), and the Key Takeaways bullet to use consistent L1 read-modify-write framing throughout — no DRAM mentioned, in alignment with line 108's "output buffer is in L1" prerequisite.
- **Item 2 (in0_block_w / Dst capacity claim — incorrect):** Removed the incorrect sentence claiming "`packer_l1_acc` enables larger `in0_block_w` values without exceeding Dst capacity." Replaced with a correct statement: `packer_l1_acc` keeps the output partial sum in L1 across K-block iterations, reducing NoC traffic. Added an explicit clarification that `in0_block_w` is constrained by L1 storage for SrcA/SrcB input tiles (not Dst capacity), that Dst capacity separately constrains `out_subblock_h × out_subblock_w`, and that `packer_l1_acc` has no effect on either of those constraints.
- **Item 3 (BFP8/HiFi2 throughput table ~21 → ~23 t/s/u):** Updated the Fidelity × Throughput Tradeoff table row "BFP8 weights, HiFi2" from "~21 t/s/u" to "~23 t/s/u" to match the inline explanatory text (line ~192) which uses 23 → 28 t/s/u as the source data point. The table and text now cite the same figure for this configuration, making the stated +22% BFP8→BFP4 gain internally consistent.

---

# B Review — Chapter 1: Hardware and TTNN Foundations — Pass 10

1. **`math_fidelity_and_data_formats.md`, line 102 — "reducing NoC traffic" / "NoC overhead" is factually wrong for `packer_l1_acc`.** Line 102 states: "This keeps the output partial sum in L1 across K-block iterations, reducing NoC traffic. The result is a meaningful reduction in NoC overhead for large-K matmuls." The `packer_l1_acc` optimization is entirely intra-core: the Packer reads the L1 output buffer, adds the Dst tile value, and writes the result back — all within the same Tensix core's own L1 SRAM. None of that involves the NoC. The NoC connects different cores and cores to DRAM controllers; an intra-core Packer read of its own core's L1 does not traverse the NoC at all. The overhead being eliminated is an extra **L1 read** per K-block write event (the read half of the read-modify-write cycle), which is correctly identified as "extra L1 read overhead" on the very same line 92. Calling it "NoC traffic" and "NoC overhead" on line 102 contradicts both the hardware architecture (NoC is inter-core, not intra-core) and the accurate description on line 92. A reader building a mental model of where `packer_l1_acc` reduces latency will incorrectly attribute the benefit to inter-core communication rather than intra-core L1 access, causing misdiagnosis when profiling. Fix: replace "reducing NoC traffic" and "NoC overhead" with the correct characterization — eliminating the extra intra-core L1 read (the read-modify-write read overhead) per K-block write event.

No further issues found.

## Agent A Change Log — Pass 10 Fix

**`math_fidelity_and_data_formats.md`**

- **Item 1 (line ~102, "NoC traffic" framing for packer_l1_acc):** Replaced "reducing NoC traffic. The result is a meaningful reduction in NoC overhead for large-K matmuls." with accurate intra-core L1 language: "This eliminates the extra L1 read from the read-modify-write cycle on the L1 output buffer: instead of reading the existing partial sum from L1, adding Dst, and writing back, the Packer accumulates in-place (L1 += Dst), removing the extra read entirely. The result is reduced L1 read overhead per K-block, which becomes significant for large-K matmuls." This aligns line 102 with the correct characterization already present at line 92 ("extra L1 read overhead") and removes the incorrect NoC framing — the operation is entirely intra-core and the NoC plays no role.


---

# B Review — Chapter 1: Hardware and TTNN Foundations — Pass 11

No feedback — chapter approved.

VERDICT: APPROVED

# B Review — Chapter 1: Hardware and TTNN Foundations — Pass 12

1. **`math_fidelity_and_data_formats.md`, Fidelity × Throughput table (line ~171–173)** — The "~1.5× baseline" label for the BFP8/HiFi2 row is a minor but real inaccuracy. The table shows 23 t/s/u vs a 14 t/s/u baseline (BF16/HiFi4), giving 23 ÷ 14 ≈ 1.64×, which rounds more accurately to ~1.6× than ~1.5×. Elsewhere in the file, "~1.5×" is used for HiFi3 vs HiFi4 (111 ÷ 74 ≈ 1.50 — correct there). A reader consulting this table to estimate throughput gains before benchmarking will underestimate the BFP8/HiFi2 uplift by about 10 percentage points. Fix: change "~1.5× baseline" to "~1.6× baseline" in the BFP8/HiFi2 row.

No other factual errors found. The compression removal of the two numbered step lists and the closing re-explanation paragraph from `packer_l1_acc in Depth` left the section coherent: the mechanism is correctly described in the inline field documentation (lines 77–79) and the Key Takeaways (line 184), and the "When packer_l1_acc helps most" bullets and the `in0_block_w` constraint clarification (line 85) survive intact. No load-bearing content was lost.

VERDICT: CHANGES NEEDED

## Agent A Change Log — Pass 12 Fix

**`math_fidelity_and_data_formats.md`**

- **Item 1 (Fidelity × Throughput table, BFP8/HiFi2 speedup):** Changed "~1.5× baseline" to "~1.6× baseline" in the BFP8/HiFi2 row of the Fidelity × Throughput Tradeoff Summary table. The table's own figures (23 t/s/u for BFP8/HiFi2 vs 14 t/s/u for BF16/HiFi4 baseline) give 23 ÷ 14 ≈ 1.64×, which rounds to ~1.6×, not ~1.5×. "~1.5×" is correct for HiFi3 vs HiFi4 (111 ÷ 74 ≈ 1.50×) and remains unchanged in the fidelity levels table. No prose references to this specific ratio were found elsewhere in the file.

# B Review — Chapter 1: Hardware and TTNN Foundations — Pass 13

No feedback — chapter approved.

VERDICT: APPROVED
