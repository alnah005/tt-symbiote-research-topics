# Chapter 4: Speculative Decoding with MTP on TT Hardware

## Prerequisites

- Chapter 1: MTP Foundations (architecture, training objective)
- Chapter 2: MTP Weights and Memory (weight layout, 304.6 MiB BF16 head, 160M params)
- Chapter 3: MTP in HuggingFace (confirmed: MTP head is training-only in standard HF; `model.generate()` never calls it; a custom generation loop is required)

## Chapter Overview

Chapters 1–3 established what the MTP head is and confirmed that using it at inference time requires bypassing `GenerationMixin` entirely. This chapter answers the next question: **if we do write a custom loop to invoke the MTP head, does it actually improve throughput on TT hardware?**

The short answer is nuanced:

- The algorithm is architecturally sound and maps cleanly onto speculative decoding.
- At **batch size 1** on **memory-bandwidth-bound** TT hardware (P150, T3K), with `mtp_num_hidden_layers = 1` (N=1), the expected speedup is **(1+α)/2 < 1** for all acceptance rates α < 1. MTP with N=1 does not improve throughput in this regime.
- The benefit materializes at **batch sizes B > 1**, where the verification pass processes B×2 tokens in parallel and amortizes the bandwidth cost.
- The value of this chapter for the project is the **implementation blueprint** carried forward into Chapter 5 (TTNN custom loop).

## Key Finding

> **MTP speculative decoding with N=1 (Qwen3.6-35B-A3B) provides no throughput improvement at batch size 1 on memory-bandwidth-bound TT hardware.** The verification pass costs one full backbone traversal — the same as a regular decode step — while the expected gain is only α additional tokens. Speedup = (1+α)/2 < 1 for α < 1. The algorithm is correct and worthwhile to implement for larger batch sizes.

## Files in This Chapter

| File | Contents |
|---|---|
| [speculative_decoding_primer.md](speculative_decoding_primer.md) | Algorithm, expected-token formula, why spec decoding targets bandwidth-bound hardware |
| [mtp_as_draft_model.md](mtp_as_draft_model.md) | How MTP maps to the draft-model role; step-by-step algorithm for Qwen3.6 N=1 |
| [throughput_analysis_on_tt_hardware.md](throughput_analysis_on_tt_hardware.md) | Bandwidth-bound cost model; speedup derivation; breakeven table |
| [acceptance_rate_estimation.md](acceptance_rate_estimation.md) | Estimated acceptance rate ranges by domain, domain dependence, empirical measurement approach |

---
**Next:** [speculative_decoding_primer.md](speculative_decoding_primer.md)
