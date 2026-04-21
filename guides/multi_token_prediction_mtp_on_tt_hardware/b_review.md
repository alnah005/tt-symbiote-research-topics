## Guide-Level B Feedback — Pass 1

No feedback — guide approved.

Cross-chapter consistency verified across all five chapters:

**H=4096 hidden dimension:** Ch02 (weight inventory: Q=4096×32768, K=4096×1024, V=4096×1024, O=32768×4096; ~160M total params), Ch04 (C_decode ≈ 243 ms basis calculation), Ch05 (input shapes [batch,1,H=4096], memory placement 304.6 MiB) — consistent throughout.

**~160M params / 304.6 MiB BF16 MTP head:** Ch02 (detailed weight breakdown), Ch05 (memory placement, overhead computation 1.06 ms / 243 ms ≈ 0.4%) — consistent.

**labels=None gate (training-only):** Ch03 (HuggingFace forward pass analysis, model.generate() → labels=None), Ch05 (TTNNMTPHead scope note, use_mtp flag) — consistent.

**No-resampling constraint on rejection path:** Ch04 (derived algebraically: resampling collapses E[tokens/cycle] to 2 and speedup to 1.0), Ch05 (CRITICAL blockquote in speculative_decode_loop_integration.md) — consistent.

**C_decode ≈ 243 ms (P150 baseline):** Ch04 (throughput analysis), Ch05 (memory_placement_for_mtp.md, fixed from 50-100 ms in B pass 1) — consistent.

**verify_logits[..., 0, :] = distribution for position t+2 (the token after x_t1):** Ch04 (acceptance rate derivation), Ch05 (fixed comment in speculative_decode_loop_integration.md) — consistent.

**E[tokens/cycle] = 1+α for K=1; speedup < 1 at batch=1 BW-bound:** Ch04 (throughput analysis main finding), Ch05 (testing_and_validation.md acceptance rate harness) — consistent.

**MTP KV cache 128 MiB = 1.1% of backbone KV cache (94 layers × 128 MiB each):** Ch04 (derived), Ch05 (memory_placement_for_mtp.md, DRAM placement rationale) — consistent.

**Weight key pattern model.future_prediction[0].*:** Ch01 (architecture), Ch02 (inventory), Ch03 (loading behavior), Ch05 (TTNNMTPHead weight key patterns) — consistent.

**Overall narrative:** Guide correctly establishes MTP as training-only in HuggingFace (Ch01-03), derives that MTP spec-decoding speedup < 1 at batch=1 BW-bound regime (Ch04), then provides the TTNN implementation plan with appropriate CRITICAL/SILENT FAILURE annotations for the rejection sampling path (Ch05). Chapter conclusions chain correctly.
