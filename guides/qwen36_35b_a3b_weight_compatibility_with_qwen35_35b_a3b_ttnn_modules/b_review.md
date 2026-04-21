## Guide-Level B Feedback — Pass 1

No feedback — guide approved.

Cross-chapter consistency verified across all six chapters:

**vocab_size = 151,936:** Ch1 (establishes field), Ch4 (bos_token_id=248044 > 151,936 — out-of-range), Ch5 (embed_tokens shape [151936, 4096]), Ch6 (migration verdict, CI assertion) — consistent throughout.

**partial_rotary_factor = 0.25 → rotary_dim = 32:** Ch3 (establishes HF resolution path for both Qwen3.5 and Qwen3.6), Ch6 (defensive fallback, migration step 2 comment) — consistent.

**H=4096 hidden dimension:** Ch5 (MTP block share backbone H=4096; 304.6 MiB ≈ 160M params × 2 bytes; embed_tokens shape [151936, 4096]), Ch6 (migration steps) — consistent and cross-verified against MTP guide Ch02/Ch04.

**MTP gate condition (labels is not None AND self.training is True):** Ch5 (`mtp_architecture.md`, `loading_recipe.md`), Ch6 (`compatibility_verdict.md`, `migration_steps.md`) — consistent.

**filter_mtp_keys prefix "model.future_prediction":** Ch5 (defined), Ch6 (reused, Step 3) — consistent.

**304.6 MiB / ~160M params MTP head:** Ch5 (weight key inventory), Ch6 (migration step 3, compatibility verdict) — consistent.

**bos_token_id = 248044 silent-failure risk:** Ch4 (detailed analysis), Ch5 (embed_tokens shape reference), Ch6 (medium risk, steps 4 and 5) — consistent risk assessment and mitigation.

**Weight tensor shapes identical (backbone):** Ch2 (establishes), Ch5 (notes MTP keys are additive, not modifications), Ch6 (none risk row in master table) — consistent.

**86 ms/token throughput target:** Ch4 (established from Qwen3.5 baseline), Ch6 (Step 7 smoke test target) — consistent.

**Overall narrative:** Guide correctly frames Qwen3.6 as a drop-in replacement for Qwen3.5 with four targeted pipeline changes. Each chapter conclusion feeds cleanly into the next. Migration steps cover all four risk items from the master compatibility table (one step per low-risk item, two steps for the medium-risk bos_token_id item).
