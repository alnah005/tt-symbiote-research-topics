## Guide-Level B Feedback — Pass 1

No feedback — guide approved.

Cross-chapter consistency verified across all six chapters:

**rotary_dim = 64 (partial_rotary_factor=0.5, head_dim=128):** Ch2 (qwen36_rope_config.md establishes), Ch4 (gap analysis and implementation), Ch5 (operation_cost_breakdown.md: 256 bytes/position = 64 × 2 tables × 2 bytes), Ch6 (index.md key numbers table, _gather_sections final shape) — consistent throughout.

**mrope_section = [11, 11, 10], sum = 32 = rotary_dim//2:** Ch1 (section_dimension_assignment.md), Ch2 (qwen36_rope_config.md), Ch4 (gather_operation_on_ttnn.md), Ch5 (kernel_launch_overhead.md: sections 22,22,20 not tile-aligned), Ch6 (integration_steps.md, correctness_validation.md) — consistent throughout.

**cos/sin table size = 8 MiB (2 × 32768 × 64 × 2 bytes):** Ch4 (pre_computed_cos_sin_strategy.md), Ch5 (memory_access_analysis.md), Ch6 (index.md key numbers table) — consistent.

**5 additional TTNN kernel dispatches (3 embedding + 2 concat):** Ch4 (gather_operation_on_ttnn.md), Ch5 (operation_cost_breakdown.md, kernel_launch_overhead.md), Ch6 (index.md, tracing_and_program_cache_considerations.md) — consistent.

**~25–50 µs overhead < 0.02% of ~250 ms decode step (P150):** Ch5 (prefill_vs_decode_comparison.md Key Finding), Ch6 (Option B rationale in tracing, index.md) — consistent.

**Text-only M-RoPE ≡ standard 1D RoPE (algebraically proven):** Ch3 (mathematical_equivalence_proof.md full proof), Ch5 (operation_cost_breakdown.md notes text decode is identical), Ch6 (Option B recommendation, Test 1 torch.equal criterion) — consistent.

**h+w random-access formula ~168 KiB at S=1024 (fixed in Ch5 B pass 1):** Ch5 (prefill_vs_decode_comparison.md, operation_cost_breakdown.md), Ch6 (references Ch5 for performance numbers) — consistent after fix.

**_gather_sections column boundaries 2*s_t (fixed in Ch6 B pass 1):** Ch4 (establishes the 3-lookup + 2-concat approach), Ch6 (integration_steps.md now uses `[0:2*s_t, 2*s_t:2*(s_t+s_h), 2*(s_t+s_h):]`) — consistent after fix; consistent with HuggingFace `mrope_section * 2` convention.

**PCC > 0.9999 threshold for device validation (not 0.99):** Ch6 (justified by Ch3 algebraic equivalence + 64-layer error accumulation) — no cross-chapter contradiction.

**Program cache 100% hit at decode; +3 misses per new seq_len at prefill:** Ch5 (kernel_launch_overhead.md: mrope_section sizes not tile-aligned, fusion deferred), Ch6 (tracing_and_program_cache_considerations.md) — consistent.

**Overall narrative:** Guide correctly establishes M-RoPE foundations (Ch1–2), proves text-only degeneracy (Ch3), designs the 3-lookup+2-concat TTNN implementation (Ch4), quantifies performance cost as negligible (Ch5), and provides integration, validation, and trace-compatibility guidance (Ch6). Chapter conclusions chain correctly. The Option B recommendation in Ch6 is correctly grounded in Ch3's algebraic proof.
