# B Review — Tracy Guide Index — Pass 1

## Issues Found

No errors found.

All checked facts match authoritative values:
- Pattern A (fused): 3 dispatches per expert — correct.
- Pattern B (unfused): 4 dispatches per expert — correct.
- CCL scaling law: O(num_active_tokens × d_model), no division by num_chips — correct.
- Qwen 235B (top_k=8, num_experts=128): expert_capacity = seq_len / 16 — correct.
- Memory-bound condition threshold: seq_len < ~40,960 — correct.
- Benchmark protocol: 20 timed iterations, median + p95, warm-up ≥ 3 — correct.
- T3K ethernet bandwidth: ~7 GB/s effective per link — correct.

Note: The index does not explicitly state Wormhole B0 core count (80) or L1/core (1.5 MB). These facts are absent, not wrong; no factual error is present.

## Verdict

No feedback — guide index approved.
