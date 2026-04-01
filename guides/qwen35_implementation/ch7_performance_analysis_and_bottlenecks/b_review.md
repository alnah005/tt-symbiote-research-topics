# Agent B Review — Chapter 7: Performance Analysis and Bottlenecks

## Pass 1

1. [latency_breakdown.md] Total syncs listed as "~81" but PERF.md Total row states "~70". Individual rows sum to 30+50+1=81 — table and PERF.md inconsistency noted.
2. [sync_overhead.md] "~40 rotary setup syncs" attributed to HfRotarySetup.get_rot_mats() — incorrect. Method returns cached matrices without syncing. The 40 syncs came from `partial_rope_fn` (3 to_torch + 2 from_torch per attention layer).
3. [index.md] Efficiency "~6.7%" — should be "~6.8%" (11.7 ÷ 172 = 0.068).

## Pass 2

No feedback — chapter approved.

## Pass 3

No feedback — chapter approved.

## Pass 4

No feedback — chapter approved.

## Pass 5 (final)

No feedback — chapter approved.
