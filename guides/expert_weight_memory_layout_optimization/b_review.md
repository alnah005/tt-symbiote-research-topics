# B Review — Expert Weight Guide Index — Pass 1

## Issues Found

No errors found.

All checked facts match authoritative values:
- Per-expert BF16 total: 84.0 MB (3 × 7168 × 2048 × 2 = 88,080,384 bytes ≈ 84.0 MB) — correct.
- Qwen crossover effective_M: ~556 (437 × 2048 / 1611 ≈ 555.97) — correct.
- Mixtral crossover effective_M: ~451 — correct.
- Decode regime rule: batch_size × top_k ≤ 16 — correct.
- L1 weight double-buffer formula: 2 × in0_block_w × per_core_N_t × tile_size_bytes (no M_t term) — correct.
- Reshard overhead (Mixtral): ~2.3 s (24 tensors/layer × 32 layers × 3 ms/tensor = 2,304 ms) — correct.
- T3K ethernet bandwidth: ~7 GB/s effective per link — correct.
- Wormhole B0 Tensix cores: 80 (8×10) — correct.
- Wormhole B0 L1 per core: 1.5 MB — correct.
- Wormhole B0 ridge point: ~437 FLOP/byte — correct.

## Verdict

No feedback — guide index approved.
