## Pass 7

**No feedback — chapter approved.**

All numerical values, formulas, and conceptual claims verified against ground truth:

- `mtp_num_hidden_layers=1` and `mtp_use_dedicated_embeddings=false` correctly stated (index.md line 17, training file lines 9–10).
- Main loss normalizer T-1 (training file line 73) and MTP loss normalizer T-2 (line 79) are both correct.
- Expert pool: 256 routed + 1 shared = 257 total, correctly stated on lines 104 and 112 of the training file (the Pass 6 error has been resolved).
- Per-layer parameter arithmetic: 257 × 2 × 2048 × 512 ≈ 537M, 40 layers ≈ 21B, MTP overhead ≈ 50M/21,000M ≈ 0.24% — all correct.
- Speculative decoding acceptance criterion `min(1, p/q)` (inference file line 41) and rejection correction `norm(max(0, p-q))` (line 46) are both correct per Leviathan et al., 2023.
- `C_mtp/C_main ≈ 0.035` and speedup at α=0.65 ≈ 1.59× (inference file lines 88–92) are both correct.
- MTP predicts token at t+2 from hidden states at position t — consistently stated throughout all three files.
- DeepSeek-V3 comparison table (training file lines 128–136) accurately reflects shared design choices.

---

## Pass 6

1. **mtp_architecture_and_training.md, line 104** — The sentence reads "almost all of which are MoE layers containing **128** routed expert FFN sub-networks (plus 1 shared expert)." The correct number is **256** routed experts, not 128. This contradicts both the ground-truth config (256 routed + 1 shared) and the model's own text twelve lines later (line 112: "**256** routed expert FFN sub-networks"). A reader implementing the MoE layer or reasoning about parameter counts from this sentence alone would arrive at the wrong figure. Fix: change "128 routed expert FFN sub-networks" to "256 routed expert FFN sub-networks" on line 104.
