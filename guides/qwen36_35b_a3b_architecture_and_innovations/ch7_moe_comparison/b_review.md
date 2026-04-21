## Pass 1

1. **`cross_model_moe_comparison.md`, line 17 — DeepSeek-V3 per-expert params wrong in summary table.**
   The table states `~88M` for DeepSeek-V3 per-expert params. The correct value is `~44M`: each expert has three matrices each of shape [7168×2048] or [2048×7168], giving 3 × 7168 × 2048 = 44,040,192 ≈ 44M parameters. The `~88M` figure is exactly 2× too large (likely from counting each matrix twice or confusing the combined gate+up weight with both matrices separately). The body text on line 31 correctly states `~44M`, so the table and body contradict each other — a reader using the table will get a wrong number.
   **Fix:** Change `~88M` to `~44M` in the summary table.

2. **`cross_model_moe_comparison.md`, lines 17 and 61 — Gemma4 per-expert params wrong in both table and body.**
   The table (line 17) and body text (line 61) both state `~25M` for Gemma4-26B-A4B per-expert params. The correct value is `~12.6M`: Gemma4 has hidden_size=2048 and expert intermediate=2048, giving 3 × 2048 × 2048 = 12,582,912 ≈ 12.6M parameters per expert. The `~25M` figure is approximately 2× too large.
   **Fix:** Change `~25M` to `~12.6M` in both the summary table (line 17) and the body text (line 61).

3. **`cross_model_moe_comparison.md`, line 141 — Incorrect lead-in count of 10,240 for routed expert weight tensors.**
   The sentence opens with "The 10,240 routed expert weight tensors (256 × 40 layers × 1 tensor per matrix direction…)". The count 10,240 = 256 × 40 omits the factor of 3 for the three weight matrices per expert (W_gate, W_up, W_down). The correct count is 256 × 3 × 40 = 30,720, which the parenthetical immediately states. The lead-in number is flatly wrong and a reader will get a wrong answer from it before reaching the correction.
   **Fix:** Replace "The 10,240 routed expert weight tensors (256 × 40 layers × 1 tensor per matrix direction — the actual count is 256 × 3 × 40 = 30,720 matrices)" with "The 30,720 routed expert weight matrices (256 experts × 3 matrices × 40 layers)".

## Pass 2

1. **`qwen36_moe_architecture.md`, line 154 — "10,240+" is wrong; same class of error as Pass 1 item 3, different file.**
   The TTNN Deployment Implications bullet for quantization reads: "bfp4 expert weight quantization dramatically reduces DRAM pressure for the 10,240+ weight tensors (256 experts × 3 matrices × 40 layers, minus the 3 shared expert matrices per layer)."
   The parenthetical correctly evaluates to 30,720 (256 × 3 × 40), but the leading figure "10,240+" is wrong by a factor of 3. 10,240 = 256 × 40, omitting the ×3 for the three SwiGLU matrices per expert. A reader skimming the bullet will take "10,240+" as the count; the parenthetical on the same line contradicts it. Pass 1 fixed the identical error in `cross_model_moe_comparison.md` line 141 but left this instance in `qwen36_moe_architecture.md` unfixed.
   **Fix:** Replace "10,240+ weight tensors (256 experts × 3 matrices × 40 layers, minus the 3 shared expert matrices per layer)" with "30,720 routed expert weight matrices (256 experts × 3 matrices × 40 layers), plus 120 shared expert weight matrices (1 × 3 × 40)".

## Pass 3

**No feedback — chapter approved.**
