# B Review — Chapter 3: Accuracy Analysis — Pass 1

2 factual errors found.

---

**Error 1**

- File: `qwen_vs_deepseek_accuracy_comparison.md`
- Line: ~31
- Wrong: "K_t = 64 (latent key dimension)"
- Correct: K_t = 64 here is the down-projection tile count, computed as d_ff / 32 = 2048 / 32 = 64. It is not the latent key dimension of the KV cache. The latent key dimension (from DeepSeek-V3's MLA design) is a separate architectural parameter. Conflating the two is a factual error.

---

**Error 2**

- File: `accuracy_metrics_for_moe.md`
- Line: ~50 (table row for bfloat8_b gate/up)
- Wrong: bfloat8_b gate/up PCC listed as 0.97–0.98
- Correct: 0.98–0.99. Both `bfloat16_vs_bfloat8_vs_bfloat4_accuracy.md` (lines 23–24) and `index.md` (line 41) consistently report 0.98–0.99 for bfloat8_b gate/up projections. Gate and up projections are less sensitive than the down projection and their observed PCC under bfloat8_b is consistently one band higher. The 0.97–0.98 figure is appropriate for the bfloat8_b down projection range, not for gate/up.

## Agent A Change Log — B Feedback Pass 1
- qwen_vs_deepseek_accuracy_comparison.md: Fixed K_t=64 parenthetical from "(latent key dimension)" to "(d_ff/32 = 2048/32 = 64 tiles — K-loop depth for down projection matmul)"
- accuracy_metrics_for_moe.md: Fixed bfloat8_b gate/up PCC range from 0.97-0.98 to 0.98-0.99 (consistent with bfloat16_vs_bfloat8_vs_bfloat4_accuracy.md and index.md)

---

# B Review — Chapter 3: Accuracy Analysis — Pass 2

## Pass 1 Fix Verification

Both Pass 1 fixes are correctly applied:

- **Error 1 fix confirmed**: `qwen_vs_deepseek_accuracy_comparison.md` line 31 now reads `K_t = 64 (d_ff / 32 = 2048 / 32 = 64 tiles — the K-loop depth for the down projection matmul)`. Parenthetical is correct.
- **Error 2 fix confirmed**: `accuracy_metrics_for_moe.md` line 50 table row now shows `bfloat8_b | gate/up | 0.98–0.99`. Value is correct and consistent with `bfloat16_vs_bfloat8_vs_bfloat4_accuracy.md` and `index.md`.

## New Error Found

**Error 3**

- File: `qwen_vs_deepseek_accuracy_comparison.md`
- Line: 31
- Wrong: "DeepSeek-V3 uses a compressed KV cache design with K_t = 64 (d_ff / 32 = 2048 / 32 = 64 tiles — the K-loop depth for the down projection matmul)."
- Correct: The Pass 1 fix correctly updated the K_t parenthetical, but the lead-in clause "uses a compressed KV cache design with" was left intact and is now contradictory. K_t = 64 here is a property of the FFN down-projection matmul geometry (d_ff / 32 tiles), not of the KV cache architecture. The sentence conflates two unrelated subsystems: DeepSeek-V3's MLA compressed KV cache is a separate design feature that has nothing to do with the d_ff = 2048 intermediate dimension of its MoE expert FFN blocks. The clause must be replaced with one that accurately introduces the FFN geometry. Suggested replacement for the offending clause: "DeepSeek-V3's expert FFN down projection has K_t = 64 (d_ff / 32 = 2048 / 32 = 64 tiles — the K-loop depth for the down projection matmul)."

## Agent A Change Log — B Feedback Pass 2
- qwen_vs_deepseek_accuracy_comparison.md line 31: Replace "DeepSeek-V3 uses a compressed KV cache design with K_t = 64 ..." with "DeepSeek-V3's expert FFN down projection has K_t = 64 ..." to remove the false KV-cache framing left over from the Pass 1 partial fix.

## Agent A Change Log — B Feedback Pass 2
- qwen_vs_deepseek_accuracy_comparison.md: Fixed lead-in clause — removed "compressed KV cache design" attribution; sentence now correctly introduces K_t=64 as the expert FFN down-projection K-loop tile count (d_ff/32)

---

# B Review — Chapter 3: Accuracy Analysis — Pass 3

Pass 2 fix verified. No feedback — chapter approved.
