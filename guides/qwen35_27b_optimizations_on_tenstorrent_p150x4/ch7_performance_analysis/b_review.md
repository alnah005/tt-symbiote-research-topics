# Agent B (Critic) Review -- Chapter 7

## Finding 1: Conv1d dispatch count arithmetic is wrong

**File:** `bottleneck_analysis.md`, section "3. Conv1d Shift Register Overhead"

**Claim:** "4 `ttnn.copy` operations per layer for the state shift, plus 4 `ttnn.multiply` and 3 `ttnn.mac` operations for the weighted sum. [...] 48 * (4 copies + 4 multiplies + 3 macs) = 528 dispatches per decode step."

**Problem:** The source code (`gdn.py` lines 281-288) shows 4 `ttnn.copy` + 1 `ttnn.multiply` + 3 `ttnn.mac` = 8 ops per layer. The first tap uses a single `ttnn.multiply`, and taps 1-3 each use `ttnn.mac`. The chapter claims 4 multiplies, but the code has only 1. The correct total is `48 * 8 = 384` dispatches, not 528.

**Fix:** Change "4 `ttnn.multiply`" to "1 `ttnn.multiply`", change "48 * (4 copies + 4 multiplies + 3 macs) = 528" to "48 * (4 copies + 1 multiply + 3 macs) = 384", and update the summary table row from "11 ttnn ops per layer" to "8 ttnn ops per layer".

---

No other factual errors found. All other claims checked and verified:
- Per-layer state arithmetic (384 pairs, 16 tiles, 12.6 MB) matches model_config.py constants
- NOC transaction counts (6,144 reads + 6,144 writes = 12,288 per layer; 44 reads per pair) match reader_gdn_fused.cpp
- GDN-to-attention cost ratio (2.26x = 9.78/4.33) is arithmetically correct
- Total state I/O (48 * 12.6 MB * 2 = ~1.2 GB) is correct
- RMS norm + SiLU described as separate post-kernel dispatches matches the actual Python decode path in gdn.py (lines 328-339), despite the kernel header suggesting they are fused
- Recurrence equation is a valid simplification of the implemented DeltaNet update
