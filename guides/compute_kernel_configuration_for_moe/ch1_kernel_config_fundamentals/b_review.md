# B Review — Chapter 1: Compute Kernel Config Fundamentals — Pass 1

1. **fp32_dest_acc_en.md, line 141 — Incorrect claim about L1 accumulation buffer dtype**
   The file states: "the L1 accumulation buffer used by the packer holds float32 values (because the packer serializes from the float32 destination register). This requires twice the L1 space for the accumulation buffer compared to the bfloat16 case."
   This conflates the destination register dtype with the packer L1 buffer dtype. On Tensix hardware the packer converts values to the output dtype (bfloat16) when writing to L1/DRAM; it does not necessarily preserve float32 in the L1 accumulation buffer even when `fp32_dest_acc_en=True`. If the L1 accumulation buffer is always bfloat16-wide regardless of `fp32_dest_acc_en`, the "twice the L1 space" statement is wrong and will lead implementers to over-budget L1 for the combined (`fp32_dest_acc_en=True`, `packer_l1_acc=True`) case. Fix: verify the actual dtype of the packer L1 accumulation buffer when `fp32_dest_acc_en=True`; if it is bfloat16, remove the "twice the L1 space" claim and correct the interaction description accordingly.

2. **wormhole_compute_kernel_config_api.md, line 212 / index.md, line 54 — `math_approx_mode=True` in HIFI2 config is inert but presented as intentional**
   Both files document `COMPUTE_KERNEL_CONFIG_HIFI2` with `math_approx_mode=True`, while the same API file (line 103) explicitly states "`math_approx_mode` has no effect on the FPU matrix multiply path." The comment on line 212 says "Acceptable for HiFi2 regime; see Chapter 4," which implies this field does something meaningful in this config. A reader implementing or extending this config for a pure matmul (no fused activation) would be materially misled into thinking `math_approx_mode=True` serves a precision or performance purpose here. Fix: clarify in the HIFI2 config block that `math_approx_mode=True` is carried forward from the source model's config for completeness but is inert for pure expert matmuls with no fused transcendental activation; setting it to `False` would produce identical numerical results and identical throughput.

3. **fp32_dest_acc_en.md, lines 93–97 — Summary table K_t values are inconsistent with the K dimension label**
   The table header says "K depth" and lists `K_t=224` for gate/up and `K_t=64` for down. However, the column label is `K depth` while the values are `K_t` (number of K tiles, not K in elements). The table also lists `d_model=7168` and `d_ff=2048` as the K-dimension source, which is correct, but a reader computing `K_t` independently gets 7168/32=224 and 2048/32=64 only if they know the tile size is 32. The tile size is stated in `math_fidelity_overview.md` (line 60) but not in this file. This is a minor comprehension gap, but it becomes a correctness issue because the note below the table (line 99) uses these K_t values to argue about sensitivity: "Even though down has a shallower K-loop (K_t=64 vs K_t=224)" — a reader who does not know the 32-element tile size cannot verify these numbers from the information in this file alone, and may incorrectly infer that down projections always have shallower K than gate/up projections as a general rule. Fix: add a one-line parenthetical in the table (e.g., "K_t = K/32, tile width = 32 elements") so the values are self-contained and verifiable.

No additional correctness issues were found. The factual content of the four primary field descriptions, the canonical config field values, the bfloat16/float32 format descriptions, the MathFidelity enum integer codes, and the K-depth error-accumulation reasoning are all internally consistent.

# B Review — Chapter 1: Compute Kernel Config Fundamentals — Pass 2

**Verifying Pass 1 fixes:**

1. **fp32_dest_acc_en.md — "twice the L1 space" claim:** Fixed. Line 143 now correctly states the L1 accumulation buffer holds bfloat16 (not float32) and explicitly says there is no "twice the L1 space" overhead from `fp32_dest_acc_en`. Fix verified.

2. **math_approx_mode inert for pure matmuls:** Fixed. Both `wormhole_compute_kernel_config_api.md` (lines 103, 212, 239) and `index.md` (line 54) now state clearly that `math_approx_mode=True` is inert for pure matmuls with no fused transcendental activation and is carried forward only for traceability. Fix verified.

3. **K_t definition added to fp32_dest_acc_en.md:** Fixed. Line 93 now has "K_t = K/32 (tile width = 32 elements)" in a note above the summary table. Fix verified.

**Remaining correctness issue found:**

1. **wormhole_compute_kernel_config_api.md, line 133 — `K_t` used in formula without definition in this file**
   The `packer_l1_acc` field description states: "This eliminates `(K_t / in0_block_w - 1)` DRAM reads of the output buffer per core, where `K_t = K/32` is the number of K tiles." On re-reading, `K_t` is now defined inline in this sentence ("`K_t = K/32`"), so this is self-contained. No correctness issue here — the definition was present in this file already.

**Conclusion after full re-read:**

All three Pass 1 fixes are correctly applied and verified. No new correctness issues were introduced by the fixes. The chapter is internally consistent across all four files on:
- Field semantics and defaults
- Canonical config field values
- bfloat16/float32 format descriptions
- MathFidelity enum integer codes and ordering
- K-depth error-accumulation reasoning
- The fp32_dest_acc_en / packer_l1_acc interaction (L1 buffer dtype is bfloat16)
- math_approx_mode scope (SFPU only, inert for pure matmul FPU path)

**No feedback — chapter approved.**
