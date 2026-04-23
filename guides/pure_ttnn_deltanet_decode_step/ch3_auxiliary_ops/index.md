# Chapter 3 — Causal Conv1D and Gated RMSNorm Without Host Readback

Chapter 2 derived a complete on-device TTNN decomposition for the recurrent delta rule step — the primary host-crossing bottleneck in `TTNNQwen3LinearAttention`. This chapter closes the remaining two host-crossing gaps: the causal conv1d state update and the gated RMSNorm. Both are simpler than the recurrence. The causal conv1d update is a sliding-window shift-and-convolve on a fixed state buffer, fully expressible as a sequence of existing TTNN slice, concat, mul, and sum ops with no new kernel development. The gated RMSNorm (`FusedRMSNormSwishGate`) is a composition of `ttnn.rms_norm`, `ttnn.silu`, and `ttnn.mul`, all already available in the TTNN API.

By the end of this chapter you will have a complete TTNN implementation sketch for both operations, with tensor shapes, memory configs, and availability tags, and will understand the tile-alignment constraint that determines the correct memory layout for the conv state.

---

## Prerequisites

- Chapter 1 (`ch1_trace_breakage_audit/`): identifies the causal conv1d update and gated RMSNorm as host-crossing operations that must be eliminated
- Chapter 2 (`ch2_ttnn_decomposition/`): establishes the on-device tensor lifecycle model that the conv state must join; provides state tensor memory config reference

---

## Learning Objectives

1. Express the causal conv1d decode-time state update as a sequence of TTNN primitives without any C extension or PyTorch kernel call
2. Identify the correct memory layout for `conv_state` given that its innermost dimension K=4 is not tile-aligned, and explain the ROW_MAJOR vs. TILE padding trade-off
3. Express `FusedRMSNormSwishGate` as a three-op TTNN composition and verify numerical equivalence with the PyTorch reference
4. State the per-step L1 memory footprint for the gated RMSNorm inputs and explain why they can be held in L1

---

## Files in Reading Order

1. [`causal_conv1d_update_ttnn.md`](./causal_conv1d_update_ttnn.md) — TTNN decomposition of the causal conv1d state update; state shift via slice+concat; depthwise convolution via mul+sum; memory layout for a K=4 non-tile-aligned state
2. [`gated_rmsnorm_ttnn.md`](./gated_rmsnorm_ttnn.md) — TTNN decomposition of `FusedRMSNormSwishGate`; three-op composition of rms_norm + silu + mul; numerical equivalence analysis; L1 memory footprint

---

## What's Next

After this chapter the only remaining gap before the full decode path is on-device is the causal conv1d TTNN wiring (Task 3 in Chapter 7) and the gated RMSNorm wiring (Task 4 in Chapter 7). Both depend on the conv state being migrated to an on-device TTNN tensor (Task 1), which is addressed in Chapter 7's task list.

Chapter 4 (`ch4_gdn_fused_kernel/`) examines the `gdn_full_fused_inplace` kernel from the Qwen3.5-27B Blackhole implementation for reuse potential on Wormhole T3K.
