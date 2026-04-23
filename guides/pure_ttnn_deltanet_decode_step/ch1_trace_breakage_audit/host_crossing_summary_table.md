# Host-Crossing Summary Table

This file consolidates all host-crossing calls identified in `forward_pass_walkthrough.md` into a single reference table. Each row names one operation that crosses the Wormhole device-host boundary during a decode step, specifies the tensors involved, classifies the trace-break mechanism, and assigns a fix priority. By the end of this file the reader has a single artifact that can be used directly to plan the implementation work in Chapter 7.

---

## Trace-Break Mechanism Taxonomy

Before reading the table, review the four mechanism tags used in the Priority column:

| Tag | Definition |
|---|---|
| `HOST_KERNEL_LAUNCH` | A non-TTNN kernel (C extension, Triton, or PyTorch op) is dispatched outside the TTNN command queue. Metal Trace cannot record or replay this dispatch. Always breaks trace. |
| `TO_TORCH` | `ttnn.to_torch` or an implicit device-to-host DMA transfer occurs, including the blocking `synchronize_device` that precedes it. Metal Trace cannot insert a blocking device sync into a static command stream. |
| `FROM_TORCH` | `ttnn.from_torch` allocates a new device buffer at runtime. Metal Trace requires all device buffer addresses to be fixed at capture time; dynamic allocation inside the trace bracket is forbidden. |
| `PYTHON_BRANCH` | Data-dependent Python control flow (an `if` statement whose condition depends on a device tensor value) cannot be captured as a static sequence. Not present in the current `TTNNQwen3LinearAttention.forward` but listed for completeness. |

A single operation can trigger multiple mechanisms. All three mechanisms must be absent for an operation to be trace-compatible.

---

## Host-Crossing Call Table

> **Note:** Source file paths are relative to the tt-metal repository root. Exact line numbers are marked as approximate (±10 lines) because they shift as the codebase evolves; the structural call pattern is stable.

| Step | Operation | Source file and line | Tensors read from device | Tensors written to device | Trace-break mechanism | Priority to fix |
|---|---|---|---|---|---|---|
| 2 | `causal_conv1d_update` (C extension call) | `models/experimental/tt_symbiote/modules/qwen_attention.py` ~L180 | `mixed_qkv` [1, 8192, 1] = 16 KB; `conv_state` [1, 8192, 4] = 64 KB | `mixed_qkv` (updated) [1, 8192, 1] = 16 KB; `conv_state` (new) [1, 8192, 4] = 64 KB | `TO_TORCH`, `HOST_KERNEL_LAUNCH`, `FROM_TORCH` | **P2** — Medium latency impact (~160 KB round-trip per layer); medium complexity (TTNN slice + concat + mul + sum; see Chapter 3) |
| 3 | Decay gate: `torch.exp`, `F.softplus`, `torch.sigmoid` on `a_t`, `b_t` | `models/experimental/tt_symbiote/modules/qwen_attention.py` ~L200 | `a_t` [1, 1, 32] = 64 B; `b_t` [1, 1, 32] = 64 B | none (`g_t` and `beta_t` remain as host tensors; no `ttnn.from_torch` call) | `TO_TORCH` | **P3** — Low latency impact (128 B device→host; sync stall is the real cost); lowest complexity (wiring only; all TTNN ops exist) |
| 4 | `recurrent_gated_delta_rule` (flash-linear-attention Triton/PyTorch) | `models/experimental/tt_symbiote/modules/qwen_attention.py` ~L230 | `q_tilde` [1, 1, 32, 128] = 8 KB; `k_tilde` [1, 1, 32, 128] = 8 KB; `v` [1, 1, 32, 128] = 8 KB; `g_t` [1, 1, 32] = 64 B; `beta_t` [1, 1, 32] = 64 B; `S_prev` [1, 32, 128, 128] = 1,024 KB | `o_t` [1, 1, 32, 128] = 8 KB; `S_new` [1, 32, 128, 128] = 1,024 KB | `TO_TORCH`, `HOST_KERNEL_LAUNCH`, `FROM_TORCH` | **P1** — Dominant latency impact (~2 MB PCIe round-trip + sync per layer × 30 layers ≈ 9–21 ms total; see Chapter 6); medium-high complexity (TTNN 6-op decomposition; see Chapter 2) |
| 5 | `FusedRMSNormSwishGate` (PyTorch `nn.Module`) | `models/experimental/tt_symbiote/modules/qwen_attention.py` ~L260 | `o_t` [1, 1, 4096] = 8 KB; `z` [1, 1, 4096] = 8 KB | `output` [1, 1, 4096] = 8 KB | `TO_TORCH`, `FROM_TORCH` | **P3** — Low latency impact (~24 KB round-trip + sync); lowest complexity (wiring only: `ttnn.rms_norm` + `ttnn.silu` + `ttnn.mul`; see Chapter 3) |

---

## Fix Priority Rationale

Priority is assigned on two axes: (1) decode latency impact — how many microseconds does eliminating this crossing save at B=1 across all 30 DeltaNet layers? — and (2) implementation complexity — does this require a new kernel, or is it a wiring change over existing TTNN ops?

**P1 — Step 4 (recurrent gated delta rule):** The state matrix S [1, 32, 128, 128] is 1 MB. Transferring it device-to-host and back on every decode step across 30 layers costs approximately 60 MB of PCIe traffic per decode step. At 16 GB/s effective PCIe 4.0 bandwidth this is ~3.75 ms in pure transfer time, plus synchronization stalls pushing the realistic total to 9–21 ms. This is the dominant source of DeltaNet decode latency by far. Fixing this crossing requires wiring a 6-operation TTNN decomposition (Chapter 2) and migrating state storage to on-device tensors (`device_state_persistence.md`, also a prerequisite for P2).

**P2 — Step 2 (causal conv1d update):** The conv state [1, 8192, 4] is 64 KB, plus the mixed_qkv tensor at 16 KB per direction. The round-trip is approximately 160 KB per layer × 30 layers = ≈4.69 MB, estimated at 0.3–1.5 ms total plus sync overhead. This is significant but secondary to the state matrix. Fixing this requires a TTNN `slice` + `concat` + `mul` + `sum` composition (Chapter 3) and migrating `conv_state` to on-device storage (prerequisite shared with P1).

**P3 — Steps 3 and 5 (decay gate + gated RMSNorm):** These operations transfer a total of less than 50 KB combined per layer. The latency impact is dominated by synchronization stalls (one `synchronize_device` per `ttnn.to_torch` call), not data volume. However, synchronization stalls scale with the number of layers (30 × 2 syncs each for steps 3 and 5), so the cumulative overhead can reach hundreds of microseconds. Both fixes are pure wiring changes: all required TTNN ops exist and are `[AVAILABLE — needs wiring]`. These should be fixed alongside or immediately after P1 and P2.

---

## Dependency Order

Some fixes depend on others being in place first:

```
device_state_persistence.md change (on-device S and conv_state)
    |
    +--- P1: Step 4 recurrent delta rule (Chapter 2) -- requires on-device S
    |
    +--- P2: Step 2 causal conv1d (Chapter 3)        -- requires on-device conv_state
    |
    P3: Step 3 decay gate (wiring only)              -- independent; shares ttnn.to_torch removal with P1
    P3: Step 5 gated RMSNorm (wiring only)           -- independent; output feeds into Step 6 which is already on-device
```

Steps 3 and 5 can technically be wired at any time, but their fixes only become visible (i.e., stop triggering `TO_TORCH`) once the tensors they operate on (`a_t`, `b_t`, `o_t`, `z`) are produced and consumed entirely on-device. If step 4 still calls `ttnn.to_torch(o_t)` to feed the `recurrent_gated_delta_rule` host kernel, wiring step 5's gated RMSNorm to TTNN does not eliminate the `TO_TORCH` on `o_t` — step 4 still reads it from device. The correct order is: fix state persistence first (prerequisite), then fix P1 (step 4), then fix P2 (step 2), then fix P3 (steps 3 and 5).

---

## Not in This Table: Already Trace-Compatible Operations

Steps 1 and 6 — the input projections and the output projection — are fully trace-compatible and require no changes.

- **Step 1** (`in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b` via `ttnn.linear`, followed by all-gather): all dispatches are on-device TTNN ops recorded correctly by the trace capture.
- **Step 6** (`out_proj` via `ttnn.linear`, followed by all-gather): same — fully on-device and trace-compatible.

These steps are excluded from the table above because no fix is needed.

---

**Next:** [`device_state_persistence.md`](./device_state_persistence.md)
