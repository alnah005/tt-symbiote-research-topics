# Plan: Pure TTNN DeltaNet Decode Step Without Host Readback

## 1. Audience

**Primary reader:** A hardware-aware ML systems engineer or tt-metal kernel developer who:
- Understands the DeltaNet / Gated Delta Net recurrence at the mathematical level (state matrix S, delta rule update, scalar decay gate) — either from direct prior reading or from the companion guide "Gated Delta Net and Gated Attention on T3K"
- Has working knowledge of the TTNN Python API: can read and write `ttnn.matmul`, `ttnn.mul`, `ttnn.linear`, `ttnn.rms_norm`, and shape manipulation ops (`ttnn.reshape`, `ttnn.permute`)
- Understands Metal Trace at a conceptual level: knows that `ttnn.begin_trace_capture` / `ttnn.execute_trace` record a static device command stream, and knows that any host-side Python or CPU operation (including `torch` calls and `ttnn.from_torch`) inside the capture bracket breaks the trace
- Is familiar with the T3K 1×8 Wormhole mesh (tensor parallelism, column-sharded linear ops, all-gather CCL) at the level covered in the Gated Delta Net guide, Chapter 6
- Has not yet studied what specific PyTorch fallback calls exist inside `TTNNQwen3LinearAttention`, why they break trace, or how to replace them with on-device TTNN ops

**What the reader already knows:**
- The mathematical recurrence for Gated DeltaNet decode: `S_t = g_t * S_{t-1} + k̃_t ⊗ (β_t * (v_t − g_t * S_{t-1}^T k̃_t))`; all tensor shapes for this operation at batch=1, T=1 on the Qwen3.6-35B-A3B configuration
- How the hybrid Qwen3.6-35B-A3B decoder interleaves DeltaNet linear attention layers and full-attention layers, and the overall model-level trace goal
- How Metal Trace works for standard attention-only models (e.g., the `_capture_decode_trace_text` path in `generator.py`) and why host-device synchronization points (like `synchronize_device` and PyTorch kernel launches) must be eliminated before a model can be traced end-to-end

**What the reader will learn from this guide:**
- Which specific Python/PyTorch calls inside `TTNNQwen3LinearAttention` break Metal Trace, what each one computes, and what tensors cross the device–host boundary during a single decode step
- A complete TTNN-native decomposition of the DeltaNet recurrence (recurrent gated delta rule step, causal conv1d update, and gated RMSNorm), expressed as a sequence of TTNN primitives with matching tensor shapes and memory configs
- What the `gdn_full_fused_inplace` kernel (from the Qwen3.5-27B Blackhole implementation) computes, its architecture-specific assumptions, and whether it can be reused on Wormhole T3K or requires adaptation
- Whether existing TTNN or TT-Metalium scan / selective-scan primitives (Mamba SSM kernels, parallel prefix scan) can be adapted for the DeltaNet state update — and a clear answer for each candidate
- The tensor shape and memory layout requirements for keeping the DeltaNet state matrix on-device across decode steps: DRAM vs. L1 trade-offs, tile alignment constraints, and the per-device memory footprint under head-parallel sharding on T3K
- The host-CPU round-trip latency for the current `recurrent_gated_delta_rule` fallback at batch=1 (including `ttnn.to_torch`, kernel launch, `ttnn.from_torch`), and the expected latency after a pure on-device implementation
- The PCC accuracy threshold acceptable for a TTNN recurrence kernel, how to measure it, and why numerical sensitivity of the DeltaNet state update is lower than for softmax attention

---

## 2. Chapter List

---

### Chapter 1 — Why the Current Implementation Breaks Trace

**Description:** Audits the `TTNNQwen3LinearAttention` forward pass step-by-step to identify every host-crossing call, explains what each call computes and why it is incompatible with Metal Trace, and defines the exact on-device / off-device boundary that must be moved.

**Directory:** `ch1_trace_breakage_audit/`

**Files and content:**

- **`index.md`** — Chapter overview, learning objectives, and file navigation.
  - States the chapter's goal: give the reader a precise map of which operations in `TTNNQwen3LinearAttention.forward` are on-device and which are host-side, before proposing any fix
  - Lists the three section files in reading order
  - Defines the chapter's key deliverable: a table of every host-crossing call with: operation name, calling line in the source file, tensors read from device, tensors written to device, and the trace-break mechanism (CPU kernel launch, `ttnn.from_torch`, `synchronize_device`, or Python data-dependent branching)

- **`forward_pass_walkthrough.md`** — Step-by-step walkthrough of `TTNNQwen3LinearAttention.forward` at decode time (B=1, T=1).
  - Step 1 — Input projections (on-device TTNN):
    - `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b` via `ttnn.linear` (column-sharded over 8 T3K devices); all-gather async follows to restore replicated hidden state for subsequent ops; these steps are trace-compatible
    - Note: `in_proj_a` and `in_proj_b` have very small output dimensions (num_v_heads = 4 for 35B-A3B); they may use replicated weight with plain `ttnn.linear` rather than column-sharded; still on-device and trace-compatible
  - Step 2 — Causal conv1d update (host-crossing):
    - Call to `causal_conv1d_update` from the `causal-conv1d` C extension (CUDA/CPU); this reads `mixed_qkv` from device (via implicit `ttnn.to_torch`), runs the sliding-window convolution on CPU/GPU, and writes updated `mixed_qkv` and `conv_state` back to device via `ttnn.from_torch`
    - Tensors crossing boundary: `mixed_qkv` [B, 8192, 1] read from device; `conv_state` [B, 8192, 4] read from device and written back; `mixed_qkv` output [B, 8192, 1] written to device
    - Trace-break mechanism: `ttnn.from_torch` inside the forward path allocates a new device buffer — incompatible with static command stream
  - Step 3 — Decay gate and update rate computation (host or device, small tensors):
    - `α_t = −exp(A_log) * softplus(a_t + dt_bias)` and `g_t = exp(α_t)` and `β_t = σ(b_t)` operate on tiny tensors [B, 1, num_v_heads]; if computed in Python (with torch scalar ops), this is a host-crossing; if already on-device (which TTNN supports via `ttnn.exp`, `ttnn.softplus`, `ttnn.sigmoid`), it is trace-compatible; document the current implementation status
    - Tensors: `a_t` [B, 1, num_v_heads] from `in_proj_a`, `b_t` [B, 1, num_v_heads] from `in_proj_b`; `A_log` and `dt_bias` are learned scalars (weight tensors)
  - Step 4 — Recurrent gated delta rule step (host-crossing, primary bottleneck):
    - Call to `recurrent_gated_delta_rule` from `flash-linear-attention` (Triton CUDA kernel or pure-PyTorch fallback); receives Q̃, K̃, V, g, β, and the previous state S as PyTorch tensors on CPU/CUDA device; returns new output and updated state
    - To supply PyTorch tensors: current implementation calls `ttnn.to_torch` on Q̃, K̃, V (which are on-device TTNN tensors after the projection and split); also reads S from device (either `ttnn.to_torch` or storing S as a plain torch tensor between steps)
    - Tensors crossing boundary: Q̃ [B, 1, num_v_heads, 128], K̃ [B, 1, num_v_heads, 128], V [B, 1, num_v_heads, 128], g [B, 1, num_v_heads], β [B, 1, num_v_heads] — all read from device; S_prev [B, num_v_heads, 128, 128] read from device; S_new [B, num_v_heads, 128, 128] and `core_attn_out` [B, 1, num_v_heads, 128] written back to device
    - Trace-break mechanisms: `ttnn.to_torch` (host readback with implicit device sync), PyTorch kernel launch (non-TTNN dispatch), `ttnn.from_torch` for outputs
  - Step 5 — Gated RMSNorm (host or device):
    - `FusedRMSNormSwishGate` applies RMSNorm then element-wise Swish gating; if run as a PyTorch module on CPU/CUDA, it requires host readback of `core_attn_out` and `z`; if composable from `ttnn.rms_norm` + `ttnn.silu` + `ttnn.mul`, it can be on-device
    - Document current implementation: whether this calls a C extension or a plain Python/TTNN composition
    - Tensors: `core_attn_out` [B, 1, num_v_heads × d_v], `z` [B, 1, num_v_heads × d_v]; output [B, 1, value_dim]
  - Step 6 — Output projection (on-device TTNN):
    - `out_proj` via `ttnn.linear` (row-sharded) followed by all-gather; trace-compatible

- **`host_crossing_summary_table.md`** — Consolidated table of all host-crossing calls with trace-break analysis.
  - Table columns: Step | Operation | Source file and line | Tensors read from device | Tensors written to device | Trace-break mechanism | Priority to fix
  - Rows for each crossing identified in `forward_pass_walkthrough.md`: causal conv1d update, recurrent gated delta rule step, (conditional) decay gate computation, (conditional) gated RMSNorm
  - For each row: classify the trace-break mechanism using one of: `HOST_KERNEL_LAUNCH` (a non-TTNN kernel is dispatched — always breaks trace), `TO_TORCH` (explicit or implicit device readback — breaks trace by forcing device sync), `FROM_TORCH` (device buffer allocation — breaks trace by introducing dynamic allocation into the static command stream), `PYTHON_BRANCH` (data-dependent Python control flow that cannot be captured as a static sequence)
  - Priority column: rank by (1) decode latency impact (how many milliseconds does eliminating this crossing save at B=1), (2) implementation complexity (simple TTNN composition vs. new kernel required)
  - Footer note: steps 1 and 6 (input/output projections, all-gather) are already trace-compatible and require no changes

- **`device_state_persistence.md`** — How the DeltaNet recurrent state S and conv state are currently managed between decode steps.
  - Describes the `TTNNQwenPagedAttentionKVCache` cache object and how it stores `recurrent_states` (dict mapping layer_idx to S tensor) and `conv_states` (dict mapping layer_idx to conv_state tensor) between decode calls
  - Documents the current data type of these stored tensors: plain PyTorch tensors (CPU or CUDA), not TTNN on-device tensors; this is the root cause of the host crossing at step 4 — the state is not on the Wormhole device between decode steps
  - Describes what must change: S and conv_state must be allocated as `ttnn.Tensor` objects on the Wormhole mesh device, persisting in DRAM between decode steps, never moved to host; state updates must be performed in-place (or by replacing the tensor binding) via TTNN ops only
  - Describes the state tensor shape and memory config needed for on-device persistence: S per layer [B, num_v_heads, d_k, d_v] = [1, 32, 128, 128] BF16 DRAM-resident; conv_state per layer [B, mixed_dim, conv_kernel_size] = [1, 8192, 4] BF16 DRAM-resident; both must be tile-aligned for `ttnn.matmul` compatibility (tile size 32×32 in BF16)
  - Notes that the state persistence change is a prerequisite for — and independent of — the kernel implementation changes; it can be implemented (as a refactor of the cache class) before the TTNN kernel implementations are complete

---

### Chapter 2 — TTNN Decomposition of the Recurrent Delta Rule Step

**Description:** Derives a complete TTNN-native implementation of the DeltaNet recurrent decode step as a composition of existing TTNN primitives, specifying tensor shapes, memory configs, and program configs at each operation, and identifies where a fused kernel provides a latency advantage over the composed form.

**Directory:** `ch2_ttnn_decomposition/`

**Files and content:**

- **`index.md`** — Chapter overview, learning objectives, and file navigation.
  - States the chapter goal: answer Q1 and Q2 from the guide spec — what the recurrence computes, and whether TTNN primitives can express it without host readback
  - Emphasizes that the decomposition in this chapter is the minimal correct implementation; Chapter 5 covers kernel fusion for latency
  - Lists the three section files in reading order

- **`recurrence_math_and_tensor_ops.md`** — The DeltaNet recurrence decomposed into named tensor operations with full shape annotations.
  - Restates the Gated DeltaNet recurrence for a single head at decode step t (cross-reference: Gated Delta Net guide, Ch2):
    ```
    S_decayed  = g_t * S_{t-1}                         [d_k, d_v] = [128, 128]
    retrieval  = S_decayed^T @ k̃_t                     [d_v] = [128]
    error      = β_t * (v_t − retrieval)               [d_v] = [128]
    write      = k̃_t ⊗ error                           [d_k, d_v] = [128, 128]  (outer product)
    S_t        = S_decayed + write                      [d_k, d_v] = [128, 128]
    o_t        = S_t @ q̃_t                             [d_v] = [128]
    ```
  - For the full multi-head batch: shapes become [B, num_v_heads, d_k, d_v] and [B, num_v_heads, d_v] and [B, num_v_heads, d_k]; operations are batched over B and num_v_heads
  - TTNN primitive for each named operation:
    - `S_decayed = g_t * S_{t-1}`: `ttnn.mul(g_t, S_prev)` where g_t is broadcast from [B, 1, num_v_heads, 1, 1] to [B, 1, num_v_heads, d_k, d_v]; `ttnn.multiply` with scalar broadcast; memory config: DRAM for S, L1 for result if possible
    - `retrieval = S_decayed^T @ k̃_t`: `ttnn.matmul(S_decayed, k̃_t, transpose_a=True)`; shapes [B, num_v_heads, d_k, d_v] × [B, num_v_heads, d_v, 1] → [B, num_v_heads, d_k, 1]; requires k̃_t reshaped to column vector
    - `error = β_t * (v_t − retrieval)`: `ttnn.sub(v_t, retrieval)` then `ttnn.mul(beta_t, error_raw)`; all [B, num_v_heads, d_v, 1] shapes
    - `write = k̃_t ⊗ error`: outer product via `ttnn.matmul(k̃_t, error^T)`; k̃_t reshaped [B, num_v_heads, d_k, 1], error^T [B, num_v_heads, 1, d_v]; result [B, num_v_heads, d_k, d_v]
    - `S_t = S_decayed + write`: `ttnn.add(S_decayed, write)`; output is new state tensor
    - `o_t = S_t @ q̃_t`: `ttnn.matmul(S_t, q̃_t)`; [B, num_v_heads, d_k, d_v] × [B, num_v_heads, d_v, 1] → [B, num_v_heads, d_k, 1]; reshape to [B, 1, num_v_heads × d_v] for downstream
  - Notes on tile alignment: d_k = d_v = 128 is divisible by the tile dimension (32); operations require TILE layout; any [d_v] or [d_k] vector must be padded to [d_v] = [128] rows (exact multiple of 32) — no padding needed for this configuration
  - Notes on memory config for each intermediate: S (DRAM, persistent), intermediate vectors (L1, ephemeral within the decode step), write matrix [d_k, d_v] (L1 if < 1.5 MB = 32 KB for single head — fits easily)

- **`ttnn_ops_per_step.md`** — Ordered list of TTNN operations for one complete decode step, with program config and memory config for each.
  - Operation sequence (continuing from the end of the projection and all-gather steps from Chapter 1):
    1. `ttnn.split` or `ttnn.slice`: split the all-gathered QKV output [B, 1, 8192] into Q [B, 1, 2048], K [B, 1, 2048], V [B, 1, 4096]; `[AVAILABLE]`
    2. `ttnn.reshape` + `ttnn.repeat_interleave` (or `ttnn.repeat`): reshape K/Q to [B, 1, 16, 128] then repeat to [B, 1, 32, 128]; reshape V to [B, 1, 32, 128]; `[AVAILABLE]`
    3. Decay gate: `ttnn.exp(ttnn.mul(-ttnn.exp(A_log), ttnn.softplus(ttnn.add(a_t, dt_bias))))` where a_t [B, 1, num_v_heads]; result g_t [B, 1, num_v_heads]; `[AVAILABLE — needs wiring]`
    4. Update rate: `ttnn.sigmoid(b_t)` → β_t [B, 1, num_v_heads]; `[AVAILABLE — needs wiring]`
    5. L2 normalize K̃ and Q̃: `ttnn.normalize_hw` or manual `ttnn.mul(K, ttnn.rsqrt(ttnn.sum(K*K, dim=-1, keepdim=True) + eps))`; `[AVAILABLE — needs wiring]`
    6. `S_decayed = ttnn.mul(g_t_broadcast, S_prev)`: g_t reshaped [B, num_v_heads, 1, 1] broadcast to state shape; `[AVAILABLE — needs wiring]`
    7. `retrieval = ttnn.matmul(S_decayed, k̃_reshaped)`: k̃ [B, num_v_heads, 128, 1]; output [B, num_v_heads, 128, 1]; `[AVAILABLE — needs wiring]`
    8. `error = ttnn.mul(beta_t_broadcast, ttnn.sub(v_reshaped, retrieval))`; `[AVAILABLE — needs wiring]`
    9. `write = ttnn.matmul(k̃_reshaped, error_transposed)`: outer product, output [B, num_v_heads, 128, 128]; `[AVAILABLE — needs wiring]`
    10. `S_new = ttnn.add(S_decayed, write)`: in-place if possible; `[AVAILABLE — needs wiring]`
    11. `o_t = ttnn.matmul(S_new, q̃_reshaped)`: output [B, num_v_heads, 128, 1]; `[AVAILABLE — needs wiring]`
    12. `ttnn.reshape(o_t, [B, 1, num_v_heads * d_v])`: flatten for gated RMSNorm; `[AVAILABLE]`
  - For each operation: table row with operation | TTNN API | input shapes | output shape | memory config (L1 or DRAM) | program config (MatmulMultiCoreReuseMultiCast or None) | availability tag
  - Total TTNN ops: 12 per decode step (excluding projections and all-gather); all are `[AVAILABLE]` with wiring changes; no new kernel development required for the composed form

- **`state_tensor_memory_config.md`** — Memory layout specification for the DeltaNet state matrix on T3K.
  - State tensor per layer (full, before sharding): [B, num_v_heads, d_k, d_v] = [1, 32, 128, 128] BF16 = 1,048,576 bytes ≈ 1 MB per layer
  - Under head-parallel sharding (4 heads per device): [B, 4, 128, 128] BF16 = 131,072 bytes ≈ 128 KB per device per layer
  - Memory config for persistent state: `ttnn.DRAM_MEMORY_CONFIG` with `ttnn.TILE_LAYOUT`; DRAM is required because state persists across decode steps and L1 is not preserved between kernel invocations
  - Why DRAM and not L1 for persistence: L1 SRAM on Wormhole is private per-core and is reclaimed between program dispatches; DRAM buffers allocated via `ttnn.allocate_tensor_on_device` persist until explicitly deallocated; DRAM state read/write is the dominant latency (analyzed in Chapter 4)
  - Tile alignment: d_k = d_v = 128; tile size = 32; 128/32 = 4 tiles per dimension; state tensor is exactly 4×4 tiles per head — no padding required; this is a favorable alignment case
  - L1 feasibility during kernel execution: per-head state [128, 128] × 2 bytes BF16 = 32 KB; a single Tensix core has 1.5 MB L1; the per-head state (32 KB) fits easily in one core's L1; for the fused kernel design in Chapter 5, keeping one head's state in L1 during the kernel's execution is feasible
  - Total DRAM for state across all 30 DeltaNet layers at B=1: 30 × 128 KB = 3.84 MB per device — negligible in the 12 GB DRAM budget
  - Conv state per layer (under sharding): [1, mixed_dim/8, conv_kernel_size] = [1, 1024, 4] BF16 = 8,192 bytes = 8 KB per device per layer; 30 layers = 240 KB total — trivial
  - Initialization: state S must be initialized to zeros on-device during model setup; use `ttnn.zeros` with the appropriate device and memory config

---

### Chapter 3 — Causal Conv1D and Gated RMSNorm Without Host Readback

**Description:** Derives on-device TTNN implementations for the two auxiliary host-crossing operations — the causal conv1d state update and the gated RMSNorm — and describes the specific TTNN primitives and tensor reshaping needed for each.

**Directory:** `ch3_auxiliary_ops/`

**Files and content:**

- **`index.md`** — Chapter overview, learning objectives, and file navigation.
  - States the chapter goal: close the remaining two host-crossing gaps (causal conv1d update and gated RMSNorm) after the recurrent delta rule step is resolved by Chapter 2
  - Notes that the causal conv1d update is simpler than the recurrence (it is a sliding-window operation on a fixed buffer) and can be expressed without any new kernel primitives
  - Notes that the gated RMSNorm is already fully composable from `ttnn.rms_norm`, `ttnn.silu`, and `ttnn.mul`; the gap is wiring, not kernel development
  - Lists the two section files in reading order

- **`causal_conv1d_update_ttnn.md`** — TTNN decomposition of the causal conv1d state update at decode time.
  - Mathematical definition of the causal conv1d decode update: for a 1D causal convolution with kernel size K=4 and a rolling state buffer `conv_state` [B, channels, K], the output for a new input vector x [B, channels, 1] is:
    ```
    conv_state[:, :, 0:K-1] = conv_state[:, :, 1:K]   (shift left by 1)
    conv_state[:, :, K-1]   = x                         (append new input)
    output = sum_{i=0}^{K-1} conv_weight[i] * conv_state[:, :, i]
    ```
    where `conv_weight` [channels, K] is the learned depthwise convolution weight (same weight applied independently per channel)
  - Channels = mixed_dim = key_dim × 2 + value_dim = 2048 × 2 + 4096 = 8192; K = 4; per-device channels (under sharding) = 8192 / 8 = 1024
  - TTNN implementation of the state shift: `ttnn.slice(conv_state, start=[0,0,1], end=[B,channels,K])` to get the last K-1 elements, then `ttnn.concat([shifted, x_reshaped], dim=-1)` to form the new state; or equivalently, implement as a roll operation via `ttnn.roll` if available, or manual slice + concat
  - TTNN implementation of the convolution output: `ttnn.sum(ttnn.mul(conv_state, conv_weight_broadcast), dim=-1)` where `conv_weight` [channels, K] is broadcast over the batch dimension; result [B, channels, 1]
  - Memory config for `conv_state`: DRAM-resident [B, channels/8, K] BF16 TILE_LAYOUT; K=4 is below the minimum tile dimension of 32; conv_state must be padded to [B, channels/8, 32] for TILE layout, using only the last K=4 columns; alternatively, keep in ROW_MAJOR layout (DRAM) which allows non-tile-aligned sizes
  - Trade-off: ROW_MAJOR DRAM is simpler (no padding) but slower for reads; TILE layout requires padding K from 4 to 32 (8× storage overhead for K dimension) but enables tile-based DMA; for K=4 and channels=1024 per device, the unpadded size is 8 KB and padded size is 65 KB — both are negligible; recommend TILE layout for consistency with the state matrix and matmul-based ops
  - Availability tag: `[AVAILABLE — needs wiring]` for all component ops (`ttnn.slice`, `ttnn.concat`, `ttnn.mul`, `ttnn.sum`); no new kernel required
  - Note on the decode path vs. prefill path: this chapter covers the decode update only (update one slot); the prefill conv1d (`causal_conv1d_fn`) processes the full sequence and is covered separately in Chapter 6

- **`gated_rmsnorm_ttnn.md`** — TTNN decomposition of the `FusedRMSNormSwishGate` operation.
  - Mathematical definition: given input x [B, 1, value_dim] and gate z [B, 1, value_dim]:
    ```
    x_normed = RMSNorm(x)                   (per-element, with learned weight w_norm [value_dim])
    gate_act = SiLU(z) = z * σ(z)           (Swish / SiLU activation)
    output   = x_normed * gate_act           (element-wise product)
    ```
    where value_dim = num_v_heads × d_v = 32 × 128 = 4096 for Qwen3.6-35B-A3B
  - TTNN implementation:
    - `ttnn.rms_norm(x, weight=w_norm, eps=1e-6)` → x_normed [B, 1, 4096]; `[AVAILABLE]`
    - `ttnn.silu(z)` → gate_act [B, 1, 4096]; `[AVAILABLE]`
    - `ttnn.mul(x_normed, gate_act)` → output [B, 1, 4096]; `[AVAILABLE]`
  - Memory config: all tensors L1-resident for this step (4096 × 2 bytes = 8 KB per tensor — well within L1)
  - Numerical equivalence: `ttnn.rms_norm` uses the standard formulation `x / sqrt(mean(x²) + ε) * w`; `ttnn.silu` computes `x * sigmoid(x)`; the composed form is numerically equivalent to the fused PyTorch `FusedRMSNormSwishGate` up to BF16 rounding; expected PCC > 0.999 against the reference
  - Fusion opportunity: `ttnn.rms_norm` + `ttnn.silu` + `ttnn.mul` can potentially be fused into a single kernel to reduce memory round-trips; this is a latency optimization rather than a correctness requirement; see Chapter 5 for fusion discussion
  - Availability tag: `[AVAILABLE — needs wiring]`; no new kernel required

---

### Chapter 4 — The `gdn_full_fused_inplace` Kernel: Reuse vs. Adapt

**Description:** Examines the `gdn_full_fused_inplace` custom TT-Metalium kernel referenced in the Qwen3.5-27B Blackhole implementation, documents what it computes and what architecture-specific assumptions it encodes, and determines how much can be reused on the Wormhole T3K versus what must be rewritten.

**Directory:** `ch4_gdn_fused_kernel/`

**Files and content:**

- **`index.md`** — Chapter overview, learning objectives, and file navigation.
  - States the chapter goal: answer Q3 from the guide spec — understand `gdn_full_fused_inplace`, assess reuse potential on T3K
  - Notes that this chapter informs the kernel development strategy in Chapter 5: reuse vs. rewrite is a binary that determines whether Chapter 5's recommended path is "port and tune" or "write from scratch"
  - Lists the two section files in reading order

- **`gdn_full_fused_inplace_analysis.md`** — Analysis of the `gdn_full_fused_inplace` kernel: what it computes, how it is structured, and what architecture it targets.
  - Source location: identify the file path in the tt-metal / tt-symbiote repository where `gdn_full_fused_inplace` is defined; expected location in `models/experimental/tt_symbiote/ops/` or `models/experimental/gdn/` (exact path to be verified during guide writing)
  - What it computes: the full 6-operation recurrent Gated DeltaNet decode step (decay, retrieve, error, outer product, add, output query) fused into a single TT-Metalium kernel that keeps the state in L1 during execution and streams input/output through L1 circular buffers without intermediate DRAM round-trips
  - Key kernel parameters to document: (1) tile size (likely 32×32 BF16), (2) number of heads processed per core, (3) CBs (circular buffers) used for each intermediate, (4) data format (BF16, FP32 accum, or mixed), (5) core grid used (how many Tensix cores), (6) explicit or implicit state layout (ROW_MAJOR vs TILE), (7) whether the state is read from / written to DRAM or L1
  - Architecture-specific assumptions to document:
    - Blackhole Tensix core configuration: number of FPUs, SFPU units, CB sizes, L1 size per core (Blackhole has larger L1 per core than Wormhole: 1.5 MB for Wormhole vs. 2 MB for Blackhole); any kernel that statically allocates CB space based on Blackhole L1 must be rechecked for Wormhole
    - RISCV data movement programs: the kernel likely uses DATA_MOVEMENT RISCV cores to issue DMA reads/writes; any architecture-specific address offset calculations or NOC routing assumptions must be verified against Wormhole
    - Accumulation format: Blackhole supports FP32 accumulation in FPUs natively; Wormhole has a different accumulation path; any kernel using `FP32_DEST_ACC` or equivalent compile-time flags must be checked
    - Matrix engine (FPU) dimensions: Wormhole FPU tile size is 32×32; if the kernel hardcodes Wormhole FPU dimensions, it is already compatible; if it hardcodes Blackhole-specific FPU widths (64×64 for Blackhole B0), it is not
  - Reuse assessment (3-level classification):
    - `REUSABLE_AS_IS`: the kernel compiles and passes correctness tests on Wormhole T3K without any source changes; this is the best case
    - `REUSABLE_WITH_TUNING`: the kernel is architecturally compatible with Wormhole but requires changes to CB size constants, core grid selection, or data format flags to match Wormhole's L1 and FPU specs; rewriting a handful of `constexpr` declarations is sufficient
    - `REQUIRES_REWRITE`: the kernel makes fundamental use of a Blackhole-specific hardware feature (e.g., a compute unit or NoC capability) not available on Wormhole; a new kernel must be written using the same algorithmic structure but targeting Wormhole

- **`wormhole_t3k_adaptation.md`** — Specific changes required to run `gdn_full_fused_inplace` (or an equivalent kernel) on Wormhole T3K.
  - If reuse classification is `REUSABLE_AS_IS`: document the verification test to confirm correctness (run against the PyTorch reference, compute PCC, check state matrix after N decode steps)
  - If reuse classification is `REUSABLE_WITH_TUNING`: list each required constant change with the Blackhole value, the Wormhole value, and the file location
  - If reuse classification is `REQUIRES_REWRITE`: provide a Wormhole-targeted kernel design sketch:
    - Core grid: use a (8, 4) or (4, 8) grid (available on Wormhole, 32 Tensix cores) to process heads in parallel; with 4 heads per T3K device (post sharding), assign one core per head and use the 4-core row
    - CB layout: CB0 = state S [128×128×2 = 32 KB], CB1 = k̃ input [128×2 = 256 bytes padded to tile], CB2 = v input, CB3 = g/β scalars, CBOUT = output o_t; total CB use per core ≈ 36 KB well within 1.5 MB L1
    - Compute program (RISCV compute core): implement the 6 operations using TT-Metalium compute API (`tile_matmul`, `add_tiles`, `mul_tiles_bcast_scalar`); keep state in CB0 across the 6 ops; DMA program reads S from DRAM into CB0, reads k̃/v/g/β from L1 (small), writes S back to DRAM and o_t to output buffer
    - Tile reads from DRAM: state [128×128] = 16 tiles (4×4); at DRAM bandwidth 288 GB/s, 32 KB read takes ~0.11 µs; per-head; ×4 heads = ~0.45 µs; ×30 layers = ~13.5 µs for the state reads alone — this is the dominant kernel latency
  - Multi-device sharding note: under head-parallel sharding, each T3K device processes its 4 heads independently; the kernel does not need cross-device communication; each device's kernel is identical and independent, which simplifies implementation
  - Availability tag for the outcome: either `[REUSABLE — port and tune]` or `[GAP — requires new kernel targeting Wormhole]`

---

### Chapter 5 — Scan and Recurrence Primitives Survey

**Description:** Surveys existing TTNN and tt-metal kernels for scan-like and recurrence-like operations — including the Mamba SSM selective scan kernel and any parallel prefix scan implementations — and determines whether any can be adapted for the DeltaNet state update, or whether the composed TTNN form from Chapter 2 is the correct starting point.

**Directory:** `ch5_scan_primitives_survey/`

**Files and content:**

- **`index.md`** — Chapter overview, learning objectives, and file navigation.
  - States the chapter goal: answer Q6 from the guide spec — inventory existing scan / recurrence kernels and assess DeltaNet adaptation potential
  - Notes the key structural difference between DeltaNet and SSMs: the DeltaNet state update is a rank-1 outer product write with a data-dependent retrieval step; SSM (Mamba) state update is a rank-1 outer product write with a scalar decay and no retrieval; this structural similarity motivates the survey
  - Lists the three section files in reading order

- **`mamba_ssm_kernel_review.md`** — Review of the Mamba SSM selective scan kernel in tt-metal for DeltaNet adaptation potential.
  - Source location: identify file paths for `ttnn.selective_scan` or equivalent Mamba-specific kernel in tt-metal (expected in `ttnn/cpp/ttnn/operations/experimental/ssm/` or similar); document the kernel's interface and what it computes
  - Mamba SSM state update (for reference): `h_t = A * h_t-1 + B_t * x_t` where A is a diagonal decay matrix (learned, per-channel), B_t is a per-step input-dependent vector [d_model → d_state], x_t is the input [d_model]; the output is `y_t = C_t * h_t` where C_t is a per-step query vector; the state h_t [d_model, d_state] has the same shape as DeltaNet's S [d_k, d_v]
  - Structural comparison to DeltaNet:
    - Similarity: both maintain a 2D state matrix; both update via outer product (B_t ⊗ x_t for Mamba, k̃_t ⊗ error_t for DeltaNet); both read from the state via a vector dot product (C_t^T h_t for Mamba, S^T k̃_t for DeltaNet)
    - Key difference: DeltaNet requires the retrieval `S^T k̃_t` inside the error computation before the write; Mamba's write does not depend on the current state (B_t ⊗ x_t is independent of h_t); this makes DeltaNet's recurrence non-separable in a way that Mamba's is not
    - Consequence for reuse: the Mamba selective scan kernel's inner loop is structurally different from DeltaNet's inner loop; the outer product accumulation idiom can be borrowed, but the retrieval-then-error step requires additional logic that the Mamba kernel does not contain
  - Reuse classification: `PARTIAL_REUSE — outer product and state-read idioms are borrowable; the kernel cannot be used as-is but provides reference patterns for the TT-Metalium implementation`
  - Document specific kernel patterns that can be borrowed: DMA pattern for streaming state from DRAM into L1 CB; tile-level matrix-vector multiply using `matmul_tiles`; outer product via a custom `outer_product_tiles` sequence or via `matmul_tiles` with transposed dimensions; scalar broadcast multiply via `mul_tiles_bcast_scalar`

- **`parallel_prefix_scan_review.md`** — Review of parallel prefix scan implementations in tt-metal for DeltaNet applicability.
  - Source location: search for prefix scan / parallel scan ops in `ttnn/cpp/ttnn/operations/` and `tt_metal/impl/`; document any found (expected: may exist for cumulative sum or associative scan in the context of Mamba or GLA)
  - DeltaNet scan feasibility analysis: the inter-chunk recurrence during prefill (S_{c+1} = g_C * S_c + Δ_C where Δ_C is a d_k × d_v matrix) is associative and could in principle be parallelized via a parallel prefix scan; however, each scan operand is a (d_k + 1) × d_v matrix (state plus correction), making the scan memory cost O(T/C × d_k × d_v); for T=8192, C=64, d_k=d_v=128: 128 scan operands × 128 × 128 × 2 bytes = 4 MB — acceptable if held in DRAM
  - Why parallel prefix scan is not the recommended path for decode: at decode time (T=1), there is only one recurrence step per layer; there is nothing to parallelize across; the scan is a single sequential update of S_t; parallel prefix scan primitives are irrelevant for decode
  - Why parallel prefix scan is not the recommended path for prefill either: the within-chunk operations (the WY-decomposition that handles intra-chunk token dependencies) are not expressible as a standard scalar-times-matrix associative scan; the inter-chunk scan is the only part that is associative, and for reasonable sequence lengths (T < 256K) the number of chunks (T/C = T/64) is small enough that a sequential Python loop over chunks is not a bottleneck compared to the within-chunk matmuls
  - Conclusion: no existing parallel prefix scan primitive in tt-metal can be directly adapted for DeltaNet; the recommended path for decode is the composed TTNN form (Chapter 2) or the fused kernel (Chapter 4 / Chapter 6); for prefill, a Python chunk loop calling TTNN matmuls is the correct first implementation

- **`gla_and_related_kernel_survey.md`** — Survey of GLA (Gated Linear Attention) and other linear attention kernels in tt-metal or tt-transformers.
  - Search for any existing TTNN or TT-Metalium implementation of GLA, RetNet, or vanilla linear attention (outside of DeltaNet); document findings
  - Expected finding: no existing tt-metal / tt-transformers kernel for GLA or RetNet; the only linear attention implementation is `TTNNQwen3LinearAttention` for DeltaNet, which currently falls back to `flash-linear-attention` Triton kernels
  - If any related kernel is found: assess whether its state-update idiom (outer product, state decay, state-vector multiply) is closer to DeltaNet's pattern than the Mamba kernel; update reuse assessment accordingly
  - Summary table: candidate kernel | what it computes | structural similarity to DeltaNet | reuse classification
  - Conclusion: the best path forward is either (a) wire the composed TTNN form from Chapter 2 as an immediate fix for trace compatibility, with latency acceptance pending measurement, or (b) port / adapt `gdn_full_fused_inplace` from Blackhole (Chapter 4) for optimal latency

---

### Chapter 6 — Latency Impact and Numerical Accuracy

**Description:** Measures the host-CPU round-trip latency for the current PyTorch fallback at decode B=1 (including device-to-host transfer, kernel execution, and host-to-device transfer), estimates the expected latency for a pure on-device implementation, and establishes PCC accuracy thresholds and sensitivity analysis for the DeltaNet state update.

**Directory:** `ch6_latency_and_accuracy/`

**Files and content:**

- **`index.md`** — Chapter overview, learning objectives, and file navigation.
  - States the chapter goal: answer Q5 and Q7 from the guide spec — measure host round-trip latency and establish accuracy requirements
  - Notes that the latency measurement section uses empirical profiling methodology (not just analytic estimates) and provides numbers that should be re-measured on the target hardware after implementation
  - Lists the three section files in reading order

- **`host_roundtrip_latency.md`** — Host-CPU round-trip latency breakdown for the current `recurrent_gated_delta_rule` fallback.
  - Latency components to measure and report:
    - `ttnn.to_torch` for Q̃, K̃, V from device to host: involves device-to-CPU memory copy + device synchronization; at B=1, tensor sizes [1, 1, 32, 128] × 3 = 3 × 8,192 bytes = 24,576 bytes total; PCIe bandwidth estimate: ~16 GB/s x86 PCIe 4.0 → ~1.5 µs theoretical; actual includes dispatch + sync overhead, expected 10–50 µs
    - `ttnn.to_torch` for S_prev from device: [1, 32, 128, 128] = 1,048,576 bytes BF16 ≈ 1 MB; at 16 GB/s PCIe → ~62.5 µs theoretical; actual may be 100–300 µs with sync overhead
    - `recurrent_gated_delta_rule` kernel execution on CPU/CUDA: for PyTorch-native (non-Triton) execution at B=1, expected 10–100 µs on CPU (small matrix ops); for Triton on GPU if available, much lower but not applicable on Wormhole inference machines
    - `ttnn.from_torch` for S_new and o_t back to device: [1, 32, 128, 128] + [1, 1, 32, 128] ≈ 1,057 KB ≈ 1 MB; PCIe upload ≈ 62.5 µs theoretical; actual 100–300 µs
    - Total estimated round-trip: 300–700 µs per DeltaNet layer × 30 layers ≈ 9–21 ms per decode step — this is a substantial fraction of the decode budget
  - How to measure: instrument with `time.perf_counter_ns()` around each `ttnn.to_torch` and `ttnn.from_torch` call and around the `recurrent_gated_delta_rule` call; run 100 warmup steps then report mean and p99 over 1000 steps; separate per-layer measurements from end-to-end decode step measurements
  - Note: these latency figures are **the primary motivation** for the on-device implementation; even if the composed TTNN form (Chapter 2) has higher DRAM bandwidth utilization than the fused kernel (Chapter 4), it eliminates the PCIe transfer completely and is expected to reduce the DeltaNet contribution to decode latency by ~10-20× at B=1

- **`on_device_latency_estimate.md`** — Expected latency for the pure on-device TTNN implementation.
  - Analytic estimate for the composed TTNN form (12 ops from Chapter 2):
    - Dominant cost: state S read from DRAM [128 KB per device at 4 heads × 128 × 128 × 2 bytes] + state write back [128 KB]; at 288 GB/s DRAM bandwidth: 256 KB / 288 GB/s ≈ 0.9 µs per layer; × 30 layers = ~27 µs for all DeltaNet state reads/writes
    - Kernel launch overhead for 12 ops × 30 layers = 360 TTNN op dispatches; dispatch latency per op ≈ 1–5 µs (empirical from other TTNN models); total dispatch overhead ≈ 0.36–1.8 ms — this is the dominant cost for the composed form, not DRAM bandwidth
    - Recommendation: dispatch overhead is the primary motivation for the fused kernel (Chapter 4); a single fused kernel reduces 12 dispatches to 1 per layer, from 360 total dispatches to 30
  - Analytic estimate for the fused kernel form:
    - 30 TTNN op dispatches total; at 5 µs each → 150 µs dispatch overhead
    - State DRAM read/write: 30 × 256 KB / 288 GB/s = 26.7 µs
    - Total estimated latency: ~177 µs for all 30 DeltaNet layers — compared to 9–21 ms for the host fallback, a 50–120× improvement
  - Prefill note: prefill uses `chunk_gated_delta_rule` (sequence length T); a Python loop over T/64 chunks calling TTNN matmuls is discussed here; at T=8192, there are 128 chunks; within each chunk, the dominant matmuls are [64, 128] × [128, 128] = 8×8×4 tiles; tile matmul latency on Wormhole ≈ ~10 µs per matmul (conservative estimate); 128 chunks × ~4 matmuls per chunk × 10 µs = ~5 ms per layer; ×30 layers = ~150 ms for prefill DeltaNet; this is within the expected prefill budget and does not need to be optimized before decode

- **`pcc_accuracy_thresholds.md`** — Acceptable PCC accuracy for the TTNN recurrence kernel and sensitivity analysis.
  - PCC (Pearson Correlation Coefficient) definition: `PCC(a, b) = cov(a, b) / (std(a) * std(b))`; used throughout tt-metal testing to quantify numerical similarity between reference and test tensors; a PCC of 1.0 means perfect linear correlation; a PCC of 0.999 means near-identical tensors with very small relative error
  - Standard PCC thresholds in tt-transformers testing:
    - 0.9999 for lossless operations (reshape, permute, simple elementwise)
    - 0.999 for matmul-derived operations (matrix multiply, attention output)
    - 0.99 for operations with significant BF16 rounding (long chains of accumulations)
    - 0.98 for acceptable under BF16 (some fused ops, long sequences)
  - Recommended PCC threshold for the DeltaNet recurrent state update: 0.999 per decode step (measuring S_new against the PyTorch reference); this is achievable in BF16 for the 6-operation sequence since no long accumulation chains are involved
  - Sensitivity of model output to state errors: DeltaNet's recurrence is a fixed-point-like state update (the delta rule minimizes error between the state's prediction and the target value); small numerical errors in S_new do not accumulate exponentially because the decay gate g_t < 1 contracts the state toward zero each step; errors from step t are multiplied by g_t^{T-t} by step T; with g_t ≈ 0.9 (typical value), a step-t error is reduced by (0.9)^{T-t} at the final step; for T=100 decode steps, initial errors are reduced by (0.9)^{100} ≈ 2.7×10^{-5} — negligible
  - PCC measurement methodology: write a test that runs 200 decode steps with the TTNN implementation and the PyTorch reference implementation in parallel; compare S tensor and o_t output after each step; report per-step PCC and cumulative drift (L2 norm of S_ttnn - S_pytorch after each step)
  - Acceptable model-level degradation: downstream metric is next-token prediction accuracy (perplexity or top-1 accuracy); if per-step PCC > 0.999, model-level degradation is expected to be < 0.1 perplexity point (estimate; to be verified with end-to-end evaluation); this is within the acceptable margin for BF16 vs. FP32 quantization error, which is already known to be acceptable for Qwen3 inference

---

### Chapter 7 — Implementation Roadmap and Trace Integration

**Description:** Consolidates findings from all prior chapters into a prioritized implementation roadmap for replacing the `TTNNQwen3LinearAttention` host fallback with a pure on-device TTNN implementation, describes the changes to state tensor lifecycle management, and specifies the verification steps needed before declaring Metal Trace compatibility.

**Directory:** `ch7_implementation_roadmap/`

**Files and content:**

- **`index.md`** — Chapter overview, learning objectives, and file navigation.
  - States the chapter goal: synthesize everything into a concrete plan that a developer can follow to implement the on-device DeltaNet decode step and achieve Metal Trace compatibility
  - Emphasizes that this chapter does not introduce new concepts; it references findings from Chapters 1–6 to derive a sequenced task list
  - Lists the three section files in reading order

- **`task_list_and_priority.md`** — Ordered implementation task list with priority, estimated complexity, and cross-chapter references.
  - **Task 1 (Priority: Critical, Complexity: Low) — Refactor state tensor storage to on-device TTNN tensors:**
    - Change `TTNNQwenPagedAttentionKVCache.recurrent_states` and `.conv_states` from dicts of PyTorch tensors to dicts of `ttnn.Tensor` objects allocated on the Wormhole mesh device via `ttnn.zeros` + `ttnn.allocate_tensor_on_device`
    - Memory config: DRAM, TILE layout; shapes as specified in Chapter 2, `state_tensor_memory_config.md`
    - This task eliminates the `ttnn.to_torch(S_prev)` and `ttnn.from_torch(S_new)` round-trips; it is a prerequisite for all subsequent tasks and can be implemented before any kernel changes
    - References: Ch1 `device_state_persistence.md`, Ch2 `state_tensor_memory_config.md`
  - **Task 2 (Priority: High, Complexity: Low) — Wire decay gate and update rate ops to TTNN:**
    - Replace Python-level `torch.exp`, `torch.softplus`, `torch.sigmoid` scalar computations for g_t and β_t with `ttnn.exp`, `ttnn.softplus`, `ttnn.sigmoid` on on-device tensors
    - Requires `in_proj_a` and `in_proj_b` output tensors to remain on-device (not moved to host for the scalar ops); verify that current implementation does not do `ttnn.to_torch` on these small tensors
    - References: Ch1 `forward_pass_walkthrough.md` (Step 3), Ch2 `ttnn_ops_per_step.md` (ops 3–4)
  - **Task 3 (Priority: High, Complexity: Medium) — Wire causal conv1d update to TTNN:**
    - Replace `causal_conv1d_update` C extension call with TTNN `ttnn.slice` + `ttnn.concat` + `ttnn.mul` + `ttnn.sum` sequence
    - Requires conv_state to be on-device (Task 1 prerequisite); requires conv_weight to be loaded to device as a TTNN tensor during model setup
    - References: Ch3 `causal_conv1d_update_ttnn.md`
  - **Task 4 (Priority: High, Complexity: Low) — Wire gated RMSNorm to TTNN:**
    - Replace `FusedRMSNormSwishGate` module call with `ttnn.rms_norm` + `ttnn.silu` + `ttnn.mul` composition
    - No new kernel needed; all primitives are `[AVAILABLE]`
    - References: Ch3 `gated_rmsnorm_ttnn.md`
  - **Task 5 (Priority: Critical, Complexity: Medium) — Wire recurrent delta rule step to TTNN:**
    - Replace `recurrent_gated_delta_rule` call with the 6-operation TTNN sequence from Chapter 2
    - Requires Tasks 1 and 2 to be complete (state on-device, g_t and β_t on-device); write state update in-place by overwriting the DRAM tensor via `ttnn.copy` or `ttnn.assign`
    - Use the composed form initially (12 TTNN ops per step); dispatch overhead will be measured and compared against the fused kernel option from Task 6
    - References: Ch2 `recurrence_math_and_tensor_ops.md`, Ch2 `ttnn_ops_per_step.md`
  - **Task 6 (Priority: Medium, Complexity: High) — Port or implement fused `gdn_full_fused_inplace` kernel for Wormhole:**
    - Evaluate `gdn_full_fused_inplace` per Chapter 4 analysis; if `REUSABLE_WITH_TUNING`, port and tune; if `REQUIRES_REWRITE`, implement a new TT-Metalium kernel using the design sketch from Ch4 `wormhole_t3k_adaptation.md`
    - This task is a latency optimization over Task 5; it is not required for trace compatibility, only for achieving the expected throughput improvement
    - Target: reduce DeltaNet decode latency from ~1 ms (composed TTNN form with dispatch overhead) to ~177 µs (fused form); see Ch6 `on_device_latency_estimate.md`
  - **Task 7 (Priority: Low, Complexity: Medium) — Python chunk loop for prefill (`chunk_gated_delta_rule` in TTNN):**
    - Implement a Python loop over T/64 chunks that calls TTNN matmuls for the within-chunk WY-decomposition and inter-chunk state transfer
    - Not required for decode trace compatibility; needed for end-to-end prefill on-device execution
    - Expected prefill latency: ~150 ms for 30 layers at T=8192 (see Ch6 `on_device_latency_estimate.md`)

- **`trace_integration_checklist.md`** — Step-by-step checklist for integrating the on-device DeltaNet decode into Metal Trace.
  - Prerequisites before trace integration: Tasks 1–5 from `task_list_and_priority.md` must be complete and passing correctness tests (PCC > 0.999 per step)
  - Step 1 — Pre-allocate state tensors as on-device TTNN tensors during model setup (warm-up phase), not inside the trace capture bracket; state is initialized to zeros and updated in-place during decode; buffer handles persist across trace replays
  - Step 2 — Ensure no `ttnn.from_torch` or `ttnn.to_torch` is called inside the trace bracket; verify by wrapping the trace capture in a guard that raises on any host-tensor creation
  - Step 3 — Verify that in-place state update (`ttnn.copy` or `ttnn.assign` writing S_new into the persistent DRAM buffer) is compatible with trace replay: in-place writes to pre-allocated DRAM buffers are trace-compatible because the buffer address is fixed at capture time and does not change across replays
  - Step 4 — Run `ttnn.begin_trace_capture` / `ttnn.end_trace_capture` on the full decoder stack (including DeltaNet layers and full-attention layers); confirm no errors raised during capture
  - Step 5 — Execute trace for 10 decode steps; compare each step's output against the non-traced reference; PCC must exceed 0.999 at each step
  - Step 6 — Run a 1000-step decode loop with trace; verify that state does not diverge (L2 norm of S_traced - S_reference should remain bounded, not growing with step count)
  - Step 7 — Profile the traced decode loop with Tracy; confirm that the DeltaNet layers no longer appear as CPU-side execution gaps in the device trace timeline
  - Step 8 — Measure per-step decode latency with and without trace; confirm the expected ~10–20× reduction in DeltaNet contribution to decode latency
  - Note: the `synchronize_device` removal (the other prerequisite for full-stack trace, from the companion topic "trace-safe cos/sin pre-replication") must also be complete before end-to-end trace works; this guide's implementation is independent of that fix and either can be done first

- **`verification_and_testing.md`** — Test matrix and verification methodology for the on-device DeltaNet decode implementation.
  - Unit test for each task:
    - Task 1 (state on-device): verify S tensor is a `ttnn.Tensor` with correct shape, memory config, and layout; verify it persists between two sequential decode calls without host transfer
    - Task 3 (conv1d TTNN): run `causal_conv1d_update` reference vs. TTNN slice+concat+mul+sum on 100 random inputs; assert PCC > 0.9999 (exact match expected since this is exact arithmetic, not matrix multiply)
    - Task 4 (gated RMSNorm TTNN): run `FusedRMSNormSwishGate` reference vs. `ttnn.rms_norm + ttnn.silu + ttnn.mul`; assert PCC > 0.999
    - Task 5 (recurrent delta rule TTNN): run `recurrent_gated_delta_rule` reference vs. TTNN 6-op sequence for 200 steps; assert per-step output PCC > 0.999 and per-step state PCC > 0.999; assert cumulative state drift (L2 norm) is bounded
    - Task 6 (fused kernel): same test as Task 5 but comparing fused kernel output vs. PyTorch reference; also compare fused kernel vs. composed TTNN form to verify consistency
  - Integration test: run a 10-layer prefix of the Qwen3.6-35B-A3B decoder (5 DeltaNet layers + 5 full-attention layers) on a T3K with the full on-device implementation; compare logits against the reference (mixed PyTorch/TTNN) implementation; assert PCC > 0.99 on logits
  - Trace-specific test: run the integration test with trace enabled; assert that `ttnn.execute_trace` completes without error and that outputs match the non-traced run within PCC > 0.99
  - Performance regression test: assert that per-step decode latency for the DeltaNet contribution (30 layers) is < 500 µs with the composed TTNN form and < 200 µs with the fused kernel form (measured targets based on Chapter 6 estimates; update thresholds after empirical measurement)

---

## 3. Conventions

### Notation

The following symbols are used consistently across all chapters:

| Symbol | Meaning |
|---|---|
| H | Model hidden dimension (2048 for Qwen3.6-35B-A3B) |
| T | Sequence length (number of tokens) |
| B | Batch size |
| d_k | Key/query head dimension (128) |
| d_v | Value head dimension (128) |
| num_v_heads | Number of value heads per DeltaNet layer (32 for 9B; 4 for 35B-A3B — use model-specific value) |
| num_k_heads | Number of key/query heads per DeltaNet layer (16; repeated to match num_v_heads) |
| mixed_dim | Dimension of the combined QKV projection output (key_dim × 2 + value_dim = 2048 × 2 + 4096 = 8192) |
| S | DeltaNet recurrent state matrix, S ∈ R^{d_k × d_v} per head (128 × 128) |
| g_t | Scalar decay gate at step t, g_t = exp(α_t) ∈ (0,1] |
| β_t | Update rate at step t, β_t = σ(b_t) ∈ (0,1) |
| k̃_t, q̃_t | L2-normalized key and query vectors, shape [d_k] = [128] per head |
| K | Causal conv1d kernel size (K = 4) |
| value_dim | Total value projection output dimension (num_v_heads × d_v = 4096) |
| PCC | Pearson Correlation Coefficient; used as the numerical accuracy metric |
| BF16 | Brain Float 16 — the primary data format for weights and activations on Wormhole |

### Memory Config Notation

Memory configuration is described using a standard two-field tuple throughout this guide:

- `(DRAM, TILE)` — tensor stored in off-chip GDDR6 DRAM, in tile layout (32×32 BF16 tiles); the standard layout for persistent tensors (state matrix, KV cache)
- `(L1, TILE)` — tensor stored in on-chip L1 SRAM, in tile layout; the standard layout for ephemeral intermediate tensors that fit in L1 and do not persist across kernel invocations
- `(DRAM, ROW_MAJOR)` — tensor stored in DRAM in row-major (non-tiled) byte layout; used when tensor dimensions are not tile-aligned; avoided where possible because it forces scalar DMA reads

### Availability Tags

Each TTNN operation or kernel is labeled with one of:

| Tag | Meaning |
|---|---|
| `[AVAILABLE]` | The TTNN API exists and is currently used in the model code |
| `[AVAILABLE — needs wiring]` | The TTNN API exists but is not yet connected in `TTNNQwen3LinearAttention`; no kernel development needed |
| `[PARTIAL REUSE]` | An existing kernel (e.g., Mamba SSM) shares algorithmic structure but cannot be used as-is; patterns can be borrowed |
| `[REUSABLE — port and tune]` | The `gdn_full_fused_inplace` kernel can be adapted for Wormhole with constant changes only |
| `[GAP — requires new kernel]` | No existing op covers this operation; a new TT-Metalium kernel must be written |

### Formatting Rules

- **All TTNN API names** are written in `monospace code` with fully qualified form (e.g., `ttnn.matmul`, `ttnn.rms_norm`, `ttnn.begin_trace_capture`)
- **All tensor shapes** use bracket notation with named dimensions (e.g., `[B, num_v_heads, d_k, d_v]`); never use unnamed dimension labels without defining them first
- **All equations** use the notation table above; any newly introduced symbol is defined immediately before the equation in which it first appears; equations involving matrix operations use `@` for matrix multiply and `⊗` for outer product
- **Numeric examples** use Qwen3.6-35B-A3B configuration (H=2048, d_k=d_v=128, num_v_heads=32, num_k_heads=16, 40 layers, 30 DeltaNet + 10 full-attention) as the primary reference; the 9B variant is noted where it differs
- **FLOPs and bytes calculations** are always shown step-by-step with intermediate quantities labeled; no unexplained final numbers
- **Hardware specs** always cite the chip variant and configuration (e.g., "Wormhole chip in T3K — one ASIC at 288 GB/s DRAM bandwidth, 1.5 MB L1 per Tensix core")
- **Source code references** use file-relative paths from the tt-metal repository root in `monospace code` with the class or function name appended after a colon, e.g., `models/experimental/tt_symbiote/modules/qwen_attention.py:TTNNQwen3LinearAttention`
- **Callout blocks** use blockquote syntax with a bold label: `> **Note:**`, `> **Warning:**`, `> **Key insight:**`; no emoji in any file
- **Each file** begins with an H1 title and a one-paragraph orientation stating what the reader will know by the end of the file
- **Every chapter's `index.md`** ends with a "What's next" section listing files in reading order

### Terminology

- "Host readback" or "host crossing" means any operation that moves tensor data from the Wormhole device to CPU memory or triggers a host-side device synchronization; these terms are synonymous in this guide
- "Metal Trace" or "trace" refers to the `ttnn.begin_trace_capture` / `ttnn.end_trace_capture` / `ttnn.execute_trace` API; "trace-compatible" means the operation can be included inside a `begin_trace_capture` / `end_trace_capture` bracket without breaking replay correctness
- "Composed TTNN form" refers to the 12-operation TTNN sequence from Chapter 2 that implements the recurrent delta rule step using existing TTNN primitives (no new kernel); it is trace-compatible after Tasks 1–5
- "Fused kernel" refers to a single TT-Metalium custom kernel that implements all 6 operations of the recurrent delta rule step in one dispatch (either a ported `gdn_full_fused_inplace` or a newly written Wormhole kernel); it is also trace-compatible after implementation but has lower dispatch overhead than the composed form
- "DeltaNet" and "Gated Delta Net" are used interchangeably in this guide; "DeltaNet" is the shorter form used in code identifiers and table rows; "Gated Delta Net" is used in prose and section titles
- "State matrix" always refers to S ∈ R^{d_k × d_v} for a single head; "full state tensor" refers to [B, num_v_heads, d_k, d_v] across all heads and batch
- "T3K" refers to the Tenstorrent T3000 system with 4 × Wormhole n300 cards (8 Wormhole chips total) in a 1×8 logical mesh; "per device" means per Wormhole ASIC (not per n300 card)
- "Prefill" = processing the full prompt (T > 1) using the chunked variant; "decode" = generating one token (T = 1) using the recurrent variant; these terms match the convention in the companion Gated Delta Net guide
- "PCC" is always Pearson Correlation Coefficient computed element-wise over flattened tensors; the computation is `numpy.corrcoef(a.flatten(), b.flatten())[0, 1]` or the TTNN test utility equivalent

---

## 4. Cross-Chapter Dependencies

| Chapter | Depends on | Concepts carried forward |
|---|---|---|
| Ch 1 — Why the Current Implementation Breaks Trace | (none; optional: companion Gated Delta Net guide Ch2 for recurrence math) | Host-crossing classification table; list of all on-device vs. host-side operations in `TTNNQwen3LinearAttention`; state persistence problem definition; trace-break mechanisms taxonomy |
| Ch 2 — TTNN Decomposition of the Recurrent Delta Rule Step | Ch 1 (the operation sequence from the walkthrough defines what needs to be replaced; state persistence problem from Ch1 drives the memory config choices) | Complete TTNN operation sequence (12 ops) for the recurrent step; tensor shapes for Q̃, K̃, V, g, β, S at each intermediate; state tensor memory config specification; L1 feasibility result for per-head state during kernel execution |
| Ch 3 — Causal Conv1D and Gated RMSNorm Without Host Readback | Ch 1 (identifies conv1d and gated RMSNorm as host-crossing gaps to fix), Ch 2 (establishes on-device tensor lifecycle model that conv_state must join) | TTNN conv1d decode op sequence (slice + concat + mul + sum); TTNN gated RMSNorm op composition (rms_norm + silu + mul); tile alignment constraints for conv_state; availability tags for all ops |
| Ch 4 — The `gdn_full_fused_inplace` Kernel | Ch 2 (the 6-operation structure defined in Ch2 is what the fused kernel also implements; the L1 feasibility result from Ch2 establishes that the state fits in L1 during kernel execution) | Reuse classification (`REUSABLE_WITH_TUNING` or `REQUIRES_REWRITE`); Wormhole adaptation list (specific constant changes or full rewrite design sketch); core grid and CB layout for the Wormhole kernel |
| Ch 5 — Scan and Recurrence Primitives Survey | Ch 2 (the composed TTNN form is the baseline; the survey asks whether any existing kernel is a better starting point than the composed form) | Reuse classification for Mamba SSM kernel (`PARTIAL_REUSE`); conclusion that no existing scan primitive directly applies to decode; GLA kernel survey result; confirmation that composed TTNN form (Ch2) is the correct immediate implementation |
| Ch 6 — Latency Impact and Numerical Accuracy | Ch 1 (host crossing latency estimates depend on the tensor sizes documented in Ch1); Ch 2 (on-device latency estimates use the 12-op count and DRAM memory config from Ch2); Ch 4 (fused kernel latency estimate uses the core grid and dispatch count from Ch4) | Host round-trip latency measurement (300–700 µs per layer, 9–21 ms total); on-device composed form latency estimate (~1 ms total, dispatch-dominated); on-device fused kernel latency estimate (~177 µs total); PCC threshold (0.999 per step); error decay argument (g_t < 1 bounds cumulative drift) |
| Ch 7 — Implementation Roadmap and Trace Integration | All prior chapters | Task list with cross-chapter references; trace integration checklist; verification test matrix; performance regression thresholds |

**Specific forward references to flag during writing:**

- Ch 1 (`device_state_persistence.md`) references Ch 2 for the exact memory config that the on-device state tensor requires
- Ch 2 (`ttnn_ops_per_step.md`) references Ch 4 for the fused kernel alternative to the 12-op composed form and Ch 5 for the scan primitive survey conclusion
- Ch 2 (`state_tensor_memory_config.md`) references Ch 7 (Task 1) for the code change needed to implement on-device state persistence
- Ch 3 (`causal_conv1d_update_ttnn.md`) references Ch 7 (Task 3) for the implementation task and Ch 1 (Task 1 prerequisite) for state on-device requirement
- Ch 4 (`wormhole_t3k_adaptation.md`) references Ch 2 (`state_tensor_memory_config.md`) for the DRAM bandwidth figure that drives the core grid sizing
- Ch 5 (`parallel_prefix_scan_review.md`) references Ch 2 (`recurrence_math_and_tensor_ops.md`) for the derivation of why the DeltaNet recurrence is not associative in the raw form
- Ch 6 (`host_roundtrip_latency.md`) references Ch 1 (`host_crossing_summary_table.md`) for the tensor sizes that determine PCIe transfer time
- Ch 6 (`on_device_latency_estimate.md`) references Ch 4 for the fused kernel dispatch count and Ch 2 for the composed form dispatch count
- Ch 6 (`pcc_accuracy_thresholds.md`) references Ch 7 (`verification_and_testing.md`) for the test methodology that measures PCC
- Ch 7 (`task_list_and_priority.md`) references Ch 4 (Task 6 fused kernel) and Ch 6 (latency targets) for the priority ordering
- Ch 7 (`trace_integration_checklist.md`) references the companion topic "trace-safe cos/sin pre-replication in TTNNQwen3FullAttention" as the other prerequisite for full-stack trace capture that is independent of this guide's scope
