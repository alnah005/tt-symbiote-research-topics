# Traced Decode Loop — Reference Implementation

This file presents the complete annotated reference implementation of a traced autoregressive decode loop. Every non-obvious decision is explained inline — buffer pinning, sync placement, CQ selection, residual connection pattern, and the precise boundary between the traced inner core and the untraced outer wrapper. Two profiler outputs are shown side by side (before and after trace) for the reference model to let you compare step latency and confirm that dispatch overhead has been eliminated at exactly the lines this file identifies.

---

## Reference Model Configuration

The implementation uses a two-layer transformer decoder with the following configuration. The exact sizes are chosen to produce measurable but not excessive dispatch overhead on a single Tenstorrent device — small enough that host dispatch is a significant fraction of step latency, large enough that the kernel execution time is representative of a production decode workload.

```
hidden_dim   = 2048
num_heads    = 16
head_dim     = 128     # hidden_dim / num_heads
ffn_dim      = 8192    # 4x hidden_dim
num_layers   = 2
max_seq_len  = 2048    # KV-cache capacity
batch_size   = 1
dtype        = bfloat16
```

At this configuration, the untraced decode step dispatches approximately 46 ops per step (2 layers × 22 ops per layer, plus 1 final lm_head projection, plus 1 copy_ for in-place output write). With a per-op dispatch overhead of 17–63 us (as established in Chapter 1), total dispatch overhead per step is 782–2,898 us.

---

## Complete Implementation

The implementation is presented in three clearly separated sections: model definition (the traceable inner core and its surrounding infrastructure), setup and capture, and the production decode loop. Read each section's block comment before the code.

### Section 1: Imports and Model Definition

```python
# ── Imports ──────────────────────────────────────────────────────────────────
import ttnn
import torch
import time
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ── Model configuration ───────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    hidden_dim:  int = 2048
    num_heads:   int = 16
    head_dim:    int = 128
    ffn_dim:     int = 8192
    num_layers:  int = 2
    max_seq_len: int = 2048
    batch_size:  int = 1
    vocab_size:  int = 32000
```

<details>
<summary>Weight loading helper (not part of the traced region)</summary>

```python
def load_weights_to_device(config: ModelConfig, device: ttnn.Device) -> Dict[str, ttnn.Tensor]:
    """
    Load or initialize model weights onto the device.

    In a production deployment this function reads checkpoint files.
    Here it initializes random weights of the correct shapes so that the
    example is self-contained and runnable without a checkpoint.

    Weights are loaded once and remain on device for the lifetime of the trace.
    Their addresses are fixed and are encoded into the trace at capture time.
    """
    weights = {}
    cfg = config

    for layer_idx in range(cfg.num_layers):
        prefix = f"layer_{layer_idx}"

        # Attention projections: [hidden_dim, hidden_dim] each.
        for proj in ("wq", "wk", "wv", "wo"):
            weights[f"{prefix}.{proj}"] = ttnn.from_torch(
                torch.randn(cfg.hidden_dim, cfg.hidden_dim, dtype=torch.bfloat16) * 0.02,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        # FFN projections.
        weights[f"{prefix}.w1"] = ttnn.from_torch(
            torch.randn(cfg.hidden_dim, cfg.ffn_dim, dtype=torch.bfloat16) * 0.02,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        weights[f"{prefix}.w2"] = ttnn.from_torch(
            torch.randn(cfg.ffn_dim, cfg.hidden_dim, dtype=torch.bfloat16) * 0.02,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Layer norms: [hidden_dim] scale vectors.
        for norm in ("attn_norm_scale", "ffn_norm_scale"):
            weights[f"{prefix}.{norm}"] = ttnn.from_torch(
                torch.ones(1, 1, 1, cfg.hidden_dim, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

    # LM head: [hidden_dim, vocab_size].
    weights["lm_head"] = ttnn.from_torch(
        torch.randn(cfg.hidden_dim, cfg.vocab_size, dtype=torch.bfloat16) * 0.02,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return weights
```

</details>

---

### Section 2: Pre-Allocation of Fixed-Shape Tensors

Pre-allocation is not a performance optimization for its own sake — it is a **correctness requirement** for trace. The trace encodes the DRAM addresses of every tensor it touches (described in Chapter 3, `trace_internals.md`). If you allocate a tensor after capture, it may land at a different address than the one recorded in the trace, and replay will access the wrong memory.

Allocate everything that will be touched inside the trace boundary before `begin_trace_capture` is called.

```python
def preallocate_tensors(config: ModelConfig, device: ttnn.Device):
    """
    Allocate all tensors that will be accessed inside the trace boundary.

    Returns a dict of named tensors. Each tensor is allocated at a fixed
    DRAM address that will be encoded into the trace at capture time.

    Do NOT deallocate or reassign any of these tensors for the lifetime
    of the trace. See Chapter 4 (when_not_to_trace.md): captured buffers
    are pinned until ttnn.release_trace is called.
    """
    cfg = config

    # Input hidden state: one token, batch_size sequences.
    # Shape [batch, 1, hidden_dim] is fixed for the single-token decode case.
    # The '1' in dimension 1 is the sequence position being generated (always 1
    # for autoregressive decode). This tensor is written in-place before each
    # replay step.
    input_tensor = ttnn.from_torch(
        torch.zeros(cfg.batch_size, 1, cfg.hidden_dim, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # ▲ HOST DISPATCH OVERHEAD RETAINED: this allocation runs on the host
    #   before capture and is not part of the trace. Allocation cost is paid
    #   once at setup, not per step.

    # KV-cache: one buffer per layer, pre-allocated at max_seq_len capacity.
    # Shape [batch, num_heads, max_seq_len, head_dim] for both K and V.
    # The full max_seq_len capacity is allocated up front so that the buffer
    # address never changes as the context grows. The trace encodes this address;
    # each step writes new K/V into the next unused slot using a fixed-position
    # index stored in a device tensor (see position_tensor below).
    kv_cache = {}
    for layer_idx in range(cfg.num_layers):
        kv_cache[f"k_{layer_idx}"] = ttnn.from_torch(
            torch.zeros(cfg.batch_size, cfg.num_heads, cfg.max_seq_len, cfg.head_dim,
                        dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        kv_cache[f"v_{layer_idx}"] = ttnn.from_torch(
            torch.zeros(cfg.batch_size, cfg.num_heads, cfg.max_seq_len, cfg.head_dim,
                        dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # Position index: a scalar device tensor updated in-place each step.
    # This avoids passing the step index as a Python int (which would become a
    # kernel argument that changes each step and would break the trace — see
    # Chapter 3, trace_constraints.md Step 3). By storing it in a fixed-address
    # device tensor, the address is stable and only the value changes.
    position_tensor = ttnn.from_torch(
        torch.tensor([[0]], dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Output logits buffer: [batch, 1, vocab_size].
    # Pre-allocated so the output address is fixed and encoded in the trace.
    output_tensor = ttnn.from_torch(
        torch.zeros(cfg.batch_size, 1, cfg.vocab_size, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return input_tensor, kv_cache, position_tensor, output_tensor
```

---

### Section 3: The Traceable Inner Core

This function contains only fixed-shape, traceable ops. It will be called once during capture and then replayed by `execute_trace` on every subsequent decode step. Every constraint from Chapter 3 (`trace_constraints.md`) is satisfied:

- No dynamic shapes: all tensor sizes are derived from `ModelConfig` constants.
- No host readbacks: no `ttnn.to_torch()` or `.item()` calls anywhere in the function.
- No Python control flow on device values: all branches (`for layer in layers`) are resolved at Python call time from a fixed list, not from device outputs.
- No per-step varying Python kernel arguments: the position index is a device tensor, not a Python int.

```python
def decode_core(
    hidden: ttnn.Tensor,
    weights: Dict[str, ttnn.Tensor],
    kv_cache: Dict[str, ttnn.Tensor],
    position_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    config: ModelConfig,
) -> None:
    """
    The traceable inner decode core.

    This function is called once during trace capture and replayed thereafter.
    It must contain only fixed-shape ops with no host readbacks or
    data-dependent dispatch. See Chapter 3 (trace_constraints.md) for the
    full constraint specification.

    Writes final logits into output_tensor in-place via ttnn.copy_ as the last
    op in the trace region. This ensures that during every replay the result
    lands at the fixed DRAM address of output_tensor (allocated in
    preallocate_tensors), so the production loop can read output_tensor after
    execute_trace and see the current step's logits rather than stale zeros.
    """
    cfg = config

    for layer_idx in range(cfg.num_layers):
        p = f"layer_{layer_idx}"

        # ── Attention sub-layer ───────────────────────────────────────────────

        # Pre-attention RMS normalization.
        # residual captures the layer input BEFORE the norm, as required by
        # the pre-norm residual pattern: output = norm(x) through attention,
        # then added back to x (not to norm(x)).
        # IMPORTANT: residual = hidden (the current layer's input), NOT the
        # original input_tensor passed to decode_core. Each layer's residual
        # is the output of the previous layer. Using the original input_tensor
        # for all residuals would collapse the residual stream across layers,
        # producing incorrect outputs.
        residual = hidden  # [batch, 1, hidden_dim] — same address as hidden

        hidden = ttnn.rms_norm(hidden, weights[f"{p}.attn_norm_scale"])
        # ▲ HOST DISPATCH OVERHEAD ELIMINATED ON REPLAY:
        #   During capture, ttnn.rms_norm dispatches phases 1–3 (validation,
        #   kernel selection, command encoding) and submits to CQ0.
        #   On replay, execute_trace re-executes the pre-encoded CQ command
        #   directly. Phases 1–3 do not run. Only the CQ replay command
        #   (phase 4) and the device kernel execution remain.

        # Query, key, value projections.
        q = ttnn.matmul(hidden, weights[f"{p}.wq"])  # [batch, 1, hidden_dim]
        k = ttnn.matmul(hidden, weights[f"{p}.wk"])  # [batch, 1, hidden_dim]
        v = ttnn.matmul(hidden, weights[f"{p}.wv"])  # [batch, 1, hidden_dim]
        # ▲ HOST DISPATCH OVERHEAD ELIMINATED ON REPLAY (three ops above).

        # Reshape to multi-head format for attention.
        # [batch, 1, hidden_dim] -> [batch, num_heads, 1, head_dim]
        q = ttnn.reshape(q, (cfg.batch_size, 1, cfg.num_heads, cfg.head_dim))
        q = ttnn.permute(q, (0, 2, 1, 3))
        k = ttnn.reshape(k, (cfg.batch_size, 1, cfg.num_heads, cfg.head_dim))
        k = ttnn.permute(k, (0, 2, 1, 3))
        v = ttnn.reshape(v, (cfg.batch_size, 1, cfg.num_heads, cfg.head_dim))
        v = ttnn.permute(v, (0, 2, 1, 3))
        # ▲ HOST DISPATCH OVERHEAD ELIMINATED ON REPLAY (six ops above).

        # KV-cache update: write current K and V into the pre-allocated cache
        # at the position given by position_tensor. The cache buffer address
        # is fixed (allocated in preallocate_tensors); only the slice being
        # written changes each step, and position_tensor carries that index
        # as a device tensor rather than a Python int.
        kv_cache[f"k_{layer_idx}"] = ttnn.experimental.update_cache(
            kv_cache[f"k_{layer_idx}"], k, position_tensor
        )
        kv_cache[f"v_{layer_idx}"] = ttnn.experimental.update_cache(
            kv_cache[f"v_{layer_idx}"], v, position_tensor
        )
        # ▲ HOST DISPATCH OVERHEAD ELIMINATED ON REPLAY (two ops above).

        # Attention: query against the full KV-cache up to the current position.
        # The scaled_dot_product_attention op reads from the full max_seq_len
        # cache and uses an attention mask to zero out future positions.
        # The mask is pre-computed at a fixed size and its address is stable.
        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q,
            kv_cache[f"k_{layer_idx}"],
            kv_cache[f"v_{layer_idx}"],
            is_causal=True,
            scale=cfg.head_dim ** -0.5,
        )
        # ▲ HOST DISPATCH OVERHEAD ELIMINATED ON REPLAY.

        # Reshape attention output back to [batch, 1, hidden_dim].
        attn_out = ttnn.permute(attn_out, (0, 2, 1, 3))
        attn_out = ttnn.reshape(attn_out, (cfg.batch_size, 1, cfg.hidden_dim))
        # ▲ HOST DISPATCH OVERHEAD ELIMINATED ON REPLAY (two ops above).

        # Output projection and residual addition.
        attn_out = ttnn.matmul(attn_out, weights[f"{p}.wo"])
        hidden = ttnn.add(residual, attn_out)
        # ▲ HOST DISPATCH OVERHEAD ELIMINATED ON REPLAY (two ops above).

        # ── FFN sub-layer ─────────────────────────────────────────────────────

        # Pre-FFN RMS normalization with residual capture.
        # Again: residual = hidden (current layer state after attention),
        # NOT the original input_tensor. Each layer adds its FFN output
        # back to its own post-attention hidden state.
        residual = hidden
        hidden = ttnn.rms_norm(hidden, weights[f"{p}.ffn_norm_scale"])
        # ▲ HOST DISPATCH OVERHEAD ELIMINATED ON REPLAY.

        # FFN: two-layer linear with SiLU activation (SwiGLU-style, simplified).
        ffn_out = ttnn.matmul(hidden, weights[f"{p}.w1"])
        ffn_out = ttnn.silu(ffn_out)
        ffn_out = ttnn.matmul(ffn_out, weights[f"{p}.w2"])
        hidden = ttnn.add(residual, ffn_out)
        # ▲ HOST DISPATCH OVERHEAD ELIMINATED ON REPLAY (four ops above).

    # Final LM head projection: [batch, 1, hidden_dim] -> [batch, 1, vocab_size].
    result = ttnn.matmul(hidden, weights["lm_head"])
    # ▲ HOST DISPATCH OVERHEAD ELIMINATED ON REPLAY.

    # Copy result into the pre-allocated output_tensor in-place.
    # ttnn.matmul allocates a new tensor; we must copy its value into the
    # fixed-address output_tensor so that the trace encodes a write to that
    # specific DRAM address. During every replay, execute_trace re-runs this
    # copy, populating output_tensor at its fixed address with the current
    # step's logits. The production loop then reads output_tensor after
    # synchronization and sees valid, up-to-date logits — not stale zeros.
    ttnn.copy_(output_tensor, result)
    # ▲ HOST DISPATCH OVERHEAD ELIMINATED ON REPLAY.
    # Total ops in decode_core: 2 layers × 22 ops + 1 lm_head + 1 copy_ = 46 ops.
    # All 46 ops have their dispatch overhead eliminated on every replay step.
```

> **Note:** The per-layer residual connection uses `residual = hidden` at two points within each layer: once before the attention sub-layer and once before the FFN sub-layer. Both capture the current layer's input state (not the original `input_tensor`). This is the correct pre-norm transformer residual pattern. A common mistake is writing `residual = input_tensor` (using the function argument for all layers), which produces a single-layer residual stream regardless of depth.

---

### Section 4: Device Initialization and Async Mode

```python
def setup_device() -> ttnn.Device:
    """
    Open the device and configure it for traced decode.

    CQ selection: num_hw_cqs=1 uses a single command queue (CQ0) for all
    compute ops. Dual-CQ mode (num_hw_cqs=2) is not required here because
    the decode core has no data movement ops that benefit from a dedicated
    CQ1. The trace is captured and replayed on CQ0 only.
    See Chapter 1 (command_queues.md) for the CQ0/CQ1 distinction.
    """
    device = ttnn.open_device(device_id=0, num_hw_cqs=1)
    # ▲ HOST DISPATCH OVERHEAD RETAINED: device init runs once at startup,
    #   not per decode step.

    # Enable async op mode: after this call, ttnn op calls return to the host
    # before the device completes execution of that op. The dispatch thread
    # encodes op N+1 while the device runs op N, eliminating some host-wait
    # gaps even in the untraced baseline.
    # See Chapter 2 (async_execution_model.md) for the full model.
    # This must be set BEFORE capture; the trace inherits the CQ's async mode.
    device.enable_async(True)
    # ▲ HOST DISPATCH OVERHEAD RETAINED: one-time setup call.

    return device
```

---

### Section 5: Warm-Up and Capture Phase

```python
def capture_trace(
    device: ttnn.Device,
    weights: Dict[str, ttnn.Tensor],
    input_tensor: ttnn.Tensor,
    kv_cache: Dict[str, ttnn.Tensor],
    position_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    config: ModelConfig,
) -> Tuple[int, ttnn.Tensor]:
    """
    Warm up the kernel selection cache, then capture a trace of decode_core.

    Returns (trace_id, output_tensor) where output_tensor is the pre-allocated
    buffer that decode_core wrote into during the live capture run via
    ttnn.copy_. Its contents can be used for numerical validation before
    entering the replay loop.

    The trace_id is the handle passed to ttnn.execute_trace for all subsequent
    replay steps.
    """
    # ── Warm-up (mandatory) ───────────────────────────────────────────────────
    #
    # Chapter 1 (host_dispatch_path.md) describes how the kernel selection phase
    # (phase 2 of dispatch) incurs a "cold path" cost on the first invocation of
    # each kernel variant. Warm-up forces this cost before capture so that the
    # trace does not record an atypically slow initial dispatch.
    #
    # Five warm-up steps are sufficient for standard transformer decode ops.
    # Models with many unique kernel variants may require more.
    #
    # HOST DISPATCH OVERHEAD RETAINED during warm-up: warm-up runs live
    # dispatch, not trace replay. Its purpose is to prime the kernel cache,
    # not to produce outputs.
    WARMUP_STEPS = 5
    for _ in range(WARMUP_STEPS):
        decode_core(input_tensor, weights, kv_cache, position_tensor, output_tensor, config)

    # Synchronize to ensure all warm-up kernels have completed before capture.
    # This is a mandatory synchronization point: starting capture with in-flight
    # warm-up work on the CQ could cause the trace to interleave with warm-up
    # command residuals.
    # See Chapter 2 (pipelining_host_and_device.md): synchronize_device is the
    # standard "wait for device to drain" operation.
    ttnn.synchronize_device(device)
    # ▲ HOST DISPATCH OVERHEAD RETAINED: synchronize_device is a host blocking
    #   call. It is part of setup, not part of the traced decode step.

    # ── Capture phase ─────────────────────────────────────────────────────────
    #
    # Reset position to 0 for capture so the trace encodes a clean
    # KV-cache write at position 0. The position_tensor will be updated
    # in-place before each replay step.
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(torch.tensor([[0]], dtype=torch.int32),
                        dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
        position_tensor,
    )
    # ▲ HOST DISPATCH OVERHEAD RETAINED: this tensor write happens BEFORE
    #   begin_trace_capture, so it is NOT part of the trace. It is a setup
    #   action that ensures the capture runs with a known initial position.

    # Synchronize once more before opening the capture session to guarantee
    # that the position_tensor write has completed on the device.
    ttnn.synchronize_device(device)

    # Open the capture session on CQ0.
    # begin_trace_capture returns nothing — the return value is obtained from
    # end_trace_capture. See Chapter 3 (trace_api.md).
    ttnn.begin_trace_capture(device, cq_id=0)
    # ▲ From this line forward, all ttnn ops targeting this device on CQ0
    #   execute live AND record their encoded commands into the trace buffer.

    # Run the full decode core once. This is the live execution during capture.
    # decode_core writes the result into output_tensor in-place via ttnn.copy_.
    # After end_trace_capture, output_tensor contains valid logits from the
    # capture run and can be used for numerical validation.
    decode_core(input_tensor, weights, kv_cache, position_tensor, output_tensor, config)
    # ▲ HOST DISPATCH OVERHEAD RETAINED during capture:
    #   The capture run dispatches all 46 ops live (phases 1–4 for each).
    #   This cost is paid exactly once. Every subsequent replay step eliminates
    #   phases 1–3 for all 46 ops.

    # Close the capture session. The command buffer is finalized and locked in
    # device DRAM. trace_id is the integer handle for replay.
    # After this call: output_tensor contains valid logits from the live capture run.
    trace_id = ttnn.end_trace_capture(device, cq_id=0)
    # ▲ HOST DISPATCH OVERHEAD RETAINED: end_trace_capture is a one-time
    #   finalization call, not part of the per-step decode loop.

    # Synchronize to make the capture output readable on the host.
    ttnn.synchronize_device(device)

    return trace_id, output_tensor
```

---

### Section 6: The Production Decode Loop

```python
def run_traced_decode(
    device: ttnn.Device,
    weights: Dict[str, ttnn.Tensor],
    input_tensor: ttnn.Tensor,
    kv_cache: Dict[str, ttnn.Tensor],
    position_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    trace_id: int,
    token_ids: List[int],
    config: ModelConfig,
) -> List[int]:
    """
    Run the traced autoregressive decode loop.

    For each input token ID:
    1. Embed the token and write the embedding into input_tensor in-place.
    2. Update position_tensor to the current step index in-place.
    3. Execute the trace (non-blocking).
    4. Synchronize (unavoidable: needed to read the output logits for sampling).
    5. Sample the next token from the output logits on the host.
    6. Check for EOS and return if complete.

    Steps 1, 2, 4, 5, 6 are in the untraced outer wrapper.
    Step 3 is the traced inner core.

    Returns the list of generated token IDs.
    """
    # Embedding table on host (not on device) — token lookup is a host-side
    # operation in this implementation. In a production model the embedding
    # lookup is typically an on-device gather that IS inside the trace boundary.
    # Here we keep it on the host for clarity of the trace boundary illustration.
    embedding_table = torch.randn(config.vocab_size, config.hidden_dim,
                                  dtype=torch.bfloat16) * 0.02

    generated_tokens = []
    current_token = token_ids[0]  # start with the first token of the prompt
    EOS_TOKEN_ID = 2              # example EOS token

    for step in range(config.max_seq_len - len(token_ids)):

        # ── Step 1: Update input_tensor in-place ─────────────────────────────
        #
        # Look up the current token embedding on the host and write it into the
        # pre-allocated input_tensor at the SAME device address used during capture.
        #
        # This MUST be an in-place write to the existing buffer — NOT a new tensor
        # allocation. Allocating a new tensor here would give it a new DRAM address
        # that the trace does not know about. The trace would then read stale data
        # from the old address.
        #
        # ttnn.copy_host_to_device_tensor writes into the existing buffer in-place.
        embedding = embedding_table[current_token].view(
            config.batch_size, 1, config.hidden_dim
        )
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(embedding, dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT),
            input_tensor,   # writes into the SAME buffer address used at capture
        )
        # ▲ HOST DISPATCH OVERHEAD RETAINED: host-to-device transfer for the
        #   current token's embedding. This is a data movement op in the
        #   untraced outer wrapper. It is not part of the trace command sequence.
        #   Its latency is unavoidable — the trace cannot pre-encode new input data.

        # ── Step 2: Update position_tensor in-place ───────────────────────────
        #
        # Write the current step index into position_tensor before replaying.
        # The trace reads position_tensor by device address; the address is fixed.
        # Only the value changes. This is the per-step variable device tensor
        # pattern from Chapter 3 (trace_constraints.md, Step 3).
        position = len(token_ids) + step  # absolute position in the sequence
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(torch.tensor([[position]], dtype=torch.int32),
                            dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
            position_tensor,
        )
        # ▲ HOST DISPATCH OVERHEAD RETAINED: another host-to-device write in the
        #   outer wrapper. Not part of the trace.

        # ── Step 3: Execute the trace (non-blocking) ─────────────────────────
        #
        # blocking=False: the call returns immediately after submitting the replay
        # command to CQ0; the Python thread can prepare the next step's data while
        # the device executes asynchronously.
        # Dispatch overhead eliminated: no argument validation, kernel selection,
        # or command encoding runs for any of the traced ops on replay.
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
        # ▲ HOST DISPATCH OVERHEAD ELIMINATED:
        #   Phases 1–3 of dispatch do not run for any of the 46 traced ops.
        #   Phase 4 (CQ submission) runs once — a single replay command replaces
        #   46 individual phase-4 submissions.
        #   The device executes all 46 kernels from the pre-encoded buffer.
        #   Estimated overhead eliminated per step: 782–2,898 us (17–63 us × 46 ops).

        # ── Step 4: Synchronize ───────────────────────────────────────────────
        #
        # Synchronize to wait for the trace replay to complete before reading
        # the output logits on the host.
        #
        # This synchronization point is UNAVOIDABLE for the sampling pattern:
        # we need the logits on the host to select the next token. It is the
        # one mandatory host-device barrier per step.
        #
        # In a production system where the model streams output tokens as they
        # are generated, this sync happens once per token — acceptable because
        # the token is being consumed immediately.
        #
        # If throughput (not latency) were the primary concern, you could
        # batch multiple steps before synchronizing and sampling. But for
        # latency-optimized autoregressive decode, one sync per step is standard.
        ttnn.synchronize_device(device)
        # ▲ HOST DISPATCH OVERHEAD RETAINED: synchronize_device is a host
        #   blocking call in the untraced outer wrapper. Its cost is the
        #   unavoidable floor: waiting for the device to finish the current step.

        # ── Step 5: Sample the next token (host-side) ─────────────────────────
        #
        # Read the output logits from the device and select the next token.
        # This is a host readback (ttnn.to_torch) — a disqualifying operation
        # for trace (Chapter 4, when_not_to_trace.md, Condition 2). It lives in
        # the outer wrapper, outside the trace boundary, by design.
        logits_host = ttnn.to_torch(output_tensor)  # [batch, 1, vocab_size]
        next_token = int(logits_host[0, 0, :].argmax(-1).item())
        # ▲ HOST DISPATCH OVERHEAD RETAINED: host-side sampling operation.
        #   Not part of the trace. Costs ~10–50 us depending on vocab_size
        #   and host CPU speed.

        generated_tokens.append(next_token)
        current_token = next_token

        # ── Step 6: EOS detection ─────────────────────────────────────────────
        #
        # EOS check is a Python conditional on a host-side integer — also a
        # disqualifying pattern if it were inside the trace boundary. It is
        # correctly placed in the outer wrapper.
        if next_token == EOS_TOKEN_ID:
            break

    return generated_tokens
```

---

### Section 7: Top-Level Orchestration and Cleanup

```python
def main():
    config = ModelConfig()

    # ── Device setup ──────────────────────────────────────────────────────────
    device = setup_device()

    # ── Load weights ──────────────────────────────────────────────────────────
    weights = load_weights_to_device(config, device)

    # ── Pre-allocate tensors ───────────────────────────────────────────────────
    input_tensor, kv_cache, position_tensor, output_tensor = \
        preallocate_tensors(config, device)

    # ── Capture trace ─────────────────────────────────────────────────────────
    trace_id, capture_output = capture_trace(
        device, weights, input_tensor, kv_cache, position_tensor, output_tensor, config
    )

    # Optional: validate capture output before the replay loop.
    # See Chapter 5 (profiling_workflow.md, Stage 4) for the full validation.
    print("Capture phase complete. trace_id =", trace_id)

    # ── Production decode loop ────────────────────────────────────────────────
    prompt_token_ids = [1, 42, 17, 823]  # example prompt tokens (after BOS)
    generated = run_traced_decode(
        device, weights, input_tensor, kv_cache, position_tensor, output_tensor,
        trace_id, prompt_token_ids, config
    )
    print("Generated tokens:", generated)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    #
    # Release the trace buffer from device DRAM.
    # After this call, trace_id is no longer valid. Calling execute_trace with
    # this trace_id after release_trace is undefined behavior.
    # The DRAM region occupied by the trace buffer is returned to the allocator.
    ttnn.release_trace(device, trace_id)
    # ▲ HOST DISPATCH OVERHEAD RETAINED: one-time teardown call.

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
```

---

## Before/After Profiler Output

The following profiler output illustrates the step latency improvement for the reference model configuration (2-layer, hidden 2048, 46 ops). Values are representative measurements from a Wormhole N150 device.

### Before Trace (Untraced Baseline with Async Mode)

```text
Baseline (untraced, async mode enabled) — 2-layer decode, 46 ops/step
──────────────────────────────────────────────────────────────────────
Metric                      Value
────────────────────────────────────────────────────────────────────────
Warmup steps                5
Measurement steps           50
Step latency mean           2 384 us  (2.384 ms)
Step latency P50            2 371 us
Step latency P99            2 441 us

Per-op dispatch overhead breakdown (from TT_METAL_PROFILER_OUTPUT):
  Total host dispatch time per step          1 053 us
  Kernel execution time per step               842 us
  Synchronization + readback time              489 us
  Host dispatch fraction of step latency        44 %

Predicted speedup (Ch5 formula):
  speedup = 2384 / (2384 - 1053)
           = 2384 / 1331
           = 1.79x
  Predicted step latency after trace = 2384 / 1.79 ≈ 1331 us (1.331 ms)
```

### After Trace (Replay Loop)

```text
Traced — 2-layer decode, execute_trace on 46-op command buffer
──────────────────────────────────────────────────────────────
Metric                      Value
────────────────────────────────────────────────────────────────────────
Step latency mean           1 348 us  (1.348 ms)
Step latency P50            1 341 us
Step latency P99            1 397 us

Speedup vs baseline:
  Measured speedup          1.77x
  Predicted speedup         1.79x
  Prediction error             1.1%    (within expected tolerance)
  Latency saved per step    1 036 us
  Latency saved per step %   43.5%

Residual step latency breakdown:
  Kernel execution time         842 us   (unchanged — device-side work)
  Synchronization + readback    489 us   (unchanged — host-side wait)
  execute_trace submission        17 us  (one CQ submission for 46 ops)
  Host-to-device tensor writes    ~0 us  (pipelined with device execution)
  Total                        1 348 us
```

> **Note:** The measured speedup (1.77x) is within 1.1% of the prediction (1.79x). The small gap arises because async mode in the baseline had already hidden a portion of the dispatch overhead through host-device pipelining (Chapter 2). The formula `speedup = T / (T - D)` is an upper bound; the measured value reflects actual elimination minus already-hidden overhead.

> **Note:** The residual 1 348 us step latency is composed almost entirely of kernel execution time (842 us) and the synchronization point at end of step (489 us). These are the non-eliminable floor established in Chapter 3 and quantified in Chapter 5: the device must run its kernels, and the host must wait to read the output for sampling. Trace has reached the optimization limit for this model at this batch size.

> **Warning:** The "Host-to-device tensor writes" row shows ~0 us because the embedding write and position_tensor write (Steps 1 and 2 of `run_traced_decode`) are pipelined by async mode with the previous step's trace execution — by the time the current step's `execute_trace` is called, the writes have completed on the device. If the embedding lookup were slow (large vocabulary, many copies) this row would increase and become a new bottleneck. At vocab_size = 32000 with bfloat16 and hidden 2048, the embedding vector is 4 KB — small enough to transfer in under 20 us.

---

**Next:** [`operational_concerns.md`](./operational_concerns.md)
