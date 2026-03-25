# Prefill/Decode Pipeline

This file covers the end-to-end LLM inference pipeline as implemented in tt-transformers: the chunked prefill loop, the single-token decode loop, batching strategies, and the warm-up and trace capture sequence required before production serving begins. The goal is to give an accurate, implementation-level picture of how prefill and decode calls are structured in code, with the API signatures and loop bounds pinned to confirmed facts.

For the attention kernels invoked inside these forward functions, see Ch2. For the matmul program configs and weight formats used inside each forward call, see Ch3.

---

## The Two Forward Functions

tt-transformers exposes two primary forward functions for LLM inference: one for prefill and one for decode. They have different signatures, different input shapes, and different performance characteristics.

### `prefill_forward_text`

[confirmed] Signature:

```python
def prefill_forward_text(
    tokens: Tensor,       # [batch, seq_len]  — padded token ids for all sequences in the batch
    page_table: Tensor,   # [batch, max_pages] — page table mapping sequence positions to KV blocks
    kv_cache_len: int,    # number of KV positions already populated before this prefill call
    prompt_lens: list[int],  # actual (unpadded) prompt length for each sequence in the batch
) -> Tensor:              # [batch, 1, vocab_size]
```

Key points:

- `tokens` is padded to a common `seq_len` within the batch, but the model only processes up to `prompt_lens[i]` real tokens for sequence `i`. Padding positions are ignored during chunked prefill iteration (see below).
- `kv_cache_len` is the number of token positions already present in the KV cache from a prior prefill call (relevant when prefill is called in multiple phases, for example during speculative decoding recovery or multi-turn conversations). For a fresh prefill from position 0, `kv_cache_len=0`.
- The return shape `[batch, 1, vocab_size]` contains the logits for the final real token of each sequence — the token that will be used to sample the first generated token.

### `decode_forward`

[confirmed] Signature:

```python
def decode_forward(
    tokens: Tensor,         # [batch, 1]  — one new token per sequence
    page_table: Tensor,     # [batch, max_pages]
    current_pos: int,       # scalar int — the position index being decoded (same for all sequences in the batch)
    enable_trace: bool = True,  # whether to use the pre-captured TTNN execution trace
) -> Tensor:                # [batch, vocab_size]
```

Key points:

- `tokens` has shape `[batch, 1]`: one token per sequence per decode step.
- `current_pos` is a **single scalar integer**, not a list or per-sequence tensor. [confirmed] This means all sequences in the batch are treated as being at the same position during a given decode step. In static batching (see Batching section), this is exact because all sequences start together and advance together. In continuous batching, per-sequence position tracking requires a different mechanism outside this function.
- `enable_trace=True` is the default for production decode. [confirmed] The trace replay path is significantly faster than the non-trace path because it bypasses the TTNN Python dispatch overhead on every step. Trace capture must have been performed before `enable_trace=True` is used (see Warm-Up and Trace Capture section).
- The return shape `[batch, vocab_size]` is the unnormalized logit vector for each sequence; the caller applies sampling or argmax to select the next token.

---

## Chunked Prefill Mechanics

### Why Chunking Is Necessary

During prefill the model processes the entire prompt in one or more forward passes. For short prompts a single `prefill_forward_text` call suffices. For long prompts — especially in 64K or 131K context configurations — the attention score matrix for a single pass may not fit in the L1 SRAM of the Tensix cores assigned to the attention kernel.

[confirmed] The maximum chunk length is controlled by `MAX_PREFILL_CHUNK_SIZE`. This is a model- and hardware-specific constant. The constraint comes from the attention score computation: for a chunk of length `C` attending to a KV sequence of growing length, the score matrix (Q tiles × KV tiles) must fit in the 120 KB L1 per core. Larger chunk sizes increase the per-core L1 pressure and can cause correctness failures or OOM if set too large.

### The Chunked Prefill Loop

[confirmed] The loop bound for chunked prefill is `prompt_lens.max().item()`, not the padded tensor dimension `tokens.shape[1]`. This is a critical distinction: iterating up to `tokens.shape[1]` would process padding tokens and corrupt the KV cache with spurious entries.

A correct chunked prefill loop structure:

```python
chunk_size = MAX_PREFILL_CHUNK_SIZE
max_prompt_len = prompt_lens.max().item()   # confirmed: loop bound is this, NOT tokens.shape[1]

for chunk_start in range(0, max_prompt_len, chunk_size):
    chunk_end = min(chunk_start + chunk_size, max_prompt_len)

    # Slice the token tensor to the current chunk window.
    # For sequences shorter than chunk_end, the extra positions in this chunk
    # are padding and will be masked by prompt_lens inside prefill_forward_text.
    chunk_tokens = tokens[:, chunk_start:chunk_end]   # [batch, chunk_len]

    logits = prefill_forward_text(
        tokens=chunk_tokens,
        page_table=page_table,
        kv_cache_len=chunk_start,   # positions already written to KV cache before this chunk
        prompt_lens=[
            max(0, min(pl, chunk_end) - chunk_start)
            for pl in prompt_lens
        ],
    )
    # logits is used only from the last chunk (when the final real token is processed).
```

The loop only iterates as far as the longest real prompt in the batch. Sequences that are shorter than the current `chunk_start` have already been fully processed; their tokens for this chunk are padding and are handled by the model's internal masking logic.

### KV Cache Growth During Chunked Prefill

Each chunk call appends new K and V vectors to the paged KV cache starting at position `kv_cache_len` (passed as the `kv_cache_len` argument). The page allocator must have pre-assigned sufficient physical blocks for the full prompt length before the first chunk call. If a sequence exhausts its allocated blocks mid-prefill, the page table update path (described in Ch2) allocates additional blocks between chunk calls.

[INFERRED] For very long contexts (e.g., 131K tokens at `MAX_PREFILL_CHUNK_SIZE=512`), chunked prefill performs approximately 256 forward passes. The latency of the prefill phase (TTFT) scales linearly with the number of chunks. Increasing `MAX_PREFILL_CHUNK_SIZE` reduces TTFT but increases per-chunk L1 pressure; the optimal chunk size is model- and context-length-specific.

---

## Decode Loop Structure

### Single-Step Decode

Each decode step takes the most recently generated token for each sequence in the batch, runs `decode_forward`, and samples the next token:

```python
# After prefill, the first generated token is sampled from the prefill logits.
next_tokens = sample(prefill_logits)   # [batch, 1]

current_pos = max_prompt_len           # confirmed: scalar int, not per-sequence

for step in range(max_new_tokens):
    logits = decode_forward(
        tokens=next_tokens,            # [batch, 1]
        page_table=page_table,
        current_pos=current_pos,       # confirmed: single scalar int
        enable_trace=True,             # confirmed: production path
    )
    next_tokens = sample(logits)       # [batch, 1]
    current_pos += 1                   # advance position scalar by 1 each step
```

The KV cache update (writing the new K and V vectors for the current token at `current_pos`) happens inside `decode_forward`. The page table must have a valid block entry for `current_pos // block_size` before `decode_forward` is called; if a new block boundary is crossed, the host allocates a new page and updates `page_table` before the call.

### `current_pos` Is a Scalar

[confirmed] `current_pos` is a single scalar integer, not a list or tensor. This design choice has two implications:

1. All sequences in the batch are assumed to be at the same decode position. In static batching (all sequences started at the same time and advanced together), this is exact. In configurations where sequences have different prompt lengths or started at different times, the caller must manage position alignment externally.

2. The trace compiled for `decode_forward` embeds `current_pos` semantics at the kernel level. Because TTNN traces are captured for a specific configuration, the trace assumes a single uniform position per step. Per-sequence position tracking (as in `cur_pos_tensor` for the underlying `paged_scaled_dot_product_attention_decode` kernel described in Ch2) is handled inside the implementation of `decode_forward`, not exposed as a per-sequence argument at this API level.

---

## Warm-Up and Trace Capture

Correct initialization of tt-transformers for production serving requires two distinct phases: JIT warm-up and trace capture. These are separate operations with different purposes and must be called in the correct order.

### Phase 1: JIT Warm-Up (Compilation Only)

[confirmed] `warmup_model_decode()` and `warmup_model_prefill()` both perform JIT kernel compilation only. They do **not** capture execution traces.

When these functions are called, TTNN compiles the RISC-V kernel binaries for all ops in the forward graph and caches them in `TT_METAL_CACHE`. Subsequent calls with the same op configurations (shapes, dtypes, program configs) reuse the cached binaries without recompilation.

```python
# Warm-up compiles all kernels. No trace is captured here.
model.warmup_model_decode()    # confirmed: JIT compilation only, no trace capture
model.warmup_model_prefill()   # confirmed: JIT compilation only, no trace capture
```

[INFERRED] In the reference implementation, `warmup_model_decode()` is called before `warmup_model_prefill()`. The confirmed facts do not establish a required ordering between the two; the observed sequence in the codebase may reflect convention rather than a hard dependency. Implementers should consult the tt-transformers source to confirm whether ordering matters for their configuration.

The warm-up calls use real (or dummy) input tensors of the correct shapes to drive compilation. The output of warm-up calls is discarded; their sole purpose is to populate the kernel cache.

### Phase 2: Trace Capture

[confirmed] Trace capture is a separate step performed using `ttnn.begin_trace_capture` and `ttnn.end_trace_capture`.

```python
import ttnn

# Capture the decode trace after warm-up has completed.
trace_id = ttnn.begin_trace_capture(device, cq_id=0)
_ = model.decode_forward(
    tokens=dummy_tokens,
    page_table=page_table,
    current_pos=0,
    enable_trace=False,   # run eagerly during capture, not via a prior trace
)
ttnn.end_trace_capture(device, trace_id, cq_id=0)
```

After capture, subsequent calls with `enable_trace=True` replay the recorded command sequence without re-invoking the Python dispatch path. This is the mechanism behind the production performance of the decode loop: replaying a trace eliminates Python overhead and reduces dispatch latency per decode step.

### Precision Changes Invalidate All Traces

[confirmed] After any change to the precision settings of the model (for example, switching from BFP8 to BFP4 weights, or changing math fidelity levels), all previously captured traces must be re-captured. The trace records specific kernel binaries and memory layouts; a precision change causes different kernels to be selected, making the old trace stale.

The required sequence after a precision change is:

1. Update the model's precision configuration.
2. Re-run `warmup_model_decode()` and `warmup_model_prefill()` to recompile affected kernels.
3. Re-capture all traces with `ttnn.begin_trace_capture` / `ttnn.end_trace_capture`.

Failing to re-capture after a precision change results in the old (now-incorrect) trace being replayed, which can produce silently wrong outputs.

### Summary of Initialization Sequence

| Step | Call | Purpose |
|---|---|---|
| 1 | `warmup_model_decode()` | JIT-compile decode kernels; no trace [confirmed] |
| 2 | `warmup_model_prefill()` | JIT-compile prefill kernels; no trace [confirmed] |
| 3 | `ttnn.begin_trace_capture` / forward / `ttnn.end_trace_capture` | Record decode execution trace [confirmed] |
| 4 (if precision changes) | Repeat steps 1–3 | Re-compile and re-capture after any precision setting change [confirmed] |

---

## Batching Strategies

### Decode Is Memory-Bandwidth-Bound

[confirmed] Decode is memory-bandwidth-bound. In each decode step, `tokens` has shape `[batch, 1]` — one token per sequence. The model must load all weight matrices from DRAM regardless of how many tokens it is processing. With a single token per sequence (`M=1` in the matmul sense), the arithmetic intensity of the weight matmuls is very low: nearly all time is spent on DRAM reads, not on compute.

Increasing batch size amortizes the fixed cost of loading each weight tile across more tokens per step. [confirmed] If the weight tile for a given layer is loaded once and used to compute outputs for `B` tokens in the same decode step, the effective cost per token is (DRAM read cost) / B plus a (compute cost) / B term. For DRAM-bandwidth-limited operations, doubling batch size nearly halves the per-token time, until some other constraint is reached.

### Static Batching

In static batching, all sequences in a batch start together (at the same prompt position) and the batch runs until all sequences have finished generating. The decode loop advances `current_pos` by 1 each step, and the batch completes when every sequence has reached its `max_new_tokens` limit.

[INFERRED] Static batching is simpler to implement and maps cleanly to the scalar `current_pos` constraint: because all sequences start at the same time with the same prompt length, they are at the same `current_pos` at every decode step. The tradeoff is efficiency: if one sequence in a batch of 32 finishes after 50 tokens while others continue to 2K tokens, that sequence's KV cache slot is wasted for the remaining 1950 steps.

| Property | Static Batching |
|---|---|
| `current_pos` semantics | Scalar int; exact for all sequences |
| Scheduling complexity | Low; single loop counter |
| KV cache utilization | Can be low if sequence lengths vary widely |
| Head-of-line blocking | Yes; short sequences wait for the longest |

### Continuous Batching

[INFERRED] In continuous batching, new requests are inserted into the batch as existing sequences complete. Rather than waiting for all sequences to finish, the serving system detects completed sequences (those that generated an EOS token or reached the maximum length), frees their KV cache blocks, and fills the vacated batch slots with new requests.

Continuous batching improves DRAM utilization and reduces head-of-line blocking compared to static batching, at the cost of additional scheduling complexity. Because different sequences in the batch may be at different decode positions, continuous batching cannot use the scalar `current_pos` interface directly for sequences that differ by more than a few steps; the implementation must handle position offset bookkeeping.

[INFERRED] The paged KV cache architecture (see Ch2) is a prerequisite for continuous batching: pages can be freed and reallocated for new sequences without copying or defragmenting the KV DRAM allocation.

| Property | Continuous Batching |
|---|---|
| `current_pos` semantics | Requires external position management for heterogeneous sequences |
| Scheduling complexity | Higher; tracks per-sequence completion and page allocation |
| KV cache utilization | Near 100% of allocated pool (only last partial block wasted per sequence) |
| Head-of-line blocking | Eliminated; new requests fill completed slots immediately |

### Batch Size Limits

[confirmed] Batch size is bounded by KV cache pool capacity: the total number of physical blocks must be sufficient for all active sequences to grow to their maximum length simultaneously. For sizing details, see `kv_cache_capacity_planning.md`.

[INFERRED] Beyond KV cache capacity, batch size is also bounded by the activation memory required per decode step in L1 (the `[batch, 1]` token tensor and the intermediate activations grow with batch) and by the DRAM bandwidth available for simultaneous weight reads across all batch elements. In practice, KV cache DRAM capacity tends to be the binding constraint before activation memory becomes an issue.

---

## Prefill/Decode Interleaving Considerations

In a production serving system, prefill and decode operations for different requests may overlap in time: while one batch is in mid-decode, new requests arrive that require prefill before they can join the decode batch.

[INFERRED] On Tenstorrent hardware, prefill and decode use different program configs, different math fidelity settings, and in the case of chunked prefill, different `WormholeComputeKernelConfig` objects. Switching between prefill and decode modes within a serving session requires:

1. Completing the active decode trace step (trace replay cannot be interrupted mid-step).
2. Calling the prefill forward pass with the appropriate prefill-mode config.
3. Re-entering the decode trace loop with the updated page table and the new sequence added to the batch.

[confirmed] Because decode traces are captured for a specific batch size and configuration, adding a new sequence to the batch after trace capture would require the new sequence to fill an already-allocated batch slot (replacing a completed sequence's slot), not expand the batch beyond its captured size. This is the mechanism that makes continuous batching compatible with trace replay.

---

## Key Takeaways

- The chunked prefill loop bound is `prompt_lens.max().item()`, not `tokens.shape[1]`. Iterating to the padded tensor dimension would write spurious KV cache entries from padding tokens. [confirmed]
- `decode_forward`'s `current_pos` argument is a scalar integer shared across all sequences in the batch, not a per-sequence tensor. Static batching aligns all sequences to the same position; continuous batching requires external position management. [confirmed]
- `warmup_model_decode()` and `warmup_model_prefill()` perform JIT kernel compilation only — no trace is captured during warm-up. [confirmed] Trace capture is a separate step using `ttnn.begin_trace_capture` / `ttnn.end_trace_capture`. [confirmed] The reference implementation calls decode warm-up before prefill warm-up; whether this ordering is required has not been confirmed. [INFERRED]
- Increasing decode batch size amortizes the fixed DRAM weight-loading cost across more tokens per step, improving per-token throughput until KV cache capacity or activation memory becomes the binding constraint. [confirmed / INFERRED]
- Any precision setting change invalidates all previously captured traces; re-warm-up and re-capture are required before resuming production decode. [confirmed]

---

## Further Reading

- Ch2 `paged_attention_kv_cache.md` — paged KV cache structure, `paged_update_cache`, page table data type and shape constraints
- Ch3 `weight_layout_and_quantization.md` — BFP4/BFP8 weight formats and their interaction with math fidelity
- Ch4 `memory_hierarchy.md` — DRAM bandwidth characteristics and the decode memory-bandwidth bottleneck
- Ch5 `ccl_and_ethernet.md` — CCL latency and its contribution to per-decode-step latency under tensor parallelism
- TTNN trace API: `ttnn.begin_trace_capture`, `ttnn.end_trace_capture`, `ttnn.execute_trace` — TT-Metal documentation

---

**Next:** [`kv_cache_capacity_planning.md`](./kv_cache_capacity_planning.md)
