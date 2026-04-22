# Chunked Prefill

## Overview

dots.ocr supports context lengths up to `max_position_embeddings=131072`, but processing a long prefill sequence in a single pass would exceed the L1 SRAM capacity of Wormhole N300 devices. The `Generator` class solves this by splitting the prefill phase into fixed-size chunks, each of which fits within the device's L1 budget. This file explains why OCR inputs require chunked prefill, how the chunking loop works, and how the relevant environment variables and TP=2 synchronization requirements interact.

### Why Long Sequences

A single OCR document image processed at 896×1344 resolution produces 1536 vision tokens after the vision encoder. A multi-page scan or an image containing dense printed text — the primary use case for dots.ocr — can produce thousands of additional text tokens when the model generates a structured transcription. With `max_position_embeddings=131072`, the model supports contexts far beyond what a standard short-context LLM would handle.

Even a modest OCR workload with a few hundred words of surrounding prompt context can push the total prefill sequence length past 4,096 tokens. At 131,072 tokens, the prefill input is 32× that length. The prefill phase is where this cost is paid: every token in the input must be processed through all 28 transformer layers before decode begins.

### L1 SRAM Constraint

Wormhole N300 provides a fixed L1 SRAM per Tensix core. During a prefill forward pass, the intermediate activations for a sequence of length S scale as O(S × hidden_size) per layer, and the attention scores scale as O(S²) in the general case (though TTNN uses a causal mask that reduces this in practice). For long sequences, these activations do not fit in L1 simultaneously.

Chunked prefill resolves this by processing a window of `chunk_size` tokens at a time. Each chunk's activations fit within L1, and the KV cache is populated incrementally — each chunk writes its KV entries before the next chunk reads from the updated cache. This trades a single large L1 allocation for a sequence of small ones, at the cost of more kernel launches per prefill.

> **Note:** The chunk size is a compile-time or startup-time decision, not a per-request decision. All requests served by a running `Generator` instance share the same chunk size.

### Chunked Prefill in `Generator`

The prefill entry point is `Generator.prefill_forward_text()`. At a high level, the loop structure is:

```python
for user_idx in range(batch_size):
    inputs = prepare_inputs_prefill(tokens[user_idx], ...)
    chunk_size = get_max_prefill_chunk_size(seq_len)
    for chunk_start in range(0, seq_len, chunk_size):
        chunk_end = min(chunk_start + chunk_size, seq_len)
        # Run one prefill forward pass for this chunk
        logits = model.prefill_forward(inputs[chunk_start:chunk_end], ...)
        # KV cache is updated in-place for positions [chunk_start, chunk_end)
    # After all chunks, decode phase can begin for this user
```

`get_max_prefill_chunk_size()` computes the chunk size from `DOTS_MAX_SEQ_LEN_WH_LB` and the current sequence length, ensuring the chunk is at least the minimum Wormhole-safe tile size. `prepare_inputs_prefill()` handles position embedding offsets so that each chunk's attention positions are correct relative to the full sequence.

> **Note:** The KV cache is populated cumulatively. Chunk N's attention sees KV entries from chunks 0 through N-1 as read-only history, plus its own newly computed entries. This is standard causal chunked prefill behavior — no attention mask adjustments are needed beyond what the position IDs encode.

### Env Vars for Chunked Prefill

See [t3k_submesh_and_env_vars.md](t3k_submesh_and_env_vars.md) for the full env var reference. Two variables are directly relevant to chunked prefill:

- `DOTS_MAX_SEQ_LEN`: caps total sequence length. Setting this too low silently truncates long OCR documents without a runtime error.
- `DOTS_MAX_SEQ_LEN_WH_LB`: sets the minimum chunk size.

> **Warning:** `DOTS_MAX_SEQ_LEN_WH_LB` is a lower bound on chunk size, not a target chunk size. The actual chunk size used may be larger, depending on `DOTS_MAX_SEQ_LEN` and the sequence length of the current request. Do not assume that setting `DOTS_MAX_SEQ_LEN_WH_LB=512` will always produce 512-token chunks.

### Synchronization at TP=2

At TP=2, the two devices in the 1×2 submesh are tightly coupled: every TTNN op is dispatched to both devices simultaneously, and each op requires both devices to complete before the next op can begin. This coupling extends to chunked prefill.

The chunk boundaries must be identical on both devices. There is no mechanism to run device 0 through chunk N while device 1 is still on chunk N-1; the Galaxy link's collective operations (all-reduce for the attention output, all-gather for the QKV projection) require both devices to be at the same point in the computation graph at all times.

In practice this means:

- Chunk size is a property of the submesh, not of individual devices.
- `DOTS_MAX_SEQ_LEN_WH_LB` applies uniformly across both devices.
- Any future modification that attempts to give different chunk sizes to different devices would break the TP collective ops.

### TTFT Impact

Time-to-first-token (TTFT) is the latency from receiving a request to emitting the first generated token. It is dominated by the prefill phase when input sequences are long, because decode is autoregressive (one token per step) while prefill processes all input tokens before any output is produced.

TTFT scales approximately linearly with the number of prefill chunks:

```
num_chunks = ceil(seq_len / chunk_size)
TTFT ≈ num_chunks × (time_per_chunk) + (decode_step_1_latency)
```

For a concrete example: a 1536-token vision sequence processed with a chunk size of 512 tokens requires:

```
ceil(1536 / 512) = 3 prefill chunks
```

Three complete prefill forward passes run before decode begins. If each prefill chunk takes 80 ms on the 1×2 submesh, TTFT is at minimum 240 ms before the first decode step.

`perf/benchmark.py` measures TTFT as part of its standard metrics suite. When tuning `DOTS_MAX_SEQ_LEN_WH_LB`, use the benchmark to verify that increasing the minimum chunk size (and thus reducing chunk count) does not push per-chunk latency above the point where L1 spills occur, which would cause a non-linear latency increase.

> **Note:** For very short inputs (e.g., a small image with minimal surrounding text), the full sequence may fit in a single chunk, in which case chunked prefill degenerates to standard single-pass prefill with no overhead.

**Next:** [Chapter 5 — Index](../ch5/index.md)
