# Multi-Device Decode: Llama 3.1 70B on T3K

This file traces a single decode step of Llama 3.1 70B running on a T3K (8-chip Wormhole Ethernet ring) at tensor parallelism degree TP=8. It assumes familiarity with the single-device decode sequence covered in `single_device_decode.md`. The focus here is on how each step in that sequence changes under TP=8: which weights are split, which dimensions they are split along, and where collective communication operations are inserted.

---

## Model Configuration

| Parameter | Value | Source |
|---|---|---|
| Architecture | Llama 3.1 70B | [confirmed] |
| Hardware | T3K: 8 Wormhole chips connected in an Ethernet ring | [confirmed] |
| Tensor parallelism | TP=8 | [confirmed] |
| Decoder layers | 80 | [confirmed] |
| Hidden dimension | 8192 | [confirmed per standard 70B architecture] |
| Q heads | 64 | [confirmed] |
| KV heads (`n_kv_heads`) | 8 | [confirmed] |
| Head dimension | 128 | [confirmed] |
| GQA group size | 8 (64 Q heads / 8 KV heads) | [confirmed] |
| Q heads per device | 8 (64 / TP=8) | [confirmed] |
| KV heads per device | 1 (8 / TP=8) | [confirmed] |
| Precision: MLP layers | BFP4 (LoFi) | [confirmed] |
| Precision: Attention layers | BFP8 (HiFi2) | [confirmed] |
| Precision: LM head + embedding | BF16 (HiFi4) | [confirmed] |

### T3K Topology

[confirmed] The T3K is an Ethernet ring of 8 Wormhole chips. The ring topology is relevant for collective operations: `ttnn.all_gather`, `ttnn.reduce_scatter`, and `ttnn.all_reduce` use the Ethernet links to pass data around the ring. Collective latency is a function of the ring diameter and the payload size.

[confirmed] T3K is an Ethernet ring, not a mesh. Each chip has a direct Ethernet connection to its two neighbors in the ring; chips that are not neighbors communicate through intermediate hops.

Cross-reference: Ch5 `ccl_and_ethernet.md` — Ethernet ring topology, collective latency, and bandwidth characteristics.

---

## GQA at TP=8: No KV Head Replication

The Llama 3.1 70B model has 64 Q heads and 8 KV heads. At TP=8, each device receives:

- **8 Q heads** (64 / 8 = 8) — [confirmed]
- **1 KV head** (8 / 8 = 1) — [confirmed]

[confirmed] Because TP equals the number of KV heads exactly (TP=8, n_kv_heads=8), each device gets exactly one KV head. No KV head replication is required. Each device computes GQA locally: its 8 Q heads attend over its single local KV head.

This is a favorable property of the 70B configuration. If TP were larger than n_kv_heads (e.g., TP=16 on a hypothetical 16-chip ring), some devices would need replicated KV heads, increasing KV cache memory and introducing additional communication. At TP=8 = n_kv_heads=8, the split is clean.

Cross-reference: Ch5 `tensor_parallelism.md` — GQA head distribution under TP, KV head replication rules. Ch2 `flash_decode.md` — GQA attention kernel.

---

## Weight Splitting Conventions

Under TP=8, weight matrices are split across devices according to two patterns:

### Column-Parallel (split along output dimension)

Used for: QKV projections, FF1 gate projection, FF3 up projection.

A column-parallel weight of shape `[hidden, N]` is split as:
- Device k receives `W_k` with shape `[hidden, N/8]`

Each device operates on the full input activation but produces only a 1/8 slice of the output. The slices are combined by subsequent communication or simply consumed locally if the next op is row-parallel.

### Row-Parallel (split along input dimension)

Used for: Output projection (attention), FF2 down projection.

A row-parallel weight of shape `[M, hidden]` is split as:
- Device k receives `W_k` with shape `[M/8, hidden]`

Each device computes a partial dot product using its local input slice (the output of the preceding column-parallel op) and its row shard. The partial results must be summed across all 8 devices to produce the correct output. This is done with `ttnn.reduce_scatter`. [confirmed]

Cross-reference: Ch5 `tensor_parallelism.md` — column-parallel and row-parallel weight splitting.

---

## Decode Step Sequence at TP=8

The following traces a single decode step through **one decoder layer** of Llama 3.1 70B on T3K at TP=8. The token input to each layer is `[batch, 1, 8192]` (one token per sequence, hidden dimension 8192). Each device holds a shard of the activations for its portion of the computation.

---

### Step 1 — RMS Norm (pre-attention)

- Input (each device): `[batch, 1, 8192]` — the full hidden-dimension activation is replicated across all 8 devices at this point (it was reconstructed by the residual add at the end of the previous layer).
- Output: `[batch, 1, 8192]` (BF16, all devices hold the same normalized activation)
- RMS norm is not parallelized; each device applies it independently to the full activation. The norm weight is replicated on all devices.
- This op runs on the SFPU.

Cross-reference: Ch1 `tensix_architecture.md` — SFPU.

---

### Step 2 — QKV Projection (column-parallel)

Each device holds a column shard of `W_QKV`.

- Full weight shape: `[8192, (64+8+8)*128]` = `[8192, 10240]`
  - 64 Q heads + 8 K heads + 8 V heads, each of dimension 128
- Per-device weight shard: device k holds `[8192, (8+1+1)*128]` = `[8192, 1280]`
  - 8 Q head columns + 1 K head column + 1 V head column for that device
- Input (each device): `[batch, 1, 8192]` (full, replicated)
- Output (each device): `[batch, 1, 1280]`, split into Q_k `[batch, 1, 8, 128]`, K_k `[batch, 1, 1, 128]`, V_k `[batch, 1, 1, 128]`

[INFERRED] The column-parallel QKV projection does not require an `all_gather` after the matmul; each device's local output is the correct Q, K, V shard for the heads assigned to that device. The full Q tensor is never assembled across devices during attention — each device attends only its own Q heads over its own KV head (see Step 4).

- Weight format: BFP8, HiFi2 — [confirmed]
- Matmul config: DRAM-sharded — [confirmed]

Cross-reference: Ch3 `matmul_program_configs.md` — DRAM-sharded matmul. Ch5 `tensor_parallelism.md` — column-parallel QKV.

---

### Step 3 — RoPE on Q and K

- Q_k: `[batch, 1, 8, 128]` → rotated Q_k
- K_k: `[batch, 1, 1, 128]` → rotated K_k
- V_k is not rotated.
- `current_pos` is a scalar integer. [confirmed] Because all sequences in the batch are at the same position, the same rotation is applied on all 8 devices. No communication is required.
- RoPE runs on the SFPU.

Cross-reference: Ch6 `prefill_decode_pipeline.md` — `current_pos` scalar semantics.

---

### Step 4 — Paged SDPA Decode (per-device, GQA-local)

Each device runs the paged SDPA decode kernel entirely locally over its own Q heads and its own KV head.

- Q_k input: `[batch, 8, 1, 128]` (8 Q heads on this device)
- K_k new: `[batch, 1, 1, 128]` (1 KV head on this device)
- V_k new: `[batch, 1, 1, 128]`
- KV cache for this device: `[1, 1, n_blocks * block_size, 128]`
  - Each device has its own KV cache containing only its single KV head.
  - [confirmed] The first dimension of the KV cache is 1, not batch; the paged block pool is shared across sequences within this device's KV shard.
- Page table: `[batch, max_pages]` — same page table on all devices (pages map to abstract block indices; each device's KV cache holds a separate physical allocation)
- GQA local: 8 Q heads attending over 1 KV head; group size = 8 on each device (64 total Q / 8 total KV, all local on each device) — [confirmed]
- Output: `[batch, 8, 1, 128]` (context vector per local Q head)

No cross-device communication is required during attention because each device has all the KV data it needs (exactly 1 KV head) and all the Q heads that attend to it (exactly 8 Q heads). This is the favorable consequence of TP=8 = n_kv_heads=8.

Cross-reference: Ch2 `flash_decode.md` — paged SDPA decode kernel. Ch2 `paged_attention_kv_cache.md` — KV cache shape and paged block pool. Ch5 `tensor_parallelism.md` — GQA with TP.

---

### Step 5 — Output Projection (row-parallel) + `ttnn.reduce_scatter`

The attention output from Step 4 is projected back to the full hidden dimension. This is a row-parallel matmul.

- Input (each device): `[batch, 1, 8*128]` = `[batch, 1, 1024]` (8 Q heads concatenated)
- Per-device weight shard: `W_O_k` with shape `[1024, 8192]` (row shard)
  - Full output projection weight is `[8192, 8192]`; device k holds `[1024, 8192]`
- Each device computes a partial contribution to the output: `[batch, 1, 8192]` partial sum
- Weight format: BFP8, HiFi2 — [confirmed]
- Matmul config: DRAM-sharded — [confirmed]

After the matmul, the 8 partial sums across devices must be combined. [confirmed] This uses `ttnn.reduce_scatter`:

- `ttnn.reduce_scatter` sums the partial contributions across all 8 devices AND shards the result, so each device ends up with a `[batch, 1, 8192/8]` = `[batch, 1, 1024]` slice of the full output.

[confirmed] The output projection uses `ttnn.reduce_scatter`, not `ttnn.all_reduce`. `reduce_scatter` is preferred here because the result is immediately used by the next all_gather (or by a subsequent sharded operation), making the scatter distribution the efficient choice rather than replicating the full result on every device.

Cross-reference: Ch5 `ccl_and_ethernet.md` — `ttnn.reduce_scatter` semantics and when to prefer it over `ttnn.all_reduce`.

---

### Step 6 — `ttnn.all_gather` (reconstruct full activation for residual add)

The residual add requires the full `[batch, 1, 8192]` activation on each device. After Step 5's `reduce_scatter`, each device holds only a `[batch, 1, 1024]` shard.

[INFERRED] `ttnn.all_gather` is called to replicate the full `[batch, 1, 8192]` activation across all 8 devices before the residual add. This restores the replicated state needed for the next RMS norm (Step 7) and the post-attention residual connection.

- Input per device: `[batch, 1, 1024]` shard
- Output per device: `[batch, 1, 8192]` full activation (replicated on all devices)

Cross-reference: Ch5 `ccl_and_ethernet.md` — `ttnn.all_gather`.

---

### Step 7 — Residual Add (post-attention)

- Input 1: `[batch, 1, 8192]` (from Step 6 all_gather)
- Input 2: `[batch, 1, 8192]` (residual entering this decoder layer, replicated on all devices)
- Output: `[batch, 1, 8192]` (replicated on all devices)
- Element-wise add; SFPU.

---

### Step 8 — RMS Norm (pre-MLP)

- Input (each device): `[batch, 1, 8192]` (replicated)
- Output: `[batch, 1, 8192]` (BF16, replicated)
- Each device computes independently; no communication.

---

### Step 9 — FF1 Gate Projection (column-parallel) + FF3 Up Projection (column-parallel)

For Llama 3.1 70B, the MLP expands from 8192 to 28672 (the standard 70B intermediate size). Both FF1 (gate) and FF3 (up) are column-parallel.

**FF1 Gate:**
- Full weight: `[8192, 28672]`
- Per-device shard: `[8192, 3584]` (28672 / 8)
- Input: `[batch, 1, 8192]` (full, replicated)
- Output per device: `[batch, 1, 3584]`
- Weight format: BFP4, LoFi — [confirmed]
- Matmul config: DRAM-sharded — [confirmed]

**FF3 Up:**
- Full weight: `[8192, 28672]`
- Per-device shard: `[8192, 3584]`
- Input: `[batch, 1, 8192]` (full, replicated)
- Output per device: `[batch, 1, 3584]`
- Weight format: BFP4, LoFi — [confirmed]
- Matmul config: DRAM-sharded — [confirmed]

The column-parallel MLP projections do not require a collective after the matmul; the output slices are consumed locally by the SiLU and element-wise multiply in the next step.

Cross-reference: Ch3 `weight_layout_and_quantization.md` — BFP4 format. Ch5 `tensor_parallelism.md` — column-parallel FF1/FF3.

---

### Step 10 — SiLU Activation + Element-wise Multiply

- Gate shard: `[batch, 1, 3584]` (from Step 9 FF1, on this device)
- Up shard: `[batch, 1, 3584]` (from Step 9 FF3, on this device)
- SiLU applied to gate shard → `[batch, 1, 3584]`
- Element-wise multiply with up shard → `[batch, 1, 3584]`
- SFPU; no communication.

---

### Step 11 — FF2 Down Projection (row-parallel) + `ttnn.reduce_scatter`

The activated MLP hidden state is projected back to the full hidden dimension. Row-parallel.

- Input per device: `[batch, 1, 3584]` (the local MLP shard from Step 10)
- Full weight: `[28672, 8192]`
- Per-device shard: `W_down_k` with shape `[3584, 8192]` (row shard)
- Each device computes partial contribution: `[batch, 1, 8192]`
- Weight format: BFP4, LoFi — [confirmed]
- Matmul config: DRAM-sharded — [confirmed]

After the matmul, `ttnn.reduce_scatter` combines the 8 partial sums:

- [confirmed] `ttnn.reduce_scatter` sums the partial contributions and distributes the result, so each device ends up with a `[batch, 1, 1024]` shard of the summed output.

[confirmed] `ttnn.reduce_scatter` is used, not `ttnn.all_reduce` — same rationale as Step 5.

Cross-reference: Ch5 `ccl_and_ethernet.md` — `ttnn.reduce_scatter`. Ch5 `tensor_parallelism.md` — row-parallel MLP down projection.

---

### Step 12 — `ttnn.all_gather` + Residual Add (post-MLP)

[INFERRED] `ttnn.all_gather` gathers the `[batch, 1, 1024]` shards from all 8 devices into the full `[batch, 1, 8192]` activation on each device.

After all_gather:
- Input 1: `[batch, 1, 8192]` (from all_gather)
- Input 2: `[batch, 1, 8192]` (residual from Step 7, replicated)
- Output: `[batch, 1, 8192]` (replicated, input to next decoder layer's Step 1)

This residual add completes one decoder layer. The sequence repeats for all 80 decoder layers.

---

### Step 13 — Final RMS Norm + Vocab-Parallel LM Head + `ttnn.all_reduce`

After all 80 decoder layers, the final output is passed through a last RMS norm and then the LM head.

**Final RMS Norm:**
- Input (each device): `[batch, 1, 8192]` (replicated)
- Output: `[batch, 1, 8192]` (BF16, replicated)

**Vocab-Parallel LM Head:**

[confirmed] The LM head weight is split along the vocabulary dimension (vocab-parallel). Each device holds a column shard of the weight covering 1/8 of the vocabulary output positions, but receives the full hidden-dimension input.

- Full weight: `[8192, vocab_size]`
- Per-device shard: `W_lm_k` with shape `[8192, vocab_size/8]`
- Input (each device): `[batch, 1, 8192]` (replicated)
- Output per device: `[batch, vocab_size/8]` — partial logits for this device's vocab shard
- Weight format: BF16, HiFi4 — [confirmed]

After the LM head matmul, `ttnn.all_reduce` is called: [confirmed]

- `ttnn.all_reduce` is the correct collective here because each device needs the **full** `[batch, vocab_size]` logit tensor to sample the next token. Unlike the row-parallel projections in the attention and MLP blocks (which use `reduce_scatter` because the result is only needed in a distributed form for the next layer), the LM head output must be fully assembled on every device.
- Output per device after `all_reduce`: `[batch, vocab_size]` (full logit tensor, replicated)

[confirmed] The LM head uses `ttnn.all_reduce`, not `ttnn.all_gather`. `ttnn.all_reduce` is used here so that every device ends up with the complete logit vector needed for next-token sampling.

Cross-reference: Ch5 `ccl_and_ethernet.md` — `ttnn.all_reduce` semantics. Ch5 `tensor_parallelism.md` — vocab-parallel LM head.

---

## Collective Communication Summary

The table below shows every collective operation in one decoder layer of the TP=8 decode step, in order.

| Location | Collective | Input per device | Output per device | Why |
|---|---|---|---|---|
| After Step 5 output projection | `ttnn.reduce_scatter` | `[batch, 1, 8192]` partial | `[batch, 1, 1024]` shard | Sum row-parallel partial contributions; keep result sharded |
| After Step 5 (before residual) | `ttnn.all_gather` | `[batch, 1, 1024]` shard | `[batch, 1, 8192]` full | Reconstruct full activation for residual add |
| After Step 11 down projection | `ttnn.reduce_scatter` | `[batch, 1, 8192]` partial | `[batch, 1, 1024]` shard | Sum row-parallel partial contributions; keep result sharded |
| After Step 11 (before residual) | `ttnn.all_gather` | `[batch, 1, 1024]` shard | `[batch, 1, 8192]` full | Reconstruct full activation for residual add |
| After Step 13 LM head | `ttnn.all_reduce` | `[batch, vocab/8]` partial logits | `[batch, vocab]` full logits | Assemble full logit tensor on every device; every device needs full logits for next-token sampling |

[confirmed] The row-parallel output projections (Step 5 attention output, Step 11 FF2 down) use `ttnn.reduce_scatter`, not `ttnn.all_reduce`.
[confirmed] The LM head uses `ttnn.all_reduce`, not `ttnn.all_gather`.

Per-layer collective count: 4 (two `reduce_scatter` + two `all_gather`). Plus one `all_reduce` after the final LM head.

---

## How the Single-Device Picture Scales to 8 Chips

The table below maps each step of the single-device Llama 3.1 8B walkthrough (`single_device_decode.md`) to its TP=8 counterpart on T3K, and notes what changes.

| Step | Single Device (8B / N150) | T3K at TP=8 (70B) | Key Change |
|---|---|---|---|
| RMS Norm | Applied to `[batch, 1, 4096]` locally | Applied to `[batch, 1, 8192]` locally; replicated | Hidden dim scales; no new communication |
| QKV Projection | Single device, all 32 Q + 8 KV heads | Column-parallel; 8 Q + 1 KV head per device | Weight shard; no post-op collective needed |
| RoPE | All 32 Q + 8 KV heads locally | 8 Q + 1 KV head per device; `current_pos` still scalar | Head count per device drops |
| Paged SDPA Decode | 32 Q heads, 8 KV heads, GQA group=4 | 8 Q heads, 1 KV head per device, GQA group=8 locally; no cross-device communication | Each device attends locally |
| Output Projection | Single matmul `[4096, 4096]` | Row-parallel `[1024, 8192]` per device + `reduce_scatter` | Collective inserted |
| Residual Add | Local | `all_gather` before add to reconstruct full activation | Collective inserted |
| RMS Norm (pre-MLP) | Local | Local; replicated after all_gather | No change in structure |
| FF1 Gate | BFP4 or BFP8 `[4096, 14336]` | BFP4 column-parallel `[8192, 3584]` per device | Weight shard; hidden and intermediate dim scale |
| FF3 Up | BFP4 or BFP8 `[4096, 14336]` | BFP4 column-parallel `[8192, 3584]` per device | Same as FF1 |
| SiLU + Multiply | Local on `[batch, 1, 14336]` | Local on `[batch, 1, 3584]` shard | Operates on shard |
| FF2 Down | BFP4 or BFP8 `[14336, 4096]` | BFP4 row-parallel `[3584, 8192]` per device + `reduce_scatter` | Collective inserted |
| Residual Add (post-MLP) | Local | `all_gather` before add | Collective inserted |
| LM Head | BF16 `[4096, vocab]` local | BF16 vocab-parallel `[8192, vocab/8]` per device + `all_reduce` | Vocab split; collective inserted |

---

## Key Takeaways

- Llama 3.1 70B on T3K runs 80 decoder layers with TP=8 across an Ethernet ring of 8 Wormhole chips. [confirmed]
- 64 Q heads and 8 KV heads at TP=8 gives exactly 8 Q heads and 1 KV head per device. No KV head replication is required; each device runs GQA entirely locally. [confirmed]
- Column-parallel projections (QKV, FF1 gate, FF3 up) do not require a post-op collective; each device produces the correct local shard directly.
- Row-parallel projections (output projection, FF2 down) require `ttnn.reduce_scatter` to sum partial contributions across devices. `reduce_scatter` is used, not `ttnn.all_reduce`. [confirmed]
- The LM head is vocab-parallel; after the LM head matmul, `ttnn.all_reduce` assembles the full logit tensor on every device. `all_reduce` is used, not `ttnn.all_gather`. [confirmed]
- Each decoder layer contributes 4 collective operations: two `reduce_scatter` and two `all_gather`. Across 80 layers, collective overhead is a significant component of total decode latency on the T3K.
- All MLP layers use BFP4 (LoFi) in production. All attention linear layers use BFP8 (HiFi2). The LM head and embedding use BF16 (HiFi4). [confirmed]
- `current_pos` remains a scalar integer at the `decode_forward` API level in the T3K configuration, just as in the single-device case. All 8 devices advance in lockstep. [confirmed]

---

## Further Reading

- `single_device_decode.md` — the single-device baseline that this file extends
- Ch2 `flash_decode.md` — paged SDPA decode kernel and GQA support
- Ch5 `tensor_parallelism.md` — column-parallel, row-parallel, and vocab-parallel weight splitting; GQA with TP
- Ch5 `ccl_and_ethernet.md` — `ttnn.all_gather`, `ttnn.reduce_scatter`, `ttnn.all_reduce` semantics; Ethernet ring topology and collective latency
- Ch6 `prefill_decode_pipeline.md` — `current_pos` scalar semantics; warm-up and trace capture initialization

---

**End of Chapter 7**
