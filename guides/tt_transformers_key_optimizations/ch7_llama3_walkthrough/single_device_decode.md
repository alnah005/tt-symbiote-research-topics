# Single-Device Decode: Llama 3.1 8B on N150

This file traces a single decode step of Llama 3.1 8B running on a single N150 (Wormhole chip). It covers model configuration, weight loading conventions, the two precision modes (accuracy vs performance), and the complete 13-step operation sequence for one decoder layer — with explicit cross-references to the chapter that introduced each optimization.

For the multi-device extension of this walkthrough (Llama 3.1 70B on T3K at TP=8), see `multi_device_decode.md`.

---

## Model Configuration

| Parameter | Value | Source |
|---|---|---|
| Architecture | Llama 3.1 8B | [confirmed] |
| Hardware | Single N150 (Wormhole chip) | [confirmed] |
| Decoder layers | 32 | [confirmed] |
| Hidden dimension | 4096 | [confirmed] |
| Q heads | 32 | [confirmed] |
| KV heads (`n_kv_heads`) | 8 | [confirmed] |
| Head dimension | 128 | [confirmed] |
| GQA group size | 4 (32 Q heads / 8 KV heads) | [confirmed] |
| Paged KV cache shape | `[1, 8, n_blocks * block_size, 128]` | [confirmed] |

The GQA group size of 4 means each KV head is shared across 4 Q heads during attention. On a single device all 8 KV heads reside locally; no replication or inter-device communication is needed for attention.

---

## Precision Configurations

Llama 3.1 8B on N150 supports two precision modes. The choice is made at model load time and applies uniformly across all 32 decoder layers.

### Accuracy Mode

| Component | Weight Format | Math Fidelity | Note |
|---|---|---|---|
| All MLP linear layers | BFP8 | HiFi2 | [confirmed] |
| All attention linear layers | BFP8 | HiFi2 | [confirmed] |
| LM head | BF16 | HiFi4 | [confirmed] |
| Token embedding | BF16 | HiFi4 | [confirmed] |

[confirmed] Accuracy mode yields approximately **23 tokens/second/user** on N150.

[confirmed] The math fidelity pairing for BFP8 is HiFi2. BFP8 weights retain 8-bit mantissa precision; HiFi2 uses the upper 8 mantissa bits of the activations in the FPU accumulation. This pairing is sufficient for full output quality in most tasks.

### Performance Mode

| Component | Weight Format | Math Fidelity | Note |
|---|---|---|---|
| MLP linear layers | BFP4 | LoFi | [confirmed] |
| Attention linear layers | BFP8 | HiFi2 | [confirmed] |
| LM head | BF16 | HiFi4 | [confirmed] |
| Token embedding | BF16 | HiFi4 | [confirmed] |

[confirmed] Performance mode yields approximately **28 tokens/second/user** on N150.

[confirmed] The math fidelity pairing for BFP4 is LoFi. LoFi uses only the upper 4 mantissa bits of activations in the FPU. This matches the reduced precision of BFP4 weights and avoids wasting cycles on mantissa bits that the weight format cannot represent. Applying HiFi2 to BFP4 weights would consume more compute cycles for no accuracy benefit.

[confirmed] Attention linear layers remain at BFP8 in performance mode. Attention weights are smaller (heads × head_dim is much smaller than the MLP projection dimensions of 4096 → 14336 and back), so the bandwidth savings from dropping to BFP4 would be proportionally smaller while the risk to output quality (particularly in the output projection) is higher.

### Why BFP4 MLP Gives ~22% Higher Throughput

[confirmed] Decode is memory-bandwidth-bound. With `M=1–32` tokens per step, the arithmetic intensity of every linear layer is too low to saturate the FPUs; nearly all wall time is spent reading weight tiles from DRAM.

[confirmed] BFP4 has a 3.56x bandwidth advantage vs BF16. BFP8 has a 1.88x bandwidth advantage vs BF16. This means BFP4 weights are loaded approximately 1.89x faster than BFP8 weights for the same layer (3.56 / 1.88 ≈ 1.89). The MLP layers account for most of the total weight volume per layer (the FF1, FF3, and FF2 projections together far exceed the QKV and output projections in parameter count), so switching MLP weights to BFP4 while keeping attention at BFP8 yields the majority of the achievable bandwidth gain. The result is approximately 22% higher throughput: 28 t/s/u vs 23 t/s/u. [confirmed]

### LM Head and Embedding: Always BF16

[confirmed] The language model head (the final linear projection from hidden dimension to vocabulary size) and the token embedding table are always kept in BF16 with HiFi4 math fidelity, regardless of whether the model is in accuracy or performance mode.

These components are not quantized because they directly determine the logit distribution from which the next token is sampled. Quantization artifacts at the LM head can disproportionately affect low-probability tokens and shift the shape of the distribution in ways that accumulate across a generation sequence.

---

## Weight Loading

### Pre-Transposed Layout

[confirmed] All weight matrices in tt-transformers are stored in pre-transposed form at load time. The stored layout is `[K, N]` where `K = in_features` and `N = out_features`.

In standard PyTorch convention, a linear layer weight has shape `[out_features, in_features]` and the forward pass computes `x @ W.T`. In tt-transformers, the transpose is materialized once during model loading, so the stored tensor is already `[in_features, out_features]`. The forward pass then computes `x @ W_stored` directly, with no transpose op at runtime.

This is an optimization in both memory layout (TTNN tile-format storage matches the matmul access pattern without a runtime permutation) and op count (eliminating a transpose from the hot path of every linear layer in every decode step).

Cross-reference: Ch3 `weight_layout_and_quantization.md` — weight storage conventions and BFP tile layout.

### DRAM-Sharded Matmul for All Decode Linear Layers

[confirmed] All linear layers during decode use DRAM-sharded matmul. The weight tensor remains in DRAM; the TTNN matmul kernel streams weight tiles directly into the matrix FPU without staging them through L1 first. This is correct for the decode regime because the weight volume is far larger than L1 capacity (120 KB per core on Wormhole) and the computation is bandwidth-bound rather than compute-bound.

[confirmed] The DRAM sharding splits the weight matrix across multiple DRAM banks, allowing multiple NoC channels to stream weight data in parallel. The activation tensor (shape `[batch, 1, hidden_dim]` in the decode case) fits in L1 and is broadcast or replicated across the cores as needed by the program config.

Cross-reference: Ch3 `matmul_program_configs.md` — DRAM-sharded matmul program config selection. Ch4 `memory_hierarchy.md` — DRAM bank parallelism and NoC bandwidth.

---

## Decode Step Sequence

The following traces a single decode step through **one decoder layer** of Llama 3.1 8B. The same sequence repeats for all 32 layers; residuals and norms connect layers in the standard pre-norm transformer pattern.

The token input to each layer is shape `[batch, 1, 4096]` — one token per sequence, hidden dimension 4096.

`current_pos` is a scalar integer representing the position being decoded. [confirmed] It is the same value for all sequences in the batch.

---

### Step 1 — RMS Norm (pre-attention)

Before any projection, the input activation is normalized by a learned RMS norm.

- Input: `[batch, 1, 4096]` (BF16 activations)
- Output: `[batch, 1, 4096]` (BF16)
- The RMS norm weight (`g`, the element-wise scale) is BF16.
- This op is not a matmul; it runs on the SFPU (scalar unit) of each Tensix core rather than the matrix FPU.

No quantization is applied to the RMS norm itself. The output is kept in BF16 and feeds directly into the QKV projection.

Cross-reference: Ch1 `tensix_architecture.md` — Tensix SFPU for element-wise ops.

---

### Step 2 — QKV Projection (linear)

The normalized input is linearly projected to produce Q, K, and V tensors.

- Input: `[batch, 1, 4096]`
- Weight: `W_QKV` with pre-transposed layout `[4096, (32+8+8)*128]` = `[4096, 6144]`
  - 32 Q heads + 8 K heads + 8 V heads, each of dimension 128
  - [confirmed] n_heads=32, n_kv_heads=8, head_dim=128
- Output: `[batch, 1, 6144]`, then split into Q `[batch, 1, 32, 128]`, K `[batch, 1, 8, 128]`, V `[batch, 1, 8, 128]`
- Weight format: BFP8 (both accuracy and performance mode) — [confirmed]
- Math fidelity: HiFi2 — [confirmed]
- Matmul config: DRAM-sharded — [confirmed]

Cross-reference: Ch3 `matmul_program_configs.md` — DRAM-sharded matmul. Ch3 `weight_layout_and_quantization.md` — pre-transposed layout, BFP8 format.

---

### Step 3 — Rotary Position Embedding (RoPE) on Q and K

[confirmed] RoPE is applied to Q and K using `current_pos` (the scalar decode position) to compute the rotation angles for this step.

- Q: `[batch, 1, 32, 128]` → rotated Q `[batch, 1, 32, 128]`
- K: `[batch, 1, 8, 128]` → rotated K `[batch, 1, 8, 128]`
- V is not rotated.
- This op runs on the SFPU (element-wise sinusoidal multiply-add pattern).
- `current_pos` is the scalar integer position; the rotation matrix is computed from this scalar. Because all sequences in the batch are at the same `current_pos`, a single rotation computation applies to all.

[confirmed] `current_pos` is a scalar int, not a per-sequence tensor, at the `decode_forward` API level.

Cross-reference: Ch6 `prefill_decode_pipeline.md` — `current_pos` scalar semantics.

---

### Step 4 — Paged SDPA Decode (attention)

Attention is computed using the flash decode kernel operating on the paged KV cache.

- Q input: `[batch, 32, 1, 128]` (rotated; heads moved to dim 1 for the kernel)
- K new: `[batch, 8, 1, 128]` (rotated)
- V new: `[batch, 8, 1, 128]`
- KV cache shape: `[1, 8, n_blocks * block_size, 128]` — [confirmed]
  - First dimension is 1, not batch; the paged block pool is shared across all sequences.
  - 8 KV heads, each of dimension 128.
- Page table: `[batch, max_pages]`, int32 — maps sequence positions to physical KV blocks
- GQA: each of the 8 KV heads is shared across 4 Q heads (group size = 32/8 = 4)
- Output: `[batch, 32, 1, 128]` (one context vector per Q head per sequence)

The paged SDPA decode kernel (`paged_scaled_dot_product_attention_decode`) uses flash decode: it streams KV tiles from DRAM, accumulates the softmax-weighted sum incrementally, and never materializes the full `[batch, n_heads, 1, seq_len]` score matrix. This keeps L1 pressure constant regardless of context length.

[confirmed] The attention linear layers (QKV projection and output projection) use BFP8 in both accuracy and performance mode.

Cross-reference: Ch2 `flash_decode.md` — flash decode algorithm and paged SDPA decode kernel. Ch2 `paged_attention_kv_cache.md` — KV cache shape `[1, n_kv_heads, n_blocks * block_size, head_dim]`, page table conventions, `paged_update_cache`.

---

### Step 5 — Output Projection (linear)

The attention output is projected back to the model hidden dimension.

- Input: `[batch, 1, 32*128]` = `[batch, 1, 4096]` (heads concatenated)
- Weight: `W_O` pre-transposed `[4096, 4096]`
- Output: `[batch, 1, 4096]`
- Weight format: BFP8 (both modes) — [confirmed]
- Math fidelity: HiFi2 — [confirmed]
- Matmul config: DRAM-sharded — [confirmed]

Cross-reference: Ch3 `matmul_program_configs.md`. Ch3 `weight_layout_and_quantization.md`.

---

### Step 6 — Residual Add (post-attention)

The output projection result is added to the residual stream from before Step 1.

- Input 1: `[batch, 1, 4096]` (from Step 5)
- Input 2: `[batch, 1, 4096]` (residual from previous layer or embedding)
- Output: `[batch, 1, 4096]`
- Element-wise add; runs on the SFPU.

---

### Step 7 — RMS Norm (pre-MLP)

A second RMS norm is applied to the residual stream before the MLP block.

- Input: `[batch, 1, 4096]` (BF16)
- Output: `[batch, 1, 4096]` (BF16)
- Same SFPU execution as Step 1.

---

### Step 8 — FF1 Gate Projection (linear, gated MLP)

Llama 3.1 uses a SwiGLU-style gated MLP. The MLP expands the hidden dimension from 4096 to 14336.

- Input: `[batch, 1, 4096]`
- Weight: `W_gate` pre-transposed `[4096, 14336]`
- Output: `[batch, 1, 14336]` — the gate signal
- **Accuracy mode**: BFP8 weight, HiFi2 fidelity — [confirmed]
- **Performance mode**: BFP4 weight, LoFi fidelity — [confirmed]
- Matmul config: DRAM-sharded — [confirmed]

The BFP4 bandwidth advantage (3.56x vs BF16) is most visible here: the gate and up projections each have a `[4096, 14336]` weight matrix. In performance mode, loading these in BFP4 vs BFP8 is approximately 1.89x faster for each of these large matrices.

Cross-reference: Ch3 `weight_layout_and_quantization.md` — BFP4 format (2-bit mantissa + shared exponent per tile), LoFi math fidelity pairing. Ch4 `memory_hierarchy.md` — bandwidth-bound decode analysis.

---

### Step 9 — FF3 Up Projection (linear, gated MLP)

- Input: `[batch, 1, 4096]`
- Weight: `W_up` pre-transposed `[4096, 14336]`
- Output: `[batch, 1, 14336]` — the up-projected signal
- **Accuracy mode**: BFP8 weight, HiFi2 fidelity — [confirmed]
- **Performance mode**: BFP4 weight, LoFi fidelity — [confirmed]
- Matmul config: DRAM-sharded — [confirmed]

In a gated MLP, FF1 (gate) and FF3 (up) are computed independently from the same normed residual. They can be fused or sequenced; in tt-transformers they are separate DRAM-sharded matmul ops.

---

### Step 10 — SiLU Activation + Element-wise Multiply

The gate signal from FF1 is passed through SiLU (sigmoid linear unit) and then multiplied element-wise with the up-projected signal from FF3.

- Gate: `[batch, 1, 14336]` from Step 8
- Up: `[batch, 1, 14336]` from Step 9
- SiLU output: `[batch, 1, 14336]`
- After element-wise multiply: `[batch, 1, 14336]`
- SiLU runs on the SFPU; the multiply is also an SFPU element-wise op.

The output of this step is the activated MLP hidden state.

Cross-reference: Ch1 `tensix_architecture.md` — SFPU for non-linear activations.

---

### Step 11 — FF2 Down Projection (linear)

The activated hidden state is projected back down to the model hidden dimension.

- Input: `[batch, 1, 14336]`
- Weight: `W_down` pre-transposed `[14336, 4096]`
- Output: `[batch, 1, 4096]`
- **Accuracy mode**: BFP8 weight, HiFi2 fidelity — [confirmed]
- **Performance mode**: BFP4 weight, LoFi fidelity — [confirmed]
- Matmul config: DRAM-sharded — [confirmed]

`W_down` at `[14336, 4096]` is the largest single weight matrix in the layer. In performance mode, loading this in BFP4 rather than BFP8 provides the largest per-matrix bandwidth saving of any operation in the decode step.

---

### Step 12 — Residual Add (post-MLP)

The down projection output is added back to the residual stream entering the MLP.

- Input 1: `[batch, 1, 4096]` (from Step 11)
- Input 2: `[batch, 1, 4096]` (residual from Step 6)
- Output: `[batch, 1, 4096]`
- Element-wise add on the SFPU.

This output becomes the input to the RMS norm at Step 1 of the **next** decoder layer, for each of the 32 layers.

---

### Step 13 — Final RMS Norm + LM Head (after all 32 layers)

After all 32 decoder layers have run their Steps 1–12, a final RMS norm is applied to the last layer's output, followed by the LM head projection to vocabulary logits.

**Final RMS Norm:**
- Input: `[batch, 1, 4096]` (output of layer 32 Step 12)
- Output: `[batch, 1, 4096]` (BF16)

**LM Head:**
- Input: `[batch, 1, 4096]`
- Weight: `W_lm_head` pre-transposed `[4096, vocab_size]`
- Output: `[batch, vocab_size]` (logits)
- **Always BF16 weight, HiFi4 fidelity — in both accuracy and performance mode** — [confirmed]

[confirmed] The LM head is never quantized to BFP8 or BFP4 in either precision mode. See "LM Head and Embedding: Always BF16" above for the rationale.

Cross-reference: Ch1 `math_fidelity_and_data_formats.md` — HiFi4 fidelity level; BF16 data format. Ch3 `weight_layout_and_quantization.md` — when not to quantize.

---

## Summary: Optimizations by Step

The table below consolidates which optimization from which chapter applies at each step of the decode sequence.

| Step | Operation | Optimization Applied | Source Chapter |
|---|---|---|---|
| 1, 7 | RMS Norm | SFPU element-wise execution; no quantization | Ch1 |
| 2 | QKV Projection | DRAM-sharded matmul; BFP8 weight; pre-transposed layout; HiFi2 | Ch3 |
| 3 | RoPE | SFPU sinusoidal; scalar `current_pos` for batch-uniform rotation | Ch1, Ch6 |
| 4 | Paged SDPA Decode | Flash decode; paged KV cache `[1, 8, blocks, 128]`; GQA group=4 | Ch2 |
| 5 | Output Projection | DRAM-sharded matmul; BFP8 weight; pre-transposed layout; HiFi2 | Ch3 |
| 6, 12 | Residual Add | SFPU element-wise | Ch1 |
| 8 | FF1 Gate Projection | DRAM-sharded matmul; BFP4 (perf) or BFP8 (acc); LoFi or HiFi2 | Ch3, Ch4 |
| 9 | FF3 Up Projection | DRAM-sharded matmul; BFP4 (perf) or BFP8 (acc); LoFi or HiFi2 | Ch3, Ch4 |
| 10 | SiLU + Multiply | SFPU non-linear + element-wise multiply | Ch1 |
| 11 | FF2 Down Projection | DRAM-sharded matmul; BFP4 (perf) or BFP8 (acc); LoFi or HiFi2 | Ch3, Ch4 |
| 13 | Final Norm + LM Head | LM head always BF16 + HiFi4; DRAM-sharded matmul | Ch1, Ch3 |

---

## Warm-Up and Trace Capture

Before the decode loop begins, the model must complete two initialization phases. These are not specific to Llama 3.1 8B but are required for all tt-transformers inference:

1. **Warm-up** (JIT compilation only): `warmup_model_decode()` and `warmup_model_prefill()` are called. These compile RISC-V kernel binaries for all ops in the decode forward graph and cache them in `TT_METAL_CACHE`. No trace is captured. [confirmed]

2. **Trace capture**: After warm-up, a decode trace is captured using `ttnn.begin_trace_capture` / `ttnn.end_trace_capture`. The production decode loop replays this trace with `enable_trace=True` to eliminate Python dispatch overhead. [confirmed]

[confirmed] Switching between accuracy mode and performance mode invalidates all previously captured traces. The full warm-up and trace capture sequence must be repeated after any precision setting change.

Cross-reference: Ch6 `prefill_decode_pipeline.md` — warm-up vs trace capture, initialization sequence, and the distinction between JIT compilation and trace recording.

---

## Key Takeaways

- Llama 3.1 8B on N150 runs 32 decoder layers, each consisting of 13 steps in the decode sequence: RMS norm, QKV projection, RoPE, paged SDPA decode, output projection, residual add, RMS norm, FF1 gate, FF3 up, SiLU + multiply, FF2 down, residual add. The LM head runs once after all 32 layers. [confirmed]
- Performance mode (BFP4 MLP, ~28 t/s/u) is approximately 22% faster than accuracy mode (BFP8 MLP, ~23 t/s/u) because decode is bandwidth-bound and BFP4 has a 3.56x bandwidth advantage vs BF16 (vs 1.88x for BFP8). [confirmed]
- The math fidelity pairing is fixed by the weight format: BFP4 → LoFi, BFP8 → HiFi2, BF16 → HiFi4. Using a higher fidelity than the weight format can represent wastes compute cycles with no quality benefit. [confirmed]
- All decode linear layers use DRAM-sharded matmul with pre-transposed `[K, N]` weight layout. No runtime transpose is needed. [confirmed]
- The LM head and token embedding are always BF16 + HiFi4, in both accuracy and performance mode. [confirmed]
- The paged KV cache shape is `[1, n_kv_heads, n_blocks * block_size, head_dim]` = `[1, 8, ..., 128]`. The first dimension is 1, not batch size. [confirmed]

---

**Next:** [`multi_device_decode.md`](./multi_device_decode.md)
