# Practical Implications for Text-Only Inference

## 1. HuggingFace Text-Only Path

When `Qwen2_5_VLForConditionalGeneration.generate()` is called with a text-only input (no pixel values, no video frames), the internal `get_rope_index()` function constructs the 3D position ID tensor as:

```python
position_ids = torch.arange(S, dtype=torch.long).unsqueeze(0).unsqueeze(0).repeat(3, 1, 1)
# shape [3, 1, S]; all three rows identical — [0, 1, 2, ..., S-1]
```

The `Qwen2_5_VLRotaryEmbedding.forward()` method is then called, which performs the three-gather + duplication construction. Because all three rows are identical, the output equals standard 1D partial RoPE cos/sin, as proved in `mathematical_equivalence_proof.md`. The M-RoPE code path is active; the M-RoPE *effect* on output values is absent. Every token is encoded with its sequential text position across all 64 rotary dimensions.

## 2. The Existing TTNN Text-Only Path

The existing `TTNNRotaryPositionEmbedding` class in tt-symbiote:

- Precomputes a standard 1D cos/sin table of shape `[max_seq_len, rotary_dim]` at initialization, using `rope_theta = 1000000.0` and `rotary_dim = 64`.
- At forward time, performs a contiguous slice of the table at the current position index: `cos_table[cur_pos : cur_pos + seq_len]`.
- Applies partial RoPE by rotating the first 64 of 128 head dimensions and concatenating the remaining 64 unchanged.
- Accepts a 1D `[seq_len]` position index, not a 3D `[3, batch, seq_len]` tensor.

By the equivalence proof in `mathematical_equivalence_proof.md`, the cos/sin values produced by this standard table lookup are identical to the values that M-RoPE would produce for the same text-only position IDs. The rotate-half application is unchanged.

**No changes are needed to `TTNNRotaryPositionEmbedding` for text-only Qwen3.6-35B-A3B inference.**

## 3. Key Finding and Scoping Decision

The equivalence proof defines the scope of M-RoPE TTNN work precisely:

- M-RoPE TTNN support is needed **only** when the input batch contains vision tokens (image patches or video frame patches).
- A correct implementation must gate on whether vision tokens are present in the sequence and route accordingly.
- For text-only batches, routing through the standard `TTNNRotaryPositionEmbedding` path is correct, complete, and avoids all M-RoPE overhead.

The integration plan in Chapter 6 implements this gating at the attention module level by inspecting the input type before selecting the RoPE path.

## 4. Risk of Over-Engineering

A premature M-RoPE implementation that always constructs 3D position IDs and always performs the three-gather construction — even for text-only batches — introduces costs with no correctness benefit:

- Three `ttnn.embedding` lookups instead of one contiguous slice per decode step.
- One `ttnn.concat` operation per layer per step to reassemble the half-frequency vector.
- Increased host-side dispatch count (~5 additional TTNN op dispatches per decode step — quantified in Ch5 `kernel_launch_overhead.md`).
- No change in output values relative to the standard path for text-only inputs.

For a Qwen3.6 deployment that is primarily used for text inference with occasional vision inputs, always taking the M-RoPE path degrades the common case with no upside. The implementation strategy in Chapter 4 preserves the text-only fast path explicitly.

> **Key Finding:** The current `TTNNRotaryPositionEmbedding` text-only path is numerically correct for Qwen3.6-35B-A3B text inference. M-RoPE TTNN support is scoped to vision-input batches only. Chapter 4 details how to add the vision path while preserving the text-only fast path.

---
**Next:** [`mrope_section_always_active.md`](./mrope_section_always_active.md)
