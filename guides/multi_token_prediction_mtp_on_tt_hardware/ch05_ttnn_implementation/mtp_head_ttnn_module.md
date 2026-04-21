# TTNNMTPHead Module Design

## Module Design Overview

`TTNNMTPHead` is a self-contained TTNN module analogous to `TTNNQwen3FullAttention` and `TTNNQwen3MoE` in the existing tt-transformers codebase. It wraps the single MTP transformer block present in Qwen3.6-35B-A3B (`mtp_num_hidden_layers = 1`) and exposes a `forward()` method compatible with the three-pass speculative decode loop described in Chapter 4 (`ch04_speculative_decoding_with_mtp/`).

The module is deliberately narrow in scope:

- It wraps exactly one transformer decoder block (the MTP head block).
- It reuses existing TTNN attention and FFN primitives — no new kernel development is required.
- It does not modify the backbone module or its computational graph in any way.
- It holds a reference to the shared `lm_head` (the embedding table tied to the backbone) rather than allocating a second copy.

A `use_mtp: bool` constructor argument allows the module to be instantiated but silenced without changing call sites in the generation loop, enabling clean A/B benchmarking and graceful fallback.

## Inputs

The `forward()` method takes two tensors:

### `backbone_hidden_state`

- Shape: `[batch, 1, H]` where `H = 4096` for Qwen3.6-35B-A3B
- Dtype: BF16
- Source: the final hidden state produced by the backbone's last layer **after** the backbone's final layer norm, corresponding to the most recently processed token position
- This is the same tensor that would be fed to `lm_head` to produce `primary_logits`; it is extracted before or alongside the `lm_head` call in the primary backbone pass (Step 1 of the decode cycle)

### `x_t1_embedding`

- Shape: `[batch, 1, H]`
- Dtype: BF16
- Source: `model.embed_tokens(x_t1)` — the embedding of the just-confirmed token `x_t1` sampled from `primary_logits` in Step 1
- `x_t1` is always accepted (it is a sample from the target backbone distribution `p`); no accept/reject decision applies to it (see Chapter 4, `ch04_speculative_decoding_with_mtp/`)

### Input Combination

For Qwen3.6-35B-A3B, the MTP block's input is formed by **summing** `backbone_hidden_state` and `x_t1_embedding` element-wise, then passing the result through the MTP transformer block. This is model-specific; other architectures may use concatenation followed by a projection instead. The sum is performed in BF16 on-device.

## Internal Components

`TTNNMTPHead` contains one transformer decoder block with the following characteristics:

- **Attention**: same `head_dim` and `num_heads` configuration as the backbone layers. Reuses existing TTNN multi-head attention (or grouped-query attention) primitives already exercised by backbone layers. No architectural difference from a backbone attention block.
- **FFN**: dense (not MoE). The MTP head uses a standard dense feed-forward network despite the backbone using MoE layers. This is consistent with the design intent: the MTP head is a lightweight module and does not inherit the MoE routing infrastructure. Reuses existing TTNN dense FFN primitives.
- **Norms**: layer norms at standard positions within the transformer block, loaded from the `model.future_prediction[0].*` weight keys.

No new TTNN kernel development is required. Every sub-operation in the MTP block has a direct counterpart in the existing backbone implementation.

## Output

`forward()` returns `draft_logits` of shape `[batch, 1, vocab_size]` where `vocab_size = 151936` for Qwen3.6-35B-A3B.

`draft_logits` is produced by applying the shared `lm_head` to the MTP block's output hidden state. The `lm_head` weight is the same embedding table used by the backbone; it is already loaded and resident in DRAM as part of backbone initialization. `TTNNMTPHead` holds a reference to this tensor — it does not copy or re-allocate it.

When `use_mtp=False`, `forward()` returns `None` immediately without performing any computation.

## Weight Loading

MTP head checkpoint keys follow the pattern `model.future_prediction[0].*`. This prefix covers:

- Norm weights for the MTP block's layer norms
- Attention projection weights (`q_proj`, `k_proj`, `v_proj`, `o_proj`)
- FFN projection weights

These are mapped to TTNN tensor allocations in BF16 during module initialization, placed in DRAM interleaved (see `memory_placement_for_mtp.md` for rationale).

The `lm_head` weight is **not** loaded by `TTNNMTPHead`. It is shared with the backbone module and already loaded as part of backbone initialization. `TTNNMTPHead.__init__` receives a reference to the backbone's `lm_head` TTNN tensor and stores it. Duplicating this weight would waste ~290 MiB of DRAM and is unnecessary.

> **Key Finding:** Total new DRAM allocation for `TTNNMTPHead` is approximately 304.6 MiB (MTP block weights only, excluding the shared `lm_head`). This is 0.4% of the backbone's ~70 GiB footprint. Weight loading from checkpoint is straightforward with no key remapping beyond the `model.future_prediction[0].*` prefix filter.

## Toggle Flag

`TTNNMTPHead` accepts a `use_mtp: bool` constructor argument (default `True`).

When `use_mtp=False`:
- Module initialization skips weight loading entirely (no DRAM allocation, no checkpoint reads for MTP keys)
- `forward()` is a no-op returning `None`
- The backbone module and its weights are unaffected in both cases

This flag enables the generation loop to support both standard autoregressive decode (`use_mtp=False`) and MTP-based speculative decode (`use_mtp=True`) with a single code path. Throughput benchmarking and correctness regression testing both rely on toggling this flag (see `testing_and_validation.md`).

> **[CRITICAL]** The backbone computational graph must be identical regardless of the `use_mtp` flag value. If `TTNNMTPHead` accidentally shares mutable state with the backbone (e.g., a reference to a backbone KV cache buffer rather than a distinct allocation), enabling MTP could silently corrupt backbone inference. Verify via the backbone non-regression test in `testing_and_validation.md` before any throughput measurement.

## Code Sketch

The following Python pseudocode illustrates the module structure. Actual implementation will use tt-transformers conventions for device placement, weight loading helpers, and TTNN tensor management.

```python
class TTNNMTPHead:
    def __init__(self, device, config, backbone_lm_head: ttnn.Tensor, use_mtp: bool = True):
        self.use_mtp = use_mtp
        if use_mtp:
            # Load MTP block weights from checkpoint under model.future_prediction[0].*
            # Norm weights
            self.input_norm_weight = load_weight("model.future_prediction[0].norm.weight", device)
            self.post_attn_norm_weight = load_weight("model.future_prediction[0].post_attention_layernorm.weight", device)
            # Attention projections
            self.q_proj = load_weight("model.future_prediction[0].self_attn.q_proj.weight", device)
            self.k_proj = load_weight("model.future_prediction[0].self_attn.k_proj.weight", device)
            self.v_proj = load_weight("model.future_prediction[0].self_attn.v_proj.weight", device)
            self.o_proj = load_weight("model.future_prediction[0].self_attn.o_proj.weight", device)
            # Dense FFN projections (not MoE)
            self.gate_proj = load_weight("model.future_prediction[0].mlp.gate_proj.weight", device)
            self.up_proj   = load_weight("model.future_prediction[0].mlp.up_proj.weight", device)
            self.down_proj = load_weight("model.future_prediction[0].mlp.down_proj.weight", device)
            # Shared lm_head — reference only, no copy
            self.lm_head = backbone_lm_head

    def forward(
        self,
        backbone_hidden: ttnn.Tensor,   # [batch, 1, H]
        x_t1_emb: ttnn.Tensor,          # [batch, 1, H]
        mtp_kv_cache: ttnn.Tensor,
    ) -> ttnn.Tensor | None:
        if not self.use_mtp:
            return None

        # Combine backbone hidden state and token embedding (sum, model-specific)
        h = ttnn.add(backbone_hidden, x_t1_emb)   # [batch, 1, H]

        # Single MTP transformer block
        h = self.mtp_block_forward(h, mtp_kv_cache)

        # Apply shared lm_head to produce draft logits
        draft_logits = ttnn.linear(h, self.lm_head)  # [batch, 1, vocab_size]
        return draft_logits

    def mtp_block_forward(self, h: ttnn.Tensor, kv_cache: ttnn.Tensor) -> ttnn.Tensor:
        # Pre-norm, attention, residual, post-norm, dense FFN, residual
        # Reuses existing TTNN attention and FFN primitives
        residual = h
        h = rms_norm(h, self.input_norm_weight)
        h = self_attention(h, self.q_proj, self.k_proj, self.v_proj, self.o_proj, kv_cache)
        h = ttnn.add(h, residual)
        residual = h
        h = rms_norm(h, self.post_attn_norm_weight)
        h = dense_ffn(h, self.gate_proj, self.up_proj, self.down_proj)
        h = ttnn.add(h, residual)
        return h
```

## References

- Chapter 1: `ch01_mtp_foundations/` — MTP architecture, `mtp_num_hidden_layers`, dense FFN vs. MoE distinction
- Chapter 2: `ch02_mtp_weights_and_memory/` — Weight key pattern `model.future_prediction[0].*`, 304.6 MiB BF16 footprint
- Chapter 3: `ch03_mtp_in_huggingface/` — Shared `lm_head` (tied embedding), weight loading behavior
- Chapter 4: `ch04_speculative_decoding_with_mtp/` — Role of `draft_logits` in the acceptance check
- Chapter 5: `memory_placement_for_mtp.md` — DRAM placement rationale for MTP weights
- Chapter 5: `testing_and_validation.md` — Backbone non-regression test, PCC correctness check
