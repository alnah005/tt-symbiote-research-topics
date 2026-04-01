# Forward Signature: Uniform Interface and Residual Path

## The Uniform Signature

`DeltaNetDecoderBlock.forward` exposes the same signature as `TransformerBlock.forward`:

```python
def forward(
    self,
    x: ttnn.Tensor,
    current_pos,
    rot_mats_global=None,
    rot_mats_local=None,
    user_id=0,
    mode="decode",
    page_table=None,
    chunk_page_table=None,
    chunk_start_idx=None,
    kv_cache=None,
    batch_size=1,
) -> ttnn.Tensor:
```

This matters because both the 27B demo (`demo.py`) and the A3B demo (`demo_a3b.py`) run
the layer loop as:

```python
for layer in model.layers:
    x = layer(x, current_pos=tt_pos, rot_mats_global=rot_mats, mode=Mode.DECODE)
```

The loop passes the same positional and keyword arguments to every layer. When a layer
holds a `GatedDeltaNet`, those arguments are silently ignored by the sub-module — they
are accepted by the outer block signature but never forwarded to `GatedDeltaNet.forward`.

## Silent Ignore of DeltaNet-Irrelevant Arguments

The docstring in `qwen35_decoder.py` lists the four arguments that DeltaNet layers drop:

```
# DeltaNet layers ignore: current_pos, rot_mats, page_table, kv_cache
```

The dispatch logic inside `forward` is:

```python
if hasattr(self.attention, "initialize_states"):
    # DeltaNet: only needs the input
    attn_out = self.attention.forward(attn_in)
else:
    # GatedAttention: needs current_pos and rot_mats for RoPE
    attn_out = self.attention.forward(
        attn_in,
        current_pos=current_pos,
        rot_mats=rot_mats_global,
        mode=mode,
    )
```

The complete set of dropped arguments:

- **DeltaNet-irrelevant** (not forwarded to `GatedDeltaNet.forward`): `current_pos`, `rot_mats_global`, `page_table`, `kv_cache`
- **Signature-parity-only** (accepted but unused by any sub-module): `rot_mats_local`, `user_id`, `chunk_page_table`, `chunk_start_idx`, `batch_size`

For details on why position information is absent from the DeltaNet recurrence, see Chapter 2.

## Residual Memory Layout

The first action in `forward` is to coerce the input to the residual memory configuration:

```python
skip_mem_cfg = self.args.get_residual_mem_config(mode, self.prefetcher)
x = ttnn.to_memory_config(x, skip_mem_cfg)
residual = x
```

`get_residual_mem_config` returns a DRAM-sharded memory config appropriate for the
current mode and prefetcher state. The same config is used for both residual adds:

```python
# Post-attention residual add
attn_out = ttnn.to_memory_config(attn_out, skip_mem_cfg)
hidden_states = ttnn.add(residual, attn_out, memory_config=skip_mem_cfg)
```

```python
# Post-MLP final output add
out = ttnn.add(
    residual,
    hidden_states,
    memory_config=skip_mem_cfg,
    dtype=activation_dtype or ttnn.bfloat16,
)
```

Using a DRAM-sharded config for the skip connections avoids L1 pressure during the
norm and MLP compute that follows. The tensor is sharded across DRAM banks so that
reads in the subsequent `ttnn.add` are spread across memory controllers.

## Full Forward Data Flow

The forward pass through one decoder block:

$$x_0 \leftarrow \text{to\_mem\_config}(x,\, \text{skip\_cfg})$$

$$x_{\text{attn}} = \text{AttentionNorm}(x_0)$$

$$a = \text{Attention}(x_{\text{attn}})$$

$$x_1 = x_0 + \text{to\_mem\_config}(a,\, \text{skip\_cfg})$$

$$x_{\text{ff}} = \text{FFNorm}(x_1)$$

$$f = \text{FeedForward}(x_{\text{ff}})$$

$$\text{out} = x_1 + f$$

The residual variable is reassigned after the attention add:

```python
hidden_states = ttnn.add(residual, attn_out, memory_config=skip_mem_cfg)
residual = hidden_states
if mode == Mode.PREFILL:
    x.deallocate(True)
ttnn.deallocate(attn_out)
```

In PREFILL mode, the original `x` tensor is explicitly deallocated after the first
residual add because prefill tensors can be large (sequence-length batched) and should
not persist in DRAM while the MLP runs.

## L1 CB Clash Workaround for hidden\_dim=17408

The comment in `forward` captures a hardware constraint:

```python
# MLP uses DRAM interleaved input with auto-selected matmul (program_config=None
# in model_config) to avoid L1 CB clash with hidden_dim=17408 on Blackhole.
ff_norm_config = self.args.get_norm_config("ff", mode, self.prefetcher)
hidden_states = self.ff_norm(hidden_states, mode, norm_config=ff_norm_config)
hidden_states = self.feed_forward.forward(hidden_states, mode)
```

The 27B model has `hidden_size = 5120` and `intermediate_size = 17408`. This value is
read directly from the HuggingFace model config (`config.json`) and stored verbatim as
`args.hidden_dim` — it is not derived from the `calculate_hidden_dim` formula used by
Llama-style models. `17408` is the SwiGLU intermediate size: the width of the gate and
up projections ($W_1, W_3 \in \mathbb{R}^{5120 \times 17408}$) and the input width of
the down projection ($W_2 \in \mathbb{R}^{17408 \times 5120}$).

On Blackhole, the tile-based circular buffer (CB) registers used by the matmul kernel
have a fixed L1 budget. When the output tensor of the first MLP matmul has 17408
columns, a pre-programmed `program_config` with an explicit CB allocation will conflict
with the CB allocation of the subsequent in-place operations.

The fix is to pass `program_config=None` in the model config for this MLP layer. TTNN
then auto-selects a matmul program that fits in L1 without a clash. The trade-off is a
small reduction in throughput compared to an explicitly tuned program config, but this
is preferable to a device hang.

## Activation Dtype Tuning

The final residual add applies a per-layer dtype from `decoders_optimizations`:

```python
activation_dtype = self.args.decoders_optimizations.get_tensor_dtype(
    decoder_id=self.layer_num, tensor=TensorGroup.ACTIVATION
)

out = ttnn.add(
    residual,
    hidden_states,
    memory_config=skip_mem_cfg,
    dtype=activation_dtype or ttnn.bfloat16,
)
```

`decoders_optimizations` is a per-model configuration object that can assign different
output dtypes to different layers. If a layer has no explicit override, the dtype
defaults to `ttnn.bfloat16`. This hook allows, for example, routing certain layers
through `bfp8` activations to reduce DRAM bandwidth without rewriting the forward logic.
Chapter 6 (Weight Precision) covers the full `decoders_optimizations` API.

---

**Next:** [`mlp_dispatch.md`](./mlp_dispatch.md)
