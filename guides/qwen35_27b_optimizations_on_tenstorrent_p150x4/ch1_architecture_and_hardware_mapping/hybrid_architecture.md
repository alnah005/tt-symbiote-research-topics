# Hybrid Architecture: 48 GDN + 16 Full Attention Layers

Qwen3.5-27B is not a standard transformer. It uses a **hybrid architecture** that mixes two distinct layer types across its 64 layers: Gated DeltaNet (GDN) linear attention layers and standard multi-head full attention layers. This design trades the quadratic-in-sequence-length KV cache of full attention for a fixed-size recurrence state in GDN layers, dramatically reducing memory consumption for long sequences while preserving the modeling power of full attention at regular intervals.

## The 3+1 Repeating Pattern

The 64 layers follow a strict repeating pattern defined by the `layer_types` list in the HuggingFace config (`config.json`):

```
Layers  0- 2: linear_attention, linear_attention, linear_attention
Layer      3: full_attention
Layers  4- 6: linear_attention, linear_attention, linear_attention
Layer      7: full_attention
...
Layers 60-62: linear_attention, linear_attention, linear_attention
Layer     63: full_attention
```

This gives exactly **48 GDN layers** (indices 0, 1, 2, 4, 5, 6, ..., 60, 61, 62) and **16 full attention layers** (indices 3, 7, 11, ..., 63). The HF config specifies `full_attention_interval: 4`, meaning every 4th layer is full attention.

The pattern has a direct consequence for the L1 state management optimization covered in Chapter 6: groups of 3 consecutive GDN layers naturally form a "window" that executes between attention layers, and the rolling window strategy exploits this grouping.

## Model Dimensions

### Full Attention Dimensions

These are specified in the HuggingFace config and stored in `Qwen35ModelArgs`:

| Parameter | Config Key | Value |
|-----------|-----------|-------|
| Hidden size | `hidden_size` | 5120 |
| Attention Q heads (`n_heads`) | `num_attention_heads` | 24 |
| Attention KV heads (`n_kv_heads`) | `num_key_value_heads` | 4 |
| Attention head dim (`head_dim`) | `head_dim` | 256 |
| MLP intermediate (`hidden_dim`) | `intermediate_size` | 17408 |
| Partial RoPE dim | `ROPE_DIM` | 64 (of 256) |
| RoPE theta | `rope_theta` | 10,000,000.0 |

The attention layers use grouped-query attention (GQA) with a 6:1 ratio (24 Q heads to 4 KV heads). They also diverge from standard transformers in several ways covered in Chapter 2: partial RoPE (only 64 of 256 head dims are rotated), QK L2 normalization, and sigmoid output gating.

### GDN Dimensions

The GDN architecture constants are **not** in the standard HuggingFace config fields. They are defined as module-level constants in `model_config.py`:

```python
GDN_Nk = 16   # Key heads
GDN_Dk = 128  # Key head dim
GDN_Nv = 48   # Value heads
GDN_Dv = 128  # Value head dim
GDN_CONV_KERNEL_SIZE = 4
```

These yield aggregate dimensions used in TP splitting: QKV = 10240, Z = 6144, KEY = 2048, VALUE = 6144.

The GDN layers have an asymmetric head structure: 16 key heads but 48 value heads. During the recurrence, each key head is expanded (repeated) to cover 3 value heads (repeat factor = `Nv / Nk = 3`). This is analogous to GQA in attention but applied to the linear recurrence.

## GDN Recurrence State vs KV Cache

Unlike full attention layers whose KV cache grows linearly with sequence length, each GDN layer maintains a **fixed-size** recurrence state of shape `[B * Nv_TP, Dk, Dv]` regardless of context length. At TP=4 with batch size 32, this totals **576 MB across 48 GDN layers per device**. This fixed footprint makes long-context generation tractable on the P150x4, but the DRAM bandwidth cost of this state is the primary decode bottleneck (see Chapter 7 for the full memory budget analysis).

## The `Transformer` Class Construction Flow

The `Transformer` class in `model.py` extends the framework `TTTransformer` and builds the model in two phases:

### Phase 1: Build with Attention as Default

```python
class Transformer(TTTransformer):
    def __init__(self, args, dtype, mesh_device, state_dict, weight_cache_path, ...):
        super().__init__(
            args=args,
            dtype=dtype,
            mesh_device=mesh_device,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            attention_class=Qwen35Attention,
            rope_setup_class=Qwen35PartialRopeSetup,
            ...
        )
```

The parent `TTTransformer.__init__()` builds all 64 layers using `Qwen35Attention` as the attention module. It also sets up the embedding, RMSNorm (with the `rms_norm_add_unit_offset = True` override for Qwen3.5's GemmaRMSNorm format), MLP, and LM head.

### Phase 2: Swap GDN Layers

Immediately after the parent constructor returns, the `Transformer.__init__()` iterates over all 64 layers and **replaces** the attention module on GDN layers:

```python
for i in range(args.n_layers):
    if args.layer_types[i] == "linear_attention":
        self.layers[i].attention = TtGatedDeltaNet(
            mesh_device=mesh_device,
            tt_ccl=self.tt_ccl,
            args=args,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=i,
            dtype=dtype,
            ...
        )
```

This swap-after-construction pattern exists because the framework `TTTransformer` only takes a single `attention_class` argument. Since Qwen3.5-27B needs two different attention implementations (full attention and GDN), the solution is to build with one and replace the other.

### Phase 3: Load and Wire Weights

The `_load_and_wire_attention_weights()` method iterates over all 64 layers and loads the appropriate mesh tensors based on `layer_type`:

- **Full attention layers** (`"full_attention"`): loads fused Q+gate (`wqkv`), separate K, V, output (`wo`), and QK norm weights (`q_norm`, `k_norm`). The Q+gate weight is prepared by `prepare_attn_qg()`. K and V weights may be replicated via `replicate_kv_weight()` when `kv_replication` is enabled.

- **GDN layers** (`"linear_attention"`): loads fused QKVZ weight (Q, K, V interleaved by `prepare_gdn_qkv()` then concatenated with Z), fused A+B projection, output projection, per-head parameters (`A_log`, `dt_bias`, `norm_w`), and 4 conv tap weights prepared by `prepare_conv_taps()`.

Each weight tensor is converted to the appropriate mesh tensor format using `_shard_w()` (column or row sharded), `_replicate()` (replicated on all devices), or `_shard_small()` (small per-head tensors sharded across devices). All weights use `ttnn.bfloat8_b` precision except norms and small parameters which remain `bfloat16`.

## Factory Function

The `create_qwen35_model()` factory function in `model.py` provides a single entry point:

```python
model = create_qwen35_model(
    mesh_device,
    model_path="~/models/Qwen3.5-27B-FP8",
    max_batch_size=32,
    max_seq_len=131072,
    dtype=ttnn.bfloat8_b,
)
```

It loads the state dict via `load_qwen35_state_dict()` (which handles FP8 block-wise dequantization with 128x128 blocks), creates `Qwen35ModelArgs`, and constructs the full `Transformer`. Weight caching goes to `~/models/Qwen3.5-27B-mesh-tp4/framework/`.

---

**Next:** [`tp_sharding_strategy.md`](./tp_sharding_strategy.md)
