# Hybrid Architecture: 48 GDN + 16 Full Attention Layers

Qwen3.5-27B is not a standard transformer. It uses a **hybrid architecture** that mixes two distinct layer types across its 64 layers: Gated DeltaNet (GDN) linear attention layers and standard multi-head full attention layers. This design trades the quadratic-in-sequence-length KV cache of full attention for a fixed-size recurrence state in GDN layers, dramatically reducing memory consumption for long sequences while preserving the modeling power of full attention at regular intervals.

## The 3+1 Repeating Pattern

The 64 layers follow a strict repeating pattern defined by the `layer_types` list in the HuggingFace config (`config.json`, also read by the framework `ModelArgs` base class):

```
Layers  0- 2: linear_attention, linear_attention, linear_attention
Layer      3: full_attention
Layers  4- 6: linear_attention, linear_attention, linear_attention
Layer      7: full_attention
...
Layers 60-62: linear_attention, linear_attention, linear_attention
Layer     63: full_attention
```

This gives exactly **48 GDN layers** (indices 0, 1, 2, 4, 5, 6, ..., 60, 61, 62) and **16 full attention layers** (indices 3, 7, 11, ..., 63). The config specifies `"full_attention_interval": 4` (see `config.json:12`), meaning every 4th layer is full attention.

The pattern has a direct consequence for the L1 state management optimization covered in Chapter 6: groups of 3 consecutive GDN layers naturally form a "window" that executes between attention layers, and the rolling window strategy exploits this grouping. The `enable_l1_state()` method in `model.py` hardcodes `self._l1_window = 3` (see `model.py:227`) explicitly because it matches this architectural pattern.

## Model Dimensions

### Full Attention Dimensions

These are specified in the HuggingFace config (`config.json`) and parsed into `Qwen35ModelArgs` via the standard `ModelArgs` base class:

| Parameter | Config Key | Value |
|-----------|-----------|-------|
| Hidden size | `hidden_size` | 5120 |
| Attention Q heads (`n_heads`) | `num_attention_heads` | 24 |
| Attention KV heads (`n_kv_heads`) | `num_key_value_heads` | 4 |
| Attention head dim (`head_dim`) | `head_dim` | 256 |
| MLP intermediate (`hidden_dim`) | `intermediate_size` | 17408 |
| Partial RoPE dim | `partial_rotary_factor` × `head_dim` | 64 (of 256) |
| RoPE theta | `rope_theta` | 10,000,000.0 |

The partial RoPE dimension derives from `partial_rotary_factor: 0.25` in `config.json:55` applied to `head_dim = 256`, giving $256 \times 0.25 = 64$. The constant `ROPE_DIM = 64` in `model_config.py:40` captures this value, and `Qwen35ModelArgs.__init__()` sets `self.rope_dim = ROPE_DIM` (see `model_config.py:246`).

The attention layers use grouped-query attention (GQA) with a 6:1 ratio (24 Q heads to 4 KV heads). They also diverge from standard transformers in several ways covered in Chapter 2: partial RoPE (only 64 of 256 head dims are rotated), QK L2 normalization, and sigmoid output gating (`"attn_output_gate": true` in `config.json:9`).

### GDN Dimensions

The GDN architecture constants are defined as module-level constants in `model_config.py` (lines 28-32) because they are not parsed by the standard `ModelArgs` base class from the HF config, even though the values do appear in `config.json` under custom keys (`linear_num_key_heads`, `linear_num_value_heads`, etc.):

```python
GDN_Nk = 16   # Key heads         (config.json: linear_num_key_heads)
GDN_Dk = 128  # Key head dim      (config.json: linear_key_head_dim)
GDN_Nv = 48   # Value heads       (config.json: linear_num_value_heads)
GDN_Dv = 128  # Value head dim    (config.json: linear_value_head_dim)
GDN_CONV_KERNEL_SIZE = 4           (config.json: linear_conv_kernel_dim)
```

These yield aggregate dimensions used in TP splitting (see `model_config.py:34-37`):

$$\text{GDN QKV DIM} = N_k \cdot D_k + N_k \cdot D_k + N_v \cdot D_v = 16 \times 128 + 16 \times 128 + 48 \times 128 = 10240$$

$$\text{GDN Z DIM} = N_v \cdot D_v = 48 \times 128 = 6144$$

$$\text{GDN KEY DIM} = N_k \cdot D_k = 16 \times 128 = 2048$$

$$\text{GDN VALUE DIM} = N_v \cdot D_v = 48 \times 128 = 6144$$

The GDN layers have an asymmetric head structure: 16 key heads but 48 value heads. The repeat factor is $N_v / N_k = 48 / 16 = 3$: during the recurrence, each key head covers 3 value heads. This is analogous to GQA in full attention but applied to the linear recurrence state.

## GDN Recurrence State vs KV Cache

Unlike full attention layers whose KV cache grows linearly with sequence length, each GDN layer maintains a **fixed-size** recurrence state regardless of context length. The state is allocated in `TtGatedDeltaNet.reset_state()` (see `gdn.py:164`) as:

```python
self.rec_states = _to_mesh(
    torch.zeros(B * self.Nv_TP, self.Dk, self.Dv, dtype=torch.bfloat16)
)
```

The shape is `[B * Nv_TP, Dk, Dv]` where `Nv_TP = GDN_Nv / tp = 12` per device at TP=4, `Dk = 128`, and `Dv = 128`. Each token processed updates this state in-place via the DeltaNet recurrence (see Chapter 3 for recurrence mechanics). There is no accumulation of past tokens in memory — only the current state matrix matters.

This fixed footprint makes long-context generation tractable on the P150x4. The DRAM bandwidth cost of reading and writing the recurrence state on every decode step is the primary GDN decode bottleneck; the L1 rolling window optimization (Chapter 6) addresses this by keeping the states for groups of 3 GDN layers in L1 SRAM during their execution window.

## The `Transformer` Class Construction Flow

The `Transformer` class in `model.py` extends the framework `TTTransformer` and builds the model in two phases (see `model.py:41-95`):

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

The parent `TTTransformer.__init__()` builds all 64 layers using `Qwen35Attention` as the attention module. It also sets up the embedding, RMSNorm (with the `rms_norm_add_unit_offset = True` override at `model_config.py:245` for Qwen3.5's GemmaRMSNorm format), MLP, and LM head.

### Phase 2: Swap GDN Layers

Immediately after the parent constructor returns, `Transformer.__init__()` iterates over all 64 layers and **replaces** the attention module on every `"linear_attention"` layer (see `model.py:76-91`):

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
            transformation_mats=self.trans_mats_dict,
            configuration=args,
            ...
        )
```

This swap-after-construction pattern exists because the framework `TTTransformer` accepts only a single `attention_class` argument. Since Qwen3.5-27B needs two different attention implementations (full attention and GDN), the solution is to build with one (`Qwen35Attention`) and replace the other (`TtGatedDeltaNet`). `TtGatedDeltaNet` matches the `Attention` constructor signature so that the `TransformerBlock` framework layer can hold it without modification (see `gdn.py:72-74`).

### Phase 3: Load and Wire Weights

The `_load_and_wire_attention_weights()` method (see `model.py:97-216`) iterates over all 64 layers and loads the appropriate mesh tensors based on `layer_type`:

- **Full attention layers** (`"full_attention"`): loads fused Q+gate (`wqkv`), separate K and V (`wk`, `wv`), output (`wo`), and QK norm weights (`q_norm`, `k_norm`). The Q+gate weight is prepared by `prepare_attn_qg()`. K and V weights may be replicated via `replicate_kv_weight()` when `args.kv_replication` is `True`.

- **GDN layers** (`"linear_attention"`): loads fused QKVZ weight (Q, K, V interleaved by `prepare_gdn_qkv()` then concatenated with the Z projection per device), fused A+B projection, output projection, per-head parameters (`A_log`, `dt_bias`, `norm_w`), and 4 conv tap weights prepared by `prepare_conv_taps()`.

Each weight tensor is converted to the appropriate mesh tensor format using `_shard_w()` (column or row sharded), `_replicate()` (replicated on all devices), or `_shard_small()` (small per-head tensors sharded across devices). All main projection weights use `ttnn.bfloat8_b` precision; norms and small per-head parameters (`A_log`, `dt_bias`, `norm_w`) remain `bfloat16`.

## Factory Function

The `create_qwen35_model()` factory function in `model.py` (lines 350-401) provides a single entry point:

```python
model = create_qwen35_model(
    mesh_device,
    model_path="~/models/Qwen3.5-27B-FP8",
    max_batch_size=32,
    max_seq_len=131072,
    dtype=ttnn.bfloat8_b,
)
```

It loads the state dict via `load_qwen35_state_dict()` (which handles FP8 block-wise dequantization with 128×128 blocks), creates `Qwen35ModelArgs`, and constructs the full `Transformer`. Weight caching goes to `~/models/Qwen3.5-27B-mesh-tp4/framework/` (see `model.py:385`).

---

**Next:** [`tp_sharding_strategy.md`](./tp_sharding_strategy.md)
