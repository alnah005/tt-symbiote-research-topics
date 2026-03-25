# Implementation Plan: TTNNBailingMoEAttentionT3K (Updated 2026-03-25)

## 1. Problem Description

### Objective
Implement `TTNNBailingMoEAttentionT3K` - a T3K-optimized attention module for the Bailing/Ling MoE model that:
- Runs on T3K (1x8 mesh device) with tensor parallelism
- Supports paged attention for efficient KV cache management
- Has separate `forward_prefill` and `forward_decode` paths
- Targets ~500 lines of code for maintainability

### Current State
The existing `TTNNBailingMoEAttention` class (lines 2198-2957 in `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/attention.py`) provides:
- Basic paged attention support via `TTNNPagedAttentionKVCache`
- Distributed linear operations via CCL manager
- Separate Q/K/V projections for handling `num_kv_heads < num_devices`
- Working prefill and decode paths

However, it lacks the optimizations found in `tt-transformers/tt/attention.py`:
- Column-parallel QKV with proper weight sharding (uses `ShardTensor2dMesh`)
- Row-parallel output with `reduce_scatter`
- Optimized memory configurations for decode (`L1_MEMORY_CONFIG`)
- `nlp_create_qkv_heads_decode` for efficient head creation
- Fused `paged_fused_update_cache` for decode

### Target Model Configuration (Ling-mini-2.0 / BailingMoeV2)
- `hidden_size`: 2048
- `num_attention_heads`: 16 (Q heads)
- `num_key_value_heads`: 4 (KV heads) - GQA with 4:1 ratio
- `head_dim`: 128
- `partial_rotary_factor`: 0.5 (partial RoPE on 64 dims)
- `use_qk_norm`: True (query/key layer norms)

### T3K Constraints (from research cache)
- 8 devices in 1x8 linear mesh
- `num_kv_heads=4 < num_devices=8` requires special handling
- Per-link bandwidth: ~12.5 GB/s unidirectional
- `cluster_axis=1` for collectives on T3K
- Use `ttnn.Topology.Linear` (not Ring) for T3K

### Key Deficiencies to Address

1. **Suboptimal decode path**: Uses generic SDPA, not `paged_scaled_dot_product_attention_decode`

2. **Missing T3K-specific optimizations**: No `nlp_create_qkv_heads_decode`, no fused RoPE, no optimized memory configs

3. **Inefficient collective ops**: Uses basic `all_gather`, not async variants with proper semaphore management

4. **No weight pre-sharding**: Weights converted at runtime, not pre-sharded via `ShardTensor2dMesh`

---

## 2. Analysis of Existing Code

### Reference: tt-transformers Attention (`/home/ttuser/salnahari/tt-metal/models/tt_transformers/tt/attention.py`)

The production-ready attention implementation demonstrates T3K-optimized patterns:

**Weight Loading Pattern (lines 221-254):**
```python
# QKV weights sharded across devices using ShardTensor2dMesh
qkv_list = []
for i in range(self.num_devices_per_group):
    wq_selected = torch.chunk(state_dict[f"{wq_str}.weight"], self.num_devices_per_group, dim=0)[i]
    wk_selected = torch.chunk(state_dict[f"{wk_str}.weight"], self.num_devices_per_group, dim=0)[i]
    wv_selected = torch.chunk(state_dict[f"{wv_str}.weight"], self.num_devices_per_group, dim=0)[i]
    qkv = torch.cat([wq.T, wk.T, wv.T], dim=-1)
    qkv_list.append(qkv)
qkv_cat = torch.cat(qkv_list, dim=-1).unsqueeze(0).unsqueeze(0)

self.wqkv = ttnn.as_tensor(
    qkv_cat,
    dtype=self.wqkv_dtype,
    mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(3, 2), mesh_shape=config.cluster_shape),
)
```

**Decode Forward (lines 482-780):**
1. QKV matmul with DRAM-sharded weights + HiFi2 compute
2. `all_reduce` across devices (cluster_axis=1)
3. `nlp_create_qkv_heads_decode` for efficient head splitting
4. Q/K norm via `RMSNorm` (optional)
5. RoPE via `rotary_embedding_llama` or `rotary_embedding_llama_fused_qk`
6. `paged_fused_update_cache` (K and V in one call)
7. `paged_scaled_dot_product_attention_decode` for SDPA
8. `nlp_concat_heads_decode`
9. Output matmul with `all_gather_matmul_async` (fused) or separate all_gather + wo linear
10. `all_reduce` or `reduce_scatter` for final output

**Prefill Forward (lines 782-1021):**
1. QKV matmul (may reshape for long sequences > MAX_QKV_MM_SEQ_LEN)
2. `all_reduce`
3. `nlp_create_qkv_heads` for head splitting (not decode version)
4. Q/K norm
5. RoPE via `rotary_embedding_llama`
6. `paged_fill_cache` for KV cache
7. Standard SDPA with `scaled_dot_product_attention` (is_causal=True)
8. `nlp_concat_heads`
9. `all_gather_async` + wo linear
10. `all_reduce` or `reduce_scatter`

### Reference: Current TTNNBailingMoEAttention

Key patterns to preserve from existing implementation:
- `_use_separate_qkv`: Handles `num_kv_heads < num_devices` by splitting fused QKV
- `TTNNLinearIColShardedWRowSharded`: Column-parallel input, row-sharded weights, reduce_scatter after
- `TTNNLinearIReplicatedWColSharded`: Replicated input, column-sharded weights (for K/V)
- `_maybe_all_gather`: CCL-managed all_gather via `device_state.ccl_manager`
- `_to_replicated`: Converts all-gathered tensor to replicated topology for paged kernels
- `_apply_partial_rope`: Handles `partial_rotary_factor < 1.0`
- `_apply_qk_norm`: Query/key layer normalization

### Linear Module Patterns (`/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/linear.py`)

**TTNNLinearIColShardedWRowSharded (lines 126-176):**
- Input sharded on dim=-1 (hidden_size across devices)
- Weight sharded on dim=-2 (input dimension)
- Uses `reduce_scatter_minimal_async` after matmul (dim=3, cluster_axis=1)
- Appropriate for: QKV projection (column-parallel), output projection

**TTNNLinearIReplicatedWColSharded (lines 256-276):**
- Input replicated across devices (via prior all_gather)
- Weight sharded on dim=-1 (output dimension)
- No CCL after matmul - output is column-sharded
- Appropriate for: K/V projection when `num_kv_heads < num_devices`

---

## 3. Step-by-Step Implementation Plan

### Step 1: Create Module Skeleton (~40 lines)

**File**: `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/bailing_attention_t3k.py`

```python
"""T3K-optimized attention for BailingMoE models with paged attention."""

from dataclasses import dataclass
from typing import Optional
import torch
import ttnn

from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.modules.attention import TTNNPagedAttentionKVCache, PagedAttentionConfig


@dataclass
class BailingAttentionT3KConfig:
    """Configuration for T3K attention."""
    hidden_size: int = 2048
    num_attention_heads: int = 16
    num_kv_heads: int = 4
    head_dim: int = 128
    partial_rotary_factor: float = 0.5
    use_qk_norm: bool = True
    max_batch_size: int = 32
    max_seq_len: int = 8192
    # Paged attention config
    block_size: int = 64
    max_num_blocks: int = 2048
    # T3K config
    num_devices: int = 8
    # Dtype config
    wqkv_dtype: ttnn.DataType = ttnn.bfloat8_b
    wo_dtype: ttnn.DataType = ttnn.bfloat8_b
    kv_cache_dtype: ttnn.DataType = ttnn.bfloat8_b
```

### Step 2: Implement Weight Initialization (~100 lines)

```python
class TTNNBailingMoEAttentionT3K(TTNNModule):
    def __init__(self, config: BailingAttentionT3KConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.scale = self.head_dim ** -0.5
        self.n_local_heads = self.num_heads // config.num_devices  # 16/8 = 2
        self.n_local_kv_heads = max(1, self.num_kv_heads // config.num_devices)  # 4/8 = 0.5 -> 1

        # Weights - set in from_torch
        self.wqkv = None
        self.wo = None
        self.q_norm = None
        self.k_norm = None

        # KV cache - initialized in init_kv_cache
        self.k_cache = None
        self.v_cache = None
        self.page_table = None

        # Compute configs - set in move_weights_to_device_impl
        self.compute_kernel_config = None
        self.sdpa_decode_program_config = None
        self.transformation_mats = None

    @classmethod
    def from_torch(cls, torch_attn, mesh_device, config: BailingAttentionT3KConfig, layer_idx: int = 0):
        """Create from PyTorch attention with T3K weight sharding."""
        module = cls(config, layer_idx)
        module._mesh_device = mesh_device

        # Extract and transpose weights (HF format: [out, in] -> TTNN: [in, out])
        qkv_weight = torch_attn.query_key_value.weight  # [q_size + 2*kv_size, hidden]
        wo_weight = torch_attn.dense.weight  # [hidden, hidden]

        q_size = config.num_attention_heads * config.head_dim
        kv_size = config.num_kv_heads * config.head_dim

        # For T3K with num_kv_heads < num_devices, we need special handling:
        # Q: sharded normally (16 heads / 8 devices = 2 heads/device)
        # K/V: replicated (4 heads < 8 devices - can't shard evenly)

        # Build per-device QKV chunks following tt-transformers pattern
        qkv_list = []
        for i in range(config.num_devices):
            # Q: chunk evenly
            q_chunk = torch.chunk(qkv_weight[:q_size, :], config.num_devices, dim=0)[i]
            # K/V: replicate full weight (will be sliced per-device in forward)
            # Note: For 4 KV heads on 8 devices, pairs of devices share KV heads
            kv_device_idx = i // 2  # devices 0-1 share head 0, 2-3 share head 1, etc.
            k_chunk = torch.chunk(qkv_weight[q_size:q_size+kv_size, :], max(1, config.num_kv_heads), dim=0)[kv_device_idx % config.num_kv_heads]
            v_chunk = torch.chunk(qkv_weight[q_size+kv_size:, :], max(1, config.num_kv_heads), dim=0)[kv_device_idx % config.num_kv_heads]

            # Transpose and concat: [in, out_q + out_k + out_v]
            qkv_chunk = torch.cat([q_chunk.T, k_chunk.T, v_chunk.T], dim=-1)
            qkv_list.append(qkv_chunk)

        qkv_cat = torch.cat(qkv_list, dim=-1).unsqueeze(0).unsqueeze(0)

        module.wqkv = ttnn.as_tensor(
            qkv_cat,
            dtype=config.wqkv_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(3, 2), mesh_shape=(1, config.num_devices)),
        )

        # Output projection: row-parallel (input sharded, output reduced)
        wo_transposed = wo_weight.T.unsqueeze(0).unsqueeze(0)
        module.wo = ttnn.as_tensor(
            wo_transposed,
            dtype=config.wo_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(2, 3), mesh_shape=(1, config.num_devices)),
        )

        # Q/K norms (if enabled)
        if config.use_qk_norm and hasattr(torch_attn, 'query_layernorm'):
            from models.experimental.tt_symbiote.modules.normalization import TTNNRMSNorm
            module.q_norm = TTNNRMSNorm.from_torch(torch_attn.query_layernorm)
            module.k_norm = TTNNRMSNorm.from_torch(torch_attn.key_layernorm)

        return module
```

### Step 3: Implement KV Cache Setup (~60 lines)

```python
def init_kv_cache(self, mesh_device, paged_config: PagedAttentionConfig = None):
    """Initialize paged KV cache on device."""
    if paged_config is None:
        paged_config = PagedAttentionConfig(
            block_size=self.config.block_size,
            max_num_blocks=self.config.max_num_blocks,
            batch_size=self.config.max_batch_size,
        )

    # KV cache shape: [max_blocks, local_kv_heads, block_size, head_dim]
    # For T3K with 4 KV heads on 8 devices, each device gets 1 KV head (with replication)
    cache_shape = (
        paged_config.max_num_blocks,
        self.n_local_kv_heads,
        paged_config.block_size,
        self.head_dim,
    )

    self.k_cache = ttnn.zeros(
        cache_shape,
        dtype=self.config.kv_cache_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    self.v_cache = ttnn.zeros(
        cache_shape,
        dtype=self.config.kv_cache_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Page table: replicated across all devices
    page_table = torch.arange(paged_config.max_num_blocks, dtype=torch.int32)
    page_table = page_table.reshape(paged_config.batch_size, paged_config.blocks_per_sequence)

    self.page_table = ttnn.from_torch(
        page_table,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    self.paged_config = paged_config
```

### Step 4: Implement forward_decode (~120 lines)

This is the critical path for token generation performance.

```python
def forward_decode(
    self,
    x: ttnn.Tensor,              # [1, 1, batch, hidden_size] sharded
    current_pos: ttnn.Tensor,    # [batch] int32, replicated
    rot_mats: tuple,             # (cos, sin) transformation matrices
    page_table: ttnn.Tensor = None,
) -> ttnn.Tensor:
    """Optimized decode forward for T3K with paged attention."""

    # 1. QKV projection with DRAM-sharded matmul
    xqkv = ttnn.linear(
        x,
        self.wqkv,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=self.compute_kernel_config,
    )

    # 2. All-reduce QKV across devices (sum partial products from row-parallel matmul)
    xqkv = ttnn.experimental.all_reduce_async(
        xqkv,
        cluster_axis=1,
        num_links=1,
        topology=ttnn.Topology.Linear,
        multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_ar_semaphore_handles(1),
        barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # 3. Split into Q, K, V heads using optimized decode kernel
    q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
        xqkv,
        num_heads=self.n_local_heads,
        num_kv_heads=self.n_local_kv_heads,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn.deallocate(xqkv)

    # 4. Apply Q/K norms if enabled
    if self.config.use_qk_norm and self.q_norm is not None:
        q = self.q_norm(q)
        k = self.k_norm(k)

    # 5. Apply RoPE (fused Q/K version for efficiency)
    if hasattr(self, 'use_fused_qk_rope') and self.use_fused_qk_rope:
        q, k = ttnn.experimental.rotary_embedding_llama_fused_qk(
            q, k, rot_mats[0], rot_mats[1], self.transformation_mats
        )
    else:
        q = ttnn.experimental.rotary_embedding_llama(
            q, rot_mats[0], rot_mats[1], self.transformation_mats, is_decode_mode=True
        )
        k = ttnn.experimental.rotary_embedding_llama(
            k, rot_mats[0], rot_mats[1], self.transformation_mats, is_decode_mode=True
        )

    # 6. Update paged KV cache (fused K+V update)
    pt = page_table if page_table is not None else self.page_table
    ttnn.experimental.paged_fused_update_cache(
        self.k_cache, k,
        self.v_cache, v,
        update_idxs_tensor=current_pos,
        page_table=pt,
    )
    ttnn.deallocate(k)
    ttnn.deallocate(v)

    # 7. Paged SDPA decode
    attn_output = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        q,
        self.k_cache,
        self.v_cache,
        page_table_tensor=pt,
        cur_pos_tensor=current_pos,
        scale=self.scale,
        program_config=self.sdpa_decode_program_config,
        compute_kernel_config=self.compute_kernel_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(q)

    # 8. Concat heads
    attn_output = ttnn.experimental.nlp_concat_heads_decode(
        attn_output,
        num_heads=self.n_local_heads,
    )

    # 9. All-gather for output projection input
    attn_output = ttnn.experimental.all_gather_async(
        attn_output,
        dim=3,
        num_links=1,
        topology=ttnn.Topology.Linear,
        multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_ag_semaphore_handles(1),
        barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # 10. Output projection
    output = ttnn.linear(
        attn_output,
        self.wo,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=self.compute_kernel_config,
    )
    ttnn.deallocate(attn_output)

    # 11. Reduce-scatter for final output
    output = ttnn.experimental.reduce_scatter_minimal_async(
        output,
        dim=3,
        cluster_axis=1,
        num_links=1,
        topology=ttnn.Topology.Linear,
        multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_rs_semaphore_handles(1),
        barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return output
```

### Step 5: Implement forward_prefill (~100 lines)

```python
def forward_prefill(
    self,
    x: ttnn.Tensor,              # [1, 1, seq_len, hidden_size] sharded
    rot_mats: tuple,             # (cos, sin) rotation matrices
    user_id: int = 0,
    page_table: ttnn.Tensor = None,
) -> ttnn.Tensor:
    """Prefill forward for prompt processing on T3K."""
    seq_len = x.shape[-2]

    # Handle long sequences by reshaping
    MAX_QKV_MM_SEQ_LEN = 8192
    if seq_len > MAX_QKV_MM_SEQ_LEN:
        x = ttnn.reshape(x, [1, seq_len // MAX_QKV_MM_SEQ_LEN, MAX_QKV_MM_SEQ_LEN, -1])

    # 1. QKV projection
    xqkv = ttnn.linear(
        x,
        self.wqkv,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=self.compute_kernel_config,
    )

    # 2. All-reduce across devices
    xqkv = ttnn.experimental.all_reduce_async(
        xqkv,
        cluster_axis=1,
        num_links=1,
        topology=ttnn.Topology.Linear,
        multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_ar_semaphore_handles(1),
        barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    if seq_len > MAX_QKV_MM_SEQ_LEN:
        xqkv = ttnn.reshape(xqkv, [1, 1, seq_len, -1])
    ttnn.deallocate(x)

    # 3. Split heads (prefill version, not decode)
    q, k, v = ttnn.experimental.nlp_create_qkv_heads(
        xqkv,
        num_heads=self.n_local_heads,
        num_kv_heads=self.n_local_kv_heads,
        transpose_k_heads=False,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(xqkv)

    # 4. Q/K norm
    if self.config.use_qk_norm and self.q_norm is not None:
        q = self.q_norm(q)
        k = self.k_norm(k)

    # 5. RoPE (prefill mode)
    if q.dtype != ttnn.bfloat16:
        q = ttnn.typecast(q, ttnn.bfloat16)
    if k.dtype != ttnn.bfloat16:
        k = ttnn.typecast(k, ttnn.bfloat16)

    q = ttnn.experimental.rotary_embedding_llama(
        q, rot_mats[0], rot_mats[1], self.transformation_mats, is_decode_mode=False
    )
    k = ttnn.experimental.rotary_embedding_llama(
        k, rot_mats[0], rot_mats[1], self.transformation_mats, is_decode_mode=False
    )

    # 6. Fill paged cache
    pt = page_table if page_table is not None else self.page_table
    k_8b = ttnn.typecast(k, self.config.kv_cache_dtype)
    v_8b = ttnn.typecast(v, self.config.kv_cache_dtype)

    ttnn.experimental.paged_fill_cache(self.k_cache, k_8b, pt, batch_idx=user_id)
    ttnn.experimental.paged_fill_cache(self.v_cache, v_8b, pt, batch_idx=user_id)

    # 7. SDPA (causal)
    q_8b = ttnn.typecast(q, ttnn.bfloat8_b)
    attn_output = ttnn.transformer.scaled_dot_product_attention(
        q_8b, k_8b, v_8b,
        is_causal=True,
        scale=self.scale,
        compute_kernel_config=self.compute_kernel_config,
        program_config=self.sdpa_prefill_program_config,
    )
    ttnn.deallocate(q)
    ttnn.deallocate(k)
    ttnn.deallocate(v)

    # 8. Concat heads
    attn_output = ttnn.reshape(attn_output, [1, self.n_local_heads, -1, self.head_dim])
    attn_output = ttnn.experimental.nlp_concat_heads(attn_output, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # Reshape for long sequences
    if seq_len > 1024:
        attn_output = ttnn.reshape(attn_output, [1, seq_len // 1024, 1024, -1])

    # 9. All-gather
    attn_output = ttnn.experimental.all_gather_async(
        attn_output,
        dim=3,
        num_links=1,
        topology=ttnn.Topology.Linear,
        multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_ag_semaphore_handles(1),
        barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # 10. Output projection + reduce
    output = ttnn.linear(attn_output, self.wo, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(attn_output)

    if seq_len > 1024:
        output = ttnn.reshape(output, [1, 1, seq_len, -1])

    output = ttnn.experimental.reduce_scatter_minimal_async(
        output,
        dim=3,
        cluster_axis=1,
        num_links=1,
        topology=ttnn.Topology.Linear,
        multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_rs_semaphore_handles(1),
        barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return output
```

### Step 6: Implement Unified Forward and Helpers (~80 lines)

```python
def forward(
    self,
    hidden_states: ttnn.Tensor,
    position_embeddings: tuple,
    attention_mask: Optional[ttnn.Tensor] = None,
    past_key_values = None,
    cache_position: Optional[torch.Tensor] = None,
    mode: str = None,  # "prefill" or "decode", auto-detected if None
    user_id: int = 0,
    page_table: ttnn.Tensor = None,
    **kwargs,
) -> tuple:
    """Unified forward dispatching to decode or prefill."""

    # Unwrap TorchTTNNTensor if needed
    if hasattr(hidden_states, "to_ttnn"):
        hidden_states = hidden_states.to_ttnn

    seq_length = hidden_states.shape[-2]

    # Auto-detect mode based on sequence length
    if mode is None:
        mode = "decode" if seq_length == 1 else "prefill"

    # Convert position_embeddings to rotation matrices
    rot_mats = self._prepare_rot_mats(position_embeddings)

    if mode == "prefill":
        output = self.forward_prefill(
            hidden_states,
            rot_mats,
            user_id=user_id,
            page_table=page_table,
        )
    else:
        current_pos = self._get_current_pos(cache_position)
        output = self.forward_decode(
            hidden_states,
            current_pos,
            rot_mats,
            page_table=page_table,
        )

    return output, None, past_key_values

def _prepare_rot_mats(self, position_embeddings):
    """Convert cos/sin position embeddings to transformation matrices."""
    cos, sin = position_embeddings

    # Ensure tensors are on device and replicated
    if isinstance(cos, torch.Tensor):
        cos = ttnn.from_torch(
            cos.to(torch.bfloat16),
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
    if isinstance(sin, torch.Tensor):
        sin = ttnn.from_torch(
            sin.to(torch.bfloat16),
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )

    return (cos, sin)

def _get_current_pos(self, cache_position):
    """Convert cache_position to ttnn tensor for decode."""
    if cache_position is None:
        raise ValueError("cache_position required for decode mode")

    if isinstance(cache_position, torch.Tensor):
        return ttnn.from_torch(
            cache_position.flatten().to(torch.int32),
            device=self.device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    return cache_position

def move_weights_to_device_impl(self):
    """Initialize compute configs when device is available."""
    super().move_weights_to_device_impl()

    self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
        self.device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    grid = self.device.compute_with_storage_grid_size()
    self.sdpa_decode_program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        q_chunk_size=0,  # Decode mode
        k_chunk_size=0,
        exp_approx_mode=False,
    )

    self.sdpa_prefill_program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        q_chunk_size=256,
        k_chunk_size=256,
        exp_approx_mode=False,
    )

    # Initialize transformation matrices for RoPE
    # (Implementation depends on partial_rotary_factor)
    self._init_transformation_mats()

def _init_transformation_mats(self):
    """Initialize transformation matrices for rotary embeddings."""
    # For partial_rotary_factor=0.5, only first 64 dims get RoPE
    rotary_dim = int(self.head_dim * self.config.partial_rotary_factor)
    # Create identity-like transformation for non-fused RoPE path
    self.transformation_mats = None  # Will be created on first use
```

---

## 4. Success Criteria

### Functional Requirements
1. **Prefill correctness**: Output matches `TTNNBailingMoEAttention` within PCC > 0.99
2. **Decode correctness**: Token generation produces coherent text
3. **Paged attention functional**: KV cache updates correctly across decode steps
4. **T3K parallelism**: All 8 devices utilized with proper weight sharding

### Performance Requirements
1. **Decode latency**: < 15ms per token for batch_size=1 (baseline target)
2. **Prefill throughput**: > 500 tokens/sec for prompt processing
3. **Memory efficiency**: KV cache fits in DRAM with paged allocation
4. **CCL overhead**: < 20% of total decode time for collectives

### Code Quality
1. **Line count**: ~500 lines (excluding comments/docstrings)
2. **No code duplication**: Reuse existing linear/norm/rope modules where possible
3. **Clear separation**: Distinct prefill/decode paths
4. **Testable**: Unit tests for each major component

---

## 5. Risks and Mitigations

### Risk 1: num_kv_heads < num_devices (4 KV heads on 8 devices)
**Problem**: Standard sharding fails when `num_kv_heads=4 < num_devices=8`.

**Mitigation Options**:
- **Option A (Recommended)**: Pair devices to share KV heads (devices 0-1 share head 0, etc.)
- **Option B**: Replicate all KV heads on all devices (memory overhead)
- **Option C**: Use separate K/V projections with `TTNNLinearIReplicatedWColSharded`

**Selected Approach**: Option A - pairs devices, maintaining efficiency while supporting the configuration.

### Risk 2: Partial RoPE (partial_rotary_factor=0.5)
**Problem**: `rotary_embedding_llama` expects full head_dim rotation.

**Mitigation**:
- Use non-fused RoPE path from `TTNNRotaryPositionEmbedding`
- Or: Split head_dim [0:64] for RoPE, concat with [64:128] passthrough
- Note: tt-transformers uses `transformation_mats["prefill"]` / `["decode"]` for this

### Risk 3: CCL Semaphore Management
**Problem**: T3K requires proper semaphore cycling for async collectives to avoid deadlocks.

**Mitigation**:
- Use `device_state.ccl_manager.get_and_cycle_*_semaphore_handles(1)` pattern
- Follow pattern from `TTNNLinearIColShardedWRowSharded.forward()`
- Ensure matching number of collectives on all devices

### Risk 4: Memory Configuration Mismatch
**Problem**: Different memory configs between operations can cause L1 fragmentation or DRAM thrashing.

**Mitigation**:
- Start with `ttnn.DRAM_MEMORY_CONFIG` everywhere (safe default)
- Profile decode path with Tracy
- Selectively move hot tensors to L1: Q after QKV split, SDPA inputs/outputs

### Risk 5: Page Table Topology
**Problem**: Paged attention kernels expect replicated page tables, but default mesh placement may shard.

**Mitigation**:
- Always use `ttnn.ReplicateTensorToMesh(mesh_device)` for page tables and `cur_pos_tensor`
- Validate in `init_kv_cache` that page_table is replicated

### Risk 6: Sequence Length Constraints
**Problem**: SDPA and matmul kernels have maximum sequence length limits.

**Mitigation**:
- Reshape long sequences: `[1, 1, seq, dim]` -> `[1, seq//1024, 1024, dim]` for matmuls
- Use chunked prefill for very long sequences (> 8192)
- Document sequence length limits in config

---

## 6. File Structure

```
models/experimental/tt_symbiote/modules/
    bailing_attention_t3k.py      # NEW FILE (~500 lines)
    attention.py                   # Existing (keep TTNNBailingMoEAttention for fallback)
    linear.py                      # Existing (reuse distributed linear classes)
    normalization.py               # Existing (reuse TTNNRMSNorm)
    rope.py                        # Existing (may need partial RoPE additions)

models/experimental/tt_symbiote/tests/
    test_bailing_attention_t3k.py  # NEW: unit tests for T3K attention
```

---

## 7. Testing Strategy

### Unit Tests
1. **Weight loading**: Verify sharding produces correct per-device weights
2. **QKV projection**: Output shapes and values match reference
3. **RoPE**: Partial rotary produces correct rotated/unrotated split
4. **Paged cache**: Update and retrieval return correct values
5. **Collectives**: all_reduce/all_gather/reduce_scatter produce expected results

### Integration Tests
1. **Full forward pass**: Single layer, prefill + 10 decode steps
2. **Multi-layer**: 4 layers with paged cache, verify no memory leak
3. **Comparison test**: PCC vs `TTNNBailingMoEAttention` on same input

### Performance Tests
1. **Decode latency**: Measure time-to-first-token and token/sec
2. **Tracy profiling**: Identify bottlenecks in decode path
3. **Memory tracking**: Monitor L1/DRAM usage across decode steps

---

## 8. Implementation Order

| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | Module skeleton + config dataclass | `bailing_attention_t3k.py` with class structure |
| 2 | `from_torch` + weight sharding | Working weight initialization |
| 3 | `init_kv_cache` + `forward_prefill` | Prefill passes unit tests |
| 4 | `forward_decode` + paged SDPA | Decode passes unit tests |
| 5 | Integration + optimization | Full forward works, Tracy profiling |

---

## 9. Critical Reference Files

| File | Purpose |
|------|---------|
| `/home/ttuser/salnahari/tt-metal/models/tt_transformers/tt/attention.py` | Production T3K attention patterns |
| `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/attention.py` | Current Bailing attention (lines 2198-2957) |
| `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/linear.py` | Distributed linear modules |
| T3K Mesh Device Optimizations guide | Topology and collective patterns |
| TT Transformers Key Optimizations guide | Performance patterns |

---

## 10. Estimated Line Count

| Component | Lines |
|-----------|-------|
| Config dataclass | 30 |
| `__init__` | 40 |
| `from_torch` | 80 |
| `init_kv_cache` | 50 |
| `forward_decode` | 100 |
| `forward_prefill` | 90 |
| `forward` + helpers | 80 |
| `move_weights_to_device_impl` | 30 |
| **Total** | **~500** |
