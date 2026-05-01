# Ling Attention Implementations Comparison

This file contains the full implementations of Ling (BailingMoeV2) attention from both PyTorch/HuggingFace and TTNN for comparison.

---

## PyTorch/HuggingFace Implementation

Source: `/home/ttuser/.cache/huggingface/modules/transformers_modules/inclusionAI/Ling-mini-2.0/ae2925e082ef9e311fbbb01f2720006611bbdb69/modeling_bailing_moe_v2.py`

### Helper Functions

```python
# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings on the first half or full tensor
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
```

### BailingMoeV2Attention (Eager/Manual Attention)

```python
# Copied from transformers.models.llama.modeling_llama.LlamaAttention with Llama->BailingMoeV2
class BailingMoeV2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: BailingMoeV2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim or self.hidden_size // self.num_heads
        partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
        self.rope_dim = int(self.head_dim * partial_rotary_factor)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        self.query_key_value = nn.Linear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=config.use_qkv_bias,
        )

        if self.config.use_qk_norm:
            self.query_layernorm = BailingMoeV2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.key_layernorm = BailingMoeV2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.dense = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.use_bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(bsz, q_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)

        query_states, key_states, value_states = qkv.split(
            [self.num_heads, self.num_key_value_heads, self.num_key_value_heads], dim=-2
        )
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if self.config.use_qk_norm:
            query_states = self.query_layernorm(query_states)
            key_states = self.key_layernorm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        kv_seq_len = key_states.shape[-2]
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.dense(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
```

### BailingMoeV2SdpaAttention (SDPA Attention)

```python
# Copied from transformers.models.llama.modeling_llama.LlamaSdpaAttention with Llama->BailingMoeV2
class BailingMoeV2SdpaAttention(BailingMoeV2Attention):
    """
    BailingMoeV2 attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `BailingMoeV2Attention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from BailingMoeV2Attention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "BailingMoeV2Model is using BailingMoeV2SdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()

        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(bsz, q_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)

        query_states, key_states, value_states = qkv.split(
            [self.num_heads, self.num_key_value_heads, self.num_key_value_heads], dim=-2
        )
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if self.config.use_qk_norm:
            query_states = self.query_layernorm(query_states)
            key_states = self.key_layernorm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None:
            kv_seq_len = key_states.shape[-2]
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.dense(attn_output)

        return attn_output, None, past_key_value
```

---

## TTNN Implementation

Source: `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/attention.py`

### TTNNBailingMoEAttention Class

```python
class TTNNBailingMoEAttention(TTNNModule):
    """TTNN Attention for BailingMoeV2 (Ling-mini-2.0 model).

    Supports both standard DynamicCache and TTNNPagedAttentionKVCache
    for paged attention with on-device KV storage.
    """

    def __init__(self):
        super().__init__()
        self.num_heads = None
        self.num_kv_heads = None
        self.head_dim = None
        self.hidden_size = None
        self.use_qk_norm = False
        self.partial_rotary_factor = 1.0
        self.is_causal = True
        self.scaling = None

        self.query_key_value = None
        self.dense = None
        self.query_layernorm = None
        self.key_layernorm = None
        self.rope = None
        self.sdpa = None

        # Separate Q, K, V projections for distributed mode when num_kv_heads < num_devices
        # In this case, Q is sharded (num_heads >= num_devices) but K/V must be replicated
        self._use_separate_qkv = False
        self.q_proj = None
        self.k_proj = None
        self.v_proj = None

    @property
    def _is_distributed(self):
        """Check if running in distributed mode with CCL manager."""
        return (
            self.device_state is not None
            and hasattr(self.device_state, "ccl_manager")
            and self.device_state.ccl_manager is not None
        )

    def _maybe_all_gather(self, tensor):
        """All-gather tensor across mesh devices if in distributed mode."""
        if not self._is_distributed:
            return tensor.to_ttnn if hasattr(tensor, "to_ttnn") else tensor
        t = tensor.to_ttnn if hasattr(tensor, "to_ttnn") else tensor
        gathered = ttnn.experimental.all_gather_async(
            t,
            dim=-1,
            multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_ag_semaphore_handles(1),
            barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
            num_links=1,
            topology=ttnn.Topology.Linear,
        )
        ttnn.synchronize_device(self.device)
        # Ensure output is BFLOAT16 for compatibility with downstream ops (e.g., RoPE)
        if gathered.dtype != ttnn.bfloat16:
            gathered = ttnn.typecast(gathered, ttnn.bfloat16)
        return gathered

    def _to_replicated(self, tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Convert a multi-device tensor to an explicitly replicated tensor.

        After all-gather the data is identical on every device but the mesh
        topology metadata differs from ReplicateTensorToMesh. Paged-attention
        kernels require the replicated topology, so we round-trip through the
        host for decode tokens (tiny tensors, negligible overhead).
        """
        if self.device.get_num_devices() <= 1:
            return tensor
        t = tensor
        if isinstance(t, TorchTTNNTensor):
            t = t.to_ttnn
        orig_shape = list(t.shape)
        mesh_composer = ttnn.ConcatMeshToTensor(self.device, dim=0)
        t_torch = ttnn.to_torch(t, mesh_composer=mesh_composer)
        t_torch = t_torch[: orig_shape[0]]
        return ttnn.from_torch(
            t_torch,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            dtype=t.dtype,
            layout=t.layout,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    @classmethod
    def from_torch(cls, torch_attn, distributed: bool = True):
        """Create TTNNBailingMoEAttention from BailingMoeV2Attention/SdpaAttention.

        Args:
            torch_attn: PyTorch BailingMoeV2 attention module
            distributed: Whether to use distributed linear/norm modules for mesh devices.
                         Defaults to True for multi-device compatibility.

        Note:
            When distributed=True and the model has fewer KV heads than devices (e.g.,
            Ling-mini-2.0 with 4 KV heads on 8 devices), this method automatically
            splits the fused QKV projection into separate Q, K, V projections:
            - Q projection uses TTNNLinearIColShardedWRowSharded (sharded, since num_heads >= num_devices)
            - K/V projections use TTNNLinearIReplicatedWColSharded (replicated input, col-sharded output)
            This allows running on more devices than KV heads.
        """
        from models.experimental.tt_symbiote.modules.normalization import TTNNRMSNorm

        new_attn = cls()
        new_attn._fallback_torch_layer = torch_attn

        # Extract attention configuration
        config = torch_attn.config
        new_attn.num_heads = config.num_attention_heads
        new_attn.num_kv_heads = config.num_key_value_heads
        new_attn.head_dim = config.hidden_size // config.num_attention_heads
        new_attn.hidden_size = config.hidden_size
        new_attn.partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        new_attn.use_qk_norm = getattr(config, "use_qk_norm", False)
        new_attn.scaling = new_attn.head_dim**-0.5

        # Select linear class based on distributed mode
        LinearCls = TTNNLinearIColShardedWRowSharded if distributed else TTNNLinear
        LinearClsOut = TTNNLinearIReplicatedWColSharded if distributed else TTNNLinear
        NormCls = TTNNDistributedRMSNorm if distributed else TTNNRMSNorm

        if distributed:
            # In distributed mode, we need separate Q, K, V projections to handle
            # the case where num_kv_heads < num_devices. This allows:
            # - Q projection to be sharded (num_heads >= num_devices typically)
            # - K/V projections to be replicated (num_kv_heads < num_devices)
            new_attn._use_separate_qkv = True

            # Split the fused query_key_value weight into separate Q, K, V weights
            qkv_weight = torch_attn.query_key_value.weight  # [(num_heads + 2*num_kv_heads) * head_dim, hidden_size]
            q_size = new_attn.num_heads * new_attn.head_dim  # e.g., 16 * 128 = 2048
            kv_size = new_attn.num_kv_heads * new_attn.head_dim  # e.g., 4 * 128 = 512

            q_weight = qkv_weight[:q_size, :]
            k_weight = qkv_weight[q_size : q_size + kv_size, :]
            v_weight = qkv_weight[q_size + kv_size :, :]

            # Handle bias if present
            q_bias = k_bias = v_bias = None
            if torch_attn.query_key_value.bias is not None:
                qkv_bias = torch_attn.query_key_value.bias
                q_bias = qkv_bias[:q_size]
                k_bias = qkv_bias[q_size : q_size + kv_size]
                v_bias = qkv_bias[q_size + kv_size :]

            # Create temporary torch.nn.Linear modules for from_torch
            import torch.nn as nn

            q_linear = nn.Linear(new_attn.hidden_size, q_size, bias=q_bias is not None)
            q_linear.weight.data = q_weight
            if q_bias is not None:
                q_linear.bias.data = q_bias

            k_linear = nn.Linear(new_attn.hidden_size, kv_size, bias=k_bias is not None)
            k_linear.weight.data = k_weight
            if k_bias is not None:
                k_linear.bias.data = k_bias

            v_linear = nn.Linear(new_attn.hidden_size, kv_size, bias=v_bias is not None)
            v_linear.weight.data = v_weight
            if v_bias is not None:
                v_linear.bias.data = v_bias

            # Q projection: sharded input, row-sharded output (num_heads >= num_devices)
            new_attn.q_proj = LinearCls.from_torch(q_linear)

            # K/V projections: replicated input, col-sharded output (num_kv_heads < num_devices)
            # Using TTNNLinearIReplicatedWColSharded allows K/V to work even when
            # num_kv_heads < num_devices because the input is replicated
            new_attn.k_proj = TTNNLinearIReplicatedWColSharded.from_torch(k_linear)
            new_attn.v_proj = TTNNLinearIReplicatedWColSharded.from_torch(v_linear)

            # No fused QKV in distributed mode with separate projections
            new_attn.query_key_value = None
        else:
            # Non-distributed mode: use fused query_key_value projection
            new_attn._use_separate_qkv = False
            new_attn.query_key_value = LinearCls.from_torch(torch_attn.query_key_value)

        # Create dense (output) projection
        new_attn.dense = LinearClsOut.from_torch(torch_attn.dense)

        # Create QK normalization layers if enabled
        # QK norms operate on head_dim (128), not hidden_size (2048), so always use
        # non-distributed version. Distributed norm tries to shard head_dim // 32 = 4
        # chunks across 8 devices, which fails.
        if new_attn.use_qk_norm:
            new_attn.query_layernorm = TTNNRMSNorm.from_torch(torch_attn.query_layernorm)
            new_attn.key_layernorm = TTNNRMSNorm.from_torch(torch_attn.key_layernorm)

        # Create RoPE and SDPA modules
        # When partial_rotary_factor < 1.0, use non-distributed RoPE which handles
        # partial rotary correctly. TTNNDistributedRotaryPositionEmbedding's underlying
        # rotary_embedding_llama kernel requires cos.shape[-1] == head_dim.
        # This follows the same pattern as TTNNQwen3FullAttention.
        uses_partial_rotary = new_attn.partial_rotary_factor < 1.0
        if uses_partial_rotary:
            new_attn.rope = TTNNRotaryPositionEmbedding()
        else:
            new_attn.rope = TTNNDistributedRotaryPositionEmbedding() if distributed else TTNNRotaryPositionEmbedding()
        new_attn.sdpa = TTNNSDPAAttention()
        new_attn.core_grid = ttnn.CoreGrid(y=8, x=8)

        return new_attn

    def preprocess_weights_impl(self):
        """Preprocess weights for TTNN operations.

        Note: Base class handles calling preprocess_weights() on all child modules.
        """
        super().preprocess_weights_impl()

    def move_weights_to_device_impl(self):
        """Move weights to device and initialize SDPA config.

        Note: Base class handles calling move_weights_to_device() on all child modules.
        """
        super().move_weights_to_device_impl()

        # Initialize SDPA config when device is available
        if self.sdpa.program_config is None:
            self.sdpa.program_config = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(self.core_grid.x, self.core_grid.y),
                q_chunk_size=256,
                k_chunk_size=256,
                exp_approx_mode=False,
            )
            self.sdpa.decode_program_config = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(self.core_grid.x, self.core_grid.y),
                q_chunk_size=0,
                k_chunk_size=0,
                exp_approx_mode=False,
            )
            self.sdpa.compute_kernel_config = ttnn.init_device_compute_kernel_config(
                self.device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )

    def _split_qkv(self, qkv: ttnn.Tensor, batch_size: int, seq_length: int):
        """Split fused QKV tensor into separate Q, K, V tensors.

        Args:
            qkv: Fused QKV tensor of shape [batch, seq, (num_heads + 2*num_kv_heads) * head_dim]
            batch_size: Batch size
            seq_length: Sequence length

        Returns:
            Tuple of (query_states, key_states, value_states)
        """
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim

        # Split along last dimension
        query_states = ttnn.slice(qkv, (0, 0, 0), (batch_size, seq_length, q_size))
        key_states = ttnn.slice(qkv, (0, 0, q_size), (batch_size, seq_length, q_size + kv_size))
        value_states = ttnn.slice(qkv, (0, 0, q_size + kv_size), (batch_size, seq_length, q_size + 2 * kv_size))

        # Reshape to [batch, seq, num_heads, head_dim]
        query_states = ttnn.reshape(query_states, (batch_size, seq_length, self.num_heads, self.head_dim))
        key_states = ttnn.reshape(key_states, (batch_size, seq_length, self.num_kv_heads, self.head_dim))
        value_states = ttnn.reshape(value_states, (batch_size, seq_length, self.num_kv_heads, self.head_dim))

        # Transpose to [batch, heads, seq, head_dim]
        query_states = ttnn.permute(query_states, (0, 2, 1, 3))
        key_states = ttnn.permute(key_states, (0, 2, 1, 3))
        value_states = ttnn.permute(value_states, (0, 2, 1, 3))

        return query_states, key_states, value_states

    def _apply_qk_norm(self, query_states: ttnn.Tensor, key_states: ttnn.Tensor):
        """Apply QK normalization if enabled.

        Args:
            query_states: Query tensor [batch, heads, seq, head_dim]
            key_states: Key tensor [batch, heads, seq, head_dim]

        Returns:
            Tuple of (normalized_query, normalized_key)
        """
        if not self.use_qk_norm:
            return query_states, key_states

        # Reshape for normalization: [batch, heads, seq, head_dim] -> [batch*heads*seq, head_dim]
        batch_size, num_heads, seq_length, head_dim = query_states.shape
        batch_kv, num_kv_heads, seq_length_k, head_dim_k = key_states.shape

        # Apply normalization
        q_reshaped = ttnn.reshape(query_states, (batch_size * num_heads * seq_length, head_dim))
        k_reshaped = ttnn.reshape(key_states, (batch_kv * num_kv_heads * seq_length_k, head_dim_k))

        q_normed = self.query_layernorm(q_reshaped)
        k_normed = self.key_layernorm(k_reshaped)

        # Unwrap TorchTTNNTensor if needed
        if hasattr(q_normed, "to_ttnn"):
            q_normed = q_normed.to_ttnn
        if hasattr(k_normed, "to_ttnn"):
            k_normed = k_normed.to_ttnn

        # Ensure BFLOAT16 dtype for compatibility with downstream RoPE ops
        if q_normed.dtype != ttnn.bfloat16:
            q_normed = ttnn.typecast(q_normed, ttnn.bfloat16)
        if k_normed.dtype != ttnn.bfloat16:
            k_normed = ttnn.typecast(k_normed, ttnn.bfloat16)

        # Reshape back
        query_states = ttnn.reshape(q_normed, (batch_size, num_heads, seq_length, head_dim))
        key_states = ttnn.reshape(k_normed, (batch_kv, num_kv_heads, seq_length_k, head_dim_k))

        return query_states, key_states

    def _apply_partial_rope(
        self,
        query_states: ttnn.Tensor,
        key_states: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
    ):
        """Apply partial RoPE based on partial_rotary_factor.

        Args:
            query_states: Query tensor [batch, heads, seq, head_dim]
            key_states: Key tensor [batch, heads, seq, head_dim]
            cos: Cosine position embeddings
            sin: Sine position embeddings

        Returns:
            Tuple of (rotated_query, rotated_key)
        """
        # The RoPE module handles partial rotary embedding internally based on cos/sin dimensions
        # cos/sin should already be sized according to partial_rotary_factor
        query_states, key_states = self.rope(query_states, key_states, cos, sin)

        # Handle TorchTTNNTensor wrapping
        if hasattr(query_states, "to_ttnn"):
            query_states = query_states.to_ttnn
        if hasattr(key_states, "to_ttnn"):
            key_states = key_states.to_ttnn

        return query_states, key_states

    def _forward_prefill(
        self,
        hidden_states: ttnn.Tensor,
        position_embeddings: tuple,
        attention_mask: Optional[ttnn.Tensor],
        past_key_values,
        cache_position: Optional[torch.LongTensor],
    ) -> tuple:
        """Forward pass for prefill phase."""
        batch_size, seq_length = hidden_states.shape[0], hidden_states.shape[1]

        # Ensure proper layout
        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        if self._use_separate_qkv:
            # Separate Q, K, V projections path (for distributed mode with num_kv_heads < num_devices)
            # Q projection can use sharded input (TTNNLinearIColShardedWRowSharded)
            query_states = self.q_proj(hidden_states)

            # K/V projections need replicated (all-gathered) input since they use
            # TTNNLinearIReplicatedWColSharded which expects full tensor width
            if self.device.get_num_devices() > 1:
                hidden_states_replicated = ttnn.all_gather(hidden_states, dim=-1, num_links=1)
            else:
                hidden_states_replicated = hidden_states

            key_states = self.k_proj(hidden_states_replicated)
            value_states = self.v_proj(hidden_states_replicated)

            # Deallocate the gathered tensor if we created one
            if self.device.get_num_devices() > 1:
                ttnn.deallocate(hidden_states_replicated)

            # All-gather projection outputs for reshape (distributed mode produces sharded outputs)
            # _maybe_all_gather also handles TorchTTNNTensor unwrapping
            query_states = self._maybe_all_gather(query_states)
            key_states = self._maybe_all_gather(key_states)
            value_states = self._maybe_all_gather(value_states)

            # Reshape to [batch, seq, num_heads, head_dim]
            query_states = ttnn.reshape(query_states, (batch_size, seq_length, self.num_heads, self.head_dim))
            key_states = ttnn.reshape(key_states, (batch_size, seq_length, self.num_kv_heads, self.head_dim))
            value_states = ttnn.reshape(value_states, (batch_size, seq_length, self.num_kv_heads, self.head_dim))

            # Transpose to [batch, heads, seq, head_dim]
            query_states = ttnn.permute(query_states, (0, 2, 1, 3))
            key_states = ttnn.permute(key_states, (0, 2, 1, 3))
            value_states = ttnn.permute(value_states, (0, 2, 1, 3))
        else:
            # Fused QKV path (non-distributed or compatible distributed mode)
            qkv = self.query_key_value(hidden_states)
            if hasattr(qkv, "to_ttnn"):
                qkv = qkv.to_ttnn

            # Split into Q, K, V
            query_states, key_states, value_states = self._split_qkv(qkv, batch_size, seq_length)

        # Apply QK normalization if enabled
        query_states, key_states = self._apply_qk_norm(query_states, key_states)

        # Apply RoPE
        cos, sin = position_embeddings

        # Handle position embeddings - they should be REPLICATED across devices, not sharded
        # The framework default shards inputs, but cos/sin must be identical on all devices
        from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

        def _ensure_replicated_tensor(t, name):
            """Convert tensor to TTNN with proper replication for multi-device.

            The framework default shards inputs, but position embeddings (cos/sin)
            must be identical on all devices, so we need to gather and re-replicate.
            """
            num_devices = self.device.get_num_devices() if hasattr(self.device, "get_num_devices") else 1

            # If it's already an TTNN tensor with wrong sharding, we need to convert back and re-convert
            if isinstance(t, ttnn.Tensor):
                # Check if tensor appears to be sharded (last dim smaller than expected)
                t_shape = list(t.shape)
                # Position embeddings should have shape like [1, 1, seq_len, rotary_dim] or [batch, seq_len, rotary_dim]
                # If last dim is divided by num_devices, it's been sharded
                if num_devices > 1 and t_shape[-1] < 32:  # rotary_dim is typically >= 32
                    # Tensor was sharded, need to convert back and re-convert with replication
                    # Use mesh_composer to gather the tensor first
                    torch_t = ttnn.to_torch(
                        t,
                        mesh_composer=ttnn.ConcatMesh2dToTensor(self.device, self.device.shape, (0, -1)),
                    )
                    mesh_mapper = ttnn.ReplicateTensorToMesh(self.device)
                    return ttnn.from_torch(
                        torch_t.to(torch.bfloat16),
                        device=self.device,
                        layout=ttnn.TILE_LAYOUT,
                        dtype=ttnn.bfloat16,
                        mesh_mapper=mesh_mapper,
                    )
                return t

            # If it's a TorchTTNNTensor, extract the original torch tensor and re-convert
            if isinstance(t, TorchTTNNTensor):
                if t.elem is not None:
                    torch_t = t.elem
                else:
                    torch_t = ttnn.to_torch(t.ttnn_tensor if t.ttnn_tensor is not None else t.to_ttnn)
                mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if num_devices > 1 else None
                return ttnn.from_torch(
                    torch_t.to(torch.bfloat16),
                    device=self.device,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    mesh_mapper=mesh_mapper,
                )

            elif isinstance(t, torch.Tensor):
                mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if num_devices > 1 else None
                return ttnn.from_torch(
                    t.to(torch.bfloat16),
                    device=self.device,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    mesh_mapper=mesh_mapper,
                )
            return t

        cos = _ensure_replicated_tensor(cos, "cos")
        sin = _ensure_replicated_tensor(sin, "sin")

        # Ensure query/key states are BFLOAT16 for RoPE compatibility
        if query_states.dtype != ttnn.bfloat16:
            query_states = ttnn.typecast(query_states, ttnn.bfloat16)
        if key_states.dtype != ttnn.bfloat16:
            key_states = ttnn.typecast(key_states, ttnn.bfloat16)

        query_states, key_states = self._apply_partial_rope(query_states, key_states, cos, sin)

        # Handle KV cache
        use_paged = isinstance(past_key_values, TTNNPagedAttentionKVCache)
        if past_key_values is not None:
            layer_idx = self._fallback_torch_layer.layer_idx

            if use_paged:
                past_key_values.paged_fill_on_device(
                    key_states,
                    value_states,
                    layer_idx=layer_idx,
                    batch_idx=0,
                )
            else:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                torch_tensors = [TorchTTNNTensor(key_states), TorchTTNNTensor(value_states)]
                orig_shapes = [key_states.shape, value_states.shape]

                torch_tensors = [
                    torch_tensor.to_torch[: orig_shape[0], : orig_shape[1], : orig_shape[2], : orig_shape[3]]
                    for orig_shape, torch_tensor in zip(orig_shapes, torch_tensors)
                ]

                key_states, value_states = past_key_values.update(
                    *torch_tensors,
                    layer_idx,
                    cache_kwargs,
                )
                key_states, value_states = [TorchTTNNTensor(key_states), TorchTTNNTensor(value_states)]
                key_states = ttnn.to_device(key_states.to_ttnn, self.device)
                value_states = ttnn.to_device(value_states.to_ttnn, self.device)

        # Apply SDPA
        attn_output = self.sdpa(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            is_causal=self.is_causal,
            transpose_output=True,
        )

        if hasattr(attn_output, "to_ttnn"):
            attn_output = attn_output.to_ttnn

        # Reshape and project output
        attn_output = ttnn.reshape(attn_output, (batch_size, seq_length, self.num_heads * self.head_dim))
        attn_output = self.dense(attn_output)

        # Return format matches HuggingFace: (attn_output, attn_weights, past_key_values)
        return attn_output, None, past_key_values

    def _forward_decode_paged(
        self,
        hidden_states: ttnn.Tensor,
        position_embeddings: tuple,
        attention_mask: Optional[ttnn.Tensor],
        past_key_values: "TTNNPagedAttentionKVCache",
        cache_position: Optional[torch.LongTensor],
    ) -> tuple:
        """Decode path using paged attention with on-device KV cache."""
        batch_size, seq_length = hidden_states.shape[0], hidden_states.shape[1]

        # Ensure proper layout
        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        if self._use_separate_qkv:
            # Separate Q, K, V projections path (for distributed mode with num_kv_heads < num_devices)
            # Q projection can use sharded input (TTNNLinearIColShardedWRowSharded)
            query_states = self.q_proj(hidden_states)

            # K/V projections need replicated (all-gathered) input since they use
            # TTNNLinearIReplicatedWColSharded which expects full tensor width
            if self.device.get_num_devices() > 1:
                hidden_states_replicated = ttnn.all_gather(hidden_states, dim=-1, num_links=1)
            else:
                hidden_states_replicated = hidden_states

            key_states = self.k_proj(hidden_states_replicated)
            value_states = self.v_proj(hidden_states_replicated)

            # Deallocate the gathered tensor if we created one
            if self.device.get_num_devices() > 1:
                ttnn.deallocate(hidden_states_replicated)

            # All-gather projection outputs for reshape (distributed mode produces sharded outputs)
            # _maybe_all_gather also handles TorchTTNNTensor unwrapping
            query_states = self._maybe_all_gather(query_states)
            key_states = self._maybe_all_gather(key_states)
            value_states = self._maybe_all_gather(value_states)

            # Reshape to [batch, seq, num_heads, head_dim]
            query_states = ttnn.reshape(query_states, (batch_size, seq_length, self.num_heads, self.head_dim))
            key_states = ttnn.reshape(key_states, (batch_size, seq_length, self.num_kv_heads, self.head_dim))
            value_states = ttnn.reshape(value_states, (batch_size, seq_length, self.num_kv_heads, self.head_dim))

            # Transpose to [batch, heads, seq, head_dim]
            query_states = ttnn.permute(query_states, (0, 2, 1, 3))
            key_states = ttnn.permute(key_states, (0, 2, 1, 3))
            value_states = ttnn.permute(value_states, (0, 2, 1, 3))
        else:
            # Fused QKV path (non-distributed or compatible distributed mode)
            qkv = self.query_key_value(hidden_states)
            if hasattr(qkv, "to_ttnn"):
                qkv = qkv.to_ttnn

            # Split into Q, K, V
            query_states, key_states, value_states = self._split_qkv(qkv, batch_size, seq_length)

        # Apply QK normalization if enabled
        query_states, key_states = self._apply_qk_norm(query_states, key_states)

        # Apply RoPE
        cos, sin = position_embeddings
        if isinstance(cos, torch.Tensor):
            mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None
            cos = ttnn.from_torch(
                cos.to(torch.bfloat16),
                device=self.device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=mesh_mapper,
            )
        if isinstance(sin, torch.Tensor):
            mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None
            sin = ttnn.from_torch(
                sin.to(torch.bfloat16),
                device=self.device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=mesh_mapper,
            )

        # Ensure query/key states are BFLOAT16 for RoPE compatibility
        if query_states.dtype != ttnn.bfloat16:
            query_states = ttnn.typecast(query_states, ttnn.bfloat16)
        if key_states.dtype != ttnn.bfloat16:
            key_states = ttnn.typecast(key_states, ttnn.bfloat16)

        query_states, key_states = self._apply_partial_rope(query_states, key_states, cos, sin)

        layer_idx = self._fallback_torch_layer.layer_idx

        # Resolve cache position
        if cache_position is None:
            cur_pos = past_key_values.get_seq_length(layer_idx)
            cache_position_tensor = torch.tensor([cur_pos], dtype=torch.int32)
        else:
            cp = cache_position
            if isinstance(cp, TorchTTNNTensor):
                cp = cp.to_torch
            if isinstance(cp, ttnn.Tensor):
                mesh_composer = None
                if hasattr(cp, "device") and cp.device() is not None and cp.device().get_num_devices() > 1:
                    mesh_composer = ttnn.ConcatMeshToTensor(cp.device(), dim=0)
                cp = ttnn.to_torch(cp, mesh_composer=mesh_composer)
            cache_position_tensor = cp.flatten()[:batch_size].to(torch.int32)

        mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None
        cur_pos_tt = ttnn.from_torch(
            cache_position_tensor,
            device=self.device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mesh_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Permute B H S D -> S B H D for paged kernels
        query_states = ttnn.permute(query_states, (2, 0, 1, 3))
        key_states = ttnn.permute(key_states, (2, 0, 1, 3))
        value_states = ttnn.permute(value_states, (2, 0, 1, 3))

        # Multi-device: convert all-gathered topology -> replicated for paged kernels
        if self.device.get_num_devices() > 1:
            query_states = self._to_replicated(query_states)
            key_states = self._to_replicated(key_states)
            value_states = self._to_replicated(value_states)

        # Update paged KV cache
        tile_size = 32
        shard_h = ((self.num_kv_heads + tile_size - 1) // tile_size) * tile_size

        core_grid = ttnn.CoreGrid(y=1, x=batch_size)
        shard_cfg = ttnn.create_sharded_memory_config(
            shape=(shard_h, self.head_dim),
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
        key_states = ttnn.to_memory_config(key_states, shard_cfg)
        value_states = ttnn.to_memory_config(value_states, shard_cfg)

        past_key_values.paged_update_on_device(
            key_states,
            value_states,
            layer_idx=layer_idx,
            current_pos=cur_pos_tt,
        )
        ttnn.deallocate(key_states)
        ttnn.deallocate(value_states)

        # ⚠️ POTENTIAL BUG: This may double-count seq_length since paged_update_on_device
        # already increments _seq_lengths internally
        past_key_values._seq_lengths[layer_idx] += seq_length
        if layer_idx == 0:
            past_key_values._seen_tokens += seq_length

        # Paged SDPA decode
        # Use the same cur_pos_tt for both paged_update_on_device and paged_sdpa_decode
        # This matches the Qwen implementation semantics
        attn_output = past_key_values.paged_sdpa_decode(
            query_states,
            layer_idx,
            current_pos=cur_pos_tt,
            scale=self.scaling,
            program_config=self.sdpa.decode_program_config,  # Use decode config (q_chunk_size=0, k_chunk_size=0)
            compute_kernel_config=self.sdpa.compute_kernel_config,
        )

        # Convert back to [B, S, H*D] for output projection
        attn_output = ttnn.permute(attn_output, (1, 0, 2, 3))  # [B, 1, H, head_dim]
        attn_output = ttnn.reshape(attn_output, (batch_size, seq_length, self.num_heads * self.head_dim))
        attn_output = self.dense(attn_output)

        # Return format matches HuggingFace: (attn_output, attn_weights, past_key_values)
        return attn_output, None, past_key_values

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        position_embeddings: tuple,
        attention_mask: Optional[ttnn.Tensor] = None,
        past_key_values=None,
        cache_position: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple:
        """Forward pass through BailingMoE attention.

        Args:
            hidden_states: Input tensor [batch, seq, hidden_size]
            position_embeddings: Tuple of (cos, sin) for RoPE
            attention_mask: Optional attention mask
            past_key_values: KV cache (TTNNPagedAttentionKVCache or DynamicCache)
            cache_position: Position in cache for decode
            position_ids: Position IDs (unused, for compatibility)
            **kwargs: Additional arguments

        Returns:
            Tuple of (output, None)
        """
        # Handle TorchTTNNTensor input
        if hasattr(hidden_states, "to_ttnn"):
            hidden_states = hidden_states.to_ttnn

        seq_length = hidden_states.shape[1]
        use_paged = isinstance(past_key_values, TTNNPagedAttentionKVCache)

        if use_paged and seq_length == 1:
            return self._forward_decode_paged(
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values,
                cache_position,
            )

        return self._forward_prefill(
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values,
            cache_position,
        )
```

---

## Key Differences Summary

| Aspect | PyTorch (HuggingFace) | TTNN |
|--------|----------------------|------|
| **QKV Projection** | Fused `query_key_value` linear | Separate Q/K/V in distributed mode |
| **QKV Split** | `qkv.split([num_heads, num_kv_heads, num_kv_heads], dim=-2)` | `ttnn.slice()` along last dim |
| **QK Normalization** | Direct call on 4D tensor `[B,H,S,D]` | Reshape to 2D, normalize, reshape back |
| **RoPE** | `apply_rotary_pos_emb()` with partial support | `TTNNRotaryPositionEmbedding` module |
| **GQA Handling** | Explicit `repeat_kv()` to expand K/V | Relies on SDPA kernel internally |
| **is_causal (decode)** | `is_causal=False` when `q_len==1` | Always `is_causal=True` |
| **KV Cache Update** | `past_key_value.update()` returns tensors | `paged_update_on_device()` + manual tracking |
| **Seq Length Tracking** | Cache handles internally | ⚠️ Manual increment (potential double-count bug) |
| **Attention** | `torch.nn.functional.scaled_dot_product_attention` | `ttnn.sdpa()` or `paged_sdpa_decode()` |
| **Output Shape** | `[B, H, S, D]` -> transpose -> `[B, S, H*D]` | Same flow with `ttnn.permute/reshape` |

## Identified Issues

1. **Double Sequence Length Increment (lines 2885-2887)**: After `paged_update_on_device()`, there's a manual increment of `_seq_lengths[layer_idx]`. However, `paged_update_on_device` already increments this internally, potentially causing double-counting.

2. **is_causal Mismatch**: PyTorch sets `is_causal=False` when `q_len==1` (decode), but TTNN always uses `is_causal=True`. This may not cause issues in practice since for single-token decode there's only one position to attend to.

3. **GQA Expansion**: PyTorch explicitly expands K/V with `repeat_kv()` before attention. TTNN relies on the kernel to handle GQA internally, which should be equivalent but worth verifying.
