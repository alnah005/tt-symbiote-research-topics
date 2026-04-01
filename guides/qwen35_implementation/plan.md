# Guide Plan: Qwen3.5 Implementation on TT-Metal / Blackhole P100A

## Audience

This guide targets ML engineers and hardware-aware software engineers who are already comfortable
with transformer model architectures (attention, MLP, RMSNorm, RoPE), PyTorch tensor operations,
and the basics of TT-Metal / TTNN (tensors, device placement, `ttnn.linear`, memory configs).
Readers do not need prior knowledge of linear attention, DeltaNet recurrence, Mixture of Experts,
or Blackhole-specific hardware constraints. The guide builds that knowledge chapter by chapter.


## Chapter List

---

### Chapter 1 — Model Architecture Overview

**Description:** Establishes the two Qwen3.5 model variants (27B dense and 35B-A3B MoE), their
layer composition, hyperparameter tables, and the hybrid DeltaNet + full-attention design before
any implementation details are introduced.

**Files:**

- `ch1_architecture/index.md` — navigation and reading order for Chapter 1
  - Lists the two files in this chapter and the order to read them
  - Notes which hyperparameters are referenced again in later chapters (head_dim, hidden_size, layer_types)
  - Cross-references Chapter 2 (DeltaNet) and Chapter 3 (GatedAttention) for layer-type detail

- `ch1_architecture/model_variants.md` — side-by-side comparison of all supported Qwen3.5 variants
  - Full hyperparameter table: layers, hidden size, DeltaNet V-heads / K-heads, Attn Q/KV heads,
    MLP type, active params, DRAM footprint (bfp4), and the hardware each variant targets
    (27B → P100A dense; 35B-A3B → P100A MoE; 122B-A10B → Galaxy 6U; 397B-A17B → Galaxy 6U+)
  - Explanation of the hybrid layer ratio: 3/4 DeltaNet layers + 1/4 full-attention layers
    (e.g., 27B = 48 DeltaNet + 16 full attention; 35B-A3B = 30 + 10)
  - How layer_types list in model config drives per-layer dispatch in the forward loop
  - Why the 35B-A3B is the recommended entry point: fits P100A, faster than the 27B, beats CPU baselines
  - Comparison with llama.cpp AmpereOne baseline (A3B: 11.7 tok/s TTNN vs 9.05 tok/s CPU Q4_K)

- `ch1_architecture/layer_types_and_hyperparams.md` — detailed breakdown of each layer type
  - DeltaNet layer hyperparameters: linear_num_key_heads, linear_num_value_heads, linear_key_head_dim,
    linear_value_head_dim, linear_conv_kernel_dim, GQA ratio (num_v_heads / num_k_heads)
  - Full-attention layer hyperparameters: n_heads, n_kv_heads, head_dim, partial_rotary_factor,
    rope_theta, norm_eps
  - MoE-specific hyperparameters (35B-A3B only): num_experts (256), num_experts_per_tok (8),
    moe_intermediate_size (512), shared_expert_intermediate_size (512)
  - Vocabulary size (248,320) and why the LM head stays on host
  - Token embedding approach: embedding table kept on host CPU as float32 tensors to avoid
    DRAM pressure and because embedding lookup for a single token is cheap

---

### Chapter 2 — GatedDeltaNet: Linear Attention on Blackhole

**Description:** Deep dive into the GatedDeltaNet module — its projections, conv1d state, the
DeltaNet recurrence equations, why the recurrence must run in float32 on the host, and the
alternative fully-fused device kernel path.

**Files:**

- `ch2_gated_deltanet/index.md` — navigation and reading order for Chapter 2
  - Reading order: recurrence_math → projections_and_conv → host_recurrence → fused_kernel
  - Prerequisites: Chapter 1 (layer types, DeltaNet hyperparameters)
  - Forward references to Chapter 6 (weight precision) for dtype choices in this module

- `ch2_gated_deltanet/recurrence_math.md` — the DeltaNet recurrence equations and their meaning
  - The five-step recurrence at each token step:
    1. `state *= exp(gate)` — exponential decay of the key-value memory matrix
    2. `kv_mem = einsum('hkv,hk->hv', state, k)` — retrieve current value estimate for key k
    3. `delta = (v - kv_mem) * beta` — delta correction weighted by beta gate
    4. `state += einsum('hk,hv->hkv', k, delta)` — rank-1 update to memory matrix
    5. `output = einsum('hkv,hk->hv', state, q)` — read output for query q
  - Gate computation: `g = -A_log.exp() * softplus(a + dt_bias)`, decay = `exp(g)`
  - Beta gate: `beta = sigmoid(b_projection)` — controls how aggressively delta is applied
  - L2 normalization of Q and K before the recurrence (prevents state explosion)
  - GQA expansion: K and Q are projected with fewer heads (num_k_heads) then repeated
    (repeat_interleave by gqa_ratio) to match num_v_heads for the outer-product state update
  - Gated RMSNorm after recurrence: `output_normed * silu(z)` fused post-processing

- `ch2_gated_deltanet/projections_and_conv.md` — device-side weight loading and conv1d state
  - Fused input projection: four weights (in_proj_qkv, in_proj_z, in_proj_b, in_proj_a) are
    concatenated along the output dimension into a single `in_proj_all` matmul to save dispatch
  - Projection split sizes: [conv_dim, value_dim, num_v_heads, num_v_heads] for qkv / z / b / a
  - Conv1d implementation as a circular ring buffer of `conv_kernel_size` (4) device tensors:
    each step overwrites the oldest slot with `ttnn.copy` (in-place, preserves tensor address
    for Metal Trace compatibility), then computes a weighted sum over all slots
  - Conv weight layout: `conv_kernel_size` separate device tensors, one per kernel position,
    each shape [1, 1, B_pad, conv_dim] in bfloat16 DRAM
  - dt_bias, A_log (as -exp), and norm.weight stored as constant device tensors in bfloat16
  - State initialization: `initialize_states` allocates conv ring buffer rows and the recurrent
    state tensor [batch_size, H, head_k_dim, head_v_dim] in float32 on device DRAM
  - `_conv_rows[oldest]` in-place copy pattern and `_oldest` pointer rotation

- `ch2_gated_deltanet/host_recurrence.md` — why recurrence runs on host in float32
  - The Blackhole fp32 circular buffer (CB) constraint: SrcB register is 19-bit TF32; element-wise
    ops with fp32 input CBs hang the device; this prevents implementing the full recurrence
    entirely on-device in fp32 using standard element-wise TTNN ops
  - Quantization error accumulation: running the recurrence in bf16 compounds across 30+ DeltaNet
    layers and produces garbage output; float32 on host is the correct baseline
  - Historical host-recurrence flow (now superseded by fused kernel): `to_torch` → float32 NumPy
    recurrence → `from_torch` for each layer per token, introducing one sync per DeltaNet layer
  - Current design: the fused kernel path (`ttnn.experimental.gated_delta_net`) performs the
    full recurrence on-device in float32 using SFPU binary path and `init_sfpu + copy_tile` for
    fp32 read (workarounds documented in README's "Blackhole fp32 CB Reference" section)
  - The recurrent state tensor is explicitly kept as `ttnn.float32` on device DRAM; the fused
    kernel reads and writes it in-place via `ttnn.copy(result[1], self._dev_state)`

- `ch2_gated_deltanet/fused_kernel.md` — the Metalium gated_delta_net fused kernel
  - Kernel location: `ttnn/cpp/ttnn/operations/experimental/ssm/gated_delta_net/device/`
  - Kernel inputs: conv_out, z_flat, ba_flat, dt_bias, neg_A_exp, state (fp32), norm_w
  - Kernel outputs: (output [1, H, B, D], new_state [batch, H, K, D]) — state updated in-place
    via `ttnn.copy` in the Python wrapper
  - PCC achieved: 0.999997 vs host float32 reference — essentially lossless
  - Why it is not used in production today: `from_torch` for state upload dominates when there
    is no Metal Trace; each call to `ttnn.from_torch` for the state tensor costs ~1–2 ms, making
    the fused path slower than the current host-sync approach without Trace
  - Path to production: Metal Trace eliminates Python dispatch and from_torch overhead; the fused
    kernel becomes the preferred path once Trace is integrated
  - Output reshape: kernel produces [1, H, B, D]; for batch > 1, `ttnn.permute([0,2,1,3])`
    then reshape to [1, 1, -1, value_dim] before the out_proj linear

---

### Chapter 3 — GatedAttention: Full-Attention Layers

**Description:** Covers the GatedAttention module that handles the 1/4 full-attention layers in
Qwen3.5, focusing on the output gate mechanism, Qwen3.5's partial RoPE with corrected frequencies,
and how the standard Attention base class is extended via hooks.

**Files:**

- `ch3_gated_attention/index.md` — navigation and reading order for Chapter 3
  - Reading order: partial_rope → output_gate → forward_flow
  - Prerequisites: Chapter 1 (attention hyperparameters), Chapter 2 (GQA concept)
  - Notes that GatedAttention inherits from the standard Attention class; only the deltas
    from standard attention are documented here

- `ch3_gated_attention/partial_rope.md` — Qwen3.5 partial RoPE implementation
  - Qwen3.5 rotates only the first `rotary_dim = head_dim * partial_rotary_factor = 64` dims
    of each head (head_dim = 256, partial_rotary_factor = 0.25)
  - Why the standard `rotary_embedding_llama` cannot be used:
    1. It computes frequencies using head_dim=256 instead of the correct rotary_dim=64
    2. Its transformation matrix pairs dims 128 apart (half of head_dim) instead of 32 apart
       (half of rotary_dim)
    3. cos/sin are in Meta interleaved format but Qwen3.5 Q/K weights are in HF format
  - Solution: corrected cos/sin matrices precomputed using `1/theta^(2i/rotary_dim)` instead
    of `1/theta^(2i/head_dim)`, then patched into the existing rope_setup matrices
  - How the patch works in demo_a3b.py: after constructing HfRotarySetup, the cos/sin matrices
    are pulled to host, corrected frequencies are written into the rotary slice positions, and
    non-rotary positions are set to cos=1.0/sin=0.0 so standard device RoPE leaves them unchanged
  - 27B variant uses RotarySetup (Meta interleaved); 35B-A3B uses HfRotarySetup — difference
    in how cos/sin positions are addressed when patching (half_rotary vs half_head offsets)
  - Historical host-based custom_rope_fn approach (in _setup_partial_rope): roundtrip of ~14 KB
    (Q and K for one token), negligible latency, but adds 2 host-device syncs per attention layer;
    superseded by the corrected-matrix approach which stays fully on device

- `ch3_gated_attention/output_gate.md` — the output gate mechanism
  - Gate weight: `q_proj_gate.weight` of shape (hidden_size, n_heads * head_dim); stored as
    bfloat16 on device DRAM; created by splitting the 2x q_proj in qwen35_utils.py
  - Gate computation: `sigmoid(x @ gate_weight)` where x is the layer input (not Q projection)
  - Why x must be copied before the parent Attention.forward call: the parent deallocates x
    after the QKV matmul; `ttnn.add(x, 0)` creates a new buffer that is not aliased
  - `pre_wo_hook` hook mechanism: GatedAttention registers `_apply_gate` as `self.pre_wo_hook`;
    the standard Attention.forward calls this hook on the post-softmax attention output before
    the WO projection, allowing the gate to be applied without changing the parent's forward logic
  - Memory config handling: attn_output moved to DRAM_MEMORY_CONFIG before multiply to ensure
    compatible configs between gate and attention output tensors; gate and _gate_input freed
    explicitly after use

- `ch3_gated_attention/forward_flow.md` — full forward pass through GatedAttention
  - Entry: `GatedAttention.forward` saves `_gate_input = ttnn.add(x, 0)` then delegates to
    `Attention.forward` which handles QKV projections, KV cache, RoPE, softmax, and WO
  - GQA: n_kv_heads=4 (27B) or n_kv_heads=2 (35B-A3B); KV heads expanded via repeat_interleave
    inside the base Attention class; ratio = n_heads / n_kv_heads
  - Q/K per-head RMSNorm with zero-centered weights: weight initialized to zeros; applied as
    `x * (1 + w)` (add_unit_offset pattern); separate q_norm.weight and k_norm.weight tensors
  - KV cache management: 10 attention layers (A3B) or 16 (27B); paged KV cache supported
    via PagedAttentionConfig; cache stores bf16 tensors
  - Hook invocation sequence within Attention.forward: QKV → RoPE → softmax attention →
    pre_wo_hook (gate application) → WO projection → return

---

### Chapter 4 — Decoder Block and Uniform Dispatch

**Description:** Explains how DeltaNetDecoderBlock provides a uniform forward signature that
allows the model loop to dispatch identically to both DeltaNet and full-attention layers,
and how the mlp_class parameter enables MoE substitution at construction time.

**Files:**

- `ch4_decoder/index.md` — navigation and reading order for Chapter 4
  - Reading order: block_structure → forward_signature → mlp_dispatch
  - Prerequisites: Chapters 2 and 3 (the two attention variants); Chapter 5 (MoE) can be read
    after this chapter since mlp_class just accepts any callable

- `ch4_decoder/block_structure.md` — DeltaNetDecoderBlock construction
  - The two attention branches controlled by `attention_class` parameter:
    - None → instantiates GatedDeltaNet and calls `initialize_states(batch_size)`
    - GatedAttention class → instantiates GatedAttention with transformation_mats=None and
      configuration=args (no prefetcher for Qwen3.5 use case)
  - MLP branch: `mlp_class or MLP`; for 27B dense layers, standard MLP is used with bfp8 weights;
    for A3B all layers, Qwen35MoE is passed as mlp_class from the demo build loop
  - Separate `mlp_dtype` and `mlp_weight_cache_path` parameters allow different dtype and cache
    paths for the MLP vs the attention weights
  - Two DistributedNorm wrappers (attention_norm and ff_norm) with RMSNorm inside, loaded from
    attention_norm.weight and ffn_norm.weight keys in the state dict
  - How the 35B-A3B build loop in demo_a3b.py constructs all 40 layers uniformly using
    DeltaNetDecoderBlock with attention_class=None (DeltaNet) or GatedAttention, always with
    mlp_class=Qwen35MoE
  - How the 27B build loop in demo.py constructs DeltaNetDecoderBlock for linear_attention layers
    and TransformerBlock (with GatedAttention) for full_attention layers

- `ch4_decoder/forward_signature.md` — uniform forward signature and residual path
  - Signature: `forward(x, current_pos, rot_mats_global, rot_mats_local, user_id, mode,
    page_table, chunk_page_table, chunk_start_idx, kv_cache, batch_size)`
  - DeltaNet layers silently ignore current_pos, rot_mats, page_table, and kv_cache; the
    dispatch logic inside forward checks for `hasattr(self.attention, 'initialize_states')` to
    determine which calling convention to use
  - Residual layout: input moved to `get_residual_mem_config(mode, prefetcher)` DRAM-sharded
    config; the skip connection add uses this config throughout
  - Memory config note for MLP: DRAM interleaved input with program_config=None (auto matmul
    selection) to avoid L1 CB clash with hidden_dim=17408 (27B) on Blackhole
  - Activation dtype: `decoders_optimizations.get_tensor_dtype(layer_num, ACTIVATION)` applied
    to the final residual add output; allows per-layer dtype tuning

- `ch4_decoder/mlp_dispatch.md` — how mlp_class enables dense vs MoE selection
  - The factory pattern: `mlp_cls = mlp_class or MLP`; both MLP and Qwen35MoE share the same
    constructor signature (mesh_device, tt_ccl, args, state_dict, weight_cache_path, layer_num,
    dtype, model_config, prefetcher)
  - For the 27B dense model: MLP uses standard SwiGLU with w1/w2/w3 weights in bfp8; no
    routing or expert selection
  - For the 35B-A3B MoE model: Qwen35MoE is substituted; the DeltaNetDecoderBlock does not
    know or care which MLP class it holds; forward calls `self.feed_forward.forward(hidden, mode)`
    identically in both cases
  - State dict prefix isolation: MoE keys use `feed_forward.*` after conversion in qwen35_utils;
    the mlp_weight_cache_path parameter allows caching expert weights separately from attention

---

### Chapter 5 — Mixture of Experts (35B-A3B)

**Description:** Documents the Qwen35MoE implementation: router matmul on device, topk/softmax
on host, shared expert overlap, fused SwiGLU gate+up projection, bfp4 expert weights, and the
DRAM budget implications.

**Files:**

- `ch5_moe/index.md` — navigation and reading order for Chapter 5
  - Reading order: architecture_overview → router_and_routing → expert_computation → dram_budget
  - Prerequisites: Chapter 4 (MLP dispatch pattern); Chapter 6 (weight precision) should be
    read after this chapter to understand why bfp4 is chosen

- `ch5_moe/architecture_overview.md` — MoE structure and design choices
  - 256 routed experts + 1 shared expert per layer; 8 experts selected per token
  - Expert architecture: SwiGLU with intermediate size 512; input hidden size 2048
  - Shared expert: same SwiGLU structure but always active; gated by a learned scalar sigmoid
    applied to a single linear(x, shared_expert_gate.weight) output of shape [1,1,B,1]
  - Why host topk: reading 256 router logit floats per layer is cheap (~1 KB); doing topk on
    device would require a sync anyway to retrieve the expert IDs for dispatching; keeping topk
    on host avoids a custom device kernel while keeping routing flexible
  - Batched decode assumption: for same-prompt batch (A3B demo default), all batch rows route
    identically so row 0 representative routing is valid; per-row routing for different prompts
    is identified as future work in the code comments
  - Forward ordering designed to overlap shared expert compute with routing sync: device queues
    shared expert matmuls and router matmul before the first `ttnn.to_torch` (the sync); shared
    expert results are ready on device while the CPU executes topk/softmax

- `ch5_moe/router_and_routing.md` — router implementation and host routing flow
  - Router weight: [1, 1, hidden, num_experts] in bfloat16 on device DRAM (256 floats output,
    small enough for bf16 without precision concern)
  - Router forward: `router_logits = ttnn.linear(x, router_weight_tt, L1)` — stays in L1 to
    avoid DRAM roundtrip on a small tensor
  - Sync point: `ttnn.to_torch(router_logits).float()[0,0,0,:num_experts]` — extracts 256 floats
    from device to CPU; this is the one mandatory sync per MoE layer per token
  - Host routing: `torch.topk(logits_cpu, 8)` then `F.softmax(topk_vals, dim=-1)` for routing
    weights; produces `topk_ids` (expert indices) and `weights` (normalized softmax scores)
  - Shared expert gate: computed on device using shared_expert_gate.weight [1,1,hidden,1];
    sigmoid applied; shared_out multiplied by gate before accumulation

- `ch5_moe/expert_computation.md` — per-expert matmuls and accumulation
  - Expert weight layout: fused gate+up projection as a single [1,1,hidden,2*intermediate] tensor
    per expert in bfp4; down projection [1,1,intermediate,hidden] per expert in bfp4
  - Loading pattern: 256 expert_gate_up + 256 expert_down tensors stored in Python lists;
    indexed by topk_ids at forward time
  - Fused gate+up: one matmul (saves a dispatch vs two separate matmuls); output split with
    `ttnn.split(gate_up, moe_intermediate_size, dim=3)` → gate_out and up_out
  - SwiGLU activation: `ttnn.mul(gate_out, up_out, input_tensor_a_activations=[SILU])` fuses
    the silu(gate_out) * up_out into a single TTNN operation
  - Accumulation loop: for each of the 8 selected experts, compute expert_out, scale by routing
    weight (ttnn.multiply with scalar if w != 1.0), and accumulate with ttnn.add into result
    (initialized to shared_out so the shared expert is included for free)
  - Memory config: all intermediate tensors in L1_MEMORY_CONFIG (small decode tensors fit in L1,
    avoiding DRAM roundtrips for the 8-expert loop)

- `ch5_moe/dram_budget.md` — DRAM usage breakdown and bfp4 rationale
  - A3B total: ~15.7 GB of 28 GB Blackhole DRAM
    - Expert weights: 256 * (gate_up + down) * 40 layers at bfp4 = 12.8 GB
    - Shared expert weights at bfp8 = 0.8 GB
    - DeltaNet projections at bfp8 = 1.2 GB
    - Attention QKV + WO + gate at bf16 = 0.5 GB
    - Router + shared gate at bf16 = 0.1 GB
    - KV cache (10 layers) at bf16 = 0.3 GB
  - 27B total: ~25 GB of 28 GB
    - MLP w1/w2/w3 at bfp8 = 17.1 GB (dominant)
    - DeltaNet projections at bfp8 = 5.4 GB
    - Attention weights at bf16 = 2.2 GB
  - Why bfp4 for routed expert weights: 256 experts per layer across 40 layers at bfp8 would
    require ~25.6 GB for experts alone, exceeding DRAM; bfp4 halves this to 12.8 GB, leaving
    headroom for non-expert weights and activations
  - Why shared experts stay at bfp8: always-active path has higher impact on output quality;
    shared expert is only 40 * 3 weights = 120 matmul weight tensors, negligible DRAM cost

---

### Chapter 6 — Weight Precision, DRAM Layout, and Weight Conversion

**Description:** Covers how Qwen3.5 HuggingFace checkpoints are converted to the internal meta
format used by the TTNN model, including the MoE key protection mechanism, the q_proj gate
extraction, and the dtype choices for each weight category.

**Files:**

- `ch6_weights/index.md` — navigation and reading order for Chapter 6
  - Reading order: dtype_choices → hf_to_meta_conversion → moe_key_protection
  - Prerequisites: Chapter 4 (weight key prefixes), Chapter 5 (MoE key structure)
  - This chapter is primarily reference material; readers can proceed to Chapter 7 without it

- `ch6_weights/dtype_choices.md` — per-weight-category dtype selection and rationale
  - bfp4 (bfloat4_b): routed expert gate_up and down weights; DRAM-critical path; minor quality
    impact because 8-expert averaging smooths individual expert quantization noise
  - bfp8 (bfloat8_b): DeltaNet projections (in_proj_all, out_proj), MLP weights for 27B dense,
    shared expert weights; good balance of quality and DRAM footprint
  - bf16: attention QKV + WO + gate weights, router weight, shared_expert_gate weight, RMSNorm
    weights, KV cache; used where tensors are small or where precision matters most
  - float32: recurrent state tensor (`_dev_state`); must be fp32 to avoid compound quantization
    error across 30+ DeltaNet layers; stored as `ttnn.float32` in DRAM
  - Compute kernel config for projections: `WormholeComputeKernelConfig(HiFi2, fp32_dest_acc=False,
    packer_l1_acc=True)` — chosen for balance of throughput and precision on Blackhole

- `ch6_weights/hf_to_meta_conversion.md` — qwen35_utils.py conversion pipeline
  - Entry point: `convert_hf_to_meta_qwen35(state_dict, head_dim, n_heads, n_kv_heads)`
  - Step 1 — MoE key extraction: `_is_moe_key` matches any key containing `mlp.experts`,
    `mlp.gate.`, or `mlp.shared_expert`; those keys are popped before any transforms to
    protect them from renaming patterns in `split_hf_keys` and `map_hf_to_meta_keys`
  - Step 2 — split_hf_keys: splits fused keys like gate_up_proj or qkv_proj for non-MoE
    weights; DeltaNet linear_attn keys pass through unchanged (no standard pattern match)
  - Step 3 — q_proj gate extraction for full_attention layers:
    - Qwen3.5 q_proj.weight is [n_heads * head_dim * 2, hidden_size]; interleaved per-head
      layout: [Q_h0(hd), G_h0(hd), Q_h1(hd), G_h1(hd), ...]
    - Reshape to [n_heads, head_dim*2, hidden_size]; slice [:, :head_dim, :] → q_proj.weight;
      slice [:, head_dim:, :] → q_proj_gate.weight
    - Same interleaved split applied to q_proj.bias if present
    - No `reverse_permute` applied: Qwen3.5 uses HF-style RoPE with partial_rotary_factor,
      so weights are already in the correct format
    - k_proj, q_norm, k_norm weights pass through unchanged
    - linear_attn keys passed through entirely (no meta key mapping needed)
  - Step 4 — map_hf_to_meta_keys: renames self_attn → attention, etc.; linear_attn keys pass
    through because they don't match any replacement pattern
  - Step 5 — MoE key re-insertion with only `mlp → feed_forward` rename; all expert weight
    tensor shapes preserved exactly (3D packed tensors remain 3D)

- `ch6_weights/moe_key_protection.md` — why MoE keys need special handling
  - Expert weights are packed 3D tensors: gate_up_proj [256, 2*intermediate, hidden] and
    down_proj [256, hidden, intermediate]; these would be incorrectly split by split_hf_keys
    if not extracted first (split_hf_keys looks for gate_proj + up_proj patterns and would
    attempt to slice dimension 0, corrupting the 256-expert batch dimension)
  - `map_hf_to_meta_keys` would rename `gate_proj` inside expert paths to `w1`, breaking the
    key lookup in Qwen35MoE.__init__ which expects `experts.gate_up_proj`
  - The `_is_moe_key` predicate and pop-protect-reinsert pattern ensures zero transforms are
    applied to expert weights; only the top-level `mlp → feed_forward` prefix rename is applied
    on reinsertion to match the model's `feed_forward.*` key namespace

---

### Chapter 7 — Performance Analysis and Bottlenecks

**Description:** Quantitative breakdown of where time is spent per token on A3B (86 ms/token),
the sync overhead budget, Python dispatch cost, theoretical device compute limit, and the
concrete bottlenecks that prevent reaching the ~172 tok/s theoretical maximum.

**Files:**

- `ch7_performance/index.md` — navigation and reading order for Chapter 7
  - Reading order: latency_breakdown → sync_overhead → bottleneck_analysis
  - Prerequisites: all prior chapters (must understand all layer types to interpret the breakdown)
  - Forward reference to Chapter 8 (Optimization Roadmap) for solutions to each bottleneck

- `ch7_performance/latency_breakdown.md` — per-component timing on A3B (86 ms/token)
  - DeltaNet (30 layers): 54 ms, 30 syncs — host recurrence dominates; 1 sync per layer
  - Full attention (10 layers): 18 ms, 10+40 syncs — RoPE setup (HfRotarySetup) contributes
    extra syncs; 10 syncs for layer-end sync + up to 40 additional for rot_mats computation
  - Norm + LM head: 14 ms, 1 sync — host embedding (248K vocab too large for device);
    `ttnn.synchronize_device` called once after LM head matmul and logits retrieval
  - Total: 86 ms, ~70 syncs → 11.7 tok/s (A3B with batch=32) / 6.28 tok/s (27B)
  - Timing methodology: `time.perf_counter` around the full layer loop + norm + LM head +
    `ttnn.synchronize_device`; step 0 = compile, step 1 = program cache warmup, step 2+ = steady
  - Per-user vs throughput: demo_a3b.py reports `batch_size / avg_dt` as total tok/s and
    `1.0 / avg_dt` as per-user tok/s

- `ch7_performance/sync_overhead.md` — sync cost decomposition
  - Each `ttnn.to_torch` or `ttnn.synchronize_device` forces a full device flush; ~0.5 ms average
    cost per sync on P100A → 70 syncs * ~0.5 ms = ~35 ms total sync overhead
  - DeltaNet sync source: previously from `to_torch`/`from_torch` for host recurrence; now from
    the fused kernel's `ttnn.copy` for state update (still one sync-equivalent per layer)
  - MoE sync: `ttnn.to_torch(router_logits)` once per MoE layer — 30 syncs for A3B DeltaNet
    layers + 10 for attention layers; router logit sync is 256 floats (~1 KB)
  - LM head sync: `ttnn.to_torch(logits_tt)` after final matmul — 1 sync for 248K logits
  - Python dispatch overhead: ~26 ms — each TTNN op dispatches a command to the device command
    queue from Python; with ~70 ops per layer * 40 layers, dispatch latency accumulates
  - Device compute: ~20 ms — actual hardware execution time for all matmuls, norms, and activations

- `ch7_performance/bottleneck_analysis.md` — root-cause analysis of each bottleneck
  - Bottleneck 1: Python dispatch (~26 ms) — TTNN ops are dispatched from Python one at a time;
    Metal Trace records and replays the device command sequence without Python involvement,
    eliminating this overhead entirely for the traced region
  - Bottleneck 2: from_torch overhead for state updates — uploading the recurrent state after
    each DeltaNet step or writing embeddings from host; without Metal Trace, each from_torch
    must synchronize to complete the transfer before the next op can proceed
  - Bottleneck 3: DeltaNet host recurrence sync (legacy) — one `to_torch` per layer per token
    for the pre-fused-kernel path; still present as the state copy sync
  - Bottleneck 4: Host embedding + CPU LM head — 248K vocab size makes device embedding lookup
    impractical; LM head matmul [B, 5120/2048] x [5120/2048, 248K] is feasible on device but
    the logits must come to host for argmax
  - Bottleneck 5: MoE expert routing sync — 256 router logits to host per MoE layer; small data
    but unavoidable without device-side topk and expert dispatch (future kernel work)
  - Theoretical limit: 172 tok/s based on DRAM bandwidth (15.7 GB weights / DRAM BW); current
    efficiency is 6.3% (11.7 / 172)

---

### Chapter 8 — Optimization Roadmap and Testing

**Description:** Documents the three main optimization opportunities (Metal Trace, multi-CQ overlap,
per-row MoE routing), then covers the full testing infrastructure: reference scripts without model
download, PCC tests for each module, fused kernel tests, and how to run and extend the test suite.

**Files:**

- `ch8_optimization_and_testing/index.md` — navigation and reading order for Chapter 8
  - Reading order: optimization_roadmap → testing_infrastructure → running_tests
  - Prerequisites: Chapter 7 (bottleneck analysis) motivates the optimization roadmap;
    Chapters 2–5 (module implementations) needed to understand what each test validates

- `ch8_optimization_and_testing/optimization_roadmap.md` — planned optimizations and current status
  - Metal Trace: records all device ops during a "trace capture" step; subsequent tokens replay
    the trace without Python dispatch or from_torch overhead; eliminates ~26 ms dispatch + state
    upload cost; prerequisite for enabling the fused DeltaNet kernel in production; blocked
    on ensuring all tensor addresses are stable (conv ring buffer uses ttnn.copy in-place writes
    precisely to preserve addresses for trace compatibility)
  - Multi-CQ overlap: use a second command queue (CQ1) to issue `from_torch` writes (e.g., next
    token embedding) while CQ0 executes the current token's compute; decouples memory transfer
    latency from device compute latency; particularly valuable for embedding upload and state writes
  - Per-row MoE routing: current implementation reads row 0 of router_logits as representative
    for all batch rows (valid only when all rows have the same prompt); supporting different prompts
    in a batch requires per-row topk and token grouping by selected expert (group-by-expert
    dispatch pattern); identified in code comments as "future work"
  - Fused DeltaNet kernel enabling path: Metal Trace → from_torch overhead goes to zero →
    fused kernel (PCC 0.999997) becomes the preferred path; expected to bring DeltaNet contribution
    from 54 ms to near-device-compute-bound

- `ch8_optimization_and_testing/testing_infrastructure.md` — test files, their purpose, and structure
  - Reference scripts (no model download needed, no device required for pure-Python checks):
    - `reference/test_deltanet_pcc.py`: validates a single DeltaNet layer end-to-end against HF
      reference implementation (`torch_recurrent_gated_delta_rule`); loads raw safetensors;
      includes the full HF reference functions copied inline to avoid import issues; checks output
      norm and cosine similarity; requires HF_MODEL env var set to local checkpoint dir
    - `reference/test_deltanet_multi.py`: 20 sequential tokens through one DeltaNet layer;
      tracks per-step PCC and state norm to detect state divergence over time; useful for
      validating that the circular conv buffer and recurrent state remain in sync with the reference
    - `reference/test_attention_pcc.py`: validates a single GatedAttention layer; auto-detects
      the first full_attention layer index from config.json layer_types; compares TT output vs HF
      reference including partial RoPE, per-head RMSNorm, GQA, and output gate
  - PCC test suite (requires model download and device):
    - `tests/test_pcc.py` (27B): TestDeltaNetPCC and TestGatedAttentionPCC classes; each test
      loads one layer's safetensors shard, runs HF reference forward, then runs TTNN forward with
      MinimalArgs (no full ModelArgs needed); PCC threshold 0.99; uses `load_layer_weights` to
      load only the needed shard files from the safetensors index
    - `tests/test_a3b_pcc.py` (A3B): TestDeltaNetPCC (single + multi-step), TestMoEPCC, and
      TestFusedKernelPCC classes; TestFusedKernelPCC does not need model download — it constructs
      synthetic random tensors matching kernel interface and checks kernel output vs Python
      reference; PCC thresholds: 0.99 for layer tests, 0.998 for fused kernel output,
      0.999 for fused kernel state
  - Weight conversion utilities used in tests: `convert_hf_to_meta_qwen35` imported from
    `models.tt_transformers.tt.load_checkpoints`; applied to the per-layer state dict before
    constructing GatedDeltaNet or GatedAttention with MinimalArgs
  - PCC metric: Pearson Correlation Coefficient computed as cosine similarity of mean-centered
    vectors; implemented identically in both test files; cosine_similarity used in reference
    scripts; PCC utility function used in pytest test classes

- `ch8_optimization_and_testing/running_tests.md` — how to run each test and what to expect
  - Environment setup: `HF_MODEL` env var must point to a downloaded HuggingFace checkpoint dir
    (or a HF hub ID for auto-download via snapshot_download); device_id=0 assumed throughout
  - Quick test (no model download): `pytest models/demos/qwen35/tests/test_a3b_pcc.py::TestFusedKernelPCC -v -s`
    validates the fused Metalium kernel with synthetic data; no checkpoint needed
  - Full A3B PCC: `HF_MODEL=Qwen/Qwen3.5-35B-A3B pytest models/demos/qwen35/tests/test_a3b_pcc.py -v -s`
    runs all three test classes; requires model download (~17 GB)
  - Full 27B PCC: `HF_MODEL=Qwen/Qwen3.5-27B pytest models/demos/qwen35/tests/test_pcc.py -v -s`
  - Reference scripts (run directly, not via pytest): `HF_MODEL=/path python reference/test_deltanet_pcc.py`
    etc.; these print per-step statistics to stdout; useful for debugging divergence
  - Demo invocations: `HF_MODEL=Qwen/Qwen3.5-35B-A3B python models/demos/qwen35/demo/demo_a3b.py
    --prompt "..." --max_tokens 80 --batch_size 32`; step 0 is compile, step 1 is warmup,
    step 2+ are steady-state timing measurements
  - Expected PCC values: DeltaNet single-step ≥ 0.99; DeltaNet multi-step (10 steps) min ≥ 0.95
    (state accumulation degrades slightly); GatedAttention ≥ 0.99; MoE ≥ 0.99; fused kernel
    output ≥ 0.998, state ≥ 0.999


## Conventions

- **File prefixes:** source files are referred to by their path relative to `tt-metal_p100_qwen35/`;
  implementation files under `models/tt_transformers/tt/` and demo files under `models/demos/qwen35/`
- **Tensor shape notation:** dimensions are listed as `[batch, heads, seq, dim]` or abbreviated
  to `[B, H, S, D]`; the tile-padded batch dimension is denoted `B_pad` (always 32 for P100A decode)
- **dtype abbreviations:** bfp4 = `ttnn.bfloat4_b`; bfp8 = `ttnn.bfloat8_b`; bf16 = `ttnn.bfloat16`;
  fp32 = `ttnn.float32` or Python `torch.float32`
- **TTNN memory config abbreviations:** DRAM = `ttnn.DRAM_MEMORY_CONFIG`; L1 = `ttnn.L1_MEMORY_CONFIG`
- **PCC:** Pearson Correlation Coefficient; values close to 1.0 indicate high numerical agreement;
  threshold for acceptable match is 0.99 unless otherwise specified
- **Host vs device:** "host" means CPU/Python; "device" means Blackhole P100A via TTNN; "sync"
  means a host-device synchronization point (typically `ttnn.to_torch` or `ttnn.synchronize_device`)
- **Layer numbering:** layers are 0-indexed; in A3B, layers 0–29 are DeltaNet (linear_attention)
  and layers 30–39 are full-attention; determined by `args.layer_types[i]`
- **Code references:** inline code uses `backtick` notation; file references use absolute paths
  anchored to the `tt-metal_p100_qwen35/` tree root


## Cross-Chapter Dependencies

| Chapter | Depends on |
|---------|------------|
| Ch 1 — Architecture Overview | (none — entry point) |
| Ch 2 — GatedDeltaNet | Ch 1 (DeltaNet hyperparameters, layer_types) |
| Ch 3 — GatedAttention | Ch 1 (attention hyperparameters, partial_rotary_factor); Ch 2 (GQA concept) |
| Ch 4 — Decoder Block | Ch 2 (GatedDeltaNet interface); Ch 3 (GatedAttention interface) |
| Ch 5 — MoE | Ch 4 (mlp_class dispatch pattern); Ch 1 (A3B layer counts and hidden size) |
| Ch 6 — Weight Precision & Conversion | Ch 4 (key prefixes); Ch 5 (MoE key structure and 3D tensor layout) |
| Ch 7 — Performance Analysis | Ch 2 (recurrence syncs); Ch 3 (RoPE syncs); Ch 5 (routing syncs); Ch 6 (DRAM budget) |
| Ch 8 — Optimization & Testing | Ch 7 (bottleneck motivation for roadmap); Ch 2–5 (module internals for test understanding) |

Key concept flows that span multiple chapters:
- The float32 recurrence constraint (Ch 2, Blackhole fp32 CB) → host sync bottleneck (Ch 7) → Metal Trace solution (Ch 8)
- The GQA ratio (Ch 1 hyperparameters) → K/V repeat_interleave in DeltaNet (Ch 2) and in GatedAttention (Ch 3)
- The bfp4 expert weight choice (Ch 6 dtype) → DRAM budget (Ch 5) → 15.7 GB total (Ch 7 breakdown)
- The q_proj interleaved gate layout (Ch 6 conversion) → gate_weight tensor in GatedAttention (Ch 3)
- The fused gate+up SwiGLU (Ch 5 expert computation) → one dispatch saved per expert → total dispatch cost (Ch 7)
