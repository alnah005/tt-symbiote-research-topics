# Formulating Batched Matmul for MoE

This file explains how to transform router output into the tensor layout required for a batched expert FFN matmul in TTNN. It covers the gather-and-pad step, the expert capacity formula, the resulting activation and weight tensor shapes, and how the batch dimension drives kernel dispatch. A worked example traces through a complete decode-regime forward pass.

---

## 1. From Router Output to Batched Tensor

### 1.1 The Dispatch-Combine Pattern

A MoE layer routes each input token to $k$ experts (here $k=8$ for Qwen3.5-35B). After routing, the computation for one expert is conceptually:

```
output_tokens_for_expert_e = FFN_e(tokens_assigned_to_expert_e)
```

The naive implementation loops over all 256 experts in Python, calling `ttnn.matmul` once per expert. This has two problems:

1. **256 kernel dispatches per MoE layer per forward step.** Each dispatch carries overhead (argument validation, NoC setup, kernel launch). At decode speed, dispatch overhead is not negligible.
2. **The grid idles between calls.** While expert $e$ is computing, all cores not assigned to it are idle. Utilization is at most $1/E$ of peak in the worst case.

The batched matmul approach eliminates both problems by expressing all 256 expert computations as a single TTNN op over a leading batch dimension of size $E=256$.

### 1.2 Gathering Tokens by Expert

After the router produces per-token expert indices (shape `[B×S, k]`) and per-token expert scores (shape `[B×S, k]`), the first step is to collect, for each expert $e$, the set of tokens assigned to it.

The result of this gather is a dense activation tensor of shape `[E, C, H]` where:

- $E = 256$ — one slot per expert
- $C$ — expert capacity (see Section 2)
- $H = 7168$ — `d_model`, the hidden dimension

Each `[C, H]` slice along the first dimension holds the tokens that have been routed to expert $e$, zero-padded to capacity $C$.

```python
import torch
import ttnn

# router_indices: [total_tokens, k] — which k experts each token is sent to
# router_scores:  [total_tokens, k] — softmax weights for combining expert outputs
# hidden_states:  [total_tokens, H] — input activations, H=7168

total_tokens = B * S   # B sequences, S tokens each
E = 256
k = 8
H = 7168
C = compute_capacity(k, B, S, E)  # see Section 2

# Build expert_token_counts and dispatch_indices on CPU, then transfer
# expert_token_buffer: [E, C, H] — zero-initialized
expert_token_buffer = torch.zeros(E, C, H, dtype=torch.bfloat16)

# For each (token, expert_slot) pair, write token into the expert's buffer
for token_idx in range(total_tokens):
    for slot in range(k):
        expert_id = router_indices[token_idx, slot]
        capacity_slot = token_to_capacity_slot[token_idx, slot]  # pre-computed
        if capacity_slot < C:
            expert_token_buffer[expert_id, capacity_slot, :] = hidden_states[token_idx, :]
        # tokens exceeding capacity are dropped (a routing loss term penalizes this)

# Transfer to device in tile layout
expert_tokens_ttnn = ttnn.from_torch(
    expert_token_buffer,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
# Shape on device: [E, C, H] = [256, C, 7168]
```

> **Warning:** The gather loop above is illustrative. In a production TTNN implementation, this gather must be implemented as a device op or a pre-compiled kernel — not a Python loop over tokens — to avoid host-side bottlenecks. The host-side cost of constructing the `[E, C, H]` buffer grows linearly with $B \times S$ and becomes significant at large sequence lengths. See `performance_profile_batched.md` for a latency breakdown.

### 1.3 The Role of the Batch Dimension in Kernel Dispatch

When `ttnn.matmul` receives a 3D activation tensor of shape `[E, C, H]` and a 3D weight tensor of shape `[E, H, D]`, it interprets the leading $E$ dimension as a batch dimension and dispatches one matmul per batch slice. The output is `[E, C, D]`.

Internally, TTNN's kernel dispatch unrolls the batch dimension: the program config's `per_core_M` and `per_core_N` parameters are specified for the per-expert `[C, H]` × `[H, D]` matmul, and the batch loop runs across all $E$ experts. This means the grid is utilized for all $E$ experts within a single kernel invocation.

> **Tip:** Because the batch dimension is unrolled inside the kernel, the grid shape and `per_core_M` / `per_core_N` values must be chosen based on the per-expert problem size `[C, H]` × `[H, D]`, not the full `[E×C, H]` matrix. This is the key distinction between the batched formulation and a flat `[E×C, H]` dense matmul.

---

## 2. Padding to Expert Capacity

### 2.1 Why Padding Is Required

TTNN requires static tensor shapes for program caching and tracing (see Chapter 2, `ttnn_programming_model.md`). The number of tokens routed to each expert varies per forward pass due to stochastic routing decisions. If tensor shapes were allowed to vary, the program cache would miss on nearly every step, causing recompilation overhead.

The solution is to allocate a fixed-size token buffer per expert — the expert capacity $C$ — and zero-pad any expert that receives fewer than $C$ tokens. The shape `[E, C, H]` is constant across forward passes; only the non-padding values change.

### 2.2 Capacity Formula

The expert capacity is:

$$C = \left\lceil \frac{k \times B \times S}{E} \right\rceil$$

This is the expected number of tokens per expert, rounded up. The reasoning: $B \times S$ total tokens each activate $k$ experts, producing $k \times B \times S$ total (token, expert) assignments distributed across $E$ experts. Under uniform routing, each expert receives $k \times B \times S / E$ assignments on average.

A capacity factor $\alpha \geq 1$ is sometimes applied as an oversubscription buffer:

$$C = \left\lceil \alpha \times \frac{k \times B \times S}{E} \right\rceil$$

With $\alpha = 1.0$, tokens that would overflow a full expert are dropped. Load balancing losses during training push routing toward uniformity to minimize drops.

**Example — decode, B=32, S=1:**

$$C = \left\lceil \frac{8 \times 32 \times 1}{256} \right\rceil = \left\lceil \frac{256}{256} \right\rceil = 1$$

**Example — prefill, B=1, S=2048:**

$$C = \left\lceil \frac{8 \times 1 \times 2048}{256} \right\rceil = \left\lceil \frac{16384}{256} \right\rceil = 64$$

**Example — prefill, B=4, S=2048:**

$$C = \left\lceil \frac{8 \times 4 \times 2048}{256} \right\rceil = \left\lceil \frac{65536}{256} \right\rceil = 256$$

### 2.3 FLOP Efficiency

FLOP efficiency is the fraction of computation performed on real (non-padded) tokens:

$$\text{FLOP efficiency} = \frac{\text{filled slots}}{C \times E} = \frac{k \times B \times S}{C \times E}$$

Under uniform routing with $\alpha = 1.0$, this approaches 1.0 as $k \times B \times S$ becomes large. At decode with $B=32$, $k=8$, $S=1$, $C=1$:

$$\text{FLOP efficiency} = \frac{8 \times 32 \times 1}{1 \times 256} = \frac{256}{256} = 1.0$$

This is the optimistic uniform case. With imbalanced routing (some experts receive 0 tokens, others receive >1), actual efficiency can fall well below 1.0 even when the formula suggests otherwise.

> **Tip:** The capacity formula gives an average-case expected value. In practice, routing imbalance causes the actual filled-slots count to be lower than $k \times B \times S / E$ for most experts on any given forward pass. At decode, where $C=1$, a single over-subscribed expert forces capacity overflow token drops; at $C > 1$ there is headroom to absorb imbalance without dropping tokens.

---

## 3. Expert Weight Tensor Layout

### 3.1 Gate and Up Projections: `[E, H, D]`

A standard SwiGLU or GeGLU expert FFN has two first-layer projections (gate and up) and one down projection:

```
gate_out = activation(token @ W_gate)   # [C, H] @ [H, D] → [C, D]
up_out   = token @ W_up                 # [C, H] @ [H, D] → [C, D]
expert_hidden = gate_out * up_out       # element-wise product: [C, D]
down_out = expert_hidden @ W_down       # [C, D] @ [D, H] → [C, H]
```

For the batched formulation, each weight tensor is stacked across all $E$ experts:

- **Gate projection weight:** `[E, H, D]` — [D UNVERIFIED — verify against Qwen3 Technical Report]
- **Up projection weight:** `[E, H, D]` — [D UNVERIFIED — verify against Qwen3 Technical Report]
- **Down projection weight:** `[E, D, H]` — [D UNVERIFIED — verify against Qwen3 Technical Report]

### 3.2 Storage on Device

The weight tensors are stored in DRAM in tile layout (`ttnn.TILE_LAYOUT`) with BFP8 dtype to reduce memory footprint and DRAM bandwidth:

```python
# Gate and up projection weights: [E, H, D] in BFP8
# E=256, H=7168, D=[D UNVERIFIED — verify against Qwen3 Technical Report]
W_gate = ttnn.from_torch(
    gate_weight_torch,       # [256, 7168, D] torch tensor
    dtype=ttnn.bfloat8_b,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)

W_up = ttnn.from_torch(
    up_weight_torch,         # [256, 7168, D]
    dtype=ttnn.bfloat8_b,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)

# Down projection weights: [E, D, H] in BFP8
W_down = ttnn.from_torch(
    down_weight_torch,       # [256, D, 7168]
    dtype=ttnn.bfloat8_b,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
```

The leading $E$ dimension is the batch dimension from TTNN's perspective. Each expert's weight slice is a contiguous `[H, D]` or `[D, H]` matrix in DRAM; the kernel reads each expert's slice in turn during the batched dispatch.

> **Warning:** The weights for all 256 experts must fit in DRAM. On a single Wormhole B0 device (12 GB DRAM), the total weight footprint for gate + up + down in BFP8 is approximately $3 \times E \times H \times D \times \text{bytes\_per\_BFP8\_element}$. With $E=256$, $H=7168$, D=[D UNVERIFIED — verify against Qwen3 Technical Report], this must be verified against device memory capacity. For 8 T3K devices with expert parallelism (Chapter 7), each chip holds $E/8 = 32$ experts, reducing per-chip weight memory by 8×.

### 3.3 Contiguous Expert Slices and DRAM Locality

TTNN reads the batch dimension sequentially during a batched matmul. For good DRAM access locality, the weight tensor should be laid out so that all tiles of expert $e$'s weight matrix are contiguous in memory before expert $e+1$'s tiles begin. The standard `[E, H, D]` torch tensor layout (major to minor: E → H → D) produces this layout automatically when converted to TTNN tile layout.

---

## 4. Worked Example: Decode Forward Pass

This section traces through a complete single-decode-step batched expert FFN matmul.

**Setup:**
- $B = 32$ sequences, $S = 1$ token per sequence (decode)
- $E = 256$ experts, $k = 8$ top-K routing
- $H = 7168$ (`d_model`)
- $D$ = [D UNVERIFIED — verify against Qwen3 Technical Report]
- Weight dtype: BFP8; Activation dtype: BF16

### Step 1: Router Output

The router produces:

```
router_indices: [32, 8]   — for each of 32 tokens, 8 expert IDs (values in [0, 255])
router_scores:  [32, 8]   — softmax weights, sum to 1.0 per token
```

Total (token, expert) assignments: $32 \times 8 = 256$.

### Step 2: Compute Expert Capacity

$$C = \left\lceil \frac{k \times B \times S}{E} \right\rceil = \left\lceil \frac{8 \times 32 \times 1}{256} \right\rceil = 1$$

### Step 3: Gather into Expert Buffer

Build `expert_tokens: [256, 1, 7168]` by placing each token into its expert's capacity slot:

```
expert_tokens[expert_id, 0, :] = hidden_states[token_idx, :]
```

For the $\lceil C / 32 \rceil$ tile calculation: since $C=1 < 32$, TTNN pads $C$ to 32 (the minimum tile dimension) when converting to tile layout. The effective padded shape is `[256, 32, 7168]` in tile layout, with only the first token slot populated.

$$M_t = \lceil C / 32 \rceil = \lceil 1 / 32 \rceil = 1$$

### Step 4: Matmul Shapes

**Gate projection (first linear layer):**

```
A: [E, C_pad, H]  = [256, 32, 7168]   (BF16, tile layout; C padded to 32)
B: [E, H, D]      = [256, 7168, D]    (BFP8, tile layout; D UNVERIFIED)
C: [E, C_pad, D]  = [256, 32, D]      (BF16 output; D UNVERIFIED)
```

Tile counts for the per-expert matmul:

$$M_t = \lceil 32 / 32 \rceil = 1$$
$$K_t = \lceil 7168 / 32 \rceil = 224$$
$$N_t = \lceil D / 32 \rceil \quad \text{[N\_t UNVERIFIED — verify against Qwen3 Technical Report]}$$

**Down projection (second linear layer):**

```
A: [E, C_pad, D]  = [256, 32, D]      (BF16, tile layout; D UNVERIFIED)
B: [E, D, H]      = [256, D, 7168]    (BFP8, tile layout; D UNVERIFIED)
C: [E, C_pad, H]  = [256, 32, 7168]   (BF16 output)
```

### Step 5: Scatter and Combine

After the down projection, scatter the per-expert outputs back to the original token positions and combine with router scores:

```python
# output_buffer: [total_tokens, H] = [32, 7168]
# expert_output: [256, 32, 7168] (only [:, 0, :] is valid; rest is padding)
for token_idx in range(total_tokens):
    for slot in range(k):
        expert_id = router_indices[token_idx, slot]
        score     = router_scores[token_idx, slot]
        output_buffer[token_idx, :] += score * expert_output[expert_id, 0, :]
```

> **Note:** As with the gather, this scatter must be implemented as a device op in production code. The Python loop is used here only to illustrate the semantics.

### Summary of Decode Example Shapes

| Tensor | Shape | Dtype | Notes |
|--------|-------|-------|-------|
| `hidden_states` | `[32, 7168]` | BF16 | Input to MoE layer |
| `expert_tokens` | `[256, 32, 7168]` | BF16 (tile) | After gather + pad to tile boundary |
| `W_gate`, `W_up` | `[256, 7168, D]` | BFP8 (tile) | D UNVERIFIED |
| `gate_out`, `up_out` | `[256, 32, D]` | BF16 (tile) | D UNVERIFIED |
| `expert_hidden` | `[256, 32, D]` | BF16 (tile) | After SwiGLU; D UNVERIFIED |
| `W_down` | `[256, D, 7168]` | BFP8 (tile) | D UNVERIFIED |
| `expert_output` | `[256, 32, 7168]` | BF16 (tile) | After down projection |
| `output_buffer` | `[32, 7168]` | BF16 | After scatter + weighted combine |

---

## Next Steps

Proceed to [program_configs_batched.md](program_configs_batched.md) to select and validate a `MatmulMultiCoreReuseMultiCastProgramConfig` for the shapes derived here, including L1 budget checks for both the decode ($C=1$) and prefill ($C=64$) regimes.
