# Recurrence Math: The DeltaNet Update Rule

## What DeltaNet Is

DeltaNet is a linear-attention mechanism derived from "Gated Delta Networks with Softmax
Attention" (Yang et al., 2025). Unlike softmax attention, which computes $O(T^2)$ dot products
over the full sequence, DeltaNet maintains a fixed-size key-value memory matrix $S$ that is
updated at each token step in $O(1)$ time (constant with respect to sequence length). This is
the property that makes it practical to run as the dominant layer type in a large model.

The matrix $S$ has shape [num_v_heads, head_k_dim, head_v_dim]. For Qwen3.5-27B:
[48, 128, 128]. For Qwen3.5-35B-A3B: [32, 128, 128].

---

## The Five-Step Recurrence

At each token step the following five operations are applied in order. The notation uses
per-head indexing (one copy of these equations executes for each of the `num_v_heads` heads).

### Step 1 — Exponential Decay

$$S_t \leftarrow S_t \cdot \exp(g_t)$$

$g_t$ is a per-head scalar gate with value $\leq 0$, so $\exp(g_t) \in (0, 1]$. The matrix $S$
is multiplied element-wise by this scalar. This is the "forgetting" step: old memories decay
geometrically over time. When $g_t = 0$ the memory is preserved exactly; as $g_t \to -\infty$
the memory is wiped.

### Step 2 — Retrieve Current Value Estimate

$$\mathbf{m}_t = \sum_k S_t[k, :] \cdot k_t[k] = S_t^\top \mathbf{k}_t$$

Read out the current memory's estimate of the value associated with key $\mathbf{k}_t$.
$\mathbf{m}_t$ has dimension `head_v_dim`.

### Step 3 — Delta Correction

$$\boldsymbol{\delta}_t = (\mathbf{v}_t - \mathbf{m}_t) \cdot \beta_t$$

$\boldsymbol{\delta}_t$ is the error between the true value $\mathbf{v}_t$ and the memory's
current estimate $\mathbf{m}_t$, scaled by the beta gate $\beta_t \in (0, 1)$. When $\beta_t
= 1$ the full correction is applied; when $\beta_t = 0$ no update happens and the memory is
read-only at this step.

### Step 4 — Rank-1 Update

$$S_t \leftarrow S_t + \mathbf{k}_t \otimes \boldsymbol{\delta}_t$$

The outer product $\mathbf{k}_t \otimes \boldsymbol{\delta}_t$ has shape
[head_k_dim, head_v_dim], matching $S$. This adds the delta correction, distributed over
all key dimensions proportional to the key vector. It is a "write" operation with rank 1
(one pair of vectors fully specifies the update).

### Step 5 — Read Output

$$\mathbf{o}_t = S_t^\top \mathbf{q}_t$$

Multiply the updated state by the query vector to produce the per-head output.
$\mathbf{o}_t$ has dimension `head_v_dim`.

### Complete Reference Implementation

The following is a simplified excerpt of the Python reference used in the test suite
(`reference/test_deltanet_pcc.py`). It makes the five steps explicit. The actual source
also contains `if not output_final_state: last_recurrent_state = None` before the
return statement, which is omitted here for clarity:

```python
def torch_recurrent_gated_delta_rule(
    query, key, value, g, beta, initial_state, output_final_state,
    use_qk_l2norm_in_kernel=False
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key   = l2norm(key,   dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]
    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim)

    for i in range(sequence_length):
        q_t    = query[:, :, i]
        k_t    = key[:, :, i]
        v_t    = value[:, :, i]
        g_t    = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)  # Step 1 decay factor
        beta_t = beta[:, :, i].unsqueeze(-1)                   # Step 3 correction scale

        last_recurrent_state = last_recurrent_state * g_t                           # Step 1
        kv_mem  = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)            # Step 2
        delta   = (v_t - kv_mem) * beta_t                                           # Step 3
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)  # Step 4
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)       # Step 5

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state
```

---

## Gate Computation

The gate $g_t$ is not projected directly. It is computed from two learned components:

```math
g_t = -\exp(\texttt{A\_log}) \cdot \text{softplus}\!\left(a_t + \texttt{dt\_bias}\right)
```

- `A_log` is a learned parameter (shape [num_v_heads]) stored as the log of a positive decay
  rate; its exponential is always positive.
- $a_t$ is the per-token, per-head output of the `in_proj_a` projection (shape [num_v_heads]).
- `dt_bias` is a learned bias (shape [num_v_heads]) that shifts the effective time-step size.
- $\text{softplus}(x) = \log(1 + e^x)$ ensures the inner term is positive.
- The leading $-$ sign ensures $g_t \leq 0$, making $\exp(g_t) \leq 1$ (valid decay).

In the `GatedDeltaNet` constructor, the implementation precomputes `−exp(A_log)`
and stores it as `_neg_A_exp_dev` on device, eliminating one operation per token:

```python
A_exp = load_param("A_log").float().exp()
self._neg_A_exp_dev = ttnn.from_torch(
    (-A_exp).reshape(1, self.num_v_heads, 1, 1).bfloat16(),
    layout=ttnn.TILE_LAYOUT, device=mesh_device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
```

---

## Beta Gate

$$\beta_t = \sigma(b_t)$$

$b_t$ is the output of `in_proj_b` (shape [num_v_heads]), and $\sigma$ is the sigmoid function.
$\beta_t \in (0, 1)$ per head. A large $\beta_t$ means "write aggressively"; a small $\beta_t$
means "mostly read, minimal update."

---

## L2 Normalization of Q and K

Before entering the recurrence, both query and key vectors are L2-normalized:

$$\hat{\mathbf{q}} = \frac{\mathbf{q}}{\sqrt{\|\mathbf{q}\|_2^2 + \epsilon}}, \qquad
  \hat{\mathbf{k}} = \frac{\mathbf{k}}{\sqrt{\|\mathbf{k}\|_2^2 + \epsilon}}$$

The reference implementation uses $\epsilon = 10^{-6}$:

```python
def l2norm(x, dim=-1, eps=1e-6):
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm
```

This normalization is critical for stability. Without it, the outer product
$\mathbf{k}_t \otimes \boldsymbol{\delta}_t$ can have unbounded magnitude, causing the state
matrix $S$ to grow without bound over many token steps. The flag
`use_qk_l2norm_in_kernel=True` is always set in the Qwen3.5 call path.

After normalization, the query is additionally scaled by $1 / \sqrt{\text{head k dim}}$:

$$\mathbf{q}_{\text{scaled}} = \hat{\mathbf{q}} \cdot \frac{1}{\sqrt{d_k}}$$

This is the same scaling used in standard dot-product attention and prevents the dot product
$\mathbf{q}^\top \mathbf{k}$ from growing too large in deeper models.

---

## GQA Expansion

Qwen3.5 uses Grouped Query Attention (GQA) in the DeltaNet layers. The number of key/query
heads (`num_k_heads`) is smaller than the number of value heads (`num_v_heads`). The ratio is:

```math
\text{gqa ratio} = \texttt{num\_v\_heads} / \texttt{num\_k\_heads}
```

For Qwen3.5-27B: 48 / 16 = 3. For Qwen3.5-35B-A3B: 32 / 16 = 2.

After projecting and conv-filtering Q and K (shape [num_k_heads, head_k_dim]), they are
expanded to [num_v_heads, head_k_dim] via `repeat_interleave`:

```python
# After reshaping q, k to [batch, seq, num_k_heads, head_k_dim]:
if gqa_ratio > 1:
    q = q.repeat_interleave(gqa_ratio, dim=2)
    k = k.repeat_interleave(gqa_ratio, dim=2)
```

Each K-head is repeated `gqa_ratio` times so it aligns with one group of V-heads for the
outer product. This reduces projection cost — projecting $K$ and $Q$ to a smaller head count
saves parameter budget — while keeping the state matrix $S$ at full [num_v_heads, K, V] size.

---

## Gated RMSNorm (Post-Recurrence)

After the recurrence produces output $\mathbf{o}_t$ (shape [num_v_heads, head_v_dim]), it is
passed through a gated RMSNorm before the output projection:

$$\text{output normed} = \mathbf{o}_t \cdot \left(\text{mean}(\mathbf{o}_t^2) + \epsilon\right)^{-1/2} \cdot \mathbf{w}_{\text{norm}}$$

$$\text{final output} = \text{output normed} \cdot \text{SiLU}(z_t)$$

where $z_t$ is the output of `in_proj_z` (shape [num_v_heads, head_v_dim]) and $\mathbf{w}_{\text{norm}}$
is the per-dimension learned scale from `norm.weight` (shape [head_v_dim]).

The SiLU gate $\text{SiLU}(z_t) = z_t \cdot \sigma(z_t)$ acts as a learned mask over the
normalized output, allowing each head to selectively suppress or amplify dimensions. This is
analogous to the gating mechanism in SwiGLU MLPs.

The reference implementation (`Qwen3_5RMSNormGated`):

```python
class Qwen3_5RMSNormGated(nn.Module):
    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states
        if gate is not None:
            hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)
```

In the fused kernel path, this entire post-recurrence block (RMSNorm + SiLU gate) is computed
inside `ttnn.experimental.gated_delta_net` using the `norm_w` and `z_flat` inputs.

---

**Next:** [`projections_and_conv.md`](./projections_and_conv.md)
