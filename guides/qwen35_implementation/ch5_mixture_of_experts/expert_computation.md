# Expert Computation — Fused Gate+Up, SwiGLU Fusion, bfp4 Indexing, and L1 Accumulation

## Expert Weight Layout and bfp4 Storage

Each of the 256 routed experts has two weight tensors on device:

| Tensor | PyTorch source shape | On-device shape | dtype |
|--------|---------------------|-----------------|-------|
| `expert_gate_up[e]` | `[1, 1, hidden, 2*intermediate]` | `[1, 1, 2048, 1024]` | `bfloat4_b` |
| `expert_down[e]` | `[1, 1, intermediate, hidden]` | `[1, 1, 512, 2048]` | `bfloat4_b` |

The HuggingFace checkpoint stores all 256 experts' gate+up weights as a single packed 3D tensor
of shape $[256, 1024, 2048]$ (i.e., $[N_{\text{exp}}, 2m, d]$) and the down projections as
$[256, 2048, 512]$ (i.e., $[N_{\text{exp}}, d, m]$). During `__init__`, each expert's slice
is extracted, transposed, and loaded individually:

```python
raw_gate_up = state_dict[f"{prefix}.experts.gate_up_proj"]  # [256, 2*intermediate, hidden]
raw_down = state_dict[f"{prefix}.experts.down_proj"]        # [256, hidden, intermediate]
intermediate = self.moe_intermediate_size                   # 512

self.expert_gate_up = []
self.expert_down = []
for e in range(self.num_experts):
    gu = raw_gate_up[e].T.unsqueeze(0).unsqueeze(0).contiguous()  # [1,1,hidden,2*intermediate]
    self.expert_gate_up.append(
        ttnn.as_tensor(
            gu,
            dtype=expert_dtype,       # ttnn.bfloat4_b
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name(f"experts.{e}.gate_up"),
        )
    )
    dn = raw_down[e].T.unsqueeze(0).unsqueeze(0).contiguous()
    self.expert_down.append(
        ttnn.as_tensor(
            dn,
            dtype=expert_dtype,       # ttnn.bfloat4_b
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name(f"experts.{e}.down"),
        )
    )
```

The `.T` transposes from $[\text{out}, \text{in}]$ to $[\text{in}, \text{out}]$ so that
`ttnn.linear(x, W)` computes $\mathbf{x} W$ as a matrix multiply $[B, d] \times [d, \text{out}]$.

At forward time, expert selection by index is simply a Python list lookup:

```python
eid = topk_ids[i].item()        # integer in [0, 255]
self.expert_gate_up[eid]        # O(1) list access
self.expert_down[eid]
```

The 256 `expert_gate_up` tensors and 256 `expert_down` tensors are plain Python lists of TTNN
device tensors, all resident in DRAM throughout the model's lifetime.

## Fused Gate+Up Matmul

The gate and up projections are fused into a single `ttnn.linear` call using the $[d, 2m]$ weight matrix, then split to recover the two halves. See `architecture_overview.md` ("Per-Expert SwiGLU Architecture") for the rationale and shape diagrams. The concrete TTNN calls are:

```python
gate_up = ttnn.linear(x, self.expert_gate_up[eid], memory_config=L1)
gate_out, up_out = ttnn.split(gate_up, self.moe_intermediate_size, dim=3)
ttnn.deallocate(gate_up)
```

## SwiGLU Fusion

The SwiGLU nonlinearity is:

$$\text{hidden} = \text{SiLU}(\text{gate\_out}) \odot \text{up\_out}$$

where $\text{SiLU}(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}$.

TTNN fuses the SiLU activation and elementwise multiply into a single kernel call using the
`input_tensor_a_activations` parameter:

```python
hidden = ttnn.mul(
    gate_out,
    up_out,
    input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
    memory_config=L1,
)
ttnn.deallocate(gate_out)
ttnn.deallocate(up_out)
```

The `input_tensor_a_activations=[SILU]` argument instructs the TTNN multiply kernel to apply
SiLU to `gate_out` before computing the elementwise product — one kernel launch instead of two
(SiLU then multiply separately).

After the fused multiply, `gate_out` and `up_out` are explicitly deallocated to free L1 space
before the next expert's tensors are allocated. With 8 experts executing sequentially in L1,
careful deallocation prevents L1 overflow.

## Down Projection and Routing Weight Scale

The down projection maps the expert's intermediate representation back to hidden size:

$$\text{expert\_out} = \text{hidden}\, W_{\text{down}} \in \mathbb{R}^{B \times d}$$

```python
expert_out = ttnn.linear(hidden, self.expert_down[eid], memory_config=L1)
ttnn.deallocate(hidden)
```

The result is then scaled by the softmax routing weight $w_i$ for this expert:

```python
if w != 1.0:
    expert_out = ttnn.multiply(expert_out, w, memory_config=L1)
```

The `w != 1.0` guard skips the scalar multiply if the weight happens to be exactly 1.0 (which
can occur when `num_experts_per_tok = 1`). For the standard 8-expert case this is virtually
never true, so the guard is a no-cost micro-optimization.

## L1 Accumulation Loop — Full Forward Pass

The complete expert loop accumulates 8 weighted expert outputs into `result`, which is
initialized to `shared_out`:

```python
# --- Routed experts (top-k, fused gate+up, L1 intermediates) ---
result = shared_out
for i in range(self.num_experts_per_tok):
    eid = topk_ids[i].item()
    w = weights[i].item()

    # Fused gate+up: 1 matmul -> split -> SwiGLU (saves 1 matmul dispatch per expert)
    gate_up = ttnn.linear(x, self.expert_gate_up[eid], memory_config=L1)
    gate_out, up_out = ttnn.split(gate_up, self.moe_intermediate_size, dim=3)
    ttnn.deallocate(gate_up)
    hidden = ttnn.mul(
        gate_out,
        up_out,
        input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
        memory_config=L1,
    )
    ttnn.deallocate(gate_out)
    ttnn.deallocate(up_out)

    expert_out = ttnn.linear(hidden, self.expert_down[eid], memory_config=L1)
    ttnn.deallocate(hidden)

    # Weighted accumulation
    if w != 1.0:
        expert_out = ttnn.multiply(expert_out, w, memory_config=L1)
    result = ttnn.add(result, expert_out, memory_config=L1)
    ttnn.deallocate(expert_out)

return result
```

Key properties of the accumulation:

- All intermediate tensors (`gate_up`, `gate_out`, `up_out`, `hidden`, `expert_out`) use
  `L1_MEMORY_CONFIG`. The decode batch tensor `x` has shape $[1, 1, 32, 2048]$ in bfloat16
  = 128 KB, and expert intermediates are at most $[1, 1, 32, 1024]$ in bfloat16 = 64 KB.
  These fit comfortably in Blackhole's L1 SRAM alongside the matmul circular buffers.

- The loop dispatches 4 TTNN ops per expert (linear, split, fused-mul, linear) plus a scale
  and an add — `ttnn.split` and `ttnn.mul` are two separate kernel dispatches — for a total
  of approximately 6 device operations × 8 experts = ~48 dispatches per MoE layer per token.
  All ~48 are enqueued to the device command queue before any Python synchronization is needed.

## Reference: PyTorch MoE Forward

The `test_a3b_pcc.py` reference function shows the mathematical equivalence using standard
PyTorch:

```python
gate_up = weights[f"{p}mlp.experts.gate_up_proj"]  # [256, 1024, 2048]
down = weights[f"{p}mlp.experts.down_proj"]         # [256, 2048, 512]

routed_out = torch.zeros_like(hidden_states)
for i in range(NUM_EXPERTS_PER_TOK):
    eid = topk_ids[i].item()
    w = routing_weights[i].item()
    gate_w = gate_up[eid, :MOE_INTERMEDIATE, :]   # [512, 2048]
    up_w   = gate_up[eid, MOE_INTERMEDIATE:, :]   # [512, 2048]
    down_w = down[eid]                             # [2048, 512]
    gate_out = F.silu(F.linear(hidden_states, gate_w))
    up_out   = F.linear(hidden_states, up_w)
    expert_out = F.linear(gate_out * up_out, down_w)
    routed_out += w * expert_out

return shared_out + routed_out
```

The TTNN implementation is numerically equivalent: the fused `gate+up` matmul followed by
`ttnn.split` produces the same $[\text{gate\_out}, \text{up\_out}]$ as the separate PyTorch
`F.linear` calls. The PCC test in `TestMoEPCC.test_single_layer` validates this with a
threshold of 0.99.

---

**Next:** [`dram_budget.md`](./dram_budget.md)
