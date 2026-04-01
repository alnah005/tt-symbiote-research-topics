# Architecture Overview — MoE Structure and Design Choices

## The 256+1 Expert Structure

Each of the 40 transformer layers in Qwen3.5-35B-A3B replaces the dense MLP with a
Mixture-of-Experts block. The block contains:

- **256 routed experts**, each a small SwiGLU MLP with intermediate size 512.
- **1 shared expert**, an identical SwiGLU MLP that is always active for every token.

Per-token, 8 of the 256 routed experts are selected (top-8 routing). The total active parameter
count for one MoE forward pass is therefore:

$$\text{active experts} = 8\ (\text{routed}) + 1\ (\text{shared})$$

With hidden size $d = 2048$ and intermediate size $m = 512$, each routed SwiGLU expert has:

$$\text{params per expert} = d \cdot 2m + m \cdot d = 2048 \cdot 1024 + 512 \cdot 2048 = 3{,}145{,}728$$

> **Note:** This formula reflects the **routed expert fused layout**, where gate and up projections are stored as a single $[d \times 2m]$ matrix. For the **shared expert**, the projections are stored as three separate matrices: `gate_proj` $[d \times m]$ + `up_proj` $[d \times m]$ + `down_proj` $[m \times d]$ = $3dm$ parameters — the same total count, but three distinct weight tensors rather than two.

The model configuration constants in `Qwen35MoE.__init__` map directly to the Qwen3.5 config:

```python
self.hidden_size = args.dim               # 2048
self.num_experts = args.num_experts       # 256
self.num_experts_per_tok = args.num_experts_per_tok  # 8
self.moe_intermediate_size = args.moe_intermediate_size          # 512
self.shared_expert_intermediate_size = args.shared_expert_intermediate_size  # 512
```

## Per-Expert SwiGLU Architecture

Every expert (routed and shared) computes a SwiGLU MLP:

$$\text{hidden} = \text{SiLU}(\mathbf{x} W_{\text{gate}}) \odot (\mathbf{x} W_{\text{up}})$$

$$\text{output} = \text{hidden}\, W_{\text{down}}$$

where $W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{d \times m}$ and $W_{\text{down}} \in \mathbb{R}^{m \times d}$.

For the **routed experts**, the gate and up projections are **fused into a single weight matrix**:

$$W_{\text{gate+up}} \in \mathbb{R}^{d \times 2m}$$

This means one matmul dispatch instead of two, followed by a `ttnn.split` to recover
$W_{\text{gate}}$ (first $m$ columns) and $W_{\text{up}}$ (last $m$ columns).

The **shared expert** does not use the fused layout. It loads `gate_proj` and `up_proj` as two
separate weight tensors (`shared_w1` and `shared_w3`), so two independent matmuls are dispatched
for its gate and up projections.

## Shared Expert Gating

The shared expert output is **additionally gated** by a learned scalar. After computing the
shared expert's SwiGLU output (shape $[1, 1, B, d]$), a separate weight
`shared_expert_gate.weight` projects the input token to a single scalar:

$$g = \sigma(\mathbf{x}\, W_{\text{shared gate}})$$

where $W_{\text{shared gate}} \in \mathbb{R}^{d \times 1}$ and $\sigma$ is the sigmoid function.

The final shared output is:

$$\text{shared out} = \text{SwiGLU}(\mathbf{x}) \cdot g$$

This gate is on device (stored in `self.shared_gate_weight_tt` as bf16) and does **not** require
a host sync. Its shape on device is `[1, 1, hidden, 1]` — a single column projection.

## Why Host Top-k

The routing decision (which 8 of 256 experts to invoke) is made on the host CPU rather than on
device. There are two reasons:

1. **Sync is unavoidable regardless.** Dispatching to specific experts by index requires the
   host to know which expert IDs were selected. Whether top-k runs on device or on host, the
   host still needs to read back the expert IDs before it can enqueue the 8 expert matmuls.
   Running top-k on host avoids a custom device kernel with no net sync savings.

2. **Data volume is negligible.** The router logits tensor is shape $[1, 1, B, 256]$. Only row 0
   is read back (one representative token); the DMA transfer is 512 bytes (256 × 2 bf16 bytes) — see `router_and_routing.md` for the full breakdown.

The code comment in `qwen35_moe.py` line 165 notes the tradeoff explicitly:

```python
# For batched decode with same prompt, all rows route identically.
# For different prompts, per-row routing would need token grouping (future work).
logits_cpu = ttnn.to_torch(router_logits).float()[0, 0, 0, : self.num_experts]
```

## Batched Decode Assumption

The demo (`demo_a3b.py`) supports batch sizes of 1 or 32. When `batch_size = 32`, all 32 rows
of the input tensor `x` carry **the same token embedding** (the same prompt replicated):

```python
for bi in range(batch_size):
    x_pad[0, 0, bi, :] = emb
```

Under this assumption, all batch rows produce identical router logits. Reading row 0 and using
its top-8 as the routing decision for all rows is exact — not an approximation.

For **different prompts per batch row**, each row would need independent routing, which would
require token grouping, separate expert dispatches per unique expert set, and a scatter-add
accumulation. This is identified as future work in the source.

## Forward-Pass Overlap Design

The ordering of operations in `Qwen35MoE.forward` is deliberately designed to maximize
overlap between shared expert computation and the routing sync:

```
Device queue:   shared_w1 matmul → shared_w3 matmul → SwiGLU → shared_w2 matmul
Device queue:   router matmul → (result sits in L1)
Device queue:   shared_gate matmul → sigmoid → multiply into shared_out
                                                     ↓
CPU sync:       ttnn.to_torch(router_logits)    ← blocks here, device finishes queued ops
CPU compute:    torch.topk + F.softmax
                                                     ↓
Device loop:    8 × [gate_up matmul → split → SwiGLU → down matmul → scale → add]
```

By the time `ttnn.to_torch(router_logits)` returns (the one mandatory sync), the device has
already completed all shared expert matmuls. The CPU then runs top-k/softmax while those
device results wait in L1, and the expert loop starts immediately after.

This design means the CPU top-k/softmax latency is **hidden behind** device shared-expert
compute rather than paid sequentially.

---

**Next:** [`router_and_routing.md`](./router_and_routing.md)
