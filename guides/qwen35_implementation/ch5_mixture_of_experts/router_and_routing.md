# Router and Routing — Weight Layout, Sync Point, and Host Computation

## Router Weight Layout

The router is a single linear projection from hidden space to expert logits. In the HuggingFace
checkpoint it is stored under the key `mlp.gate.weight` with shape $[256, 2048]$ — 256 experts,
each with a 2048-dimensional weight vector.

In `Qwen35MoE.__init__`, this weight is transposed, unsqueezed, and loaded to device as a
bfloat16 DRAM tensor:

```python
# Router weight on device: [1, 1, hidden, num_experts] for ttnn.linear
router_w = state_dict[f"{prefix}.gate.weight"].T.unsqueeze(0).unsqueeze(0).contiguous()
self.router_weight_tt = ttnn.as_tensor(
    router_w,
    dtype=ttnn.bfloat16,
    device=mesh_device,
    layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    cache_file_name=cache_name("gate"),
)
```

The resulting on-device shape is:

```
router_weight_tt: [1, 1, 2048, 256]   # [1, 1, hidden_size, num_experts]
```

The `.T` transpose converts the HF layout $[256, 2048]$ to $[2048, 256]$ so that `ttnn.linear`
can compute $\mathbf{x} W$ directly where $\mathbf{x}$ is $[1, 1, B, 2048]$ and $W$ is
$[1, 1, 2048, 256]$.

**Why bfloat16 for the router?** The router weight is tiny (256 × 2048 × 2 bytes = 1 MB per
layer) and its logits directly determine expert selection — a quantization error that shifts
the argmax to a wrong expert degrades output quality. Keeping the router in bf16 costs negligible
DRAM at 40 layers × 1 MB = 40 MB total.

## Device Router Matmul

During the forward pass, the router logit computation is a standard `ttnn.linear` call:

```python
# Router logits on device: [1,1,B,256] -- only sync 256 floats, not 2048
router_logits = ttnn.linear(x, self.router_weight_tt, memory_config=L1)
```

The result `router_logits` has shape $[1, 1, B, 256]$ and is placed in L1 (not DRAM) because:

- The tensor is small ($B \times 256$ bfloat16 values, e.g., $32 \times 256 \times 2 = 16\ \text{KB}$
  for batch=32).
- It will be read back to host immediately; keeping it in L1 avoids an extra DRAM roundtrip
  before the `ttnn.to_torch` call.

This matmul is dispatched to the command queue **after** the shared expert matmuls and before the
`ttnn.to_torch` sync. The device executes the full queue (shared expert + router) concurrently
with no intervening Python-level barriers.

## The One Mandatory Sync

The routing sync is the **only host-device synchronization required by the MoE layer per token**:

```python
# --- Sync: read router logits (row 0 representative for all batch items) ---
logits_cpu = ttnn.to_torch(router_logits).float()[0, 0, 0, : self.num_experts]
ttnn.deallocate(router_logits)
```

Breaking this down:

- `ttnn.to_torch(router_logits)` blocks the host until all previously dispatched device ops
  complete and the tensor data is available on CPU. This is the sync point.
- `.float()` converts from bfloat16 to float32 for numerically precise top-k comparison.
- `[0, 0, 0, :self.num_experts]` extracts exactly the first 256 logit values from row 0,
  discarding any tile-padding.
- `ttnn.deallocate(router_logits)` immediately frees the L1 buffer.

The DMA transfer volume is:

$$256 \times 2\ \text{bytes (bf16 logits transferred)} = 512\ \text{bytes}$$

The router logits are produced by `ttnn.linear` against a `ttnn.bfloat16` weight, so the tensor
on device is bf16. `ttnn.to_torch` DMA-copies 512 bytes of bf16 data from device to host; the
subsequent `.float()` call widens those values to float32 (1024 bytes) on the CPU, but that
conversion happens after the transfer. In practice, `ttnn.to_torch` syncs the device command
queue and DMA-copies the entire tile-padded tensor before slicing, but the logically relevant
data is just those 256 bf16 values.

## Host Top-k and Softmax

After the sync, routing runs entirely on CPU:

```python
topk_vals, topk_ids = torch.topk(logits_cpu, self.num_experts_per_tok)
weights = F.softmax(topk_vals, dim=-1)
```

**Step 1 — Top-k selection.** `torch.topk(logits_cpu, 8)` returns:

- `topk_vals`: the 8 largest logit values, shape $[8]$, float32.
- `topk_ids`: the 8 corresponding expert indices (integers in $[0, 255]$), shape $[8]$.

The logits are **not** softmaxed before top-k. Raw logit ordering is used for selection.

**Step 2 — Routing weights.** `F.softmax(topk_vals, dim=-1)` normalizes the top-8 logit values
into a probability distribution:

$$w_i = \frac{e^{l_i}}{\sum_{j=1}^{8} e^{l_j}}, \quad i = 1, \ldots, 8$$

These weights satisfy $\sum_{i=1}^{8} w_i = 1$ and are used to scale each expert's output
contribution before accumulation.

The reference implementation in `test_a3b_pcc.py` confirms this two-step process:

```python
logits = token @ router_w.T
topk_vals, topk_ids = torch.topk(logits, NUM_EXPERTS_PER_TOK)
routing_weights = F.softmax(topk_vals, dim=-1)
```

## Shared Expert Gate

The shared expert gating runs entirely on device, requiring no sync. The `shared_expert_gate.weight`
maps the input to a single scalar gate value per batch row:

```python
# Shared expert gate on device: sigmoid(x @ gate_weight) -> [1,1,B,1]
gate = ttnn.linear(x, self.shared_gate_weight_tt, memory_config=L1)
gate = ttnn.sigmoid(gate, memory_config=L1)
shared_out = ttnn.mul(shared_out, gate, memory_config=L1)
ttnn.deallocate(gate)
```

The weight `self.shared_gate_weight_tt` was loaded from `shared_expert_gate.weight` as:

```python
gate_w = state_dict[f"{prefix}.shared_expert_gate.weight"].T.unsqueeze(0).unsqueeze(0).contiguous()
self.shared_gate_weight_tt = ttnn.as_tensor(
    gate_w,
    dtype=ttnn.bfloat16,
    device=mesh_device,
    layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    cache_file_name=cache_name("shared_expert_gate"),
)
```

On-device shape:

```
shared_gate_weight_tt: [1, 1, 2048, 1]   # [1, 1, hidden_size, 1]
```

After `ttnn.linear`, `gate` has shape $[1, 1, B, 1]$ — one scalar per batch row. After
`ttnn.sigmoid`, this becomes a value in $(0, 1)$ that modulates the shared expert output.
Because the gate shape $[1, 1, B, 1]$ broadcasts correctly against `shared_out` shape
$[1, 1, B, 2048]$, the single `ttnn.mul` applies the gate to all hidden dimensions simultaneously.

---

**Next:** [`expert_computation.md`](./expert_computation.md)
