# TTNNGemma4FFN

The feed-forward network in every Gemma 4 31B decoder layer uses a Gated Linear
Unit with GELU activation (GeGLU). The FFN structure and weight shapes are
identical across all 60 layers --- there is no per-layer-type variation in the
FFN, unlike the attention module.

## GeGLU Architecture

The GeGLU FFN computes:

```math
\text{FFN}(x) = W_{down} \cdot \left(\text{GELU}_{\tanh}(W_{gate} \cdot x) \odot W_{up} \cdot x\right)
```

where:

- $W_{gate}$: `[5376, 21504]` --- gate projection
- $W_{up}$: `[5376, 21504]` --- up projection
- $W_{down}$: `[21504, 5376]` --- down projection
- $\text{GELU}_{\tanh}$: GELU with tanh approximation (`gelu_pytorch_tanh`)
- $\odot$: element-wise multiplication

The gate and up projections operate in parallel on the same input. Their outputs
are combined through the gating mechanism before the down projection reduces
the dimension back to `hidden_size`.

## Forward Pass Dataflow

```text
hidden_states [1, 1, 5376]   (replicated on all devices)
      |
      +------> gate_proj [5376, 21504/8] -----> gate_out [1, 1, 2688]  (per device)
      |                                              |
      |                                         gelu_pytorch_tanh
      |                                              |
      |                                              v
      +------> up_proj [5376, 21504/8] -------> up_out [1, 1, 2688]    (per device)
                                                     |
                                                     v
                                          ttnn.mul(gelu(gate_out), up_out)
                                                     |
                                                     v
                                          intermediate [1, 1, 2688]     (per device)
                                                     |
                                          down_proj [2688, 5376]
                                                     |
                                          ttnn.all_reduce
                                                     |
                                                     v
                                          output [1, 1, 5376]           (replicated)
```

## TTNN Pseudocode

### Option 1: Separate Gate and Up Projections

```python
class TTNNGemma4FFN(TTNNModule):
    def __init__(self, config, mesh_device):
        super().__init__()
        # Column-parallel: shard output dim across 8 devices
        # Per-device shapes: [5376, 2688] each
        self.gate_proj = TTNNLinearIReplicatedWColSharded(
            in_features=5376,
            out_features=21504,   # sharded to 21504/8 = 2688 per device
        )
        self.up_proj = TTNNLinearIReplicatedWColSharded(
            in_features=5376,
            out_features=21504,
        )
        # Row-parallel: shard input dim across 8 devices
        # Per-device shape: [2688, 5376]
        self.down_proj = TTNNLinearIColShardedWRowSharded(
            in_features=21504,    # sharded to 2688 per device
            out_features=5376,
        )

    def forward(self, hidden_states):
        # Gate path: project and activate
        gate_output = self.gate_proj(hidden_states)       # [B, 1, 2688] per device
        gate_output = ttnn.gelu(gate_output, fast_and_approximate_mode=True)

        # Up path: project (no activation)
        up_output = self.up_proj(hidden_states)            # [B, 1, 2688] per device

        # Gating: elementwise multiply
        intermediate = ttnn.mul(gate_output, up_output)    # [B, 1, 2688] per device

        # Down projection + all-reduce (handled internally by the module)
        output = self.down_proj(intermediate)              # [B, 1, 5376] replicated

        return output
```

### Option 2: Fused Gate+Up Projection

An alternative approach fuses the gate and up projections into a single matmul:

```python
class TTNNGemma4FFN(TTNNModule):
    def __init__(self, config, mesh_device):
        super().__init__()
        # Fused gate+up: [5376, 43008] -> sharded to [5376, 5376] per device
        self.gate_up_proj = TTNNLinearIReplicatedWColSharded(
            in_features=5376,
            out_features=43008,   # 21504 * 2, sharded to 43008/8 = 5376 per device
        )
        self.down_proj = TTNNLinearIColShardedWRowSharded(
            in_features=21504,
            out_features=5376,
        )

    def forward(self, hidden_states):
        # Single fused matmul for gate + up
        gate_up = self.gate_up_proj(hidden_states)         # [B, 1, 5376] per device

        # Split into gate and up halves
        gate_output, up_output = ttnn.split(gate_up, 2, dim=-1)
        # gate_output: [B, 1, 2688], up_output: [B, 1, 2688]

        # Activate gate
        gate_output = ttnn.gelu(gate_output, fast_and_approximate_mode=True)

        # Gating
        intermediate = ttnn.mul(gate_output, up_output)    # [B, 1, 2688]

        # Down projection + all-reduce
        output = self.down_proj(intermediate)              # [B, 1, 5376]

        return output
```

### Trade-offs

| Aspect | Separate (Option 1) | Fused (Option 2) |
|--------|--------------------|--------------------|
| Matmul count | 3 (gate + up + down) | 2 (gate_up + down) |
| Weight memory per device | 2 x [5376, 2688] + [2688, 5376] = ~57.8 MB | [5376, 5376] + [2688, 5376] = ~57.8 MB |
| Peak activation memory | 2 x [B, 1, 2688] held simultaneously | [B, 1, 5376] then split; same peak |
| Extra ops | None | `ttnn.split` after fused matmul |
| Matmul efficiency | Two smaller matmuls may underutilize cores | One larger matmul better saturates compute |
| Program config complexity | Two identical configs for gate/up | One config for larger fused matmul |

The fused approach reduces the number of kernel launches from 2 to 1 for the
gate+up step, which can reduce host-side dispatch overhead. However, the split
operation adds a small cost. For decode (B=1, S=1), both matmuls are
memory-bandwidth-bound, so the fused approach primarily saves launch overhead.

**Recommendation:** Start with separate projections (Option 1) for
implementation simplicity and easier debugging against the HuggingFace
reference. Switch to the fused approach as an optimization once numerical
correctness is validated.

## GELU Activation

The `gelu_pytorch_tanh` activation is the tanh-approximated GELU:

```math
\text{GELU}_{\tanh}(x) = 0.5 \cdot x \cdot \left(1 + \tanh\!\left(\sqrt{\frac{2}{\pi}} \cdot (x + 0.044715 \cdot x^3)\right)\right)
```

In TTNN, this maps to:

```python
ttnn.gelu(gate_output, fast_and_approximate_mode=True)
```

The `fast_and_approximate_mode=True` flag selects the tanh approximation rather
than the exact GELU (which uses `erf`). This matches the PyTorch behavior of
`torch.nn.functional.gelu(x, approximate="tanh")`.

## Weight Shapes and Sharding

All FFN weights are sharded identically across the 8 devices, regardless of
whether the parent decoder layer uses sliding or global attention.

### Column-Parallel (Gate, Up)

| Projection | Full Shape | Per-Device Shape (TP=8) | Per-Device Bytes (BF16) |
|------------|-----------|------------------------|------------------------|
| Gate | `[5376, 21504]` | `[5376, 2688]` | 28.9 MB |
| Up | `[5376, 21504]` | `[5376, 2688]` | 28.9 MB |

The output dim 21504 divides cleanly by 8: 21504 / 8 = 2688. Each device
computes a 2688-wide slice of both the gate and up projections.

### Row-Parallel (Down)

| Projection | Full Shape | Per-Device Shape (TP=8) | Per-Device Bytes (BF16) |
|------------|-----------|------------------------|------------------------|
| Down | `[21504, 5376]` | `[2688, 5376]` | 28.9 MB |

The input dim 21504 is sharded by 8, matching the column-parallel output from
the gate/up step. The down projection's partial outputs are summed across
devices via `ttnn.all_reduce` (handled internally by
`TTNNLinearIColShardedWRowSharded`).

### Per-Device FFN Weight Total

| Component | Per-Device Bytes (BF16) |
|-----------|------------------------|
| Gate | 28.9 MB |
| Up | 28.9 MB |
| Down | 28.9 MB |
| **Total** | **~86.7 MB** |

Across all 60 layers: 60 x 86.7 MB = **~5,202 MB per device** (BF16).
At BFP8: **~2,601 MB per device**.

The FFN weights dominate the per-layer memory budget, accounting for roughly
72% of each sliding layer's weight storage and 57% of each global layer's
weight storage (see
[Chapter 6 --- Weight Sharding](../ch6_tp_sharding/weight_sharding.md)).

## Program Config Recommendations

### Gate and Up Projections: [5376, 2688] per device

For B=1 decode, the activation is `[1, 1, 5376]` and the weight is
`[5376, 2688]`. This is a matrix-vector multiply (the activation's sequence
dimension is 1).

Recommended program config approach:

- **Weight storage:** DRAM-sharded. The `[5376, 2688]` weight at BF16 is 28.9
  MB, too large for L1. Store in DRAM with a DRAM-sharded memory config for
  streaming reads during the matmul.
- **Activation placement:** L1. The input `[1, 1, 5376]` is 10.7 KB at BF16,
  fitting comfortably in L1.
- **Output placement:** L1. The output `[1, 1, 2688]` is 5.4 KB at BF16.
- **Compute kernel:** Use the 1D matmul program config optimized for tall-skinny
  weight matrices with a single-row activation.

### Down Projection: [2688, 5376] per device

The activation is `[1, 1, 2688]` and the weight is `[2688, 5376]`. Again a
matrix-vector multiply at B=1.

Recommended program config approach:

- **Weight storage:** DRAM-sharded. The `[2688, 5376]` weight at BF16 is 28.9
  MB.
- **Activation placement:** L1.
- **Output placement:** L1. The partial output `[1, 1, 5376]` is 10.7 KB.
- **Post-matmul:** The `TTNNLinearIColShardedWRowSharded` module handles the
  all-reduce internally, producing a replicated `[1, 1, 5376]` output on all
  devices.

### Batched Decode

At higher batch sizes (B > 1), the activation becomes `[B, 1, 5376]`. The
matmul transitions from memory-bandwidth-bound (B=1) toward compute-bound as B
increases. At B=32, the activation is `[32, 1, 5376]` (344 KB at BF16), still
fitting in L1.

For B >= 8, consider switching from DRAM-sharded weights to a 2D matmul program
config that better utilizes the compute grid. The exact crossover point depends
on profiling.

## CCL Operations

The FFN contributes exactly **one `ttnn.all_reduce`** per decoder layer,
occurring after the down projection. Combined with the all-reduce after the
attention O projection, each decoder layer performs **two all-reduce operations
total**.

The all-reduce payload for the FFN is `B x 1 x 5376 x 2 = 10,752 bytes` at
B=1 (BF16). This is latency-bound, not bandwidth-bound, as discussed in
[Chapter 6 --- Weight Sharding](../ch6_tp_sharding/weight_sharding.md#all-reduce-after-row-parallel-matmuls).

---

**Next:** [`ple_module.md`](./ple_module.md)
