# Fused Activations

## Current Implementation

### Standalone Activation Modules

TT-Symbiote defines three standalone activation modules in `modules/activation.py`:

```python
class TTNNSilu(TTNNModule):
    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        if input_tensor.layout != ttnn.TILE_LAYOUT:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, ...)
        tt_output = ttnn.silu(input_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return tt_output

class TTNNReLU(TTNNModule):
    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        if input_tensor.layout != ttnn.TILE_LAYOUT:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, ...)
        tt_output = ttnn.relu(input_tensor, memory_config=input_tensor.memory_config())
        return tt_output

class TTNNGelu(TTNNModule):
    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        if input_tensor.layout != ttnn.TILE_LAYOUT:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, ...)
        tt_output = ttnn.gelu(input_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return tt_output
```

Each of these reads its input from DRAM, applies a single elementwise function, and writes the result back to DRAM.

### Existing Fusion: TTNNLinearActivation

TT-Symbiote already has a partial fusion pattern in `modules/linear.py` (line 329):

```python
class TTNNLinearActivation(TTNNModule):
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)      # TTNNLinear -> DRAM
        hidden_states = self.activation(hidden_states)  # ttnn.silu/gelu -> DRAM
        return hidden_states
```

This is used by `TTNNLinearSilu` (line 367) and `TTNNLinearGelu` (line 353):

```python
class TTNNLinearSilu:
    @classmethod
    def from_parameters(cls, weight, bias=None, linear_class=TTNNLinear):
        return TTNNLinearActivation.from_parameters(weight, linear_class, ttnn.silu, nn.SiLU(), bias)

class TTNNLinearGelu:
    @classmethod
    def from_parameters(cls, weight, bias=None, linear_class=TTNNLinear):
        return TTNNLinearActivation.from_parameters(weight, linear_class, ttnn.gelu, nn.GELU(), bias)
```

**The limitation:** Although the classes are named "fused," the implementation still executes two separate TTNN kernel calls -- `ttnn.linear()` followed by `ttnn.silu()` or `ttnn.gelu()`. The matmul output is written to DRAM, then the activation reads it back. This is a "Python-level fusion" (single module call) but not a "kernel-level fusion" (single hardware kernel).

### SwiGLU Pattern in MoE MLP

The GLM-4 and LLaMA MLP uses the SwiGLU activation pattern, visible in `TTNNGlm4MoeMLP.forward()` (moe.py, line 677):

```python
class TTNNGlm4MoeMLP(TTNNModule):
    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x_gate = self.gate_proj(x)    # TTNNLinearSilu: linear + silu -> DRAM
        x_up = self.up_proj(x)        # TTNNLinear: linear -> DRAM
        x = ttnn.mul(x_gate, x_up)    # elementwise mul -> DRAM
        x = self.down_proj(x)         # TTNNLinear: linear -> DRAM
        return x
```

This is also the pattern in `Glm4MoeMLP.forward()` (moe.py, line 399):

```python
def forward(self, x):
    down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    return down_proj
```

The SwiGLU pattern involves: `gate_proj(x)` then `silu` then `mul(up_proj(x))` then `down_proj`. Currently this is 4 separate kernel launches with 3 intermediate DRAM tensors.

## Performance Bottleneck Analysis

### Per-Activation DRAM Overhead

For a model with `hidden_size=4096`, `intermediate_size=11008` (LLaMA-7B), `seq_len=1024`:

| Op in MLP | Output Tensor (BF16) | DRAM Write | DRAM Read |
|-----------|---------------------|-----------|----------|
| `gate_proj` (linear) | [1, 1024, 11008] = 21.5 MB | Yes | Yes (silu) |
| `silu` | [1, 1024, 11008] = 21.5 MB | Yes | Yes (mul) |
| `up_proj` (linear) | [1, 1024, 11008] = 21.5 MB | Yes | Yes (mul) |
| `mul` | [1, 1024, 11008] = 21.5 MB | Yes | Yes (down_proj) |
| `down_proj` (linear) | [1, 1024, 4096] = 8 MB | Yes | Yes (next layer) |

**Total MLP intermediate DRAM traffic: ~94 MB written + ~94 MB read = ~188 MB per layer.**

The `silu` activation alone accounts for 43 MB of unnecessary DRAM traffic (21.5 MB write + 21.5 MB read). It reads the entire `gate_proj` output from DRAM, applies a simple elementwise function, and writes it back -- only for `mul` to immediately read it again.

### Frequency

Every decoder layer has an MLP block. For a 32-layer model, the total MLP DRAM overhead is ~6 GB per forward pass. Fusing the activation into the matmul eliminates ~1.4 GB of that.

## TT-Lang Kernel Designs

### Design 1: Fused Linear + Activation

Fold the activation function into the matmul's output tile processing. Following the DFB patterns from [Chapter 1](../ch1_programming_model/index.md):

```python
@ttl.operation(grid="auto")
def fused_linear_activation(
    x_in: ttnn.Tensor,       # [batch, seq_len, in_features]
    weight: ttnn.Tensor,     # [in_features, out_features]
    bias: ttnn.Tensor,       # [out_features] or None
    out: ttnn.Tensor,        # [batch, seq_len, out_features]
    activation: str,         # "silu", "gelu", "relu"
) -> None:
    in_tiles = x_in.shape[-1] // ttl.TILE_SHAPE[0]
    out_tiles = weight.shape[-1] // ttl.TILE_SHAPE[1]
    seq_tiles = x_in.shape[-2] // ttl.TILE_SHAPE[0]

    x_dfb = ttl.make_dataflow_buffer_like(x_in, shape=(1, 1), block_count=2)
    w_dfb = ttl.make_dataflow_buffer_like(weight, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        for st in range(seq_tiles):
            for ot in range(out_tiles):
                # Accumulate matmul
                acc = ttl.math.fill(0)
                for kt in range(in_tiles):
                    with x_dfb.wait() as x_blk, w_dfb.wait() as w_blk:
                        acc += x_blk @ w_blk

                # Add bias (if present)
                # ... bias broadcast ...

                # Apply activation IN-REGISTER (no DRAM round-trip)
                if activation == "silu":
                    acc = ttl.math.silu(acc)
                elif activation == "gelu":
                    acc = ttl.math.gelu(acc)
                elif activation == "relu":
                    acc = ttl.math.relu(acc)

                # Write activated result
                with out_dfb.reserve() as out_blk:
                    out_blk.store(acc)
```

**This eliminates one full DRAM write+read cycle per activation.** The matmul accumulator is transformed in-register before being written out.

### Design 2: Fused SwiGLU Kernel

The full SwiGLU pattern -- `silu(x @ W_gate) * (x @ W_up)` -- can be computed in a single kernel that interleaves the two matmuls tile-by-tile:

```python
@ttl.operation(grid="auto")
def fused_swiglu(
    x_in: ttnn.Tensor,        # [batch, seq_len, hidden_size]
    w_gate: ttnn.Tensor,      # [hidden_size, intermediate_size]
    w_up: ttnn.Tensor,        # [hidden_size, intermediate_size]
    out: ttnn.Tensor,         # [batch, seq_len, intermediate_size]
) -> None:
    hidden_tiles = x_in.shape[-1] // ttl.TILE_SHAPE[0]
    inter_tiles = w_gate.shape[-1] // ttl.TILE_SHAPE[1]
    seq_tiles = x_in.shape[-2] // ttl.TILE_SHAPE[0]

    x_dfb = ttl.make_dataflow_buffer_like(x_in, shape=(1, 1), block_count=2)
    wg_dfb = ttl.make_dataflow_buffer_like(w_gate, shape=(1, 1), block_count=2)
    wu_dfb = ttl.make_dataflow_buffer_like(w_up, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        for st in range(seq_tiles):
            for it in range(inter_tiles):
                # Accumulate gate projection
                gate_acc = ttl.math.fill(0)
                for kt in range(hidden_tiles):
                    with x_dfb.wait() as x_blk, wg_dfb.wait() as wg_blk:
                        gate_acc += x_blk @ wg_blk

                # Apply SiLU in-register
                gate_activated = ttl.math.silu(gate_acc)

                # Accumulate up projection
                up_acc = ttl.math.fill(0)
                for kt in range(hidden_tiles):
                    with x_dfb.wait() as x_blk, wu_dfb.wait() as wu_blk:
                        up_acc += x_blk @ wu_blk

                # Elementwise multiply in-register
                result = gate_activated * up_acc

                # Single DRAM write
                with out_dfb.reserve() as out_blk:
                    out_blk.store(result)
```

**Savings:** Eliminates the `gate_proj` output, `silu` output, and `up_proj` output intermediates. Only the final `gate_activated * up_acc` result is written to DRAM.

### Design 3: Fused SwiGLU with Combined Gate/Up Weight

When the gate and up weights are stored as a single `[hidden_size, 2*intermediate_size]` tensor (as in the Qwen `gate_up_proj`), the kernel can issue a single matmul with double output width, then split in-register:

```python
@ttl.operation(grid="auto")
def fused_swiglu_combined(
    x_in: ttnn.Tensor,          # [batch, seq_len, hidden_size]
    w_gate_up: ttnn.Tensor,     # [hidden_size, 2*intermediate_size]
    out: ttnn.Tensor,           # [batch, seq_len, intermediate_size]
) -> None:

    @ttl.compute()
    def compute():
        for st in range(seq_tiles):
            for it in range(inter_tiles):
                # Single matmul accumulator with double width
                gate_acc = ttl.math.fill(0)
                up_acc = ttl.math.fill(0)
                for kt in range(hidden_tiles):
                    with x_dfb.wait() as x_blk:
                        with wgu_gate_dfb.wait() as wg_blk:
                            gate_acc += x_blk @ wg_blk
                        with wgu_up_dfb.wait() as wu_blk:
                            up_acc += x_blk @ wu_blk

                # Fused SiLU + multiply in-register
                result = ttl.math.silu(gate_acc) * up_acc

                with out_dfb.reserve() as out_blk:
                    out_blk.store(result)
```

**Additional benefit:** The input tiles `x_blk` are read once from DRAM and reused for both gate and up projections, halving input DRAM reads.

## Integration with TT-Symbiote

Following the integration contract from [Chapter 6](../ch6_integration_strategy/index.md), the fused kernels replace the existing `TTNNLinearActivation` and `TTNNGlm4MoeMLP` classes:

```python
class TTNNFusedLinearSilu(TTNNModule):
    """Drop-in replacement for TTNNLinearSilu with true kernel-level fusion."""

    def preprocess_weights_impl(self):
        self.tt_weight = preprocess_linear_weight(self.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        return fused_linear_activation(input_tensor, self.tt_weight, self.tt_bias, output, "silu")


class TTNNFusedSwiGLU(TTNNModule):
    """Replaces TTNNGlm4MoeMLP gate+silu+up+mul with a single kernel."""

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        intermediate = fused_swiglu(x, self.tt_w_gate, self.tt_w_up, intermediate_buf)
        return self.down_proj(intermediate)
```

The `TTNNFusedSwiGLU` module still calls `down_proj` as a separate kernel because the down projection has different dimensions (intermediate_size -> hidden_size) and is often paired with a reduce-scatter for tensor parallelism. Fusing it in would require the TT-Lang kernel to also handle the collective communication, which is a separate optimization.

## Expected Benefit

| Fusion | DRAM Saved Per Layer (LLaMA-7B scale) | Launch Reduction | Applicability |
|--------|---------------------------------------|-----------------|---------------|
| Linear + SiLU | 43 MB (gate_proj output) | 2 -> 1 | Every MLP with SiLU |
| Linear + GELU | 43 MB (dense output) | 2 -> 1 | ViT, BERT FFN |
| SwiGLU (gate + silu + up + mul) | 86 MB (gate + silu + up intermediates) | 4 -> 1 | LLaMA, Qwen, GLM-4 MLP |
| SwiGLU + combined weight | 86 MB + halved input reads | 4 -> 1 | Qwen (gate_up_proj) |

For a 32-layer LLaMA model during prefill (seq_len=1024):
- **SwiGLU fusion saves ~2.75 GB** of DRAM traffic per forward pass (86 MB x 32 layers)
- At ~300 GB/s DRAM bandwidth, that is ~9 ms saved
- Combined with the 4x reduction in kernel launches, total MLP latency improvement is estimated at 15-25%

### Activation Fusion is the Gateway

Activation fusion is the recommended first TT-Lang integration target because:

1. **Low complexity:** The kernel is a straightforward extension of matmul with a post-processing step
2. **High frequency:** Every layer has at least one MLP with an activation
3. **Existing precedent:** `TTNNLinearSilu` and `TTNNLinearGelu` already define the module interface; only the kernel implementation changes
4. **No collective communication:** Unlike MoE or attention, activations are purely local compute with no inter-device coordination

This makes it an ideal candidate for proving the TT-Lang integration workflow described in [Chapter 6](../ch6_integration_strategy/index.md) before tackling the more complex MoE and attention fusion targets.

---

**Next:** [Chapter 8 -- Developer Workflow and Multi-Device Considerations](../ch8_workflow_and_multidevice/index.md)
