# Host Recurrence: Why the DeltaNet Update Runs on CPU

## The Problem in One Sentence

On Blackhole, the SrcB register is 19-bit TF32. If a circular buffer (CB) feeding the SrcB
slot is formatted as fp32, the hardware stalls; therefore, standard element-wise TTNN ops
cannot perform fp32-in, fp32-out computation on Blackhole today. This prevents implementing
the DeltaNet state update entirely on device using standard operations.

---

## The Blackhole fp32 CB Constraint

Blackhole's compute engine has a strict type restriction on its source registers:

- **SrcA** can accept a tile read from an fp32, bfloat16, or bfloat8_b circular buffer.
- **SrcB** is a 19-bit TF32 register. Reading a tile from an fp32 CB into SrcB causes the
  hardware to hang (deadlock / stall with no error recovery).

This means any TTNN op that routes its second operand through SrcB — which is the normal
path for element-wise binary ops (`ttnn.multiply`, `ttnn.add`) — will hang if that operand
CB is fp32.

The DeltaNet recurrence requires fp32 because it accumulates outer products across many token
steps. Each step computes:

$$S \leftarrow S \cdot \exp(g)$$
$$\boldsymbol{\delta} = (\mathbf{v} - S^\top\mathbf{k}) \cdot \beta$$
$$S \leftarrow S + \mathbf{k} \otimes \boldsymbol{\delta}$$

> **Note:** The update cannot be written as the single compressed form $S \leftarrow S \cdot \exp(g) + \mathbf{k} \otimes \boldsymbol{\delta}$ without loss of meaning: $\boldsymbol{\delta}$ is computed from the *already-decayed* state $S \cdot \exp(g)$, not from the state before decay. The three-line sequential form above is the correct reading order (see the five-step recurrence in [`recurrence_math.md`](./recurrence_math.md)).

With bfloat16 (7 mantissa bits), the incremental rounding error per step is on the order of
$2^{-7} \approx 0.008$. Across 30 DeltaNet layers each updating $S$ per token, the error
compounds to produce garbage output. This was confirmed empirically: running the recurrence
in bfloat16 on device produces PCC values near 0 against the float32 reference after even
a few tokens.

The `PERF.md` file documents this as a known bottleneck:

> bf16 element-wise ops on Blackhole fp32 state: SrcB is 19-bit TF32, prevents all-device recurrence

---

## Workarounds Used in the Fused Kernel

The fused kernel `ttnn.experimental.gated_delta_net` (described fully in
[`fused_kernel.md`](./fused_kernel.md)) works around this restriction using three techniques
documented in the README's "Blackhole fp32 CB Reference" section:

1. **`init_sfpu` + `copy_tile` for fp32 read.**
   Instead of loading an fp32 CB tile into SrcB via the normal unpack path (which hangs),
   the kernel uses `init_sfpu` to put the SFPU into a state that can read from the fp32 CB,
   then `copy_tile` to load the fp32 tile into the SFPU destination register directly. This
   bypasses the SrcB register entirely.

2. **SFPU binary path for fp32 operations.**
   Arithmetic involving the fp32 state is performed through the SFPU (the scalar floating-point
   unit) rather than through the FPU matrix path that routes through SrcA/SrcB. The SFPU binary
   path supports fp32 natively and is not subject to the TF32 SrcB restriction.

3. **`binary_dest_reuse_tiles` for mixed fp32-DST + bf16-CB operations.**
   When mixing an fp32 accumulator in the DST register with a bfloat16 input CB (e.g.,
   applying the beta gate to the delta correction), `binary_dest_reuse_tiles` allows the DST
   register to retain its fp32 precision while reading the bfloat16 input, avoiding a
   precision-downgrade roundtrip.

These are non-trivial Metalium kernel engineering constraints, not standard TTNN API usage.
The README explicitly flags them:

> SrcB register is 19-bit (TF32). Element-wise ops with fp32 input CBs hang.
> Workarounds: init_sfpu+copy_tile for fp32 read, SFPU binary path for fp32 ops,
> binary_dest_reuse_tiles for mixed fp32-DST + bf16-CB operations.
> matmul_tiles has a separate unpack path that handles fp32 CBs.

---

## Historical Host-Recurrence Flow

Before the fused kernel existed, the recurrence ran on the host CPU as follows. For each
DeltaNet layer per token:

1. **`ttnn.to_torch`** — transfer the current device state tensor `_dev_state`
   (shape [batch, H, K, D]) to the host. This is a synchronization point: the device
   finishes all pending ops and the CPU waits for the DMA transfer.

2. **Float32 recurrence on CPU** — the five-step update (decay, retrieve, delta, rank-1
   update, read) is executed in Python/NumPy as float32 arithmetic.

3. **`ttnn.from_torch`** — transfer the updated state back to device DRAM before the
   output projection matmul.

This introduced one host-device sync per DeltaNet layer per token. With 30 DeltaNet layers
in the A3B model, the profiling breakdown shows:

| Component | Time | Syncs |
|-----------|------|-------|
| DeltaNet (30 layers) | 54 ms | 30 |
| Attention (10 layers) | 18 ms | 50 |
| norm + LM head | 14 ms | 1 |
| **Total** | **86 ms** | **~81** |

DeltaNet accounts for 54 ms of the 86 ms per-token total. The full 86 ms breaks down as approximately:
- ~35 ms total sync overhead (PCIe transfer + kernel launch latency)
- ~26 ms Python dispatch (looping over layers, slicing, calling into TTNN)
- ~20 ms device compute (projections, conv, out_proj)

The theoretical compute limit is ~5.8 ms/token (172 tok/s). Current efficiency is ~6.7%
(measured ~11.6 tok/s (= 1000 ms ÷ 86 ms/token) divided by theoretical 172 tok/s = ~6.7%).
The host recurrence path is the dominant bottleneck.

---

## Why bf16 Recurrence Is Incorrect

To make this concrete: with bfloat16 the state matrix $S$ has 7 mantissa bits. After each
token the absolute error on any element of $S$ grows by at most:

$$\epsilon_{\text{step}} \approx 2^{-7} \cdot \|\mathbf{k} \otimes \boldsymbol{\delta}\|_\infty$$

Over $T$ tokens the error in $S$ can grow as $O(T \cdot \epsilon_{\text{step}})$. Because
the output $\mathbf{o} = S^\top \mathbf{q}$ integrates over all `head_k_dim = 128` elements
of $S$, even a per-element error of 0.01 produces an output error of order $128 \times 0.01
= 1.28$ per head. With 30 layers the errors compound multiplicatively through the residual
stream, quickly producing outputs that are entirely uncorrelated with the float32 reference.

The test `test_deltanet_multi.py` was written specifically to observe this degradation. It
runs 20 sequential decode steps and prints the PCC at each step:

```python
for step in range(NUM_STEPS):
    x = torch.randn(1, 1, hidden_size)
    # ... reference float32 recurrence ...
    # ... TTNN forward (fused kernel path) ...
    pcc = F.cosine_similarity(ref_output.unsqueeze(0), tt_result.unsqueeze(0)).item()
    print(
        f"  Step {step:2d}: PCC={pcc:.6f}  ref_norm={ref_output.norm():.3f}  "
        f"tt_norm={tt_result.norm():.3f}  state_norm={ref_state.norm():.3f}"
    )
```

With the float32 fused kernel path, PCC remains above 0.999 across all 20 steps. With a
bfloat16 state, PCC drops below 0.9 within 3–5 steps.

---

## Current Design: Float32 State on Device

With the fused kernel active, the state tensor `_dev_state` is kept on device in float32
and the recurrence runs entirely within the kernel without a host roundtrip. The initialization
code for `_dev_state` (the `ttnn.from_torch` call inside `initialize_states`) is shown in
[`projections_and_conv.md`](./projections_and_conv.md).

The fused kernel receives `_dev_state` as an input, computes the updated state, and returns
it as a second output. The caller then copies the updated state back into `_dev_state` using
an in-place copy (to preserve the tensor address for Trace):

```python
result = ttnn.experimental.gated_delta_net(
    conv_out, z_flat, ba_flat,
    self._dt_bias_dev, self._neg_A_exp_dev,
    self._dev_state, self._norm_w_dev,
    scale=self.scale, norm_eps=self.args.norm_eps,
    key_dim=self.key_dim, gqa_ratio=self.gqa_ratio,
)
output_tt = result[0]
ttnn.copy(result[1], self._dev_state)   # in-place update, preserves address
ttnn.deallocate(result[1])
```

The `ttnn.copy` is the device-side equivalent of what was previously done as a host
`torch.Tensor` assignment. Because it runs on device and returns immediately (no sync), the
30-layer overhead drops from 54 ms to the device compute time alone.

The reason this is not yet the default production path is explained in
[`fused_kernel.md`](./fused_kernel.md): without Metal Trace, the `from_torch` call required
to upload the state on each token (in the absence of persistent device memory) still
dominates.

---

**Next:** [`fused_kernel.md`](./fused_kernel.md)
