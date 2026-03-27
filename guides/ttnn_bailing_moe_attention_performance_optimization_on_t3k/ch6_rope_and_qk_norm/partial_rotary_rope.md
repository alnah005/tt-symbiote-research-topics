# Partial Rotary RoPE and Non-Distributed Execution

## Overview

The Ling model sets `partial_rotary_factor=0.5`, meaning rotary positional encodings are applied only to the first half of each head's feature dimensions (`rotary_dim = 64` out of `head_dim = 128`). This single parameter choice has a direct structural consequence in the TTNN execution path: the distributed RoPE kernel (`TTNNDistributedRotaryPositionEmbedding`) is disabled and the non-distributed variant (`TTNNRotaryPositionEmbedding`) runs in its place. On a T3K 8-chip mesh, non-distributed execution means the RoPE kernel does not exploit the mesh's parallelism for this operation — it runs identically on every chip without coordination, or in the worst case runs only on one chip while others wait.

This file establishes: (1) why `partial_rotary_factor < 1.0` causes the dispatch branch to select the non-distributed kernel, (2) what performance cost this imposes on T3K, (3) whether padding the cos/sin tables to full `head_dim` would allow the distributed kernel to be re-enabled while preserving partial-rotary semantics, and (4) whether a post-embedding slice to discard padded dimensions is cheaper than running non-distributed RoPE.

All cost estimates are `[ESTIMATE]` unless annotated otherwise.

## Why `partial_rotary_factor < 1.0` Disables the Distributed Kernel

### The Dispatch Branch

In the TTNN model code for BailingMoE attention, the selection between distributed and non-distributed RoPE is a conditional branch gated on whether the rotary dimension equals the full head dimension:

```python
# Conceptual representation of the dispatch logic in attention.py
rotary_dim = int(self.partial_rotary_factor * self.head_dim)  # = int(0.5 * 128) = 64

if rotary_dim == self.head_dim:
    # Full rotary: every element of head_dim participates.
    # The cos/sin tables have shape (seq_len, head_dim) and can be sharded
    # across chips along the head_dim axis without any masking logic.
    rope_op = TTNNDistributedRotaryPositionEmbedding(...)
else:
    # Partial rotary: only the first rotary_dim elements are rotated;
    # elements [rotary_dim : head_dim] are passed through unchanged.
    # The cos/sin tables have shape (seq_len, rotary_dim), which does not
    # divide evenly across chips when rotary_dim != head_dim in general,
    # and the distributed kernel's shard structure assumes the full head_dim.
    rope_op = TTNNRotaryPositionEmbedding(...)
```

The root constraint is that `TTNNDistributedRotaryPositionEmbedding` is designed around the assumption that the cos/sin tables span the full `head_dim`. Its mesh-sharding strategy divides the head dimension across the 8 chips: each chip handles `head_dim / 8 = 128 / 8 = 16` elements of the rotary computation. This works cleanly when all `head_dim` elements participate in RoPE.

When `partial_rotary_factor < 1.0`, only `rotary_dim = 64` elements participate. The remaining `head_dim - rotary_dim = 64` elements are identity-mapped (pass-through). The distributed kernel has no built-in mechanism to apply RoPE to a prefix of `head_dim` and pass through the suffix: its loop structure and cos/sin table layout both presuppose full coverage. Using the distributed kernel with a `rotary_dim`-sized cos/sin table would produce incorrect results for the pass-through elements, which is why the dispatch selects the non-distributed variant instead.

### Structure of the Non-Distributed Kernel

`TTNNRotaryPositionEmbedding` runs entirely within a single chip's Tensix cores. Each core holds one HEIGHT_SHARDED shard — one head's worth of data, shape `(32, 128)` tile-padded. The kernel applies the partial-rotary computation:

```
For elements [0 : rotary_dim=64]:   apply cos/sin rotation
For elements [rotary_dim : head_dim]: copy unchanged (identity)
```

The cos/sin table used is pre-computed on host and stored in DRAM or L1, with shape `(1, rotary_dim)` = `(1, 64)` at decode seq_len=1. The kernel reads these 64 values and applies the standard complex rotation:

```
q_rot[i]   = q[i]   * cos[i] - q[i + rotary_dim/2] * sin[i]
q_rot[i+rotary_dim/2] = q[i] * sin[i] + q[i + rotary_dim/2] * cos[i]
```

for `i` in `[0, rotary_dim/2)` = `[0, 32)`.

This kernel executes on a single chip's cores. On T3K, each chip independently runs the identical RoPE computation on its replicated copy of the Q and K tensors. The 8 chips do not communicate during RoPE — the operation is fully data-parallel across the mesh by virtue of tensor replication, not by mesh sharding of the RoPE computation itself.

## Performance Cost of Non-Distributed RoPE on T3K

### Execution Mode on T3K

When `TTNNRotaryPositionEmbedding` is dispatched on a T3K mesh, each chip executes the same kernel on its local copy of Q and K. The mode of execution is **replicated non-distributed**: all 8 chips perform the full RoPE computation. There is no reduction over chips, no communication, and no result that is assembled from partial contributions. The execution is correct because Q and K are replicated tensors (each chip holds the full tensor — see Chapter 4, `decode_tensor_lifecycle.md`).

### Kernel Latency Estimate

The RoPE kernel on a single chip processes:
- Q: `(1, 16, 32, 128)` tile-padded, 16 heads, 128 elements per head, `rotary_dim=64` active elements per head
- K: `(1, 4, 32, 128)` tile-padded, 4 heads, 128 elements per head

For Q, with 16 cores each handling one head's shard `(32, 128)`:
- Active compute: 32 multiplications and 32 additions per core for the rotation, over 32 elements × 128 / 32 = 128 active-row elements per core. At decode, only 1 token row is active (the rest are tile-padding zeros). Effective arithmetic per core: ≈ 2 × 64 = 128 FP ops.
- At Wormhole's single-core SFPU throughput of ~500 GFLOPS [ESTIMATE] this is sub-nanosecond arithmetic.
- Kernel dispatch overhead dominates: estimated **5–10 µs** per RoPE call [ESTIMATE].

The kernel itself is fast. The surrounding data movement (T1a, T1b, T2a, T2b — totalling ≈ 64 µs) is what makes RoPE expensive in the decode path, not the arithmetic.

### Would the Distributed Kernel Be Faster?

`TTNNDistributedRotaryPositionEmbedding` shards the RoPE computation across the 8 chips in the mesh, with each chip computing RoPE for `head_dim / 8 = 16` elements of each head. After the local computation, a collective communication step (or direct tensor assembly) reconstructs the full-head result. On T3K the chips communicate over Ethernet for this reconstruction.

For Ling's configuration at decode batch=1:

```
Per-chip compute payload: (seq_len=1) × (num_heads) × (head_dim/8=16) = 256 elements
CCL payload for reconstruction: 16 heads × 16 elements × 2 bytes (BF16) × 7 exchanges ≈ 3.6 KB
CCL latency at T3K Ethernet: dominated by synchronization overhead ≈ 3–10 µs [ESTIMATE]
```

The distributed kernel's arithmetic work is 8× less per chip than the non-distributed kernel, but the arithmetic is already negligible. The CCL overhead to reassemble the result adds 3–10 µs. The net effect is that **distributed and non-distributed RoPE have comparable latency at decode batch=1**, with the distributed variant potentially slightly slower due to the CCL synchronization overhead.

This analysis has an important corollary: the reason for choosing the non-distributed variant is not performance — it is **correctness** (the distributed kernel cannot handle `partial_rotary_factor < 1.0`). Even if the distributed kernel were adaptable, switching to it for Ling's configuration would not yield a latency improvement at batch=1.

Table: Non-distributed vs. distributed RoPE latency comparison at decode batch=1 [ESTIMATE]

| Metric | `TTNNRotaryPositionEmbedding` (non-distributed) | `TTNNDistributedRotaryPositionEmbedding` (full `partial_rotary_factor=1.0`) |
|---|---|---|
| Per-chip arithmetic | Full `rotary_dim` computation on all elements | `head_dim / 8` elements per chip |
| CCL communication | None | ≈ 3–10 µs synchronization overhead |
| Kernel dispatch | ≈ 5–10 µs | ≈ 5–10 µs + CCL dispatch |
| Total kernel latency | ≈ 5–15 µs [ESTIMATE] | ≈ 8–20 µs [ESTIMATE] |
| Surrounding DRAM↔L1 transitions (T1+T2) | ≈ 64 µs | ≈ 64 µs (same transition structure) |
| **Total RoPE step cost** | **≈ 70–80 µs** | **≈ 72–84 µs** |

The distributed kernel offers no advantage at batch=1 decode — the dominant cost is the DRAM↔L1 data movement, not the kernel arithmetic. The performance concern from using non-distributed RoPE is therefore not a direct latency penalty; it is an **opportunity cost**: the non-distributed path prevents exploiting mesh parallelism for RoPE, which would matter at batch > 1 or prefill, but not at single-token decode.

## Can Cos/Sin Tables Be Padded to Full `head_dim`?

### The Padding Strategy

The idea is to pad the cos/sin tables from `(1, rotary_dim=64)` to `(1, head_dim=128)`, filling the padded region with values that produce identity behavior:
- Pad `cos` with 1.0 for indices `[rotary_dim : head_dim]`
- Pad `sin` with 0.0 for indices `[rotary_dim : head_dim]`

With this padding, the standard full-head rotary formula:

```
q_rot[i]   = q[i] * cos[i] - q[i + H/2] * sin[i]
q_rot[i+H/2] = q[i] * sin[i] + q[i + H/2] * cos[i]
```

applied to all `i` in `[0, H/2)` = `[0, 64)` would produce:

- For `i` in `[0, rotary_dim/2=32)`: `cos[i]` and `sin[i]` are the actual rotation values → correct rotation applied.
- For `i` in `[32, 64)`: `cos[i] = 1.0`, `sin[i] = 0.0` → formula gives `q_rot[i] = q[i] * 1 - q[i+64] * 0 = q[i]`, and `q_rot[i+64] = q[i] * 0 + q[i+64] * 1 = q[i+64]` → identity pass-through.

This is **mathematically equivalent** to partial-rotary with `rotary_dim=64` **if and only if** `TTNNDistributedRotaryPositionEmbedding` uses a rotation offset of `rotary_dim//2 = 32` (pairing `q[i]` with `q[i+32]` for `i in [0,32)`). However, the full-head distributed kernel may use an offset of `head_dim//2 = 64` instead (pairing `q[i]` with `q[i+64]` for `i in [0,64)`). These are different pairings: if the kernel uses `head_dim//2=64` as its offset, elements `q[0:32]` are paired with `q[64:96]` rather than `q[32:64]`, and the identity padding in positions `[32:64]` and `[96:128]` would be applied to the wrong rotation pairs — producing incorrect results.

> **Verify before implementing:** Confirm that `TTNNDistributedRotaryPositionEmbedding` supports a `rotary_offset` or equivalent parameter that can be set to `rotary_dim//2 = 32` (not `head_dim//2 = 64`). Without this configurable offset, element pairing will be wrong and the padded strategy produces incorrect output. Do not use this approach until this parameter is confirmed to exist and is set correctly.

Padding the cos/sin tables would therefore allow `TTNNDistributedRotaryPositionEmbedding` to be used with full `head_dim=128` coverage while preserving the partial-rotary semantics, **conditional on the kernel supporting the correct rotation offset**.

### Implementation Requirements

To enable this strategy:

1. **Cos/sin table construction** — At model initialization or at each prefill step, generate the cos/sin tables with shape `(seq_len, head_dim)` where:
   - Columns `[0 : rotary_dim]` = standard RoPE frequencies as currently computed
   - Columns `[rotary_dim : head_dim]` = `cos=1.0`, `sin=0.0`

2. **KV cache cos/sin storage** — If cos/sin tables are cached per sequence position, the cached tensors must be re-generated or resized to `head_dim` width.

3. **Kernel invocation** — Pass the padded `(seq_len, head_dim)` cos/sin tables to `TTNNDistributedRotaryPositionEmbedding` instead of the current `(seq_len, rotary_dim)` tables passed to `TTNNRotaryPositionEmbedding`.

4. **Dispatch branch update** — Change the condition from `rotary_dim == head_dim` to always select the distributed kernel (or introduce a new condition that selects the distributed kernel when padded cos/sin tables are provided).

```python
# Modified initialization: pad cos/sin tables to full head_dim
def _build_padded_cos_sin(rotary_dim, head_dim, max_seq_len, base_freq):
    # Standard frequencies for [0 : rotary_dim]
    freqs = 1.0 / (base_freq ** (torch.arange(0, rotary_dim, 2) / rotary_dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)
    cos_partial = torch.cos(freqs)   # shape: (max_seq_len, rotary_dim // 2)
    sin_partial = torch.sin(freqs)   # shape: (max_seq_len, rotary_dim // 2)

    # Pad to head_dim // 2 with identity values
    pad_cols = head_dim // 2 - rotary_dim // 2   # = 32 for Ling
    cos_padded = torch.cat([cos_partial, torch.ones(max_seq_len, pad_cols)], dim=-1)
    sin_padded = torch.cat([sin_partial, torch.zeros(max_seq_len, pad_cols)], dim=-1)

    # Interleave into shape (max_seq_len, head_dim) expected by distributed kernel
    return cos_padded, sin_padded   # each shape: (max_seq_len, head_dim // 2)
```

### Memory Cost of Padding

The padded cos/sin tables are `head_dim` wide instead of `rotary_dim` wide: 2× the storage. For Ling at `head_dim=128`, the padded table stores `head_dim // 2 = 64` elements per position vs. the current 32. At BF16, over a maximum sequence length of `S_max`:

```
Per-table size: S_max × (head_dim // 2) × 2 bytes (BF16)
              = S_max × 64 × 2 = S_max × 128 bytes
Both tables (cos + sin): S_max × 256 bytes
```

For `S_max = 4096`: 4096 × 256 = 1,048,576 bytes ≈ 1 MB. For `S_max = 32768`: 32768 × 256 ≈ 8 MB. This is a negligible increase in DRAM usage and has no measurable impact on decode latency.

### Would This Enable Faster Execution?

As established in the comparison table above, the distributed kernel at batch=1 is not faster than the non-distributed kernel — it is marginally slower due to CCL overhead. The cos/sin padding strategy therefore does **not** improve single-token decode latency on its own.

The strategy's value is for larger batch sizes and prefill:
- At batch > 1, the distributed kernel partitions RoPE work across chips, reducing per-chip arithmetic.
- During prefill with long sequences, the distributed kernel reduces per-chip load by `1/8`.

For Ling's single-token decode use case, the padding approach is a code hygiene improvement (using the more general kernel) rather than a performance optimization.

## Alternative: Post-Distributed-RoPE Slice to Discard Padded Dimensions

An alternative strategy avoids modifying the cos/sin tables. Instead:

1. **Apply distributed RoPE to the full `head_dim`** using actual (non-padded) full-coverage cos/sin tables — which would apply rotary encoding to all 128 elements instead of just 64.
2. **Post-RoPE, apply a mask or slice** that zeroes out or restores the original values for elements `[rotary_dim : head_dim]`, effectively discarding the rotation applied to those elements.

This is not semantically equivalent to partial-rotary, because full-coverage RoPE has already modified elements `[rotary_dim : head_dim]` with real frequencies. A post-hoc slice `q[:, :, :, :rotary_dim]` would restore only the rotary-encoded portion and discard the identity portion, which changes the embedding. A mask-and-restore operation would need to blend the RoPE output with the original un-rotated values:

```python
# Post-embedding blend: restore original values for [rotary_dim : head_dim]
q_rope_full = TTNNDistributedRotaryPositionEmbedding(q_heads, cos_full, sin_full)
# q_rope_full has RoPE applied to all 128 elements

# Restore the original values for elements [64 : 128]
# This requires keeping q_heads in L1 while RoPE executes, then blending
q_output = ttnn.where(
    mask_full_head,       # True for [0:64], False for [64:128]
    q_rope_full,          # use rotated values for [0:64]
    q_heads_original,     # use original values for [64:128]
)
```

This blend operation requires storing both the pre- and post-RoPE tensors in L1 simultaneously, doubling the L1 footprint, and adds a `ttnn.where` kernel call (with its own dispatch overhead of ≈ 5–10 µs [ESTIMATE]).

### Cost Comparison: Post-Slice vs. Non-Distributed Native

Table: Strategy cost comparison for partial-rotary handling [ESTIMATE]

| Strategy | Additional kernels | CCL overhead | Arithmetic correctness | Net latency vs. baseline |
|---|---|---|---|---|
| Baseline: non-distributed `TTNNRotaryPositionEmbedding` | None | None | Correct natively | 0 (reference) |
| Padded cos/sin + distributed kernel | None (pad at init) | ≈ 3–10 µs synchronization | Correct (identity pad) | ≈ +3–10 µs (overhead from CCL) |
| Full RoPE + post-blend `ttnn.where` | 1 `ttnn.where` call + L1 for original copy | ≈ 3–10 µs (distributed kernel) | Correct with blend | ≈ +8–20 µs (CCL + where kernel) |
| Full RoPE + post-slice `ttnn.slice` | 1 `ttnn.slice` + `ttnn.concat` back | None if non-distributed | Incorrect (elements [64:128] are rotated) | Not viable |

The post-blend approach is strictly worse than non-distributed RoPE at batch=1: it adds both the CCL overhead of the distributed kernel and the `ttnn.where` blend overhead, while providing no latency reduction.

The padded cos/sin approach incurs CCL overhead, making it also marginally slower than non-distributed RoPE at batch=1.

**Conclusion:** For single-token decode, the non-distributed `TTNNRotaryPositionEmbedding` is the correct and performant choice. The alternatives are appropriate only for prefill or multi-token batch decode.

## Recommendation and Implementation Sketch

### Recommendation

For Ling's `partial_rotary_factor=0.5` at decode batch=1 on T3K:

1. **Keep `TTNNRotaryPositionEmbedding` (non-distributed)** for single-token decode. Switching to the distributed kernel does not improve latency and adds CCL overhead.

2. **Implement the padded cos/sin strategy for prefill and batched decode** — this enables a unified code path using `TTNNDistributedRotaryPositionEmbedding` for all sequence lengths, exploiting mesh parallelism during prefill without affecting decode latency.

3. **Focus RoPE-related optimization effort on the surrounding transitions** (T1a, T1b, T2a, T2b) rather than the kernel itself. The kernel is not the bottleneck; the DRAM↔L1 data movement is. The highest-impact change for the full RoPE+norm path is the Priority 1 fusion described in `qk_norm_latency.md`.

### Implementation Sketch: Padded Cos/Sin for Prefill Unified Path

```python
# In TTNNBailingMoEAttention.__init__ or model preparation:

def _build_rope_tables(self, max_seq_len: int, device_mesh):
    """Build cos/sin tables padded to head_dim for distributed RoPE."""
    rotary_dim = int(self.partial_rotary_factor * self.head_dim)  # 64
    head_dim   = self.head_dim                                     # 128

    # Standard frequencies for the active rotary dimensions
    inv_freq = 1.0 / (self.rope_theta ** (
        torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim
    ))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)   # (max_seq_len, rotary_dim // 2)

    # Compute partial cos/sin
    cos_p = torch.cos(freqs)   # (max_seq_len, rotary_dim // 2)
    sin_p = torch.sin(freqs)   # (max_seq_len, rotary_dim // 2)

    # Pad to head_dim // 2 with identity: cos=1, sin=0
    pad = head_dim // 2 - rotary_dim // 2   # 32 padding columns
    cos_full = torch.cat([cos_p, torch.ones(max_seq_len, pad)], dim=-1)  # (max_seq_len, 64)
    sin_full = torch.cat([sin_p, torch.zeros(max_seq_len, pad)], dim=-1) # (max_seq_len, 64)

    # Interleave format expected by distributed kernel: (1, 1, max_seq_len, head_dim // 2)
    cos_table = cos_full.unsqueeze(0).unsqueeze(0).to(torch.bfloat16)
    sin_table = sin_full.unsqueeze(0).unsqueeze(0).to(torch.bfloat16)

    # Transfer to device mesh as ReplicateTensorToMesh
    self.cos_cached = ttnn.from_torch(
        cos_table, dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device_mesh),
        device=device_mesh,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    self.sin_cached = ttnn.from_torch(
        sin_table, dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device_mesh),
        device=device_mesh,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _apply_rope(self, q: ttnn.Tensor, k: ttnn.Tensor,
                position_ids: torch.Tensor, mode: str) -> tuple:
    """
    Apply RoPE using the unified padded-table path.
    mode='decode': single token, use non-distributed kernel (no CCL gain at batch=1).
    mode='prefill': long sequence, use distributed kernel to exploit mesh parallelism.
    """
    cos_slice = self.cos_cached[:, :, position_ids, :]   # (1, 1, seq_len, head_dim // 2)
    sin_slice = self.sin_cached[:, :, position_ids, :]   # (1, 1, seq_len, head_dim // 2)

    if mode == 'decode':
        # Non-distributed: correct and marginally faster at batch=1.
        # The kernel pairs q[i] with q[i + rotary_dim//2] for i in [0, rotary_dim//2),
        # so it needs exactly rotary_dim//2 cos/sin values (one per rotation pair),
        # not rotary_dim values. The padded table has head_dim//2=64 columns total;
        # we slice [:rotary_dim//2] = [:32] to pass only the active rotation values.
        q_rope = TTNNRotaryPositionEmbedding(
            q, cos_slice[:, :, :, :self.rotary_dim // 2],
            sin_slice[:, :, :, :self.rotary_dim // 2],
            rotary_dim=self.rotary_dim,
        )
        k_rope = TTNNRotaryPositionEmbedding(
            k, cos_slice[:, :, :, :self.rotary_dim // 2],
            sin_slice[:, :, :, :self.rotary_dim // 2],
            rotary_dim=self.rotary_dim,
        )
    else:
        # Distributed: exploits mesh for prefill; padded tables ensure
        # elements [rotary_dim : head_dim] receive identity transformation
        q_rope = TTNNDistributedRotaryPositionEmbedding(q, cos_slice, sin_slice)
        k_rope = TTNNDistributedRotaryPositionEmbedding(k, cos_slice, sin_slice)

    return q_rope, k_rope
```

This implementation sketch isolates the decode vs. prefill dispatch to a single method, making it straightforward to validate that both paths produce numerically identical output (the padded cos/sin tables applied by the distributed kernel are equivalent to the partial-rotary tables applied by the non-distributed kernel). Numerical equivalence can be verified with a unit test comparing the two paths on a fixed Q/K pair.

### Expected Outcome

- **Decode latency**: unchanged from the baseline non-distributed path. The dominant costs remain the DRAM↔L1 transitions (T1, T2) and the QK norm round-trip. These are addressed by Priority 1 in Chapter 4.
- **Prefill throughput**: improved for long sequences by exploiting mesh parallelism in the distributed RoPE kernel.
- **Code quality**: unified cos/sin table management; single kernel variant used across decode/prefill once batch > 1 is considered.

---

**Next:** [Chapter 7 — Profiling and Bottleneck Identification](../ch7_profiling_and_bottleneck_identification/index.md)
