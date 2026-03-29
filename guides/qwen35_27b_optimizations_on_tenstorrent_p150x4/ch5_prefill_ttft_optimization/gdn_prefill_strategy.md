# GDN Prefill Strategy: Batched Projections with Sequential Recurrence

GDN layers present a fundamental challenge for prefill optimization: unlike attention layers where all tokens can be processed in parallel via flash SDPA, the DeltaNet recurrence has a strict sequential dependency. Each token's recurrence state depends on the previous token's output -- the state update `state += outer(k_col, delta)` feeds forward into the next token's readout `output = q @ state`. There is no way to parallelize the recurrence across tokens without changing the algorithm.

The GDN prefill strategy in `TtGatedDeltaNet.forward_prefill()` (`gdn.py`, lines 578-726) addresses this with a hybrid approach: batch everything that can be batched (projections), then iterate per-token only for the parts that must be sequential (conv1d + recurrence).

## B=1 Prefill State Initialization

Before the first prefill token, `_init_prefill_states()` (`gdn.py`, lines 501-520) creates a separate set of B=1 states that are independent of the B=32 decode states:

```python
# Conv states: 4 shift register slots, each [1, 1, qkv_dim_tp]
self._prefill_conv_states = [
    _to_mesh(torch.zeros(1, 1, self.qkv_dim_tp))
    for _ in range(self.conv_kernel_size)     # conv_kernel_size = 4
]

# Recurrence state: [1 * Nv_TP, Dk, Dv] = [12, 128, 128] per device
self._prefill_rec_states = _to_mesh(
    torch.zeros(1 * self.Nv_TP, self.Dk, self.Dv)
)

# Fused kernel output buffer: [Nv_TP, 1, Dv] = [12, 1, 128] per device
self._prefill_fused_output = _to_mesh(
    torch.zeros(1 * self.Nv_TP, 1, self.Dv)
)
```

These B=1 states are allocated via `ttnn.ReplicateTensorToMesh` -- each device gets an identical copy, all initialized to zero. The separation from decode states is important: the B=32 decode states hold `[32*12, 128, 128]` per device (12 MB), while the B=1 prefill states hold `[12, 128, 128]` per device (approximately 393 KB). This smaller footprint means prefill state operations are faster and use less DRAM bandwidth.

## The Hybrid Prefill Pipeline

The `forward_prefill()` method proceeds in three phases:

### Phase 1: Batched Projections

As detailed in [`batched_projections.md`](./batched_projections.md), the QKVZ and AB projections are computed once for the full `[1, 1, seq_len, dim]` input:

```
qkvz_all: [1, 1, seq_len, qkvz_dim_tp = 4096]  -- Q, K, V, Z for all tokens
ab_all:   [1, 1, seq_len, Nv_TP * 2   = 24]     -- a, b gates for all tokens
```

### Phase 2: Per-Token Sequential Loop

The main loop iterates over `seq_len` tokens (lines 636-700). For each token `t`:

**Step 1 -- Token slicing.** Extract token `t` from the precomputed projection tensors:

```python
qkvz_t = ttnn.slice(qkvz_all, (0, 0, t, 0), (1, 1, t+1, qkvz_dim_tp))
qkvz_t = ttnn.reshape(qkvz_t, (1, B_pf, qkvz_dim_tp))  # B_pf = 1

ab_t = ttnn.slice(ab_all, (0, 0, t, 0), (1, 1, t+1, Nv_TP * 2))
ab_t = ttnn.reshape(ab_t, (1, B_pf, Nv_TP * 2))
```

The QKVZ tensor is then split into QKV and Z components:

```python
qkv_tt = ttnn.slice(qkvz_t, (0, 0, 0), (1, B_pf, qkv_dim_tp))
z_tt   = ttnn.slice(qkvz_t, (0, 0, qkv_dim_tp), (1, B_pf, qkvz_dim_tp))
```

Similarly, AB is split into separate `a_tt` and `b_tt` tensors.

**Step 2 -- Conv1d shift register.** The same 4-tap causal conv1d used in decode (see Chapter 3) runs on the B=1 prefill conv states:

```python
ttnn.copy(states[1], states[0])
ttnn.copy(states[2], states[1])
ttnn.copy(states[3], states[2])
ttnn.copy(qkv_tt, states[3])

conv_acc = ttnn.multiply(states[0], tw["conv_taps"][0])
for j in range(1, self.conv_kernel_size):
    conv_acc = ttnn.mac(states[j], tw["conv_taps"][j], conv_acc)
conv_out = ttnn.silu(conv_acc)
```

Each conv state is `[1, 1, qkv_dim_tp]` -- the same shift register pattern as decode, but with B=1 instead of B=32.

**Step 3 -- Fused recurrence kernel.** The same `gdn_full_fused_inplace` kernel from Chapter 4 processes the token, but with `num_pairs = B_pf * Nv_TP = 1 * 12 = 12` instead of the decode-time `num_pairs = 32 * 12 = 384`:

```python
gdn_full_fused_inplace(
    conv_out, a_tt, b_tt,
    self.neg_exp_A, tw["dt_bias"], tw["norm_w"],
    self.scale_tt, self.rms_scale_tt, self.rms_eps_tt,
    self._prefill_rec_states, self._prefill_fused_output,
    num_pairs=num_pairs_pf,           # 12
    num_cores=min(96, num_pairs_pf),  # 12
    Nv_TP=Nv_TP, Nk_TP=Nk_TP, repeat_factor=repeat_factor,
    key_dim_tp=key_dim_tp,
)
```

The kernel reads from and writes to `self._prefill_rec_states`, updating the recurrence state in-place. The output is written to `self._prefill_fused_output`.

**Step 4 -- Post-kernel processing.** The fused kernel output goes through RMS norm and SiLU gating with Z:

```python
out_r = ttnn.reshape(self._prefill_fused_output, (B_pf, Nv_TP, Dv))
out_n = ttnn.rms_norm(out_r, weight=tw["norm_w"], epsilon=1e-6)
out_f = ttnn.reshape(out_n, (1, B_pf, self.value_dim_tp))

z_act = ttnn.silu(z_tt)
gated = ttnn.multiply(out_f, z_act)
```

The gated output for each token is appended to `gated_outputs`.

### Phase 3: Batched Output Projection

After the loop completes, all per-token outputs are concatenated and projected:

```python
gated_seq = ttnn.concat(gated_outputs, dim=1)  # [1, seq_len, value_dim_tp]
gated_seq = ttnn.reshape(gated_seq, (1, 1, seq_len, self.value_dim_tp))

out_partial = ttnn.linear(
    gated_seq, tw["out"],
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    program_config=out_progcfg,
    compute_kernel_config=self.compute_cfg,
)
```

This single 2D matmul replaces `seq_len` individual output projection dispatches. The result goes through an all-reduce across TP devices to produce the final `[1, 1, seq_len, dim]` output.

## Memory Management

The per-token loop is careful about deallocation. Within each iteration:

- `qkvz_t` and `ab_t` (sliced from the precomputed tensors) are deallocated after splitting
- `conv_out`, `a_tt`, and `b_tt` are deallocated after the fused kernel call
- Intermediate tensors (`out_r`, `out_n`, `z_act`, `out_f`) are deallocated after use
- Only the final `gated` output for each token is retained in the `gated_outputs` list

The precomputed `qkvz_all` and `ab_all` tensors persist throughout the loop and are deallocated after all tokens are processed (line 702-703). This means DRAM holds both the full-sequence projection results and the growing list of per-token outputs simultaneously -- a tradeoff of memory for the dispatch overhead savings.

## Why Not Parallelize the Recurrence?

The DeltaNet recurrence has a true data dependency: token `t`'s state is a function of token `t-1`'s state. Parallel scan algorithms exist for certain recurrence forms, but they require associative operations and typically double the compute. The current sequential approach is simpler and, combined with batched projections, already achieves the 5.3x speedup target. Chunked parallel recurrence (processing groups of tokens with inter-chunk sequential updates) remains a potential future optimization noted in Chapter 7.

---

**Previous:** [`batched_projections.md`](./batched_projections.md) | **Next:** [`state_replication.md`](./state_replication.md)
