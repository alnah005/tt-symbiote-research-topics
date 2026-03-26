# Distributed Alternative for QK Normalization

## Why `TTNNDistributedRMSNorm` Does Not Apply to the Current Decode Path

`TTNNDistributedRMSNorm` (lines 100–151 of `normalization.py`) is designed for a specific scenario: the tensor being normalized has its reduction dimension (typically hidden_size) **split across devices**, and the partial statistics must be gathered before the final normalization step can complete. Its forward pass (lines 127–151) is:

1. Each device holds `input[:, :, hidden_size/N]` — a col-shard of the full hidden state (3-D input is first unsqueezed to 4-D at lines 129–130).
2. `ttnn.rms_norm_pre_all_gather` computes per-device partial sum-of-squares over the local shard.
3. `ttnn.experimental.all_gather_async` gathers the partial statistics from all N=8 devices.
4. `ttnn.rms_norm_post_all_gather` uses the full global statistics to normalize the local shard.

In the current `TTNNBailingMoEAttention` decode path, by the time `_apply_qk_norm` is called (line 2659 of `attention.py`), Q and K have already been through `_maybe_all_gather`. Each device holds the **complete** Q tensor `[1, B, 16, 128]` and the complete K tensor `[1, B, 4, 128]`. There is nothing sharded across devices at this point.

Applying `TTNNDistributedRMSNorm` here would require:
1. Artificially re-splitting the already-gathered Q and K across devices.
2. Running `rms_norm_pre_all_gather` on the split shards.
3. Running another `all_gather_async` to gather the statistics.
4. Running `rms_norm_post_all_gather`.

This is strictly worse than the current approach: it adds a shard operation and a full all_gather communication round that serves no purpose because the data was already fully replicated. `TTNNDistributedRMSNorm` applies to the MoE feedforward hidden state (which is col-sharded before QK projection) — not to Q and K after they have been gathered.

---

## The Pre-All-Gather Q Norm Alternative

The constructive alternative is to move the Q norm **before** the all-gather, while Q is still in the col-sharded layout that `TTNNLinearIColShardedWRowSharded` produces via reduce-scatter (Chapter 2). This eliminates the DRAM→L1 transition for Q entirely by performing the norm in-place on the pre-gather shards before they are communicated.

### Why this is feasible for Q

After the Q reduce-scatter, each device holds a col-shard of Q along the last dimension. The shape is:

```
Q per device (pre-all-gather): [B, 1, (H/N) * D] = [B, 1, 256]
```

With H=16, N=8, D=128: each device holds 2 complete Q heads worth of data (H/N = 2), giving 2 × D = 256 elements per token per device.

When the Q norm is applied **per head**, the reduction is over D=128 within each head vector. Each of the H/N=2 heads on a given device is entirely local to that device — no elements from those heads reside on any other device. Therefore:

- The norm can run independently on each device.
- No cross-device communication is needed.
- No all-gather is needed before or after the norm.
- The norm input shape per device would be `[B * (H/N), D]` = `[B*2, 128]` after a reshape.

This is an entirely intra-device operation. The regular (non-distributed) `ttnn.rms_norm` can be used, operating on the col-sharded tensor on each device independently.

The benefit: if the norm runs before the all-gather, Q is still in its pre-gather layout when normalized. The subsequent Q all-gather then communicates the already-normalized Q heads. The DRAM→L1 transition at lines 2655–2656 of `attention.py` (which currently moves Q to L1 for the reshape) becomes unnecessary for Q: the norm can happen on Q in its post-reduce-scatter layout before the all-gather is issued.

At B=32, this saves a 131,072-byte DRAM→L1 copy per decode step per device.

### Why this is NOT directly feasible for K

K is produced by `TTNNLinearIReplicatedWColSharded` (line 2375 of `attention.py`), which projects the replicated hidden state to a col-sharded output. The col-sharding is over the output (KV) dimension, not the head dimension directly:

```
K per device (pre-all-gather): [B, 1, Hkv/N * D] = [B, 1, 64]
```

With Hkv=4, N=8, D=128: Hkv/N = 0.5. **Each device holds only half of a KV head** — 64 elements out of the 128-element D dimension. This half-head slice is not a valid input for per-head RMSNorm over D=128, because the reduction dimension (D=128) is split across two devices.

Applying `ttnn.rms_norm` on these 64-element slices would compute the norm over the half-head vector, not the full D=128 head vector. This would produce different output from the correct per-head norm and break the logit scale invariant that QK norm is designed to enforce.

To norm K correctly before the all-gather, a distributed norm would be required — specifically, one that gathers partial statistics across the pair of devices holding the two halves of each KV head. Based on the Hkv=4, N=8 parameters (Hkv/N = 0.5), each KV head's D=128 elements are inferred to span exactly 2 devices in the col-sharded layout. This is the `TTNNDistributedRMSNorm` pattern, but applied to a 2-device sub-group rather than the full 8-device mesh. No existing TTNN infrastructure supports sub-group distributed norms with a 2-device topology in this codebase.

**Conclusion for K:** the norm must remain post-all-gather, where each device holds the full `[1, B, Hkv, D]` K tensor and the reduction over D=128 is intra-device. The DRAM→L1 transition for K (32,768 bytes at B=32) cannot be eliminated with the pre-all-gather approach.

### Impact summary of pre-all-gather Q norm

| | Current approach | Pre-all-gather Q norm |
|---|---|---|
| Q DRAM→L1 transition | 131,072 bytes per step (B=32) | Eliminated |
| K DRAM→L1 transition | 32,768 bytes per step (B=32) | Still required |
| Q all-gather | Before norm | After norm (moved) |
| K all-gather | Before norm | Before norm (unchanged) |
| Cross-device communication for Q norm | None | None |
| Cross-device communication for K norm | None | None |
| Correctness risk | None (current baseline) | Low — Q norm semantics preserved because each device's 2 heads are complete |
| Host touches added | None | None |
| Code complexity change | None (current baseline) | Requires splitting norm call from `_apply_qk_norm` into separate pre-gather step for Q |
| Combined with inline K norm (no module dispatch) | Not applied | Optional independent improvement |

The pre-all-gather Q norm saves the larger of the two transitions (131 KB vs 32 KB) but requires restructuring the call sequence in `_forward_decode_paged`: the Q norm must be inserted between the Q reduce-scatter output and the `_maybe_all_gather(query_states)` call, and the Q `ttnn.to_memory_config(... L1_MEMORY_CONFIG)` at line 2656 must be removed. The K path is unchanged.

One additional consideration: the Q norm weight must be applied using the per-head RMSNorm convention. Because the pre-gather Q shard holds 2 complete heads (each of D=128 elements), the norm applies over the D=128 last dimension — which is what the weight shape `[32, 128]` provides (expanded to 32 rows for tile compatibility; see `normalization.py` line 85). The weight's D=128 dimension matches the input's D=128 head dimension, so no change to the weight shape is needed. The pre-gather input would be reshaped to `[B*(H/N), D]` = `[B*2, 128]` before the norm; validating that `ttnn.rms_norm` accepts a `[32, 128]` weight for a variable-row input of width 128 should be confirmed during implementation.

---

## Fusion Opportunity: QK Norm and the Projection Outputs

A more aggressive optimization would fuse the QK norm directly with the output of the Q or K projection, eliminating the norm as a separate dispatched op.

### How `TTNNGlm4MoeLiteAttention` handles normalization

`TTNNGlm4MoeLiteAttention` (starting at line 1493 of `attention.py`) uses a different architecture (MLA — Multi-head Latent Attention) and does not have a direct QK norm equivalent. Instead, it normalizes the compressed KV latent (`kv_a_layernorm`) after the first KV projection and before the second KV projection. This normalization is inline in `_project_qkv` (line 1689 of `attention.py`):

```python
k_pass_flat = ttnn.rms_norm(k_pass_flat, weight=self._kv_a_ln_weight, epsilon=self._kv_a_ln_eps)
```

This is a **direct `ttnn.rms_norm` call** with the weight pre-loaded as a device tensor (`self._kv_a_ln_weight`, populated at line 1582 of `attention.py`). There is no `TTNNRMSNorm` module dispatch involved. The weight is stored as a `ReplicateTensorToMesh` tensor in DRAM and accessed directly.

Importantly, `TTNNGlm4MoeLiteAttention.from_torch` (line 1540 of `attention.py`) uses:
```python
NormCls = TTNNDistributedRMSNorm if distributed else TTNNRMSNorm
```
for `q_a_layernorm` and `kv_a_layernorm` module objects. The `q_a_layernorm` module is called in the forward path (at line 1660, applied to the Q LoRA latent). The `kv_a_layernorm` module object is constructed but its forward is not called in the decode path — the KV latent normalization runs entirely via the inline `ttnn.rms_norm` at line 1689 using `self._kv_a_ln_weight`. The inline `ttnn.rms_norm` at line 1689 is thus the normalization that parallels the `TTNNBailingMoEAttention` post-all-gather QK norm.

The key difference is that `TTNNGlm4MoeLiteAttention` calls `ttnn.rms_norm` directly with a pre-loaded device weight (`self._kv_a_ln_weight`) rather than dispatching through a `TTNNRMSNorm` module object. This eliminates the Python overhead of the module's `forward` method call, the `hasattr` checks, and the conditional typecast guards.

### Applying the same pattern to `TTNNBailingMoEAttention`

For `TTNNBailingMoEAttention`, a fusion-style approach would:

1. Store `query_layernorm.tt_weight` and `key_layernorm.tt_weight` as direct device tensors on the `TTNNBailingMoEAttention` object (as `move_weights_to_device_impl` already does via the child `TTNNRMSNorm` modules, but exposed at the parent level).
2. Replace the `self.query_layernorm(q_reshaped)` and `self.key_layernorm(k_reshaped)` calls in `_apply_qk_norm` with direct `ttnn.rms_norm(q_reshaped, weight=self._q_norm_weight, epsilon=...)` calls.
3. Remove the `hasattr` and `ttnn.typecast` guards that exist to handle the module fallback path.

This does not eliminate the DRAM→L1 transition (that requires the pre-all-gather approach above) but does reduce Python dispatch overhead per norm call. For a fast decode loop running thousands of steps, the reduction in Python overhead from removing module dispatch and conditional checks is measurable.

In practice the combined optimization would be: move Q norm before all-gather (eliminates Q DRAM→L1 transition) and inline K norm as a direct `ttnn.rms_norm` call post-all-gather (removes module dispatch overhead for K).

The pre-all-gather Q norm is a targeted, low-risk optimization that removes the largest of the two QK-norm-induced DRAM→L1 transitions identified in Chapter 3. The K norm transition (32,768 bytes at B=32) is structurally unavoidable given the current Hkv=4, N=8 configuration where KV heads do not divide evenly across devices, and should be accepted as a fixed cost.

---

**Next:** [Chapter 6 — Math Fidelity, Compute Kernel Settings, and SDPA Configuration](../chapter_06_math_fidelity/index.md)
