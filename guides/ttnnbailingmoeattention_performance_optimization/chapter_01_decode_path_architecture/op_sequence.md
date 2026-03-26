# Op Sequence — `_forward_decode_paged` (Lines 2610–2799)

This document annotates every operation in `TTNNBailingMoEAttention._forward_decode_paged`
in execution order. For each step the following information is given:

- **What it does** — the operation performed
- **Why it is there** — the functional or layout requirement that demands this step
- **Input** — tensor name(s), shape, layout, memory location entering the op
- **Output** — tensor name(s), shape, layout, memory location leaving the op
- **Host touch** — whether the operation crosses the PCIe bus to/from the host CPU

Source file: `models/experimental/tt_symbiote/modules/attention.py`

---

## Step 1 — `ttnn.to_layout` → TILE_LAYOUT (conditional)

**Lines:** 2621–2622

```python
if hidden_states.layout != ttnn.TILE_LAYOUT:
    hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
```

**What it does:** Ensures that `hidden_states` is in TILE_LAYOUT before any linear
projection. TTNN linear kernels require tile-aligned input.

**Why it is there:** The Bailing MoE model's MoE routing layer can produce output in
ROW_MAJOR_LAYOUT under certain paths. The conversion is conditional — if the tensor is
already in TILE_LAYOUT (the common case during steady-state decoding) this step is a
no-op.

**Input:** `hidden_states`, shape `[B, 1, d_model]` = `[B, 1, 2048]`, ROW_MAJOR or
TILE, DRAM, col-sharded across N=8 devices (last dim sharded: each device holds
`[B, 1, 256]`).

**Output:** `hidden_states`, shape `[B, 1, 2048]`, TILE_LAYOUT, DRAM, col-sharded.

**Host touch:** No.

**Performance note:** This step is a no-op in the steady-state decode loop. It adds a
Python-level branch and a layout check, but no device work if the condition is false.

---

## Step 2 — `q_proj(hidden_states)` via `TTNNLinearIColShardedWRowSharded`

**Lines:** 2624

```python
query_states = self.q_proj(hidden_states)
```

**What it does:** Projects `hidden_states` from `d_model=2048` to the full Q space
`H * D = 16 * 128 = 2048` using a tensor-parallel matmul with reduce-scatter output.

**Why it is there:** `TTNNLinearIColShardedWRowSharded` (defined in `linear.py`, lines
126–176) performs `ttnn.linear` with the weight matrix sharded on its row dimension
(each of N=8 devices holds `[d_model/N, q_dim]` = `[256, 2048]` of the weight). The
input is also col-sharded on the last dimension (`[B, 1, 256]` per device). The local
matmul produces a partial sum of shape `[B, 1, 2048]` per device; these are reduced
and scattered via `ttnn.experimental.reduce_scatter_minimal_async` on `dim=3` with
`cluster_axis=1`, producing a col-sharded result where each device holds `[B, 1, 256]`
of the Q output.

**Input:** `hidden_states`, `[B, 1, 2048]` logical, col-sharded (each device `[B, 1, 256]`),
TILE_LAYOUT, DRAM.

**Output:** `query_states`, `[B, 1, 2048]` logical, col-sharded (each device `[B, 1, 256]`),
TILE_LAYOUT, DRAM.

**Host touch:** No. The reduce-scatter is an async device collective.

**Performance note:** The reduce-scatter output here is reconstituted by `_maybe_all_gather` at step 5a; see step 5a for the round-trip analysis.

---

## Step 3 — `ttnn.all_gather(hidden_states, dim=-1, num_links=1)`

**Lines:** 2626

```python
hidden_states_replicated = ttnn.all_gather(hidden_states, dim=-1, num_links=1)
```

**What it does:** All-gathers the col-sharded `hidden_states` across all N=8 devices,
producing a fully-replicated copy of the complete `[B, 1, 2048]` hidden state on each
device.

**Why it is there:** `TTNNLinearIReplicatedWColSharded` (used for K and V projections)
stores its weights sharded on the output (column) dimension but expects a **replicated**
input. The hidden states arriving from the MoE layer are col-sharded after the MoE
expert FFN reduce-scatter, so they must be gathered before `k_proj` and `v_proj` can be
called.

**Input:** `hidden_states`, `[B, 1, 2048]` logical, col-sharded, TILE_LAYOUT, DRAM.

**Output:** `hidden_states_replicated`, `[B, 1, 2048]`, fully replicated on all 8
devices, TILE_LAYOUT, DRAM.

**Host touch:** No. Synchronous ring collective on device; this call blocks until
complete before the next op is dispatched.

**Performance note:** This all-gather transfers `B * 1 * 2048 * 2 bytes = 4096 * B`
bytes total across all links. At batch=1 this is 4 KiB — small, but the synchronous
`ttnn.all_gather` introduces a synchronization barrier that prevents overlap with any
prior device work. Chapter 2 analyzes whether this gather can be merged with the Q
gather from step 5a.

---

## Step 4a — `k_proj(hidden_states_replicated)` via `TTNNLinearIReplicatedWColSharded`

**Lines:** 2627

```python
key_states = self.k_proj(hidden_states_replicated)
```

**What it does:** Projects the replicated `hidden_states` from `d_model=2048` to the
K space `Hkv * D = 4 * 128 = 512`, with weights sharded on the column (output)
dimension. Each device computes a local matmul and retains `[B, 1, 512/N]` = `[B, 1, 64]`
of the K output.

**Why it is there:** K (and V) use fewer heads than Q (`Hkv=4` vs `H=16`), so K
projection has output dimension 512. Using `TTNNLinearIReplicatedWColSharded` means each
device performs a full `[B, 1, 2048] × [2048, 64]` matmul with no inter-device
communication — the communication burden is placed on the subsequent `_maybe_all_gather`
instead of on the projection.

**Input:** `hidden_states_replicated`, `[B, 1, 2048]`, replicated, TILE_LAYOUT, DRAM.

**Output:** `key_states`, `[B, 1, 512]` logical, col-sharded (each device `[B, 1, 64]`),
TILE_LAYOUT, DRAM.

**Host touch:** No.

---

## Step 4b — `v_proj(hidden_states_replicated)` via `TTNNLinearIReplicatedWColSharded`

**Lines:** 2628

```python
value_states = self.v_proj(hidden_states_replicated)
```

**What it does:** Identical to step 4a but produces V. Shape `[B, 1, 512]` logical,
col-sharded.

**Input/Output:** Same as step 4a, for V.

**Host touch:** No.

**Note:** After step 4b, `hidden_states_replicated` is deallocated:
```python
ttnn.deallocate(hidden_states_replicated)  # line 2629
```
This frees the replicated copy from DRAM immediately to reclaim memory.

---

## Step 5a — `_maybe_all_gather(query_states)`

**Lines:** 2631

```python
query_states = self._maybe_all_gather(query_states)
```

**What it does:** All-gathers the col-sharded Q output from step 2 to produce a
fully-replicated Q tensor on all 8 devices. The helper also typecasts to bfloat16 if
needed (lines 2284–2286 of `attention.py`).

**Why it is there:** The paged attention kernel and RoPE kernel require each device to
hold the full Q tensor for the heads it will process. After reduce-scatter, each device
only has `H/N = 2` heads worth of Q data.

**Input:** `query_states`, `[B, 1, 2048]` logical, col-sharded, TILE_LAYOUT, DRAM.

**Output:** `query_states`, `[B, 1, 2048]`, replicated, TILE_LAYOUT, DRAM, bfloat16.

**Host touch:** No.

**Performance note:** This all-gather transfers the same data volume as step 3 (`4096 * B`
bytes for Q), immediately after the reduce-scatter that created the col-sharded Q. The
reduce-scatter in step 2 + all-gather in step 5a is a communication round-trip that
results in no net change to the data distribution relative to what was available before
step 2. This is the primary structural inefficiency analyzed in Chapter 2.

---

## Step 5b — `_maybe_all_gather(key_states)`

**Lines:** 2632

```python
key_states = self._maybe_all_gather(key_states)
```

**What it does:** All-gathers the col-sharded K output from step 4a. Each device
receives the complete `[B, 1, 512]` K tensor.

**Input:** `key_states`, `[B, 1, 512]` logical, col-sharded, TILE_LAYOUT, DRAM.

**Output:** `key_states`, `[B, 1, 512]`, replicated, TILE_LAYOUT, DRAM, bfloat16.

**Host touch:** No.

---

## Step 5c — `_maybe_all_gather(value_states)`

**Lines:** 2633

```python
value_states = self._maybe_all_gather(value_states)
```

**What it does:** All-gathers V, same pattern as step 5b.

**Input:** `value_states`, `[B, 1, 512]` logical, col-sharded, TILE_LAYOUT, DRAM.

**Output:** `value_states`, `[B, 1, 512]`, replicated, TILE_LAYOUT, DRAM, bfloat16.

**Host touch:** No.

**Note:** Steps 5a, 5b, and 5c are three separate synchronous `ttnn.all_gather` calls,
each introducing a synchronization barrier. Combined with step 3, there are four
all-gather barriers in the projection phase alone. Chapter 2 provides the authoritative
comparison of this count against `TTNNQwen3FullAttention` and what reduction is possible.

---

## Step 6 — `ttnn.reshape` Q/K/V to `[1, B, H, D]` (S B H D decode format)

**Lines:** 2644–2646

```python
query_states = ttnn.reshape(query_states, (1, batch_size, self.num_heads, self.head_dim))
key_states   = ttnn.reshape(key_states,   (1, batch_size, self.num_kv_heads, self.head_dim))
value_states = ttnn.reshape(value_states, (1, batch_size, self.num_kv_heads, self.head_dim))
```

**What it does:** Reinterprets the flat projection output as the four-dimensional S B H D
decode layout. For Q: `[B, 1, 2048]` → `[1, B, 16, 128]`. For K/V: `[B, 1, 512]` →
`[1, B, 4, 128]`.

**Why it is there:** All downstream kernels — `rotary_embedding_llama` (decode mode),
`paged_update_on_device`, `paged_sdpa_decode`, and `nlp_concat_heads_decode` — expect
tensors in the `[seq=1, batch, heads, head_dim]` layout. The comment at line 2642 notes
that this reshape directly avoids the `concat + _to_replicated + nlp_create_qkv_heads_decode`
round-trip used in other attention implementations.

**Input:** `query_states` `[B, 1, 2048]`, `key_states` `[B, 1, 512]`, `value_states`
`[B, 1, 512]`, TILE_LAYOUT, DRAM, replicated.

**Output:** `query_states` `[1, B, 16, 128]`, `key_states` `[1, B, 4, 128]`,
`value_states` `[1, B, 4, 128]`, TILE_LAYOUT, DRAM, replicated.

**Host touch:** No. TTNN reshape is a metadata-only operation when the data layout is
compatible; no data movement occurs.

---

## Step 7 — `ttnn.typecast` Q/K/V to bfloat16 (conditional)

**Lines:** 2648–2653

```python
if query_states.dtype != ttnn.bfloat16:
    query_states = ttnn.typecast(query_states, ttnn.bfloat16)
if key_states.dtype != ttnn.bfloat16:
    key_states = ttnn.typecast(key_states, ttnn.bfloat16)
if value_states.dtype != ttnn.bfloat16:
    value_states = ttnn.typecast(value_states, ttnn.bfloat16)
```

**What it does:** Ensures Q, K, and V are in bfloat16 before QK norm and RoPE. The
`_maybe_all_gather` in step 5 already performs a typecast if needed (lines 2284–2286),
so these guards are typically no-ops.

**Why it is there:** Defensive guard — `ttnn.rms_norm`, `rotary_embedding_llama`, and
`paged_sdpa_decode` all require bfloat16 input; if any upstream path produces a
different dtype these casts prevent silent failures.

**Input/Output:** Same shapes as step 6; dtype normalized to bfloat16.

**Host touch:** No.

---

## Step 8a/8b — `ttnn.to_memory_config(query_states / key_states, L1_MEMORY_CONFIG)`

**Lines:** 2656–2657

```python
query_states = ttnn.to_memory_config(query_states, ttnn.L1_MEMORY_CONFIG)
key_states   = ttnn.to_memory_config(key_states,   ttnn.L1_MEMORY_CONFIG)
```

**What it does:** Copies Q and K from DRAM interleaved into L1 interleaved memory.
V is **not** moved to L1 here — it goes directly to HEIGHT_SHARDED at step 16b.

**Why it is there:** `ttnn.reshape` (used inside `_apply_qk_norm`) does not operate on
sharded tensors. The `_apply_qk_norm` function reshapes Q from `[1, B, 16, 128]` to
`[B*16, 128]` and K from `[1, B, 4, 128]` to `[B*4, 128]` before calling
`ttnn.rms_norm`. Moving to `L1_MEMORY_CONFIG` (interleaved, non-sharded) ensures the
reshape is supported. The comment at line 2655 states:
```python
# Move to L1 for QK norm (reshape doesn't work on sharded tensors)
```

**Input:** Q `[1, B, 16, 128]`, K `[1, B, 4, 128]`, TILE_LAYOUT, DRAM interleaved.

**Output:** Q `[1, B, 16, 128]`, K `[1, B, 4, 128]`, TILE_LAYOUT, L1 interleaved.

**Host touch:** No. Data movement is DRAM→L1 via NoC.

**Performance note:** This is a DRAM-to-L1 copy for `B * 16 * 128 * 2 = 4096 * B` bytes
of Q and `B * 4 * 128 * 2 = 1024 * B` bytes of K. At batch=1 this is 5 KiB total. Small
in absolute terms, but this transition is caused entirely by QK norm's reshape
requirement — it is an avoidable data movement if QK norm were reordered or made
compatible with sharded input. Analyzed in detail in Chapter 3 and Chapter 5.

---

## Step 9 — `_apply_qk_norm` (reshape, `ttnn.rms_norm`, reshape back)

**Lines:** 2659

```python
query_states, key_states = self._apply_qk_norm(query_states, key_states)
```

Implemented at lines 2454–2493. The decode path within `_apply_qk_norm`:

```python
# lines 2464–2468 (decode_mode branch)
q_reshaped = ttnn.reshape(query_states, (batch_size * num_heads, head_dim))       # [B*16, 128]
k_reshaped = ttnn.reshape(key_states,   (batch_kv * num_kv_heads, head_dim_k))   # [B*4, 128]
q_normed = self.query_layernorm(q_reshaped)   # TTNNRMSNorm.forward -> ttnn.rms_norm
k_normed = self.key_layernorm(k_reshaped)     # TTNNRMSNorm.forward -> ttnn.rms_norm
# ... typecast guards (lines 2481-2484) ...
query_states = ttnn.reshape(q_normed, (1, batch_size, num_heads, head_dim))   # [1, B, 16, 128]
key_states   = ttnn.reshape(k_normed, (1, batch_kv, num_kv_heads, head_dim))  # [1, B, 4, 128]
```

**What it does:** Applies per-head RMS normalization to Q and K. This prevents attention
logit explosion in deep MoE architectures. The norms run independently on each device
(non-distributed) using `TTNNRMSNorm` — which calls `ttnn.rms_norm` with replicated
weights (line 95 of `normalization.py`). The reshape to 2D is required because
`ttnn.rms_norm` normalizes over the last dimension and expects a 2D or 4D input without
conflicting shard constraints.

**Why it is there:** `use_qk_norm=True` in the Bailing MoE model config. This is a
model-architecture requirement, not an optimization artifact.

**Input:** Q `[1, B, 16, 128]`, K `[1, B, 4, 128]`, TILE_LAYOUT, L1 interleaved.

**Output:** Q `[1, B, 16, 128]`, K `[1, B, 4, 128]`, TILE_LAYOUT, L1 interleaved,
bfloat16.

**Host touch:** No. The `TTNNRMSNorm` weights are pre-loaded on device as replicated
tensors.

**Performance note:** The weights for `query_layernorm` and `key_layernorm` are of
shape `[32, 128]` (expanded in `normalization.py` line 85: `weight.unsqueeze(0).expand(32, -1)`).
At `head_dim=128`, the comment at line 2380 of `attention.py` explains:
```python
# QK norms use non-distributed version (head_dim too small to shard across devices)
```
Sharding `head_dim=128` across N=8 devices gives 16 elements per device — below the
32-element tile width minimum for `TTNNDistributedRMSNorm`. Chapter 5 analyzes whether
this constraint is truly binding and proposes applying QK norm before the all-gather as
an alternative.

---

## Step 10 — `cache_position_tensor` construction and `ttnn.from_torch` → `cur_pos_tt`

**Lines:** 2663–2685

```python
if cache_position is None:
    cur_pos = past_key_values.get_seq_length(layer_idx)
    cache_position_tensor = torch.tensor([cur_pos], dtype=torch.int32)
else:
    cp = cache_position
    if isinstance(cp, TorchTTNNTensor):
        cp = cp.to_torch
    if isinstance(cp, ttnn.Tensor):
        mesh_composer = None
        if hasattr(cp, "device") and cp.device() is not None and cp.device().get_num_devices() > 1:
            mesh_composer = ttnn.ConcatMeshToTensor(cp.device(), dim=0)
        cp = ttnn.to_torch(cp, mesh_composer=mesh_composer)   # device → host
    cache_position_tensor = cp.flatten()[:batch_size].to(torch.int32)

mesh_mapper = ttnn.ReplicateTensorToMesh(self.device)
cur_pos_tt = ttnn.from_torch(
    cache_position_tensor,
    device=self.device,
    dtype=ttnn.int32,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    mesh_mapper=mesh_mapper,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
```

**What it does:** Produces `cur_pos_tt`, a device-resident `int32` tensor of shape `[B]`
containing the current KV cache write position for each sequence in the batch. The
`ttnn.from_torch` call uploads this from host to device with `ReplicateTensorToMesh`
topology so all 8 devices hold identical copies.

**Why it is there:** `paged_update_on_device` and `paged_sdpa_decode` require an on-device
`cur_pos` tensor to know which KV cache slot to write to and which positions to attend
over. The kernel does not read the position from host; it must be a device-resident
tensor to avoid stalling the device command queue.

**Input:** `cache_position` — either `None`, a `torch.Tensor`, a `ttnn.Tensor`, or a
`TorchTTNNTensor` (host-side Python value). If `cache_position` is a `ttnn.Tensor`, a
`ttnn.to_torch` call is made first (device→host, PCIe transfer).

**Output:** `cur_pos_tt`, shape `[B]` (e.g., `[1]` at batch=1), int32, ROW_MAJOR,
DRAM, replicated across 8 devices.

**Host touch:** **Yes — always.** The `ttnn.from_torch` call on line 2678 is a
host→device PCIe transfer every decode step. If `cache_position` is a `ttnn.Tensor`,
there is additionally a device→host transfer (the `ttnn.to_torch` at line 2674) before
the upload.

**Performance note:** This is one of two host-touching hot spots in the decode path and
is analyzed in depth in Chapter 4. The `ttnn.from_torch` for a `[B]` int32 tensor is
small in data volume but the call itself requires Python-level dispatch overhead and
PCIe bus setup. At batch=1 the payload is 4 bytes, but the overhead of the call is
typically tens of microseconds.

---

## Step 11 — `BailingRotarySetup.get_cos_sin_for_decode(cache_position_tensor)`

**Lines:** 2696

```python
cos_ttnn, sin_ttnn = self._rotary_setup.get_cos_sin_for_decode(cache_position_tensor)
```

Implemented in `rope.py` lines 420–472.

**What it does:** Performs an on-device embedding table lookup to retrieve the cos and
sin values for the current decode positions. The cos/sin tables
(`cos_cache_row_major`, `sin_cache_row_major`) are pre-loaded in DRAM during
`move_weights_to_device_impl` as row-major tensors of shape `[max_seq_len, head_dim]`.
The `ttnn.embedding` call indexes into these tables using `pos_ttnn` (the position
indices) and produces cos/sin of shape `[1, B, head_dim]`, which is then unsqueezed and
transposed to `[1, B, 1, D]`.

**Why it is there:** RoPE requires position-specific cos/sin values. The embedding
lookup retrieves the correct row from the pre-computed cache rather than recomputing
them. The result is the cos/sin tensor for the specific decode step position.

**Host touch:** **Yes (always).** the `ttnn.from_torch` PCIe upload runs every step;
an additional `ttnn.to_torch` device-drain may also occur if `position_ids` arrives as
a `ttnn.Tensor`. `get_cos_sin_for_decode` (rope.py lines 425–435) checks whether
`position_ids` is a `ttnn.Tensor`; if so it calls `ttnn.to_torch` to convert it to a
host tensor before the `ttnn.from_torch` re-upload at line 444. Since
`cache_position_tensor` in the normal `_forward_decode_paged` path is already a
`torch.Tensor` (constructed at step 10), the device→host copy is skipped. However,
`ttnn.from_torch` at line 444 is always called, uploading the `[1, B]` uint32 position
index tensor to device on every decode step.

**Input:** `cache_position_tensor`, shape `[B]`, host `torch.int32`.

**Output:** `cos_ttnn` `[1, B, 1, D]` = `[1, B, 1, 128]`, TILE_LAYOUT, DRAM,
replicated. Same for `sin_ttnn`.

**Performance note:** The `ttnn.from_torch` inside `get_cos_sin_for_decode` is a second
host-device PCIe transfer per decode step (for a tiny `[1, B]` uint32 tensor). Chapter 4
proposes maintaining a persistent device-resident position tensor updated in-place to
eliminate both the step-10 and step-11 PCIe uploads.

---

## Step 12a/12b — `ttnn.to_memory_config` cos/sin to HEIGHT_SHARDED

**Lines:** 2701–2709

```python
rope_shard_mem = ttnn.create_sharded_memory_config(
    shape=(ttnn.TILE_SIZE, self.head_dim),   # (32, 128)
    core_grid=batch_grid,
    strategy=ttnn.ShardStrategy.HEIGHT,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)
cos_ttnn = ttnn.to_memory_config(cos_ttnn, rope_shard_mem)
sin_ttnn = ttnn.to_memory_config(sin_ttnn, rope_shard_mem)
```

**What it does:** Moves cos and sin from DRAM interleaved into L1 HEIGHT_SHARDED memory
with shard shape `(32, 128)` = `(TILE_SIZE, head_dim)`. One core per batch slot holds
one tile-row of cos/sin.

**Why it is there:** `ttnn.experimental.rotary_embedding_llama` in decode mode requires
HEIGHT_SHARDED L1 input for cos, sin, Q, and K. The shard spec `(TILE_SIZE, head_dim)`
matches the expected per-core data shape for one decode token per head.

**Input:** cos/sin `[1, B, 1, 128]`, TILE_LAYOUT, DRAM.

**Output:** cos/sin `[1, B, 1, 128]`, TILE_LAYOUT, L1 HEIGHT_SHARDED, shard `(32, 128)`
per core.

**Host touch:** No.

---

## Step 12c/12d — `ttnn.to_memory_config` Q and K to HEIGHT_SHARDED

**Lines:** 2711–2727

```python
q_shard_mem = ttnn.create_sharded_memory_config(
    shape=(ttnn.TILE_SIZE, self.head_dim),   # (32, 128)
    core_grid=batch_grid,
    strategy=ttnn.ShardStrategy.HEIGHT,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)
query_states = ttnn.to_memory_config(query_states, q_shard_mem)

k_shard_mem = ttnn.create_sharded_memory_config(
    shape=(ttnn.TILE_SIZE, key_states.shape[-1]),  # (32, 128)
    core_grid=batch_grid,
    ...
)
key_states = ttnn.to_memory_config(key_states, k_shard_mem)
```

**What it does:** Moves Q and K from L1 interleaved (post-QK norm) to L1 HEIGHT_SHARDED
with shard `(32, 128)` per core, matching the cos/sin shard spec from step 12a/12b.

**Why it is there:** `rotary_embedding_llama(is_decode_mode=True)` requires all four
inputs (Q, K, cos, sin) to be HEIGHT_SHARDED in L1. Q and K currently sit in L1
interleaved after `_apply_qk_norm`; this transition reshards them in-place within L1
(no DRAM round-trip for Q and K).

**Input:** Q `[1, B, 16, 128]`, K `[1, B, 4, 128]`, TILE_LAYOUT, L1 interleaved.

**Output:** Q `[1, B, 16, 128]`, K `[1, B, 4, 128]`, TILE_LAYOUT, L1 HEIGHT_SHARDED,
shard `(32, 128)` per core.

**Host touch:** No.

**Performance note:** This is a second memory config transition for Q and K (the first
was at step 8). They moved from DRAM→L1 interleaved at step 8, and now move from L1
interleaved→L1 HEIGHT_SHARDED at step 12c/12d. Chapter 3 identifies that the DRAM→L1
move at step 8 and the L1 reshard here could in principle be fused into a single
DRAM→L1 HEIGHT_SHARDED move, eliminating the intermediate interleaved state.

---

## Step 13 — `BailingRotarySetup.get_trans_mat_decode_sharded(batch_size)`

**Lines:** 2729

```python
trans_mat = self._rotary_setup.get_trans_mat_decode_sharded(batch_size)
```

Implemented at `rope.py` lines 478–498.

**What it does:** Returns a pre-created HEIGHT_SHARDED transformation matrix for the
RoPE rotation operation. The matrix maps `[x_even, x_odd]` → `[-x_odd, x_even]` for
the interleaved rotation format used by `rotary_embedding_llama`. The matrix shape is
`[1, 1, B * TILE_SIZE, TILE_SIZE]` = `[1, 1, B*32, 32]`, sharded so each core holds
one `(32, 32)` tile.

**Why it is there:** `rotary_embedding_llama` performs the RoPE rotation as a batched
matrix multiply internally; it requires this pre-built transformation matrix to be
resident in L1 in HEIGHT_SHARDED layout, matching the batch grid of the other inputs.

**Input:** None (lazy cache lookup keyed by `batch_size`).

**Output:** `trans_mat`, `[1, 1, B*32, 32]`, TILE_LAYOUT, L1 HEIGHT_SHARDED (fetched
from `_trans_mat_decode_sharded_cache`).

**Host touch:** No after first call. On the first call for a given `batch_size`,
`ttnn.from_torch` is called once to create and cache the device tensor (lazy
initialization). Subsequent calls return the cached tensor.

---

## Step 14 — `ttnn.experimental.rotary_embedding_llama` for Q

**Lines:** 2731–2733

```python
query_states = ttnn.experimental.rotary_embedding_llama(
    query_states, cos_ttnn, sin_ttnn, trans_mat, is_decode_mode=True
)
```

**What it does:** Applies Rotary Position Embedding to Q. In decode mode, the kernel
expects all four inputs in HEIGHT_SHARDED L1 with consistent shard specs (one
`(32, head_dim)` tile per batch slot per head). The kernel computes
`q_out = q * cos + rotate_half(q) * sin` using the Meta interleaved rotation convention
via the `trans_mat` matrix.

**Why it is there:** RoPE encodes positional information relative to the KV cache
positions into Q and K, enabling the model to attend correctly to cached tokens.

**Input:** Q `[1, B, 16, 128]` HEIGHT_SHARDED L1; cos/sin `[1, B, 1, 128]` HEIGHT_SHARDED
L1; trans_mat `[1, 1, B*32, 32]` HEIGHT_SHARDED L1.

**Output:** `query_states` `[1, B, 16, 128]`, TILE_LAYOUT, L1 HEIGHT_SHARDED.

**Host touch:** No.

---

## Step 15 — `ttnn.experimental.rotary_embedding_llama` for K

**Lines:** 2734–2736

```python
key_states = ttnn.experimental.rotary_embedding_llama(
    key_states, cos_ttnn, sin_ttnn, trans_mat, is_decode_mode=True
)
```

**What it does:** Same operation as step 14, applied to K. Shape `[1, B, 4, 128]`.

**Input:** K `[1, B, 4, 128]` HEIGHT_SHARDED L1; cos/sin/trans_mat as above.

**Output:** `key_states` `[1, B, 4, 128]`, TILE_LAYOUT, L1 HEIGHT_SHARDED.

**Host touch:** No.

---

## Step 16a/16b — Re-shard K and V to `kv_mem` HEIGHT_SHARDED for paged_update

**Lines:** 2739–2754

```python
num_cores = batch_size
compute_grid = self.device.compute_with_storage_grid_size()
shard_grid = ttnn.num_cores_to_corerangeset(num_cores, compute_grid, True)
kv_vol = key_states.volume() // key_states.padded_shape[-1] // num_cores
kv_shard = ttnn.ShardSpec(
    shard_grid,
    [kv_vol, key_states.padded_shape[-1]],  # different from rope_shard_mem
    ttnn.ShardOrientation.ROW_MAJOR,
)
kv_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, kv_shard)
key_states   = ttnn.to_memory_config(key_states,   kv_mem)
value_states = ttnn.to_memory_config(value_states, kv_mem)
```

**What it does:** Reshards K to a new HEIGHT_SHARDED configuration compatible with
`paged_update_on_device`. V is moved from DRAM directly to this `kv_mem` configuration
(V was never moved to L1 before this point — it skipped steps 8–15). The new shard
spec uses `[kv_vol, padded_head_dim]` per core, where `kv_vol` is computed as
`total_elements / padded_head_dim / num_cores`. This differs from the `rope_shard_mem`
spec of `(TILE_SIZE, head_dim)` = `(32, 128)`.

**Why it is there:** `paged_update_on_device` requires a specific HEIGHT_SHARDED shard
layout for K and V — the shard height encodes the full per-core KV volume, whereas the
RoPE kernel expected each shard to be exactly one tile-row `(32, 128)`. These two specs
are not identical, necessitating a reshard.

**Input:** K `[1, B, 4, 128]` L1 HEIGHT_SHARDED (rope spec `(32, 128)` per core);
V `[1, B, 4, 128]` DRAM replicated.

**Output:** K `[1, B, 4, 128]` L1 HEIGHT_SHARDED (`kv_mem` spec); V `[1, B, 4, 128]`
L1 HEIGHT_SHARDED (`kv_mem` spec).

**Host touch:** No.

**Performance note:** K undergoes two HEIGHT_SHARDED configurations: `rope_shard_mem`
for RoPE (step 12d/15) and `kv_mem` for paged_update (step 16a). This is the second
reshard of K within L1. Chapter 3 investigates whether the paged_update kernel's shard
spec requirement can be made compatible with the RoPE shard spec.

---

## Step 17 — `past_key_values.paged_update_on_device`

**Lines:** 2756–2761

```python
past_key_values.paged_update_on_device(
    key_states,
    value_states,
    layer_idx=layer_idx,
    current_pos=cur_pos_tt,
)
```

**What it does:** Writes the current step's K and V tensors into the on-device paged
KV cache at the positions specified by `cur_pos_tt`. The KV cache lives in device DRAM
and is addressed by `layer_idx` and `cur_pos_tt`. This kernel runs fully on-device; no
data is returned to the host.

**Why it is there:** The paged KV cache stores all previous K/V states for the ongoing
decode session. This write step must happen before `paged_sdpa_decode` in step 19 so
that the current token's K/V are included in the attention computation.

**Input:** K `[1, B, 4, 128]` L1 HEIGHT_SHARDED; V `[1, B, 4, 128]` L1 HEIGHT_SHARDED;
`cur_pos_tt` `[B]` int32 DRAM replicated.

**Output:** Written into the on-device KV cache; no returned tensor.

**Host touch:** No.

---

## Step 18 — `ttnn.deallocate` K and V

**Lines:** 2762–2763

```python
ttnn.deallocate(key_states)
ttnn.deallocate(value_states)
```

**What it does:** Frees the L1 buffers holding the current step's K and V tensors. After
the KV cache write in step 17, these tensors are no longer needed — all future attention
for this step reads from the KV cache (not from `key_states`/`value_states` directly).

**Why it is there:** L1 memory is a scarce resource on Wormhole. Freeing K and V
promptly after the cache write allows the L1 freed to be reused by subsequent ops (SDPA,
concat heads, dense). Without explicit deallocation, these buffers would remain
allocated until the end of the function.

**Input/Output:** K and V freed from L1; no tensor output.

**Host touch:** No.

---

## Step 19 — `past_key_values.paged_sdpa_decode`

**Lines:** 2766–2773

```python
attn_output = past_key_values.paged_sdpa_decode(
    query_states,
    layer_idx,
    current_pos=cur_pos_tt,
    scale=self.scaling,
    program_config=self.sdpa.decode_program_config,
    compute_kernel_config=self.sdpa.compute_kernel_config,
)
```

**What it does:** Computes scaled dot-product attention over the full paged KV cache for
the current decode step. This is the compute-intensive attention kernel. It reads Q from
L1 HEIGHT_SHARDED, reads K and V from the on-device DRAM KV cache, computes
`softmax(Q * K^T / sqrt(D)) * V`, and produces the attention output. The `compute_kernel_config`
uses HiFi4, `fp32_dest_acc_en=True`, `packer_l1_acc=True` (set at lines 2434–2440).
The `decode_program_config` uses `q_chunk_size=0, k_chunk_size=0` (kernel picks defaults)
and `exp_approx_mode=False`.

**Why it is there:** Core attention computation. The paged implementation reads K/V from
non-contiguous pages in the DRAM KV cache, handling variable-length sequences and batch
sequences at different cache positions. `self.scaling = head_dim ** -0.5 = 128 ** -0.5`.

**Input:** Q `[1, B, 16, 128]` L1 HEIGHT_SHARDED; KV cache in DRAM (not a direct tensor
argument — accessed by layer index and position); `cur_pos_tt` `[B]` int32 DRAM.

**Output:** `attn_output`, shape `[1, B, 16, 128]`, TILE_LAYOUT (memory location depends
on SDPA kernel output config).

**Host touch:** No.

**Performance note:** SDPA is typically the most compute-intensive op in the decode path,
especially for long contexts where the KV cache spans many pages. Chapter 6 analyzes the
HiFi4 configuration and whether switching to HiFi2 is safe.

---

## Step 20 — `ttnn.to_memory_config(attn_output, sdpa_output_memcfg)`

**Lines:** 2776–2783

```python
sdpa_output_memcfg = ttnn.create_sharded_memory_config(
    shape=(32, self.head_dim),               # (32, 128)
    core_grid=ttnn.CoreGrid(y=1, x=batch_size),
    strategy=ttnn.ShardStrategy.HEIGHT,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)
attn_output = ttnn.to_memory_config(attn_output, sdpa_output_memcfg)
```

**What it does:** Reshards the SDPA output to the HEIGHT_SHARDED spec expected by
`nlp_concat_heads_decode`. The shard shape is `(32, head_dim)` = `(32, 128)` per core,
placed on a `1×B` core grid (one core per batch slot).

**Why it is there:** `ttnn.experimental.nlp_concat_heads_decode` requires its input in
exactly this HEIGHT_SHARDED configuration. The SDPA kernel's output memory config may
differ from this spec.

**Input:** `attn_output` `[1, B, 16, 128]`, TILE_LAYOUT, (various memory).

**Output:** `attn_output` `[1, B, 16, 128]`, TILE_LAYOUT, L1 HEIGHT_SHARDED,
shard `(32, 128)` per core on `CoreGrid(y=1, x=B)`.

**Host touch:** No.

---

## Step 21 — `ttnn.experimental.nlp_concat_heads_decode`

**Lines:** 2785–2788

```python
attn_output = ttnn.experimental.nlp_concat_heads_decode(
    attn_output,
    num_heads=self.num_heads,
)
```

**What it does:** Concatenates all attention heads back into a single hidden dimension,
producing `[1, 1, B, d_model]` = `[1, 1, B, 2048]`. This is the inverse of the head
split performed when Q/K/V were reshaped at step 6.

**Why it is there:** The dense projection in step 22 expects input of shape
`[1, 1, B, d_model]` — a contiguous hidden representation per batch element, not
per-head tensors.

**Input:** `attn_output` `[1, B, 16, 128]`, TILE_LAYOUT, L1 HEIGHT_SHARDED.

**Output:** `attn_output` `[1, 1, B, 2048]`, TILE_LAYOUT (memory config as produced by
kernel, typically L1 or DRAM).

**Host touch:** No.

---

## Step 22 — `dense(attn_output)` via `TTNNLinearIReplicatedWColSharded`

**Lines:** 2789

```python
attn_output = self.dense(attn_output)
```

**What it does:** Projects the concatenated attention output from `d_model=2048` back to
`d_model=2048` (the output projection). Uses `TTNNLinearIReplicatedWColSharded`, which
expects replicated input and produces col-sharded output. Each device computes a local
`[1, 1, B, 2048] × [2048, 256]` matmul (weight sharded on last dim), retaining
`[1, 1, B, 256]` of the output.

**Why it is there:** The output projection maps from the multi-head attention space back
to the model's hidden dimension space. The output will feed into the next MoE layer,
which expects col-sharded input — consistent with `TTNNLinearIReplicatedWColSharded`'s
output topology.

**Input:** `attn_output` `[1, 1, B, 2048]`, TILE_LAYOUT, replicated (or moved to
DRAM by the dense forward).

**Output:** `attn_output` `[1, 1, B, 2048]` logical, col-sharded (each device
`[1, 1, B, 256]`), TILE_LAYOUT, DRAM.

**Host touch:** No.

---

## Step 23 — `ttnn.slice` (conditional, `batch_size < 32`)

**Lines:** 2794–2795

```python
if batch_size < 32:
    attn_output = ttnn.slice(attn_output, [0, 0, 0, 0], [1, 1, batch_size, attn_output.shape[-1]])
```

**What it does:** Removes the batch-dimension padding that was added to align to 32
(the TTNN tile size). If the true batch size is less than 32, the output tensor was
padded to shape `[1, 1, 32, 2048]`; this slice removes the padding rows, returning
`[1, 1, B, 2048]`.

**Why it is there:** TTNN tile layout requires dimensions to be multiples of 32. When
`B < 32`, the tensors were implicitly padded to `[1, 1, 32, 2048]` throughout the
forward pass. The slice restores the true batch size before returning.

**Input:** `attn_output` `[1, 1, 32, 2048]` (padded), TILE_LAYOUT, DRAM.

**Output:** `attn_output` `[1, 1, B, 2048]`, TILE_LAYOUT, DRAM.

**Host touch:** No.

---

## Step 24 — `ttnn.reshape` to `[B, S, d_model]`

**Lines:** 2797

```python
attn_output = ttnn.reshape(attn_output, (batch_size, seq_length, -1))
```

**What it does:** Reshapes the 4D attention output `[1, 1, B, 2048]` to the 3D format
`[B, 1, 2048]` that the surrounding model code expects.

**Why it is there:** The model's main forward pass processes hidden states as 3D tensors
`[batch, seq_len, hidden_size]`. The 4D layout used internally by attention kernels is
collapsed back to 3D here.

**Input:** `attn_output` `[1, 1, B, 2048]`, TILE_LAYOUT, DRAM.

**Output:** `attn_output` `[B, 1, 2048]`, TILE_LAYOUT, DRAM, col-sharded. Returned
along with `None` (no attention weights) and `past_key_values` (updated KV cache
reference).

**Host touch:** No. Reshape is a metadata operation.

---

## Critical Path Summary

At batch=1 decode, the ops with the most significant latency contributions are:

| Category | Steps | Bottleneck reason |
|----------|-------|-------------------|
| Collective communication (reduce-scatter) | 2 | `reduce_scatter_minimal_async` inside Q projection (`TTNNLinearIColShardedWRowSharded`) |
| Collective communication (all-gather) | 3, 5a, 5b, 5c | Four synchronous `all_gather` calls; each is a barrier |
| Host-device transfers | 10, 11 | `ttnn.from_torch` each decode step |
| Memory layout transitions | 8a/8b, 12a–12d, 16a/16b, 20 | Nine `to_memory_config` calls |
| Compute | 19 (SDPA) | Attention over full KV cache; compute-bound for long contexts |

Setup-only ops with negligible per-step cost: step 1 (no-op when already TILE),
step 13 (cached after first call), step 18 (deallocation metadata only).

---

**Next:** [tensor_layouts.md](./tensor_layouts.md) — reference table of tensor shapes
and layouts at each of the 24 steps, with a memory location transition diagram.
