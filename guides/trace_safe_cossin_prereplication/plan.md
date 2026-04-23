# Plan: Trace-Safe Pre-Replication of Position Embeddings in TTNNQwen3FullAttention

---

## 1. Audience

**Primary audience:** ML engineers working on the tt-symbiote / TTNN inference stack who are
responsible for enabling end-to-end Metal Trace capture for the Qwen3.6-35B-A3B hybrid decoder.
They have been asked to make `TTNNQwen3FullAttention` compatible with `TracedRun` and need to
understand why the current `_ensure_replicated` helper breaks trace capture and how to fix it.

**What they already know:**

- The three-phase TTNN trace API: `ttnn.begin_trace_capture`, `ttnn.end_trace_capture`,
  `ttnn.execute_trace`, and the requirement that captured ops must not allocate new device buffers
  during replay
- The `TracedRun` execution mode in tt-symbiote: how `_capture_trace` orchestrates a compile run
  followed by a capture bracket, how `_alloc_kwarg_tensor` pre-allocates input buffers, and how
  `ttnn.copy` is used inside the traced region to update those buffers without reallocating
- The existing `_decode_cur_pos` pre-allocation pattern in `move_weights_to_device_impl`: how a
  scalar position tensor is pre-allocated as a device buffer before trace capture and updated via
  `ttnn.copy` at each decode step so that the trace replay sees a stable device address
- Tensor Parallelism (TP) on T3K: column-sharding and row-sharding of weight matrices, how `ttnn.
  from_torch` creates sharded or replicated tensors on the mesh, and why cos/sin tables are
  typically replicated (every device needs the full table to rotate its head shard)
- `TTNNQwen3FullAttention`: the full-attention layer in the Qwen3.6 hybrid decoder — its
  `forward` method, the role of `TTNNRotaryPositionEmbedding`, and the `_ensure_replicated`
  helper that was added to fix a runtime crash when cos/sin arrived sharded instead of replicated
- Basic familiarity with TTNN memory configs: `DRAM_MEMORY_CONFIG`, `L1_MEMORY_CONFIG`,
  `ROW_MAJOR_LAYOUT`, `TILE_LAYOUT`, and the difference between interleaved and sharded tensors

**What they do NOT need to know in advance:**

- Why `ttnn.from_torch` is incompatible with Metal Trace (this guide explains it in Chapter 1)
- How pre-allocated replicated buffers work at the device level and why `ttnn.copy` to such a
  buffer is trace-safe while `ttnn.from_torch` is not
- What layout and memory config cos/sin tensors must carry to pass through `ttnn.unsqueeze` and
  `ttnn.experimental.rotary_embedding` without triggering a re-layout
- Whether `TracedRun._alloc_kwarg_tensor` already handles cos/sin or whether new pre-allocation
  logic must be added to `move_weights_to_device_impl`
- How to preserve the existing warm-up guard in `TTNNRotaryPositionEmbedding.forward` that
  detects wrongly-sharded inputs via the `rotary_dim % 64 != 0` heuristic

---

## 2. Chapter List

---

### Chapter 1 — Why `ttnn.from_torch` Breaks Metal Trace

**Description:** Establishes the precise reason `_ensure_replicated` is trace-incompatible by
explaining what Metal Trace records, what replay forbids, and why any host-side buffer allocation
inside a captured region — including the one hidden inside `ttnn.from_torch` — causes replay to
produce incorrect or undefined results.

**Directory:** `ch1_trace_incompatibility/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Diagram: three-phase trace lifecycle (compile run → capture run → replay), annotated to show
    where `_ensure_replicated` is called relative to the capture bracket
  - Glossary of terms introduced in this chapter: host operation, device buffer, buffer address
    stability, command buffer, Metal Trace
  - "What's next" section listing files in reading order

- `what_trace_records.md`
  - Explain that `ttnn.begin_trace_capture` records the sequence of device commands — kernel
    dispatches, DMA transfers, semaphore operations — enqueued on the `MeshCommandQueue` during
    the capture run; the command buffer stores concrete device memory addresses as they exist at
    capture time
  - Explain what replay does: re-issues those exact commands against those exact device memory
    addresses without any Python re-execution or host-side buffer allocation
  - Explain what trace does NOT record: Python control flow, tensor shape recomputation, any
    operation that executes on the host CPU rather than the device, and — critically — any
    allocation of new device buffers
  - Show the invariant that must hold: every device buffer touched during the capture run must
    exist at the same address on every subsequent replay call; if an op allocates a new buffer
    during replay (at a different address), the recorded commands reference stale addresses,
    causing silent data corruption or a device crash

- `from_torch_is_a_host_operation.md`
  - Explain the call chain of `ttnn.from_torch` with `device=mesh_device`: the tensor data is
    first assembled in host DRAM (a Python/C++ host-side allocation), then transferred to device
    via DMA; crucially, a new device buffer is allocated for the destination on every call
  - Explain that this new device buffer allocation is a host operation that cannot be recorded by
    the trace: from the trace's perspective, a brand-new device address appears during replay that
    was not present during capture, making every command that references the new buffer invalid
  - Show why this is not immediately obvious: `ttnn.from_torch` looks like a TTNN op (it has a
    `ttnn.` prefix) but is fundamentally a host-to-device data staging call — it is as
    trace-incompatible as a Python `torch.zeros` call inside a captured region
  - Contrast with `ttnn.copy` into a pre-allocated buffer: `ttnn.copy` uses the existing device
    buffer address baked into the trace; no allocation occurs; the DMA command recorded at capture
    time is valid for every subsequent replay because the destination address is stable

- `ensure_replicated_call_site.md`
  - Locate `_ensure_replicated` in `TTNNQwen3FullAttention.forward` and identify exactly where it
    is called relative to the trace bracket in a `TracedRun` execution
  - Explain the original bug it was solving: cos/sin tensors arriving as TP-sharded tensors
    (each device holding a shard of the frequency table) instead of replicated (every device
    holding the full table), which caused `ttnn.experimental.rotary_embedding` to crash because
    it requires the full cos/sin table on each device
  - Show that `_ensure_replicated` calls `ttnn.from_torch` when the input is detected as sharded,
    which allocates a new replicated device buffer — precisely the operation that breaks trace
  - Identify the fix required: replicated cos/sin buffers must be allocated before the trace
    capture begins and updated via `ttnn.copy` inside the traced region, so that the buffer
    address is stable across all replays

---

### Chapter 2 — The `_decode_cur_pos` Pre-Allocation Pattern

**Description:** Explains the existing pattern for pre-allocating a scalar position tensor as a
stable device buffer before trace capture, using `_decode_cur_pos` in
`move_weights_to_device_impl` as the canonical example, and extracts the generalizable pattern
that will be applied to cos/sin buffers.

**Directory:** `ch2_cur_pos_pattern/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Diagram: lifecycle of `_decode_cur_pos` from allocation in `move_weights_to_device_impl`
    through `ttnn.copy` update in the decode loop to replay access inside the trace
  - Recap of Chapter 1 prerequisites: why buffer address stability is required for trace replay
  - "What's next" section listing files in reading order

- `decode_cur_pos_walkthrough.md`
  - Walk through `move_weights_to_device_impl` and identify where `_decode_cur_pos` is allocated:
    the exact `ttnn.from_torch` call (or equivalent) that creates the device tensor before any
    trace capture begins, what dtype and shape it has, and what memory config and layout are used
  - Explain that this allocation is performed during the model's weight-to-device transfer phase,
    which happens before `TracedRun._capture_trace` is ever called — the buffer address is
    therefore stable by the time the capture bracket opens
  - Show how `_decode_cur_pos` is updated at each decode step: the host-side current position
    integer is wrapped in a `torch.tensor`, converted, and written into the pre-allocated device
    buffer using `ttnn.copy`; the `ttnn.copy` call is inside the traced region and is trace-safe
    because it uses the stable, pre-allocated destination address
  - Identify the key properties that make this pattern work: (1) allocation before capture, (2)
    fixed shape and dtype throughout the decode loop, (3) update via `ttnn.copy` (not
    `ttnn.from_torch`) inside the trace

- `pattern_generalization.md`
  - Extract the generalizable pre-allocation pattern from `_decode_cur_pos`:
    1. In `move_weights_to_device_impl` (or equivalent pre-capture setup), allocate the device
       tensor using `ttnn.from_torch` (or `ttnn.zeros`) with the full target shape, dtype, layout,
       and memory config; store a reference to the tensor on the module instance
    2. Before trace capture begins, verify that the tensor is on device and has the correct
       replication factor (for TP models, this means the `TensorToMesh` mapping must produce a
       replicated tensor — one full copy on every device)
    3. Inside the traced region (at the top of the forward method), update the tensor content via
       `ttnn.copy(source, destination)` where `destination` is the pre-allocated device tensor;
       no new buffer is allocated
    4. Pass the now-updated device tensor to downstream ops as a normal `ttnn.Tensor`
  - Identify what makes cos/sin different from `_decode_cur_pos`: (a) cos/sin change at every
    decode step (different position means different sin/cos values), just like `_decode_cur_pos`;
    (b) cos/sin have a non-scalar shape (`[1, 1, seq_len, rotary_dim]` or similar) that must be
    compatible with `ttnn.unsqueeze` and `ttnn.experimental.rotary_embedding`; (c) cos/sin must
    be replicated across TP devices (every device needs the full table for its head shard)
  - State the design decision: follow the `_decode_cur_pos` pattern exactly, but with a 2D or
    4D pre-allocated buffer and a replicated `TensorToMesh` mapping

- `traced_run_alloc_kwarg_tensor.md`
  - Locate `TracedRun._alloc_kwarg_tensor` and describe what it pre-allocates: which kwargs it
    inspects (typically the primary input activation tensor), what shape it uses, and how it
    determines the mesh placement
  - Determine whether `_alloc_kwarg_tensor` currently handles cos/sin keyword arguments (e.g.,
    `cos` and `sin` kwargs passed into the attention module's `forward`): if it does, document
    the current layout and mesh mapping; if it does not, document the gap
  - Explain the limitation of `_alloc_kwarg_tensor` for cos/sin: even if it pre-allocates cos/sin
    buffers, it must allocate them as replicated tensors (not sharded) and with the correct layout
    for `ttnn.experimental.rotary_embedding`; document whether the current implementation
    satisfies these constraints or whether `move_weights_to_device_impl` must be extended instead
  - State the conclusion: the recommended location for cos/sin pre-allocation is
    `move_weights_to_device_impl`, following the exact `_decode_cur_pos` model, rather than
    relying on `_alloc_kwarg_tensor`, because `move_weights_to_device_impl` provides explicit
    control over the mesh mapping and layout

---

### Chapter 3 — Pre-Allocating Replicated cos/sin Buffers

**Description:** Provides the concrete implementation plan for pre-allocating cos/sin buffers as
replicated tensors in `move_weights_to_device_impl`, covering the required shape, dtype, layout,
memory config, and `TensorToMesh` mapping, and explains how these choices are constrained by the
downstream ops `ttnn.unsqueeze` and `ttnn.experimental.rotary_embedding`.

**Directory:** `ch3_prereplication_impl/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Quick-reference table: the required attributes of the pre-allocated cos/sin buffer
    (shape, dtype, layout, memory config, mesh mapping)
  - Recap of Chapter 2 prerequisites: the pre-allocation pattern and why placement in
    `move_weights_to_device_impl` is preferred over `_alloc_kwarg_tensor`
  - "What's next" section listing files in reading order

- `downstream_op_constraints.md`
  - Work backwards from `ttnn.experimental.rotary_embedding` to derive the required cos/sin tensor
    attributes:
    - Shape: `ttnn.experimental.rotary_embedding` expects cos and sin tensors shaped
      `[1, 1, seq_len, rotary_dim]` (or a broadcastable form); document the exact expected shape
      and whether `ttnn.unsqueeze` is called on the cos/sin before passing them in (which would
      require the pre-allocated buffer to have one fewer dimension)
    - Layout: `ttnn.experimental.rotary_embedding` requires `TILE_LAYOUT` for compute; if the
      pre-allocated buffer is in `ROW_MAJOR_LAYOUT`, a layout conversion will be triggered inside
      the trace — document whether this is trace-safe or whether the buffer must be pre-allocated
      in `TILE_LAYOUT`
    - Memory config: document whether `DRAM_MEMORY_CONFIG` or `L1_MEMORY_CONFIG` is required;
      for a decode-step cos/sin buffer of shape `[1, 1, 1, rotary_dim]` (single token), L1 is
      feasible; for prefill, DRAM is required; identify which the pre-allocated buffer must target
    - dtype: cos/sin are computed in `float32` or `bfloat16` depending on the model config;
      document the dtype used by `TTNNRotaryPositionEmbedding` and whether it matches what
      `ttnn.experimental.rotary_embedding` expects
  - Identify any shape transformations performed by `_ensure_replicated` or `forward` between
    the cos/sin source and the `ttnn.experimental.rotary_embedding` call site; each such
    transformation must be either (a) moved outside the trace (performed at pre-allocation time)
    or (b) verified to be trace-safe (no new buffer allocation)

- `replicated_mesh_mapping.md`
  - Explain what "replicated" means for a tensor on a T3K TP mesh: the `TensorToMesh` mapper
    produces one full copy of the tensor data on every device in the mesh, so each device can
    apply rotary embedding to its own head shard without cross-device communication
  - Contrast with the sharded case that triggered the original crash: a sharded cos/sin table
    has each device holding only a slice of the table, which is insufficient for each device to
    rotate its full head shard
  - Identify the correct `TensorToMesh` mapping to use when calling `ttnn.from_torch` for the
    pre-allocation: `ReplicateTensorToMesh(mesh_device)` (or the equivalent tt-symbiote helper)
  - Explain how to verify replication at runtime: `ttnn.get_device_tensors(cos_tensor)` should
    return a list with one tensor per device, each with the same shape as the original; document
    this as a debug assertion to add during warm-up
  - Note the memory cost: replicating a `[1, 1, 1, rotary_dim]` decode-step cos/sin buffer
    across 8 T3K devices uses `8 * rotary_dim * 2 bytes` — negligible; document the prefill
    case and whether a different strategy is needed for prefill traces

- `move_weights_impl_changes.md`
  - Provide the concrete code change plan for `move_weights_to_device_impl` in
    `TTNNQwen3FullAttention` (or its parent class if cos/sin allocation is shared):
    1. After existing weight transfers, add a `_cos_replicated` and `_sin_replicated` tensor
       allocation using `ttnn.from_torch(torch.zeros(...), dtype=..., layout=TILE_LAYOUT,
       memory_config=DRAM_MEMORY_CONFIG, mesh_mapper=ReplicateTensorToMesh(mesh_device))` with
       the correct shape derived from `downstream_op_constraints.md`
    2. Store references as `self._cos_replicated` and `self._sin_replicated`
    3. Document that these tensors are allocated once and reused for every decode step; they are
       never deallocated until the module is destroyed
  - Provide the corresponding change to `TTNNQwen3FullAttention.forward`:
    1. Remove (or guard) the `_ensure_replicated` call
    2. At the top of `forward`, perform `ttnn.copy(cos, self._cos_replicated)` and
       `ttnn.copy(sin, self._sin_replicated)` to update the stable buffers with the current
       step's cos/sin values
    3. Use `self._cos_replicated` and `self._sin_replicated` in place of the original `cos` and
       `sin` throughout the rest of `forward`
  - Note that the `ttnn.copy` calls must be inside the traced region (they must be part of the
    captured command sequence so that each replay updates the buffers before the rotary embedding
    op reads them); they must NOT be placed before `begin_trace_capture` because the whole point
    is that the trace itself handles the per-step update

---

### Chapter 4 — Trace Safety of `ttnn.copy` to a Replicated Destination

**Description:** Answers the question of whether `ttnn.copy` from a replicated source to a
pre-allocated replicated destination is trace-safe by examining what the copy op records in the
command buffer and verifying that no new device buffers are allocated during replay.

**Directory:** `ch4_copy_trace_safety/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Answer-first summary: `ttnn.copy` into a pre-allocated buffer is trace-safe; this chapter
    provides the supporting analysis
  - Recap of the distinction (from Chapter 1) between host operations (trace-unsafe) and pure
    device DMA operations (trace-safe when using stable addresses)
  - "What's next" section listing files in reading order

- `what_copy_records.md`
  - Explain what the `ttnn.copy(source, destination)` call enqueues on the `MeshCommandQueue`:
    a DMA transfer from the source device buffer address to the destination device buffer address
  - Distinguish `ttnn.copy` from `ttnn.clone` and from assignment: `ttnn.copy` writes into an
    existing destination buffer without allocating a new one; `ttnn.clone` always allocates a new
    buffer and is therefore trace-unsafe if called inside a captured region
  - Explain what "replicated destination" means in the DMA command: the copy is issued once per
    device in the mesh, with each device transferring from its local source shard to its local
    destination shard; both addresses are stable because both the source (passed-in cos/sin) and
    destination (pre-allocated `_cos_replicated`) are on-device before the trace begins
  - State the requirement: the source tensor (the `cos` argument passed to `forward` at each
    decode step) must also be an on-device tensor at a stable address — if it is a freshly
    computed host tensor, a `ttnn.from_torch` would be needed to move it to device, which is
    trace-unsafe; document how the caller is expected to provide cos/sin as device tensors

- `source_tensor_stability.md`
  - Identify where cos/sin tensors come from at decode time in `TTNNQwen3FullAttention.forward`:
    are they computed on-the-fly from a position index, or are they pre-computed and stored as
    device tensors in `TTNNRotaryPositionEmbedding`?
  - If cos/sin are computed per-step on host and then moved to device: document the exact
    location where `ttnn.from_torch` is called for the cos/sin, and explain why this call must
    be moved outside the traced region (to before `begin_trace_capture`) or replaced with an
    on-device computation (e.g., a precomputed table indexed by `_decode_cur_pos`)
  - If cos/sin are already pre-computed device tensors (e.g., from `TTNNRotaryPositionEmbedding`'s
    internal table): verify that the lookup mechanism (slicing or gathering from a stable device
    table) is trace-safe; document whether this path exists and can be used directly
  - State the correct design: cos/sin at decode time should be sliced from a pre-computed DRAM
    table (a fixed device tensor allocated before trace capture), using `_decode_cur_pos` to index
    into the table; this slice is itself a view into a stable device buffer and is therefore safe
    to use as the source in `ttnn.copy`

- `replay_correctness_verification.md`
  - Describe how to verify that `ttnn.copy` to a replicated destination works correctly across
    multiple consecutive replay calls:
    1. Capture the trace with cos/sin values for position 0
    2. Update `_cos_replicated` and `_sin_replicated` with values for position 1 (this update
       is inside the trace, so execute the trace once)
    3. Execute the trace a second time with new cos/sin values for position 2
    4. After each replay, read back the output tensor to host and compare against a non-traced
       reference forward pass with the same cos/sin values
  - Explain what failure looks like if the copy is not trace-safe: the output of replay iteration
    N+1 would use the cos/sin values from iteration N (because the copy did not execute during
    replay), producing numerically incorrect hidden states that do not match the reference
  - Provide a PCC threshold for the comparison: > 0.999 in BF16 against a float32 reference;
    identify this as a regression test that must pass before the pre-replication change is merged

---

### Chapter 5 — Warm-Up Guard Preservation

**Description:** Analyzes whether the `rotary_dim % 64 != 0` guard in
`TTNNRotaryPositionEmbedding.forward` — which detects wrongly-sharded cos/sin inputs during
warm-up — remains effective after the pre-replication change, and proposes any adjustments needed.

**Directory:** `ch5_warmup_guard/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Context: the guard was added as a safety mechanism to surface the TP sharding bug early
    rather than allowing silent numerical errors from a partially-rotated cos/sin table
  - Answer-first summary: the guard continues to fire correctly during warm-up because warm-up
    runs before the trace capture bracket; after the change, the guard sees the pre-allocated
    replicated buffer instead of the raw sharded input, which is correct behavior
  - "What's next" section listing files in reading order

- `guard_mechanism_analysis.md`
  - Locate the `rotary_dim % 64 != 0` guard in `TTNNRotaryPositionEmbedding.forward` and explain
    what it detects: when cos/sin are sharded across TP devices, each device's shard has
    `rotary_dim / num_devices` columns, which is not a multiple of 64 for typical Qwen3 configs;
    the guard checks for this anomaly and raises an error
  - Explain when the guard fires: it runs during every call to `TTNNRotaryPositionEmbedding.
    forward`, including the warm-up compile run that precedes trace capture
  - Explain why the guard still works after pre-replication: the pre-allocated `_cos_replicated`
    buffer is replicated (full `rotary_dim` columns on each device), so the guard sees the correct
    column count and does not raise; only a misconfigured pre-allocation (e.g., using a sharded
    mesh mapper by mistake) would trigger the guard — which is exactly the intended behavior
  - Identify one edge case: during the very first forward call before `move_weights_to_device_impl`
    has run (e.g., if forward is called without initialization), `self._cos_replicated` does not
    exist yet; document whether a `hasattr` check or an `__init__` default is needed

- `guard_adequacy_after_change.md`
  - Assess whether the `rotary_dim % 64 != 0` check remains a sufficient signal for wrong
    sharding after the pre-replication change:
    - Before the change: the guard caught wrong sharding of the raw cos/sin input argument
    - After the change: the `_cos_replicated` buffer is always replicated (the copy from the raw
      input to the pre-allocated buffer converts sharded to replicated); the guard now checks
      `_cos_replicated` at its output point in the rotary embedding forward — it still catches
      misconfigured replication
  - Recommend adding a second check at the point where `_cos_replicated` is populated (i.e.,
    immediately after the `ttnn.copy` or during `move_weights_to_device_impl`) to verify the
    tensor is truly replicated across all devices, as an explicit assertion rather than relying on
    the shape heuristic alone
  - Recommend a new warm-up-only debug assertion: after `move_weights_to_device_impl` completes,
    call `ttnn.get_device_tensors(self._cos_replicated)` and assert that each per-device tensor
    has shape `[..., rotary_dim]` with `rotary_dim` equal to the expected full value; this
    assertion is O(1) in device ops and can be guarded by a `TTNN_DEBUG` environment variable

- `non_tile_aligned_rotary_dim_interaction.md`
  - Address the interaction between the pre-replication change and the related open issue of
    non-tile-aligned `rotary_dim` values (referenced in the research topic context as a separate
    pending topic)
  - Explain that tile alignment (multiples of 32 for `TILE_LAYOUT`) affects whether the
    pre-allocated cos/sin buffer can be stored in `TILE_LAYOUT` without padding: if
    `rotary_dim = 64` (the Qwen3 case), it is tile-aligned and no padding is needed; if
    `rotary_dim` is non-tile-aligned, the pre-allocated buffer must either use `ROW_MAJOR_LAYOUT`
    (which may require a layout conversion before `ttnn.experimental.rotary_embedding`) or be
    padded to the next tile boundary
  - State that the pre-replication plan described in Chapters 3 and 4 applies cleanly when
    `rotary_dim % 32 == 0`; for non-tile-aligned cases, the layout choice in Chapter 3's
    `downstream_op_constraints.md` must be revisited
  - Cross-reference the separate "partial rotary embedding numerical correctness" research topic
    as the correct location for the non-tile-aligned analysis; this guide focuses on the
    trace-safety concern for the tile-aligned Qwen3 case

---

### Chapter 6 — End-to-End Integration and Test Strategy

**Description:** Consolidates the changes from Chapters 3–5 into a sequenced integration
checklist and a concrete test plan that verifies trace correctness, numerical accuracy, and
warm-up guard preservation before and after the pre-replication change is merged.

**Directory:** `ch6_integration_and_testing/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Prerequisite: the reader should have read Chapters 3, 4, and 5 before following this
    integration plan; Chapter 1 and 2 provide background but are not required for the checklist
  - Scope: this chapter covers integration of pre-replication into `TTNNQwen3FullAttention` and
    `move_weights_to_device_impl`; it does not cover changes to the prefill path (which has a
    different cos/sin lifecycle)
  - "What's next" section listing files in reading order

- `integration_checklist.md`
  - Pre-conditions to verify before beginning the change:
    - [ ] Confirm that `TTNNQwen3FullAttention` has a `move_weights_to_device_impl` method (or
          an equivalent hook called before trace capture); identify the exact class and file path
    - [ ] Confirm that `_decode_cur_pos` pre-allocation in the same file uses `TILE_LAYOUT` and
          `DRAM_MEMORY_CONFIG`; use these as the baseline attributes for cos/sin pre-allocation
    - [ ] Confirm the shape of cos/sin tensors as they arrive in `TTNNQwen3FullAttention.forward`
          at decode time (seq_len=1): document as `[batch, 1, 1, rotary_dim]` or equivalent
    - [ ] Confirm that `ttnn.experimental.rotary_embedding` does not internally call
          `ttnn.from_torch` or allocate intermediate buffers that depend on the cos/sin shape
    - [ ] Confirm that the device is opened with a `trace_region_size` that accommodates the
          additional `ttnn.copy` ops (two per decode step — one for cos, one for sin); this is
          negligible relative to the full decoder trace budget
  - Implementation steps in order:
    1. Add `_cos_replicated` and `_sin_replicated` allocation to `move_weights_to_device_impl`
       with `TILE_LAYOUT`, `DRAM_MEMORY_CONFIG`, and `ReplicateTensorToMesh`
    2. Add the debug assertion in `move_weights_to_device_impl` to verify replication
    3. Replace `_ensure_replicated(cos)` and `_ensure_replicated(sin)` in `forward` with
       `ttnn.copy(cos, self._cos_replicated)` and `ttnn.copy(sin, self._sin_replicated)`
    4. Update all downstream references within `forward` from `cos`/`sin` to
       `self._cos_replicated` / `self._sin_replicated`
    5. Keep `_ensure_replicated` as a private helper but remove it from the traced call path;
       optionally retain it for warm-up-only diagnostic use

  - Post-implementation checks:
    - [ ] Run the existing non-traced forward pass test and confirm numerical parity with the
          pre-change baseline (PCC > 0.999)
    - [ ] Run the warm-up guard test: deliberately pass a sharded cos/sin tensor during warm-up
          and confirm the guard still raises
    - [ ] Run two consecutive traced decode steps and compare outputs to a non-traced reference

- `test_plan.md`
  - **Test 1 — Non-traced forward correctness:**
    - Setup: load `TTNNQwen3FullAttention` with `move_weights_to_device_impl` including the new
      pre-allocated buffers; run a single forward pass with `TracedRun` disabled
    - Input: random BF16 query/key/value of shape `[batch, n_heads, 1, head_dim]`, cos/sin of
      shape matching the pre-allocated buffer, `_decode_cur_pos` = 0
    - Validation: PCC > 0.999 against a float32 CPU reference; confirm output tensor shape is
      unchanged
    - Purpose: ensures the `ttnn.copy` replacement does not change the numerical result of a
      single forward pass
  - **Test 2 — Trace capture and single-replay correctness:**
    - Setup: compile run (one non-traced forward), then `begin_trace_capture` / one forward /
      `end_trace_capture`, then one `execute_trace`
    - Validation: output of the traced replay matches the non-traced reference (PCC > 0.999)
    - Purpose: confirms that the pre-allocated buffers are correctly populated during the capture
      run and that the traced command sequence is valid
  - **Test 3 — Multi-step replay consistency:**
    - Setup: run 8 consecutive `execute_trace` calls, each time updating cos/sin via `ttnn.copy`
      before the trace executes (this happens inside the trace, but the test drives the outer
      loop with different position indices)
    - Validation: each step's output matches the non-traced reference for the same position index
    - Purpose: confirms that `ttnn.copy` inside the trace correctly updates the buffers before
      the rotary embedding op reads them on each replay iteration
  - **Test 4 — Warm-up guard preservation:**
    - Setup: bypass `move_weights_to_device_impl` and directly call `TTNNQwen3FullAttention.
      forward` with sharded cos/sin tensors (as would occur if pre-allocation were missing or
      misconfigured)
    - Validation: `TTNNRotaryPositionEmbedding.forward` raises the expected error about wrong
      cos/sin dimensionality
    - Purpose: confirms the guard is not silently disabled by the pre-replication change
  - **Test 5 — Full hybrid decoder trace (integration smoke test):**
    - Setup: run the full `TTNNQwen3FullAttention`-containing decoder block (including both
      full-attention and delta-net layers) under `TracedRun` with a realistic batch size
    - Validation: output PCC > 0.99 against non-traced reference; no `ttnn.from_torch` calls
      observed inside the trace (verified by enabling a debug hook that logs host operations)
    - Purpose: end-to-end smoke test confirming that pre-replication unblocks full-stack trace
      capture for the Qwen3.6-35B-A3B hybrid decoder

- `prefill_scope_note.md`
  - Document explicitly that this guide addresses the decode trace path only; the prefill path
    for `TTNNQwen3FullAttention` has a different cos/sin lifecycle (sequence-length-varying cos/sin
    that cannot be pre-allocated to a fixed shape) and requires separate treatment
  - Note that prefill traces for variable sequence lengths are typically handled by a separate
    trace per supported sequence length (see the `trace_id_prefill` keying strategy in
    `Generator`); cos/sin pre-allocation for prefill would require one pre-allocated buffer per
    supported sequence length
  - Recommend deferring prefill cos/sin pre-allocation until after the decode trace path is
    verified; the `_ensure_replicated` call can be retained for the prefill path in the
    short term because the prefill trace (if enabled) is captured once per sequence length
    at warm-up time and is not subject to the same per-step update requirement

---

## 3. Conventions

### Terminology

| Term | Meaning in this guide |
|---|---|
| Metal Trace | The TTNN subsystem that records device commands during a capture run and replays them verbatim; accessed via `ttnn.begin_trace_capture`, `ttnn.end_trace_capture`, `ttnn.execute_trace` |
| capture run | The single forward pass executed inside the `begin_trace_capture` / `end_trace_capture` bracket; its device commands are recorded into the trace command buffer |
| replay | Any subsequent call to `ttnn.execute_trace`; re-issues the recorded device commands without host re-execution or buffer reallocation |
| host operation | Any operation that runs on the CPU host and is not recorded by Metal Trace; `ttnn.from_torch` is the canonical example |
| buffer address stability | The requirement that every device tensor touched during a capture run exists at the same device memory address on every subsequent replay |
| `_decode_cur_pos` | A scalar device tensor pre-allocated in `move_weights_to_device_impl` and updated via `ttnn.copy` inside the traced region at each decode step; serves as the canonical pattern for pre-allocated decode-step inputs |
| `_ensure_replicated` | The helper method in `TTNNQwen3FullAttention` that calls `ttnn.from_torch` to produce a replicated cos/sin tensor when the input is detected as sharded; trace-incompatible because `ttnn.from_torch` allocates a new device buffer |
| pre-replication | The technique of allocating a replicated device buffer before trace capture and updating it via `ttnn.copy` inside the traced region; the fix that replaces `_ensure_replicated` |
| `_cos_replicated` / `_sin_replicated` | The pre-allocated replicated device tensors introduced by this fix; allocated in `move_weights_to_device_impl`, updated via `ttnn.copy` inside `forward` |
| `move_weights_to_device_impl` | The method called during model initialization (before any trace capture) that transfers weight tensors and auxiliary tensors to device; the correct location for pre-replication allocation |
| `TracedRun` | The tt-symbiote execution mode class that captures and replays Metal Traces; wraps `@trace_enabled` module `forward` calls |
| `_alloc_kwarg_tensor` | A method on `TracedRun` that pre-allocates device buffers for certain keyword arguments; examined in Chapter 2 to determine whether it handles cos/sin |
| replicated tensor | A device tensor where every device in the mesh holds a full copy of the data; the required distribution mode for cos/sin in TP inference |
| sharded tensor | A device tensor where each device holds a different slice of the data along some axis; the incorrect distribution mode for cos/sin that caused the original crash |
| `ReplicateTensorToMesh` | The tt-symbiote / TTNN mesh mapper that produces a replicated tensor on `ttnn.from_torch` |
| warm-up guard | The `rotary_dim % 64 != 0` check in `TTNNRotaryPositionEmbedding.forward` that detects wrongly-sharded cos/sin inputs and raises an error |
| non-tile-aligned `rotary_dim` | A `rotary_dim` value that is not a multiple of 32; causes `TILE_LAYOUT` to require padding; handled by the separate "partial rotary embedding numerical correctness" research topic |
| PCC | Pearson Correlation Coefficient; used in TTNN tests to compare output tensors against a float32 reference; threshold is > 0.999 for BF16 rotary embedding outputs |
| T3K | An 8-device Wormhole mesh arranged as a 1×8 logical ring; the target hardware for Qwen3.6-35B-A3B TP inference |
| DRAM | Device DRAM on each Wormhole chip; the recommended memory tier for pre-allocated cos/sin buffers in this guide |
| L1 | On-chip SRAM on each Wormhole core; not recommended for cos/sin pre-allocation due to limited capacity and the need to accommodate batch-size variation |
| `TILE_LAYOUT` | TTNN tensor layout aligned to 32×32 tiles; required by `ttnn.experimental.rotary_embedding` for compute |
| `ROW_MAJOR_LAYOUT` | TTNN tensor layout without tile alignment; lower memory overhead for small tensors but requires conversion to `TILE_LAYOUT` before compute ops |
| `ttnn.copy` | TTNN op that writes into an existing destination buffer without allocation; trace-safe when both source and destination are pre-existing device tensors |
| `ttnn.clone` | TTNN op that allocates a new destination buffer and copies into it; trace-unsafe inside a captured region |

### Notation

- All TTNN Python API symbols use inline code with the `ttnn.` prefix: `ttnn.begin_trace_capture`,
  `ttnn.copy`, `ttnn.experimental.rotary_embedding`.
- Class and method names in the tt-symbiote codebase use inline code without the module prefix:
  `TTNNQwen3FullAttention`, `move_weights_to_device_impl`, `_ensure_replicated`.
- File paths are given relative to the tt-symbiote repository root in inline code:
  `models/tt_symbiote/nn/attention/qwen3_full_attention.py`.
- Tensor shapes use square brackets with commas: `[batch, n_heads, 1, head_dim]`.
- The `rotary_dim` symbol always refers to the number of real dimensions receiving RoPE rotation
  (an even integer); it is `2 * (number of rotation pairs)`.
- Trace phase labels are typeset in bold when referring to the phase lifecycle: **compile run**,
  **capture run**, **replay**.
- Hardware device counts on T3K are written as "8 devices" or "T3K mesh (1×8)"; never abbreviated
  to a number alone.

### Formatting Rules

- Each `.md` file begins with an H1 title matching the file's topic, followed by a one-paragraph
  orientation that states what the reader will know by the end of the file.
- Code patterns are shown in fenced code blocks with `python` language tags; comments annotate
  non-obvious lines with `# why:` prefix.
- Callout blocks use blockquote syntax with a bold label:
  - `> **Note:**` for clarifications that prevent common misunderstandings
  - `> **Warning:**` for actions that will silently produce wrong results or crash the device
  - `> **Key Finding:**` for direct answers to the five research questions
  - `> **Trace Invariant:**` for statements of the buffer-address-stability contract
- Every chapter's `index.md` ends with a "What's next" section listing the files in that chapter
  in reading order.
- Checklist items use `- [ ]` GitHub-flavored markdown syntax.
- No emoji in any file.
- Forward references to other chapters use relative markdown links:
  `../ch3_prereplication_impl/downstream_op_constraints.md`.
- When a code snippet shows the pre-allocation pattern, the `# BEFORE:` / `# AFTER:` comment
  convention is used to distinguish the removed `_ensure_replicated` call from the replacement
  `ttnn.copy` call.

---

## 4. Cross-Chapter Dependencies

```
Chapter 1 (Why ttnn.from_torch Breaks Metal Trace)
  - Introduces: host operation definition, buffer address stability, what trace records,
    what replay forbids, _ensure_replicated call site and the original sharding bug
  - Required by: all subsequent chapters (the incompatibility rationale motivates every
    subsequent design decision)

Chapter 2 (The _decode_cur_pos Pre-Allocation Pattern)
  - Depends on: Chapter 1 (buffer address stability constraint, ttnn.copy vs ttnn.from_torch
    distinction)
  - Introduces: _decode_cur_pos lifecycle, the generalizable pre-allocation pattern,
    TracedRun._alloc_kwarg_tensor and its limitations for cos/sin, rationale for placing
    cos/sin pre-allocation in move_weights_to_device_impl
  - Required by: Chapter 3 (the pre-allocation design follows the _decode_cur_pos model),
    Chapter 6 (integration checklist references the pattern from Ch2)

Chapter 3 (Pre-Allocating Replicated cos/sin Buffers)
  - Depends on: Chapter 1 (why _ensure_replicated must be replaced),
    Chapter 2 (_decode_cur_pos pattern to adapt, move_weights_to_device_impl placement)
  - Introduces: downstream op constraints (TILE_LAYOUT, DRAM_MEMORY_CONFIG, shape),
    ReplicateTensorToMesh mapping, concrete code changes to move_weights_to_device_impl
    and TTNNQwen3FullAttention.forward
  - Required by: Chapter 4 (the ttnn.copy trace-safety analysis assumes the pre-allocation
    design from Ch3), Chapter 5 (the warm-up guard analysis assumes _cos_replicated exists
    as described in Ch3), Chapter 6 (integration steps implement the design from Ch3)

Chapter 4 (Trace Safety of ttnn.copy to a Replicated Destination)
  - Depends on: Chapter 1 (what trace records, DMA commands vs host allocations),
    Chapter 3 (the pre-allocated _cos_replicated buffer whose copy-into is being analyzed)
  - Introduces: what ttnn.copy enqueues in the command buffer, source tensor stability
    requirement, replay correctness verification methodology, PCC threshold for comparison
  - Required by: Chapter 6 (test_plan.md references the replay correctness verification
    from Ch4 as Test 3)

Chapter 5 (Warm-Up Guard Preservation)
  - Depends on: Chapter 1 (the original _ensure_replicated guard context),
    Chapter 3 (_cos_replicated buffer whose replication is what the guard now checks)
  - Introduces: rotary_dim % 64 guard mechanism, why it still fires correctly after
    pre-replication, debug assertion recommendation, non-tile-aligned rotary_dim interaction
    and scope boundary with the separate partial RoPE topic
  - Required by: Chapter 6 (integration_checklist.md and test_plan.md reference the guard
    and the debug assertion from Ch5)

Chapter 6 (End-to-End Integration and Test Strategy)
  - Depends on: all prior chapters
    - Ch1: trace incompatibility motivation (drives why the checklist items are ordered
      as they are)
    - Ch2: _decode_cur_pos pattern (the checklist verifies parity with this baseline)
    - Ch3: concrete code changes (the integration steps implement them)
    - Ch4: replay correctness (Tests 2 and 3 implement the verification protocol)
    - Ch5: warm-up guard (Test 4 implements the guard preservation check)
  - Introduces: no new concepts; synthesizes prior chapters into ordered integration steps,
    a five-test verification plan, and a prefill scope note that defers prefill cos/sin
    pre-allocation to future work
  - Serves as the operational reference for the engineer implementing the fix
```

**Explicit forward references to flag in chapter content:**

- **Ch1 → Ch2:** `ensure_replicated_call_site.md` identifies that a stable buffer is needed;
  flag readers that the `_decode_cur_pos` pattern showing how to create one is in Ch2.
- **Ch2 → Ch3:** `pattern_generalization.md` identifies the three properties of a correct
  pre-allocation; flag readers that the concrete implementation for cos/sin is in Ch3.
- **Ch2 → Ch3:** `traced_run_alloc_kwarg_tensor.md` concludes that `move_weights_to_device_impl`
  is preferred; flag readers that the specific code changes are in Ch3.
- **Ch3 → Ch4:** `move_weights_impl_changes.md` introduces `ttnn.copy` as the update mechanism;
  flag readers that the trace-safety analysis for `ttnn.copy` is in Ch4.
- **Ch3 → Ch5:** `replicated_mesh_mapping.md` mentions the warm-up guard; flag readers that the
  guard analysis after pre-replication is in Ch5.
- **Ch4 → Ch6:** `replay_correctness_verification.md` describes the verification methodology;
  flag readers that the test plan implementing it is in Ch6.
- **Ch5 → Ch5 (internal):** `non_tile_aligned_rotary_dim_interaction.md` cross-references the
  separate partial RoPE research topic; this is a lateral reference, not a chapter dependency.
- **Ch5 → Ch6:** `guard_adequacy_after_change.md` recommends a debug assertion; flag readers
  that the assertion appears in the integration checklist in Ch6.
```
