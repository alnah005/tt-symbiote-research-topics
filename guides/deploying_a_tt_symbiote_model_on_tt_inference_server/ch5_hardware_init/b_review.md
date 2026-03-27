## Pass 1

**1. `device_lifecycle.md` — T3K device count is wrong (`TT_VISIBLE_DEVICES` example)**

`device_lifecycle.md` line 109 states:
> `TT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` makes all eight chips of a T3K visible.

A T3K board contains **4 chips** (four N300 cards wired together), not 8. The mesh shape for T3K is `(1, 8)` because each N300 chip exposes **2 Wormhole dies**, giving 8 dies total — but the physical device IDs as seen by `tt-smi` are 4 entries (one per N300 card), not 8. A reader following the example literally would request IDs `0–7` for a T3K, which would either fail or accidentally capture devices belonging to a second T3K on the same host. The `environment_variables_reference.md` correctly gives the T3K-on-an-8-card-host example as `TT_VISIBLE_DEVICES=0,1,2,3` / `TT_VISIBLE_DEVICES=4,5,6,7`, which contradicts the `device_lifecycle.md` claim.

**2. `model_init_responsibilities.md` — `ttnn.allocate_tensor_on_device` is not a real TTNN API**

The KV cache allocation snippet (lines 123–130) calls `ttnn.allocate_tensor_on_device(...)`. This function does not exist in the public `ttnn` API. The correct call is `ttnn.zeros(shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)` or `ttnn.Tensor(...).to(device, memory_config)`. A developer copying this snippet will get an `AttributeError` at runtime.

**3. `device_lifecycle.md` — Galaxy fabric topology is described as "1D ring" but the mesh shape entry implies a 2D mesh**

Line 51 states Galaxy clusters use a **1D ring** fabric topology. However, the mesh shape table entry for Galaxy is `ttnn.MeshShape(8, 4)` — a 2D mesh. A 2D mesh requires a **2D torus** or **2D mesh** fabric topology for efficient CCL (all-reduce) operations; a 1D ring over a 2D mesh means every cross-row collective must traverse the ring the long way, which is a known performance trap. If the server genuinely configures a 1D ring for Galaxy, this is a significant behavioral detail that needs explicit justification. As written, a reader implementing a Galaxy model will assume 1D ring and may write CCL patterns that are unexpectedly slow, or assume the topology matches the mesh dimensions when it does not.

**4. `model_init_responsibilities.md` — `deallocate_weights` guidance contradicts itself within the same paragraph**

Lines 70–71 state: "Do not call `deallocate_weights` during `initialize_vllm_model`" and then two sentences later: "If memory pressure requires it, call `deallocate_weights` after the warmup sweeps have completed and before returning." Calling it "before returning" is still inside `initialize_vllm_model`. The prohibition and the exception directly contradict each other in a way that will confuse implementers about whether the call is permitted or not.

**5. `index.md` — Non-clickable/missing file reference does not exist as a structural gap but the `Reading Order` list omits the navigation footer target**

No additional qualifying structural issue beyond items 1–4. Item 5 is withheld — no fifth issue rises to the materiality threshold.

---

*Issues 1 and 2 are implementation-breaking. Issue 3 is a potentially material conceptual mislabel. Issue 4 produces directly contradictory implementation instructions.*

## Pass 2

All four issues flagged in Pass 1 are resolved in the current files. One new issue was found.

**1. `device_lifecycle.md` — T3K mesh shape in the table contradicts the stated die count**

The mesh shape table (lines 28–35) maps `"T3K"` to `ttnn.MeshShape(1, 4)`. However, line 109 of the same file states: "T3K consists of 4 physical N300 cards (8 Wormhole dies total)." Eight dies arranged as a 1D mesh should produce `ttnn.MeshShape(1, 8)`, not `(1, 4)`. This is further confirmed by `environment_variables_reference.md` line 62, which cites `"1×8" for a T3K-optimized attention kernel"` as a realistic example, and by `model_init_responsibilities.md` line 98, which shows a mesh shape assertion example of `(1, 8)`. A developer who reads the table and sets `MESH_DEVICE=T3K` will receive a `(1, 4)` mesh and unknowingly run the model on only half the T3K's dies, silently degrading throughput and producing incorrect tensor sharding for kernels that expect 8 chips.

No further qualifying issues found.

---

*Issue 1 is implementation-breaking: it causes silent use of half the available T3K hardware.*

## Pass 3

The Pass 2 issue (T3K mesh shape `(1,4)` vs `(1,8)`) is resolved in the current files. One new qualifying issue found.

**1. `environment_variables_reference.md` — T3K device-ID count contradicts `device_lifecycle.md`**

`environment_variables_reference.md` line 67 states:
> `TT_VISIBLE_DEVICES=4,5,6,7` restricts the process to the second T3K in an eight-card host, while another worker process uses `TT_VISIBLE_DEVICES=0,1,2,3` for the first T3K.

This implies each T3K occupies **4** device IDs. However, `device_lifecycle.md` line 109 states:
> T3K consists of 8 N300 modules (16 Wormhole chips total) — `tt-smi` reports **8 device IDs** (0, 1, 2, 3, 4, 5, 6, 7) for a T3K host.

The two files directly contradict each other on how many physical device IDs a T3K consumes. If a developer follows `environment_variables_reference.md` and sets `TT_VISIBLE_DEVICES=0,1,2,3` for a T3K, they will only expose 4 of the 8 required device IDs to the `ttnn` runtime. `open_mesh_device` with `MeshShape(1, 8)` will then fail (or silently construct a degraded mesh) because only 4 devices are enumerable. The multi-T3K host partitioning example in `environment_variables_reference.md` must be corrected to use 8-ID ranges (e.g., `0,1,2,3,4,5,6,7` and `8,9,10,11,12,13,14,15`).

No further qualifying issues found.

---

*Issue 1 is implementation-breaking: following the partitioning example in `environment_variables_reference.md` will cause `open_mesh_device` to fail or produce a degraded mesh on any T3K deployment.*

## Pass 4

The Pass 3 issue (T3K device-ID count contradiction between `device_lifecycle.md` and `environment_variables_reference.md`) is resolved — both files now consistently state T3K uses 8 device IDs and cannot be split. One new qualifying issue found.

**1. `model_init_responsibilities.md` — KV cache allocation example allocates only one tensor per layer instead of two (K and V separately)**

The `allocate_kv_cache` code example (lines 121–135) creates a list of tensors using:

```python
self.kv_cache = [
    ttnn.zeros(
        shape=[num_blocks, block_size, self.n_heads, self.head_dim],
        ...
    )
    for _ in range(self.n_layers)
]
```

This allocates **one tensor per layer**. A standard transformer KV cache requires **two tensors per layer** — one for keys and one for values. As written, a developer copying this pattern allocates only half the required KV cache storage. Depending on how the model's `forward()` subsequently indexes `self.kv_cache`, this either causes an index-out-of-range error at runtime (if the model expects `kv_cache[2*i]` and `kv_cache[2*i+1]`) or silently produces incorrect attention output (if K and V are both read from the same tensor). The example should either create two tensors per layer (`range(2 * self.n_layers)` with the standard pairwise indexing convention) or explicitly document that K and V are packed into the `n_heads` dimension and show how the model unpacks them.

No further qualifying issues found.

---

*Issue 1 is implementation-breaking: a developer following the example will allocate insufficient KV cache, producing either a runtime crash or silent attention correctness errors.*

## Pass 5

No feedback — chapter approved.

## Pass 6

**1. `device_lifecycle.md` — T3K hardware description gives wrong physical card and die counts**

`device_lifecycle.md` line 109 states:
> "T3K consists of 8 N300 modules (16 Wormhole chips total), but each N300 module presents as a single device ID — so `tt-smi` reports 8 device IDs (0, 1, 2, 3, 4, 5, 6, 7) for a T3K host."

This is numerically wrong. A T3K board has **4 N300 cards**, not 8. Each N300 card contains 2 Wormhole dies, giving **8 dies total** (not 16). The `tt-smi` reports 8 device IDs because each Wormhole die presents as one device ID — not because there are 8 N300 cards. A reader who accepts the "8 N300 modules, 16 Wormhole chips" description will form a false picture of the T3K's physical layout. The correct description is: T3K has 4 N300 cards × 2 Wormhole dies each = 8 dies, enumerated by `tt-smi` as device IDs 0–7. Downstream conclusions (that 8 IDs are needed for a full T3K mesh) happen to be numerically correct, but the supporting counts are wrong, which will confuse readers cross-referencing T3K specs or reasoning about partial-board topologies.

No further qualifying issues found.

---

*Issue 1 produces a materially wrong numerical understanding of T3K hardware (die count off by 2×, card count off by 2×).*

## Pass 7

**1. `environment_variables_reference.md` — `DPL` mode described as reducing per-request latency, which is incorrect**

`environment_variables_reference.md` line 107 states:
> `DPL` — Data-parallel latency mode; splits the batch across chips for lower per-request latency

Splitting a batch across chips is data parallelism. Data parallelism processes multiple requests simultaneously on separate chips and improves throughput (requests per second), not per-request latency. Per-request latency is determined by how quickly a single request moves through the model; distributing independent requests to separate chips does not shorten the critical path for any individual request. A developer reading this description will incorrectly select `DPL` mode when optimizing for response-time-sensitive workloads, and will be surprised when single-request latency is unchanged or degraded (due to reduced per-chip batch size at high concurrency). The description should say "higher throughput" or "higher request-processing parallelism," not "lower per-request latency."

No further qualifying issues found.

---

*Issue 1 is a material conceptual misstatement that will cause incorrect mode selection in latency-sensitive deployments.*

## Pass 8

**1. `environment_variables_reference.md` — `TT_VISIBLE_DEVICES` N300 example lists only 1 device ID for a 2-die card**

Line 67 states:
> `TT_VISIBLE_DEVICES=0` for the first N300, `TT_VISIBLE_DEVICES=1` for the second

An N300 card contains 2 Wormhole dies, and `device_lifecycle.md` line 109 explicitly states "each Wormhole chip presents as a single device ID." A single N300 therefore occupies 2 consecutive device IDs (e.g., `0,1` for the first card, `2,3` for the second). Setting `TT_VISIBLE_DEVICES=0` exposes only one die to the process. When the worker then calls `ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 2), ...)` for an N300 deployment, only 1 device ID is enumerable and the open will either fail or silently produce a degenerate 1-chip mesh instead of the expected 2-chip N300 mesh. A developer following this example on a multi-N300 host will partition cards incorrectly, exposing half the dies of the first card to one process and half to the other.

No further qualifying issues found.

---

*Issue 1 is implementation-breaking: the N300 isolation example gives device ID counts that are half of what is required, causing `open_mesh_device` to fail or produce a degraded mesh.*

## Pass 9

**1. `device_lifecycle.md` — Galaxy fabric description contradicts the Galaxy mesh shape on ring dimension size**

The fabric section (line 51) states for Galaxy (`(8,4)` mesh):
> "CCL operations route data along the row dimension (the ring within each row of 8 chips)"

With `ttnn.MeshShape(8, 4)`, `rows=8` and `cols=4`. A single row therefore spans **4 chips** (one entry per column), not 8. The dimension that spans 8 chips is the **column** dimension. The description inverts the row/column relationship: it attributes 8 chips to the row when a row in this mesh contains 4 chips, and a column contains 8. A developer implementing CCL patterns for Galaxy who reads this description will orient their ring along the wrong axis — routing collectives along the 4-chip row dimension instead of the 8-chip column dimension — producing incorrect or inefficient collective communication on Galaxy hardware.

No further qualifying issues found.

---

*Issue 1 is a material factual error: the ring dimension size (8 chips) is attributed to rows when rows contain 4 chips; the correct dimension is columns. This will cause incorrect CCL axis selection for Galaxy deployments.*

## Pass 10

**1. `device_lifecycle.md` — `TT_VISIBLE_DEVICES` is described as set by the workflow before the worker starts, but the pseudocode shows it is set inside `TTWorker.__init__`**

`device_lifecycle.md` line 111 states:
> "This variable is set by the server workflow before the worker process starts."

However, the pseudocode at lines 12–21 shows `self._configure_visible_devices()` — the function that sets `TT_VISIBLE_DEVICES` — called as the first line inside `TTWorker.__init__`, which runs after the worker process has already started. `environment_variables_reference.md` (Quick Reference table) also lists `TT_VISIBLE_DEVICES` as "Set by: workflow", reinforcing the pre-start interpretation.

These two accounts are mutually exclusive. If `TT_VISIBLE_DEVICES` is set by `_configure_visible_devices()` inside `__init__`, then it is not set by the external workflow before the process starts. A developer responsible for multi-worker isolation will likely rely on the textual description and attempt to set `TT_VISIBLE_DEVICES` in their deployment scripts, assuming those values will persist. But if the worker's own `__init__` unconditionally overwrites the variable, the deployment script's value is silently discarded. Conversely, if the workflow truly sets it before process start and `_configure_visible_devices()` only reads (not writes) it, then the pseudocode comment `# sets TT_VISIBLE_DEVICES` is wrong and creates the false impression that the worker controls this value. Either way, a reader configuring multi-card partitioning will be misled about the authoritative source of `TT_VISIBLE_DEVICES` and may produce deployments where device isolation is silently broken.

No further qualifying issues found.

---

*Issue 1 is a material coherence error: the stated ownership of `TT_VISIBLE_DEVICES` (workflow vs. worker `__init__`) directly contradicts the pseudocode, and will cause incorrect multi-worker device-isolation configurations.*

## Pass 11

**1. `model_init_responsibilities.md` — `initialize_vllm_model` is presented as a module-level function in the first example and as an instance method in the second example; the two are mutually incompatible**

The "minimal sketch" at lines 16–42 defines `initialize_vllm_model` as a **module-level function** with signature:

```python
def initialize_vllm_model(
    hf_model_path: str,
    mesh_device: ttnn.MeshDevice,
    model_config: dict,
    tt_cache_path: str,
    dtype: ttnn.DataType,
    **kwargs,
) -> MyTTNNModel:
```

Later, the KV cache section (lines 98–103) shows `initialize_vllm_model` as a **method on `MyTTNNModel`**:

```python
class MyTTNNModel:
    def initialize_vllm_model(self, mesh_device, ...):
```

These two signatures are mutually exclusive. The server worker calls one specific entry point: if it expects a free function, a developer who implements it as a method on the model class will see the call fail (the worker will call `initialize_vllm_model(hf_model_path, mesh_device, ...)` without a receiver, raising a `TypeError`). If the server expects a method, the free-function example is wrong in the opposite direction. A developer using both code blocks as a template — copying the function-level signature for the entry point and the method-level pattern for the class structure — will produce code that cannot be called correctly by the worker.

No further qualifying issues found.

---

*Issue 1 is implementation-breaking: the two code examples in the same file give contradictory calling conventions for `initialize_vllm_model`, and a developer following both will produce an entry point the worker cannot call.*

## Pass 12

No feedback — chapter approved.
