# Prefill Trace Flow

This file traces the execution path for a prefill step through `models/tt_transformers/tt/generator.py`, starting at `prefill_forward_text` and following the code into `_easy_trace_prefill`, `_capture_trace_prefill`, and `_prefill_forward_trace`. By the end you will understand how `can_enable_trace` in `model_config.py` gates which sequence lengths are traceable, how the `(prefill_seq_len, model_id)` pair is used as a dictionary key so that one trace per length-and-shard combination is stored, and why paged attention is a hard prerequisite for prefill tracing.

---

## Routing at `prefill_forward_text`

`prefill_forward_text` is the entry point called by vLLM and internal tests. Early in the function it enforces the paged-attention prerequisite:

```python
if page_table is not None:
    assert isinstance(page_table, torch.Tensor), "page_table mush be torch.Tensor"
else:
    # Only paged attention is supported for prefill
    enable_trace = False
```

If no `page_table` is provided, `enable_trace` is forced to `False` regardless of the caller's intent. The page table is required because the prefill trace must include the page-table lookup that maps KV-cache blocks to physical addresses; without it the recorded graph would reference a non-existent tensor.

For each user in the batch the function computes:

```python
prefill_seq_len = get_padded_prefill_len(seq_len - num_cached_tokens)
enable_trace_current_prompt = enable_trace and self.model_args[model_id].can_enable_trace(
    prefill_seq_len, num_cached_tokens
)
```

If `enable_trace_current_prompt` is `True` the user is forwarded to `_easy_trace_prefill`; otherwise `prefill_forward_single_user_text` is called (the non-trace path).

> **Key insight:** The per-user `can_enable_trace` call means different users in the same batch can take different paths. A user whose prompt happens to land on a supported length gets the low-latency trace path; others fall back to the eager path automatically.

---

## Capture Logic: `_capture_trace_prefill`

```
Python host     ┌─────────────────────────────────────────────────────────────┐
                │  _capture_trace_prefill(prefill_ids, page_table, kv_cache,  │
                │                         model_id)                           │
                │                                                             │
                │  1. prepare_prefill_inputs_trace → host_inputs              │
                │  2. copy_host_to_device → device_inputs  (compile inputs)   │
                │  3. transform_and_embed_prefill_inputs_device(...)          │
                │  4. ttnn_prefill_forward(...)      ← compile run            │
                │     "Done Compiling Model"                                  │
                │                                                             │
                │  5. copy_host_to_device → device_inputs  (fresh trace bufs) │
                │  6. begin_trace_capture(mesh_device, cq_id=0)              │
                │  7. transform_and_embed_prefill_inputs_device(...)          │
                │  8. ttnn_prefill_forward(...)      ← captured in trace      │
                │  9. end_trace_capture(mesh_device, trace_id, cq_id=0)      │
                │     "Done Capturing Prefill Trace"                          │
                │  return trace_id, tt_out_trace, *device_inputs              │
                └─────────────────────────────────────────────────────────────┘

TTNN runtime    ┌─────────────────────────────────────────────────────────────┐
                │  Compile run allocates ops; trace bracket records them      │
                └─────────────────────────────────────────────────────────────┘

Device          ┌─────────────────────────────────────────────────────────────┐
                │  device_inputs (token IDs, page table, padding mask)        │
                │  become the live buffers that future replays write into      │
                └─────────────────────────────────────────────────────────────┘
```

Two important details in this flow:

- `transform_and_embed_prefill_inputs_device` is inside the trace bracket. This means both the embedding lookup and the position-encoding transforms are part of the recorded graph, not just the Transformer forward pass.
- A fresh `copy_host_to_device` is called **after** the compile run (step 5) to allocate new device buffers that belong inside the trace region. The compile-run device buffers (step 2) are discarded; only the post-step-5 buffers are the canonical trace inputs.

---

## The Trace Key: `_easy_trace_prefill`

```python
def _easy_trace_prefill(self, prefill_ids, page_table=None, ..., model_id=-1, prefill_seq_len=None, ...):
    trace_key = f"{prefill_seq_len}_{model_id}"
    if self.trace_id_prefill[trace_key] is None:
        trace_id, tt_out_trace, *device_inputs = self._capture_trace_prefill(
            prefill_ids, page_table=page_table, kv_cache=kv_cache, model_id=model_id, ...
        )
        self.trace_id_prefill[trace_key] = trace_id
        self.trace_inputs_prefill[trace_key] = device_inputs
        self.trace_output_prefill[trace_key] = tt_out_trace

    tt_out_trace = self._prefill_forward_trace(
        self.trace_id_prefill[trace_key],
        self.trace_inputs_prefill[trace_key],
        self.trace_output_prefill[trace_key],
        prefill_ids,
        page_table=page_table,
        model_id=model_id,
        ...
    )
    return tt_out_trace
```

The key `f"{prefill_seq_len}_{model_id}"` ensures that one trace is stored **per (sequence length, data-parallel shard) pair**. Because each sequence length produces a different graph shape (different tile counts, different loop bounds) and each data-parallel shard owns its own `mesh_device`, these are orthogonal dimensions that each require their own captured trace.

`model_id` is the data-parallel shard index for the current user. It is computed in `prefill_forward_text` as:

```python
model_id = user_id // max_batch_size_per_model
```

where `user_id` is the user's global slot index in the full batch and `max_batch_size_per_model` is the maximum number of users each shard can hold. With `data_parallel=N` the generator holds N model instances (one submesh each), and `model_id` ∈ `[0, N-1]` selects which of those instances — and which submesh — handles this user. This is the same dimension iterated as `for i in range(self.data_parallel)` in the decode path; the naming difference (`model_id` vs `i`) reflects only that decode iterates all shards in one call while prefill processes one user (and therefore one shard) at a time. `model_id` is not a model-variant selector; all N instances run the same model weights.

`self.trace_id_prefill` is a `defaultdict(lambda: None)`, so a missing key is `None`, which triggers capture; a present key re-uses the existing trace.

---

## Replay: `_prefill_forward_trace`

```python
def _prefill_forward_trace(self, trace_id, device_inputs, tt_out_trace, prefill_ids, ..., page_table=None, model_id=-1, ...):
    host_inputs = self.model[model_id].prepare_prefill_inputs_trace(prefill_ids, **prefill_kwargs)
    host_inputs = (host_inputs[0], host_inputs[3], host_inputs[4])

    device_inputs = copy_host_to_device(
        host_inputs, device_tensors=device_inputs, mesh_device=self.model_args[model_id].mesh_device
    )

    ttnn.execute_trace(self.model_args[model_id].mesh_device, trace_id, cq_id=0, blocking=False)
    return tt_out_trace
```

Every replay unconditionally calls `copy_host_to_device` into the pre-allocated `device_inputs` buffers. This is different from the decode path where the copy is sometimes skipped: for prefill the input tokens and page table always change between users, so a copy is always required.

The rotary-matrix pointers (`tt_rot_mats_prefill_global`, `tt_rot_mats_prefill_local`) extracted during capture point to the global cos/sin matrices allocated by `RotarySetup` and do not change between users, so they are not re-copied.

`ttnn.execute_trace` is called with `blocking=False`, returning immediately while the device executes the recorded graph.

---

## `can_enable_trace` in `model_config.py`

`can_enable_trace` in `models/tt_transformers/tt/model_config.py` is the single gating function:

```python
def can_enable_trace(self, prefill_seq_len, num_cached_tokens=0):
    allowed_seq_lens = self.trace_prefill_supported_seq_lens
    return (
        prefill_seq_len in allowed_seq_lens
        and prefill_seq_len <= self.max_prefill_chunk_size
        and prefill_seq_len <= self.max_seq_len
        and num_cached_tokens == 0
    )
```

Four conditions must all be satisfied:

| Condition | Reason |
|---|---|
| `prefill_seq_len in allowed_seq_lens` | Only discrete padded lengths have pre-warmed traces |
| `prefill_seq_len <= self.max_prefill_chunk_size` | Chunked prefill is not yet supported with tracing (GitHub issue #32056) |
| `prefill_seq_len <= self.max_seq_len` | Safety upper bound from the model config |
| `num_cached_tokens == 0` | Prefix caching is not yet supported with tracing |

> **Warning:** If `num_cached_tokens > 0`, `can_enable_trace` returns `False` even if the sequence length would otherwise qualify. This means users whose prompts have a cached prefix always take the non-trace path until prefix-caching trace support is added.

`self.trace_prefill_supported_seq_lens` is populated at `ModelArgs.__init__` time by calling `self.get_trace_prefill_supported_seq_lens()`, which is described in detail in `model_config_trace_settings.md`.

---

**Next:** [`model_config_trace_settings.md`](./model_config_trace_settings.md)
