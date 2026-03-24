# Prefill Warm-Up

This file explains the prefill warm-up phase in detail. By the end you will understand how `warmup_model_prefill` sweeps power-of-two sequence lengths, why the `model_id` loop runs both a compile pass and a trace capture pass on every participating mesh, what tensors are used, how the `warmup_prefill` parameter gates the warm-up on first call, and when chunked-prefill warm-up is silently skipped.

## Entry Point: `warmup_model_prefill` in `generator.py`

`warmup_model_prefill` is defined on `Generator` in `models/tt_transformers/tt/generator.py`. It is protected by an `already_warmed_up_prefill` guard so that the first call performs the full sweep and every subsequent call returns immediately:

```python
def warmup_model_prefill(self, kv_cache, enable_trace, can_sample_on_device, non_greedy_decoding_on_device):
    if self.already_warmed_up_prefill:
        return
    self.already_warmed_up_prefill = True

    sequence_lengths_to_warmup = self.model_args[0].get_warmup_prefill_supported_seq_lens()
    ...
```

The guard boolean is initialized to `False` in `__init__`. `already_warmed_up_prefill = True` is set immediately after the early-return check — before any forward pass executes. This is a single-entry gate, not a safety feature: it prevents the method body from running more than once, but it does not protect against failures inside warm-up.

> **Warning:** Because `already_warmed_up_prefill` is set to `True` before the first forward pass executes, any exception thrown during warm-up (kernel compile failure, OOM, device timeout, or any other error) leaves the guard permanently `True`. Every subsequent call to `warmup_model_prefill` hits the `if self.already_warmed_up_prefill: return` check on line 11 and returns immediately — no forward pass, no error, no log message. The model continues in an un-warmed state: no kernels compiled, no traces captured. Callers that implement a retry loop or an exception handler around `warmup_model_prefill` must explicitly reset `already_warmed_up_prefill = False` after catching an exception before calling `warmup_model_prefill` again; otherwise the retry silently does nothing.

## Sequence-Length Set: `get_warmup_prefill_supported_seq_lens`

`get_warmup_prefill_supported_seq_lens` is a method on `ModelArgs` in `models/tt_transformers/tt/model_config.py`. It assembles the full list of sequence lengths to compile by calling `calculate_prefill_warmup_seq_lens` from `models/tt_transformers/tt/common.py`.

### `capped_warmup_seq_len`

`capped_warmup_seq_len` is computed once during `ModelArgs.__init__`:

```python
self.capped_warmup_seq_len = min(self.max_prefill_chunk_size, self.max_seq_len)
```

It is the ceiling on the sequence lengths that will be warmed up; no length longer than this is ever compiled during warm-up.

### `calculate_prefill_warmup_seq_lens`

`calculate_prefill_warmup_seq_lens(max_seq_len_to_warmup, trace_supported_seq_lens)` in `models/tt_transformers/tt/common.py` builds the list as follows:

1. Call `get_all_padded_prefill_lengths(max_seq_len_to_warmup)`, which returns `[128, 1024, 2048, 4096, ...]` — 128 followed by powers of two starting from 1024 (i.e., 1024, 2048, 4096, 8192, …) up through `max_seq_len_to_warmup`. The values 256 and 512 are intentionally absent: `get_padded_prefill_len` maps any sequence length from 1 through 128 to 128, and any length from 129 through 1024 to 1024, so 256 and 512 are not valid pad-length boundaries. Warming them up would compile kernels for sizes that are never actually dispatched.
2. Append any `trace_supported_seq_lens` that are not already in the list.
3. Sort the combined list.

This ensures the warmup sweep covers both the standard power-of-two breakpoints used by `get_padded_prefill_len` and any extra lengths that the trace infrastructure requires.

`get_warmup_prefill_supported_seq_lens` then applies model-specific overrides (for example, Qwen3-32B caps at 4096 to avoid a known hang) and calls `filter_warmup_seq_lens` for device-specific filtering before returning the final sorted list.

> **Note:** `capped_warmup_seq_len` must be a power of 2; `get_warmup_prefill_supported_seq_lens` asserts this at runtime. The assertion fires on the *result* of `min(max_prefill_chunk_size, max_seq_len)`, not on `max_prefill_chunk_size` directly. If `max_seq_len` is smaller than `max_prefill_chunk_size` and is itself a power of 2, the `min()` produces a power-of-2 value and the assertion passes even if `max_prefill_chunk_size` is non-power-of-2. Conversely, if both values are non-power-of-2, or if `max_prefill_chunk_size` is the smaller and is non-power-of-2, the assertion fires. The correct trigger condition is: the assertion fires whenever `capped_warmup_seq_len` (the `min` result) is not a power of 2.

## The `model_id` Loop: Compile and Capture Per Mesh

`warmup_model_prefill` iterates over `range(self.data_parallel)`, where `self.data_parallel` equals the number of model replicas (meshes). The purpose is to control which sequence lengths each mesh warms up, and to perform the required compile-then-capture sequence on every mesh that participates in trace replay:

- **`model_id == 0`:** every sequence length in `sequence_lengths_to_warmup` is run. The first forward pass at each length compiles the TT-Metal kernels; the second (when `enable_trace=True`) captures the trace. Each submesh holds its own independent program cache, so kernels compiled on `model_id == 0` are not available on other meshes — the skip condition is simply structured so that only `model_id == 0` runs the full length set.
- **`model_id > 0`:** behaviour depends on `enable_trace`:
  - **`enable_trace=True`:** the inner loop skips any length not in `self.model_args[0].trace_prefill_supported_seq_lens`. For each trace-supported length on additional meshes, the warm-up performs **both** a compile run and a trace capture pass (two forward calls per length) — the same two-pass structure as `model_id == 0`. The compile run is required first because each submesh holds an independent program cache and has no kernels compiled yet for that length; attempting trace capture without a prior compile run on that submesh will fail. Purely compile-time lengths (those not in `trace_prefill_supported_seq_lens`) are skipped entirely on additional meshes.
  - **`enable_trace=False`:** `not enable_trace` evaluates to `True` unconditionally, so the `or` clause fires for **every** sequence length. All lengths are skipped on every `model_id > 0` replica. Only `model_id == 0` ever runs prefill warm-up when tracing is disabled.

> **Warning:** In a multi-device deployment with `enable_trace=False`, only the first mesh (`model_id == 0`) has its prefill kernels compiled during warm-up. All additional meshes receive no warmup compilation at all and will fall back to JIT compilation on their first real inference call. With `enable_trace=True`, additional meshes skip non-trace-supported lengths but do run the capture pass for trace-supported lengths; they still require their own compile run at that point because each submesh holds an independent program cache and does not inherit compiled programs from `model_id == 0`.

```python
for model_id in range(self.data_parallel):
    for supported_length in sequence_lengths_to_warmup:
        if model_id != 0 and (
            supported_length not in self.model_args[0].trace_prefill_supported_seq_lens or not enable_trace
        ):
            continue
        ...
```

## Warm-Up Token Tensors

For each `(model_id, supported_length)` pair the loop constructs zero-filled tensors:

```python
warmup_tokens = torch.zeros(batch_size, supported_length, dtype=torch.long)
warmup_prompt_lens = torch.tensor([supported_length] * batch_size, dtype=torch.long)
warmup_empty_slots = list(range(batch_size))
```

When paged attention is active, a matching page table is allocated:

```python
block_size = get_block_size(kv_cache[model_id])
num_blocks = num_blocks_in_seq(supported_length, block_size)
page_table_warmup = torch.zeros(batch_size, num_blocks, dtype=torch.int32)
```

If `kv_cache` is `None` (or its entry for `model_id` is `None`), `page_table_warmup` is left as `None`, which disables paged attention for that warm-up call.

The batch size loop iterates over `[1, 32]`; however, the `batch_size == 32` branch is currently skipped with a `continue` until batched prefill is supported. In practice only `batch_size=1` warm-up runs execute today.

## Sampling Parameters in Prefill Warm-Up

Prefill warm-up calls `_create_sampling_params` (inherited from `WarmupForwardMixin`) exactly once across the entire warm-up sweep, controlled by the `sampling_parameters_sweeped` flag. The flag is initialized to `False` **outside** the `model_id` loop, so it is shared across all `model_id` iterations:

```python
sampling_parameters_sweeped = False          # initialized outside model_id loop

for model_id in range(self.data_parallel):
    for supported_length in sequence_lengths_to_warmup:
        if model_id != 0 and (...):
            continue                          # skips the entire loop body, including the flag assignment
        ...
        if not sampling_parameters_sweeped:
            sampling_params = self._create_sampling_params(...)
        else:
            sampling_params = [None]
        ...
        sampling_parameters_sweeped = True   # set at the bottom of the loop body, after the sampling call
```

The `sampling_parameters_sweeped = True` assignment sits at the **bottom** of the `supported_length` loop body. Early-exit paths — specifically the `continue` on `model_id != 0` — jump past this assignment entirely. As a result, the flag is set on the **first `(model_id, supported_length)` pair whose loop body executes to completion**, which in practice is `model_id == 0` with the first sequence length that does not hit a `break` or `continue`. Because `model_id == 0` never hits the `model_id != 0` guard, and because the `skip_sequence_lengths` path would prevent the loop body from running for later lengths anyway, the sweep reliably executes exactly once — on the first fully-executed iteration at `model_id == 0`. What is **not** guaranteed is that it will execute on the very first sequence length; if the first length triggers `skip_sequence_lengths` and breaks out before the flag line, the next iteration would find `sampling_parameters_sweeped` still `False` and re-run the sweep.

Once the flag is set, all subsequent iterations — including those for `model_id > 0` that do execute their loop body — use `sampling_params = [None]`.

This reflects the fact that on-device sampling in prefill is sequence-length-agnostic: the sampling kernel graph is determined by the sampling config, not the prefill length or the mesh index. Sweeping once on `model_id == 0` is sufficient; additional meshes do not repeat the sampling-variant compilation.

## The `warmup_prefill` Gate in `prefill_forward_text`

`prefill_forward_text` in `generator.py` accepts a `warmup_prefill=True` keyword argument:

```python
def prefill_forward_text(
    self,
    tokens: torch.Tensor,
    page_table=None,
    kv_cache=None,
    prompt_lens=None,
    empty_slots=None,
    enable_trace=True,
    model_id_warmup=None,
    sampling_params=None,
    start_pos=None,
    return_hidden_states=False,
    warmup_prefill=True,
    **kwargs,
):
    ...
    if warmup_prefill:
        ...
        self.warmup_model_prefill(
            kv_cache=kv_cache,
            enable_trace=enable_trace,
            can_sample_on_device=sampling_on_device_enabled,
            non_greedy_decoding_on_device=sampling_on_device_enabled,  # same value: intentionally conservative
        )
```

> **Note — same value for both parameters (intentional):** Both `can_sample_on_device` and `non_greedy_decoding_on_device` receive `sampling_on_device_enabled` — the same boolean derived from whether the model's `sampling` attribute is present and `_supports_on_device_sampling` is `True`. This is verified against the source at `models/tt_transformers/tt/generator.py` (lines 300–309) and is not a transcription error.
>
> The design is intentionally conservative: `prefill_forward_text` has no knowledge of whether the caller will ever issue non-greedy requests at runtime. By passing the same value for both flags, the warm-up always captures the full 6-variant sampling sweep (greedy + `None` + 4 non-greedy Cartesian-product configs) whenever on-device sampling is supported, regardless of whether the caller believes only greedy decoding will be used. The 4 extra non-greedy variants are harmless if sampling is never exercised in production — the captured traces simply go unused — but their absence would cause a slow JIT path on the first non-greedy request in any deployment that later enables sampling. A caller who wants to restrict warm-up to 2 variants (greedy + `None`) must pass `non_greedy_decoding_on_device=False` explicitly when constructing a dedicated warm-up call, rather than relying on the `warmup_prefill` gate inside `prefill_forward_text`.

When `warmup_prefill=True` (the default), calling `prefill_forward_text` for the very first time triggers warm-up automatically before the real prefill proceeds. The `already_warmed_up_prefill` guard ensures that the warm-up block is a no-op on all subsequent calls.

Test code that wants to exercise `prefill_forward_text` without triggering warm-up passes `warmup_prefill=False` explicitly. This is common in unit tests that need to control compilation state or that supply pre-compiled op graphs.

> **Note:** Passing `warmup_prefill=False` in production code means the model is invoked before any trace has been captured. The first real prefill call will then trigger an expensive JIT-compile path. Only disable warm-up in tests that explicitly manage the compilation lifecycle.

## Skipping Warm-Up for Chunked Prefill Without Paged Attention

There is one skip path in the warm-up loop. When `page_table_warmup is None` (paged attention is disabled) and the current sequence length exceeds `self.model_args[0].max_prefill_chunk_size`, chunked prefill cannot be exercised without a page table. The code emits a warning and breaks out of the sequence-length loop:

```python
if page_table_warmup is None and max_prefill_chunk_size_cutoff(
    supported_length, self.model_args[0].max_prefill_chunk_size
):
    logger.warning(
        f"Skipping warmup for sequence lengths after: {supported_length} because they are "
        f"greater than the max prefill chunk size and paged attention is disabled"
    )
    skip_sequence_lengths = True
    break
```

If you see this warning during start-up it means the model is running without paged attention and some long sequence lengths will not have compiled kernels or captured traces. Requests with those lengths will fall back to a slow compile path at runtime.

> **Warning:** The `skip_sequence_lengths = True` flag breaks out of the inner `batch_size` loop. After the `batch_size` loop exits, the check `if skip_sequence_lengths: break` at the bottom of the `supported_length` loop body exits the **`supported_length` loop** for the current `model_id`, skipping all remaining sequence lengths for that replica. Because `skip_sequence_lengths` is initialized **outside** the `model_id` loop (and is never reset inside it), subsequent `model_id` iterations also exit the `supported_length` loop immediately on their first iteration — so no further sequence lengths are warmed up on any replica once the flag is set. The `model_id` loop itself continues to its natural end; it is the `supported_length` loop that is short-circuited on each subsequent replica.

---

**Next:** [`differentiating_warmup_from_production.md`](./differentiating_warmup_from_production.md)
