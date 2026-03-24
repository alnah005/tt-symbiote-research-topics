# Decode Warm-Up

This file explains exactly what happens during the decode warm-up phase. By the end you will understand how `WarmupForwardMixin.warmup_model_decode` builds zero-filled tensors at maximum batch size, sweeps every sampling configuration through `decode_forward` to ensure all trace variants are compiled and captured, and where the observable log boundaries appear in stdout.

## The `WarmupForwardMixin` Mixin

`WarmupForwardMixin` is defined in `models/common/warmup/warmup_utils.py`. Any generator class that needs decode warm-up inherits from it. The mixin requires the inheriting class to expose `self.decode_forward()`. The three entry points it provides are:

- `_create_decode_warmup_inputs` — allocates the zero-filled tensors.
- `_create_sampling_params` — builds the list of `SamplingParams` objects to sweep.
- `warmup_model_decode` — orchestrates the sweep and calls `decode_forward` for each config.

## Building Zero-Filled Decode Inputs

`_create_decode_warmup_inputs(max_batch_size, num_blocks)` allocates the tensors that stand in for real data during warm-up:

```python
tokens    = torch.zeros(max_batch_size, 1, dtype=torch.int32)
start_pos = torch.zeros(max_batch_size, dtype=torch.int32)
page_table = torch.zeros(max_batch_size, num_blocks, dtype=torch.int32)
```

The token tensor has shape `(max_batch_size, 1)` because decode always processes one token per user per step. `start_pos` carries the position index for each slot; zeroing it is safe because the KV-cache is also zeroed at this point. The page table is zeroed so that every slot maps to block 0, which is a valid page-table entry for a freshly allocated KV-cache. Using maximum batch size ensures the compiled and captured kernels cover the worst-case memory layout; smaller real batches will replay without recompilation.

## Building the Sampling-Config Sweep

`_create_sampling_params(can_sample_on_device, non_greedy_decoding_on_device, batch_size)` returns the list of `SamplingParams` objects that warm-up iterates over.

When `can_sample_on_device` is `False`, the method returns `[None]` immediately and `batch_size` is never read. Callers on a host-sampling-only path need not compute or validate `batch_size` before calling this function. In that case the sampling step runs on the host and no device trace needs to capture it.

When `can_sample_on_device` is `True`, the result depends on `non_greedy_decoding_on_device`:

**Case A: `non_greedy_decoding_on_device=True` — 6 variants total**

The method generates four non-greedy configs by taking the Cartesian product of `penalties ∈ {True, False}` and `log_probs ∈ {True, False}`:

| `penalties` | `log_probs` | Config description |
|-------------|-------------|------------------------------------------------|
| True        | True        | temperature=1.0, top_k=10, top_p=0.9, presence/frequency/repetition penalties, log_probs enabled |
| True        | False       | same, log_probs disabled |
| False       | True        | temperature=1.0, top_k=10, top_p=0.9, no penalties, log_probs enabled |
| False       | False       | temperature=1.0, top_k=10, top_p=0.9, no penalties, log_probs disabled |

After those four, two more entries are appended:

- A greedy config: `temperature=0.0, top_k=1, top_p=1.0` — covers the common production path where no stochastic sampling occurs.
- `None` — runs the decode path without any on-device sampling kernel, capturing the trace variant used when the caller opts out of device sampling.

Total: **6 variants** (4 non-greedy + 1 greedy + `None`).

**Case B: `non_greedy_decoding_on_device=False` — 2 variants total**

The non-greedy Cartesian product is skipped entirely. Only the two unconditional entries are appended:

- A greedy config: `temperature=0.0, top_k=1, top_p=1.0`.
- `None` — the no-sampling-kernel variant.

Total: **2 variants** (1 greedy + `None`). This is the correct count for a greedy-only on-device deployment; a caller implementing warm-up for such a configuration should expect exactly 2 `decode_forward` calls, not 6.

Each variant produces a distinct trace on device because the sampling kernel dispatch differs.

> **Key insight:** Sweeping all sampling variants during warm-up is what guarantees that every `decode_forward` path a production call can take already has a captured trace ready at inference time. A missing variant would fall back to a slow compile-and-execute path on the first real request that uses it.

## The `warmup_model_decode` Orchestrator

`warmup_model_decode` in `models/common/warmup/warmup_utils.py` ties the pieces together:

```python
def warmup_model_decode(
    self,
    kv_cache,
    enable_trace,
    max_batch_size,
    num_blocks,
    can_sample_on_device,
    non_greedy_decoding_on_device,
):
    sampling_params = self._create_sampling_params(
        can_sample_on_device, non_greedy_decoding_on_device, max_batch_size
    )
    tokens, start_pos, page_table = self._create_decode_warmup_inputs(max_batch_size, num_blocks)

    logger.info("Starting decode warmup")
    ...
    for param in sampling_params:
        logger.info(f"Warming up decode for sampling params: {param}")
        self.decode_forward(
            tokens=tokens,
            start_pos=start_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            enable_trace=enable_trace,
            read_from_device=True,
            sampling_params=param,
        )

    logger.info("Decode warmup completed")
```

All sampling configs share the same token and page-table tensors because those tensors only need to be valid in shape; their values do not affect which kernels are compiled or which trace is captured.

## Decode Warm-Up Guard

`warmup_model_decode` does not currently maintain an `already_warmed_up_decode` boolean of its own in the mixin; the caller (typically vLLM's engine setup or a demo script) is responsible for invoking it exactly once. Contrast this with `warmup_model_prefill`, which sets `self.already_warmed_up_prefill = True` on first entry and returns immediately on any subsequent call. If you add decode warm-up to a new caller, ensure you gate it similarly to avoid repeated trace capture, which would silently leak trace buffer memory.

> **Warning:** Calling `warmup_model_decode` a second time while `enable_trace=True` will attempt to capture a new trace for the same op sequence. The second capture will succeed but the first captured trace ID will be orphaned unless you explicitly release it.

## Observable Log Boundaries

The log lines emitted by `warmup_model_decode` are the clearest boundary markers in stdout:

```
INFO | Starting decode warmup
INFO | Tokens shape: torch.Size([32, 1])
INFO | Start pos shape: torch.Size([32])
INFO | Page table shape: torch.Size([32, N])
INFO | Warming up decode for sampling params: <SamplingParams ...>
...
INFO | Decode warmup completed
```

Everything that appears between `Starting decode warmup` and `Decode warmup completed` in the log is part of the warm-up phase and is not driven by any real user request. In a Tracy ops CSV the corresponding compile-phase rows will appear before any rows with a non-null `METAL TRACE ID` — meaning before both the trace-capture phase and any replay phases. For how `METAL TRACE ID` and `METAL TRACE REPLAY SESSION ID` distinguish these phases in profiling output, see [`differentiating_warmup_from_production.md`](./differentiating_warmup_from_production.md).

---

**Next:** [`warmup_prefill.md`](./warmup_prefill.md)
