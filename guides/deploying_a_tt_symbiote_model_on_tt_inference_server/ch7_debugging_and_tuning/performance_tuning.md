# Performance Tuning

This file covers the configuration knobs available to tune a deployed TT Symbiote model for throughput, latency, and memory efficiency. The levers fall into four categories: throughput, latency, memory, and block size. A section on profiling tips follows.

All configuration described here is set either in `DeviceModelSpec`, in the `vllm_args` dict, or in `override_tt_config`. None of it requires changes to your model class.

---

## Throughput Levers

### `max_concurrency` (maps to `--max-num-seqs`)

`max_concurrency` in `DeviceModelSpec` sets the maximum number of sequences the scheduler will run concurrently. It maps directly to vLLM's `--max-num-seqs`. Increasing this value amortizes the fixed per-step decode overhead — device dispatch, host synchronization, sampling — across a larger batch, increasing tokens generated per second at the cost of per-request latency.

On T3K with a 7B model, 32 concurrent sequences is a typical operating point that saturates compute without exhausting KV cache. Smaller models or higher-memory configurations can run more; larger models may need fewer.

When tuning: start at 32, run a multi-user load test, and observe decode throughput (tokens/sec across all sequences). If throughput scales linearly as you add sequences, you have not yet hit the memory or compute ceiling. If it plateaus, you are either memory-bound (KV cache exhausted) or compute-bound (decode kernel fully saturated).

### `max-num-batched-tokens`

`max-num-batched-tokens` in `vllm_args` limits the total number of tokens the scheduler will process in a single forward pass across prefill and decode combined.

- **Decode-focused workloads** (many concurrent short-generation requests): for pure-decode steps each active sequence contributes exactly one token, so the minimum useful value is `max_concurrency`. In practice, set it to `max_concurrency * block_size` (e.g., `32 × 64 = 2048`) as a conservative upper bound that also accommodates short prefill bursts without requiring a separate prefill-mode configuration. The key constraint is that the value must be ≥ `max_concurrency`; note that `block_size` (64) is a KV cache memory-allocation unit, not the decode batch width — the formula uses it simply as a generous multiplier.
- **Prefill-heavy workloads** (long prompts, summarization, RAG): set to `max_context`. This allows the scheduler to process a full long prompt in one shot rather than spreading it across multiple steps.

Mismatching this value to your workload pattern is a common cause of underutilization. A decode-focused deployment with `max-num-batched-tokens` set too high will waste scheduler cycles building padded batches; set too low it will artificially throttle concurrency.

### `tt_data_parallel`

For Galaxy cluster deployments, `tt_data_parallel` runs multiple independent model replicas, each on a subset of the available chips. Each replica handles a separate sub-batch of requests. Total cluster throughput scales proportionally with the number of replicas, assuming the load balancer distributes requests evenly.

Data parallel scaling is orthogonal to the per-replica tuning described elsewhere in this file. Tune a single replica first, then multiply out with data parallelism to hit aggregate throughput targets.

---

## Latency Levers

### `optimizations="performance"`

Setting `optimizations="performance"` in `DeviceModelSpec` activates a bundle of latency-reducing choices:

- **bfloat8 weights**: reduces DRAM bandwidth per weight load, allowing the decode kernel to run faster when memory bandwidth is the bottleneck
- **HiFi2 math fidelity**: uses a faster but slightly lower-precision math path in the TTNN matrix engine
- **TTNN trace capture**: records the decode kernel graph during warmup and replays it via `ttnn.execute_trace()` in subsequent steps, eliminating per-step Python dispatch overhead

This mode is appropriate for production deployments where output quality has been validated. The precision reduction from bfloat8 and HiFi2 is negligible for most generative tasks but should be verified for your specific model and task.

### `optimizations="accuracy"`

Setting `optimizations="accuracy"` activates the higher-precision configuration:

- **bfloat16 weights**: full 16-bit storage; higher memory bandwidth cost per decode step
- **HiFi4 math fidelity**: slower but more numerically precise matrix operations
- **No trace capture**: every decode step dispatches TTNN operations through the Python call stack

This mode is approximately 2× slower than `"performance"` for decode. Use it during integration debugging, when validating model output quality, or when diagnosing suspected numerical issues. Do not use it in production if throughput or latency targets matter.

> **Decode will be significantly slower without trace capture.** Because TTNN trace is intentionally disabled in accuracy mode, decode latency above 100ms per step is expected and is not a misconfiguration. This mode is designed for correctness validation, not production throughput. If you observe slow decode and are in accuracy mode, see [common_errors.md](./common_errors.md) — Error 8 for the distinction between expected accuracy-mode slowness and a genuine trace initialization failure under `optimizations="performance"`.

### On-Device Sampling

Moving the sampling step onto the device eliminates a round-trip between the device and the Python sampler per decode step. This requires:

1. `has_builtin_warmup=True` set in your model configuration so the worker knows warmup is handled inside `initialize_vllm_model`.
2. The sampling operations (softmax, top-k/top-p filtering, multinomial draw) implemented as TTNN ops inside the model's decode path.
3. The sampling ops included in the trace capture during warmup, so they are replayed by `ttnn.execute_trace()` along with the rest of the decode graph.

When on-device sampling is active, the decode step returns token IDs directly from the device rather than a full logit tensor, further reducing the data transfer volume per step. The gain is most visible at high concurrency where sampling overhead accumulates across many parallel sequences.

---

## Memory Levers

### `trace_region_size` in `override_tt_config`

`trace_region_size` controls the amount of DRAM reserved for TTNN trace capture buffers. The trace stores intermediate activations and operator metadata for the decode-shaped forward pass.

This value directly trades off against KV cache capacity:

- **Too large**: insufficient DRAM remains for KV cache blocks; allocation fails at startup (see [common_errors.md](./common_errors.md) — error 9).
- **Too small**: trace capture itself fails with an OOM error during warmup.

A practical starting range for a 7B model on T3K is 200–400 MB. Larger models with more layers or wider hidden dimensions will require more trace memory. Tune by starting at 200 MB and increasing until trace capture succeeds, then verify that the remaining DRAM is sufficient for your target `max_concurrency` and `max_context`.

### `worker_l1_size_bytes`

`worker_l1_size_bytes` controls how much L1 SRAM is reserved per device for worker-side buffers, including the circular buffers used to stage activations between TTNN ops. The default value is set conservatively.

Adjust this only if you are seeing L1 overflow errors (`RuntimeError: L1 allocation failed`) during compilation or trace capture. Increasing it reserves more L1 for your model's circular buffers at the cost of reducing the space available for other allocations. Do not reduce it below the default without profiling; the default is chosen to accommodate typical model buffer requirements.

---

## Block Size

The KV cache block size is fixed at 64 tokens for all TT hardware. This value is not tunable via `vllm_args` or any other configuration. All KV cache tensor shapes, index calculations, and attention mask logic in your model must use `block_size=64`. Passing a different value via `vllm_args` will either be ignored or cause an assertion failure (see [common_errors.md](./common_errors.md) — error 6).

---

## Profiling Tips

### Per-Step Timing via `VLLM_LOGGING_LEVEL`

Set `VLLM_LOGGING_LEVEL=DEBUG` before starting the server to enable verbose scheduler and engine logs. At this level, vLLM logs the time spent in each phase (prefill forward, decode forward, sampler, scheduler overhead) per step. Use these logs to identify which phase dominates your latency budget.

```bash
export VLLM_LOGGING_LEVEL=DEBUG
python -m vllm.entrypoints.openai.api_server ...
```

### Device Utilization via `tt-smi`

`tt-smi` is the TT Systems Management Interface. Run it in a separate terminal while the server is under load to observe:

- DRAM bandwidth utilization per device — low bandwidth during decode suggests the batch is too small to saturate the memory bus
- Core utilization — low utilization suggests the decode kernel is waiting on host dispatch rather than running on-device work; this points to trace not being active

### Isolating Trace Capture Failures

If trace capture fails inside the vLLM worker process, the error message is often obscured by the worker's exception handling. To reproduce it in isolation:

```python
import ttnn
mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 8))  # T3K: 1x8

# Load weights and build model exactly as in initialize_vllm_model
model = TTMyModel.initialize_vllm_model(hf_config, mesh_device, ...)

# Re-run just the trace capture section
ttnn.begin_trace_capture(mesh_device, trace_id=99)
_ = model.decode_forward(dummy_tokens, dummy_kv_cache)
ttnn.end_trace_capture(mesh_device, trace_id=99)

ttnn.close_mesh_device(mesh_device)
```

Running this outside vLLM gives you a clean Python traceback for OOM errors, compile failures, or shape mismatches in the trace capture path.

---

**Return to:** [Guide Index](../index.md)
