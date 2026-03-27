# Platform Constraints

`TTPlatform` enforces a set of constraints on the vLLM runtime configuration that directly affect how your model implementation must behave. These are not soft guidelines — they are hard platform-level overrides that the server applies before your model is ever called. Violating them in your model code produces incorrect results or runtime errors that are difficult to diagnose.

## Chunked Prefill Is Disabled

`TTPlatform` sets `enable_chunked_prefill = False`. Your model's `prefill_forward()` will always receive the entire prompt token sequence for each request in a single call. You do not need to implement any state machine to accumulate partial prefill results across multiple forward passes.

## Prefix Caching Is Disabled

`TTPlatform` does not enable prefix KV cache reuse. Your `allocate_kv_cache()` implementation does not need to track which cache blocks contain shared prompt prefixes, and your `prefill_forward()` does not need to accept or skip pre-filled block ranges. Every prefill call computes attention over its full input from scratch.

## Sampling Constraints

The server enforces the following sampling restrictions for TT hardware:

- `n = 1` only. The sampler always requests exactly one completion per prompt. Your model only needs to return one logit row per input sequence.
- `best_of` is not supported. Do not implement any beam search or multi-hypothesis scoring logic.
- `prompt_logprobs` is not supported. Your forward methods do not need to return per-token log probabilities for the input prompt. The return value is always the last-token logits tensor only.

## Tensor Parallelism and Single-Process Execution

`TTPlatform` enforces single-process execution. There is exactly one Python worker process per server instance. Multi-chip parallelism — tensor parallelism, pipeline parallelism, and data parallelism across TT devices — is handled entirely inside your model via TTNN mesh operations on the `mesh_device` passed to `initialize_vllm_model()`.

Do not attempt to use `torch.distributed` or vLLM's built-in tensor-parallel group utilities inside your model. Those mechanisms assume multiple OS-level processes and NCCL/RCCL backends; they have no integration with the TTNN mesh abstraction. Any call to `dist.init_process_group()` or `dist.all_reduce()` from within your model will either fail immediately or silently operate on a single-rank process group and return incorrect results.

## Block Size: Always 64

`TTPlatform` overrides the vLLM block size to `64` for all TT hardware configurations. The paged attention kernel that ships with `tt-vllm-plugin` is compiled for this block size and will produce incorrect results if called with any other value.

Your model must not hardcode a different block size anywhere in `allocate_kv_cache()`, `prefill_forward()`, or `decode_forward()`. Use the `block_size` argument passed to `allocate_kv_cache()` when computing tensor shapes and indexing page tables. In testing, you can assert that it equals `64` as a sanity check:

```python
def allocate_kv_cache(self, num_blocks, block_size, num_kv_heads, head_dim, dtype, **kwargs):
    assert block_size == 64, (
        f"TT paged attention kernel requires block_size=64, got {block_size}"
    )
    ...
```

## Pinned Memory: Not Available

`TTPlatform.is_pin_memory_available()` returns `False`. Do not request pinned (page-locked) CPU memory in any part of your model initialization or forward-pass code. This includes:

- Calls to `torch.Tensor.pin_memory()` on intermediate buffers.
- `pin_memory=True` arguments to `torch.zeros()`, `torch.empty()`, or `DataLoader`.
- Any TTNN utility that internally requests pinned memory as a staging buffer.

Requesting pinned memory on a system where it is unavailable raises a runtime error that terminates the worker process.

## Data-Parallel Batch Gathering

`TTPlatform.requires_gathered_batch_dp()` returns `True`. When `tt_data_parallel > 1`, vLLM gathers the full batch across all data-parallel ranks before calling your model, so your `prefill_forward()` and `decode_forward()` always receive the complete batch in a single call. You do not need to implement any scatter/gather logic inside your model to handle partial sub-batches.

Your model is responsible for internally dispatching the gathered batch across the data-parallel dimension of the `mesh_device` using TTNN mesh operations.

---

**Next:** [Chapter 4 — Weight Loading and Tokenization](../ch4_weights_and_tokenization/index.md)
