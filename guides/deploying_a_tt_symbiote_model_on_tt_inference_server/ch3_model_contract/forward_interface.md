# Forward Interface

Every model class registered with `TTModelLoader` must implement three instance methods: `prefill_forward()`, `decode_forward()`, and `allocate_kv_cache()`. The worker calls these methods directly; there is no base class to inherit from and no abstract interface to satisfy at import time — the contract is purely duck-typed. If a method is missing or its signature is wrong, the failure will occur at runtime when the worker first tries to call it.

## `prefill_forward()`

```python
def prefill_forward(
    self,
    tokens,        # torch.Tensor, shape (total_tokens,), dtype torch.int32, on CPU
    prompt_lens,   # list[int]; length of each prompt in the batch
    page_table,    # torch.Tensor or None; shape (batch_size, max_blocks_per_seq), dtype torch.int32, on CPU
    kv_cache,      # list of TTNN KV cache tensors allocated by allocate_kv_cache()
    **kwargs,      # reserved for future arguments; accept and ignore
) -> torch.Tensor:  # shape (batch_size, vocab_size)
```

`tokens` is a 1D CPU tensor of token IDs with all prompts concatenated in batch order. Use `prompt_lens` to split it: `prompt_lens[i]` gives the number of tokens belonging to the `i`-th sequence. The sum of `prompt_lens` equals `tokens.shape[0]`.

`page_table` maps sequence positions to paged KV cache block indices. Each row `page_table[i]` contains the block indices for sequence `i`; the number of columns is the maximum number of blocks any sequence in this batch requires. Entries beyond the last valid block for a given sequence are undefined and must not be read. If paged attention is not enabled, `page_table` may be `None`.

**Warning: the return tensor MUST reside on CPU.** The vLLM sampler runs entirely on CPU and accesses the logits tensor directly via PyTorch operations — it does not go through `TTModelLoader`'s device-transfer path. Returning a TTNN on-device tensor from `prefill_forward()` will cause a silent sampler failure or incorrect results rather than a clear exception.

The return value must be a `torch.Tensor` of shape `(batch_size, vocab_size)` on CPU, where row `i` contains the logits for the last token of the `i`-th prompt. Call `.cpu()` (or `ttnn.to_torch()` followed by `.cpu()`) before returning if your implementation computes logits on device. Only the last-token logits are consumed by the sampler; the runtime does not inspect intermediate token logits.

## `decode_forward()`

```python
def decode_forward(
    self,
    tokens,        # torch.Tensor, shape (batch_size, 1), dtype torch.int32, on CPU
    page_table,    # torch.Tensor, shape (batch_size, max_blocks_per_seq), dtype torch.int32, on CPU
    kv_cache,      # list of TTNN KV cache tensors allocated by allocate_kv_cache()
    **kwargs,      # reserved for future arguments; accept and ignore
) -> torch.Tensor:  # shape (batch_size, vocab_size)
```

`tokens` contains exactly one token per sequence in the batch — the last token generated in the previous step (or the last token of the prompt for the first decode step). The shape is always `(batch_size, 1)`.

`page_table` is always present in decode (unlike in prefill, where it may be `None`). It has shape `(batch_size, max_blocks_per_seq)` and the same semantics as in `prefill_forward()`: row `i` lists the KV cache block indices allocated to sequence `i`, in order.

The return value must be a `torch.Tensor` of shape `(batch_size, vocab_size)` — one logit row per sequence. Unlike `prefill_forward()`, the tensor may reside on CPU **or** remain as a TTNN on-device tensor. This leniency exists because the decode path supports two sampling modes: the on-device sampling path can consume TTNN tensors directly without a device transfer, while the CPU sampler path calls `TTModelLoader`'s transfer helper to move the tensor to CPU before invoking the sampler. Both paths are handled by the worker; implementers do not need to choose between them.

## `allocate_kv_cache()`

```python
def allocate_kv_cache(
    self,
    num_blocks,    # int; total number of paged KV cache blocks to allocate
    block_size,    # int; number of token positions per block (always 64 on TT hardware)
    num_kv_heads,  # int; number of key/value attention heads
    head_dim,      # int; dimension of each attention head
    dtype,         # ttnn.DataType; dtype for the KV cache tensors
    **kwargs,      # reserved for future arguments; accept and ignore
) -> list:         # list of TTNN tensors, one per layer (or per KV head group, per your layout)
```

`allocate_kv_cache()` is called once during worker initialization, after `initialize_vllm_model()` returns. Its return value is passed as the `kv_cache` argument to every subsequent `prefill_forward()` and `decode_forward()` call, so the tensors must remain valid for the lifetime of the worker.

### KV Cache Tensor Shape Conventions

The paged attention kernel expects KV cache tensors laid out in a specific shape. Two layouts are in use across TT Symbiote models:

**Standard layout (most models):**
```
(num_blocks, block_size, num_kv_heads, head_dim)
```

**Transposed layout (used when the kernel expects heads-first ordering):**
```
(num_blocks, num_kv_heads, block_size, head_dim)
```

Choose the layout that matches the paged attention kernel variant your model uses and document it in your model's internal comments. Mismatched layouts produce silent numerical errors, not runtime crashes, because the kernel reads whatever data is in the tensor.

A typical allocation loop:

```python
def allocate_kv_cache(
    self,
    num_blocks,
    block_size,
    num_kv_heads,
    head_dim,
    dtype,
    **kwargs,
) -> list:
    kv_cache = []
    for _ in range(self.config.num_hidden_layers):
        # Allocate key cache and value cache as separate TTNN tensors.
        k_cache = ttnn.zeros(
            shape=(num_blocks, block_size, num_kv_heads, head_dim),
            dtype=dtype,
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
        )
        v_cache = ttnn.zeros(
            shape=(num_blocks, block_size, num_kv_heads, head_dim),
            dtype=dtype,
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
        )
        kv_cache.append((k_cache, v_cache))
    return kv_cache
```

## Return Type Summary

| Method | Return shape | Allowed devices | Reason |
|---|---|---|---|
| `prefill_forward()` | `(batch_size, vocab_size)` | **CPU `torch.Tensor` only** | vLLM sampler runs on CPU and accesses logits directly — no device-transfer path is invoked |
| `decode_forward()` | `(batch_size, vocab_size)` | CPU `torch.Tensor` or TTNN on-device tensor | On-device sampling path handles TTNN tensors; `TTModelLoader` transfers to CPU for the CPU sampler path |
| `allocate_kv_cache()` | list of TTNN tensors | Must reside on `mesh_device` | — |

Both forward methods must return exactly one logit row per sequence in the batch. Returning a tensor with a leading dimension that does not match `batch_size` will cause the sampler to produce incorrect results without raising an exception.

---

**Next:** [constraints.md](./constraints.md)
