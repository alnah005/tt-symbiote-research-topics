# request_lifecycle.md

This document traces a single completion request from its arrival at the HTTP port to the returned response token. Understanding this lifecycle tells you exactly which interfaces your TT Symbiote model must implement, what state it receives per forward call, and where hardware-specific behaviour (KV cache, on-device sampling, trace) fits in.

## End-to-end flow

```
Client
  │
  │  POST /v1/completions  {"prompt": "Hello", "max_tokens": 128}
  ▼
vLLM OpenAI API server  (vllm.entrypoints.openai.api_server)
  │  tokenize → [15496, ...]
  │  create SequenceGroup, add to scheduler
  ▼
vLLM Scheduler
  │  select sequences for the next step
  │  allocate KV cache blocks via BlockSpaceManager
  ▼
TTWorker  (tt_worker.py)
  │  build TTModelInput for this step
  ▼
TTModelRunner  (tt_model_runner.py)
  │  dispatch prefill_forward() or decode_forward()
  ▼
Model implementation  (e.g., TTLlamaForCausalLM.prefill_forward / .decode_forward)
  │  run on ttnn mesh device
  │  return sampled token IDs
  ▼
TTWorker  →  vLLM Scheduler  →  API server  →  detokenize  →  HTTP response
```

## vLLM KV cache block manager

On startup, `TTModelLoader` asks the model for its KV cache configuration and pre-allocates a pool of fixed-size KV cache blocks on the Tenstorrent device using `ttnn` tensors. For TT hardware, the TT hardware default block size is **64 tokens** — set by `TTPlatform` and larger than vLLM's CUDA default (16 tokens) because Tenstorrent's tile-based compute prefers 64-token-aligned access patterns. This value is configurable via `DeviceModelSpec.vllm_args["block-size"]`; do not hard-code 64 in model code. Always configure the block size through `ModelSpec` so that it can be adjusted per deployment without modifying the model implementation.

The block manager treats each block as a page in a virtual-memory-style page table. When a new sequence arrives, the scheduler allocates one or more blocks from the free pool and records their physical block IDs in a `block_tables` — an integer tensor of shape `(batch_size, max_blocks_per_sequence)`. As the sequence grows, additional blocks are allocated and appended to the table. When a sequence finishes, its blocks are returned to the free pool.

The model implementation receives the `block_tables` as part of `TTModelInput` on every forward call. It is the model's responsibility to use the block tables to read and write the KV cache at the correct physical locations (typically via `paged_sdpa` or an equivalent paged attention kernel).

## TTModelInput

`TTModelInput` is the data-transfer object that `TTWorker` builds before each forward call and passes to `TTModelRunner`. Its fields are:

| Field | Type | Description |
|-------|------|-------------|
| `input_ids` | `torch.Tensor` `(batch_size, seq_len)` | Token IDs for this step; during decode, `seq_len=1` |
| `input_positions` | `torch.Tensor` `(batch_size, seq_len)` | Absolute position indices for each token; used to index into RoPE cos/sin tables |
| `prompt_lens` | `torch.Tensor` `(batch_size,)` | Length of the original prompt for each sequence in the batch; zero for decode-only steps |
| `block_tables` | `torch.Tensor` `(batch_size, max_blocks)` | Page table mapping logical KV cache positions to physical block IDs |
| `batch_size` | `int` | Number of active sequences in this step |
| `max_seq_len` | `int` | Maximum sequence length across all sequences in this step |
| `sampling_params` | `list[TTSamplingParams]` | Per-sequence sampling configuration, one entry per sequence in the batch; sequences in the same batch may have different temperatures, top_k, and top_p values (see below) |

### TTSamplingParams

`TTSamplingParams` carries the sampling parameters for a single sequence. `TTModelInput.sampling_params` is a list indexed by sequence position — each entry corresponds to one sequence in the batch. Sequences within the same batch may carry different values, so model authors must not assume uniform sampling parameters across the batch. Always iterate over the list rather than applying a single shared configuration:

```python
@dataclass
class TTSamplingParams:
    temperature: float       # 0.0 → greedy argmax; > 0 → softmax + multinomial
    top_k: int               # 0 → no top-k filtering; > 0 → keep top_k logits
    top_p: float             # 1.0 → no nucleus filtering; < 1.0 → nucleus sampling
```

When `temperature=0.0`, the model should return the `argmax` of the logits. When `temperature > 0`, it should apply `softmax(logits / temperature)` and sample according to `top_k` / `top_p` constraints.

## Prefill vs. decode

The two forward methods have different performance characteristics and the model should implement them separately.

### Prefill

`prefill_forward(tt_model_input: TTModelInput) -> torch.Tensor`

Prefill processes the full prompt. `input_ids` has shape `(batch_size, prompt_len)` where `prompt_len` can be hundreds or thousands of tokens. The model computes attention over the entire prompt in a single (or multi-chunk) forward pass and writes every KV pair into the allocated cache blocks identified by `block_tables`. The return value is the logits for the **last token** of each sequence — the first token that needs to be sampled.

Because prefill is compute-bound and not latency-sensitive to the same degree as decode, it does not need to be captured in a TTNN trace.

### Decode

`decode_forward(tt_model_input: TTModelInput) -> torch.Tensor`

Decode generates one token per step. `input_ids` has shape `(batch_size, 1)` — a single token per sequence. The model reads the existing KV cache entries using the `block_tables` page table, writes the new KV pair for this step into the appropriate block, and returns logits for the single new position.

Decode is repeated hundreds or thousands of times per request and is extremely latency-sensitive. A typical `decode_forward` call on TT hardware targets sub-millisecond execution time for small batch sizes.

## On-device sampling

When the `tt-vllm-plugin` is configured with on-device sampling enabled, the logits computation and token sampling steps are separated into two distinct TTNN operations and both are captured as TTNN traces during `initialize_vllm_model()` at model startup. This is why `has_builtin_warmup=True` is set in `ModelSpec` for trace-captured models — the model itself performs warmup during initialisation, so vLLM does not need to issue a separate warmup pass.

```
decode_forward():
  ├── run logits trace      # TTNN trace: compute hidden states → logits
  └── run sampling trace    # TTNN trace: apply temperature/top_k/top_p → sample token
```

TTNN trace capture eliminates Python host overhead between individual device operations. During trace capture (warm-up), every `ttnn` operation is recorded into a replay buffer. During actual inference, a single `ttnn.execute_trace()` call replays the entire buffer on the device without re-entering Python for each op. This is especially valuable for decode, where the sequence of operations is identical for every step.

The two traces (logits and sampling) are kept separate so that the sampling parameters can be updated between steps without re-capturing the logits trace. If `temperature` or `top_p` changes mid-sequence, only the sampling trace needs to be invalidated and re-captured.

### Trace lifecycle

```python
# Warm-up: capture the traces (done once at model initialisation)
with ttnn.capture_trace(mesh_device, trace_id=LOGITS_TRACE_ID):
    logits = model._forward_logits(tt_model_input)

with ttnn.capture_trace(mesh_device, trace_id=SAMPLING_TRACE_ID):
    tokens = model._sample(logits, sampling_params)

# Inference: replay the traces on every decode step
ttnn.execute_trace(mesh_device, LOGITS_TRACE_ID)
ttnn.execute_trace(mesh_device, SAMPLING_TRACE_ID)
output_tokens = ttnn.from_device(model.output_token_buffer)
```

The model must pre-allocate all input and output tensors before trace capture and reuse the same tensor addresses for every replay. This means `TTModelInput` tensors are written into pre-allocated host-pinned buffers by `TTModelRunner`, and the model reads from those buffers via persistent TTNN tensor handles.

---

**Next:** [Chapter 2 — The ModelSpec Configuration System](../ch2_model_spec/index.md)
