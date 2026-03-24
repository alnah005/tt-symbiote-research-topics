# Chapter 3 — Model Warm-Up and Its Relationship to Trace Capture

This chapter explains what happens during the model warm-up phase that runs before any user request is served. By the end of this chapter you will understand the sequence of events from device open through warm-up to the first real inference call, how warm-up embeds the compile-then-capture trace workflow described in Chapter 1, what inputs warm-up uses for decode and prefill, and how to tell warm-up operations apart from production operations in both Python and profiling output.

## Learning Objectives

- Understand the end-to-end timeline from device open to the first production inference request.
- Know how `WarmupForwardMixin.warmup_model_decode` and `warmup_model_prefill` orchestrate compile and trace capture for every sampling variant and sequence length.
- Understand the guard booleans that prevent redundant warm-up on repeated calls.
- Be able to identify warm-up rows versus production rows in a Tracy ops CSV.

## Timeline: Device Open to First Real Inference

```
[Device Open]
     |
     v
[Model weights loaded onto device]
     |
     v
[warmup_model_prefill]
     |  - compile run for each seq_len (sampling configs swept on first length only)   <-- compile phase
     |  - trace capture run for each trace-supported seq_len      <-- capture phase
     v
[warmup_model_decode]
     |  - compile + capture run for each sampling_config variant  <-- single pass (no separate compile sweep)
     v
[already_warmed_up_prefill = True, all decode traces captured]
     |
     v
[First real prefill_forward_text / decode_forward call from vLLM or demo]
```

The compile runs from Chapter 1 — where TT-Metal JIT-compiles every kernel the first time an op is dispatched — are embedded inside warm-up: for the first sequence length the compile run sweeps all sampling configs, and for all subsequent lengths only a single forward pass (with no sampling variant) is the compile run; the following pass is the trace capture.

## Files in This Chapter

Read in the following order:

1. [`warmup_decode.md`](./warmup_decode.md) — Decode warm-up: inputs, sampling variants, log boundaries, and the decode guard.
2. [`warmup_prefill.md`](./warmup_prefill.md) — Prefill warm-up: sequence-length sweep, the `model_id` loop, warm-up tensors, and the chunked-prefill skip path.
3. [`differentiating_warmup_from_production.md`](./differentiating_warmup_from_production.md) — How to distinguish warm-up calls from production calls at the Python level and in profiling output, including the `split_compile_and_trace` utility and Tracy signpost filtering.
