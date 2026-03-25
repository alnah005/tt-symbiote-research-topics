# Chapter 2: Weight Management and Precision

Precision choices and sharding strategies are TT Transformers' primary performance advantage over TT Symbiote, and understanding both pipelines in detail is the prerequisite for any meaningful integration work.

## Motivation

TT Symbiote's current weight pipeline is simple and correct: weights are preprocessed on CPU into `bfloat16` or `bfloat8_b` TTNN tensors and pushed to a single device. TT Transformers, by contrast, loads weights directly onto a multi-device mesh at construction time, selects per-layer dtypes from a structured `ModelOptimizations` / `TensorGroup` / `OpGroup` pipeline, tiles weights into DRAM banks via `create_dram_sharded_mem_config`, and caches the converted tensors on disk so that repeated runs skip the conversion entirely.

These differences have direct consequences for throughput, memory, and integration complexity. A developer porting a TT Transformers model into Symbiote must decide, for every weight tensor: which parts of the TT Transformers approach can be carried over unchanged, which require adaptation to Symbiote's module lifecycle, and which must be rewritten from scratch.

This chapter answers those questions concretely against the actual source files.

## Files in this chapter

| File | What it covers |
|---|---|
| [symbiote_weight_pipeline.md](symbiote_weight_pipeline.md) | How TT Symbiote preprocesses, stages, and moves weights; the `@trace_enabled` / `@trace_disabled` class decorators and why they matter for weight lifecycle |
| [transformers_weight_pipeline.md](transformers_weight_pipeline.md) | How TT Transformers loads, shards, and caches weights; `ttnn.as_tensor` with `cache_file_name`, `ShardTensor2dMesh`, `create_dram_sharded_mem_config`, and `load_checkpoints.py` utilities |
| [reuse_vs_rewrite.md](reuse_vs_rewrite.md) | Explicit Reuse / Adapt / Rewrite classification for every significant weight-pipeline feature |
