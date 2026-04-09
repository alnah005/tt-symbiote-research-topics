# Chapter 8 --- Developer Workflow and Multi-Device Considerations

This final chapter provides the end-to-end developer workflow for writing, testing, and deploying TT-Lang kernels within the TT-Symbiote inference pipeline. It also analyzes how TT-Lang's grid model interacts with TT-Symbiote's multi-device distribution layer, including current limitations and the recommended near-term approach.

Where earlier chapters defined the programming model ([Chapter 1](../ch1_programming_model/index.md)), compilation pipeline ([Chapter 2](../ch2_compilation_pipeline/index.md)), profiling tools ([Chapter 4](../ch4_performance_tools/index.md)), and the integration contract ([Chapter 6](../ch6_integration_strategy/index.md)), this chapter ties them together into a concrete, actionable sequence of steps a developer follows from "I want to fuse these ops" to "it runs in production inference."

## The Development Lifecycle

The lifecycle has five phases, each corresponding to a distinct set of tools and source files:

```
Design  -->  Simulate  -->  Profile  -->  Integrate  -->  Deploy
  |             |              |              |              |
  |  Pick ops   |  Functional  |  auto/       |  TTNNModule  |  TT-Symbiote
  |  from Ch7   |  sim (Ch3)   |  signpost/   |  subclass    |  pipeline
  |  targets    |  validate    |  perf_dump/  |  (Ch6)       |  end-to-end
  |             |  correctness |  perfetto    |              |  test
```

## Contents

| File | Topic |
|------|-------|
| [`development_workflow.md`](./development_workflow.md) | The 7-step development workflow: write kernel, validate with simulator, execute on device, profile, optimize via `CompilerOptions`, integrate into `TTNNModule`, and test through the TT-Symbiote pipeline. |
| [`multidevice_simplification.md`](./multidevice_simplification.md) | Current multi-device code patterns (`ShardTensor2dMesh`, `ConcatMesh2dToTensor`, `TT_CCL`, `CCLManagerConfig`), TT-Lang's grid model potential, current single-device limitation, and the recommended near-term hybrid approach. |

## Key Takeaways

1. **The workflow is incremental and each step produces a testable artifact.** A developer can validate correctness with the functional simulator (no hardware required) before ever touching a device. On-device execution and profiling are separate, explicit steps --- not a single "compile and hope" cycle.

2. **`CompilerOptions` is the primary optimization knob.** Seven boolean flags control DST maximization, FPU binary ops, block matmul lowering, auto-sync, pack-tile combining, and FP32 accumulation for reduce/matmul. These are set via decorator string, `TTLANG_COMPILER_OPTIONS` env var, or `sys.argv` flags, with a well-defined priority order (see `compiler_options.py`).

3. **Four profiling modes** cover the spectrum from quick iteration to deep analysis: auto-profile (`TTLANG_AUTO_PROFILE=1`), signpost profiling (`TTLANG_SIGNPOST_PROFILE=1`), perf dump (`TTLANG_PERF_DUMP=1`), and Perfetto trace server (`TTLANG_PERF_SERV=1`).

4. **Multi-device is currently handled at the TT-Symbiote level, not TT-Lang.** TT-Lang kernels operate on per-device tensor shards. TT-Symbiote's `DistributedConfig`, `ShardTensor2dMesh`, and `TT_CCL` continue to manage cross-device distribution and collective communication. This is the correct near-term architecture.

5. **The integration point is the `TTNNModule.forward()` method.** A fused TT-Lang kernel replaces one or more `ttnn.*` calls inside `forward()`, while `preprocess_weights_impl` and `move_weights_to_device_impl` remain unchanged. The `@run_on_devices` decorator and `@trace_enabled` decorator continue to work as-is.
