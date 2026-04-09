# TT-Lang Architecture and TT-Symbiote Integration Strategy

This guide provides a comprehensive reference for Tenstorrent software engineers who want to understand TT-Lang's programming model, compilation pipeline, and functional simulator, and then apply that knowledge to author custom fused kernels integrated into the TT-Symbiote inference pipeline. It assumes working familiarity with TTNN ops, `ttnn.Tensor`, the `TTNNModule` base class, and Tenstorrent hardware concepts (Tensix cores, NOCs, L1/DRAM, TILE_LAYOUT).

---

## How to Use This Guide

| Goal | Recommended Path | Direct Links |
|------|-----------------|--------------|
| Learn the TT-Lang DSL from scratch | Ch 1 then Ch 2 then Ch 3 | [Programming Model](ch1_programming_model/index.md), [Compilation Pipeline](ch2_compilation_pipeline/index.md), [Simulator](ch3_functional_simulator/index.md) |
| Profile and optimize a TT-Lang kernel | Ch 1 then Ch 4 | [Programming Model](ch1_programming_model/index.md), [Performance Tools](ch4_performance_tools/index.md) |
| Understand TT-Symbiote internals | Ch 5 | [Symbiote Architecture](ch5_symbiote_architecture/index.md) |
| Integrate a TT-Lang kernel into TT-Symbiote | Ch 5 then Ch 6 then Ch 8 | [Symbiote Architecture](ch5_symbiote_architecture/index.md), [Integration Strategy](ch6_integration_strategy/index.md), [Workflow](ch8_workflow_and_multidevice/index.md) |
| Identify high-value fusion opportunities | Ch 5 then Ch 7 | [Symbiote Architecture](ch5_symbiote_architecture/index.md), [Fusion Targets](ch7_fusion_targets/index.md) |
| End-to-end: write, test, and deploy a fused kernel | Ch 1 through Ch 8 in order | All chapters below |

---

## Chapter Index

| # | Chapter | Description | Key Concepts |
|---|---------|-------------|--------------|
| 1 | [Ch 1 --- TT-Lang Programming Model](ch1_programming_model/index.md) | The decorator-based DSL for authoring kernels that target Tensix cores. | `@ttl.operation`, `@ttl.compute`, `@ttl.datamovement`, DFB, `TensorBlock`, grid intrinsics |
| 2 | [Ch 2 --- TT-Lang Compilation Pipeline](ch2_compilation_pipeline/index.md) | Full lowering path from Python DSL through MLIR dialects to C++ codegen and JIT execution. | `TTLGenericCompiler`, 17-pass MLIR pipeline, `CompilerOptions`, `CompiledTTNNKernel`, `ttnn.generic_op` |
| 3 | [Ch 3 --- Functional Simulator](ch3_functional_simulator/index.md) | Pure-Python simulator for validating kernel correctness without hardware. | `BlockStateMachine`, `AccessState` lifecycle, `GreenletScheduler`, resource limits |
| 4 | [Ch 4 --- Performance Analysis Tools](ch4_performance_tools/index.md) | Environment-variable-driven profiling, signpost regions, and Perfetto trace visualization. | `TTLANG_AUTO_PROFILE`, `TTLANG_SIGNPOST_PROFILE`, `TTLANG_PERF_DUMP`, `TTLANG_PERF_SERV` |
| 5 | [Ch 5 --- TT-Symbiote Architecture and Pain Points](ch5_symbiote_architecture/index.md) | TT-Symbiote's module lifecycle, dispatch system, and module catalog with identified pain points. | `TTNNModule`, `TorchTTNNTensor`, `__torch_dispatch__`, dispatcher registry, module catalog |
| 6 | [Ch 6 --- Integration Strategy: TT-Lang Kernels in TT-Symbiote](ch6_integration_strategy/index.md) | Interface contract and code changes for using TT-Lang kernels as drop-in replacements for TTNN ops. | `CompiledTTNNKernel.__call__`, weight pipeline interaction, `forward()` method changes, compilation caching |
| 7 | [Ch 7 --- High-Value Fusion Targets](ch7_fusion_targets/index.md) | Highest-value TT-Symbiote operations for kernel fusion with concrete kernel designs. | MoE expert pipeline, fused attention (QKV + RoPE + SDPA), fused activations (SwiGLU) |
| 8 | [Ch 8 --- Developer Workflow and Multi-Device Considerations](ch8_workflow_and_multidevice/index.md) | End-to-end development lifecycle and multi-device distribution analysis. | 7-step workflow (design through deploy), `ShardTensor2dMesh`, single-device kernel + CCL hybrid approach |

---

## Quick Reference

| API / Concept | What It Does | Where to Learn More |
|---------------|-------------|---------------------|
| `@ttl.operation(grid=..., num_outs=N)` | Top-level decorator that defines a TT-Lang kernel with a grid of Tensix cores. | [Ch 1](ch1_programming_model/index.md) |
| `@ttl.compute()` / `@ttl.datamovement()` | Register the compute thread and data-movement threads within a kernel. | [Ch 1](ch1_programming_model/index.md) |
| `ttl.make_dataflow_buffer_like(tensor, shape, block_count)` | Create a circular DFB tied to a tensor's shape and dtype. | [Ch 1](ch1_programming_model/index.md) |
| `dfb.wait()` / `dfb.reserve()` | Acquire a DFB block for reading (consumer) or writing (producer). | [Ch 1](ch1_programming_model/index.md), [Ch 3](ch3_functional_simulator/index.md) |
| `ttl.copy(src, dst)` | Asynchronous DMA transfer between tensors and DFBs. | [Ch 1](ch1_programming_model/index.md) |
| `ttl.node()` / `ttl.grid_size()` | Grid intrinsics returning core coordinates and grid dimensions. | [Ch 1](ch1_programming_model/index.md) |
| `CompilerOptions` | Dataclass controlling MLIR pass behavior (`maximize_dst`, `use_block_matmul`, etc.). | [Ch 2](ch2_compilation_pipeline/index.md) |
| `CompiledTTNNKernel` | Cached, callable kernel object that accepts `ttnn.Tensor` arguments. | [Ch 2](ch2_compilation_pipeline/index.md), [Ch 6](ch6_integration_strategy/index.md) |
| `BlockStateMachine` / `AccessState` | DFB protocol enforcer: MW, MR, RW, ROR, OS state lifecycle. | [Ch 3](ch3_functional_simulator/index.md) |
| `TTLANG_AUTO_PROFILE=1` | Per-source-line cycle count instrumentation. | [Ch 4](ch4_performance_tools/index.md) |
| `TTLANG_PERF_SERV=1` | Launch interactive Perfetto trace server. | [Ch 4](ch4_performance_tools/index.md) |
| `TTNNModule` | TT-Symbiote base class: `preprocess_weights` / `move_to_device` / `forward` / `deallocate`. | [Ch 5](ch5_symbiote_architecture/index.md) |
| `TorchTTNNTensor.__torch_dispatch__` | PyTorch tensor subclass routing ATen ops to TTNN handlers. | [Ch 5](ch5_symbiote_architecture/index.md) |

---

## Prerequisites

- **TTNN and tt-metal**: Working installation of tt-metal with the TTNN op library. `TT_METAL_HOME` environment variable must be set.
- **TT-Lang**: The `ttl` Python package installed and importable (`import ttl`).
- **TT-Symbiote**: Access to the TT-Symbiote codebase (`tt-metal/models/experimental/tt_symbiote/`).
- **Hardware knowledge**: Familiarity with Tensix cores, NOC data movement, L1/DRAM memory hierarchy, and `TILE_LAYOUT`.
- **Python**: Comfortable with Python decorators, context managers, and AST-level concepts.
- **MLIR**: No prior MLIR knowledge required; relevant concepts are introduced as needed in [Ch 2](ch2_compilation_pipeline/index.md).

---

## Source Code Locations

| Component | Repository Path |
|-----------|----------------|
| TT-Lang DSL and compiler | `/localdev/salnahari/testing_dir/tt-lang/` |
| TT-Lang simulator | `/localdev/salnahari/testing_dir/tt-lang/sim/` |
| TT-Lang MLIR dialects and passes | `/localdev/salnahari/testing_dir/tt-lang/lib/Dialect/` |
| TT-Symbiote core | `/localdev/salnahari/testing_dir/tt-metal/models/experimental/tt_symbiote/` |
| TT-Symbiote modules | `/localdev/salnahari/testing_dir/tt-metal/models/experimental/tt_symbiote/modules/` |
| TT-Symbiote dispatchers | `/localdev/salnahari/testing_dir/tt-metal/models/experimental/tt_symbiote/core/dispatchers/` |
