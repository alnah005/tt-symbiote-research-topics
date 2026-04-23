# Chapter 8: Porting Strategy and Model Prioritization

## Prerequisites

This chapter synthesizes findings from all preceding chapters. You should have read:

- [Chapter 1 -- Architecture Overview](../ch1_architecture_overview/index.md): TT-DiT's `Module`/`Parameter` base classes, the four-level abstraction hierarchy, and the [comparison with TTNNModule](../ch1_architecture_overview/comparison_with_ttnnmodule.md).
- [Chapter 2 -- Parallelism and CCL](../ch2_parallelism_and_ccl/index.md): 3-axis parallelism (`DiTParallelConfig`), `CCLManager`, and the [mapping to TT-Symbiote's distributed infrastructure](../ch2_parallelism_and_ccl/mapping_to_symbiote.md).
- [Chapter 3 -- Custom Layers and Ops](../ch3_custom_layers_and_ops/index.md): normalization, linear, feedforward, convolution, and embedding layers with their [experimental TTNN ops](../ch3_custom_layers_and_ops/ttnn_experimental_ops.md).
- [Chapter 4 -- Attention and Transformer Blocks](../ch4_attention_and_transformer_blocks/index.md): joint attention, adaptive layer normalization, and the [comparison with TT-Symbiote's attention modules](../ch4_attention_and_transformer_blocks/comparison_with_symbiote_attention.md).
- [Chapter 5 -- Pipelines and Serving](../ch5_pipelines_and_serving/index.md): pipeline lifecycle, `PipelineTrace`, and the [three integration strategies](../ch5_pipelines_and_serving/mapping_to_symbiote_serving.md).
- [Chapter 6 -- Weight Loading](../ch6_weight_loading/index.md): TT-DiT's `_prepare_torch_state` pipeline vs. TT-Symbiote's three-phase `from_torch`/`preprocess`/`move` lifecycle.
- [Chapter 7 -- Tracing and Performance](../ch7_tracing_and_performance/index.md): pipeline-level vs. module-level tracing and the [recommended integration strategy](../ch7_tracing_and_performance/integration_strategy.md).

---

## The Porting Challenge

TT-DiT and TT-Symbiote solve the same fundamental problem -- running neural network inference on Tenstorrent hardware -- but they approach it from opposite directions:

**TT-DiT** is a **purpose-built, vertically integrated framework** for diffusion transformer models. Every layer is a hand-written TTNN implementation. Parallelism, collective communication, weight loading, and tracing are tightly coupled into a single coherent system optimized for the denoising-loop workload. The framework's tight coupling is both its strength (high performance, low overhead) and its limitation (every new model requires significant manual implementation effort).

**TT-Symbiote** is a **general-purpose acceleration framework** that intercepts PyTorch dispatch to route operations to TTNN. Its `TTNNModule` base class provides a standard lifecycle (`from_torch` -> `preprocess_weights` -> `move_to_device` -> `forward`), automatic tracing via `TracedRun`, and dual-execution modes (`SELRun`, `DPLRun`) for validation. The framework's generality is its strength (new models can be accelerated incrementally), but it lacks the DiT-specific primitives (joint attention, adaptive normalization, 3-axis parallelism) needed for competitive diffusion model performance.

The porting effort is therefore not a simple code migration. It requires:

1. **Building new infrastructure** in TT-Symbiote for capabilities that currently exist only in TT-DiT (CCL persistent buffers, multi-axis parallelism, submesh management).
2. **Creating new TTNNModule subclasses** for DiT-specific components (joint attention, adaptive LayerNorm, time-conditioned transformer blocks).
3. **Reconciling architectural assumptions** at every integration boundary (tracing granularity, weight loading lifecycle, memory management, tensor representation).

This chapter provides a concrete, prioritized plan for executing this work, organized around three questions:

- **What can be reused as-is?** (Components with direct TT-Symbiote equivalents)
- **What needs adaptation?** (Components that can be reimplemented as TTNNModule subclasses)
- **What requires new infrastructure?** (Capabilities that TT-Symbiote fundamentally lacks)

## Chapter Files

- [`component_assessment.md`](./component_assessment.md) -- Three-tier classification of every TT-DiT component: directly reusable, reimplementable as TTNNModule subclasses, or requiring new TT-Symbiote infrastructure. Specific components listed in each tier with effort estimates.
- [`model_prioritization.md`](./model_prioritization.md) -- Ranking of all six supported models by porting difficulty. SD3.5 recommended as the first candidate. Rationale for each model's ranking.
- [`porting_roadmap.md`](./porting_roadmap.md) -- Five-phase plan from infrastructure through production deployment. Phase deliverables, success criteria, risk factors, and open questions.

## Key Takeaways

1. **The porting challenge is primarily an infrastructure and abstraction gap, not a code volume problem.** The majority of TT-DiT's computational code (TTNN op calls in `forward()` methods) is directly compatible with TT-Symbiote's execution model. The difficulty lies in the surrounding infrastructure: CCL management, multi-axis parallelism configuration, pipeline-level tracing, and dynamic memory management.

2. **Approximately 30% of TT-DiT components are directly reusable, 40% require moderate adaptation, and 30% require new infrastructure.** The component assessment in this chapter provides a concrete mapping for every layer, block, and utility.

3. **SD3.5 is the clear first porting candidate.** It uses only 2D image generation (no Conv3d), has the most standard attention pattern among TT-DiT models, has an established test suite for validation, and exercises the core infrastructure (TP, SP, joint attention, adaptive LayerNorm) without the additional complexity of video models.

4. **An incremental, phase-gated approach minimizes risk.** Each phase produces a testable deliverable, allowing the team to validate correctness and measure performance before committing to the next phase.

5. **The long-term benefit of porting justifies the upfront cost.** Once the DiT-specific infrastructure exists in TT-Symbiote, subsequent models can be added with significantly less effort, and all models benefit from TT-Symbiote's debugging, profiling, and serving infrastructure.

---

**Next:** [`component_assessment.md`](./component_assessment.md)
