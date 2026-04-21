# Post-Training Differences: Qwen3.5 vs Qwen3.6

## What Post-Training Means

Modern large language models are developed in two broad phases:

1. **Pre-training:** The model learns general language and world knowledge from a massive corpus using next-token prediction. This produces a *base model* whose weights encode broad competence but no task-specific behavior or safety alignment.

2. **Post-training:** The base model is further trained using supervised fine-tuning (SFT), reinforcement learning from human feedback (RLHF), or reinforcement learning from AI feedback (RLAIF) on curated datasets. This shapes *behavior* — how the model responds to instructions, how it reasons, how it uses tools, and how it handles safety-critical situations — without changing the underlying architecture.

**Qwen3.5 and Qwen3.6 share the same pre-trained base.** All differences between them are post-training differences: different training data, different RL reward signals, and different inference-time techniques. The pre-trained weight initialization and architectural choices are unchanged.

This distinction is crucial for TTNN implementers: post-training differences manifest only as different weight *values* (not shapes), and as different behavioral outputs at the same input. They require no changes to hardware kernels, op graphs, or memory layouts.

---

## Agentic Coding Improvements

The dominant theme of Qwen3.6's post-training is improvement in *agentic coding* — tasks that require a model to autonomously write, execute, and debug code across multiple steps, use external tools, and recover from errors without human intervention.

### Reinforcement Learning on Agentic Tasks

Qwen3.6 was trained with RL reward signals derived from agentic coding benchmarks. Unlike standard RLHF, which uses human preferences over single-turn responses, agentic RL assigns rewards based on outcomes over multi-step trajectories:

- **Task completion reward:** Did the agent successfully complete the coding task (e.g., pass all test cases in SWE-bench)?
- **Tool use correctness:** Did the agent invoke tools (bash, file editor, search) with correct syntax and semantically valid arguments?
- **Error recovery reward:** When a tool call fails or produces unexpected output, does the agent correctly diagnose and re-attempt?
- **Efficiency penalty:** Unnecessarily long or redundant action sequences are penalized to discourage verbose scaffolding.

This RL signal creates a post-training distribution that is sharply different from Qwen3.5's, even though both models start from the same pre-trained base. The result is a model that is better calibrated for taking autonomous action in software engineering environments.

### Improved Tool Use

Tool use improvement in Qwen3.6 manifests in several concrete ways:

- **Schema adherence:** The model more reliably produces tool call JSON that matches the declared function signature, reducing parse errors in agent loops.
- **Argument grounding:** When constructing tool arguments (e.g., file paths, command strings, API parameters), the model more accurately uses values it has observed in prior context rather than hallucinating plausible-but-wrong values.
- **Tool selection:** Given a set of available tools, the model is better at selecting the minimally sufficient tool for the subtask rather than over-relying on a single general-purpose tool.

### Multi-Step Planning

For tasks that span many turns or require decomposing a high-level goal into subtasks, Qwen3.6 shows improved planning fidelity:

- The model maintains a more consistent internal plan across long trajectories, showing less tendency to drift from the original goal.
- It better tracks which subtasks have been completed, which are in progress, and which remain, reducing redundant work.
- It applies structured decomposition (e.g., identifying files to modify before making changes) more reliably.

### Error Recovery

When an execution step fails — a syntax error, a test failure, a missing dependency — Qwen3.6 shows improved recovery:

- It more accurately identifies the root cause of the failure from error output.
- It constructs targeted fixes rather than broad rewrites that may introduce new errors.
- It recognizes when a failure is non-recoverable and escalates or reports appropriately rather than entering a retry loop.

---

## Thinking Preservation

*Thinking Preservation* is an inference-time technique introduced with Qwen3.6. It is not an architectural feature — it does not add new layers, change attention patterns, or modify weight structures. It is a prompting and context management strategy combined with post-training alignment.

### The Problem It Solves

In multi-turn conversations where a model uses extended reasoning (chain-of-thought, scratchpad thinking), the intermediate reasoning steps from earlier turns are typically discarded from the context to save tokens. This causes the model to lose access to intermediate conclusions it had already reached, forcing it to re-derive them or, worse, to re-derive them incorrectly.

### How It Works

Thinking Preservation compresses and retains key conclusions from prior-turn reasoning chains rather than discarding them entirely:

1. After a reasoning turn completes, a compression pass identifies the key intermediate conclusions (e.g., "the bug is in function X because of Y; the fix requires changing Z").
2. These conclusions are serialized in a compact format and prepended to the next turn's context as a *reasoning summary*.
3. The model's post-training distribution (shaped by Qwen3.6's alignment) is specifically trained to attend to and build on these reasoning summaries, rather than treating them as noise.

### Post-Training Component

The post-training component of Thinking Preservation is the alignment signal that teaches the model to:

- Produce reasoning summaries in the expected compact format.
- Treat incoming reasoning summaries as authoritative prior context.
- Avoid re-deriving conclusions that are already summarized (reducing hallucinated re-derivations).

Without this alignment, even if reasoning summaries are injected into the context, a model trained on standard RLHF would not reliably utilize them. Qwen3.6's RL training on multi-turn agentic tasks explicitly rewards correct utilization of prior reasoning summaries.

### Why This Is Not Architectural

Thinking Preservation operates entirely through:

- The token context window (standard transformer input).
- The model's learned behavior (post-training alignment).
- Prompt engineering conventions (summary format).

No new attention mechanisms, no new weight matrices, no new positional encoding schemes, and no new operators are introduced. Any TTNN forward pass that correctly handles the standard Qwen3.5 context window will support Thinking Preservation without modification.

---

## Weight-Level Differences

### Shapes and Dtypes: Identical

Every weight tensor in Qwen3.6 has the same shape and dtype as the corresponding tensor in Qwen3.5. This follows directly from the architectural equivalence established in `config_diff.md`. The shapes are fully determined by config fields like `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `moe_intermediate_size`, etc. — all of which are identical.

For the key weight tensors:

| Tensor | Shape (both 3.5 and 3.6) | Dtype (both) |
|--------|--------------------------|--------------|
| Token embedding | `[248320, 2048]` | `bfloat16` |
| DeltaNet Q projection (per layer) | `[16 * 128, 2048]` | `bfloat16` |
| DeltaNet K projection (per layer) | `[16 * 128, 2048]` | `bfloat16` |
| DeltaNet V projection (per layer) | `[32 * 128, 2048]` | `bfloat16` |
| DeltaNet out_proj (per layer) | `[4096, 2048]` | `bfloat16` |
| Full attention Q projection (per layer) | `[16 * 256, 2048]` | `bfloat16` |
| Full attention K/V projection (per layer) | `[2 * 256, 2048]` each | `bfloat16` |
| MoE router weight (per MoE layer) | `[128, 2048]` | `bfloat16` |
| MoE expert gate/up (per expert per layer) | `[2 * 1536, 2048]` | `bfloat16` |
| MoE expert down (per expert per layer) | `[2048, 1536]` | `bfloat16` |
| Shared expert gate/up (per layer) | `[2 * 768, 2048]` | `bfloat16` |
| LM head (unembedding) | `[248320, 2048]` | `bfloat16` |

### Values: Different

The weight values differ because Qwen3.6's post-training applied a different distribution of gradient updates than Qwen3.5's. The RL on agentic tasks, the Thinking Preservation alignment, and the different fine-tuning data all shift the weight values away from Qwen3.5's distribution.

The degree of difference varies by layer and by weight type. Empirically, post-training typically produces smaller weight updates in the early embedding layers (which encode broad language knowledge that does not change) and larger updates in the later layers (which encode task-specific behavior).

### Implications for Weight Loading Code

Because shapes and dtypes are identical, any weight loading code written for Qwen3.5 will load Qwen3.6 weights without modification. This includes:

- Code that maps HuggingFace checkpoint key names to TTNN buffer handles.
- Code that performs dtype casting (e.g., `bfloat16` → `float32` for specific ops).
- Code that reshapes or transposes weights for specific TTNN kernel layouts.
- Code that shards weights across multiple devices for tensor parallelism.

No special casing for Qwen3.6 weight names, shapes, or dtypes is needed.

---

## Summary

| Aspect | Qwen3.5 | Qwen3.6 | Impact on TTNN |
|--------|---------|---------|----------------|
| Pre-trained base | shared | shared | none |
| Agentic RL training | not applied | RL on SWE-bench, tool use, multi-step tasks | none (behavior only) |
| Thinking Preservation | not applied | post-training alignment + inference-time | none (context window feature) |
| Weight shapes | reference | identical | none |
| Weight dtypes | `bfloat16` | `bfloat16` | none |
| Weight values | (3.5 alignment) | (3.6 alignment) | load different checkpoint, no code changes |

---

**Next:** [`benchmark_comparison.md`](./benchmark_comparison.md)
