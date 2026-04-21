# Research Guide Plan: Multi-Token Prediction (MTP) on TT Hardware

## Topic Context

Qwen3.6-35B-A3B exposes `mtp_num_hidden_layers: 1` in its model configuration, indicating the presence of a Multi-Token Prediction head. MTP is a training-time auxiliary objective that teaches the model to predict multiple future tokens simultaneously. Whether this head is active at inference, how it is structured, and whether it can accelerate decode throughput on Tenstorrent hardware via speculative decoding are open questions that this guide addresses.

---

## Audience

**Primary audience:** ML systems engineers and inference-runtime developers who are integrating Qwen3.6-35B-A3B (or similar MTP-equipped models) into the TT-Symbiote / `tt-transformers` stack on Tenstorrent hardware.

**What they already know:**
- Transformer architecture fundamentals: attention, FFN, residual connections, KV cache.
- Standard autoregressive decode: the token-by-token generation loop, prefill vs. decode phases.
- Basic familiarity with HuggingFace Transformers: `AutoModelForCausalLM`, `model.generate()`, model configuration files.
- Working knowledge of the TTNN tensor operation API and at least one prior Tenstorrent model bring-up.
- General awareness of speculative decoding as a concept (draft model + verifier), though not necessarily its implementation details.

**What they do NOT need to know in advance:**
- The MTP training objective or how `mtp_num_hidden_layers` affects the training graph (covered in Chapter 1).
- The exact weight layout and parameter count of the MTP head (covered in Chapter 2).
- HuggingFace's inference-time treatment of MTP heads — whether they are forwarded or skipped (covered in Chapter 3).
- How speculative decoding maps onto TT hardware execution (covered in Chapter 4).
- Practical TTNN implementation strategy for an MTP-based speculative decoder (covered in Chapter 5).

---

## Chapter List

### Chapter 1 — Multi-Token Prediction: Training Objective and Architecture

**Description:** Establishes what MTP is, why it exists as a training objective, and how the MTP head is attached to the main transformer backbone architecturally.

**Directory:** `ch01_mtp_foundations/`

**Files:**

- `index.md`
  - Chapter overview and reading guide; no prerequisites beyond baseline audience knowledge.
  - Summary of the three things this chapter establishes: (1) MTP as a training-time loss, (2) the architectural attachment point of the MTP head, and (3) the terminology used throughout the guide.
  - Navigation links to each sub-topic file.

- `mtp_training_objective.md`
  - Original MTP formulation: at training step $t$, in addition to predicting token $t+1$, the model also produces auxiliary predictions for tokens $t+2, \ldots, t+N$ using intermediate hidden states.
  - Motivation: enriches gradient signal, encourages the backbone to encode multi-step context, improves training efficiency without changing inference-time architecture for standard AR generation.
  - Key hyperparameter: `mtp_num_hidden_layers` controls how many transformer blocks the MTP head contains; for Qwen3.6-35B-A3B this is 1.
  - Loss weighting: MTP auxiliary loss is typically a weighted sum with the primary next-token cross-entropy loss; the weight is a training hyperparameter that does not appear in the inference config.
  - Comparison to other multi-step training objectives: token-level knowledge distillation, consistency regularization, and why MTP is distinct.

- `mtp_head_architecture.md`
  - Structural description of the MTP head: a stack of `mtp_num_hidden_layers` transformer decoder blocks, each receiving the main backbone's hidden state at the corresponding layer depth as input.
  - Shared vs. unshared parameters: whether the MTP head's transformer blocks share weights with any backbone layers (model-dependent; Qwen3.6 uses separate weights).
  - Input to the MTP head: the final hidden state of the main backbone after the last backbone layer, concatenated with a shifted embedding of the current prediction token.
  - Output: a sequence of `N` additional logit tensors (one per future-token position) produced from the MTP head's final hidden state via the shared language model head (`lm_head`).
  - Diagram description: backbone layers → final hidden state → MTP head block(s) → auxiliary logits; `lm_head` shared between backbone and MTP head.
  - Distinction between `mtp_num_hidden_layers: 1` (one additional transformer block) and a multi-layer MTP head.

- `qwen36_mtp_config.md`
  - Relevant fields from the Qwen3.6-35B-A3B configuration: `mtp_num_hidden_layers: 1`, hidden dimension, intermediate dimension, number of attention heads.
  - Which weight keys in the checkpoint correspond to the MTP head (naming convention: `model.mtp_*` or equivalent; to be confirmed during research).
  - Whether the MTP head uses the same attention and FFN hyperparameters as the backbone layers or has its own reduced configuration.
  - Relationship to `mtp_num_hidden_layers` in the Qwen3.5 lineage: does Qwen3.5-35B-A3B have this field, and if absent, what does that imply about when MTP was introduced.

---

### Chapter 2 — MTP Head Weight Shapes and Memory Footprint

**Description:** Catalogs the exact weight tensors comprising the MTP head, derives their sizes relative to the main model, and determines whether they require DRAM placement or could reside in L1.

**Directory:** `ch02_mtp_weights_and_memory/`

**Files:**

- `index.md`
  - Chapter overview; prerequisite: Chapter 1.
  - Goal: produce a concrete memory budget for the MTP head that informs placement decisions in later chapters.

- `mtp_weight_inventory.md`
  - Enumeration of all weight tensors in the MTP head for `mtp_num_hidden_layers: 1`: self-attention projection matrices (`W_q`, `W_k`, `W_v`, `W_o`), FFN matrices (gate, up, down projections), layer norm parameters, and any additional projection needed to combine backbone hidden state with the shifted token embedding.
  - Shape of each tensor: expressed symbolically in terms of `H` (hidden dimension), `num_heads`, `head_dim`, `intermediate_dim`; then instantiated with Qwen3.6-35B-A3B's concrete values.
  - Parameter count: total number of MTP head parameters vs. total backbone parameters; express as a percentage.
  - Whether the MTP head uses a Grouped Query Attention (GQA) configuration matching the backbone, or full multi-head attention.
  - Whether the MTP head includes a MoE FFN or a dense FFN — this is a critical distinction given the backbone's MoE architecture.

- `mtp_memory_footprint.md`
  - Weight memory in BF16 (2 bytes per parameter): total MTP head weight size in MiB.
  - Comparison to a single backbone transformer block weight size: is the MTP head heavier, lighter, or comparable per block?
  - Activation memory during an MTP head forward pass: shape `[batch, seq_len, H]` for the hidden state; intermediate buffers for attention and FFN.
  - Wormhole L1 capacity per device: aggregate L1 across all Tensix cores; whether MTP head weights fit in L1 entirely.
  - DRAM bandwidth cost of streaming MTP head weights from DRAM during decode: estimated time at Wormhole DRAM bandwidth.
  - Placement recommendation: L1 residency feasibility check given typical decode-phase L1 budget after backbone weights.

- `mtp_vs_backbone_compute_cost.md`
  - FLOP count for one MTP head forward pass at batch size 1 (single-token decode): attention FLOPs + FFN FLOPs.
  - Comparison to one backbone transformer block: ratio of MTP head FLOPs to backbone block FLOPs.
  - Comparison to full backbone forward pass: MTP head cost as a fraction of total model cost.
  - Arithmetic intensity of MTP head operations: weight-bound vs. activation-bound at decode batch sizes 1–32.
  - Implication: whether the MTP head cost is negligible overhead or a meaningful fraction of per-step latency.

---

### Chapter 3 — MTP in HuggingFace Transformers: Training-Only or Inference-Active?

**Description:** Investigates exactly how HuggingFace's implementation handles the MTP head during `model.generate()` — whether it participates in the forward pass, is silently skipped, or raises an error.

**Directory:** `ch03_mtp_in_huggingface/`

**Files:**

- `index.md`
  - Chapter overview; prerequisite: Chapter 1.
  - Framing question: from the perspective of a TT-Symbiote bring-up engineer, does the MTP head need to be ported or can it be safely ignored?

- `huggingface_mtp_forward_pass.md`
  - Walkthrough of the HuggingFace `Qwen3_5MoeForConditionalGeneration.forward()` method: where the MTP head is invoked, under what conditions.
  - The `use_cache` and `output_hidden_states` flags: do they affect whether MTP head is exercised.
  - The training/eval mode distinction: `model.train()` vs. `model.eval()` and whether MTP loss computation is gated on `self.training`.
  - Whether `model.generate()` ever triggers MTP head computation: examination of the GenerationMixin loop and what it passes to `forward()`.
  - Conclusion: definitive statement of whether MTP is training-only or inference-active in standard HuggingFace usage.

- `mtp_weight_loading_behavior.md`
  - What happens when `AutoModelForCausalLM.from_pretrained` loads a checkpoint containing MTP weights: are they loaded into the model object or ignored?
  - Whether missing or unexpected MTP weight keys trigger warnings vs. hard errors during loading.
  - The `ignore_mismatched_sizes` and `_keys_to_ignore_on_load_missing` mechanisms: does the Qwen3.6 model class use them for MTP keys?
  - Practical implication: when porting to TT-Symbiote, which weight keys must be handled (loaded into TTNN tensors) and which can be discarded without correctness impact.
  - Verification strategy: loading the model in HuggingFace with `output_attentions=False`, running a generation call, and confirming MTP head participation via hooks or logging.

- `mtp_inference_activation_scenarios.md`
  - Scenario A (standard generation): `model.generate()` with greedy or sampling — MTP head participation status.
  - Scenario B (manual speculative decoding): using the MTP head's draft logits to propose candidate tokens, then verifying with the backbone — requires explicit code; not provided by default HuggingFace `generate()`.
  - Scenario C (future HuggingFace speculative decoding integration): whether HuggingFace's `AssistantModel`-based speculative decoding can use the MTP head as the draft model.
  - Summary table: scenario, MTP head active?, code change required, correctness risk.

---

### Chapter 4 — Speculative Decoding with MTP on TT Hardware

**Description:** Explains the speculative decoding algorithm, shows how MTP's single-head architecture maps onto it, and analyzes the theoretical throughput improvement achievable on Tenstorrent hardware.

**Directory:** `ch04_speculative_decoding_with_mtp/`

**Files:**

- `index.md`
  - Chapter overview; prerequisites: Chapters 1, 2, and 3.
  - Framing: this chapter is relevant only if Chapter 3 concludes that MTP can be used at inference time (either natively or with modifications); if MTP is purely training-only, this chapter documents the theoretical case for future implementation.

- `speculative_decoding_primer.md`
  - Standard speculative decoding algorithm: a small draft model proposes $K$ tokens; the large verifier model runs a single forward pass over all $K$ draft tokens; accepted tokens are kept, the first rejected token is resampled.
  - Acceptance rate $\alpha$: the probability that a draft token matches the verifier's distribution; expected number of tokens accepted per step = $\frac{1 - \alpha^{K+1}}{1 - \alpha}$.
  - Throughput gain formula: speedup = $\frac{\text{expected accepted tokens per step}}{\text{cost of one verifier step} + \text{cost of draft step}}$.
  - Why this is compelling for memory-bandwidth-bound decode: the verifier step dominates cost; draft tokens are nearly free if the draft model is cheap.
  - Distinction between an external draft model (separate network) and a self-speculative approach (MTP head attached to the same backbone).

- `mtp_as_draft_model.md`
  - How the MTP head maps onto the draft-model role in speculative decoding: the MTP head produces $N$ auxiliary logit distributions for positions $t+1, \ldots, t+N$ from a single backbone forward pass.
  - Key advantage of MTP-based speculative decoding: the draft and backbone share the same hidden state; no separate draft model forward pass is needed.
  - Algorithm walkthrough for MTP-based speculative decode:
    1. Run backbone forward pass; obtain logit for position $t+1$ and MTP auxiliary logits for $t+2, \ldots, t+N$.
    2. Sample draft tokens $\hat{x}_{t+2}, \ldots, \hat{x}_{t+N}$ from MTP auxiliary logits.
    3. Run a single backbone forward pass with the $N$ draft tokens appended; obtain verifier logits for positions $t+2, \ldots, t+N+1$.
    4. Accept/reject each draft token using the standard speculative decoding criterion.
    5. Advance by $1 + \text{accepted}$ tokens.
  - Distinction from standard speculative decoding: draft cost folds into the backbone step; the verifier step is the only additional compute.
  - Open question: whether `mtp_num_hidden_layers: 1` produces draft logits of sufficient quality for a useful acceptance rate.

- `throughput_analysis_on_tt_hardware.md`
  - Baseline decode throughput: tokens per second for Qwen3.6-35B-A3B on a single P150 or T3K, at batch size 1.
  - Cost breakdown of one standard decode step: backbone forward pass latency (memory-bandwidth-bound at batch 1).
  - Cost of one MTP-assisted speculative step: one backbone pass (same cost) + one verification backbone pass; verification pass can potentially be batched with the next user request.
  - Expected speedup as a function of acceptance rate $\alpha$ and draft depth $N$: derive the breakeven $\alpha$ above which MTP speculative decoding is faster than standard AR.
  - Wormhole-specific considerations: the verification pass must process $N$ tokens in parallel (prefill-style); at small $N$ (e.g., $N = 4$), this is still memory-bandwidth-bound; estimate latency.
  - Estimate: for $\alpha = 0.7$ and $N = 3$, compute the expected throughput multiplier on TT hardware.
  - Comparison to external draft model speculative decoding: why MTP self-speculative decoding avoids the memory cost of loading a second model.

- `acceptance_rate_estimation.md`
  - Factors driving MTP draft acceptance rate: how well the MTP head (trained with one auxiliary transformer block) approximates the backbone's next-token distribution.
  - Literature values: DeepSeek-V3 and similar MTP-trained models report acceptance rates of 0.7–0.85 on coding and math benchmarks.
  - Domain dependence: acceptance rates vary by task; conversational text typically higher than code.
  - How to empirically measure acceptance rate: run backbone + MTP head on a held-out dataset; compute fraction of MTP draft tokens accepted by the backbone verifier.
  - Sensitivity of throughput gain to $\alpha$: table of speedup multipliers for $\alpha \in \{0.5, 0.6, 0.7, 0.8, 0.85\}$ at $N \in \{1, 2, 3, 4\}$.

---

### Chapter 5 — TTNN Implementation Strategy for MTP-Based Speculative Decoding

**Description:** Translates the speculative decoding algorithm from Chapter 4 into a concrete TTNN implementation plan, covering tensor flow, memory placement, kernel reuse, and integration with the existing TT-Symbiote generation loop.

**Directory:** `ch05_ttnn_implementation/`

**Files:**

- `index.md`
  - Chapter overview; prerequisites: Chapters 1–4.
  - Scope: this chapter assumes MTP is inference-active (or that the team has chosen to activate it); provides a porting and integration blueprint.
  - Note: if Chapter 3 concluded MTP is training-only in HuggingFace, the implementation plan here requires explicit modification of the generation loop.

- `mtp_head_ttnn_module.md`
  - Design of a `TTNNMTPHead` module: analogous to existing `TTNNQwen3FullAttention` and `TTNNQwen3MoE` modules.
  - Input: backbone final hidden state tensor `[batch, 1, H]` (decode) plus shifted token embedding `[batch, 1, H]`; combined via a linear projection or element-wise addition (model-dependent).
  - Internal components: one transformer decoder block (attention + FFN); reuse existing TTNN attention and FFN primitives; no new kernel development required if attention and FFN dimensions match the backbone.
  - Output: auxiliary logit tensor `[batch, 1, vocab_size]` via the shared `lm_head`.
  - Weight loading: map MTP checkpoint keys to TTNN tensor allocations; identify which keys can be safely omitted if MTP is disabled.
  - Toggle: a `use_mtp: bool` flag that routes execution through or around the MTP head without modifying the backbone path.

- `speculative_decode_loop_integration.md`
  - Modified generation loop structure:
    1. Standard backbone forward pass → primary logit + backbone hidden state.
    2. MTP head forward pass using backbone hidden state → auxiliary logits (draft logits for positions $t+2, \ldots, t+N$).
    3. Sample draft tokens.
    4. Verification backbone forward pass with draft tokens appended to context.
    5. Accept/reject logic; advance KV cache.
  - KV cache management: the verification pass extends the KV cache by $1 + \text{accepted}$ positions; handle variable-length KV cache updates.
  - Batch size interaction: at batch size $> 1$, acceptance decisions differ per sequence; managing variable token advancement across sequences in a batch.
  - Integration point in `tt-transformers`: which file and function to modify; where `TTNNMTPHead` is instantiated and called.

- `memory_placement_for_mtp.md`
  - MTP head weight placement decision: L1 if weights fit within the decode-phase L1 budget; DRAM otherwise (informed by Chapter 2 analysis).
  - Activation tensors during MTP head forward pass: `[1, 1, H]` at batch size 1; these easily fit in L1 and should be placed there.
  - KV cache for the MTP head's single attention layer: if the MTP head uses a KV cache (required for correct causal attention), its memory footprint at maximum sequence length.
  - Whether the MTP head KV cache can share the same DRAM buffer pool as backbone KV cache entries, or must be allocated separately.
  - Recommendation table: tensor class, recommended placement, rationale, fallback if L1 pressure occurs.

- `testing_and_validation.md`
  - Correctness check for `TTNNMTPHead`: compare auxiliary logits from TTNN against HuggingFace reference on a fixed input; acceptable BF16 tolerance.
  - End-to-end acceptance rate measurement on TT hardware: harness to run MTP speculative decode on a held-out prompt set and record empirical $\alpha$.
  - Throughput benchmark: tokens per second with MTP speculative decoding enabled vs. disabled; confirm speedup matches theoretical prediction from Chapter 4.
  - Regression tests: ensure enabling `use_mtp=True` does not alter the backbone's primary logit outputs.
  - Edge cases: empty prompt, maximum context length, very low acceptance rate causing rollback every step.

---

## Conventions

### Terminology

| Term | Definition |
|---|---|
| **MTP** | Multi-Token Prediction; an auxiliary training objective and associated head that predicts multiple future tokens. Always abbreviated as "MTP" after first use. |
| **MTP head** | The lightweight transformer block(s) appended to the backbone for MTP; identified by `mtp_num_hidden_layers` in the model config. |
| **backbone** | The main transformer stack of Qwen3.6-35B-A3B, excluding the MTP head. |
| **draft token** | A speculatively predicted token produced by the MTP head's auxiliary logit distribution. |
| **verifier** | In the speculative decoding context, the backbone's forward pass that accepts or rejects draft tokens. |
| **acceptance rate ($\alpha$)** | The per-position probability that a draft token is accepted by the verifier. |
| **draft depth ($N$)** | The number of future positions for which MTP produces auxiliary logits; equals `mtp_num_hidden_layers` for single-block MTP. |
| **AR (autoregressive)** | Standard token-by-token generation without speculative drafting. |
| **TT hardware** | Tenstorrent hardware; used as the collective term for both Wormhole-based P150 and T3K. Specific hardware is named when results differ. |
| **TTNN** | Tenstorrent's tensor operation library. API symbols are written in `code font` as `ttnn.<name>`. |
| **L1** | Per-core SRAM on Wormhole Tensix cores. Always written "L1" in prose. |
| **DRAM** | Off-chip memory on Wormhole devices. Always all-caps. |
| **HuggingFace** | Written as one word, capitalized, when referring to the HuggingFace Transformers library. Abbreviated as "HF" only in tables. |
| **lm_head** | The unembedding / language model head that projects hidden states to vocabulary logits. Written in `code font`. |
| **prefill** | The prompt-encoding phase of LLM inference. Lowercase, one word. |
| **decode** | The token-generation phase. Lowercase, one word. |

### Notation

- `H` — hidden dimension of the backbone (and MTP head, unless otherwise stated).
- `N` — draft depth; number of future positions predicted by the MTP head per step.
- `K` — in speculative decoding literature, the number of draft tokens proposed; equivalent to $N$ in this guide.
- `$\alpha$` — acceptance rate (per-position probability of accepting a draft token).
- `V` — vocabulary size.
- `B` — batch size (number of concurrent sequences).
- `S` — sequence length (number of tokens in the context window).
- Tensor shapes use bracket notation `[dim0, dim1, ...]` with lowercase symbolic names on first use per file.
- FLOPs are reported as floating-point multiply-accumulate operations (MACs × 2 = FLOPs).
- Memory sizes use MiB/GiB (binary) for on-chip capacities and MB/GB (decimal) for bandwidth-product estimates; the distinction is noted explicitly.
- Latency figures are in microseconds (µs) for operation-level measurements and milliseconds (ms) for per-step measurements.
- Throughput figures are in tokens per second (tok/s).
- Speedup multipliers are written as `X×` (e.g., `1.4×`).

### Formatting Rules

- All code snippets use fenced code blocks with explicit language tags: `python` for Python, `bash` for shell commands, `text` for pseudocode or algorithm steps.
- Mathematical expressions use LaTeX inline notation (`$...$`) for in-line formulas and display notation (`$$...$$`) for standalone equations; do not embed fractions in plain prose.
- Tables are used for comparisons (weight inventory, speedup vs. $\alpha$, memory placement decisions); prose paragraphs are used for reasoning and rationale.
- Every chapter's `index.md` must contain a "Prerequisites" section listing which prior chapters must be read first, or "None" for Chapter 1.
- Every file ends with a `## References` section listing cited papers, documentation, or other guide chapters in the format: `- [Label] Author(s), "Title", Venue/URL, Year.`
- Cross-chapter references use the form: "see Chapter N, `filename.md`" with the exact chapter number and filename.
- Abbreviations are spelled out on first use in each file, even if defined in another file's glossary.
- The guide uses American English spelling throughout.
- Benchmark result tables include a "Status" column marked `[placeholder — to be filled during research]` until empirical data is collected.
- Avoid passive voice in headings and section titles.

---

## Cross-Chapter Dependencies

```
Chapter 1 (MTP Foundations)
    ├── Chapter 2 (Weight Shapes and Memory Footprint)
    └── Chapter 3 (MTP in HuggingFace: Training-Only or Inference-Active?)
            └── Chapter 4 (Speculative Decoding with MTP on TT Hardware)
                    [also depends on Chapter 2]
                    └── Chapter 5 (TTNN Implementation Strategy)
                            [also depends on Chapters 2 and 3]
```

**Detailed dependency notes:**

- **Ch 2 → Ch 1:** Weight inventory and memory footprint analysis requires knowing the MTP head architecture (number of blocks, component types) established in Ch 1.
- **Ch 3 → Ch 1:** The HuggingFace code walkthrough requires knowing what the MTP head is and where it attaches (Ch 1); Chapter 2 is not required reading for Ch 3.
- **Ch 4 → Ch 1, Ch 2, Ch 3:** The speculative decoding analysis requires the architectural description (Ch 1), the memory and compute cost estimates (Ch 2), and the confirmed understanding of whether MTP participates at inference time (Ch 3).
- **Ch 5 → Ch 1, Ch 2, Ch 3, Ch 4:** The TTNN implementation plan requires all prior chapters: Ch 1 for module design, Ch 2 for memory placement decisions, Ch 3 for weight loading behavior, and Ch 4 for the generation loop algorithm.

**Specific concept forward-references to be aware of:**

- Ch 1 (`mtp_head_architecture.md`) states that the MTP head may or may not use a KV cache; the precise inference-time behavior is resolved in Ch 3 (`huggingface_mtp_forward_pass.md`). Ch 1 should flag this as an open question pointing forward to Ch 3.
- Ch 2 (`mtp_memory_footprint.md`) states a placement recommendation but defers the final decision to Ch 5 (`memory_placement_for_mtp.md`), which can weigh L1 budget against the rest of the decode-phase allocation.
- Ch 3 (`mtp_inference_activation_scenarios.md`) introduces the scenario taxonomy; Ch 4 and Ch 5 both build on Scenario B (manual speculative decoding). Ch 3 should note that Scenario B is the subject of Chapters 4 and 5.
- Ch 4 (`throughput_analysis_on_tt_hardware.md`) derives a breakeven acceptance rate; Ch 5 (`testing_and_validation.md`) provides the methodology to empirically measure whether that threshold is met on real hardware.
- Ch 4 (`mtp_as_draft_model.md`) describes the algorithm at a conceptual level; Ch 5 (`speculative_decode_loop_integration.md`) translates it into concrete generation loop modifications and names specific files in the `tt-transformers` codebase.
