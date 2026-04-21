# Benchmark Comparison: Qwen3.5 vs Qwen3.6

## Overview

This document presents benchmark results comparing Qwen3.5-35B-A3B and Qwen3.6-35B-A3B across three categories:

1. **Agentic coding** — the primary target of Qwen3.6's post-training improvements.
2. **General reasoning** — showing that alignment improvements did not degrade core capabilities.
3. **Vision** — showing competitive positioning for multimodal tasks.

All results are reported as `Qwen3.5 → Qwen3.6 (delta)`. A positive delta indicates improvement.

---

## Agentic Coding Benchmarks

Agentic coding benchmarks evaluate a model's ability to autonomously solve software engineering tasks across multiple steps. They are the primary axis of differentiation between Qwen3.5 and Qwen3.6.

| Benchmark | Qwen3.5-35B-A3B | Qwen3.6-35B-A3B | Delta |
|-----------|-----------------|-----------------|-------|
| SWE-bench Verified | 70.0 | 73.4 | +3.4 |
| SWE-bench Multilingual | 60.3 | 67.2 | +6.9 |
| SWE-bench Pro | 44.6 | 49.5 | +4.9 |
| Terminal-Bench 2.0 | 40.5 | 51.5 | +11.0 |
| SkillsBench Avg5 | 4.4 | 28.7 | +24.3 |
| NL2Repo | 20.5 | 29.4 | +8.9 |
| QwenWebBench | 978 | 1397 | +419 |
| Claw-Eval Avg | 65.4 | 68.7 | +3.3 |

### Benchmark Descriptions

**SWE-bench Verified** evaluates the ability to resolve real GitHub issues by modifying code in existing repositories. The "Verified" split contains issues that have been manually verified to have clear, unambiguous solutions. A score of 73.4 means the model successfully resolved 73.4% of issues autonomously.

**SWE-bench Multilingual** extends SWE-bench to repositories in languages other than Python (Java, TypeScript, Go, Rust, C++, etc.). The larger delta here (+6.9 vs +3.4 on Verified) suggests that Qwen3.6's agentic RL training included multilingual repositories and was particularly effective at generalizing across language ecosystems.

**SWE-bench Pro** is a harder variant with more complex, ambiguous, or multi-file issues that require broader repository understanding. The +4.9 improvement indicates that Qwen3.6's multi-step planning improvements are effective on harder tasks, not only the typical-difficulty issues in the Verified split.

**Terminal-Bench 2.0** evaluates command-line tool use, system administration tasks, and shell scripting. The +11.0 improvement is one of the largest absolute gains, consistent with the post-training emphasis on correct tool invocation and error recovery in terminal environments.

**SkillsBench Avg5** evaluates five specific software engineering sub-skills (e.g., testing, documentation, refactoring, debugging, code review). The dramatic +24.3 improvement is the largest relative gain in the table. This suggests that Qwen3.5 had significant headroom in these targeted skills and that Qwen3.6's fine-tuning data specifically targeted them.

**NL2Repo** measures the ability to generate or modify a repository from a natural-language specification, requiring the model to create multiple files with consistent interfaces. The +8.9 improvement reflects better multi-file coordination and planning.

**QwenWebBench** is Qwen's proprietary web agent benchmark measuring the ability to complete tasks in browser environments (searching, form filling, navigation, information extraction). The +419 point improvement (on a different scale from percentage points) is large in absolute terms; expressed as a percentage relative to the Qwen3.5 baseline, this is approximately a 43% relative improvement.

**Claw-Eval Avg** evaluates code generation and editing across a suite of real-world tasks. The +3.3 improvement is the smallest in the table, suggesting that straightforward code generation was already strong in Qwen3.5 and left less room for improvement.

### Analysis of Agentic Gains

The pattern of gains across agentic benchmarks reveals where Qwen3.6's post-training was most effective:

- **Largest gains in non-Python and multi-environment tasks** (SWE-bench Multilingual +6.9, Terminal-Bench +11.0, SkillsBench +24.3): These categories likely had the most headroom and received the most targeted RL signal.

- **Moderate gains in core Python software engineering** (SWE-bench Verified +3.4, SWE-bench Pro +4.9, Claw-Eval +3.3): Qwen3.5 was already strong here; Qwen3.6 pushes further but with diminishing returns.

- **Large gains in multi-file and web agent tasks** (NL2Repo +8.9, QwenWebBench +419): These require exactly the kind of multi-step planning and tool use coordination that Qwen3.6's RL training targeted.

---

## General Reasoning Benchmarks

These benchmarks evaluate capabilities that were not the primary target of Qwen3.6's post-training. They serve as a sanity check that improving agentic coding did not degrade general reasoning — a common concern with targeted RL fine-tuning.

| Benchmark | Qwen3.5-35B-A3B | Qwen3.6-35B-A3B | Delta |
|-----------|-----------------|-----------------|-------|
| GPQA Diamond | 84.2 | 86.0 | +1.8 |
| MMLU-Pro | 85.3 | 85.2 | -0.1 |
| AIME 2026 | 91.0 | 92.7 | +1.7 |
| LiveCodeBench v6 | 74.6 | 80.4 | +5.8 |

### Benchmark Descriptions

**GPQA Diamond** (Graduate-Level Google-Proof Q&A) evaluates expert-level scientific reasoning across biology, chemistry, and physics. The +1.8 improvement is small but positive, suggesting that agentic RL training did not harm and marginally helped structured scientific reasoning.

**MMLU-Pro** (Massive Multitask Language Understanding Professional) is a broad knowledge benchmark across 57 domains. The -0.1 change (85.3 → 85.2) is within measurement noise and should be interpreted as **no change**. This is the expected result: MMLU-Pro measures memorized factual knowledge, which post-training does not meaningfully alter.

**AIME 2026** (American Invitational Mathematics Examination) evaluates mathematical competition problem-solving. AIME 2026 refers to the February 2026 contest problems; the model was evaluated post-release in April 2026, which is standard practice. Note: the Qwen3.6 README explicitly states it uses the full AIME 2026 (I & II), and scores may differ from Qwen3.5 notes due to this expanded problem set. The +1.7 improvement (91.0 → 92.7) is small but positive, consistent with the hypothesis that better multi-step planning from agentic RL has some transfer benefit to multi-step mathematical reasoning.

**LiveCodeBench v6** evaluates competitive programming on problems collected after the model's training cutoff, minimizing contamination. The +5.8 improvement is notably larger than the other general benchmarks. This is consistent with coding being a shared capability between agentic software engineering and competitive programming: improvements in code planning and error recovery from the agentic RL likely transfer to competitive programming as well.

### Analysis: No Capability Regression

The general benchmark results confirm three things:

1. **No regression in knowledge-intensive tasks**: MMLU-Pro is essentially flat, confirming that the pre-trained base's knowledge was not degraded by post-training.

2. **Small positive transfer to adjacent reasoning tasks**: GPQA and AIME show small positive deltas, suggesting that the structured reasoning induced by agentic RL has some transfer to scientific and mathematical reasoning.

3. **Meaningful transfer to coding**: LiveCodeBench shows a +5.8 improvement, consistent with the shared skill set between competitive programming and software engineering.

---

## Vision Benchmarks

Both Qwen3.5 and Qwen3.6 are multimodal models with a shared vision encoder. The vision encoder and vision-language connector were not the focus of Qwen3.6's post-training improvements, but vision benchmark results are included here for completeness and positioning.

### Competitive Context

On standard vision-language benchmarks (MMBench, MME, MMMU, DocVQA, ChartQA, and similar), Qwen3.6-35B-A3B is competitive with or better than:

- **Claude Sonnet 4.5** (as listed in the Qwen3.6 HuggingFace model card) — a significantly larger closed model — on document understanding and chart reasoning tasks.
- **Gemma4-31B** — a similarly-sized open model — across most vision benchmarks.

### Interpretation

Because the vision encoder is architecturally identical between Qwen3.5 and Qwen3.6 (as confirmed in `config_diff.md`), vision benchmark improvements (where they exist) are attributable to post-training on vision-language instruction data that accompanied the agentic RL training, not to any change in vision architecture.

For TTNN purposes, the vision encoder implementation requires no changes between Qwen3.5 and Qwen3.6. Chapter 6 covers the vision encoder architecture in detail.

---

## Consolidated Analysis

### Where the Gains Come From

The benchmark results, taken together, tell a consistent story:

| Category | Typical Delta | Source of Improvement |
|----------|-------------|----------------------|
| Agentic coding (multi-step, tools, error recovery) | +5 to +25 | Direct target of RL training |
| Agentic coding (core Python SWE) | +3 to +5 | Partial improvement, less headroom |
| General reasoning (knowledge) | ~0 | Post-training does not change knowledge |
| General reasoning (math/science) | +1 to +2 | Small transfer from structured reasoning |
| Competitive programming | ~+6 | Shared skill set with SWE |
| Vision | competitive | Minimal change; encoder unchanged |

### Implication: No Hardware Optimization Changes Required

The performance improvements in Qwen3.6 are entirely attributable to better weight values resulting from more effective post-training. No new operators, no new memory access patterns, no new kernel requirements, and no new quantization considerations arise.

A TTNN implementation optimized for Qwen3.5-35B-A3B will deliver the same performance gains relative to Qwen3.5 on CPU/GPU as the benchmark deltas suggest — simply by loading Qwen3.6 weights. The hardware execution profile is identical: the same number of FLOPS per token, the same memory bandwidth requirements, the same MoE routing overhead, and the same attention computation cost.

There is no scenario in which switching from Qwen3.5 to Qwen3.6 weights would degrade TTNN throughput or latency. The two models are hardware-equivalent; Qwen3.6 is simply better-aligned.

---

**Next:** [Chapter 4 — Partial Rotary Embedding and M-RoPE](../ch4_rope_and_mrope/index.md)
