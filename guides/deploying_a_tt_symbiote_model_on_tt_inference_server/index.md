# Deploying a TT Symbiote Model on tt-inference-server

This guide walks you through every step required to integrate, configure, and deploy a TT Symbiote model backend with the tt-inference-server using the vLLM v1 serving stack.

---

## Prerequisites

Before starting this guide, ensure you have the following:

**Knowledge**
- Familiarity with Python class-based APIs and `@classmethod` patterns
- Basic understanding of transformer model architecture (attention, KV cache, prefill vs. decode)
- Comfort reading vLLM source code and configuration files

**Software**
- tt-inference-server repository cloned and dependencies installed
- tt-vllm-plugin installed and accessible on `PYTHONPATH`
- `VLLM_USE_V1=1` exported in your shell environment before any Python import of vLLM
- Python 3.10+ with `torch`, `ttnn`, and `transformers` available

**Hardware**
- At least one Tenstorrent Wormhole device (N300 or T3K configuration)
- For N300 single-card isolation: two device IDs available, controlled via `TT_VISIBLE_DEVICES`
- For T3K multi-card: four N300 cards (8 Wormhole chips total) with inter-card connectivity verified
- For Galaxy: 32-chip system with mesh connectivity confirmed

---

## How-to-Use

Use the table below to navigate directly to the chapters most relevant to your goal:

| Goal | Recommended Chapters |
|---|---|
| Understand the overall serving stack and how components fit together | Chapter 1 |
| Learn what `ModelSpec`, `perf_reference`, and `ModelStatusTypes` mean | Chapter 2 |
| Implement a new model backend from scratch (the Symbiote interface) | Chapters 3, 4, 5 |
| Add a new model end-to-end with a worked example | Chapters 3–6 |
| Verify your integration is correct before submitting | Chapter 6 |
| Debug a broken integration or tune inference performance | Chapter 7 |
| Just need a quick API or hardware reference | Quick Reference section below |

---

## Chapter Index

| Chapter | What You Learn |
|---|---|
| [Chapter 1: Architecture Overview](ch1_architecture_overview/index.md) | How tt-inference-server, tt-vllm-plugin, and TTPlatform compose the full serving stack; request lifecycle from HTTP to hardware |
| [Chapter 2: ModelSpec and Configuration](ch2_model_spec/index.md) | The `ModelSpec` dataclass fields, `perf_reference` dict layout, `ModelStatusTypes` thresholds (Functional / Complete), and how configuration flows from `config.json` into the model registry |
| [Chapter 3: Model Contract — Implementing the Symbiote Interface](ch3_model_contract/index.md) | The full Symbiote interface contract: `initialize_vllm_model`, `allocate_kv_cache`, `prefill_forward`, and `decode_forward` — their signatures, expected return types, and behavioral requirements |
| [Chapter 4: Weights and Tokenization](ch4_weights_and_tokenization/index.md) | How model weights are loaded onto TT hardware, tokenizer integration, and the mapping between HuggingFace checkpoint layout and tt-inference-server weight conventions |
| [Chapter 5: Hardware Initialization](ch5_hardware_init/index.md) | Mesh device setup for N300, T3K, and Galaxy topologies; `MeshShape` configurations; `TT_VISIBLE_DEVICES` isolation; device ID assignment and teardown |
| [Chapter 6: Integration Checklist and Worked Example](ch6_integration_checklist/index.md) | A step-by-step checklist for verifying a complete integration, plus a concrete worked example showing all required methods implemented for a real model |
| [Chapter 7: Debugging and Performance Tuning](ch7_debugging_and_tuning/index.md) | Diagnosing common failure modes (registry misses, shape mismatches, KV cache errors), profiling tools, and strategies for reaching `ModelStatusTypes.Complete` performance targets |

---

## Quick Reference

The facts below are the most commonly looked up during development and debugging.

### API Signatures and Return Types

| Item | Detail |
|---|---|
| `initialize_vllm_model` signature | `@classmethod initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size, max_seq_len, optimizations) -> "Self"` |
| `allocate_kv_cache` return type | `list` of `(k_cache, v_cache)` tuples, one tuple per layer; tensors allocated via `ttnn.zeros()` in `ROW_MAJOR_LAYOUT` |
| `prefill_forward` return shape | `(batch, seq_len, vocab_size)` — all-position logits, returned as a CPU PyTorch tensor |
| `decode_forward` return shape | `(batch, vocab_size)` — last-position logits, returned as a CPU PyTorch tensor |

### Hardware Constants

| Item | Value |
|---|---|
| Block size | **64** (fixed for TT hardware; cannot be overridden at runtime) |
| T3K mesh shape | `MeshShape(1, 8)` — 4 N300 cards × 2 Wormhole chips = 8 chips total; device IDs 0–7 |
| N300 isolation env var | `TT_VISIBLE_DEVICES=0,1` (one N300 card = 2 device IDs) |
| Galaxy mesh shape | `MeshShape(8, 4)` — 32 chips total; CCL ring runs along columns of 8 chips |

### Environment and Registry

| Item | Value / Requirement |
|---|---|
| `VLLM_USE_V1` | Must be set to `1` in the shell **before** any Python import of vLLM |
| Model registry lookup | `TTPlatform` prepends `"TT"` to the architecture name from `config.json` to locate the model class in `ModelRegistry` |
| Entry point for new model | Implement `initialize_vllm_model`; register via `register_tt_models()` in the plugin's registration hook |

### ModelStatusTypes Thresholds

| Status | Meaning |
|---|---|
| `ModelStatusTypes.Functional` | Model is within approximately 80% of all `perf_reference` targets |
| `ModelStatusTypes.Complete` | Model meets all `perf_reference` targets |

`perf_reference` is a flat `dict[str, float]` mapping metric names to target values defined in the model's `ModelSpec`.

---

## Source Code Location

The relevant source code for this guide is spread across two repositories:

- **tt-inference-server** — top-level server, request handling, and `TTPlatform` implementation; the primary repo for understanding how serving requests reach the model backend
- **tt-vllm-plugin** — contains the Symbiote interface definition, `ModelRegistry`, `ModelSpec`, `ModelStatusTypes`, and the `register_tt_models()` registration hook; see the Quick Reference table above for the model class lookup chain.

---

**Guide:** Deploying a TT Symbiote Model on tt-inference-server
