# dots.ocr on TT Hardware

## Guide Overview

This guide documents the TTNN port of [rednote-hilab/dots.ocr](https://huggingface.co/rednote-hilab/dots.ocr), a ~3.0B-parameter multimodal OCR model, targeting Tenstorrent hardware via the `ign/dots_ocr` branch of `tenstorrent/tt-metal`. It covers the full model architecture, the TTNN port structure, the on-device vision stack, T3K topology constraints, and current implementation status. The intended audience is ML engineers and systems integrators working on `tt_symbiote` who need to evaluate, adapt, or deploy the dots.ocr TTNN port in a production or staging environment.

## Quick Summary

| Topic | Value |
|-------|-------|
| Model | `rednote-hilab/dots.ocr` |
| Branch | `tenstorrent/tt-metal @ ign/dots_ocr` |
| Total parameters | ~3.0B (~1.78B text decoder + ~1.22B vision encoder) |
| Text decoder | 28 layers, `hidden_size=1536`, GQA 12Q/2KV, `attention_bias=True` |
| Vision encoder | 42-layer ViT, `post_norm=True`, `patch_size=14`, `spatial_merge_size=2` |
| Max TP on T3K | 2 (`gcd(12,2)=2`) |
| Confirmed PCC | >0.98 text decoder prefill (commit "prefill at 0.98") |
| Target PCC | >0.99 per `IMPLEMENTATION_STEPS.md` (not confirmed by commit history) |
| `image_token_id` | 151665 (Qwen2.5-VL uses 151655 — different by 10) |

## Chapter Guide

| Chapter | Title | Key Question Answered |
|---------|-------|----------------------|
| [1](ch1_model_architecture/index.md) | dots.ocr Model Architecture | What are the exact hyperparameters and how does dots.ocr differ from Qwen 2.5 VL? |
| [2](ch2_ttnn_port_architecture/index.md) | TTNN Port Architecture | How is the port structured and how is PCC validated? |
| [3](ch3_full_ttnn_vision_stack/index.md) | Full TTNN Vision Stack | How are all 42 ViT layers implemented on-device in TTNN? |
| [4](ch4_t3k_topology_and_gqa_constraint/index.md) | T3K Topology and GQA Constraint | Why is TP capped at 2 and how does the T3K submesh address this? |
| [5](ch5_implementation_status_and_deployment/index.md) | Implementation Status and Deployment | What is confirmed working, what needs verification, and how to integrate with `tt_symbiote`? |

## Reading Order

Read the chapters sequentially if you are new to the port — each chapter builds on the previous one's terminology and structural context.

Chapter 1 is the prerequisite for all others; it establishes the exact hyperparameters and the relationship to Qwen 2.5 VL that are referenced throughout.

Jump directly to Chapter 4 if your primary concern is topology constraints and deployment configuration on T3K hardware.

Jump directly to Chapter 5 if you need the integration checklist for `tt_symbiote` or want a summary of what is confirmed working versus what still requires verification.

## Critical Facts for tt_symbiote Integrators

- **TP is hard-capped at 2** (`gcd(12,2)=2`). Attempting TP=4 or TP=8 triggers a shape assertion failure at startup. See [Chapter 4](ch4_t3k_topology_and_gqa_constraint/index.md) for the derivation and submesh setup.

- **dots.ocr holds all 8 T3K devices** even at TP=2; schedule it as an 8-device workload. See [Chapter 4 — t3k_submesh_and_env_vars.md](ch4_t3k_topology_and_gqa_constraint/t3k_submesh_and_env_vars.md) for scheduling implications.

- **`image_token_id=151665`** differs from Qwen2.5-VL's `151655` by 10. Hardcoding the Qwen value inserts vision tokens at wrong positions with no error at encoding time.

- **PCC >0.99 is a target, not confirmed.** The only commit-verified figure is >0.98 (text decoder prefill). Run `test_vision_tower_pcc.py` and `test_e2e_pcc.py` on T3K before production. See [Chapter 5](ch5_implementation_status_and_deployment/index.md).

- **Commit 6 may be incomplete** ("Intermediate changes removing qwen reference"). Grep for residual `Qwen*` symbols before deployment. See [Chapter 5 — commit_history_and_stabilization.md](ch5_implementation_status_and_deployment/commit_history_and_stabilization.md).
