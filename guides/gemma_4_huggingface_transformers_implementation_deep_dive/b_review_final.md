# Agent B Final Review: Cross-Chapter Consistency -- Pass 1

## Issue 1: Ch4 incorrectly claims audio output_proj_dims matches text hidden_size

**Location**: `ch4_audio_encoder/index.md`, line 454

Ch4 states: "projecting from `hidden_size=1024` to `output_proj_dims=1536` to match the text decoder's hidden dimension." However, Ch2 (Section 2.2) establishes `Gemma4TextConfig.hidden_size = 2304`, not 1536. Ch6 (Section 6.3) correctly explains that the audio encoder's output (1536) feeds into `Gemma4MultimodalEmbedder`, which then projects from `multimodal_hidden_size` (1536) to `text_hidden_size` (2304). The Ch4 statement should say "to serve as input to the multimodal embedder" rather than "to match the text decoder's hidden dimension."

## Issue 2: Ch7 cites wrong section for mm_token_type_ids usage

**Location**: `ch7_preprocessing_pipelines/index.md`, line 90

Ch7 states that `mm_token_type_ids` enables "modality-specific logic during embedding (see Chapter 6, Section 6.3)." Section 6.3 covers `Gemma4MultimodalEmbedder`, which does not use `mm_token_type_ids`. The actual usage is in Section 6.5, Stage 8, where `mm_token_type_ids` drives bidirectional vision attention masking within `create_causal_mask_mapping`. The reference should point to Section 6.5 and the description should say "bidirectional attention masking for vision tokens" rather than "modality-specific logic during embedding."

## Issue 3: Navigation footer dash style is inconsistent

**Location**: All chapter footer lines

Chapters 1, 2, 6, 7, and moe_details.md use double-hyphen (`--`) in their "Next" footers (e.g., `Chapter 2 -- Configuration Hierarchy`), while Chapters 3 and 4 use em-dash (`---`). Pick one convention and apply it uniformly.

## Issue 4: Ch4 output_proj claim of matching text hidden_size repeated in end-to-end diagram

**Location**: `ch4_audio_encoder/index.md`, line 537 (end-to-end data flow diagram)

The diagram states the output `[B, T/4, 1536]` "matches the text decoder's hidden dimension, enabling direct token injection into the text decoder." This repeats the same factual error as Issue 1. The 1536-dim output does not inject directly into the text decoder; it passes through `Gemma4MultimodalEmbedder` (RMSNorm + Linear projection to 2304) first, as described in Ch6 Section 6.3. The diagram annotation should be corrected.

## Issue 5: No "Previous" navigation links in footers

**Location**: All chapter content files

Every content file has a "Next" link but none have a "Previous" link. This is a structural gap that makes backward navigation through the guide impossible without returning to the top-level index. Adding `**Previous:** [Chapter N](...)` alongside the existing "Next" links would improve navigability. This is a suggestion, not a factual error.
