# Chapter 6: Integration Plan and Testing Strategy

## Prerequisites

- Chapters 1–5 of this guide
- Text-only Qwen3.6-35B-A3B inference already working on TTNN (the starting point for all work in this chapter)

## Scope

This chapter covers the additional work needed to support **mixed text+image batches** on TTNN. It does not modify the existing text-only inference path.

> **Key Finding:** Text-only Qwen3.6-35B-A3B inference on TTNN requires zero changes to the RoPE implementation (established in Ch3). The integration work in this chapter is scoped to enabling mixed text+image batches. The primary technical challenge is implementing the 3D position ID gather described in Ch4; correctness is validated against the HuggingFace M-RoPE reference before moving to TTNN device.

The mathematical proof in Ch3 (`../ch3_text_only_reduction/mathematical_equivalence_proof.md`) shows that M-RoPE with identical position IDs on all three axes is numerically identical to standard 1D RoPE. Therefore:

- **Text-only batches**: M-RoPE support is not needed. The existing `TTNNRotaryPositionEmbedding` path is invoked unchanged.
- **Mixed text+image batches**: M-RoPE support is required. The `use_mrope=True` path is activated only when vision tokens are present in the batch.

## Files in This Chapter

| File | Contents |
|------|----------|
| `integration_steps.md` | Step-by-step integration guide: config extraction, class extension, attention module changes, position ID construction, CPU validation, TTNN device validation |
| `correctness_validation.md` | Four test cases: text-only degeneracy, single image patch, full image grid (32×32), multi-frame video |
| `tracing_and_program_cache_considerations.md` | Metal Trace compatibility, program cache hit rate analysis at decode and prefill, position ID tensor shape contract, backward compatibility |

## Chapter Roadmap

```text
Step 1: Extract mrope_section from config
         └─ Qwen3.6: [11, 11, 10] from config.rope_scaling.mrope_section

Step 2: Extend TTNNRotaryPositionEmbedding (Option A from Ch4)
         └─ Add use_mrope flag + mrope_section param
         └─ Frequency table UNCHANGED; only forward() changes

Step 3: Modify attention module forward
         └─ has_vision_tokens → dispatch to _mrope_forward path

Step 4: Build 3D position IDs for VL inputs
         └─ text: t==h==w==sequential; image: t=frame, h=row+offset, w=col+offset

Step 5: CPU reference validation (torch.equal vs HF apply_multimodal_rotary_pos_emb)

Step 6: TTNN device validation (PCC > 0.9999 vs CPU reference)
```

## Key Numbers (from Ch2, Ch4, Ch5)

| Parameter | Value | Source |
|-----------|-------|--------|
| `rotary_dim` | 64 | Ch2 (`config.partial_rotary_factor = 0.5`, `head_dim = 128`) |
| `mrope_section` | `[11, 11, 10]` | Ch2 (`config.rope_scaling.mrope_section`) |
| `max_seq_len` | 32768 | Ch2 |
| cos/sin table size | 8 MiB total | Ch4 (2 tables × 32768 × 64 × 2 bytes) |
| Additional dispatch overhead | ~25–50 µs/step | Ch5 (< 0.02% of ~250 ms decode step) |
| Additional kernel dispatches | 5 (3 gather + 2 concat) | Ch4 |

## References

- `../ch2_qwen36_mrope_config/qwen36_rope_config.md`
- `../ch2_qwen36_mrope_config/position_id_construction.md`
- `../ch3_text_only_reduction/mathematical_equivalence_proof.md`
- `../ch3_text_only_reduction/practical_implications_for_text_inference.md`
- `../ch4_ttnn_implementation/existing_ttnn_rope_gap_analysis.md`
- `../ch4_ttnn_implementation/extension_approach.md`
- `../ch4_ttnn_implementation/gather_operation_on_ttnn.md`
- `../ch5_performance_analysis/operation_cost_breakdown.md`
- `../ch5_performance_analysis/prefill_vs_decode_comparison.md`
