# Qwen3.6-35B-A3B Bring-Up Session State

**Date:** 2026-04-21
**Status:** COMPLETE — all phases passed (Phases 1–5)

---

## What Was Done This Session

### Step 1: Cloned and adapted test files (COMPLETE)

Two new files were created by copying the Qwen3.5 equivalents and updating model name references:

| New File | Copied From | Key Change |
|----------|-------------|------------|
| `tests/test_qwen3_6_35b_a3b_bottom_up.py` | `test_qwen3_5_35b_a3b_bottom_up.py` | `MODEL_NAME = "Qwen/Qwen3.6-35B-A3B"` |
| `tests/test_qwen3_6_35b_a3b.py` | `test_qwen3_5_35b_a3b.py` | `model_name = "Qwen/Qwen3.6-35B-A3B"` |

**No TTNN module changes were needed.** `modules/qwen_attention.py` and `modules/qwen_moe.py` are model-agnostic.

All paths are relative to: `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/`

---

## Test Results

### Phase 1: Individual Op Verification — ALL PASSED (6/6)

Run command used:
```bash
cd /home/ttuser/salnahari/tt-metal && unset TT_VISIBLE_DEVICES && python -m pytest \
  models/experimental/tt_symbiote/tests/test_qwen3_6_35b_a3b_bottom_up.py -v --timeout=0 \
  -k "test_ttnn_linear_qwen_weights or test_ttnn_linear_replicated_col_sharded_qwen_weights or \
      test_ttnn_linear_col_sharded_row_sharded or test_ttnn_rope_partial_rotary_qwen or \
      test_ttnn_sdpa_prefill_qwen or test_paged_kv_cache_initialization"
```

| Test | Result |
|------|--------|
| `test_ttnn_linear_qwen_weights` | PASSED |
| `test_ttnn_linear_replicated_col_sharded_qwen_weights` | PASSED |
| `test_ttnn_linear_col_sharded_row_sharded` | PASSED |
| `test_ttnn_rope_partial_rotary_qwen` | PASSED |
| `test_ttnn_sdpa_prefill_qwen` | PASSED |
| `test_paged_kv_cache_initialization` | PASSED |

**Key finding:** Qwen3.6 weights load cleanly with the existing TTNN modules. Confirmed weight compatibility with `Qwen3_5MoeForConditionalGeneration` architecture class.

---

### Phase 2: Component Tests — ALL PASSED (5/5)

Run command used:
```bash
cd /home/ttuser/salnahari/tt-metal && unset TT_VISIBLE_DEVICES && python -m pytest \
  models/experimental/tt_symbiote/tests/test_qwen3_6_35b_a3b_bottom_up.py -v --timeout=0 \
  -k "test_ttnn_moe_gate_router_qwen or test_ttnn_qwen_router_decode_accuracy or \
      test_ttnn_qwen_router_decode_expert_index_match or test_ttnn_qwen_experts_fused_sparse_matmul or \
      test_ttnn_shared_expert_mlp_qwen"
```

| Test | Result | Notes |
|------|--------|-------|
| `test_ttnn_moe_gate_router_qwen` | PASSED | |
| `test_ttnn_qwen_router_decode_accuracy` | PASSED | |
| `test_ttnn_qwen_router_decode_expert_index_match` | PASSED | |
| `test_ttnn_qwen_experts_fused_sparse_matmul` | PASSED | 15.4s heaviest test |
| `test_ttnn_shared_expert_mlp_qwen` | PASSED | PCC: 0.999908 |

---

### Phase 3: Module Tests — ALL PASSED (12/12)

Run command used (note: requires `TTNN_LINEAR_ATTN_PROJECTIONS=1`):
```bash
cd /home/ttuser/salnahari/tt-metal && unset TT_VISIBLE_DEVICES && TTNN_LINEAR_ATTN_PROJECTIONS=1 python -m pytest \
  models/experimental/tt_symbiote/tests/test_qwen3_6_35b_a3b_bottom_up.py -v --timeout=0 \
  -k "test_ttnn_qwen3_full_attention_prefill or test_ttnn_qwen3_full_attention_paged_decode or \
      test_ttnn_qwen3_full_attention_paged_decode_pcc or test_ttnn_qwen3_full_attention_multi_step_decode_drift or \
      test_ttnn_qwen3_linear_attention or test_ttnn_qwen3_linear_attention_decode_with_state or \
      test_ttnn_qwen3_linear_attention_decode_pcc or test_ttnn_qwen3_linear_attention_multi_step_decode_drift or \
      test_ttnn_qwen3_moe_full_block or test_ttnn_qwen3_moe_prefill or \
      test_ttnn_qwen3_moe_decode_with_expert_verification or test_ttnn_qwen3_moe_prefill_per_token_pcc"
```

| Test | Result |
|------|--------|
| `test_ttnn_qwen3_full_attention_prefill` | PASSED |
| `test_ttnn_qwen3_full_attention_paged_decode` | PASSED |
| `test_ttnn_qwen3_full_attention_paged_decode_pcc` | PASSED |
| `test_ttnn_qwen3_full_attention_multi_step_decode_drift` | PASSED |
| `test_ttnn_qwen3_linear_attention` | PASSED |
| `test_ttnn_qwen3_linear_attention_decode_with_state` | PASSED |
| `test_ttnn_qwen3_linear_attention_decode_pcc` | PASSED |
| `test_ttnn_qwen3_linear_attention_multi_step_decode_drift` | PASSED |
| `test_ttnn_qwen3_moe_full_block` | PASSED |
| `test_ttnn_qwen3_moe_prefill` | PASSED |
| `test_ttnn_qwen3_moe_decode_with_expert_verification` | PASSED |
| `test_ttnn_qwen3_moe_prefill_per_token_pcc` | PASSED |

Total runtime: 101.79s (1:41)

---

### Phase 4: Integration Tests — IN PROGRESS (2/7 confirmed)

**Important:** Running all 7 Phase 4 tests together in one pytest invocation causes a **segfault (exit code 139)** due to memory pressure from loading the full model multiple times. Tests must be run **one at a time** with a chip reset between each.

Run pattern for each test:
```bash
cd /home/ttuser/salnahari/tt-metal && unset TT_VISIBLE_DEVICES && tt-smi -r && \
  TTNN_LINEAR_ATTN_PROJECTIONS=1 python -m pytest \
  models/experimental/tt_symbiote/tests/test_qwen3_6_35b_a3b_bottom_up.py -v --timeout=0 \
  -k "<test_name>"
```

| Test | Result | Notes |
|------|--------|-------|
| `test_qwen3_full_attention_decoder_block` | PASSED | PCC: 0.999987, 60s runtime |
| `test_qwen3_linear_attention_decoder_block` | PASSED | 60s runtime |
| `test_qwen3_four_layer_block` | PASSED | PCC: 0.999978, ~75s |
| `test_qwen3_generation_loop` | PASSED | Token match: 1.00, ~84s |
| `test_kv_cache_consistency_multi_decode` | PASSED | 8 decode steps, 8.78s |
| `test_qwen3_four_layer_block_decode_argmax` | PASSED | PCC: 0.999980, argmax_match=True, 72s |
| `test_qwen3_generation_loop_token_match` | PASSED | Token match: 8/8 (1.00), 79s |

---

### Phase 5: E2E — ALL PASSED (2/2)

| Test | Result | Notes |
|------|--------|-------|
| `test_qwen3_6_35b_a3b_e2e` (bottom-up) | PASSED | 16 coherent tokens generated, 625s |
| `test_qwen3_6_35b_a3b` (traced) | PASSED | 128 coherent tokens generated, 881s |

Generated text was coherent in both cases. Timing stats saved to `qwen3_6_35b_a3b_timing_stats.csv`.

---

## What To Do Next

**Bring-up is COMPLETE.** All 32 tests across Phases 1–5 passed.

**Optional follow-up:**
- Mark research topics (Weight Compatibility, M-RoPE, MTP) as completed once research instance answers them
- File a PR or update model registry to include Qwen3.6-35B-A3B

---

## Historical: Original Phase 4 Plan

```bash
# Test 4.3
cd /home/ttuser/salnahari/tt-metal && unset TT_VISIBLE_DEVICES && tt-smi -r && \
  TTNN_LINEAR_ATTN_PROJECTIONS=1 python -m pytest \
  models/experimental/tt_symbiote/tests/test_qwen3_6_35b_a3b_bottom_up.py -v --timeout=0 \
  -k "test_qwen3_four_layer_block"

# Test 4.4
cd /home/ttuser/salnahari/tt-metal && unset TT_VISIBLE_DEVICES && tt-smi -r && \
  TTNN_LINEAR_ATTN_PROJECTIONS=1 python -m pytest \
  models/experimental/tt_symbiote/tests/test_qwen3_6_35b_a3b_bottom_up.py -v --timeout=0 \
  -k "test_qwen3_generation_loop"

# Test 4.5
cd /home/ttuser/salnahari/tt-metal && unset TT_VISIBLE_DEVICES && tt-smi -r && \
  TTNN_LINEAR_ATTN_PROJECTIONS=1 python -m pytest \
  models/experimental/tt_symbiote/tests/test_qwen3_6_35b_a3b_bottom_up.py -v --timeout=0 \
  -k "test_kv_cache_consistency_multi_decode"

# Test 4.6
cd /home/ttuser/salnahari/tt-metal && unset TT_VISIBLE_DEVICES && tt-smi -r && \
  TTNN_LINEAR_ATTN_PROJECTIONS=1 python -m pytest \
  models/experimental/tt_symbiote/tests/test_qwen3_6_35b_a3b_bottom_up.py -v --timeout=0 \
  -k "test_qwen3_four_layer_block_decode_argmax"

# Test 4.7
cd /home/ttuser/salnahari/tt-metal && unset TT_VISIBLE_DEVICES && tt-smi -r && \
  TTNN_LINEAR_ATTN_PROJECTIONS=1 python -m pytest \
  models/experimental/tt_symbiote/tests/test_qwen3_6_35b_a3b_bottom_up.py -v --timeout=0 \
  -k "test_qwen3_generation_loop_token_match"
```

After all Phase 4 tests pass, run **Phase 5 (E2E)**:
```bash
cd /home/ttuser/salnahari/tt-metal && unset TT_VISIBLE_DEVICES && tt-smi -r && \
  TTNN_LINEAR_ATTN_PROJECTIONS=1 python -m pytest \
  models/experimental/tt_symbiote/tests/test_qwen3_6_35b_a3b_bottom_up.py -v --timeout=0 \
  -k "test_qwen3_6_35b_a3b_e2e"
```

Then run the traced E2E test:
```bash
cd /home/ttuser/salnahari/tt-metal && unset TT_VISIBLE_DEVICES && tt-smi -r && \
  python -m pytest \
  models/experimental/tt_symbiote/tests/test_qwen3_6_35b_a3b.py -v --timeout=0 \
  -k "test_qwen3_6_35b_a3b"
```

---

## Key Operational Notes

- **Always `unset TT_VISIBLE_DEVICES` and `tt-smi -r` before each test run**
- **Always use `--timeout=0`** with pytest
- **Always use `TTNN_LINEAR_ATTN_PROJECTIONS=1`** for Phase 3+ (linear attention tests)
- **Run Phase 4+ tests one at a time** — running multiple integration tests in one process segfaults
- The 8-device T3K mesh (chips 0-7) is available and working
- HuggingFace weights for `Qwen/Qwen3.6-35B-A3B` are cached locally

## Team Lead Rules (from TEAM_PROMPT.md)

- **Only `research_topics.md` may be pushed** in the research-topics repo
- **No pushes to tt-metal** — all code changes remain local
- Architect must do research cache lookup before any planning:
  ```bash
  cd /home/ttuser/salnahari/research-topics/tt-symbiote-research-topics && git pull
  cat research_topics.md
  ```
- Research topics already pending (do not re-add):
  - Qwen3.6-35B-A3B Weight Compatibility (Pending)
  - M-RoPE Implementation on TTNN (Pending)
  - Multi-Token Prediction on TT Hardware (Pending)
