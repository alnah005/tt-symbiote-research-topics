# Chapter 5 Review -- Agent B (Correctness)

## Issue 1: Attention projection output dimensions use total heads instead of per-device heads

**File:** `batched_projections.md`, lines 23-28 and 37

**Claim:**
```
Q+gate:  x_dram @ wqkv -> [1, 1, seq_len, NH*HD*2 = 24*256*2 = 12288]
K:       x_dram @ wk   -> [1, 1, seq_len, NKV*HD  = 4*256  = 1024]
V:       x_dram @ wv   -> [1, 1, seq_len, NKV*HD  = 4*256  = 1024]
```

and:

```
gated_flat: [1, 1, seq_len, NH*HD = 6144] @ wo -> [1, 1, seq_len, dim = 5120]
```

**Problem:** These are per-device matmuls on TP-sharded weights. With TP=4, the per-device dimensions use `n_local_heads = 24 // 4 = 6` and `n_local_kv_heads = 4 // 4 = 1`. The actual per-device output shapes are:

- Q+gate: `[1, 1, seq_len, 6*256*2 = 3072]`
- K: `[1, 1, seq_len, 1*256 = 256]`
- V: `[1, 1, seq_len, 1*256 = 256]`
- Output projection input: `[1, 1, seq_len, 6*256 = 1536]`

**Source:** `model_config.py` line 278 uses `self.n_local_heads * self.head_dim * 2`; HF config has `num_attention_heads = 24` total; `attention.py` line 313 sets `NH = self.n_local_heads`. The chapter conflates total model head counts (NH=24, NKV=4) with the per-device local head counts that the code actually uses.

**Fix:** Replace `NH=24` with `NH_TP=6` and `NKV=4` with `NKV_TP=1` in the attention projection shape annotations, or explicitly note these are total-model dimensions and show the per-device values.

## Issue 2: Recurrence state size "12.6 MB" is slightly inaccurate

**File:** `gdn_prefill_strategy.md`, line 29; `state_replication.md`, line 35

**Claim:** The B=32 decode recurrence state is "12.6 MB" per device for `[384, 128, 128]` in bfloat16.

**Problem:** `384 * 128 * 128 * 2 = 12,582,912 bytes = 12.0 MiB`. The chapter says "12.6 MB" which is off by 5%. Even using decimal megabytes (1 MB = 1,000,000 bytes), it would be 12.58 MB, not 12.6 MB -- though that rounds acceptably. The discrepancy is minor but appears twice.

**Fix:** Use "approximately 12 MB" or the precise "12.6 MB (decimal)" if decimal MB is intended.

---

All other claims verified correct against source code:
- GDN projection dimensions (qkvz_dim_tp=4096, Nv_TP*2=24) match model_config.py
- Layer counts (48 GDN + 16 full attention = 64 total) match config.json layer_types
- _init_prefill_states() shapes and line references match gdn.py
- forward_prefill() line references match both attention.py and gdn.py
- Conv1d shift register logic, fused kernel call, and post-kernel processing match code
- State replication patterns (expand vs repeat, copy vs replace) match code
- KV cache replication logic matches attention.py replicate_kv_cache_to_batch()
- Cleanup/deallocation sequences match code

---

## Pass 1

Reviewer: Agent B (independent cold review against source code)
Scope: factual correctness, implementability, material conceptual accuracy

1. **`batched_projections.md` — V projection config claim is factually wrong (line 18).** The guide states "Each projection calls `create_prefill_matmul_program_config(seq_len, dim, out_dim)` directly (`attention.py`, lines 336, 344, 352)." In source (`attention.py` lines 352-356), the V projection (`vp_tt`) is built with `ttnn.linear(..., program_config=k_progcfg, ...)`, reusing the K config from line 344 — no separate `create_prefill_matmul_program_config` call is made for V. The cited line 352 points to `ttnn.linear`, not a config creation call. A downstream implementer would incorrectly infer that V has an independently computed program config, and the line citation is wrong. Fix: change "lines 336, 344, 352" to "lines 336, 344" and note that V reuses the K config (same output dimension).

2. **`gdn_prefill_strategy.md` and `state_replication.md` — B=1 recurrence state size stated as "393 KB" is wrong.** `12 * 128 * 128 * 2 bytes (bfloat16) = 393,216 bytes = 384 KiB`. The guide says "approximately 393 KB" in both files (gdn_prefill_strategy.md line 29, state_replication.md line 35). 393,216 bytes rounds to 384 KiB or 0.393 MB, not 393 KB — the guide is conflating byte count with kibibyte count. A reader doing memory budget calculations would use the wrong value. Fix: use "approximately 384 KiB" or "approximately 0.39 MB".

3. **`batched_projections.md` — attention Q+gate projection output shape uses wrong head counts (lines 27-29).** The guide states `Q+gate: [1, 1, seq_len, n_local_heads*HD*2 = 6*256*2 = 3072]` which is correct for the per-device shape, but the immediately preceding description in line 22 says "per-device TP-sharded weights, so output dimensions reflect local head counts (`n_local_heads = 6`, `n_local_kv_heads = 1` with TP=4)" — this text is accurate. However the guide description is correct; this was already flagged as fixed from a prior review. Verified accurate in this pass.

**No further actionable issues found beyond items 1 and 2 above.** All other numerical values, line references, tensor shapes, API call sequences, memory management patterns, navigation footers, display equation formatting (`$$...$$` not required as no block equations are present), and index clickable links verified correct against source.

## Pass 2

**Pass 1 issue verification:**

1. **V projection citation — FIXED.** `batched_projections.md` now correctly states "The Q+gate and K projections each call `create_prefill_matmul_program_config` directly (`attention.py`, lines 336 and 344). The V projection does not create a new config; it reuses `k_progcfg` from line 344 (`attention.py`, line 352)." This matches `attention.py` lines 352-356 where `vp_tt = ttnn.linear(..., program_config=k_progcfg, ...)`.

2. **B=1 recurrence state size — FIXED.** Both `gdn_prefill_strategy.md` (line 29) and `state_replication.md` (line 35) now read "approximately 384 KiB". The value is correct: `12 * 128 * 128 * 2 = 393,216 bytes = 384 KiB`.

**Full re-verification against source code:**

All claims re-checked against `attention.py`, `gdn.py`, `model_config.py`, and `config.json`. No new issues found:

- Attention projection dimensions (`n_local_heads=6`, `n_local_kv_heads=1`, `HD=256`, TP=4) match `config.json` (`num_attention_heads=24`, `head_dim=256`, `num_key_value_heads=4`) and `model_config.py` line 252.
- GDN dimensions (`qkvz_dim_tp=4096`, `Nv_TP=12`, `Nv_TP*2=24`, `value_dim_tp=1536`) verified from `model_config.py` lines 34-37, 263-268.
- `_init_prefill_states()` shapes match `gdn.py` lines 511-520 exactly (conv: `[1, 1, qkv_dim_tp]`; rec: `[Nv_TP, Dk, Dv]`; output: `[Nv_TP, 1, Dv]`).
- Line references in `batched_projections.md` (lines 336, 344, 352, 474-480) verified correct in `attention.py`.
- `create_prefill_matmul_program_config` description (8x8 grid, `out_subblock_h * out_subblock_w <= 4` FP32 DST constraint) matches `model_config.py` lines 138-172.
- Layer count (48 GDN + 16 full attention = 64 total) confirmed via `config.json` `layer_types` array.
- State replication patterns (`expand` for conv/KV, `repeat` for rec states), cleanup sequences, and KV cache replacement vs `ttnn.copy` distinction all match source code.
- All index.md links resolve to existing files. All four files carry correct navigation footers.
- No plain-text display equations found anywhere in the chapter.

**No feedback — chapter approved.**

## Pass 3

**All prior fixes confirmed present. Full independent re-verification against source code:**

- **Attention projection dimensions** (`n_local_heads=6`, `n_local_kv_heads=1`, `HD=256`, TP=4): confirmed via `config.json` (`num_attention_heads=24`, `num_key_value_heads=4`, `head_dim=256`) and `model_config.py` line 252. Per-device shapes in `batched_projections.md` (Q+gate=3072, K=256, V=256, output input=1536) are correct.
- **GDN dimension arithmetic** (`qkvz_dim_tp=4096`, `Nv_TP=12`, `Nv_TP*2=24`): `(GDN_QKV_DIM + GDN_Z_DIM) // 4 = (10240 + 6144) // 4 = 4096` confirmed in `model_config.py` lines 34-37, 267.
- **`_init_prefill_states()` shapes and line references** (conv `[1, 1, qkv_dim_tp]`, rec `[Nv_TP, Dk, Dv]`, output `[Nv_TP, 1, Dv]`): verified against `gdn.py` lines 501-520.
- **`forward_prefill()` line references** (`gdn.py` 578-726, projections 611-617 / 620-626, loop 636-700, output 716-722): all correct.
- **V projection reuse of `k_progcfg`** (`attention.py` line 352): confirmed, no new config created.
- **`create_prefill_matmul_program_config` description** (8x8 grid, `out_subblock_h * out_subblock_w <= 4` FP32 DST constraint, lines 146-172): verified against `model_config.py`.
- **Decode M=1 program configs at lines 291-296**: confirmed in `model_config.py`.
- **B=1 state memory sizes** (rec: `12*128*128*2 = 393,216 bytes = 384 KiB`; decode: `384*128*128*2 = 12 MB`): arithmetic verified.
- **TTFT numbers** (498 ms × 96 = 47.8 s; 94 ms × 96 = 9.024 s ≈ 9.1 s; speedup = 5.3x): correct.
- **Layer counts** (48 GDN + 16 full attention = 64): confirmed via `config.json` `layer_types` array (pattern: 3 linear_attention + 1 full_attention × 16 groups).
- **State replication patterns** (`expand` for conv/KV, `repeat` for rec): `gdn.py` lines 540-566 and `attention.py` lines 264-292 confirm.
- **KV cache replacement vs `ttnn.copy`**: `attention.py` replaces references via `ttnn.from_torch`; `gdn.py` uses `ttnn.copy` into pre-existing decode buffers. Both described correctly.
- **Conv shift register copy order** (states[1]→[0], [2]→[1], [3]→[2], qkv→[3]): `gdn.py` lines 653-656 match `gdn_prefill_strategy.md` Step 2 exactly.
- **All index.md links**: resolve to existing files, clickable.
- **Navigation footers**: present and correct in all four files.
- **No plain-text display equations**: no `$$...$$` or bare LaTeX found anywhere.

**No feedback — chapter approved.**
