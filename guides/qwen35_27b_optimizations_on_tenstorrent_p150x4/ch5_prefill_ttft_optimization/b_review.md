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
