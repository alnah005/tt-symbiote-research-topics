# Chapter 2 Review -- Correctness

## Issue 1: Prefill K/V reshape omits the NKV==1 special case

**File:** `flash_attention_prefill.md`, Section "2. Reshape to Head Format"

The guide shows K and V being reshaped to `[1, NKV, seq_len, HD]` unconditionally:

```python
k = ttnn.to_memory_config(ttnn.reshape(kp_tt, (1, NKV, seq_len, HD)), ttnn.DRAM_MEMORY_CONFIG)
v = ttnn.to_memory_config(ttnn.reshape(vp_tt, (1, NKV, seq_len, HD)), ttnn.DRAM_MEMORY_CONFIG)
```

In the actual source (`attention.py` lines 364-373), when `NKV == 1` the code takes a different path: it calls `ttnn.clone(kp_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)` and skips the reshape entirely. The reshape path only executes in the `else` branch (NKV > 1). Since TP=4 with 4 KV heads gives `NKV = 1`, the reshape path shown in the guide is never taken on P150x4. The guide should show the clone path and note the conditional.

## Issue 2: SDPA chunk sizes described as "tiles"

**File:** `flash_attention_prefill.md`, bullet under chunk size selection

The text states: "Uses `q_chunk = k_chunk = 256` tiles for maximum throughput." The values 256 and 64 are sequence-position chunk sizes passed to `q_chunk_size` / `k_chunk_size` in `SDPAProgramConfig`, not tile counts. A tile is 32 elements, so 256 positions would be 8 tiles. The word "tiles" should be removed or replaced with an accurate unit (e.g., "positions" or just drop the unit since the numbers speak for themselves).

## Issue 3: KV update shard config attribute name mismatch

**File:** `dram_sharded_decode.md`, "Internal Per-Head KV Cache" code snippet

The guide shows the shard config created as `self.kv_update_shard_cfg` (no leading underscore). In the source (`attention.py` line 88), the attribute is stored as `self._kv_update_shard_cfg` (with leading underscore, indicating a private field). This is a minor naming inconsistency but could confuse a reader cross-referencing the guide with the code.

---

## Pass 2 (post-correction)

All three issues from Pass 1 have been verified as corrected:

1. **Issue 1 (NKV==1 special case)**: `flash_attention_prefill.md` now shows the `ttnn.clone` path for NKV==1 and notes the conditional with the reshape-based else branch.
2. **Issue 2 (chunk size units)**: `flash_attention_prefill.md` now describes chunk sizes as "sequence positions" rather than "tiles."
3. **Issue 3 (attribute name)**: `dram_sharded_decode.md` now uses `self._kv_update_shard_cfg` with the leading underscore, matching the source.

No new factual errors found. All numerical values (NH=6, NKV=1, HD=256, ROPE_DIM=64, dim=5120, n_heads=24, n_kv_heads=4, TP=4), code snippets, tensor shapes, config field names, and architectural descriptions were cross-checked against `attention.py`, `rope.py`, `model_config.py`, and `config.json`.

No feedback -- chapter approved.
