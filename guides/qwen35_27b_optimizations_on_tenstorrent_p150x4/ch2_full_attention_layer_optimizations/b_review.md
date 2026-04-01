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

## Pass 3

Independent cold review. All source files read: `attention.py`, `rope.py`, `model_config.py`. All guide files read: `index.md`, `attention_architecture.md`, `dram_sharded_decode.md`, `flash_attention_prefill.md`.

**Structural checks:**
- All three content files present and reachable from `index.md` via clickable relative links.
- Navigation footers present on all content files: `attention_architecture.md` → `dram_sharded_decode.md` → `flash_attention_prefill.md` → `../ch3_gdn_layer_decode_pipeline/index.md`.
- `index.md` footer `**Next:** [`attention_architecture.md`](./attention_architecture.md)` is present and clickable.
- Display equation in `attention_architecture.md` uses `$$...$$` LaTeX delimiters (lines 118–119). No plain-text display equations found.

**Factual verification (cross-checked against source):**
- `ROPE_DIM = 64` at `model_config.py:40`: confirmed.
- `_cos_table` shape `[1, max_seq_len, 64]` from `emb.cos().unsqueeze(0)`: confirmed (`rope.py:51-53`).
- `get_rot_mats()` return shape `[1, B, 1, ROPE_DIM]` via `unsqueeze_to_4D` + `transpose(1,2)`: confirmed (`rope.py:76-77`).
- `get_prefill_rot_mats()` output `[1, 1, seq_len, 64]` via slice + reshape: confirmed (`rope.py:89-92`).
- RMSNorm formula `x / sqrt(mean(x^2) + eps)` matches `_rms_norm_dev` calling `ttnn.rms_norm(x, epsilon=1e-6)`: confirmed (`attention.py:44-46`).
- Norm before RoPE ordering in `forward_decode`: confirmed (`attention.py:167-174`).
- Q+gate decode reshape `[1, B, NH, HD*2]`, slice at `HD` boundary: confirmed (`attention.py:156-159`).
- NH=6, NKV=1, HD=256, dim=5120: internally consistent with `n_local_heads`, `n_local_kv_heads`, `head_dim` usage throughout `attention.py`.
- `DRAM_CORES=8`, N padded to multiple of `TILE_SIZE * DRAM_CORES = 256`: confirmed (`model_config.py:46-49, 82`).
- `per_core_N = n_tiles // num_cores if n_tiles >= num_cores else 1`: confirmed (`model_config.py:111`).
- Four decode program configs instantiated with `M=1`: confirmed (`model_config.py:290-296`).
- `COMPUTE_HIFI2` with `math_fidelity=HiFi2`, `math_approx_mode=True`, `fp32_dest_acc_en=True`, `packer_l1_acc=True`: confirmed (`model_config.py:177-182`).
- `self.compute_cfg = args.compute_kernel_config_hifi2`: confirmed (`attention.py:82`).
- Prefill NKV==1 path uses `ttnn.clone`, not reshape: confirmed (`attention.py:364-366`).
- `kv_update_shard_cfg` HEIGHT_SHARDED shape `(32, 256)` on 8x4 grid: confirmed (`model_config.py:313-319`).
- `attn_out_dim_tp = (n_heads * head_dim) // tp = 1536` per device: confirmed (`model_config.py:270`).
- `per_core_M=8, per_core_N=12` for 2048-token Q+gate prefill: confirmed (`model_config.py:152-153`).
- Chunk sizes 256/64 are sequence positions, not tiles: confirmed (`attention.py:442-443`).
- `_all_reduce` uses `cluster_axis=0`, `dim=3`: confirmed (`attention.py:294-300`).
- Prefill SDPA uses `scaled_dot_product_attention` with `is_causal=True`: confirmed (`attention.py:451-457`).
- Prefill output projection result all-reduced via `_all_reduce(wo_out)`: confirmed (`attention.py:484`).

**No issues found.** All numerical values, tensor shapes, code snippets, line references, and architectural descriptions verified against source.

**No feedback — chapter approved.**

---

## Pass 2

Independent cold review. All source files read: `attention.py`, `rope.py`, `model_config.py`. All guide files read: `index.md`, `attention_architecture.md`, `dram_sharded_decode.md`, `flash_attention_prefill.md`.

**Structural checks:**
- All four planned files present and linked.
- Navigation footers present on all three content files (`attention_architecture.md` → `dram_sharded_decode.md` → `flash_attention_prefill.md` → ch3 index).
- `index.md` file references are all clickable markdown links.
- Display equation in `attention_architecture.md` uses `$$...$$` (line 118-119).

**Factual verification (cross-checked against source):**
- NH=6 (24/4), NKV=1 (4/4), HD=256, ROPE_DIM=64, dim=5120: confirmed via model_config.py:40 and source constants.
- `_cos_table` shape `[1, max_seq_len, 64]`: confirmed (rope.py:51-53, `emb.cos().unsqueeze(0)`).
- `get_rot_mats()` return shape `[1, B, 1, ROPE_DIM]`: confirmed (rope.py:76-77, unsqueeze_to_4D then transpose(1,2)).
- Q+gate decode reshape and slice at HD boundary: confirmed (attention.py:156-159).
- Decode `_rms_norm_dev` → RoPE ordering: confirmed (attention.py:167-174, norm before RoPE).
- `_kv_update_shard_cfg` HEIGHT_SHARDED (32,256) on 8x4 grid: confirmed (model_config.py:313-319, attention.py:88).
- `kv_update_shard_cfg` attribute in model_config snippet (no leading underscore) and `self._kv_update_shard_cfg` in attention.py (with underscore): both correct in their respective contexts.
- `per_core_M=8, per_core_N=12` for 2048-token Q+gate prefill: confirmed (model_config.py:152-153).
- Prefill NKV==1 path uses `ttnn.clone`, not reshape: confirmed (attention.py:364-366), correctly shown in `flash_attention_prefill.md`.
- Chunk sizes 256/64 are sequence positions, not tiles: confirmed (attention.py:442-443), correctly described.
- `_all_reduce` uses `cluster_axis=0`, `dim=3`: confirmed (attention.py:294-300).
- `attn_out_dim_tp = NH*HD = 1536` per device: confirmed (model_config.py:270, `n_heads*head_dim//tp = 24*256//4`).
- SDPA decode uses `scaled_dot_product_attention_decode`; prefill uses `scaled_dot_product_attention` with `is_causal=True`: confirmed (attention.py:224, 451-457).
- `DRAM_CORES=8`, N padded to multiple of 256 (`TILE_SIZE * DRAM_CORES`): confirmed (model_config.py:45-49, 82).

**No issues found.** All numerical values, code snippets, tensor shapes, and architectural descriptions are accurate.

**No feedback — chapter approved.**

---

## Pass 1

Reviewed by Agent B (independent cold reviewer). Source files cross-checked: `attention.py`, `rope.py`, `model_config.py`, `config.json`.

**Structural gaps:** All four planned files present. Navigation footers correct on all content files. Index file links are all clickable. Display equation in `attention_architecture.md` uses `$$...$$`.

**No material errors found.** All numerical values confirmed against source:

- 64 total layers, 16 full-attention at every 4th position (`config.json`: `num_hidden_layers=64`, `full_attention_interval=4`)
- NH=6 (24 heads / TP=4), NKV=1 (4 KV heads / TP=4), HD=256, ROPE_DIM=64, dim=5120 (`config.json` + `model_config.py:40`)
- `rope_theta=10_000_000.0` matches `config.json` (`rope_theta: 10000000`)
- `_cos_table` shape `[1, max_seq_len, 64]` confirmed (`rope.py:52`: `emb.cos().unsqueeze(0)`)
- `get_rot_mats()` return shape `[1, B, 1, ROPE_DIM]` confirmed (`rope.py:76-77`: `unsqueeze_to_4D` then `transpose(1,2)`)
- `get_prefill_rot_mats()` output `[1, 1, seq_len, 64]` confirmed (`rope.py:89-92`)
- RMSNorm formula in `attention_architecture.md` matches `_rms_norm_dev` implementation (`attention.py:44-46`)
- Q+gate decode reshape to `[1, B, NH, HD*2]`, slice at `HD` boundary confirmed (`attention.py:156-159`)
- `attn_out_dim_tp = (n_heads * head_dim) // tp = 1536` matches reshape to `(1, B, NH*HD)` in `forward_decode` (`attention.py:241`, `model_config.py:270`)
- `DRAM_CORES=8` grid and `padded_n` multiple-of-256 claim confirmed (`model_config.py:46-48, 82`)
- `kv_update_shard_cfg` shape `(32, 256)` on 8x4 grid confirmed (`model_config.py:313-319`)
- `per_core_M=ceil(2048/32/8)=8`, `per_core_N=ceil(3072/32/8)=12` derivation confirmed (`model_config.py:152-153`)
- Decode SDPA uses `scaled_dot_product_attention_decode` (non-paged path, `attention.py:224`); prefill uses `scaled_dot_product_attention` with `is_causal=True` (`attention.py:451-457`)
- Chunk size formula `min(256 if seq_len>=2048 else 64, padded_seq)` confirmed (`attention.py:442-443`)
