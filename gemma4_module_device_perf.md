# Gemma 4 Module Device Performance Results

**Hardware**: T3K (8× Wormhole B0, 1×8 mesh), CHIP_FREQ=1000 MHz  
**Date**: 2026-05-03  
**Commit**: `879f59d` (gemma4 removed reshapes — `gemma4_attention.py`, `linear.py`)  
**Full reports**: `/tmp/gemma4_perf_reports/`  
**Tool**: `tt-perf-report --ignore-signposts <tracy_csv>`

---

## Device Time by Module (Stacked Report Summary)

All device times are `Device_Time_Sum_us` (microseconds) accumulated across **all phases** in the profiling run (compile + 2× warmup + 10× perf iterations for prefill and decode). Percentages are relative to that module's total device time.

---

### `linear` — Column-parallel linear (q/k/v/o projections)
Total device ops: Matmul + ReduceScatter + AllGather

| Op | Device_Time_Sum_us | % |
|----|--------------------|---|
| Matmul (dram_interleaved) | 26,184 | 59.28% |
| ReduceScatter | 11,681 | 26.45% |
| AllGather | 6,302 | 14.27% |

Host latency (50 iters): 1.2 ms/iter (IColShardedWAllReduced), 0.4 ms/iter (IReplicatedWColSharded), 0.8 ms/iter (IColShardedWRowSharded)

---

### `distributed_rmsnorm` — AllGather + LayerNorm across devices
| Op | Device_Time_Sum_us | % |
|----|--------------------|---|
| AllGather | 8,593 | 48.97% |
| LayerNormPostAllGather | 5,849 | 33.33% |
| LayerNormPreAllGather | 3,105 | 17.70% |

Host latency (100 iters): Decode 0.7 ms/iter, Prefill 0.8 ms/iter

---

### `local_rmsnorm` — Local (per-device) RMSNorm
| Op | Device_Time_Sum_us | % |
|----|--------------------|---|
| LayerNorm | 12,425 | 100% |

Host latency (100 iters): ~0.2 ms/iter across all variants

---

### `rope` — Rotary Position Embedding (sliding + global variants)
| Op | Device_Time_Sum_us | % |
|----|--------------------|---|
| RotaryEmbeddingLlama | 39,095 | 78.12% |
| Slice | 4,174 | 8.34% |
| Concat | 3,603 | 7.20% |
| TilizeWithValPadding | 2,334 | 4.66% |
| Permute | 359 | 0.72% |
| Transpose | 263 | 0.53% |
| Embeddings | 146 | 0.29% |
| Typecast | 70 | 0.14% |

Host latency (50 iters): sliding-prefill 0.3 ms/iter, sliding-decode 2.2 ms/iter, global-prefill 2.6 ms/iter

---

### `partial_rope` — Partial RoPE (factor=0.25, global attention variant)
**Status**: FAILED — intermittent Tracy internal desync (`AssertionError: Device data missing: Op not present in cpp_device_perf_report.csv for device 4`). No perf report generated.

Host latency (50 iters from log): FULL_CHUNK_DECODE 4.72 ms/iter, PARTIAL_DECODE 2.91 ms/iter, FULL_CHUNK_PREFILL 2.58 ms/iter

---

### `ffn` — Feed-Forward Network (single matmul path)
| Op | Device_Time_Sum_us | % |
|----|--------------------|---|
| Matmul (dram_interleaved) | 26,032 | 49.02% |
| ReduceScatter | 11,554 | 21.76% |
| AllGather | 10,420 | 19.62% |
| Slice | 2,272 | 4.28% |
| BinaryNg | 1,699 | 3.20% |
| Unary | 1,125 | 2.12% |

Host latency (10 iters): Decode 3.3 ms/iter, Prefill 3.3 ms/iter

---

### `fused_gate_up` — Fused gate×up projection (2× wider matmul)
| Op | Device_Time_Sum_us | % |
|----|--------------------|---|
| Matmul (dram_interleaved) | 52,429 | 48.53% |
| ReduceScatter | 23,418 | 21.68% |
| AllGather | 21,060 | 19.50% |
| Slice | 4,615 | 4.27% |
| BinaryNg | 3,438 | 3.18% |
| Unary | 3,065 | 2.84% |

Note: fused_gate_up runs both gate and up projections in a single wider matmul — device time is ~2× ffn as expected.

Host latency (10 iters): Fused decode 2.90 ms/iter, Fused prefill 3.53 ms/iter; Separate decode 3.77 ms/iter, Separate prefill 3.61 ms/iter

---

### `mlp` — Full MLP (gate_proj × up_proj → activation → down_proj)
| Op | Device_Time_Sum_us | % |
|----|--------------------|---|
| Matmul (dram_interleaved) | 25,974 | 48.99% |
| ReduceScatter | 11,533 | 21.75% |
| AllGather | 10,406 | 19.63% |
| Slice | 2,272 | 4.28% |
| BinaryNg | 1,699 | 3.20% |
| Unary | 1,133 | 2.14% |

Host latency (10 iters): Decode 3.2 ms/iter, Prefill 3.1 ms/iter

---

### `attention_sliding` — Sliding-window attention (32Q/16KV heads, head_dim=256, window=1024)
| Op | Device_Time_Sum_us | % |
|----|--------------------|---|
| Reshape | 29,736 | 44.73% |
| Matmul (dram_interleaved) | 7,725 | 11.62% |
| ReduceScatter | 5,344 | 8.04% |
| AllGather | 4,520 | 6.80% |
| Transpose | 4,027 | 6.06% |
| RotaryEmbedding | 2,282 | 3.43% |
| Slice | 2,267 | 3.41% |
| SDPA | 2,021 | 3.04% |
| LayerNorm | 1,739 | 2.62% |
| Tilize | 1,168 | 1.76% |
| Untilize | 1,166 | 1.75% |
| Concat | 1,072 | 1.61% |
| PagedUpdateCache | 943 | 1.42% |
| Matmul (width_sharded) | 861 | 1.29% |
| TilizeWithValPadding | 595 | 0.89% |
| SdpaDecode | 495 | 0.75% |
| PagedFillCache | 385 | 0.58% |

Host latency (10 iters): Prefill 8.2 ms/iter, Decode 8.5 ms/iter

---

### `attention_global` — Global attention (32Q/4KV heads, head_dim=512, K=V sharing)
| Op | Device_Time_Sum_us | % |
|----|--------------------|---|
| Reshape | 50,688 | 43.57% |
| Matmul (dram_interleaved) | 16,084 | 13.82% |
| ReduceScatter | 9,464 | 8.13% |
| Transpose | 8,067 | 6.93% |
| AllGather | 8,031 | 6.90% |
| RotaryEmbedding | 3,912 | 3.36% |
| SDPA | 3,304 | 2.84% |
| Slice | 3,206 | 2.76% |
| LayerNorm | 3,059 | 2.63% |
| Tilize | 2,134 | 1.83% |
| Untilize | 2,104 | 1.81% |
| Matmul (width_sharded) | 1,706 | 1.47% |
| Concat | 1,374 | 1.18% |
| TilizeWithValPadding | 1,161 | 1.00% |
| PagedFillCache | 700 | 0.60% |
| SdpaDecode | 685 | 0.59% |
| PagedUpdateCache | 464 | 0.40% |

Host latency (10 iters): Prefill 8.4 ms/iter, Decode 8.3 ms/iter

---

### `decoder_layer_sliding` — Full sliding decoder layer (attn + distributed rmsnorm + MLP)
**Status**: FAILED — Tracy CSV processing failed (`cpp_device_perf_report.csv not found`). Test ran successfully; device op breakdown unavailable.

Host latency (10 iters from log): Prefill 17.7 ms/iter, Decode 14.6 ms/iter

---

### `decoder_layer_global` — Full global decoder layer (attn + distributed rmsnorm + MLP)
| Op | Device_Time_Sum_us | % |
|----|--------------------|---|
| Reshape | 51,069 | 28.70% |
| Matmul (dram_interleaved) | 41,402 | 23.27% |
| AllGather | 23,062 | 12.96% |
| ReduceScatter | 20,234 | 11.37% |
| Transpose | 8,129 | 4.57% |
| Slice | 5,478 | 3.08% |
| RotaryEmbedding | 3,920 | 2.20% |
| SDPA | 3,379 | 1.90% |
| LayerNorm | 3,066 | 1.72% |
| LayerNormPostAllGather | 2,990 | 1.68% |
| Tilize | 2,145 | 1.21% |
| Untilize | 2,122 | 1.19% |
| BinaryNg | 1,918 | 1.08% |
| Matmul (width_sharded) | 1,703 | 0.96% |
| LayerNormPreAllGather | 1,618 | 0.91% |
| Concat | 1,356 | 0.76% |
| TilizeWithValPadding | 1,161 | 0.65% |
| Unary | 1,123 | 0.63% |
| PagedFillCache | 706 | 0.40% |
| SdpaDecode | 685 | 0.39% |
| PagedUpdateCache | 464 | 0.26% |

Host latency (10 iters): Prefill 20.6 ms/iter, Decode 15.1 ms/iter

---

### `decoder_layer_v2` — Decoder layer v2 (sliding variant)
| Op | Device_Time_Sum_us | % |
|----|--------------------|---|
| Matmul (dram_interleaved) | 33,026 | 25.80% |
| Reshape | 30,188 | 23.58% |
| AllGather | 18,941 | 14.79% |
| ReduceScatter | 16,723 | 13.06% |
| Slice | 4,530 | 3.54% |
| Transpose | 4,014 | 3.14% |
| LayerNormPostAllGather | 2,983 | 2.33% |
| RotaryEmbedding | 2,298 | 1.79% |
| SDPA | 2,090 | 1.63% |
| BinaryNg | 1,926 | 1.50% |
| LayerNorm | 1,741 | 1.36% |
| LayerNormPreAllGather | 1,607 | 1.25% |
| Untilize | 1,170 | 0.91% |
| Tilize | 1,164 | 0.91% |
| Unary | 1,132 | 0.88% |
| Concat | 1,073 | 0.84% |
| PagedUpdateCache | 945 | 0.74% |
| Matmul (width_sharded) | 861 | 0.67% |
| TilizeWithValPadding | 595 | 0.46% |
| SdpaDecode | 498 | 0.39% |
| PagedFillCache | 387 | 0.30% |

Host latency (10 iters): Prefill 16.8 ms/iter, Decode 15.8 ms/iter

---

### `embedding` — Token embedding lookup
**Status**: Tracy CSV empty (no device ops captured). Test passed but profiler produced no data (embedding op likely too fast / runs entirely on host path for this shape).

---

## Key Observations

1. **Reshape still dominates attention modules but is reduced**: Removing reshapes in `gemma4_attention.py` and `linear.py` (commit `879f59d`) cut Reshape device time by ~10–12%: `attention_sliding` 33,727 → 29,736 us (47.9% → 44.7%), `attention_global` 57,141 → 50,688 us (46.9% → 43.6%). Reshape is still the #1 op in all attention modules — remaining reshape overhead is the next optimization target.

2. **Reshape removal significantly improves host latency in attention**: `attention_sliding` -16%/-21% (9.8/10.7 → 8.2/8.5 ms), `attention_global` -21%/-19% (10.6/10.2 → 8.4/8.3 ms). Decoder layer sliding host latency also improved (20.0/15.7 → 17.7/14.6 ms from logs).

3. **In decoder_layer_v2, Matmul overtook Reshape as the top op** (25.80% vs 23.58%), indicating the attention path has improved sufficiently that compute is now the leading cost in the sliding decoder variant.

4. **Collective communication (AllGather + ReduceScatter) is 40-50% of linear/MLP device time**. On T3K with 8 devices, tensor-parallel communication cost is nearly equal to compute. Unchanged from previous run.

5. **SDPA is only 2-3% of attention device time** — flash attention is well-optimized; the bottleneck remains data movement (Reshape, collectives).

6. **New ops now visible in attention/decoder reports**: `TilizeWithValPadding`, `SdpaDecode`, `PagedFillCache` appear explicitly now that the dominant Reshape overhead is reduced. Combined, they account for ~1–2% of attention device time.

7. **Decoder layer = attention + rmsnorm + MLP**: The op mix confirms the full layer is a straight composition — no unexpected ops or overhead.

8. **Global vs sliding attention**: Global attention device time is ~1.7× sliding (51ms vs 30ms Reshape, 16ms vs 7.7ms Matmul), driven by larger head_dim (512 vs 256) and lower KV head count (4 vs 16). Ratio unchanged.

---

## Reproduction Steps

### Prerequisites
```bash
cd /home/ttuser/salnahari/tt-metal
source /home/ttuser/salnahari/tt_bashrc  # sets up TT_METAL_HOME, PYTHONPATH, etc.
```

### Run All Modules (Automated)

```bash
MESH_DEVICE=T3K bash run_gemma4_perf_reports.sh
```

Results land in `/tmp/gemma4_perf_reports/`:
- `<module>.log` — full pytest + Tracy output
- `<module>_perf_report.txt` — tt-perf-report output

### Run a Single Module

General pattern:
```bash
TT_METAL_DEVICE_PROFILER=1 TT_SYMBIOTE_TRACY=1 MESH_DEVICE=T3K \
    python3 -m tracy -v -r -p -m pytest \
    models/experimental/tt_symbiote/tests/<test_file>.py -s -v
```

Then find the newest CSV and generate the report:
```bash
NEWEST=$(ls -t generated/profiler/reports/*/ops_perf_results_*.csv | head -1)
tt-perf-report --ignore-signposts "$NEWEST"
```

### Per-Module Commands

| Module | Test file |
|--------|-----------|
| embedding | `test_gemma4_profile_embedding.py` |
| linear | `test_gemma4_profile_linear.py` |
| distributed_rmsnorm | `test_gemma4_profile_distributed_rmsnorm.py` |
| local_rmsnorm | `test_gemma4_profile_local_rmsnorm.py` |
| rope | `test_gemma4_profile_rope.py` |
| partial_rope | `test_gemma4_profile_partial_rope.py` |
| ffn | `test_gemma4_profile_ffn.py` |
| fused_gate_up | `test_gemma4_profile_fused_gate_up.py` |
| mlp | `test_gemma4_profile_mlp.py` |
| attention_sliding | `test_gemma4_profile_attention_sliding.py` |
| attention_global | `test_gemma4_profile_attention_global.py` |
| decoder_layer_sliding | `test_gemma4_profile_decoder_layer_sliding.py` |
| decoder_layer_global | `test_gemma4_profile_decoder_layer_global.py` |
| decoder_layer_v2 | `test_gemma4_profile_decoder_layer_v2.py` |

All test files: `models/experimental/tt_symbiote/tests/`

### Retry partial_rope / decoder_layer_sliding

Both fail intermittently due to Tracy internals. Retry with:
```bash
tt-smi -r && sleep 2
TT_METAL_DEVICE_PROFILER=1 TT_SYMBIOTE_TRACY=1 MESH_DEVICE=T3K \
    python3 -m tracy -v -r -p -m pytest \
    models/experimental/tt_symbiote/tests/test_gemma4_profile_<module>.py -s -v
NEWEST=$(ls -t generated/profiler/reports/*/ops_perf_results_*.csv | head -1)
tt-perf-report --ignore-signposts "$NEWEST"
```

### Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `TT_METAL_DEVICE_PROFILER` | `1` | Enable device-side op timing capture |
| `TT_SYMBIOTE_TRACY` | `1` | Enable signpost calls in tests |
| `MESH_DEVICE` | `T3K` | Select 8-device mesh (1×8 layout) |
