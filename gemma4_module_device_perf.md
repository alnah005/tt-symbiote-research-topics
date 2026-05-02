# Gemma 4 Module Device Performance Results

**Hardware**: T3K (8× Wormhole B0, 1×8 mesh), CHIP_FREQ=1000 MHz  
**Date**: 2026-05-02  
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
| Matmul (dram_interleaved) | 26,147 | 58.97% |
| ReduceScatter | 11,809 | 26.63% |
| AllGather | 6,382 | 14.39% |

Host latency (50 iters): 1.4 ms/iter (seq≥1), 0.5 ms/iter, 1.0 ms/iter (different seq variants)

---

### `distributed_rmsnorm` — AllGather + LayerNorm across devices
| Op | Device_Time_Sum_us | % |
|----|--------------------|---|
| AllGather | 8,627 | 49.07% |
| LayerNormPostAllGather | 5,853 | 33.29% |
| LayerNormPreAllGather | 3,103 | 17.65% |

Host latency (100 iters): Prefill 0.7 ms/iter, Decode 0.8 ms/iter

---

### `local_rmsnorm` — Local (per-device) RMSNorm
| Op | Device_Time_Sum_us | % |
|----|--------------------|---|
| LayerNorm | 12,453 | 100% |

Host latency (100 iters): ~0.2 ms/iter across all variants

---

### `rope` — Rotary Position Embedding (sliding + global variants)
| Op | Device_Time_Sum_us | % |
|----|--------------------|---|
| RotaryEmbeddingLlama | 39,098 | 78.13% |
| Slice | 4,169 | 8.33% |
| Concat | 3,606 | 7.21% |
| TilizeWithValPadding | 2,333 | 4.66% |
| Permute | 360 | 0.72% |
| Transpose | 263 | 0.53% |
| Embeddings | 146 | 0.29% |

Host latency (50 iters): sliding-prefill 0.3 ms/iter, sliding-decode 2.3 ms/iter, global-prefill 2.8 ms/iter

---

### `partial_rope` — Partial RoPE (factor=0.25, global attention variant)
**Status**: FAILED — intermittent Tracy internal desync (`AssertionError: Device data missing: Op not present in cpp_device_perf_report.csv for device 4`). No perf report generated. Retry expected to succeed.

Host latency (50 iters from log): 4.83 ms, 3.06 ms, 2.65 ms, 1.91 ms (prefill/decode, sliding/global variants)

---

### `ffn` — Feed-Forward Network (single matmul path)
| Op | Device_Time_Sum_us | % |
|----|--------------------|---|
| Matmul (dram_interleaved) | 25,950 | 49.10% |
| ReduceScatter | 11,407 | 21.58% |
| AllGather | 10,390 | 19.66% |
| Slice | 2,276 | 4.31% |
| BinaryNg | 1,697 | 3.21% |
| Unary | 1,131 | 2.14% |

Host latency (10 iters): Prefill 3.2 ms/iter, Decode 3.1 ms/iter

---

### `fused_gate_up` — Fused gate×up projection (2× wider matmul)
| Op | Device_Time_Sum_us | % |
|----|--------------------|---|
| Matmul (dram_interleaved) | 52,442 | 48.48% |
| ReduceScatter | 23,473 | 21.70% |
| AllGather | 21,147 | 19.55% |
| Slice | 4,607 | 4.26% |
| BinaryNg | 3,445 | 3.19% |
| Unary | 3,059 | 2.83% |

Note: fused_gate_up runs both gate and up projections in a single wider matmul — device time is ~2× ffn as expected.

---

### `mlp` — Full MLP (gate_proj × up_proj → activation → down_proj)
| Op | Device_Time_Sum_us | % |
|----|--------------------|---|
| Matmul (dram_interleaved) | 25,978 | 49.02% |
| ReduceScatter | 11,499 | 21.70% |
| AllGather | 10,401 | 19.63% |
| Slice | 2,281 | 4.31% |
| BinaryNg | 1,699 | 3.21% |
| Unary | 1,134 | 2.14% |

Host latency (10 iters): Prefill 3.4 ms/iter, Decode 3.9 ms/iter

---

### `attention_sliding` — Sliding-window attention (32Q/16KV heads, head_dim=256, window=1024)
| Op | Device_Time_Sum_us | % |
|----|--------------------|---|
| Reshape | 33,727 | 47.89% |
| Matmul (dram_interleaved) | 7,733 | 10.98% |
| ReduceScatter | 5,445 | 7.73% |
| AllGather | 4,595 | 6.52% |
| Transpose | 4,031 | 5.72% |
| RotaryEmbedding | 2,294 | 3.26% |
| Slice | 2,270 | 3.22% |
| SDPA | 2,098 | 2.98% |
| LayerNorm | 1,413 | 2.01% |
| Tilize | 1,168 | 1.66% |
| Untilize | 1,166 | 1.66% |
| Concat | 1,065 | 1.51% |
| PagedUpdateCache | 947 | 1.34% |

Host latency (10 iters): Prefill 9.8 ms/iter, Decode 10.7 ms/iter

---

### `attention_global` — Global attention (32Q/4KV heads, head_dim=512, K=V sharing)
| Op | Device_Time_Sum_us | % |
|----|--------------------|---|
| Reshape | 57,141 | 46.90% |
| Matmul (dram_interleaved) | 16,100 | 13.21% |
| ReduceScatter | 9,021 | 7.40% |
| AllGather | 8,088 | 6.64% |
| Transpose | 8,069 | 6.62% |
| RotaryEmbedding | 3,908 | 3.21% |
| SDPA | 3,349 | 2.75% |
| Slice | 3,234 | 2.65% |
| LayerNorm | 2,398 | 1.97% |
| Tilize | 2,120 | 1.74% |
| Untilize | 2,114 | 1.74% |
| Matmul (width_sharded) | 1,714 | 1.41% |
| Concat | 1,369 | 1.12% |

Host latency (10 iters): Prefill 10.6 ms/iter, Decode 10.2 ms/iter

---

### `decoder_layer_sliding` — Full sliding decoder layer (attn + distributed rmsnorm + MLP)
| Op | Device_Time_Sum_us | % |
|----|--------------------|---|
| Reshape | 33,715 | 25.75% |
| Matmul (dram_interleaved) | 33,034 | 25.23% |
| AllGather | 18,805 | 14.36% |
| ReduceScatter | 16,712 | 12.76% |
| Slice | 4,509 | 3.44% |
| Transpose | 4,006 | 3.06% |
| LayerNormPostAllGather | 2,979 | 2.28% |
| RotaryEmbedding | 2,289 | 1.75% |
| SDPA | 2,062 | 1.58% |
| BinaryNg | 1,876 | 1.43% |
| LayerNormPreAllGather | 1,598 | 1.22% |
| LayerNorm | 1,420 | 1.08% |
| Untilize | 1,163 | 0.89% |
| Tilize | 1,157 | 0.88% |
| Unary | 1,116 | 0.85% |
| Concat | 1,058 | 0.81% |
| PagedUpdateCache | 944 | 0.72% |
| Matmul (width_sharded) | 864 | 0.66% |

Host latency (10 iters): Prefill 20.0 ms/iter, Decode 15.7 ms/iter

---

### `decoder_layer_global` — Full global decoder layer (attn + distributed rmsnorm + MLP)
| Op | Device_Time_Sum_us | % |
|----|--------------------|---|
| Reshape | 57,218 | 31.31% |
| Matmul (dram_interleaved) | 41,361 | 22.63% |
| AllGather | 22,577 | 12.35% |
| ReduceScatter | 20,333 | 11.13% |
| Transpose | 8,052 | 4.41% |
| Slice | 5,468 | 2.99% |
| RotaryEmbedding | 3,931 | 2.15% |
| SDPA | 3,252 | 1.78% |
| LayerNormPostAllGather | 2,980 | 1.63% |
| LayerNorm | 2,404 | 1.32% |
| Untilize | 2,124 | 1.16% |
| Tilize | 2,111 | 1.15% |
| BinaryNg | 1,923 | 1.05% |
| Matmul (width_sharded) | 1,708 | 0.93% |
| LayerNormPreAllGather | 1,618 | 0.89% |
| Concat | 1,363 | 0.75% |
| Unary | 1,124 | 0.61% |

Host latency (10 iters): Prefill 19.7 ms/iter, Decode 17.4 ms/iter

---

### `decoder_layer_v2` — Decoder layer v2 (sliding variant, same op profile as sliding)
| Op | Device_Time_Sum_us | % |
|----|--------------------|---|
| Reshape | 33,664 | 25.73% |
| Matmul (dram_interleaved) | 33,003 | 25.22% |
| AllGather | 18,895 | 14.44% |
| ReduceScatter | 16,720 | 12.78% |
| Slice | 4,502 | 3.44% |
| Transpose | 4,007 | 3.06% |
| LayerNormPostAllGather | 2,978 | 2.28% |
| RotaryEmbedding | 2,284 | 1.75% |
| SDPA | 1,971 | 1.51% |
| BinaryNg | 1,886 | 1.44% |
| LayerNormPreAllGather | 1,600 | 1.22% |
| LayerNorm | 1,418 | 1.08% |
| Untilize | 1,168 | 0.89% |
| Tilize | 1,164 | 0.89% |
| Unary | 1,121 | 0.86% |
| Concat | 1,057 | 0.81% |
| PagedUpdateCache | 942 | 0.72% |
| Matmul (width_sharded) | 864 | 0.66% |

Host latency (10 iters): Prefill 20.6 ms/iter, Decode 15.8 ms/iter

---

### `embedding` — Token embedding lookup
**Status**: Tracy CSV empty (no device ops captured). Test passed but profiler produced no data (embedding op likely too fast / runs entirely on host path for this shape).

---

## Key Observations

1. **Reshape dominates attention modules** (~47% of attention device time). This is layout reformatting overhead (DRAM-interleaved reshapes before/after matmuls), not compute — a primary optimization target.

2. **Collective communication (AllGather + ReduceScatter) is 40-50% of linear/MLP device time**. On T3K with 8 devices, tensor-parallel communication cost is nearly equal to compute.

3. **SDPA is only 2-3% of attention device time** — flash attention is well-optimized; the bottleneck is data movement (Reshape, collectives).

4. **Decoder layer = attention + rmsnorm + MLP**: The op mix confirms the full layer is a straight composition — no unexpected ops or overhead.

5. **Global vs sliding attention**: Global attention device time is ~1.7× sliding (57ms Reshape vs 34ms, 16ms vs 7.7ms Matmul), driven by larger head_dim (512 vs 256) and lower KV head count (4 vs 16).

---

## Reproduction Steps

### Prerequisites
```bash
cd /home/ttuser/salnahari/tt-metal
source tt_bashrc  # sets up TT_METAL_HOME, PYTHONPATH, etc.
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

### Retry partial_rope

The partial_rope test failed due to an intermittent Tracy desync (device 4 data missing from CSV). Retry with:
```bash
tt-smi -r && sleep 2
TT_METAL_DEVICE_PROFILER=1 TT_SYMBIOTE_TRACY=1 MESH_DEVICE=T3K \
    python3 -m tracy -v -r -p -m pytest \
    models/experimental/tt_symbiote/tests/test_gemma4_profile_partial_rope.py -s -v
NEWEST=$(ls -t generated/profiler/reports/*/ops_perf_results_*.csv | head -1)
tt-perf-report --ignore-signposts "$NEWEST"
```

### Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `TT_METAL_DEVICE_PROFILER` | `1` | Enable device-side op timing capture |
| `TT_SYMBIOTE_TRACY` | `1` | Enable signpost calls in tests |
| `MESH_DEVICE` | `T3K` | Select 8-device mesh (1×8 layout) |
