# Model Variants

Qwen3.5 is a family of hybrid transformer models combining GatedDeltaNet (linear attention) layers
with full-attention layers. The codebase implements two variants that run on a single P100A
Blackhole card: the 27B dense model and the 35B-A3B Mixture-of-Experts model. Two larger variants
(122B-A10B and 397B-A17B) are architecturally documented but require multi-chip Galaxy systems.

---

## All Four Variants at a Glance

| | **27B** | **35B-A3B** | 122B-A10B | 397B-A17B |
|---|---|---|---|---|
| Total layers | 64 (48 + 16) | 40 (30 + 10) | 48 (36 + 12) | 60 (45 + 15) |
| Hidden size | 5120 | 2048 | 3072 | 4096 |
| DeltaNet V-heads | 48 | 32 | 64 | 64 |
| Attn Q / KV heads | 24 / 4 | 16 / 2 | 32 / 2 | 32 / 2 |
| MLP type | Dense SwiGLU | MoE 256 top-8 | MoE 256 top-8 | MoE 512 top-10 |
| Active parameters | 27B | 3B | 10B | 17B |
| DRAM footprint (bfp4) | ~13 GB | ~17.5 GB | ~61 GB | ~199 GB |
| Target hardware | P100A | **P100A** | Galaxy 6U | Galaxy 6U+ |

The DRAM footprint numbers are approximate and depend on precision choices. The actual measured
DRAM usage at the precisions used in this implementation (bfp8 attention / DeltaNet projections,
bfp4 MoE expert weights, bf16 attention QKV+WO, bf16 KV cache) is:

- **35B-A3B:** ~15.7 GB of 28 GB available (see PERF.md for a per-component breakdown)
- **27B:** ~25 GB of 28 GB available

---

## Hybrid Layer Ratio: 3/4 DeltaNet + 1/4 Full Attention

Every Qwen3.5 variant uses a fixed 3:1 ratio of DeltaNet layers to full-attention layers.

The ordering of layer types within a model is specified by the `layer_types` list in the HF
`config.json`. The model config loader reads this directly:

```python
self.layer_types = text_config.get("layer_types", None)
```

Entries in this list are either `"linear_attention"` (DeltaNet) or `"full_attention"`
(GatedAttention). The forward loop in both demos uses this list to dispatch:

```python
# From demo.py (27B) — per-layer dispatch
if args.layer_types[i] == "linear_attention":
    layers.append(DeltaNetDecoderBlock(...))
else:
    layers.append(TransformerBlock(..., attention_class=GatedAttention))

# From demo_a3b.py (35B-A3B) — uniform DeltaNetDecoderBlock with attention_class switch
if args.layer_types[i] == "linear_attention":
    attention_class = None        # → GatedDeltaNet
else:
    attention_class = GatedAttention
layers.append(DeltaNetDecoderBlock(..., attention_class=attention_class, mlp_class=Qwen35MoE))
```

---

## Recommended Entry Point: 35B-A3B

The 35B-A3B is the recommended variant for two reasons:

**Speed:** Despite having 35B total parameters, only 3B active parameters are used per
token (via the MoE top-8 routing). This means the per-token DRAM reads are much smaller
than for the dense 27B model, even though A3B has more total weight on device.

| Model | tok/s on P100A | Method |
|-------|---------------|--------|
| Qwen3.5-35B-A3B | **11.7** | Host recurrence + device MoE |
| Qwen3.5-27B | **6.28** | Host recurrence + device MLP |

**CPU baseline comparison:** The AmpereOne 128-core CPU running llama.cpp with Q4_K
quantization achieves 9.05 tok/s on A3B. The TTNN implementation at 11.7 tok/s exceeds
this CPU baseline on a single $999 Blackhole card.

**Hardware fit:** Both variants fit within the 28 GB DRAM of a single P100A card, but
A3B leaves significantly more headroom (15.7 GB used vs 25 GB for 27B).

---

## Per-Component DRAM Breakdown

### 35B-A3B (MoE)

| Component | Dtype | Size |
|-----------|-------|------|
| Expert weights (256 × gate+up+down, 40 layers) | bfp4 | 12.8 GB |
| Shared expert weights (40 layers) | bfp8 | 0.8 GB |
| DeltaNet projections (30 layers) | bfp8 | 1.2 GB |
| Attention QKV + WO + gate (10 layers) | bf16 | 0.5 GB |
| Router + shared gate weights | bf16 | 0.1 GB |
| KV cache (10 layers) | bf16 | 0.3 GB |
| **Total** | | **~15.7 GB / 28 GB** |

### 27B (Dense)

| Component | Dtype | Size |
|-----------|-------|------|
| DeltaNet projections (48 layers) | bfp8 | 5.4 GB |
| MLP w1 + w2 + w3 (64 layers) | bfp8 | 17.1 GB |
| Attention QKV + WO + gate (16 layers) | bf16 | 2.2 GB |
| Other | bf16 | ~0.3 GB |
| **Total** | | **~25 GB / 28 GB** |

---

## Performance Profiling (A3B, 86 ms/token)

The A3B per-token time breaks down as follows at 11.7 tok/s (86 ms/token):

| Component | Time | Host-Device Syncs |
|-----------|------|--------------------|
| DeltaNet (30 layers) | 54 ms | 30 (1 per layer) |
| Attention (10 layers) | 18 ms | 10 + 40 |
| norm + LM head | 14 ms | 1 |
| **Total** | **86 ms** | **~70** |

Time budget breakdown: ~35 ms sync overhead + ~26 ms Python dispatch + ~20 ms device
compute. The theoretical device-compute limit is ~5.8 ms/token (172 tok/s); current
efficiency is 6.3%. The dominant bottleneck is the DeltaNet host recurrence, which
requires one `to_torch` + `from_torch` round-trip per layer per token (30 syncs for
30 DeltaNet layers). Metal Trace is the primary optimization path to eliminate the
Python dispatch overhead.

---

**Next:** [`layer_types_and_hyperparams.md`](./layer_types_and_hyperparams.md)
