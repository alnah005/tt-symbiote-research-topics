# KV Cache Sharding and Memory Budget

## Sliding Layer KV Cache (50 Layers)

### Per-Device Layout

With TP=8, each device holds 2 of the 16 sliding KV heads. The sliding
window is 1,024 tokens, so the KV cache for each sliding layer is bounded
regardless of the total sequence length.

| Parameter | Value |
|-----------|-------|
| KV heads per device | 2 |
| head_dim | 256 |
| Window size | 1,024 tokens |
| Tensors | 2 (K + V) |

Per-device KV cache shape per sliding layer:

```text
K cache: [B, 2, 1024, 256]
V cache: [B, 2, 1024, 256]
```

### Memory Per Layer Per Device

```math
\text{Sliding KV per layer per device} = 2 \times B \times 2 \times 1024 \times 256 \times \text{bytes\_per\_element}
```

| Dtype | Bytes/Element | Per Layer Per Device (B=1) |
|-------|--------------|---------------------------|
| BF16 | 2 | 2 x 1 x 2 x 1,024 x 256 x 2 = 2.0 MB |
| BFP8 | 1 | 2 x 1 x 2 x 1,024 x 256 x 1 = 1.0 MB |

### Total Sliding KV Across 50 Layers

| Dtype | Per Device (B=1) |
|-------|-----------------|
| BF16 | 50 x 2.0 MB = **100.0 MB** |
| BFP8 | 50 x 1.0 MB = **50.0 MB** |

The sliding KV cache is **constant** with respect to sequence length because
the window caps storage at 1,024 tokens. This is a significant advantage of
the sliding-window design --- 50 of 60 layers contribute a fixed, predictable
memory footprint.

## Global Layer KV Cache (10 Layers)

### Per-Device Layout

Under the recommended replication strategy (see
[`sharding_strategy_analysis.md`](./sharding_strategy_analysis.md)), each
device holds all 4 global KV heads with the full sequence length. There is
no window bound --- the cache grows linearly with the number of tokens
generated.

| Parameter | Value |
|-----------|-------|
| KV heads per device | 4 (replicated) |
| head_dim | 512 |
| Window size | None (full causal) |
| Tensors | 2 (K + V) |

Per-device KV cache shape per global layer:

```text
K cache: [B, 4, S, 512]
V cache: [B, 4, S, 512]
```

where S is the current sequence length.

### Memory Per Layer Per Device

```math
\text{Global KV per layer per device} = 2 \times B \times 4 \times S \times 512 \times \text{bytes\_per\_element}
```

At B=1, BF16 (2 bytes/element):

| Seq Length (S) | Per Layer Per Device |
|----------------|---------------------|
| 1,024 | 2 x 4 x 1,024 x 512 x 2 = 8.0 MB |
| 2,048 | 2 x 4 x 2,048 x 512 x 2 = 16.0 MB |
| 4,096 | 2 x 4 x 4,096 x 512 x 2 = 32.0 MB |
| 8,192 | 2 x 4 x 8,192 x 512 x 2 = 64.0 MB |
| 16,384 | 2 x 4 x 16,384 x 512 x 2 = 128.0 MB |
| 32,768 | 2 x 4 x 32,768 x 512 x 2 = 256.0 MB |
| 65,536 | 2 x 4 x 65,536 x 512 x 2 = 512.0 MB |
| 131,072 | 2 x 4 x 131,072 x 512 x 2 = 1,024.0 MB |
| 262,144 | 2 x 4 x 262,144 x 512 x 2 = 2,048.0 MB |

At B=1, BFP8 (1 byte/element):

| Seq Length (S) | Per Layer Per Device |
|----------------|---------------------|
| 8,192 | 32.0 MB |
| 32,768 | 128.0 MB |
| 131,072 | 512.0 MB |
| 262,144 | 1,024.0 MB |

### Total Global KV Across 10 Layers

| Seq Length | BF16 (10 layers, B=1) | BFP8 (10 layers, B=1) |
|------------|----------------------|----------------------|
| 2,048 | 160 MB | 80 MB |
| 4,096 | 320 MB | 160 MB |
| 8,192 | 640 MB | 320 MB |
| 16,384 | 1,280 MB | 640 MB |
| 32,768 | 2,560 MB | 1,280 MB |
| 65,536 | 5,120 MB | 2,560 MB |
| 131,072 | 10,240 MB | 5,120 MB |
| 262,144 | 20,480 MB | 10,240 MB |

## Combined KV Cache Budget Per Device

The total KV cache per device is the sum of the constant sliding cache and
the variable global cache:

```math
\text{Total KV} = \underbrace{50 \times 2 \times 2 \times 1024 \times 256 \times \text{bpe}}_{\text{Sliding (constant)}} + \underbrace{10 \times 2 \times 4 \times S \times 512 \times \text{bpe}}_{\text{Global (grows with S)}}
```

| Seq Length | Sliding (BF16) | Global (BF16) | **Total (BF16)** | Total (BFP8) |
|------------|---------------|---------------|-----------------|-------------|
| 2,048 | 100 MB | 160 MB | **260 MB** | 130 MB |
| 4,096 | 100 MB | 320 MB | **420 MB** | 210 MB |
| 8,192 | 100 MB | 640 MB | **740 MB** | 370 MB |
| 16,384 | 100 MB | 1,280 MB | **1,380 MB** | 690 MB |
| 32,768 | 100 MB | 2,560 MB | **2,660 MB** | 1,330 MB |
| 65,536 | 100 MB | 5,120 MB | **5,220 MB** | 2,610 MB |
| 131,072 | 100 MB | 10,240 MB | **10,340 MB** | 5,170 MB |

The global KV cache dominates the total KV budget at all practical sequence
lengths beyond ~1K tokens.

## Page Table Configuration

The paged KV cache uses a page table to map logical token positions to
physical block locations in DRAM. The page table configuration depends on
the layer type.

### Sliding Layers

| Parameter | Value |
|-----------|-------|
| KV heads per device | 2 |
| head_dim | 256 |
| Max tokens in cache | 1,024 (window size) |
| Block size | 64 tokens (typical) |
| Blocks per layer per device | ceil(1,024 / 64) = 16 |
| Page table shape | `[B, 16]` |

The sliding window naturally limits the page table to 16 entries. Circular
buffer semantics can be used: when the 1,025th token arrives, the oldest
block is overwritten. The page table is updated to reflect the new mapping.
See [Chapter 5](../ch5_attention_module_design/paged_sdpa_sliding_window.md)
for the sliding window page table strategy.

### Global Layers

| Parameter | Value |
|-----------|-------|
| KV heads per device | 4 (replicated) |
| head_dim | 512 |
| Max tokens in cache | S (grows with generation) |
| Block size | 64 tokens (typical) |
| Blocks per layer per device | ceil(S / 64) |
| Page table shape | `[B, ceil(S / 64)]` |

At S=8,192: ceil(8,192 / 64) = 128 blocks per layer per device.
At S=32,768: ceil(32,768 / 64) = 512 blocks per layer per device.

Each page table entry is typically a 32-bit integer (4 bytes). The page table
memory overhead is negligible:
- At S=32,768: 512 entries x 4 bytes = 2 KB per layer per device.
- Across 10 global layers: 20 KB per device.

### Block Shape

Each KV cache block stores `block_size` tokens for all local KV heads:

| Layer Type | Block Shape (K or V) | Block Size (BF16) |
|------------|---------------------|-------------------|
| Sliding | `[2, 64, 256]` | 2 x 64 x 256 x 2 = 64 KB |
| Global | `[4, 64, 512]` | 4 x 64 x 512 x 2 = 256 KB |

Global layer blocks are 4x larger than sliding layer blocks due to having
2x the heads and 2x the head_dim.

## Total DRAM Budget Per Device

The following table summarizes the full DRAM budget per device, combining
model weights and KV cache at various sequence lengths. This uses BFP8
weights (the likely production dtype to fit within the 12 GB DRAM budget) and
BF16 KV cache.

### Weight Budget (BFP8)

| Component | Per Device |
|-----------|-----------|
| 50 sliding layers (weights) | ~2,993 MB |
| 10 global layers (weights) | ~764 MB |
| Embedding table (BF16, shared with LM head) | 262,144 x 5,376 x 2 / 8 = ~336 MB |
| Norms, cos/sin tables, misc | ~50 MB |
| **Total weights** | **~4,143 MB** |

### Combined Budget

| Seq Length | Weights (BFP8) | KV Cache (BF16) | **Total** | Headroom (of 12 GB) |
|------------|---------------|-----------------|-----------|-------------------|
| 2,048 | 4,143 MB | 260 MB | 4,403 MB | 7,885 MB |
| 4,096 | 4,143 MB | 420 MB | 4,563 MB | 7,725 MB |
| 8,192 | 4,143 MB | 740 MB | 4,883 MB | 7,405 MB |
| 16,384 | 4,143 MB | 1,380 MB | 5,523 MB | 6,765 MB |
| 32,768 | 4,143 MB | 2,660 MB | 6,803 MB | 5,485 MB |
| 65,536 | 4,143 MB | 5,220 MB | 9,363 MB | 2,925 MB |
| 131,072 | 4,143 MB | 10,340 MB | 14,483 MB | **EXCEEDS 12 GB** |

### Key Findings

1. **The model fits comfortably on T3K at up to 65K context** with BFP8
   weights and BF16 KV cache, with nearly 3 GB of headroom per device.

2. **131K+ context requires BFP8 KV cache.** With BFP8 KV cache at S=131,072,
   the KV budget drops from 10,340 MB to 5,170 MB, bringing the total to
   9,313 MB --- within the 12 GB limit with ~3.0 GB headroom.

3. **Full 256K context is infeasible with replicated global KV at BF16.**
   Even with BFP8 KV cache, 256K context requires 10,240 MB for KV alone,
   which with weights totals ~14.4 GB. Achieving full 256K context would
   require either (a) BFP4 KV cache, (b) a non-replicated global KV strategy
   (Options B or C from the sharding analysis, accepting the CCL cost), or
   (c) offloading KV cache to host memory.

4. **Sliding KV cache is negligible.** At 100 MB (BF16) or 50 MB (BFP8),
   the sliding cache across all 50 layers is smaller than a single global
   layer's cache at S=8,192. The window-bounded design is highly effective.

5. **The practical sweet spot for T3K is 8K--32K context** with BFP8 weights
   and BF16 KV cache, leaving 5--7 GB of headroom per device for activations,
   page tables, and temporary buffers.

---

**Next:** [Chapter 7 --- Decoder Layer and Full Model Assembly](../ch7_model_assembly/index.md)
