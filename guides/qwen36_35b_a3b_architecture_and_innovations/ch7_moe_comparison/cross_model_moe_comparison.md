# Cross-Model MoE Comparison

This file compares the Qwen3.6-35B-A3B MoE configuration against DeepSeek-V3 and Gemma4-26B-A4B, then analyzes the hardware utilization implications of the many-small-experts design on Tenstorrent hardware.

---

## Summary Table

| Property | Qwen3.6-35B-A3B | DeepSeek-V3 | Gemma4-26B-A4B |
|---|---|---|---|
| Total parameters | 35B | ~685B | 26B |
| Hidden size | 2048 | 7168 | 2048 |
| Total routed experts | 256 | 256 | 32 |
| Shared experts | 1 | 1 | 0 |
| Top-k routing | 8 | 8 | 1 |
| Expert intermediate size | 512 | 2048 | 2048 |
| Per-expert params | ~3.1M | ~44M | ~12.6M |
| Load balancing | Auxiliary-loss-free | Auxiliary-loss-free | Auxiliary loss |
| Active params per token | ~3B | ~37B | ~4B |

---

## DeepSeek-V3

### Configuration

DeepSeek-V3 uses the same high-level routing topology as Qwen3.6: 256 routed experts, 1 shared expert, and top-8 selection. The structural similarity ends there.

- **Expert intermediate size**: 2048 vs Qwen3.6's 512 — each DeepSeek-V3 expert is 4× wider at the bottleneck.
- **Hidden size**: 7168 vs Qwen3.6's 2048 — the input and output dimension of each expert is 3.5× larger.
- **Per-expert parameters**: $3 \times 7168 \times 2048 \approx 44\text{M}$ per expert (SwiGLU, three matrices of shape 7168×2048 or 2048×7168). This is approximately 14× larger than Qwen3.6's ~3.1M per expert.
- **Total parameters**: ~685B, compared to Qwen3.6's 35B. The ~20× total parameter ratio is driven entirely by the larger hidden size and expert width; the number of experts is the same.

### Shared Design Philosophy

Both models adopt the same many-small-experts design philosophy relative to a single large dense FFN:

- 256 experts provide fine-grained routing choices.
- Top-8 routing activates a substantial fraction of the expert bank per token (8/256 = 3.1%), blending multiple specializations.
- Auxiliary-loss-free load balancing avoids the accuracy penalty of auxiliary loss terms.

The key difference is scale: DeepSeek-V3 applies this philosophy to a model roughly 20× larger, which changes the compute and memory profile substantially but not the structural logic.

### Scale Implications

At DeepSeek-V3's scale, the per-expert matmuls are $7168 \times 2048$ — large enough to saturate accelerator compute units without any batching tricks. Qwen3.6's $2048 \times 512$ matmuls are much smaller and require explicit expert batching for efficient utilization (see hardware discussion below).

---

## Gemma4-26B-A4B

### Configuration

Gemma4-26B-A4B takes the opposite design choice: **fewer, larger experts**.

- **Total routed experts**: 32 (vs Qwen3.6's 256) — an 8× reduction.
- **Shared experts**: none — all experts are routed.
- **Top-k routing**: top-1 (only 1 expert is activated per token, vs Qwen3.6's 8).
- **Expert intermediate size**: 2048 (vs Qwen3.6's 512) — 4× wider at the bottleneck.
- **Hidden size**: 2048 — same as Qwen3.6.
- **Per-expert parameters**: $3 \times 2048 \times 2048 = 12{,}582{,}912 \approx 12.6\text{M}$ per expert (SwiGLU, intermediate=2048).
- **Total parameters**: ~26B, similar scale to Qwen3.6's 35B but with very different expert structure.

### Routing Philosophy

Gemma4's top-1 routing is the starkest difference. Each token activates exactly one expert — the one with the highest router logit. This is the same routing scheme used by the original Mixture-of-Experts papers (Switch Transformer, GShard). It has several consequences:

- **Simpler dispatch**: one expert weight set is loaded per token per layer, minimizing routing bookkeeping.
- **Less routing flexibility**: 32-choose-1 provides 32 possible FFN computations per token, compared to Qwen3.6's 256-choose-8 = $\binom{256}{8} \approx 10^{15}$ combinations.
- **No blending**: the output is a single expert's contribution, with no weighted combination of multiple specializations.
- **Higher expert utilization pressure**: with top-1 routing and 32 experts, each expert must serve a wider range of inputs, which constrains the degree of specialization achievable.

---

## Many-Small-Experts vs Fewer-Large-Experts

The comparison above frames a general design axis in MoE models. Qwen3.6 sits firmly at the many-small-experts end (256 × intermediate=512), while Gemma4 sits at the fewer-large-experts end (32 × intermediate=2048).

### Advantages of Many Small Experts (Qwen3.6 Approach)

**Finer-grained specialization.** With 256 experts each handling a narrow intermediate dimension of 512, each expert can specialize in a more specific subdomain of the training distribution. The model can carve the input space into 256 distinct regions rather than 32, potentially capturing more fine-grained distinctions in language, reasoning, and domain knowledge.

**More routing flexibility.** The 256-choose-8 routing space is combinatorially richer than 32-choose-1. Different token types can activate very different expert combinations, giving the model more expressive power in how it blends specializations. This flexibility is particularly valuable for a model that must handle diverse tasks (coding, reasoning, math, language).

**Lower per-expert memory footprint.** Each expert weight set is approximately 3.1M parameters at bfp16 — about 6.3 MB. This small size enables efficient batching of multiple expert weight sets in L1/L2 cache during a forward pass, and simplifies expert-parallel sharding without large communication payloads per expert.

### Disadvantages of Many Small Experts

**Higher routing overhead.** A 256-way softmax followed by top-8 selection and weight normalization is significantly more expensive than a 32-way softmax followed by argmax. The router itself (linear projection [2048 → 256]) has $256 \times 2048 = 524{,}288$ parameters and costs $2 \times 2048 \times 256 = 1{,}048{,}576$ FLOPs — not negligible relative to the expert computation.

**More expert weight tensors in DRAM.** The total number of routed expert weight matrices is 30,720 (plus 120 shared expert weight sets, for 30,840 total). Each must reside somewhere in the memory hierarchy. With 256 experts per layer, the probability that a given expert is needed for any particular batch of tokens is lower, meaning cold experts must be fetched from DRAM more often than in a 32-expert model.

**Poorer compute utilization per matmul.** The dominant expert matmuls are shaped [2048, 512] (gate and up projections) and [512, 2048] (down projection). These are small by accelerator standards. On most hardware, systolic arrays or tensor cores achieve peak utilization only when all dimensions are large. A [2048 × 512] matmul leaves significant compute capacity underutilized compared to the [7168 × 2048] matmuls that DeepSeek-V3 executes.

**Higher all-to-all communication volume in expert-parallel deployments.** When experts are sharded across devices, each token must be routed to the device holding its assigned expert. With top-8 routing across 256 experts, each token may require communication to up to 8 different devices. The all-to-all volume scales with top-k and the number of devices. See `guides/expert_parallelism_strategies/` for the detailed analysis.

---

## Hardware Utilization Implications for Tenstorrent

The Qwen3.6 MoE configuration presents specific challenges and opportunities on Tenstorrent T3K devices.

### Compute vs DRAM Bandwidth

Each active expert weight set is 6.3 MB at bfp16 (3 matrices × 1,048,576 elements × 2 bytes); loading 9 active expert sets per layer reads 56.7 MB, and across 40 layers approximately 2.2 GB per token per forward pass at bfp16 — dropping to ~0.55 GB at bfp4 (see [`qwen36_moe_architecture.md`](./qwen36_moe_architecture.md) for the full derivation).

**DRAM bandwidth, not compute, is the bottleneck for Qwen3.6 MoE inference.** This is the expected regime for large-expert-count MoE models with small expert dimensions.

### Expert Parallelism on T3K (8 Devices)

With 256 routed experts distributed across 8 T3K devices:

$$\frac{256 \text{ experts}}{8 \text{ devices}} = 32 \text{ experts per device}$$

This is a clean, balanced partition. Each device holds 32 expert weight sets per layer. When a token's top-8 routing selects experts distributed across devices, the token's hidden state is sent to the relevant devices via all-to-all, each device computes its assigned expert's output, and the partial outputs are aggregated.

With top-8 routing, a single token can require communication with up to 8 of the 8 devices — meaning in the worst case, every device participates in every token's routing. In practice, load balancing ensures expert activations are roughly uniform, and the expected number of device-hops per token is at most 8. The all-to-all volume is proportional to $B \times 8 \times H = B \times 8 \times 2048$ per layer, which at small batch sizes (B=1 or B=32) is manageable. See `guides/expert_parallelism_strategies/` for T3K-specific all-to-all tuning.

### Expert Batching for Compute Efficiency

The [2048, 512] expert matmuls are too small to saturate Tenstorrent compute units when executed individually. The mitigation strategy is to batch tokens across experts: if a batch of $B$ tokens activates the same expert, the matmul becomes [B × some_count, 512] for the input projection and [$B \times$ some_count, 2048] for the output — large enough to achieve higher utilization. See `guides/moe_optimization_techniques_for_ttnn/` for the token-dispatch-and-batch implementation.

### bfp4 Quantization of Expert Weights

Storing all expert weight tensors in DRAM at bfp4 is critical for two reasons:

1. **Aggregate DRAM capacity.** The 30,720 expert weight matrices at bfp16 require approximately $30{,}720 \times 1{,}048{,}576 \times 2 \approx 64\text{ GB}$. At bfp4 this drops to ~16 GB, fitting within the aggregate DRAM of a T3K system with headroom for the non-expert parameters and KV cache.

2. **DRAM bandwidth reduction.** Each expert weight load at bfp4 transfers 4× fewer bytes than at bfp16, directly reducing the per-token bandwidth cost from ~2.2 GB to ~0.55 GB per forward pass.

See `guides/weight_quantization_for_moe_experts/` and `guides/ttnn_moe_performance_optimization_on_t3k/` for quantization implementation details.

---

**Next:** [Chapter 8 -- Vision Encoder and Multimodal Integration](../ch8_vision_encoder/index.md)
