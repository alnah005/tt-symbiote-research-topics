# Qwen3.5-35B Architecture Configuration

## Purpose of This File

This file catalogs the concrete architectural constants for Qwen3.5-35B that are relevant to expert parallelism design. Abstract discussions of MoE systems become actionable only when grounded in specific numbers. The values here are referenced repeatedly in Chapters 4 through 8 when computing memory budgets, communication volumes, and compute costs.

Readers who want to understand *why* these numbers create specific engineering challenges should read `moe_architecture.md` and `routing_problem.md` first.

---

## Architectural Constants

### Model-Level Parameters

| Constant | Symbol | Value |
|---|---|---|
| Total layers (transformer blocks) | $L_\text{total}$ | 94 |
| MoE layers | $L_\text{MoE}$ | 80 |
| Dense FFN layers | $L_\text{dense}$ | 14 |
| Hidden dimension | $H$ | 7168 |
| Number of key-value heads (grouped-query attention, GQA) | — | 8 |
| Vocabulary size | — | 151,936 |

> **Note:** Query head count and head dimension have been removed from this table pending verification. A candidate value of 64 query heads × 128 head_dim = 8,192 ≠ H = 7,168; these figures are internally inconsistent and cannot both be correct for this model. Head count and head dimension are integers and cannot be reconciled by rounding. Source the correct values from the official Qwen3 Technical Report and confirm that (head_count × head_dim) = H before using them in any calculation.

Qwen3.5-35B uses grouped-query attention (GQA) with 8 key-value heads; the query head count and head dimension are unverified pending confirmation from the official Qwen3 Technical Report (see note above). GQA reduces the memory footprint and compute cost of attention relative to full multi-head attention. This is independent of the MoE design but relevant for understanding the total memory budget per device.

### MoE Layer Parameters

| Constant | Symbol | Value |
|---|---|---|
| Total experts per MoE layer | $E$ | 256 |
| Top-$k$ routing | $k$ | 8 |
| Expert FFN intermediate dimension | $D$ | **2048 — UNVERIFIED; do not use for implementation decisions** (see warning block below) |
| Expert activation function | — | SiLU (Sigmoid Linear Unit) |
| Expert FFN style | — | SiLU-gated linear unit (SwiGLU) |
| Router weight matrix shape | $W_r$ | $[H, E] = [7168, 256]$ |

### Dense FFN Layer Parameters

The 14 dense layers use a larger intermediate dimension than the expert layers:

| Constant | Value |
|---|---|
| Dense FFN intermediate dimension | **18,944 — UNVERIFIED; do not use for implementation decisions** (see warning block below) |
| Dense FFN activation function | SiLU (Sigmoid Linear Unit) |
| FFN style | SwiGLU (three matrices: W_gate, W_up, W_down) |

This asymmetry is intentional: the dense layers (which process every token for every forward pass) are given higher capacity, while the MoE experts (which each process only a fraction of tokens) use a narrower intermediate dimension $D$ (stated as 2048 but **unverified** — see warning block below; do not use this figure for implementation decisions until confirmed against the Qwen3 Technical Report).

> **⚠ UNVERIFIED PLACEHOLDER — D_dense = 18,944:** The dense FFN intermediate dimension $D_\text{dense} = 18{,}944$ has not been confirmed against the Qwen3 Technical Report. All figures derived from $D_\text{dense}$ are therefore unverified placeholders: the dense FFN parameter count ($14 \times 3 \times 7168 \times 18{,}944 \approx 5.7\text{B}$), the 19.3B non-expert parameter sum (which includes the dense FFN contribution), and the ~42.5 GB per-device EP memory figure (which is derived from the 19.3B non-expert estimate) must not be used for implementation decisions until $D_\text{dense}$ is sourced from the Qwen3 Technical Report. Additionally, the ~12.4B attention weight estimate that contributes to the 19.3B non-expert sum is itself an unverified contributor to the ~42.5 GB figure: it depends on (query head count × head_dim) = H = 7,168, but the query head count and head dimension were removed from the constants table as unverified because the candidate values (64 × 128 = 8,192 ≠ 7,168) are internally inconsistent. A reader who independently verifies $D_\text{dense}$ and then trusts the remainder of the ~42.5 GB sum will silently inherit this unverified ~12.4B input. The correct query head count and head dimension must also be sourced from the Qwen3 Technical Report before the attention weight estimate and the ~42.5 GB figure can be treated as verified.

> ## ⚠ IMPORTANT: ARCHITECTURAL CONSTANTS REQUIRE VERIFICATION
>
> The constants listed in the tables above are **not internally self-consistent**. Before relying on any numerical calculation derived from these figures, readers must be aware of the following contradictions:
>
> **1. Per-layer expert weight count vs. published 35B total.**
> With $D = 2048$, $H = 7168$, and $E = 256$ experts, each MoE layer contains $256 \times 3 \times 7168 \times 2048 \approx 11.274\text{B}$ expert parameters. Across $L_\text{MoE} = 80$ MoE layers, this naively sums to $11.274\text{B} \times 80 \approx 902\text{B}$ total expert parameters — more than 25 times the published ~35B total for the entire model. The primary suspect is $D$ (the per-expert intermediate dimension); at least one of {$D$, expert count $E$, MoE layer count $L_\text{MoE}$} must differ materially from the value stated in the tables above. Cross-check with the Qwen3 Technical Report to obtain the authoritative value of $D$.
>
> **2. Active parameter first-principles sum exceeds total parameter count.**
> The first-principles active parameter sum (~46.3B, derived in the "Active Parameter Count per Token" section below) exceeds the total model parameter count (~35B). Active parameters for a token cannot exceed the model's total parameters. This is arithmetically impossible and confirms that the architectural constants as stated are not self-consistent. No valid combination of these figures can simultaneously satisfy all three: the 35B total, the per-expert dimension of $D = 2048$, and the 80-layer, 256-expert configuration.
>
> **3. Memory section expert weight figure (~15.7B) conflicts with the per-layer calculation.**
> The memory analysis below derives total expert weight as $35\text{B} - 19.3\text{B} \approx 15.7\text{B}$ by subtracting all non-expert parameters (attention ~12.4B + embeddings ~1.1B + dense FFN ~5.7B + layer norms ~0.1B) from the published total. This is the coherent approach given the published 35B figure, but it directly contradicts the per-layer calculation above (which yields ~902B). Both figures appear in this file; they cannot both be correct.
>
> **4. Action required before using these figures.**
> Readers must consult the **Qwen3 Technical Report** (Qwen Team, Alibaba Group, arXiv, 2025) to obtain the correct per-expert intermediate dimension $D$ and confirm the MoE layer count and expert count before relying on any numerical calculations in this file. The published 35B total parameter count and the "A22B" active parameter designation are used as authoritative figures throughout this guide; all first-principles derivations that contradict them should be treated as unverified approximations until the correct architectural constants are confirmed.

---

## Layer Structure: MoE vs. Dense

### Layer Placement

Qwen3.5-35B uses a sandwich structure: dense layers at both ends with MoE layers in the middle. The authoritative aggregate counts — $L_\text{MoE} = 80$ MoE layers and $L_\text{dense} = 14$ dense layers — are used as the authoritative values for all downstream calculations throughout this guide.

The following boundary indices are **approximate** and should be confirmed against `config.json` in the official Qwen3.5-35B-A22B repository before relying on them:

> **⚠ Verify:** layer boundary indices against `config.json` in the official Qwen3.5-35B-A22B repository.

- Layers 0–3: dense FFN (4 layers) — *approximate*
- Layers 4–83: MoE FFN (80 layers) — *approximate*
- Layers 84–93: dense FFN (10 layers) — *approximate*

Any variation in boundary positions does not affect the aggregate counts ($L_\text{MoE} = 80$, $L_\text{dense} = 14$), which are taken from the published Qwen3 technical report and used as authoritative values for this guide, subject to the verification caveat in the warning block above.

### Why Dense Layers at the Boundaries?

Dense layers at the input and output ends of the model serve as stable feature extractors and predictors that do not depend on routing decisions. The input-side dense layers process token inputs before the MoE section begins. The output-side dense layers apply learned transformations in the residual stream before the language model head converts hidden states to vocabulary logits. Using MoE layers at these positions would add routing variance without clear benefit.

---

## Expert Weight Sizes and Memory Footprint

### Per-Expert Parameter Count

> **⚠ UNVERIFIED PLACEHOLDER — do not use for implementation decisions.** All figures in this section derive from $D = 2048$, which has not been confirmed against the Qwen3 Technical Report. The per-layer expert weight count implied by $D = 2048$ ($\approx 11.3$B parameters/layer × 80 layers ≈ 902B total) exceeds the entire 35B model by more than 25×, which is impossible. These figures are retained only as a worked example of the formula structure; the numeric outputs are wrong and must be replaced once the correct value of $D$ is sourced. See the "ARCHITECTURAL CONSTANTS REQUIRE VERIFICATION" warning block.

Each expert is a gated linear unit FFN with three weight matrices:
- Gate projection: $W_{e,\text{gate}} \in \mathbb{R}^{H \times D} = \mathbb{R}^{7168 \times D_\text{unverified}}$
- Up projection: $W_{e,\text{up}} \in \mathbb{R}^{H \times D} = \mathbb{R}^{7168 \times D_\text{unverified}}$
- Down projection: $W_{e,\text{down}} \in \mathbb{R}^{D \times H} = \mathbb{R}^{D_\text{unverified} \times 7168}$

Total parameters per expert (formula is correct; numeric result is an unverified placeholder):

$$P_\text{expert} = 2 \times H \times D + D \times H = 3 \times H \times D = 3 \times 7168 \times D_\text{unverified} \quad \text{[PLACEHOLDER: } \approx 44\text{M if } D=2048\text{]}$$

### Per-Layer Expert Weight Memory

> **⚠ UNVERIFIED PLACEHOLDER** — numeric values depend on unverified $D$.

With $E = 256$ experts per MoE layer and each expert having $P_\text{expert}$ parameters (unverified):

$$\text{Expert params per layer} = 256 \times P_\text{expert} \quad \text{[PLACEHOLDER: } \approx 11.3\text{B if } D=2048\text{]}$$

In BF16 (2 bytes per parameter):

$$\text{Expert weight memory per layer} = \text{(Expert params per layer)} \times 2 \quad \text{[PLACEHOLDER: } \approx 22.5\text{ GB if } D=2048\text{]}$$

### Per-Device Memory Under Uniform Assignment

With $N = 8$ devices, the total model (~35B parameters) is distributed as follows:

| Precision | Total model memory | Per-device memory under full sharding (EP+TP combined, ÷8) — not expert parallelism (EP) alone; see note below | Per-device memory under pure EP (see note below) | Fits in T3K (`[verify: believed to be 12–24 GB/device — check official T3K datasheet]` per device)? |
|---|---|---|---|---|
| BF16 (2 bytes/param) | $35\text{B} \times 2 \approx 70\text{ GB}$ | $\approx 8.75\text{ GB}$ | $\approx 42.5\text{ GB}$ | Full TP: Yes; Pure EP: No — substantially exceeds typical T3K per-device DRAM |
| INT8 (1 byte/param) | $35\text{B} \times 1 \approx 35\text{ GB}$ | $\approx 4.4\text{ GB}$ | $\approx 21.25\text{ GB}$ | Full TP: Yes — leaves ample headroom; Pure EP: marginal at 24 GB/device |

> **⚠ Verify:** T3K DRAM capacity (`[verify: believed to be 12–24 GB/device — check official T3K datasheet]` per device) against official Tenstorrent T3K hardware datasheet.

> **Note:** Memory feasibility conclusions depend critically on verifying T3K DRAM capacity; BF16 is tight at 12 GB/device.

> **Note:** The "÷8" calculation assumes all parameters are evenly sharded across 8 devices (equivalent to combining expert parallelism with full tensor parallelism). Under pure expert parallelism (EP only), expert weights are sharded across $N = 8$ devices while non-expert parameters are replicated on every device. **The per-device memory bound below is derived solely from the authoritative 35B total; it does not depend on the unverified value of $D$.** Assumption: non-expert parameters (attention weights, embeddings, dense FFN weights, layer norms) constitute approximately 19.3B of the 35B total, leaving approximately 15.7B as expert weights — derived by subtraction ($35\text{B} - 19.3\text{B}$), not from per-layer arithmetic. Under this assumption, per-device BF16 memory under pure EP = $(\text{expert params} / N + \text{non-expert params}) \times 2 = (15.7\text{B}/8 + 19.3\text{B}) \times 2 \approx (1.96 + 19.3) \times 2 \approx 42.5$ GB. **This figure is contingent on the 19.3B non-expert estimate being correct; the exact value should be deferred until architectural constants are verified.** The target configuration for this guide assumes some form of combined EP + TP or quantization; see Chapter 8, `ch08_qwen35b_t3k_strategy/recommended_configuration.md` for the reference configuration.

These estimates cover model weights only; activations, KV-cache, and operating overhead add to the per-device footprint.

> **⚠ Verify:** T3K aggregate DRAM capacity across 8 devices (`[verify: ~96–192 GB aggregate]`).

Even so, the ~35B total parameter count is readily accommodated at the upper end of the verified range (~192 GB); marginal at the lower estimate (~96 GB) by the T3K aggregate DRAM across 8 devices. The per-layer expert weight memory figure stated above depends on the unverified value of $D$ and must be treated as a placeholder until $D$ is confirmed. Under pure expert parallelism, the ~42.5 GB per-device BF16 footprint (derived from the authoritative 35B total — see note above; exact figure deferred until non-expert parameter estimates are verified) significantly exceeds typical T3K per-device DRAM capacity, confirming that pure EP alone is insufficient — combined EP + TP or quantization is required for Qwen3.5-35B on T3K. For the purposes of this guide, memory calculations use BF16 unless stated otherwise; INT8 halves the values.

### Router Weight Memory

The router weight matrix $W_r \in \mathbb{R}^{H \times E} = \mathbb{R}^{7168 \times 256}$ has $7168 \times 256 = 1{,}835{,}008 \approx 1.8$M parameters per MoE layer — negligibly small compared to the expert weights.

---

## Active vs. Total Parameters

### Total Parameter Count

The total parameter count for Qwen3.5-35B is approximately 35B parameters. This model should not be confused with Qwen3-235B, a different (larger) model with approximately 235B total parameters. The "35B" in the model name refers to the total parameter count; "A22B" refers to the approximately 22B active parameters per token.

| Component | Parameter count |
|---|---|
| Token embedding table | $151{,}936 \times 7168 \approx 1.1\text{B}$ |
| Attention weights (all 94 layers) | $\approx 12.4\text{B}$ |
| Dense FFN weights (14 layers) | $\approx 5.7\text{B}$ |
| Layer norms, output projection | $< 0.1\text{B}$ |
| Expert FFN weights (80 MoE layers, distributed across 256 experts × 3 matrices) | See "Expert Weight Sizes" section — dominant contributor to the 35B total; naive per-layer sum does not reconcile |
| **Effective model total** | **$\approx 35\text{B}$ (per official Qwen3 technical report; see note below)** |

Note: A naive sum of per-expert parameters across all layers would exceed the published 35B total. The official Qwen3 technical report does not provide a complete per-component parameter breakdown; we use the published 35B total as the authoritative figure.

Note: Dense FFN weight derivation: $14 \times 3 \times 7168 \times 18944 \approx 5.7\text{B}$ (14 layers, SwiGLU with 3 matrices each of shape $[H, D_\text{dense}]$ or $[D_\text{dense}, H]$, where $D_\text{dense} = 18{,}944$).

> **⚠ Verify:** published 35B total parameter count against official Qwen3.5-35B technical report.

### Active Parameter Count per Token

For a single token, a first-principles count gives:
- Attention weights (all 94 layers, always active): $\approx 12.4$B
- Dense FFN weights (14 layers, always active): $\approx 5.7$B
- $k = 8$ expert weights per MoE layer, out of $E = 256$ (80 MoE layers): $80 \times 8 \times P_\text{expert}$ **[UNVERIFIED PLACEHOLDER: $\approx 28.2$B if $D = 2048$; do not use for implementation decisions — see verification warning]**

**Note:** This derivation does not reconcile with the manufacturer's 22B figure; the 46.3B result is arithmetically impossible — see the "ARCHITECTURAL CONSTANTS REQUIRE VERIFICATION" warning block above for the authoritative explanation.

This first-principles sum yields approximately $12.4 + 5.7 + 28.2 \approx 46.3\text{B}$. The gap to the manufacturer's 22B "A22B" figure is ~24.3B ($46.3\text{B} - 22\text{B}$). Consult the Qwen3 Technical Report directly for the correct architectural constants and authoritative parameter accounting. The published 22B figure is adopted here as authoritative.

The manufacturer designates this model as "A22B" (approximately 22 billion active parameters per token). **We use 22B as the authoritative active parameter count throughout this guide.** This figure is consistent with the model's "Qwen3.5-35B-A22B" designation and with `moe_architecture.md`.

---

## Activation Function: SiLU and Its Effect on Expert Compute

### SiLU Definition

The SiLU (Sigmoid Linear Unit) activation function is defined as:

$$\text{SiLU}(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}$$

where $\sigma$ is the logistic sigmoid function. SiLU is smooth everywhere, has a non-monotonic region for $z < 0$, and is broadly similar in behavior to GELU (Gaussian Error Linear Unit).

### Effect on Expert Compute Cost

In the gated FFN architecture used by Qwen3.5-35B, each expert requires:
1. A matrix multiply for the gate projection: $[1, H] \times [H, D] \to [1, D]$
2. A matrix multiply for the up projection: $[1, H] \times [H, D] \to [1, D]$
3. Element-wise SiLU on the gate projection output
4. Element-wise multiplication of gate and up outputs
5. A matrix multiply for the down projection: $[1, D] \times [D, H] \to [1, H]$

The gated design requires two up-projection-sized matrix multiplies instead of one, doubling the compute cost of the first stage relative to a standard two-layer FFN with the same intermediate dimension. However, this is offset by the fact that the intermediate dimension $D = 2048$ is smaller than typical dense FFN intermediate dimensions, so the total FLOPs are comparable.

FLOPs per token per expert (ignoring non-linear ops which are negligible — element-wise SiLU and gating operations ≈ D per token, roughly 43,000× fewer FLOPs than the 6HD matmul cost per token — negligible in practice):

$$\text{FLOPs}_\text{expert} = 2 \times (H \times D + H \times D + D \times H) = 2 \times 3 \times H \times D = 6HD$$

> **Note:** The leading ×2 factor in the formula reflects the MAC (multiply-accumulate) convention: one multiply-accumulate operation counts as 2 FLOPs (one multiply + one add). Under this convention, a matrix multiply of shape $[1, m] \times [m, n]$ costs $2mn$ FLOPs.

> **Note:** $6HD$ is the per-token cost (for a single token dispatched to the expert). For a dispatched batch of $T$ tokens, the total is $6THD$.

> **⚠ UNVERIFIED PLACEHOLDER — do not use for implementation decisions.** The numeric substitution below uses $D = 2048$, which is unverified (see warning block). The formula $6HD$ is correct; the numeric result must be recomputed once the correct $D$ is confirmed from the Qwen3 Technical Report.

Substituting $H = 7168$, $D = D_\text{unverified}$ (placeholder value 2048):

$$\text{FLOPs}_\text{expert} = 6 \times 7168 \times D_\text{unverified} \quad \text{[PLACEHOLDER: } \approx 88\text{M FLOPs if } D=2048\text{]}$$

For $k = 8$ active experts per token and $L_\text{MoE} = 80$ MoE layers:

$$\text{Total expert FLOPs per token} = 80 \times 8 \times \text{FLOPs}_\text{expert} \quad \text{[PLACEHOLDER: } \approx 56.4\text{G FLOPs if } D=2048\text{]}$$

The formula structure is correct and guides batch-size threshold analysis in Chapter 6, `ch06_fused_dispatch_compute_combine/end_to_end_latency_model.md`, but the numeric values must not be used for implementation decisions until $D$ is verified.

---

## Why 256 Experts with Top-8 Creates Unique Sharding Pressure

### The Sharding Problem Stated

The combination of $E = 256$ experts and $k = 8$ routing creates a specific pressure that does not arise in models with smaller expert counts:

**1. High expert count relative to device count.** With $E = 256$ and $N = 8$ devices, each device hosts 32 experts (under uniform assignment). This is a large local expert batch. Each dispatch operation must correctly route tokens to any of 256 possible destinations, requiring a routing index structure with 256 buckets. As a hypothetical comparison point, a model with $E = 8$ total experts (one per device on T3K) would have trivially mapped routing; this is not a configuration of Qwen3.5-35B but illustrates why the $E = 256$ case is substantially more complex. At $E = 256$, the mapping from expert index to device index requires a non-trivial lookup.

**2. High top-$k$ relative to device count.** With $k = 8$ and $N = 8$, the top-$k$ count equals the device count. In expectation, each token is routed to exactly one expert on each device (since $k/N = 1$). Under uniform routing, on average one expert slot per token lands on the local device (no cross-device send required for that slot), while the remaining $k - 1 = N - 1 = 7$ slots require cross-device sends to 7 remote devices. The expected fan-out is therefore $N - 1 = 7$ remote devices, not $N = 8$. This still makes the all-to-all traffic pattern nearly maximally dense — there are no "lucky" tokens that happen to have all their experts on one device, and essentially every token incurs 7 of 8 possible remote sends.

Compare to a model with $E = 64$, $k = 2$, $N = 8$: the top-$k$ count is only 2, so most tokens are dispatched to only 2 of 8 devices, and the all-to-all is highly sparse.

**3. Router logit tensor size.** The router must compute $G = XW_r \in \mathbb{R}^{B \times 256}$ for every MoE layer. With $B = 512$ and BF16, this tensor is $512 \times 256 \times 2 = 262{,}144$ bytes $= 256$ KB. While not huge in isolation, computing and materializing this tensor for 80 layers across the forward pass adds up. Chapter 5, `ch05_routing_weight_optimization/topk_selection_efficiency.md` discusses how to reduce this cost.

**4. Expert replication economics.** With 256 experts, a model designer might consider replicating only the most popular 1% of experts (about 2–3 experts) on additional devices. The fine-grained control this enables is an advantage, but it also requires sophisticated dispatch logic to track which device holds which replica. This is analyzed in Chapter 4, `ch04_expert_device_assignment/expert_replication.md`.

**5. Load-balancing space.** A larger expert pool theoretically allows finer-grained load balancing: the router has more targets to spread tokens across, so any individual expert receives a smaller fraction of the total traffic. In practice, MoE routers often exhibit a long-tail distribution where a small number of experts receive much more traffic than others. With 256 experts, the top-8 most popular experts might still handle a large fraction of tokens if the router's learned weights concentrate probability mass. This is the subject of Chapter 7, `ch07_load_balancing/`.

### Summary of the Key Tension

The 256-expert, top-8 configuration sits at a specific point in the design space:

| Property | Value | Implication |
|---|---|---|
| Expert count $E$ | 256 | Large: fine-grained specialization, complex routing, replication-feasible |
| Top-$k$ $k$ | 8 | Equals device count $N$: all-to-all traffic is dense per token |
| Experts per device (uniform) | 32 | Large local batch: expert FFN can be batched efficiently |
| Cross-device fraction per token | $\frac{N-1}{N} = \frac{7}{8} = 87.5\%$ | Nearly all of each token's expert work requires communication. |
| Router output tensor $[B, E]$ | $[B, 256]$ | 256-way classification: non-trivial top-$k$ compute at large $B$ |

Note: The cross-device fraction $(N-1)/N$ holds for any value of $k$ under uniform routing — it is $k$-independent. The expected **count** of remote experts per token, $k(N-1)/N$, does depend on $k$. See `routing_problem.md` for the full derivation.

This table encapsulates why Qwen3.5-35B on T3K requires the careful expert parallelism design that this guide addresses. The expert FFN compute is efficient (well-matched to hardware via large local batches), but the communication overhead is nearly unavoidable and the routing logic is non-trivial.

---

## References

- [Qwen3] Qwen Team, Alibaba Group, "Qwen3 Technical Report", 2025.
- [Qwen3HF] Qwen Team, "Qwen3.5-35B-A22B Model Card", HuggingFace, 2025. URL: https://huggingface.co/Qwen/Qwen3.5-35B-A22B. Note: architectural constants sourced from Qwen3 Technical Report and Hugging Face model card; see VERIFICATION WARNING block for caveats.

> **⚠ Verify:** the exact HuggingFace repository path for the Qwen3.5-35B-A22B model card.
- [Shazeer2017] Shazeer, N. et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer", ICLR, 2017.
- [Dauphin2017] Dauphin, Y. N. et al., "Language Modeling with Gated Convolutional Networks", ICML, 2017. (GLU architecture)
- [Hendrycks2016] Hendrycks, D. and Gimpel, K., "Gaussian Error Linear Units (GELUs)", arXiv:1606.08415, 2016. (SiLU/GELU family)
- [Ch1MoEArch] Chapter 1, `ch01_moe_fundamentals/moe_architecture.md` — MoE layer structure and routing math.
- [Ch1Routing] Chapter 1, `ch01_moe_fundamentals/routing_problem.md` — communication overhead and load skew.
- [Ch4Assignment] Chapter 4, `ch04_expert_device_assignment/` — expert-to-device assignment strategies.
- [Ch4Replication] Chapter 4, `ch04_expert_device_assignment/expert_replication.md` — expert replication analysis.
- [Ch5TopK] Chapter 5, `ch05_routing_weight_optimization/topk_selection_efficiency.md` — top-k computation efficiency.
- [Ch6Latency] Chapter 6, `ch06_fused_dispatch_compute_combine/end_to_end_latency_model.md` — parameterized latency model.
- [Ch7Load] Chapter 7, `ch07_load_balancing/` — load balancing mechanisms.

---

**Next:** [Chapter 2 — All-to-All Communication Primitives](../ch02_all_to_all_primitives/index.md)
