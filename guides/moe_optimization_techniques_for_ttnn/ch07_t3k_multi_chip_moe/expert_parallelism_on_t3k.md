# Expert Parallelism on T3K

## Section 1: EP Partitioning for Qwen3.5-35B

Expert Parallelism (EP) distributes the expert FFN weight tensors across devices so that each device stores and computes only a subset of experts. On the T3K eight-chip mesh, the EP degree equals the number of devices: $N = 8$.

For Qwen3.5-35B with $E = 256$ experts:

$$E_d = \frac{E}{N} = \frac{256}{8} = 32 \text{ experts per device}$$

Each device independently stores its 32 expert weight tensors in DRAM and is responsible for computing those experts' FFN outputs during inference. No device ever needs another device's expert weights at inference time; all cross-device coordination happens through token routing, not weight exchange.

The sparsity ratio at decode is:

$$\rho = \frac{k}{E} = \frac{8}{256} = 3.1\%$$

At $B = 1$, only 8 experts fire across all 256. Under uniform routing, the expected number of active experts per device is:

$$\mathbb{E}[\text{active experts per device}] = k \cdot \frac{E_d}{E} = 8 \cdot \frac{32}{256} = 1$$

So on average only 1 of the 32 local experts is active at $B = 1$. This extreme sparsity makes `sparse_matmul` critical; see Chapter 4 ([when_sparse_matmul_wins.md](../ch04_sparse_matmul_for_moe/when_sparse_matmul_wins.md)) for the utilization threshold analysis.

**Assignment strategies** (load balancing, expert affinity, auxiliary loss tuning) are covered in the Expert Parallelism guide, Chapter 4. This chapter focuses on the TTNN-level implementation: how tensors are shaped, how all-to-all is configured, and what the resulting program configs look like.

---

## Section 2: All-to-All Communication Pattern

MoE inference under EP follows a three-phase pattern per MoE layer:

### Phase 1: Token Dispatch

The router assigns each of the $B$ input tokens to $k = 8$ experts. `ttnn.all_to_all` redistributes token activations so that each device receives the tokens assigned to its local experts.

- Input on each device: $[B, H]$ activation tensor (or the relevant sequence shard under tensor parallelism)
- After dispatch: each device receives $[T_d, H]$ where $T_d \approx B \cdot k \cdot E_d / E = B$ tokens under uniform routing
- Reshaped for batched computation: $[E_d, C, H] = [32, C, H]$

### Phase 2: Expert Compute

Each device runs its 32 local expert FFNs on the received token buffer. This is a batched matmul over local weights; see [program_configs_t3k.md](./program_configs_t3k.md) for config details.

### Phase 3: Token Combine

`ttnn.all_to_all` returns the expert output activations to the originating devices, where they are weighted and summed per the router probabilities.

### TTNN Code Example

```python
import ttnn

# --- Configuration ---
mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 8))
topology = ttnn.Topology.Linear
cluster_axis = 1
num_links = 1  # one Ethernet link per hop on T3K linear mesh

# tokens: [B, H] on each device (replicated or held by device 0 in pure EP)
# B=32, H=7168 for Qwen3.5-35B decode example
B, H = 32, 7168

# --- Phase 1: Dispatch ---
# dispatch_config maps each token to its target device(s) based on router output
dispatch_memory_config = ttnn.L1_MEMORY_CONFIG  # fits at decode; use DRAM for prefill

dispatched_tokens = ttnn.all_to_all(
    tokens,                          # [B, H] input activation
    num_links=num_links,
    memory_config=dispatch_memory_config,
    topology=topology,
    cluster_axis=cluster_axis,
)
# dispatched_tokens shape: [T_d, H] on each device, T_d ≈ B under uniform routing

# Reshape for expert-local batched matmul
E_d, C = 32, 2  # 32 local experts, capacity C=2 at B=32 (CF=1.25)
local_expert_batch = ttnn.reshape(dispatched_tokens, [E_d, C, H])
# Shape: [32, 2, 7168]

# --- Phase 2: Expert Compute (per-device) ---
# expert_weights: [E_d, H, D] in DRAM on this device
# sparsity_tensor: [E_d * M_t, K_t] uint8, constructed locally from dispatched_tokens
expert_outputs = ttnn.matmul(
    local_expert_batch,   # [32, 2, 7168]  (or sparse_matmul variant)
    expert_weights,       # [32, 7168, D]  [UNVERIFIED: D = FFN intermediate dim]
    program_config=per_chip_program_config,
    memory_config=ttnn.L1_MEMORY_CONFIG,
)
# expert_outputs shape: [32, 2, D]

# --- Phase 3: Combine ---
combine_memory_config = ttnn.L1_MEMORY_CONFIG

combined_outputs = ttnn.all_to_all(
    expert_outputs,          # [T_d, D] after reshape
    num_links=num_links,
    memory_config=combine_memory_config,
    topology=topology,
    cluster_axis=cluster_axis,
)
# combined_outputs shape: [B, D] on originating device; apply router weights and sum
```

> **Tip:** For prefill with large sequence lengths, use `memory_config=ttnn.DRAM_MEMORY_CONFIG` for both dispatch and combine. L1 may not hold the full $[T_d, H]$ buffer when $T_d$ is large.

---

## Section 3: Latency Model

Understanding where time is spent per MoE layer guides optimization decisions. The three phases contribute:

### Dispatch Latency

The all-to-all dispatch moves tokens from all devices to their target devices. On a 1×8 linear mesh, the data volume per Ethernet link is approximately:

$$V_{\text{dispatch}} = (N - 1) \cdot C \cdot E_d \cdot H \cdot 2 \text{ bytes}$$

At $B = 32$, $C = 2$, $E_d = 32$, $H = 7168$, BF16 (2 bytes):

$$V_{\text{dispatch}} = 7 \times 2 \times 32 \times 7168 \times 2 = 6,422,528 \text{ bytes} \approx 6.4 \text{ MB}$$

At ~12.5 GB/s per Ethernet link:

$$T_{\text{dispatch}} \approx \frac{6.4 \text{ MB}}{12.5 \text{ GB/s}} \approx 0.51 \text{ ms}$$

### Expert Compute Latency

Each device computes $E_d = 32$ expert FFNs on $T_d \approx B$ received tokens. The compute cost is proportional to $T_d \times H \times D$ [UNVERIFIED: D is the FFN intermediate dimension]. At small $B$ (decode), this term is small relative to the all-to-all cost.

### Combine Latency

The combine phase is symmetric with dispatch; $T_{\text{combine}} \approx T_{\text{dispatch}} \approx 0.51$ ms at $B = 32$.

### Total Per-Layer Latency Estimate

$$T_{\text{total}} \approx 2 \times T_{\text{dispatch}} + T_{\text{expert}} \approx 2 \times 0.51 + T_{\text{expert}}$$

At $B = 32$ decode, $T_{\text{expert}}$ is small (low arithmetic intensity), so:

$$T_{\text{total}} \approx 1.0+ \text{ ms per MoE layer}$$

The total inference latency is $\text{num\_layers} \times T_{\text{total}}$; the number of MoE layers in Qwen3.5-35B is [UNVERIFIED].

> **Warning:** The 6.4 MB / 12.5 GB/s estimate assumes the full volume passes over a single hop. The T3K linear topology requires multi-hop routing for devices far apart; actual latency is higher for the end-to-end all-to-all collective. Measure with `ttnn.all_to_all` benchmarks on your specific topology and `num_links` setting.

---

## Section 4: When EP Reduces Compute vs. When It Increases Communication

### EP Divides Expert FFN Compute

Without EP, a single device would need to compute all $E = 256$ expert FFNs (or at least all $k = 8$ active ones). With EP degree $N = 8$, each device computes only $E_d = 32$ experts — a $1/8$ reduction in expert-side FLOPs.

### EP Adds All-to-All Operations

EP introduces two all-to-all collectives per MoE layer. Their cost is independent of model sparsity: even if 0 tokens are routed to a device's experts, the collective still runs.

### The Break-Even Point

There is a batch size $B_{\text{cross}}$ where the FFN compute savings from EP equal the all-to-all overhead. Below $B_{\text{cross}}$, EP adds net overhead; above it, EP provides net savings.

However, for Qwen3.5-35B this analysis is largely academic: **EP is forced by memory, not chosen for speed.** The full expert weight set is too large to fit on a single Wormhole B0 device. EP is a memory partitioning requirement; the communication overhead is the cost of operating the model at all.

| Regime | Dominant cost | EP effect |
|---|---|---|
| Decode ($B \leq 32$) | All-to-all latency (~1 ms/layer) | Necessary; communication dominates |
| Prefill ($B \geq 256$) | Expert FFN compute | EP's $1/N$ compute reduction becomes meaningful |
| Memory budget | Expert weight DRAM | EP is mandatory for 35B+ models on T3K |

At decode ($B = 1$): the all-to-all cost (~0.26 ms × 2) dwarfs the expert compute time. The model cannot run on fewer devices due to memory constraints, so the communication overhead is unavoidable. The correct optimization response is to minimize per-call overhead (use `num_links=1`, place buffers in L1) and to maximize the efficiency of the expert compute phase through `sparse_matmul`.

---

## References

- Chapter 1: [routing_and_sparsity.md](../ch01_moe_architecture_fundamentals/routing_and_sparsity.md) — sparsity ratio $\rho$ definition
- Chapter 4: [when_sparse_matmul_wins.md](../ch04_sparse_matmul_for_moe/when_sparse_matmul_wins.md) — threshold analysis
- T3K guide: [collective_primitives.md](../../t3k_guide/ch02_ttnn_mesh_api/collective_primitives.md) — `ttnn.all_to_all` API reference
- Expert Parallelism guide, Chapter 4 — assignment and load-balancing strategies

---

**Next:** [sharding_strategies.md](./sharding_strategies.md)
