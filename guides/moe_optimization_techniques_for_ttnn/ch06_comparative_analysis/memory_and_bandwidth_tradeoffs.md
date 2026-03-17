# Memory and Bandwidth Tradeoffs

This file analyzes DRAM bandwidth pressure, L1 footprint, and multi-chip considerations for batched matmul and sparse matmul (SM) in Mixture-of-Experts (MoE) inference on Tenstorrent Wormhole B0. All analysis uses Qwen3.5-35B constants: $E = 256$ experts, $k = 8$ top-k, $H = 7{,}168$, $N = 8$ devices (T3K), $E_d = 32$ experts per device.

---

## 1. DRAM Bandwidth Pressure

### Wormhole B0 Memory System

Wormhole B0 provides 80 Tensix cores, each with 1.5 MB of private L1 SRAM, for 120 MB total aggregate L1. DRAM bandwidth is approximately 300 GB/s [UNVERIFIED]. The critical constraint for MoE operations is that both the activation tensor and expert weight tensors must be read from DRAM, and the gather phase (for batched matmul) adds a separate DRAM round-trip.

### Batched Matmul: Gather Phase Bandwidth Cost

Batched matmul requires a gather operation before the matmul: tokens must be moved from their original positions in the activation tensor into an expert-grouped layout of shape $[E,\, C,\, H]$, where $C$ is expert capacity.

At decode with $B=1$, $S=1$:

- $C = \lceil 8 \times 1 \times 1 / 256 \rceil = 1$ token/expert
- Expert-grouped tensor shape: $[256,\, 1,\, 7{,}168]$
- Total elements: $256 \times 1 \times 7{,}168 = 1{,}835{,}008$
- Active (non-padded) elements: $8 \times 7{,}168 = 57{,}344$ (for the 8 selected experts)
- Zero-padded elements: $248 \times 7{,}168 = 1{,}777{,}664$
- Useful data fraction: $57{,}344 / 1{,}835{,}008 = 3.1\%$

The gather reads the 1 input token and scatters it to 8 destination slots. The remaining 248 slots are zero-filled. When the downstream matmul reads this tensor from DRAM, it reads $1{,}835{,}008$ elements but only $57{,}344$ contribute to the computation — a **96.9% DRAM bandwidth waste** for the activation read.

At prefill with $B=32$, $S=2048$:

- $C = 2{,}048$ tokens/expert
- Expert-grouped tensor shape: $[256,\, 2{,}048,\, 7{,}168]$
- All 256 experts receive tokens — zero padding is negligible
- Gather moves $65{,}536$ real tokens; bandwidth utilization approaches $100\%$

This contrast explains why batched matmul's gather overhead is acceptable at prefill but wasteful at decode.

### Sparse Matmul: No Gather Required

Sparse matmul does not require a gather phase. The sparsity tensor encodes which tiles in the activation tensor are active; the kernel reads only those tiles directly:

- At decode $B=1$: 8 active experts × $\lceil H/32 \rceil = 224$ tiles per expert = $1{,}792$ active tiles read
- Skipped tiles: $248 \times 224 = 55{,}552$ tiles — never loaded from DRAM
- DRAM reads for the activation: $8 \times 7{,}168 = 57{,}344$ elements (only active rows)

The sparsity tensor itself is a small uint8 tensor; its DRAM footprint is addressed in Section 4. The key point is that sparse matmul's DRAM access for activations scales with $\rho$, not with the full tensor size.

> **Tip:** At decode with $B=1$ and $\rho = 3.1\%$, sparse matmul reads approximately $32\times$ fewer activation bytes from DRAM than batched matmul. This maps directly to a $32\times$ reduction in DRAM bandwidth for the activation load — a substantial advantage on a bandwidth-bound operation.

---

## 2. L1 Footprint Comparison

### Batched Matmul L1 Usage

Batched matmul uses circular buffers (CBs) in L1 to stage tiles for compute. The L1 footprint per core depends on the expert-batched activation block assigned to that core.

For prefill large ($B=32$, $S=2048$, $C=2{,}048$):

- The activation tensor $[256,\, 2{,}048,\, 7{,}168]$ is distributed across 80 Tensix cores
- Each core holds a subblock; the per-core L1 budget for the activation CB is roughly:
  $$\frac{256 \times 2{,}048 \times 7{,}168 \times 2\,\text{bytes}}{80\,\text{cores}} \approx 93\,\text{MB} / 80 \approx 1.16\,\text{MB/core}$$
- This approaches the per-core L1 limit of 1.5 MB, requiring careful subblock tiling to avoid L1 overflow

For decode small ($B=1$, $S=1$, $C=1$):

- Tensor shape: $[256,\, 1,\, 7{,}168]$ — much smaller in the sequence dimension
- Per-core activation CB: proportional to $C=1$ row per expert assigned to that core
- However, 248 of the 256 rows are zero-padding — L1 holds padded slots that never produce useful output

### Sparse Matmul L1 Usage

Sparse matmul only loads tiles that the sparsity tensor marks as active into L1:

- At decode $B=1$: 8 active experts out of 256 → L1 activation CB holds only $8 \times \lceil 7{,}168/32 \rceil = 8 \times 224 = 1{,}792$ active tiles
- Total activation data in L1: $1{,}792 \times 32 \times 32 \times 2\,\text{bytes} = 3{,}670{,}016\,\text{bytes} \approx 3.5\,\text{MB}$ across all cores
- Per-core: $3.5\,\text{MB} / 80 \approx 44\,\text{KB}$ — far below the 1.5 MB L1 limit

The sparsity tensor occupies additional L1:

- Shape: $[E_d \times M_t,\, K_t]$ uint8, where $M_t = \lceil C/32 \rceil$ and $K_t = \lceil H/32 \rceil = 224$
- At decode $B=1$: $M_t = \lceil 1/32 \rceil = 1$; shape = $[32 \times 1,\, 224] = [32,\, 224]$
- Size: $32 \times 224 \times 1\,\text{byte} = 7{,}168\,\text{bytes} \approx 7\,\text{KB}$
- This fits trivially in L1; it does not compete with the activation CB budget

At prefill large ($B=32$, $S=2048$):

- $M_t = \lceil 2{,}048/32 \rceil = 64$; sparsity tensor shape = $[32 \times 64,\, 224] = [2{,}048,\, 224]$
- Size: $2{,}048 \times 224 \times 1\,\text{byte} = 458{,}752\,\text{bytes} \approx 448\,\text{KB}$
- Still manageable in L1, but larger; and since $\rho \approx 1.0$ at prefill, this tensor encodes almost entirely "active" entries — no savings from using it

### L1 Summary

| Scenario | Approach | Activation CB (approx.) | Sparsity tensor | L1 pressure |
|----------|----------|------------------------|-----------------|-------------|
| Prefill, $B=32$, $S=2048$ | Batched matmul | ~1.16 MB/core | N/A | High (near limit) |
| Prefill, $B=32$, $S=2048$ | Sparse matmul | ~1.16 MB/core (all active) | 448 KB | Very high (not recommended) |
| Decode, $B=1$, $S=1$ | Batched matmul | Padded: wastes 96.9% | N/A | Low (but wasteful) |
| Decode, $B=1$, $S=1$ | Sparse matmul | ~44 KB/core (active only) | 7 KB | Very low |

---

## 3. T3K Multi-Chip Considerations

### Expert Parallelism Layout

On T3K (8 devices), Qwen3.5-35B distributes its $E = 256$ experts evenly: $E_d = 256 / 8 = 32$ experts per device. During inference, tokens must be routed to the devices holding their selected experts via all-to-all communication.

### All-to-All Overhead Is Shared

A key observation: **both batched matmul and sparse matmul face the same all-to-all communication cost**. The choice of matmul strategy does not affect how many tokens are sent between devices. The all-to-all volume depends only on the routing distribution, not on what happens after tokens arrive at their destination device.

This means the matmul strategy decision can be made independently of the communication topology. The all-to-all overhead is a fixed cost subtracted equally from both approaches' effective throughput.

### Sparse Matmul on T3K: Per-Device Sparsity Tensor

After the all-to-all dispatch, each device knows which of its 32 local experts received tokens. Each device constructs its own local sparsity tensor independently:

- Shape per device: $[E_d \times M_t,\, K_t] = [32 \times M_t,\, 224]$ uint8
- At decode $B=1$: on the device holding the 8 selected experts (or a subset thereof), $M_t = 1$; sparsity tensor = $[32,\, 224]$ = 7 KB
- On devices whose 32 local experts received no tokens: the sparsity tensor is all-zeros; the matmul is entirely skipped

The per-device sparsity tensor construction requires no cross-device coordination. Each device independently inspects its received token assignments and sets the corresponding bits.

> **Tip:** On T3K with $B=1$ decode and $\rho = 3.1\%$, approximately $8/256 = 3.1\%$ of device-expert slots receive tokens. Under uniform routing, most devices receive between 0 and 4 tokens for their 32 experts. Devices receiving 0 tokens skip their matmul entirely when using sparse matmul — a benefit not available with batched matmul (which processes the empty expert-grouped tensor).

### Batched Matmul on T3K: Local Gather

With expert parallelism, batched matmul's gather phase is local to each device. After all-to-all dispatch, each device gathers its received tokens into a local expert-grouped layout $[E_d,\, C,\, H] = [32,\, C,\, 7{,}168]$. The gather operates over only $E_d = 32$ experts rather than all 256, reducing the per-device gather cost proportionally.

At decode $B=1$: at most 8 tokens arrive across all 8 devices. On average, each device receives 1 token for its 32 experts. The gather layout is $[32,\, 1,\, 7{,}168]$ per device; 31 of 32 rows are zero-padded — still a 96.9% DRAM waste at the per-device level.

---

## 4. Sparsity Tensor Construction Overhead

### Construction Cost

Sparsity tensor construction is $O(B \times k)$ work: for each token in the batch, $k = 8$ expert indices must be recorded. At $B=32$, $k=8$: $256$ write operations. At $B=1$: $8$ write operations. This is trivially small compared to the matmul compute.

For context: a single tile matmul on Wormhole B0 involves $32 \times 32 \times 32 = 32{,}768$ multiply-accumulate operations. Constructing the sparsity tensor involves $256$ integer writes — three orders of magnitude less work.

### Transfer vs. Device-Side Construction

The sparsity tensor can be constructed on the CPU and transferred to the device, or constructed directly on-device from the router output. For the small tensors involved in decode ($\leq 7\,\text{KB}$), device-side construction is preferred to avoid a PCIe transfer on the critical path. See `ch05_sparsity_tensor_construction/constructing_from_router_output.md` for the full construction procedure.

### Must Update Every Decode Step

> **Warning:** The sparsity tensor MUST be rebuilt at every decode step. The router selects different experts for each new token; reusing a stale sparsity tensor produces silent correctness errors (the wrong expert rows are skipped or computed). There is no valid caching of the sparsity tensor across decode steps.

The $O(B \times k)$ rebuild cost is negligible compared to the matmul, so the no-reuse requirement does not impose a meaningful overhead.

---

## References

- `ch03_batched_matmul_for_moe/performance_profile_batched.md` — batched matmul gather cost and DRAM bandwidth analysis
- `ch04_sparse_matmul_for_moe/when_sparse_matmul_wins.md` — sparsity ratio and bandwidth savings derivation
- `ch04_sparse_matmul_for_moe/sparse_matmul_internals.md` — tile-skip mechanics and L1 CB management
- `ch05_sparsity_tensor_construction/sparsity_tensor_placement.md` — sparsity tensor L1 sizing and placement
- `ch05_sparsity_tensor_construction/constructing_from_router_output.md` — device-side construction procedure

## Next Steps

Proceed to `decision_guide.md` for structured decision rules, runtime sparsity measurement utilities, the hybrid strategy implementation pattern, and anti-patterns to avoid.
