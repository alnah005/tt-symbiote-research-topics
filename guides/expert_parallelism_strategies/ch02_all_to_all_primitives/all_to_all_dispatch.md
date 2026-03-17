# All-to-All Dispatch

## Purpose

This file specifies the semantics of `all_to_all_dispatch` — the TTNN (Tenstorrent tensor library) operation that routes token embeddings from their originating devices to the devices that own the target experts. It covers the pre-dispatch steps needed to construct well-formed input buffers, the distinction between sparse and dense packing, kernel-level constraints on Tenstorrent hardware, and a concrete worked example.

**Prerequisite:** `collective_communication_background.md` (this chapter) for the definition of all-to-all; Chapter 1, `routing_problem.md` for the dispatch/combine communication pattern.

---

## Semantics of `all_to_all_dispatch`

### Conceptual Contract

`all_to_all_dispatch` takes as input a set of per-device token buffers and routing metadata, and produces as output a set of per-device received-token buffers, one per source device, organized for direct consumption by local expert FFNs.

On each device $d$, the operation performs the following:

1. **Read send buffers.** For each destination device $d' \neq d$, read the pre-packed buffer of token embeddings that device $d$ is sending to device $d'$. The size of each send buffer is $C \times H$ bytes, where $C$ (see Ch. 7) is the expert capacity and $H = 7168$ is the hidden dimension.

2. **Execute all-to-all.** Transmit send buffer $[d \to d']$ to device $d'$ for all $d' \neq d$, while simultaneously receiving send buffer $[d'' \to d]$ from each $d'' \neq d$.

3. **Write receive buffers.** Assemble the received token embeddings into a contiguous receive buffer of shape $[C \times E_d, H]$, where $E_d = E/N = 32$ experts per device under uniform assignment. Tokens for expert $e$ on device $d$ occupy slots $[(e \bmod E_d) \times C : (e \bmod E_d + 1) \times C]$ in the receive buffer.

### TTNN Signature (Illustrative)

The following Python pseudocode illustrates the logical signature. Exact argument names and types are subject to the TTNN API version in use; consult the TTNN documentation for the authoritative interface.

```python
received_tokens = ttnn.all_to_all_dispatch(
    tokens,           # Tensor [B, H]: local token batch on this device
    expert_indices,   # Tensor [B, k]: routing indices (expert IDs per token)
    expert_capacity,  # int: C, max tokens per expert per forward pass
    num_experts,      # int: E = 256, total experts across all devices
    num_devices,      # int: N = 8
    dtype,            # ttnn.bfloat16 or equivalent
)
# Returns: Tensor [C * E_d, H] of tokens dispatched to this device's experts
# E_d = num_experts / num_devices = 32
```

The operation encapsulates both the pre-dispatch packing (computing send counts, building packed send buffers) and the collective transfer. The details of packing are described in the next two sections.

---

## Pre-Dispatch Steps

Before the all-to-all transfer can execute, each device must know how many tokens to send to each destination device and must pack those tokens into contiguous send buffers. These pre-dispatch steps are logically separate from the collective itself, though TTNN implementations may fuse them.

### Step 1: Computing Send Counts Per Device

Given the routing index tensor of shape $[B, k]$ (where each entry is an expert index in $\{0, \ldots, E-1\}$), each device computes a **send-count vector** of length $N$ indicating how many token slots to send to each destination device.

The mapping from expert index to device index is:

$$\text{device}(e) = \lfloor e / E_d \rfloor = \lfloor e / 32 \rfloor$$

under uniform assignment. For each token $t \in \{0, \ldots, B-1\}$ and each of its $k$ selected experts $e_{t,j}$ ($j = 0, \ldots, k-1$), the destination device is $\text{device}(e_{t,j})$. The send count for device $d'$ is:

$$\text{send\_count}[d'] = \sum_{t=0}^{B-1} \sum_{j=0}^{k-1} \mathbf{1}[\text{device}(e_{t,j}) = d']$$

Note that if both of a token's selected experts reside on the same remote device (which occurs with probability $\binom{E_d}{2}/\binom{E}{2}$ under uniform sampling without replacement), that token's embedding is sent to that device once per expert slot, and the device receives two copies of the embedding — one for each expert. The embedding is not deduplicated, because each expert processes it independently.

In pseudocode:

```python
def compute_send_counts(
    expert_indices: list,   # shape [B, k], values in {0, ..., E-1}
    E_d: int,               # experts per device = E / N
    N: int,                 # number of devices
) -> list:
    """
    Returns send_counts[d'] = number of token slots to send to device d'.
    """
    send_counts = [0] * N
    for token_experts in expert_indices:     # iterate over B tokens
        for expert_id in token_experts:      # iterate over k experts
            dest_device = expert_id // E_d
            send_counts[dest_device] += 1
    return send_counts
```

The sum of all send counts equals $B \times k$ (every token-expert pair is assigned to exactly one device):

$$\sum_{d'=0}^{N-1} \text{send\_count}[d'] = B \times k$$

### Step 2: Padding to Expert Capacity

Send counts are not transmitted directly to the all-to-all; instead, the send buffer for each destination device is padded to exactly $C \times E_d$ slots. Padding to a fixed capacity $C$ per expert slot serves two purposes:

1. **Static buffer sizing.** The all-to-all collective operates on fixed-size messages. If actual token counts per expert varied message-by-message, the collective would need to negotiate sizes, adding overhead. Fixed-size messages allow the collective to be pre-allocated and issued without a handshake.

2. **Compatibility with expert FFN tiling.** The expert FFN kernel on the receiving device expects a fixed-size input buffer of $[C, H]$ per expert, allowing the kernel's tile loop bounds to be compiled as constants.

Padding inserts zero-valued token embeddings (or a sentinel value) into unused capacity slots. The receiving device's expert FFN processes these zero-padded slots but their outputs are discarded (masked out) during the combine step, so padding does not affect model output.

The padded send buffer for destination device $d'$ has shape $[C \times E_d, H]$: $E_d$ expert slots each with $C$ token positions. Only the first $\text{actual\_count}[d', e]$ positions in expert $e$'s slot are populated with real token data; the remainder are zero.

---

## Sparse vs. Dense Packing

### The Packing Problem

Token embeddings arrive at the router in their original order in the batch (token 0, token 1, ..., token $B-1$). After routing, the $k$ expert assignments for each token scatter those embeddings across $N$ possible destination devices. A naive implementation would leave each token's embedding in its original batch-order location and transmit the full $[B, H]$ tensor to every device, relying on the receiver to find the relevant tokens.

This naive approach — equivalent to performing a broadcast and discarding irrelevant tokens — wastes bandwidth and cache. **Packing** solves this by reordering and compacting tokens into destination-sorted contiguous buffers before the collective.

### Dense Packing

In **dense packing**, each device prepares $N$ send buffers, one per destination device, each of shape $[C \times E_d, H]$. Token embeddings are copied from their batch-order locations into the appropriate position within the destination buffer, based on:

- Which device owns the target expert (determines which of the $N$ send buffers to write into).
- Which expert on that device the token is targeting (determines which expert slot within the send buffer).
- The slot index within that expert's capacity window (determined by a per-expert counter).

After packing, the send buffer for device $d'$ contains, at position $[e_\text{local} \times C + \text{slot}, :]$ the embedding of the $\text{slot}$-th token routed to expert $e' = d' \times E_d + e_\text{local}$.

The send buffer can be visualized as:

```text
Send buffer for device d' (shape [C * E_d, H]):

  rows 0        ..  C-1:         tokens for expert  d'*E_d + 0
  rows C        ..  2C-1:        tokens for expert  d'*E_d + 1
  ...
  rows (E_d-1)*C .. E_d*C-1:    tokens for expert  d'*E_d + (E_d-1)
```

### Slot Metadata

Dense packing requires **slot metadata**: a record of which original token index and routing weight corresponds to each slot in each send buffer. This metadata must be transmitted alongside (or in advance of) the token embeddings so that the receiving device can construct the correct combine operations later. In practice, the metadata is often transmitted as a separate, small all-to-all of integer index tensors immediately before or fused with the embedding dispatch.

The slot metadata for each token-expert assignment consists of:
- `token_id`: the original token index $t \in \{0, \ldots, B-1\}$ on the originating device
- `routing_weight`: the renormalized weight $\hat{w}_{t,j}$ for this expert assignment

Both fields are necessary for the combine step to reconstruct the weighted sum at the originating device.

### Scatter Indices

An efficient implementation of the packing step uses a **scatter index tensor** of shape $[B \times k, 3]$, where each row contains `(dest_device, expert_local_idx, slot_idx)` for one token-expert assignment. The scatter operation reads each token embedding from its batch-order location and writes it to the specified position in the appropriate send buffer.

On Tenstorrent hardware, this scatter is implemented as a gather from the token tensor followed by a write into the packed buffer, or as a direct core-to-core routing where each compute core writes into the L1 buffer destined for the Ethernet DMA engine.

---

## Kernel-Level Considerations on Tenstorrent Hardware

### Tile Alignment Requirements

Tenstorrent's compute cores operate on tiles — fixed-size 2D blocks of elements. The standard tile size is $32 \times 32$ elements. For BF16 data, one tile occupies $32 \times 32 \times 2 = 2048$ bytes.

For the dispatch operation to achieve peak memory bandwidth, the token embedding dimension $H$ and the batch dimension of each send buffer must be tile-aligned:

- $H = 7168$ must be a multiple of 32. $7168 / 32 = 224$. This holds exactly — there are 224 tiles per row.
- The number of token slots per expert $C$ must be a multiple of 32 for the send buffer rows to be tile-aligned. For capacity factor $CF = 1.0$ and an expected load of $B \times k / E$ tokens per expert, $C$ must be rounded up to the nearest multiple of 32.

When $C$ is not naturally a multiple of 32, zero-padding is applied to reach the next multiple. This adds a small amount of additional communication volume (at most $31 \times H \times \text{dtype\_bytes} = 31 \times 7168 \times 2 = 444{,}416$ bytes per expert, across 32 experts per device per direction) but is required for correct kernel execution.

### Shard Layout on Tenstorrent Cores

The T3K's compute grid is organized as a 2D array of Tensix cores. For the dispatch packing kernel, the token batch $[B, H]$ is typically sharded across cores along the batch dimension: each core handles a contiguous range of tokens and writes its assigned tokens into the appropriate positions in the send buffers.

The send buffers themselves are allocated in the device's DRAM and streamed into L1 (the on-chip SRAM) for the Ethernet DMA engine. The DMA engine reads from L1 and transmits over the Ethernet link. To avoid stalls, the packing kernel must complete writing each send buffer into L1 before the DMA engine needs to read it. Double-buffering (writing the next send buffer into one L1 partition while the DMA engine reads the current send buffer from another) is the standard technique for hiding this latency.

Tile alignment in the batch dimension also affects the packing kernel: if $B$ is not a multiple of 32, the last tile row of each expert's send buffer will be partially filled. The kernel must zero-initialize these partial tiles before writing, to avoid transmitting stale data in the padding region.

### Expert Capacity and L1 Footprint

The receive buffer on the destination device has shape $[C \times E_d, H]$. For $C = 16$ (one example value; the actual $C$ depends on the capacity factor and batch size), $E_d = 32$, and $H = 7168$ at BF16:

$$\text{Receive buffer size} = 16 \times 32 \times 7168 \times 2 = 7{,}340{,}032 \text{ bytes} \approx 7 \text{ MiB}$$

This example uses $C = 16$, which is illustrative; production capacity values are typically larger (e.g., $C = 32$ or higher depending on load), giving receive buffers of 14 MiB or more. Even at $C = 16$, the 7 MiB receive buffer exceeds a single Tensix core's L1 (~1.5 MB per core), so the buffer must be held in DRAM and expert FFN kernels must stream tiles from DRAM into L1 for compute, processing each expert's token batch in tiles. Detailed treatment of expert FFN tiling is in Chapter 6, `ch06_fused_dispatch_compute_combine/expert_ffn_tiling.md`.

---

## Worked Example: 4-Device Case, 8 Tokens, Top-2 Routing

This example uses $N = 4$ devices, $B = 8$ tokens per device (32 tokens total), $k = 2$ top experts per token, and $E = 8$ experts (2 per device). This simplification makes all quantities small enough to trace by hand. The T3K configuration ($N = 8$, $E = 256$, $k = 8$) follows the same logic at larger scale.

### Setup

Expert-to-device assignment (uniform, $E_d = 2$):

| Device | Local experts |
|---|---|
| 0 | 0, 1 |
| 1 | 2, 3 |
| 2 | 4, 5 |
| 3 | 6, 7 |

Capacity: $C = 4$ tokens per expert (chosen to fit this example; under uniform routing with $B = 8$ and $k = 2$, the expected load is $8 \times 2 / 8 = 2$ tokens per expert, so $C = 4$ gives capacity factor $CF = 2.0$).

### Routing Decisions on Device 0

Suppose device 0 holds tokens $t_0, t_1, t_2, t_3, t_4, t_5, t_6, t_7$ and the router produces the following expert assignments:

| Token | Expert 1 | Expert 2 | Dest Device 1 | Dest Device 2 |
|---|---|---|---|---|
| $t_0$ | 0 | 3 | 0 (local) | 1 |
| $t_1$ | 2 | 6 | 1 | 3 |
| $t_2$ | 1 | 5 | 0 (local) | 2 |
| $t_3$ | 4 | 7 | 2 | 3 |
| $t_4$ | 0 | 2 | 0 (local) | 1 |
| $t_5$ | 3 | 5 | 1 | 2 |
| $t_6$ | 6 | 1 | 3 | 0 (local) |
| $t_7$ | 7 | 4 | 3 | 2 |

### Send Counts on Device 0

Counting destination devices for all $8 \times 2 = 16$ token-expert pairs:

- Device 0 (self): $t_0 \to e_0$, $t_2 \to e_1$, $t_4 \to e_0$, $t_6 \to e_1$ — **4 slots** (local, no network transfer)
- Device 1: $t_0 \to e_3$, $t_1 \to e_2$, $t_4 \to e_2$, $t_5 \to e_3$ — **4 slots**
- Device 2: $t_2 \to e_5$, $t_3 \to e_4$, $t_5 \to e_5$, $t_7 \to e_4$ — **4 slots**
- Device 3: $t_1 \to e_6$, $t_3 \to e_7$, $t_6 \to e_6$, $t_7 \to e_7$ — **4 slots**

Send counts: `[4, 4, 4, 4]`. Under uniform routing for this toy example, every device receives exactly 4 token slots from device 0.

### Packing the Send Buffer for Device 1

The send buffer for device 1 has shape $[C \times E_d, H] = [4 \times 2, H] = [8, H]$. The local experts on device 1 are experts 2 and 3 ($e_\text{local} = 0$ for expert 2, $e_\text{local} = 1$ for expert 3).

Expert 2 receives: $t_1$, $t_4$. These go into slots 0 and 1 of the send buffer (rows 0 and 1).
Expert 3 receives: $t_0$, $t_5$. These go into slots $C + 0 = 4$ and $C + 1 = 5$ of the send buffer (rows 4 and 5).
Rows 2, 3 (remaining $C - 2 = 2$ slots for expert 2) and rows 6, 7 (remaining slots for expert 3) are zero-padded.

Send buffer for device 1 from device 0 (rows denote token slots):

```text
Row 0:  embedding of t_1    [H floats]  -> for expert 2, slot 0
Row 1:  embedding of t_4    [H floats]  -> for expert 2, slot 1
Row 2:  [zeros]                          -> expert 2 padding
Row 3:  [zeros]                          -> expert 2 padding
Row 4:  embedding of t_0    [H floats]  -> for expert 3, slot 0
Row 5:  embedding of t_5    [H floats]  -> for expert 3, slot 1
Row 6:  [zeros]                          -> expert 3 padding
Row 7:  [zeros]                          -> expert 3 padding
```

Slot metadata transmitted alongside (as a separate integer buffer):

```text
Row 0:  (token_id=1, local_expert=0, routing_weight=w_{t_1, e_2})
Row 1:  (token_id=4, local_expert=0, routing_weight=w_{t_4, e_2})
Row 2:  (token_id=-1, ...)   # padding sentinel
Row 3:  (token_id=-1, ...)   # padding sentinel
Row 4:  (token_id=0, local_expert=1, routing_weight=w_{t_0, e_3})
Row 5:  (token_id=5, local_expert=1, routing_weight=w_{t_5, e_3})
Row 6:  (token_id=-1, ...)   # padding sentinel
Row 7:  (token_id=-1, ...)   # padding sentinel
```

### All-to-All Transfer

After all 4 devices have prepared their send buffers, the all-to-all collective executes. Device 1 simultaneously:
- Sends its send buffer for device 0 to device 0
- Sends its send buffer for device 2 to device 2
- Sends its send buffer for device 3 to device 3
- Receives from devices 0, 2, and 3

After the collective, device 1's receive buffer (shape $[C \times E_d, H] = [8, H]$) contains:

```text
Rows 0..3:   tokens for expert 2, sourced from all sending devices
Rows 4..7:   tokens for expert 3, sourced from all sending devices
```

Specifically, rows 0–3 will contain the token embeddings from device 0 (rows 0–1 of device 0's send buffer) plus those from device 2 and device 3 that are also targeting expert 2. The exact assembly order within each expert's $C = 4$ slots depends on the collective's receive ordering, which is defined by the TTNN implementation and must be consistent with the slot metadata.

### Expert FFN Execution on Device 1

Device 1 runs expert 2 on rows 0–3 and expert 3 on rows 4–7 of its receive buffer. Rows corresponding to padding sentinels are either skipped (if the kernel supports masking) or computed and discarded in the combine step.

The output buffer after expert FFN has shape $[C \times E_d, H] = [8, H]$ — the same shape as the receive buffer. The slot metadata is carried alongside to the combine step.

---

## References

- [Lepikhin2021] Lepikhin, D. et al., "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding", ICLR, 2021.
- [Fedus2022] Fedus, W., Zoph, B., Shazeer, N., "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity", JMLR, 2022.
- [Rajbhandari2022] Rajbhandari, S. et al., "DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale", ICML, 2022.
- [Singh2023] Singh, S. et al., "Designing Efficient LLM Accelerators for Server Workloads", arXiv:2312.02207, 2023.
- [TTNNDocs] Tenstorrent, "TTNN API Reference", Tenstorrent Developer Documentation.
- [Ch1Routing] Chapter 1, `ch01_moe_fundamentals/routing_problem.md` — dispatch/combine pattern and cross-device volume derivation.
- [Ch2Background] Chapter 2, `collective_communication_background.md` — all-to-all formal definition and bandwidth model.
- [Ch2Combine] Chapter 2, `all_to_all_combine.md` — inverse operation: routing expert outputs back to originating devices.
- [Ch6FFNTiling] Chapter 6, `ch06_fused_dispatch_compute_combine/expert_ffn_tiling.md` — expert FFN kernel tiling and L1 management.
- [Ch7Capacity] Chapter 7, `ch07_load_balancing/capacity_factor_mechanics.md` — formal definition of expert capacity $C$ and capacity factor $CF$.

---

**Next:** [all_to_all_combine.md](./all_to_all_combine.md)
