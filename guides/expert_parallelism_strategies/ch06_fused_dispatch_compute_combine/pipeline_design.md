# Pipeline Design

## Section 1: Six Stages of a Single MoE Layer

Each MoE layer in Qwen3.5-35B executes the following six stages in strict order.
Token batches of size $B$ enter Stage 1 and exit Stage 6 as updated residual-stream
embeddings of dimension $H = 7168$.

```
Stage 1  Route          router projection + sigmoid + top-k
Stage 2  Pack/Dispatch  scatter embeddings into send buffer; all-to-all dispatch
Stage 3  Expert FFN     local matmul on received tokens for each of E_d=32 experts
Stage 4  Combine        all-to-all combine (return outputs to originating device)
Stage 5  Unpack+Accum   apply routing weights; sum k=8 expert outputs per token
Stage 6  Residual Add   output += input residual stream
```

**ASCII data flow diagram:**

```
  [B, H] tokens
      |
      v
+------------+   indices [B, k]
|  Stage 1   |   scores  [B, k]
|   Route    |-----------------------------+
+------------+                             |
      |                                    |
      v (indices)                          |
+------------+   all-to-all dispatch       |
|  Stage 2   |-->  [C*E_d, H] per dev      |
| Pack/Disp  |                             |
+------------+                             |
      |                                    |
      v                                    |
+------------+   [T_received, H] outputs   |
|  Stage 3   |   for each local expert     |
| Expert FFN |                             |
+------------+                             |
      |                                    |
      v                                    |
+------------+   all-to-all combine        |
|  Stage 4   |-->  [B, k, H] per token     |
|  Combine   |                             |
+------------+                             |
      |                                    |
      v                                    |
+------------+  <--------------------------+
|  Stage 5   |   weighted scatter-add
| Unpack+Acc |   output[b] = sum_j w_j * expert_out[b,j]
+------------+
      |
      v
+------------+
|  Stage 6   |   output += residual
| Resid Add  |
+------------+
      |
      v
  [B, H] updated tokens
```

## Section 2: Dependency Graph

The stages form a linear chain with no divergent branches. Each stage consumes the
output of exactly the stage before it:

| Producer Stage | Output | Consumer Stage |
|---|---|---|
| Stage 1 (Route) | `indices [B, k]`, `scores [B, k]` | Stage 2 (Pack/Dispatch) and Stage 5 (scores held locally) |
| Stage 2 (Pack/Dispatch) | `recv_buf_e [received_count_e, H]` per expert per device | Stage 3 (Expert FFN) |
| Stage 3 (Expert FFN) | `expert_out_e [received_count_e, H]` per expert | Stage 4 (Combine) |
| Stage 4 (Combine) | `[B, k, H]` logical buffer on originating device | Stage 5 (Unpack+Accum) |
| Stage 5 (Unpack+Accum) | `[B, H]` weighted sum | Stage 6 (Residual Add) |
| Stage 6 (Residual Add) | `[B, H]` updated embeddings | Next MoE layer or final projection |

**Which stages can overlap across different micro-batches:**

Because Stages 2 and 4 are both all-to-all collectives over separate buffers, and
Stage 1 touches only router parameters (not the dispatch/combine buffers), routing of
micro-batch $i+1$ can execute concurrently with any of Stages 2–5 of micro-batch $i$,
provided the two micro-batches use independent buffer sets.

There is no intra-stage parallelism between Stage 3 and Stage 4 for the same
micro-batch: combine (Stage 4) cannot begin until every expert FFN in Stage 3 has
produced its output, because the combine all-to-all sends exactly those outputs back
to originating devices.

## Section 3: Micro-Batch Pipelining

### Motivation

At decode batch sizes ($B \leq 32$), Stage 1 (Route) is fast: it is a single linear
projection of shape $[B, H] \times [H, E]$ followed by a top-$k$ selection. The
total time $T_{\text{route}}$ is small relative to $T_{\text{dispatch}} +
T_{\text{expert}} + T_{\text{combine}}$. By splitting $B$ tokens into two
micro-batches of $B/2$ tokens each and staggering their execution, $T_{\text{route}}$
for micro-batch $i+1$ is hidden behind the slower stages of micro-batch $i$.

### Pipeline Diagram (2 micro-batches)

```
Time -->
                         T_route    T_disp    T_expert  T_comb   T_accum
Micro-batch 0:  [Route0 |Disp0    |Expert0  |Comb0   |Accum0 ]
Micro-batch 1:           [Route1  |Disp1    |Expert1 |Comb1  |Accum1]
                          ^
                          Route1 overlaps with Disp0+Expert0+Comb0
```

With double-buffering, micro-batch 1's Route stage executes on the CPU/router cores
while micro-batch 0's tokens are in transit or being processed by expert FFNs. The
pipeline delivers one completed micro-batch every
$\max(T_{\text{route}}, T_{\text{dispatch}} + T_{\text{expert}} + T_{\text{combine}})$
cycles, plus $T_{\text{accum}}$ which is not overlapped.

### Condition for Benefit

Double-buffering reduces observed latency whenever:

$$T_{\text{route}} < T_{\text{dispatch}} + T_{\text{expert}} + T_{\text{combine}}$$

At decode ($B = 32$): $T_{\text{dispatch}} \approx 0.51$ ms (see
`end_to_end_latency_model.md`), while $T_{\text{route}}$ for 32 tokens through a
$[7168, 256]$ projection is a small fraction of that. The condition holds comfortably.

At large prefill batch sizes, $T_{\text{expert}}$ grows linearly with $B$, making the
right-hand side even larger. The condition holds there as well.

> **Tip:** The micro-batch split need not be exactly $B/2$. An unequal split can be
> used to balance Stage 3 compute time across micro-batches if load imbalance is
> observed at a particular batch size.

## Section 4: Buffer Requirements

### Per Micro-Batch Buffer Layout

Each micro-batch of size $B/2$ uses the following buffers:

- **Dispatch send buffer** (one per destination device): shape $[C \times E_d, H]$
  where $C = \lceil k \times B \times \text{CF} / E \rceil = \lceil B \times 1.25 / 32 \rceil$ and $E_d = 32$, $H = 7168$.
- **Dispatch receive buffer**: same shape $[C \times E_d, H]$ (tokens arriving from other devices).
- **Combine send buffer**: same shape $[C \times E_d, H]$ (expert outputs to send back).
- **Combine receive buffer** (one per source device): same shape $[C \times E_d, H]$.

### Double-Buffering Overhead

For double-buffering, two complete sets of dispatch and combine buffers are required
simultaneously — one for micro-batch $i$ currently in Stages 2–4, and one for
micro-batch $i+1$ beginning Stage 2.

Total double-buffer memory per device:

$$M_{\text{double}} = 2 \times 4 \times C \times E_d \times H \times 2 \text{ bytes}$$

The factor of 2 (outermost) is for the two pipeline stages in flight simultaneously;
the factor of 4 accounts for the four buffers per stage (dispatch-send, dispatch-receive,
combine-send, combine-receive); the trailing 2 bytes is BF16 element size.

At $B = 32$, $C = 2$:

$$M_{\text{double}} = 8 \times 2 \times 32 \times 7168 \times 2 = 7{,}340{,}032 \text{ bytes} \approx 7.0 \text{ MB total}$$

Individual buffer shape: $[C \times E_d, H] = [64, 7168]$; size per buffer:
$64 \times 7168 \times 2 = 917{,}504$ bytes $\approx 896$ KB.

> **Warning:** 7.0 MB exceeds the L1 capacity of a single Wormhole B0 core (1.5 MB)
> but is within the aggregate L1 of all 80 cores (120 MB). Buffers must be distributed
> across cores using height-sharding. Refer to the TTNN sharding documentation for
> `ttnn.TensorMemoryLayout.HEIGHT_SHARDED`.

### Buffer Size Summary

| Parameter | $B=1$ | $B=4$ | $B=32$ |
|---|---|---|---|
| $C$ | 1 | 1 | 2 |
| Single buffer $[C \cdot E_d, H]$ | $[32, 7168]$ | $[32, 7168]$ | $[64, 7168]$ |
| Single buffer size | 448 KB | 448 KB | 896 KB |
| Total double-buffer $M_{\text{double}}$ | 3.5 MB | 3.5 MB | 7.0 MB |

### Python Reference: Buffer Allocation

```python
import math
import ttnn

E_d = 32       # local experts per device
H = 7168       # hidden dimension
CF = 1.25      # capacity factor
dtype = ttnn.bfloat16

def capacity(B: int) -> int:
    """Compute per-device token capacity C for a micro-batch of size B."""
    return math.ceil(B * CF / E_d)

def dispatch_buffer_shape(B: int):
    C = capacity(B)
    return (C * E_d, H)   # [C*E_d, H]

def double_buffer_bytes(B: int) -> int:
    """Total bytes for double-buffered dispatch+combine buffers (BF16)."""
    C = capacity(B)
    single = C * E_d * H * 2   # 2 bytes per BF16 element
    return 8 * single           # 2 pipeline stages × 4 buffers each (dispatch-send/recv, combine-send/recv)

# Example: B=32
B = 32
print(f"C = {capacity(B)}")
print(f"Dispatch buffer shape: {dispatch_buffer_shape(B)}")
print(f"Double-buffer total: {double_buffer_bytes(B) / 1024 / 1024:.2f} MB")
# C = 2
# Dispatch buffer shape: (64, 7168)
# Double-buffer total: 7.00 MB
```

## References

- Chapter 2 of this guide: `ch02_all_to_all_primitives/all_to_all_dispatch.md`,
  `ch02_all_to_all_primitives/all_to_all_combine.md`.
- Chapter 4 of this guide: `ch04_expert_device_assignment/uniform_partitioning.md`
  (capacity factor derivation and $W_{\text{expert}}$).
- Chapter 5 of this guide: `ch05_routing_weight_optimization/router_kernel_fusion.md`
  (double-buffering mechanics).
- TT-Metalium TTNN documentation: `ttnn.TensorMemoryLayout.HEIGHT_SHARDED`.
