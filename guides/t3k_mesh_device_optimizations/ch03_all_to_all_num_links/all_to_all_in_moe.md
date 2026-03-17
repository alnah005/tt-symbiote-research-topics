# All-to-All in MoE: Dispatch, Expert Compute, and Combine

> **Quick Reference — Symbols Used in This File**
>
> | Symbol | Value (Qwen3.5-35B) | Notes |
> |---|---|---|
> | $N$ | 8 | Number of T3K devices |
> | $k$ | 8 | Top-K experts per token [D UNVERIFIED] |
> | $E$ | 256 | Total experts [D UNVERIFIED] |
> | $H$ | 7168 | Hidden dimension [D UNVERIFIED] |
> | $D$ | — | Expert FFN intermediate dimension [D UNVERIFIED] |
> | $B$ | varies | Batch size |
> | $S$ | varies | Sequence length |
> | $C$ | $\lceil k \times B \times S / E \rceil$ | Expert capacity |

Every MoE layer in a Mixture-of-Experts transformer involves three sequential phases: token routing (dispatch), expert computation, and output aggregation (combine). On T3K, the dispatch and combine phases are each implemented as a `ttnn.all_to_all` collective. This file analyzes each phase, derives the data volumes transferred, and characterizes how those volumes differ between the prefill and decode regimes.

---

## Token Routing Phase (Dispatch)

### What Happens

After the router computes top-K expert assignments for each token, the token embeddings must be sent to the devices that hold the selected experts. With $N = 8$ devices each holding $E/N = 256/8 = 32$ experts, a token whose top-K experts are spread across all devices must reach up to $N - 1 = 7$ remote devices.

For Qwen3.5-35B with $k = 8$ and $N = 8$, each token nominally selects exactly 8 experts. If those 8 experts are distributed uniformly across the 8 devices (one expert per device), then each token is sent to all 7 other devices plus retained on its own originating device — a full broadcast pattern. In practice routing is not perfectly uniform, but the full-broadcast case is the design-point for capacity planning.

Before calling `ttnn.all_to_all`, tokens must be sorted into groups by their destination device. Each group of tokens addressed to device $d$ is placed at position $d$ in the send buffer. The all-to-all then delivers each group to its target device.

```
Device 0                  Device 1   ...   Device 7
┌───────────────────────┐
│ tokens → device 0 (local retained)        │
│ tokens → device 1 ────────────────► d1    │
│ tokens → device 2 ──────────────────────► d2
│  ...                                      │
│ tokens → device 7 ─────────────────────────────► d7
└───────────────────────┘
```

Each device simultaneously sends its groups to the 7 other devices and receives groups addressed to it from the 7 other devices. The collective transfers $k \times (N-1)/N$ expert-token slots per token over Ethernet links (the remaining $k/N$ slots on average are the locally retained tokens that do not cross links).

### Per-Token Dispatch Data Volume

Each token is a vector of $H$ elements in BF16 (2 bytes per element).

For a single token being dispatched to $(N-1) = 7$ remote devices out of $N = 8$:

$$V_{\text{dispatch, per token}} = \frac{N-1}{N} \times k \times H \times 2\ \text{bytes}$$

Substituting $N = 8$, $k = 8$ [D UNVERIFIED], $H = 7168$ [D UNVERIFIED]:

$$V_{\text{dispatch, per token}} = \frac{7}{8} \times 8 \times 7168 \times 2\ \text{bytes}$$

Step by step:
- $\frac{7}{8} \times 8 = 7$
- $7 \times 7168 = 50{,}176$
- $50{,}176 \times 2 = 100{,}352\ \text{bytes per token}$

This is the volume of token-embedding data that crosses Ethernet links per token per dispatch all-to-all.

---

## Expert Compute Phase

### What Happens

After the dispatch all-to-all completes, each device holds the tokens assigned to its $E/N$ local experts. Each device runs the expert feed-forward network (FFN) as a local matrix multiply:

$$\text{output} = \text{FFN}(x; W_1, W_2)$$

For a standard two-layer FFN expert with intermediate dimension $D$ [D UNVERIFIED]:

$$\text{FLOPs per token per expert} = 2 \times H \times D + 2 \times D \times H = 4 \times H \times D\ \text{FLOPs}$$

The expert compute on the receiving device processes up to $C = \lceil k \times B \times S / E \rceil$ tokens, where $C$ is the expert capacity. The matmul shape is $(C, H) \times (H, D)$ for the up-projection and $(C, D) \times (D, H)$ for the down-projection.

This phase involves no cross-device communication. The Ethernet links are idle during expert compute, which creates an overlap opportunity: the `ttnn.all_to_all` for the next pipeline stage can be dispatched to a separate command queue while local expert compute proceeds. This is discussed further in Chapter 5, `combine_and_accumulation.md`.

---

## Combine Phase

### What Happens

After expert computation, each device holds the expert output tensors for the tokens it processed. These outputs must be routed back to the originating devices so that each token's $k$ expert outputs can be weighted and summed according to the router scores. This return routing is a second `ttnn.all_to_all` — the combine all-to-all — with the same topology and cluster axis as the dispatch.

The combine all-to-all has the same per-token data volume as the dispatch, because the expert output shape $(C, H)$ matches the input shape:

$$V_{\text{combine, per token}} = V_{\text{dispatch, per token}} = 100{,}352\ \text{bytes per token}$$

The round-trip per-token volume for one MoE layer is therefore:

$$V_{\text{round-trip, per token}} = V_{\text{dispatch}} + V_{\text{combine}} = 2 \times 100{,}352 = 200{,}704\ \text{bytes per token per MoE layer}$$

---

## Total Data Volume for a Full Forward Pass

For Qwen3.5-35B with approximately 80 MoE layers [D UNVERIFIED]:

$$V_{\text{total, per token}} = 200{,}704 \times 80 = 16{,}056{,}320\ \text{bytes per token}$$

$$\approx 16\ \text{MB per token round-trip across all MoE layers}$$

At a batch of $B = 32$ sequences each of length $S = 2048$ tokens (a representative prefill workload):

$$V_{\text{prefill, per layer}} = 100{,}352 \times B \times S = 100{,}352 \times 32 \times 2048 = 6{,}576{,}668{,}672\ \text{bytes per layer}$$

$$\approx 6.6\ \text{GB per layer per direction during prefill}$$

This volume must pass through the Ethernet links in the time budget of one prefill step, making link bandwidth the dominant constraint during prefill.

---

## Prefill vs. Decode Regimes

At large batch (prefill), the operation is throughput-bound; at small batch (decode), it is latency-bound. See `num_links_parameter.md` for `num_links` tuning by regime.

---

## T3K Linear-Chain All-to-All Implementation

T3K uses `ttnn.Topology.Linear`, not `ttnn.Topology.Ring`. This is critical: T3K is a linear chain of 8 devices (devices 0–7 connected in sequence), with no wrap-around edge between device 7 and device 0. Using `ttnn.Topology.Ring` on T3K would configure the collective algorithm to expect a link that does not exist, causing hangs or incorrect results.

The linear all-to-all on T3K proceeds in $N - 1 = 7$ sequential rounds. In each round, each device forwards its current buffer one hop along the chain. After 7 rounds, every device has received data from every other device. The total time is approximately:

$$T_{\text{all-to-all}} = (N-1) \times \left( \frac{V_{\text{per-hop}}}{n_l \times \text{BW}_{\text{link}}} + \tau_{\text{hop}} \right)$$

where $V_{\text{per-hop}}$ is the volume moved in each round. For a full all-to-all of shape $(N, C, H)$, the per-hop volume per device is $C \times H \times 2$ bytes (one device-slice of the input tensor per round).

This linear structure is a property of the T3K hardware topology. It is not a software choice and cannot be overridden. The 7-round chain is the minimum number of hops required to deliver all data to all destinations on a linear path graph. A ring topology with a wrap-around edge would reduce the worst-case hop distance to $\lfloor N/2 \rfloor = 4$ rounds, but T3K does not have that edge on a single board.

---

## Why All-to-All Volume Dominates at Small Batch

At small batch sizes (decode), the expert FFN compute per device is minimal:

$$\text{FLOPs}_{\text{expert}} = 4 \times H \times D \times C = 4 \times H \times D \times \lceil k \times B / E \rceil$$

For $B = 1$ and $E / k = 32$, each expert processes at most 1 token: $C = \lceil 8 \times 1 / 256 \rceil = 1$.

The compute for a single token through a single expert is $4 \times 7168 \times D$ FLOPs [D UNVERIFIED]. At small $C$, this matmul is extremely small — likely memory-bandwidth-bound rather than compute-bound — and completes quickly. The all-to-all latency (even at `num_links=2`) then becomes the dominant term in the per-layer wall time.

At large batch (prefill), $C$ scales with $B \times S$, and both expert compute and all-to-all volume grow proportionally. The ratio of communication to compute is approximately constant with batch size (both scale linearly), so the prefill regime is not necessarily communication-dominated in the same way. Whether prefill is compute-bound or communication-bound depends on the hardware's compute-to-bandwidth ratio and the specific tensor shapes.

---

## References

- Chapter 1, `topology_implications_for_collectives.md` — linear-chain collective algorithm and 7-round hop sequence
- Chapter 1, `ethernet_link_bandwidth.md` — per-link bandwidth and hop latency figures
- Chapter 2, `collective_primitives.md` — `ttnn.all_to_all` signature, `cluster_axis=1`, `ttnn.Topology.Linear`
- Chapter 3, `num_links_parameter.md` — bandwidth model and latency formula for `num_links` tuning
- Chapter 5, `combine_and_accumulation.md` — overlap of combine all-to-all with subsequent operations

---

**Next:** [num_links_parameter.md](./num_links_parameter.md)
