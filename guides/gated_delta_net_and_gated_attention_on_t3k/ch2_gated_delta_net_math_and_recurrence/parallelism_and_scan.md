# Parallelism and Scan: Sequence-Dimension Execution Strategies

## 1. The Sequential Dependency Problem

The Gated Delta Net recurrence for a single head is:

$$S_t = g_t \cdot S_{t-1} + \tilde{k}_t \bigl(\beta_t \cdot (v_t - g_t \cdot S_{t-1}^\top \tilde{k}_t)\bigr)^\top$$

The correction vector $\beta_t \cdot (v_t - g_t \cdot S_{t-1}^\top \tilde{k}_t)$ contains the term $S_{t-1}^\top \tilde{k}_t$ — the state from the previous step appears **inside** the quantity being written back. This is what distinguishes the delta rule from a simple gated linear recurrence.

In a simple gated linear recurrence of the form $S_t = g_t \cdot S_{t-1} + W_t$, the write matrix $W_t$ is independent of $S_{t-1}$. That structure is associative: you can compute any pair of consecutive updates as a combined operator and apply parallel prefix scan across the sequence. The Gated Delta Net recurrence does **not** have this property in its raw form because $W_t$ itself depends on $S_{t-1}$.

### Why naive parallel scan fails

A parallel prefix scan over operators $f_1, f_2, \ldots, f_T$ requires that each $f_t$ be expressible as a function only of its local inputs (not of intermediate states). Here $f_t(S) = g_t \cdot S + \tilde{k}_t (\beta_t (v_t - g_t S^\top \tilde{k}_t))^\top$ is a function of $S$, which is fine — but to compose $f_t \circ f_{t-1}$, the composed operator's write term acquires a dependency on $S_{t-2}$ through the $S_{t-1}^\top \tilde{k}_{t-1}$ retrieval inside $f_{t-1}$. Composition does not close into a fixed-form operator of bounded complexity: each composition adds a new low-rank term, so the composed operator has rank growing linearly with the number of composed steps. This prevents direct parallel scan.

---

## 2. Chunkwise Parallel Training (Practical Approach)

The practical solution used in the `fla` (flash-linear-attention) library is **chunkwise parallelism**: divide the sequence of length T into non-overlapping chunks of size C (default C = 64), and handle intra-chunk and inter-chunk computation separately.

### 2.1 Within-chunk computation via WY-decomposition

Within a single chunk of C steps starting at chunk-initial state $S_0^{(c)}$, the multi-step update can be written as:

$$S_C^{(c)} = \Gamma_C \cdot S_0^{(c)} + U_C W_C^\top$$

Where:

- $\Gamma_C$ is a scalar (product of the C decay gates within the chunk):

$$\Gamma_C = \prod_{t=1}^{C} g_t \in (0, 1)$$

- $U_C \in \mathbb{R}^{d_k \times C}$ and $W_C \in \mathbb{R}^{d_v \times C}$ are matrices whose columns encode the per-step delta corrections, accumulated via the WY-decomposition

This is the **WY representation** of a sequence of rank-1 updates. The key insight is that once we know the chunk-initial state $S_0^{(c)}$, the intra-chunk output tokens can be computed in parallel using triangular masking over the $C \times C$ inner attention scores:

1. Compute the $C \times C$ causal attention matrix within the chunk: $A = (\tilde{Q}_{\text{chunk}} \tilde{K}_{\text{chunk}}^\top) \odot \text{tril}(\text{mask})$, where the mask accounts for the decaying products $g_{t'\ldots t}$ between tokens.
2. Compute the intra-chunk output: $O_{\text{intra}} = A V_{\text{chunk}}$ — standard triangular matmul, parallelizable over the C positions.
3. Compute the inter-chunk contribution from $S_0^{(c)}$ to all C query positions. Each position $\tau$ within the chunk sees the carry-in state already decayed by the cumulative gate product from chunk start up to that position. The contribution for position $\tau$ is:

$$o_{\text{cross}}[\tau] = \Gamma_\tau \cdot S_0^{(c)\top} \tilde{q}_\tau$$

   where $\Gamma_\tau = \prod_{t=1}^{\tau} g_t \in (0, 1)$ is the cumulative decay from position 0 to $\tau$ within the chunk. Stacking over all C positions in matrix form:

$$O_{\text{cross}} = D \cdot (\tilde{Q}_{\text{chunk}} \mathbin{@} S_0^{(c)})$$

   where $D \in \mathbb{R}^{C \times C}$ is a **diagonal** matrix with $D[\tau, \tau] = \Gamma_\tau$ (i.e., D is the diagonal matrix of intra-chunk prefix products), and $\tilde{Q}_{\text{chunk}} \mathbin{@} S_0^{(c)}$ has shape $[C, d_k] \times [d_k, d_v] \to [C, d_v]$. The multiplication by D applies a distinct per-row scalar to each of the C output rows. Without D, every query position would incorrectly retrieve from the fully undecayed carry-in state $S_0^{(c)}$, over-weighting the cross-chunk contribution at every position $\tau > 1$.

4. Sum: $O_{\text{chunk}} = O_{\text{intra}} + O_{\text{cross}}$.

Steps 2 and 3 are dense matmuls that use tensor cores effectively. Step 1 is $O(C^2 d_k)$ FLOPs, bounded since C = 64 is small.

### 2.2 Inter-chunk recurrence

The chunk-to-chunk recurrence is:

$$S_0^{(c+1)} = \Gamma_C^{(c)} \cdot S_0^{(c)} + U_C^{(c)} (W_C^{(c)})^\top$$

This recurrence **is** of the form $S' = a \cdot S + B$ where $a = \Gamma_C^{(c)}$ is a scalar and $B = U_C^{(c)} (W_C^{(c)})^\top$ is a `[d_k, d_v]` matrix. Because $a$ is a scalar, the composition of two such operators is:

$$(a_2, B_2) \circ (a_1, B_1) = (a_2 \cdot a_1,\; a_2 \cdot B_1 + B_2)$$

This composition is associative and the identity element is $(1, 0)$. The inter-chunk recurrence therefore **can** be parallelized over $T/C$ chunks via a parallel prefix scan.

### 2.3 Why sequential chunk scan is preferred in practice

Despite the theoretical associativity of the inter-chunk recurrence, a parallel prefix scan over $T/C$ chunks carries a significant memory cost: each node in the scan tree must store a full `[d_k, d_v]` state matrix. With T = 262,144 (2^18), C = 64, d_k = d_v = 128:

```
Number of chunks:              T / C  =  262,144 / 64  =  4,096
Bytes per state (BF16):        d_k × d_v × 2  =  128 × 128 × 2  =  32,768 bytes  =  32 KB
Total scan workspace:          4,096 chunks × 32 KB  =  131,072 KB  =  128 MB  (per head, per layer)
```

With 32 V-heads per layer, that is 32 × 128 MB = 4,096 MB ≈ 4 GB of scan workspace per layer — completely prohibitive. In practice, `chunk_gated_delta_rule` (the prefill kernel) scans the $T/C$ chunks **sequentially**, benefiting from the within-chunk parallelism (tensor-core matmuls over the `[C, d_k]` × `[d_k, d_v]` products) without paying the associative scan workspace cost.

The chunk-level parallelism is vectorized over the batch and head dimensions, which provides the main parallelism win on GPU and Tensix cores.

### 2.4 Complexity summary (prefill)

```
Within-chunk matmul FLOPs (per chunk, per head):
  Intra:   C × C × d_k   =  64 × 64 × 128  =  524,288
  Cross:   C × d_k × d_v =  64 × 128 × 128 =  1,048,576
  Update:  d_k × d_v × C =  128 × 128 × 64 =  1,048,576
  Total per chunk:  ~2.6 M FLOPs

Total prefill FLOPs (T/C chunks, H_v heads, per layer):
  (T/C) × H_v × 2.6 M  ≈  (T/64) × 32 × 2.6 M  ≈  T × 1.3 M FLOPs/layer

At T = 8192:  ~10.6 GFLOPs per Gated Delta Net layer
At T = 262,144 (2^18):  ~341 GFLOPs per Gated Delta Net layer
```

State memory during prefill is $O(B \times H_v \times d_k \times d_v)$ regardless of T — only the current chunk-boundary state needs to be held in registers/L1 at any one time.

---

## 3. Decode: Strictly Sequential

At decode time (T = 1, one new token per forward pass), there is no parallelism to exploit along the sequence dimension. The `recurrent_gated_delta_rule` kernel executes exactly one step of the recurrence:

```
# One decode step, per head h:
S_decayed    = g_{t,h} · S_{t-1,h}                         # [d_k, d_v]  scalar-times-matrix
retrieval    = S_decayed^T k̃_{t,h}                          # [d_v]        matrix-vector
error        = β_{t,h} · (v_{t,h} − retrieval)             # [d_v]        axpy
write        = k̃_{t,h} ⊗ error                             # [d_k, d_v]  outer product
S_{t,h}      = S_decayed + write                            # [d_k, d_v]  matrix add
o_{t,h}      = S_{t,h}^T q̃_{t,h}                           # [d_v]        matrix-vector
```

FLOPs per decode step, per head:

```
S_decayed:   d_k × d_v        =  128 × 128  =  16,384  (scale)
retrieval:   d_v × d_k        =  128 × 128  =  16,384  (matvec)
error:       d_v               =  128        (sub + scale)
write:       d_k × d_v        =  16,384      (outer product)
S update:    d_k × d_v        =  16,384      (add)
output:      d_v × d_k        =  16,384      (matvec)
---
Total per head:                ≈  82,000 FLOPs  (~0.08 MFLOPs)

Total per layer (32 heads):    ≈  2.6 MFLOPs
```

Memory read/write per decode step, per head (BF16):

```
Read  S_{t-1}:   d_k × d_v × 2  =  32,768 bytes  =  32 KB
Write S_t:       d_k × d_v × 2  =  32,768 bytes  =  32 KB
Read  k̃, q̃, v:  (d_k + d_k + d_v) × 2 = 256 + 256 + 256 = 768 bytes
---
Dominant cost: reading and writing the state matrix (32 KB read + 32 KB write = 64 KB per head)
```

Arithmetic intensity per head: `~82K FLOPs / (768 + 65,536) bytes ≈ 1.24 FLOPs/byte`. This is far below the Wormhole ridge point (~100 FLOPs/byte), confirming the decode step is **heavily memory-bandwidth-bound** on T3K. The roofline analysis and implications for kernel design are covered in Chapter 5.

---

## 4. `chunk_gated_delta_rule` vs. `recurrent_gated_delta_rule`

| Property | `chunk_gated_delta_rule` (prefill) | `recurrent_gated_delta_rule` (decode) |
|----------|-----------------------------------|---------------------------------------|
| Input T | > 1 (typically C-aligned) | 1 |
| Chunk size C | 64 | 1 (degenerate) |
| Intra-chunk | Triangular matmul (tensor cores) | N/A |
| Inter-chunk | Sequential scan over T/C chunks | N/A |
| State I/O | Chunk boundary states only | Full state read+write every step |
| FLOPs/token | ~O(C · d_k · d_v) | ~O(d_k · d_v) |
| Bottleneck | Compute (matmuls over C) | Memory bandwidth (state I/O) |
| Implementation | `fla.ops.gated_delta_rule.chunk` | `fla.ops.gated_delta_rule.recurrent` |

Both kernels currently execute on CPU/GPU via the `flash-linear-attention` library. Chapter 4 catalogs which sub-operations map to TTNN primitives and which require new fused kernels for T3K execution.

---

**Next:** [`state_vs_kv_cache_memory.md`](./state_vs_kv_cache_memory.md)
