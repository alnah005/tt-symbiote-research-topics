# Circular Buffer Layout for the Windowed KV Cache

This file describes the circular (ring) buffer data structure that implements
the windowed eviction policy in device DRAM. It derives the write-pointer and
wrap-around arithmetic, shows how the buffer is exposed as a fixed-shape TTNN
tensor `[B, H, w, d]` with a companion position-offset scalar, and contrasts
this design with the grow-in-place tensor used by full-attention KV caches.

## Why Not Simply Shift the Cache on Every Step

The most obvious implementation of windowed eviction would shift all w−1
remaining entries one slot to the left in DRAM each time a new entry is
written, keeping the cache as a contiguous block with the newest entry always
at slot w−1. This is correct but catastrophically slow: a left-shift requires
reading and rewriting `(w−1) * d` elements per head per batch item per layer.
For w = 4 096 and the parameters of a 7B model, one full shift moves roughly
64 MiB of data per layer per decode step — comparable to a full model weight
forward pass. This approach is completely impractical.

A circular buffer avoids the shift entirely. Instead of moving data, it moves
a pointer. The buffer slots are reused in a round-robin fashion; only one slot
is written per decode step regardless of w. The price paid is that the cache
entries are no longer in temporal order in DRAM — they wrap around — so a small
amount of index arithmetic is needed when accessing the cache.

## Circular Buffer Fundamentals

A circular buffer of capacity C is a fixed array of C slots (indexed 0 through
C−1) accompanied by a **write pointer** `wp` (an integer in [0, C−1]). The
pointer advances by one on each write, wrapping back to 0 after reaching C−1:

$$\text{wp}_{\text{new}} = (\text{wp}_{\text{old}} + 1) \bmod C$$

When the buffer is full (all C slots contain valid data), the slot about to be
overwritten holds the oldest entry. Overwriting it therefore simultaneously
writes the newest entry and evicts the oldest — exactly the windowed eviction
semantics from `kv_cache_lifecycle.md`.

For the windowed KV cache the buffer capacity C equals the window size w:

$$C = w$$

## Slot Assignment and Position Mapping

Let `t` be the absolute token position (0-indexed from the start of the
sequence, including prompt tokens). The slot in the circular buffer to which
position t is written is:

$$\text{slot}(t) = t \bmod w$$

This is equivalent to initialising `wp = 0` before the first token and
advancing by one on each write. The slot assignment function is deterministic
and invertible given knowledge of `t` and `w`.

### Worked Example (w = 4)

```text
Token position t  | slot = t mod 4 | buffer state after write
------------------|-----------|-----------------------------------------
0                 | 0         | [t=0, ___,  ___,  ___ ]   wp=1
1                 | 1         | [t=0, t=1,  ___,  ___ ]   wp=2
2                 | 2         | [t=0, t=1,  t=2,  ___ ]   wp=3
3                 | 3         | [t=0, t=1,  t=2,  t=3]    wp=0  (full)
4                 | 0         | [t=4, t=1,  t=2,  t=3]    wp=1  (t=0 evicted)
5                 | 1         | [t=4, t=5,  t=2,  t=3]    wp=2  (t=1 evicted)
6                 | 2         | [t=4, t=5,  t=6,  t=3]    wp=3  (t=2 evicted)
7                 | 3         | [t=4, t=5,  t=6,  t=7]    wp=0  (t=3 evicted)
8                 | 0         | [t=8, t=5,  t=6,  t=7]    wp=1  (t=4 evicted)
```

After t = 3 the buffer is full and steady-state eviction begins. At t = 7,
slots 0–3 hold t=4, t=5, t=6, t=7 respectively — this IS in chronological
order because the write pointer has advanced from 0 to 3 without yet wrapping
a second time. Non-chronological ordering first appears at t = 8: slot 0 is
overwritten with t=8 while slots 1–3 still hold t=5, t=6, t=7, so the logical
ordering oldest→newest is slots [1, 2, 3, 0], wrapping from the slot
immediately after `wp` around to `wp−1`. In general, non-chronological layout
occurs whenever t ≥ 2w (i.e., the write pointer has completed at least one
full wrap).

### Reading Back the Window

To read all w entries in ascending position order (oldest first) for a query at
position t, the access pattern is:

$$\text{slots to read, oldest first} =
  \bigl(t \bmod w\bigr) + 1,\;
  \bigl(t \bmod w\bigr) + 2,\;
  \ldots,\;
  \bigl(t \bmod w\bigr) + w
  \quad \text{(all indices taken mod } w\text{)}$$

Equivalently, starting from the slot immediately after the most recently written
slot (which is the oldest slot), read w consecutive slots wrapping at w:

$$\text{slot}(t - w + 1 + i) = (t - w + 1 + i) \bmod w, \quad i = 0, 1, \ldots, w-1$$

This is a single contiguous read if `(t+1) mod w == 0` (i.e., no wrap needed
in this particular step), or a pair of contiguous reads straddling the wrap
boundary otherwise. Attention kernels on Wormhole handle this via a split read
or by treating the circular index as an offset into a gather operation, as
described in Chapter 3.

## The Position Offset Scalar

Circular buffer slot indices (0 through w−1) are physical addresses within the
DRAM tensor. Token positions (0, 1, 2, ...) are logical sequence positions
required for positional encoding (RoPE). The two are related by:

$$\text{position}(\text{slot}) = \text{slot} + k \cdot w, \quad \text{for the unique } k \geq 0 \text{ such that } 0 \leq \text{slot} < w$$

More practically, given the current write pointer `wp` and the current absolute
position `t`, the oldest position in the cache is:

$$t_{\text{oldest}} = t - w + 1$$

and it occupies slot `t_{\text{oldest}} \bmod w = (t - w + 1) \bmod w`.

Rather than storing the full position of every slot, it suffices to store a
single scalar — the **position offset** — that allows the position of any slot
to be reconstructed:

$$\text{pos\_offset} = t - w + 1 \quad \text{(absolute position of the oldest cache entry)}$$

Given `pos_offset` and a slot index `s`, the absolute position of the entry in
slot `s` is:

$$\text{position}(s) = \text{pos\_offset} + \bigl((s - \text{pos\_offset}) \bmod w\bigr)$$

RoPE implementations consume this `pos_offset` to compute the correct rotation
angles without storing a separate position integer per cache slot.

## TTNN Tensor Shape and Companion Scalar

The windowed KV cache for one layer is represented as a pair of TTNN tensors
(one for keys, one for values), each with shape:

```text
[B, H, w, d]
```

alongside a companion scalar (or 1-D integer tensor of length 1):

```text
pos_offset: int   # absolute position of the oldest cache entry = t - w + 1
```

The four dimensions of the cache tensor are:
- `B`: batch dimension — one cache per sequence in the batch
- `H`: head dimension — one independent buffer per attention head
- `w`: the circular buffer slots (NOT ordered by token position)
- `d`: head dimension (features per key/value vector)

This shape is **fixed at allocation time** and never changes. There is no
reallocation, no tensor resizing, and no data copy between decode steps. The
only mutation per decode step is:
1. Write the new key vector `k_t` into slice `[:, :, t mod w, :]`.
2. Write the new value vector `v_t` into the same slot of the value cache.
3. Increment `pos_offset` by 1 (or equivalently, record `t - w + 1` directly).

In TTNN syntax, the write is expressed as an in-place slice update:

```python
k_cache[:, :, t % w, :] = k_t        # k_t shape: [B, H, 1, d]
v_cache[:, :, t % w, :] = v_t        # v_t shape: [B, H, 1, d]
pos_offset = t - w + 1
```

The cache tensor lives in device DRAM throughout. Keys and values are written
directly to the target slot via a DRAM write; no intermediate host buffer is
involved.

## Buffer Layout Diagram

The diagram below shows the physical DRAM layout of the `[B=1, H=1, w=8, d=4]`
key cache for a single head and batch item after t = 11 (steady state, w = 8):

```text
Physical slot index (cache dim 2):
  0     1     2     3     4     5     6     7
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│ t=8 │ t=9 │t=10 │t=11 │ t=4 │ t=5 │ t=6 │ t=7 │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
         ↑
    wp = 4  (next write will go into slot 4, overwriting t=4)

Logical order (oldest→newest): slots 4,5,6,7,0,1,2,3
  i.e. positions:               t=4, t=5, t=6, t=7, t=8, t=9, t=10, t=11

pos_offset = 11 - 8 + 1 = 4   (oldest position currently in cache)
```

When the next token t=12 arrives, slot 4 (currently holding t=4) is
overwritten with t=12, and pos_offset advances to 5:

```text
  0     1     2     3     4     5     6     7
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│ t=8 │ t=9 │t=10 │t=11 │t=12 │ t=5 │ t=6 │ t=7 │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
                              ↑
                         wp = 5

Logical order (oldest→newest): slots 5,6,7,0,1,2,3,4
  i.e. positions:               t=5, t=6, t=7, t=8, t=9, t=10, t=11, t=12

pos_offset = 5
```

No data was moved. Only one slot was written.

## Contrast with Full-Attention Grow-in-Place Tensors

Full-attention KV caches use a fundamentally different allocation strategy:

| Property                 | Windowed circular buffer     | Full-attention grow-in-place   |
|--------------------------|------------------------------|--------------------------------|
| DRAM tensor shape        | Fixed: `[B, H, w, d]`        | Grows: `[B, H, T, d]` at step T |
| Allocation strategy      | Pre-allocated once           | Either pre-allocated to max-T or dynamically extended |
| DRAM bytes per step      | Write 1 slot: `B*H*d*bytes`  | Write 1 slot: `B*H*d*bytes`    |
| Eviction cost            | Zero (overwrite in place)    | None needed (no eviction)      |
| Read for attention       | w slots (possible wrap)      | T slots (always contiguous)    |
| Position→slot mapping    | `t mod w` with pos_offset    | Direct index t (no arithmetic) |
| DRAM footprint at T=131k | Constant: `B*H*w*d*bytes`    | Growing: `B*H*T*d*bytes`       |

The key practical difference is the DRAM footprint. For full attention, the
tensor must be either pre-allocated to the maximum context length (wasting DRAM
early in generation) or dynamically reallocated (incurring copy overhead at
each reallocation). For the circular buffer, the allocation is fixed at `w`
slots from the start and never changes.

The slight added complexity is the wrap-around arithmetic when reading the w
slots for attention computation. This complexity is bounded and constant in w;
it does not scale with T. Chapter 3 characterises the resulting memory access
patterns and shows how the wrap boundary affects prefetch scheduling on
Wormhole's NoC.

## Relation to the TTNN KV Cache Update Primitive

TTNN exposes a KV cache update operation (used in tt-transformers) that
encapsulates the circular write:

```python
ttnn.update_cache(cache_tensor, new_kv, update_index)
```

where `update_index = t % w`. Internally this operation performs an in-place
write to the correct slot in the `[B, H, w, d]` DRAM tensor. The
`pos_offset` scalar is maintained separately by the model's decode loop and
passed to the RoPE application and the attention mask construction.

The attention call that follows the cache update reads the full `[B, H, w, d]`
cache tensor via `ttnn.matmul` (Q × K^T) and a second `ttnn.matmul` (scores ×
V), applying the appropriate mask or gather to handle the wrap boundary. These
steps are detailed in Chapter 4 (TTNN Primitive Operations and Tensor Shapes),
which covers both the decode primitive shapes and the kernel gap analysis.

---

**Next:** [Chapter 3 — Data Dependencies and Memory Access Patterns](../ch3_data_dependencies/index.md)
