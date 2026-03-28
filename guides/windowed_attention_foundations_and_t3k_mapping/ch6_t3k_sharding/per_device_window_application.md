# Per-Device Window Application

This file analyses where and how the window boundary is enforced when the KV
cache is distributed across multiple devices. The central question is: does the
global window constraint `[T - w + 1, T]` need to be evaluated once and
propagated to all devices, or can each device independently enforce a local
variant of the same constraint? The answer differs between the two sharding
strategies.

## Global vs Per-Device Window Enforcement

### The Global Window Constraint

For a query at decode step `T` (0-indexed), windowed attention restricts
attention to KV pairs whose absolute position `t` satisfies:

```math
T - w + 1 \;\leq\; t \;\leq\; T
```

During the fill phase (`T < w`), the constraint simplifies to `0 ≤ t ≤ T`,
meaning only the `T + 1` positions written so far are valid. In steady state
(`T ≥ w - 1`), all `w` slots of the circular buffer hold valid tokens and
the constraint is satisfied implicitly by the fixed buffer capacity — no
explicit masking of out-of-window tokens is needed.

The window constraint is a single predicate over absolute token position. It
does not depend on which device a KV pair is stored on; it depends only on `T`,
`w`, and the token's absolute position `t`. This means that in principle the
window can be enforced either:

- **Globally before sharding** — the host computes the window bounds and slices
  or masks the KV tensor to contain only valid tokens, then shards the result.
- **Per-device after sharding** — each device independently determines which
  of its locally held tokens fall within the global window and masks or ignores
  the rest.

The two approaches yield identical mathematical results if implemented
correctly. They differ in complexity and in the correctness invariants that
must be maintained.

## Head-Parallel Sharding: Per-Device Window Application

### Why Per-Device Is Straightforward

Under head-parallel sharding, each device holds a subset of heads and the
**full `w`-length circular buffer** for those heads. The buffer structure on
each device is identical to the single-device circular buffer described in
Chapter 2: a `[B, H_kv/N, w, d]` tensor written in round-robin order with a
shared write pointer `update_index = T % w`.

Because every device holds the same set of `w` token positions (differing only
in which heads' vectors are stored), the window enforcement logic is completely
local and requires no inter-device coordination:

```text
Head-parallel per-device window enforcement:

  Each device i:
    k_shard : [B, H_kv/N, w, d]   (circular buffer for heads i*H_kv/N .. (i+1)*H_kv/N - 1)
    v_shard : [B, H_kv/N, w, d]
    q_shard : [B, H_q/N,  1, d]

    Global window [T-w+1, T] applies identically on all devices.
    During fill phase: pass position_mask [B, 1, 1, w] with -inf for slots T+1 .. w-1.
    At steady state: all w slots valid, no mask needed.

    No device needs information from any other device to determine which
    slots are valid. The write pointer T%w is a globally synchronised scalar.
```

The `position_mask` tensor `[B, 1, 1, w]` is the same on all devices and can
be computed once on the host and broadcast, or computed independently on each
device (since it depends only on the scalar `T` and the fixed `w`).

### Correctness Proof: Union of Per-Device Windows = Global Window

Let `W_global = { t : T - w + 1 ≤ t ≤ T }` be the global window. Under
head-parallel sharding, device `i` enforces exactly the set `W_global` against
its local shard. The union of per-device valid token sets is:

```math
\bigcup_{i=0}^{N-1} W_i = \bigcup_{i=0}^{N-1} W_{\text{global}} = W_{\text{global}}
```

The equality holds trivially: every device enforces the same predicate over
the same set of token positions `{0, 1, ..., w-1}` (the circular buffer
slots), so the union collapses to the single set `W_global`.

Each device contributes a subset of heads' attention outputs for exactly the
tokens in `W_global`. The concatenation of per-device outputs along the head
dimension reconstructs the full multi-head attention output over `W_global`,
which is the correct result. Because the window predicate operates on the time
axis and all devices hold the full `w`-length time axis, there are no boundary
issues or edge cases introduced by the sharding itself beyond those already
present in the single-device case (covered in
[`../ch2_kv_cache_management/circular_buffer_layout.md`](../ch2_kv_cache_management/circular_buffer_layout.md)
and
[`../ch4_ttnn_primitives/decode_primitives.md`](../ch4_ttnn_primitives/decode_primitives.md)).

### Practical Implementation

The host decode loop maintains one scalar `T` (the current absolute position)
and one derived scalar `update_index = T % w`. Both are used identically on
all devices. The device-side window enforcement proceeds as follows:

```text
Per-device logic at decode step T (head-parallel, steady state T ≥ w):

  1. Update circular buffer:
       ttnn.update_cache(k_shard, k_T_shard, T % w)
       ttnn.update_cache(v_shard, v_T_shard, T % w)

  2. Compute attention (no mask needed in steady state):
       output_shard = ttnn.scaled_dot_product_attention_decode(
           q_shard,    # [B, H_q/N, 1, d]
           k_shard,    # [B, H_kv/N, w, d]
           v_shard,    # [B, H_kv/N, w, d]
           attn_mask=None,          ← all w slots valid, no masking
           scale=1/sqrt(d)
       )

Per-device logic at decode step T (head-parallel, fill phase T < w):

  1. Update circular buffer (same as above).

  2. Build fill-phase mask (computed on host, same for all devices):
       position_mask = zeros(B, 1, 1, w)
       position_mask[:, :, :, T+1:] = -inf   # slots T+1..w-1 not yet written

  3. Compute attention with mask:
       output_shard = ttnn.scaled_dot_product_attention_decode(
           q_shard, k_shard, v_shard,
           attn_mask=position_mask,
           scale=1/sqrt(d)
       )
```

The `position_mask` is identical for all devices. It can be constructed once
on the host and replicated to all devices, or constructed independently on each
device using the scalar `T`.

## Sequence-Parallel Sharding: Per-Device Window Application

### The Window Enforcement Problem

Under sequence-parallel sharding, device `i` holds tokens in the window slice:

```math
\text{Device } i \text{ slice} = \Bigl[ i \cdot \frac{w}{N},\; (i+1) \cdot \frac{w}{N} - 1 \Bigr]
```

These are indices into the circular buffer (slot indices), not absolute token
positions. The mapping between slot index `s` and absolute token position `t`
depends on the write pointer:

```math
t(s) = \text{pos\_offset} + s \quad \text{with wrap-around in the circular buffer}
```

More precisely, in the circular buffer the slot at index `s` holds the token
written at step `T_write(s) = T - ((T \% w - s + w) \% w)`. The window
predicate is `T - w + 1 ≤ T_write(s) ≤ T`, which at steady state is satisfied
by all `w` slots. At the fill phase, slots `T+1` through `w-1` are
uninitialised.

The window enforcement under sequence-parallel sharding requires each device
to determine which of its local slots fall within the global window. This is
straightforward at steady state (all slots valid), but the fill phase creates
a complication: the invalid trailing slots may be distributed across multiple
devices, requiring those devices to apply partial masks.

### Global Window Applied Before Sharding

The simplest correct implementation applies the window globally before sharding:

1. The host constructs the full `position_mask` tensor `[B, 1, 1, w]` covering
   all `w` buffer slots.
2. This mask is sharded along the `w` dimension, giving each device a slice
   `[B, 1, 1, w/N]`.
3. During the all-gather that reconstructs the full KV tensor, the
   corresponding mask slices are also all-gathered to produce a global mask
   on each device.

```text
Global mask sharding (fill phase, T=3, w=16, N=8):

  Full mask: [B, 1, 1, 16]
    slots 0..3: valid (0)
    slots 4..15: -inf

  Sharded per device:
    Dev0: [B, 1, 1, 2]  slots  0..1 → [  0,   0]
    Dev1: [B, 1, 1, 2]  slots  2..3 → [  0,   0]
    Dev2: [B, 1, 1, 2]  slots  4..5 → [-inf, -inf]
    ...
    Dev7: [B, 1, 1, 2]  slots 14..15 → [-inf, -inf]

  After all-gather on mask: each device holds full [B, 1, 1, 16] mask.
  Applied to the gathered KV tensor [B, H_kv, 16, d].
```

This approach requires an additional all-gather of the mask tensor. The mask
is `B × 1 × 1 × w × 2 bytes` = `B × w × 2 bytes` total — small relative to
the KV all-gather. For `B = 32`, `w = 4096`, BF16: `32 × 4096 × 2 = 256 KiB`.
This is negligible compared to the KV all-gather volume.

### Per-Device Window Application After Sharding

An alternative is for each device to compute its local mask slice independently.
Each device knows its own slot range `[i * w/N, (i+1) * w/N - 1]` and the
global `T`. The validity of each slot in the local range is deterministic:

```text
Device i per-device mask construction (fill phase, T < w, N=8):

  w_per_dev = w // N
  local_start = i * w_per_dev     (inclusive)
  local_end   = local_start + w_per_dev - 1  (inclusive)

  For each slot s in [local_start, local_end]:
    is_valid = (s <= T)

  local_mask[b, 0, 0, s - local_start] = 0 if is_valid else -inf
```

This avoids the mask all-gather entirely but requires each device to be aware
of its own position in the sharding scheme (i.e., `i`, `w_per_dev`, and the
global `T`). These are all available to the host, which can precompute and
distribute the per-device masks.

### Edge Cases When `w` Is Not Divisible by `N`

When `w` is not evenly divisible by `N = 8`, the `w/N` slices are not all the
same length. The standard approach is to give the first `w mod N` devices one
extra slot:

```math
w_i = \begin{cases}
\lfloor w / N \rfloor + 1 & \text{if } i < w \bmod N \\
\lfloor w / N \rfloor & \text{if } i \geq w \bmod N
\end{cases}
```

This creates devices with unequal shard sizes, which complicates the all-gather
(the output tensor is no longer uniformly shaped across devices). TTNN's
`all_gather` requires uniform shard sizes along the gathered dimension. The
standard workaround is to pad the smaller shards to the size of the largest
shard before the gather:

```text
Non-divisible case (w=9, N=8):
  floor(9/8) = 1, 9 mod 8 = 1
  Dev0: 2 slots  (one extra slot)
  Dev1..Dev7: 1 slot each

  Padded to uniform size 2:
    Dev0: [B, H_kv, 2, d]   (2 real slots)
    Dev1: [B, H_kv, 2, d]   (1 real slot + 1 padding slot, masked to -inf)
    ...
    Dev7: [B, H_kv, 2, d]   (1 real slot + 1 padding slot, masked to -inf)

  After all_gather(dim=2): each device holds [B, H_kv, 16, d]
  The 7 padding slots (one per device 1–7) must be masked to -inf.
  Effective window size after masking: 9 slots.
```

The padding approach introduces wasted slots in the assembled tensor whenever
`w mod N ≠ 0`. The number of padding slots equals `N - (w mod N)`. Worst-case
padding occurs when `w mod N = 1`, which wastes `N - 1 = 7` slots on the last
device (i.e., the seven smaller-shard devices each carry one padding slot).
For `N = 8` this is at most 7 wasted slots out of the assembled `ceil(w/N) × N`
total slots, which is a rounding overhead of less than `(N-1)/w` — negligible
for any practical `w ≥ 64`.

### Correctness Invariant for Sequence-Parallel Sharding

For the union of per-device token sets to equal the global window, the
following invariant must hold after the all-gather:

**Invariant:** For every absolute token position `t` in `[T - w + 1, T]`,
there exists exactly one device `i` whose shard contains slot `s = t mod w`,
and that slot holds the correct value `(k_t, v_t)` written at step `t`.

This invariant is satisfied at steady state by the circular buffer write
discipline: each token is written to the unique slot `T mod w`, and once
written, it remains valid until the same slot is overwritten by a token `w`
steps later. At steady state all `w` slots hold valid data from the current
window, so every device's shard contains only valid tokens.

At fill phase the invariant requires tracking which slots are valid (those with
indices `0` through `T`). Slots `T+1` through `w-1` are uninitialised, and
under sequence-parallel sharding some devices may hold only uninitialised slots
during early fill steps. Those devices must mask all of their local slots to
`-inf`. The all-gather includes these `-inf` mask values, and the softmax over
the assembled attention scores correctly assigns zero attention weight to all
uninitialised positions regardless of which device holds them.

### Correctness Check: Fill Phase with Non-Uniform Valid Slots

Consider `w = 16`, `N = 8`, `T = 3` (fill phase, 4 valid tokens):

```text
Circular buffer state at T=3, w=16 (steady state not yet reached):

  Slot:  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
         t0  t1  t2  t3  --  --  --  --  --  --  --  --  --  --  --  --
                         ← invalid (never written) →

Sequence-parallel shard assignment (w/N = 2 slots per device):

  Dev0: slots  0..1  → [t0, t1]  — both valid
  Dev1: slots  2..3  → [t2, t3]  — both valid
  Dev2: slots  4..5  → [--, --]  — both invalid
  Dev3: slots  6..7  → [--, --]  — both invalid
  Dev4: slots  8..9  → [--, --]  — both invalid
  Dev5: slots 10..11 → [--, --]  — both invalid
  Dev6: slots 12..13 → [--, --]  — both invalid
  Dev7: slots 14..15 → [--, --]  — both invalid

Per-device masks:
  Dev0: [0, 0]         (both valid)
  Dev1: [0, 0]         (both valid)
  Dev2..Dev7: [-inf, -inf]

All-gather mask → [B, 1, 1, 16]: [0, 0, 0, 0, -inf, -inf, ..., -inf]
Union of valid slots = {0, 1, 2, 3} = {t0, t1, t2, t3} = W_global. ✓
```

The correctness check confirms that the union of per-device valid slot sets
equals `W_global` for this example. The same argument extends to arbitrary
`T`, `w`, and `N`: the window predicate `s ≤ T` partitions the slots into
contiguous valid and invalid ranges, and each device holds a contiguous slice
of this range. Two cases arise:

- **If `(T+1) mod (w/N) == 0`:** The valid/invalid boundary falls exactly
  between two adjacent devices. All shards are either fully valid or fully
  invalid. No partial fill-phase masking is required on any device; devices
  `0` through `(T+1)/(w/N) - 1` apply zero mask, devices `(T+1)/(w/N)`
  through `N-1` apply a full `-inf` mask.

- **Otherwise:** The boundary falls within the shard of device
  `floor(T / (w/N))`, which applies a partial mask covering slots `T+1`
  through `(floor(T / (w/N)) + 1) * (w/N) - 1` of its local shard.

### Summary of Edge Cases

| Condition | Sharding | Behaviour |
|---|---|---|
| Fill phase, all slots on one device valid | Sequence-parallel | Device applies zero mask |
| Fill phase, all slots on one device invalid | Sequence-parallel | Device masks all slots to -inf |
| Fill phase, boundary falls within a device's slice | Sequence-parallel | Device applies partial mask at slot T+1 |
| Steady state, `w` divisible by `N` | Sequence-parallel | All devices' slices fully valid; no mask |
| Steady state, `w` not divisible by `N` | Sequence-parallel | Pad to uniform shard size; mask padding slots |
| Fill phase, any | Head-parallel | Same logic as single-device; all devices apply identical global fill-phase mask |
| Steady state, `H_kv` not divisible by `N` | Head-parallel | Distribute heads unequally; last device may hold fewer heads (requires padding in output all-gather) |

## Recommendation: Enforce Window Globally Before Sharding

For sequence-parallel sharding, constructing the full `position_mask` on the
host and sharding it alongside the KV data is the more straightforward path.
This approach:

1. Requires no per-device conditional logic in the kernel or program config.
2. Handles the `w mod N ≠ 0` padding case uniformly: padding slots are masked
   to `-inf` in the global mask before sharding.
3. Is transparent to the `ttnn.scaled_dot_product_attention_decode` kernel,
   which receives a standard `[B, 1, 1, w_padded]` mask exactly as in the
   single-device case.

For head-parallel sharding (the recommended strategy from
[`sharding_strategies.md`](./sharding_strategies.md)), the window is always
applied identically on each device using a single scalar `T` and the fixed
buffer capacity `w`. No per-device mask slicing is needed except during the
fill phase, where the single host-constructed mask is replicated to all devices.
There are no boundary issues under head-parallel sharding.

---

**Next:** [Chapter 7 — Roofline Analysis and Existing Kernel Survey](../ch7_roofline_and_kernels/index.md)
