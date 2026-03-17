# SDPAProgramConfig

`SDPAProgramConfig` controls how the SDPA decode kernel is mapped onto the
Tenstorrent device compute grid. Getting this configuration wrong is one of
the most common sources of silent performance and correctness regressions in
multi-batch or GQA attention.

---

## Import and Construction

```python
from ttnn.model_preprocessing import SDPAProgramConfig

config = SDPAProgramConfig(
    compute_with_storage_grid_size=(8, 4),  # (cols, rows) = 32 cores
    q_chunk_size=32,
    k_chunk_size=32,
)
```

Pass the config to the kernel:

```python
output = ttnn.transformer.scaled_dot_product_attention_decode(
    q, k, v,
    cur_pos_tensor=cur_pos_t,
    page_table_tensor=page_table,
    program_config=config,
)
```

---

## Fields

### `compute_with_storage_grid_size`

**Type:** `tuple[int, int]` — `(cols, rows)`.

Specifies the 2D compute grid the kernel is allowed to use, expressed as
`(number_of_columns, number_of_rows)`. The total core count is `cols * rows`.

**Hard constraint:** `cols * rows >= b * nkv`.

See the constraint section in `tensor_shape_reference.md` for the full failure description.

**Do not over-provision.** Unused cores still participate in synchronization
barriers, adding latency. Size the grid to exactly what you need, or to the
next convenient rectangle that satisfies the constraint.

| Scenario | b | nkv | Required cores | Example grid |
|----------|---|-----|----------------|--------------|
| Single-batch GQA | 1 | 4 | 4 | `(4, 1)` |
| Single-batch MHA | 1 | 16 | 16 | `(8, 2)` |
| Batch-8 GQA | 8 | 4 | 32 | `(8, 4)` |
| Batch-8 MHA | 8 | 16 | 128 | `(8, 16)` |

---

### `q_chunk_size`

**Type:** `int`. Typical value: `32`.

The number of query head positions processed together in one kernel launch
chunk. Because TTNN tile layout uses 32-element tiles, `q_chunk_size=32`
aligns to exactly one tile and is the standard choice. Setting it lower (e.g.,
16) reduces register pressure, which can help when `dh` is large (512+), but
increases kernel launch overhead. Setting it higher (e.g., 64) increases
register pressure and may cause spills on devices with small L1.

For the vast majority of models with `dh <= 256`, leave `q_chunk_size=32`.

---

### `k_chunk_size`

**Type:** `int`. Typical value: `32` or `64`.

The number of K/V sequence positions processed per inner loop iteration during
the attention score computation. This controls the granularity of the online
softmax accumulation:

- **Smaller values (32):** Each chunk contributes fewer terms to the softmax
  denominator. This is more numerically stable and avoids large intermediate
  exponentials for very long sequences. Use 32 when sequence lengths exceed
  32K tokens or when debugging numerical precision.
- **Larger values (64, 128):** Fewer loop iterations for the same sequence
  length. Better throughput for moderate sequence lengths (up to ~16K). Use
  64 as a starting point for standard production configurations.

`k_chunk_size` must be a multiple of `block_size` in paged mode
(`k_chunk_size % block_size == 0`). If `block_size=32` and you set
`k_chunk_size=64`, the kernel processes 2 blocks per iteration — this is legal
because 64 is a multiple of 32. Setting `k_chunk_size=48` with `block_size=32`
would be illegal (48 % 32 = 16 ≠ 0) and raises an error at program compilation
time.

---

## Parallelization Strategy

The kernel maps the attention computation across the grid as follows:

1. The grid is flattened into a 1D list of `num_cores = cols * rows` cores.
2. The `b * nkv` work units (one per `(batch_element, kv_head)` pair) are
   distributed across the first `b * nkv` cores in row-major order: core 0
   handles `(batch=0, kv_head=0)`, core 1 handles `(batch=0, kv_head=1)`, and
   so on.
3. Each assigned core iterates over the sequence dimension in `k_chunk_size`
   steps, accumulating the partial softmax numerator and denominator (online
   softmax).
4. At the end, each core writes its result into the output tile corresponding
   to its assigned `(batch_element, kv_head)` pair.

Because each KV head produces `group_size = nh / nkv` query head outputs, one
core effectively computes attention for an entire group of query heads in
sequence. The per-core work scales with `group_size * s / k_chunk_size`
iterations.

**Implication for large `group_size`:** In MHA (`group_size=1`) each core
handles one query head. In GQA with `group_size=4`, each core handles 4 query
heads — 4x more work per core for the same sequence length. If you find GQA
decode slower than expected, this is the primary cause; consider whether the
sequence chunking can be parallelized differently (model-specific
optimization).

---

## Worked Example: Ling Model

Configuration: `b=1`, `nh=16`, `nkv=4`, `dh=128`, paged KV cache with
`block_size=32`, maximum sequence length 128K tokens.

**Step 1 — Compute required cores.**

```
b * nkv = 1 * 4 = 4 cores required
```

**Step 2 — Choose a grid.**

Minimum viable grid: `(4, 1)` — exactly 4 cores, zero idle cores. An
alternative is `(8, 1)` (8 cores); the first 4 handle the 4 work units and the
last 4 are idle. For a single batch element with 4 KV heads, `(4, 1)` is
strictly better.

**Step 3 — Choose chunk sizes.**

At 128K tokens with `block_size=32`, there are `128K / 32 = 4096` blocks per
sequence. Using `k_chunk_size=32` processes one block per inner loop iteration
(4096 iterations per core). Using `k_chunk_size=64` processes two blocks per
iteration (2048 iterations). At 128K tokens, `k_chunk_size=32` is safer for
numerical stability; use `k_chunk_size=64` only if benchmarking confirms no
precision regression.

**Resulting config:**

```python
config = SDPAProgramConfig(
    compute_with_storage_grid_size=(4, 1),  # 4 cores, matches b*nkv=4
    q_chunk_size=32,                        # one tile, standard
    k_chunk_size=32,                        # one block per iter, stable
)
```

---

## Default Config Behavior

When `program_config=None` is passed to the kernel, a default grid is selected
based on the device's total core count, not on `b * nkv`. On an 8x8 device
this typically selects a large grid (e.g., `(8, 8)` = 64 cores). For a
single-batch 4-KV-head model, 60 of those cores are idle but still participate
in synchronization, adding measurable latency. Always provide an explicit
`SDPAProgramConfig` for production inference paths.
