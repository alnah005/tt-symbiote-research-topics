# Common Pitfalls in Sparsity Tensor Construction

This file catalogs the six most common mistakes when constructing the sparsity tensor for `ttnn.sparse_matmul`. For each pitfall, the entry covers: symptom, root cause, detection method, and fix.

Cross-references to the format specification and construction pipeline are provided where relevant.

---

## P1 — Shape Mismatch Between Activation Tensor and Sparsity Tensor

**Symptom:** `ttnn.sparse_matmul` raises a shape validation error at call time. In some configurations (if the validation is not exhaustive), the call may succeed but produce silent wrong output because tile rows are mapped to the wrong expert slots.

**Root cause:** The mask has an incorrect $M_t$ (wrong value of $C$ was used to compute $\lceil C/32 \rceil$) or an incorrect $K_t$ (wrong value of $H$). Common causes:

- $C$ was computed from the *logical* token count rather than the *padded* activation tensor height. If the activation tensor is padded to the next multiple of 32 (e.g., $C=1$ padded to 32), the logical $M_t=1$ is correct, but if the activation is padded to $C=32$, $M_t$ remains 1 — that is fine. The problem arises when the padding changes the tile row count and the mask is not updated accordingly.
- $H$ was hardcoded to a wrong value (e.g., using $H=4096$ from a different model when the actual model has $H=7168$, giving $K_t=128$ instead of $K_t=224$).
- The mask was built for a single expert ($[M_t, K_t]$) but the activation tensor is the batched multi-expert form ($[E_d \times C, H]$), requiring mask shape $[E_d \times M_t, K_t]$.

**Detection:**

```python
import math

def assert_mask_shape(mask_ttnn, E_d: int, C: int, H: int):
    """Call this before every sparse_matmul during development."""
    M_t = math.ceil(C / 32)
    K_t = math.ceil(H / 32)
    expected = (E_d * M_t, K_t)
    actual = tuple(mask_ttnn.shape)
    assert actual == expected, (
        f"Sparsity tensor shape mismatch: expected {expected}, got {actual}. "
        f"Check E_d={E_d}, C={C} (M_t={M_t}), H={H} (K_t={K_t})."
    )
```

**Fix:** Always derive $M_t$ from the actual padded height of the activation tensor, not from the logical token count. After padding the activation tensor to its canonical shape, compute $M_t = \text{activation\_tensor.shape}[0] // (E_d \times 32)$.

```python
# Derive M_t from the padded activation tensor shape, not from logical C
padded_activation_height = activation_ttnn.shape[0]  # E_d * M_t * 32 after padding
M_t = padded_activation_height // (E_d * 32)
K_t = H // 32  # H must match the activation tensor's width exactly
```

See `sparsity_tensor_format.md`, Sections 1 and 7.

---

## P2 — Non-Tile-Aligned Token Counts Causing Incorrect Mask

**Symptom:** The model produces wrong outputs for experts where $C \bmod 32 \neq 0$. The error is silent — no TTNN exception is raised. The last few tokens routed to the affected expert are treated as if they do not exist, and the expert's computation for those tokens is skipped.

**Root cause:** When $C$ is not a multiple of 32, the last tile row of each expert's block in the flattened activation tensor is a partial tile — it contains $C \bmod 32$ real token slots and $32 - (C \bmod 32)$ padding zeros. If mask construction logic checks whether *all 32 slots* in a tile row are occupied (or checks only the padded-zero portion), it may incorrectly classify the partial tile row as inactive and set its mask value to `0`.

The most common instance is single-token decode with $B=1$, $S=1$:

- $C = \lceil 8 \times 1 \times 1 / 256 \rceil = 1$
- $M_t = \lceil 1 / 32 \rceil = 1$
- The activation tensor for expert $e$ has shape $[1, H]$ padded to $[32, H]$ (one real token, 31 padding rows)
- The single tile row is active if the 1 real token was routed to expert $e$
- If the mask construction checks `sum(tile_row) == 32` (all 32 slots occupied) instead of `sum(tile_row) > 0` (any slot occupied), it will always return 0 for this case, skipping every active expert

**Detection:**

```python
def validate_mask_vs_assignment(mask_torch, assignment, E_d, C, K_t):
    """
    Verify that every tile row containing a real token is marked active.
    mask_torch: [E_d * M_t, K_t], uint8
    assignment: [E_d, C], uint8
    """
    import math
    M_t = math.ceil(C / 32)
    padded = assignment.reshape(E_d * C)
    padded_ext = torch.zeros(E_d * M_t * 32, dtype=torch.uint8)
    padded_ext[:E_d * C] = padded

    for m in range(E_d * M_t):
        tile_has_token = padded_ext[m * 32: (m + 1) * 32].any().item()
        mask_val = mask_torch[m, 0].item()
        if tile_has_token and mask_val == 0:
            raise AssertionError(
                f"Tile row {m} has real tokens but mask is 0 — tokens will be silently dropped."
            )
        if not tile_has_token and mask_val == 1:
            # This is not wrong (computing an empty tile is wasteful but not incorrect)
            # but may indicate mask over-counting
            pass
```

**Fix:** Use `> 0` (not `== 32`) when determining whether a tile row is active:

```python
# Correct: any token in the tile row → active
tile_active = (tile_rows.sum(dim=1) > 0).to(torch.uint8)

# WRONG: requires all 32 slots to be occupied
# tile_active = (tile_rows.sum(dim=1) == 32).to(torch.uint8)
```

See `sparsity_tensor_format.md`, Section 7; `constructing_from_router_output.md`, Step 4.

---

## P3 — Forgetting to Update the Sparsity Tensor Between Decode Steps

**Symptom:** The model produces correct output for the first decode step but incorrect output for all subsequent steps. The error is silent and may manifest as repetitive or incoherent token generation. The magnitude of the error increases with the number of decode steps completed with the stale mask.

**Root cause:** The sparsity mask was computed once (e.g., at the start of generation) and reused in the KV-cache decode loop without being recomputed. Since routing decisions change every step (the router input changes as the KV-cache grows and the generated sequence evolves), the stale mask causes `sparse_matmul` to skip experts that are actually active for the current step and/or compute experts that are inactive.

This is the most common production bug for mask-based sparse MoE inference. It is particularly easy to introduce when refactoring a non-traced decode loop into a `ttnn.Trace`-based loop: the mask update logic may be accidentally placed outside the loop or omitted during the trace migration.

**Detection:**

```python
# Add a step counter check during development:
prev_mask = None
for step in range(max_new_tokens):
    routing_indices = router(current_embedding)
    new_mask = build_sparsity_mask(routing_indices, ...)

    if prev_mask is not None:
        # Verify that the mask changes between steps (it should, almost always)
        # A mask that never changes across 10+ steps is a strong signal of a stale-mask bug
        if (new_mask == prev_mask).all():
            import warnings
            warnings.warn(f"Step {step}: mask identical to previous step — possible stale mask bug")

    prev_mask = new_mask.clone()
    ttnn.copy_(mask_ttnn, ttnn.from_torch(new_mask, ...))
    ttnn.execute_trace(trace, mesh_device)
```

**Fix:** Recompute and overwrite the mask tensor at the start of every decode step, before executing the trace or calling `sparse_matmul` directly.

```python
for step in range(max_new_tokens):
    routing_indices = router(current_embedding)          # fresh router output
    new_mask_torch = build_sparsity_mask(routing_indices, ...)  # fresh mask
    ttnn.copy_(mask_ttnn, ttnn.from_torch(new_mask_torch, ...)) # overwrite buffer
    ttnn.execute_trace(trace, mesh_device)               # replay with fresh mask
```

See `sparsity_tensor_placement.md`, Section 6.

---

## P4 — Performance Regression from Placing Sparsity Tensor in DRAM Instead of L1

**Symptom:** `sparse_matmul` latency is higher than expected. TTNN profiler shows elevated DRAM bandwidth utilization during `sparse_matmul` kernel execution, inconsistent with the expected weight-streaming bandwidth profile. The regression is more pronounced at decode sizes where the kernel reads the mask frequently relative to weight tiles.

**Root cause:** The mask tensor was created with `ttnn.DRAM_MEMORY_CONFIG` (or with the default memory config if the default is DRAM). During kernel execution, every tile-row check requires a DRAM read to fetch the mask tile, adding NOC round-trip latency on each check.

A typical cause is copying mask creation code from a utility that uses DRAM as its default, or forgetting to specify `memory_config` in `ttnn.from_torch`.

**Detection:**

```python
# Check mask placement before calling sparse_matmul:
def assert_mask_in_l1(mask_ttnn):
    if mask_ttnn.memory_config() != ttnn.L1_MEMORY_CONFIG:
        import warnings
        warnings.warn(
            "Sparsity tensor is not in L1. This will add DRAM bandwidth overhead "
            "on every tile-row decision in sparse_matmul. "
            "Use memory_config=ttnn.L1_MEMORY_CONFIG for decode-sized masks."
        )
```

For profiling: use TTNN's built-in profiler and check the DRAM read bytes counter during the `sparse_matmul` op. A 7 KB mask in DRAM adds 7168 bytes of DRAM reads per call; across many decode steps this accumulates.

**Fix:** Use `ttnn.L1_MEMORY_CONFIG` when creating the mask tensor:

```python
mask_ttnn = ttnn.from_torch(
    mask_torch,
    dtype=ttnn.uint8,
    layout=ttnn.TILE_LAYOUT,
    device=mesh_device,
    memory_config=ttnn.L1_MEMORY_CONFIG,  # not ttnn.DRAM_MEMORY_CONFIG
)
```

For the decode regime, the mask is at most 7 KB (7 tiles) — there is no L1 capacity justification for DRAM placement. See `sparsity_tensor_placement.md`, Section 3.

---

## P5 — Wrong dtype for Mask Tensor (Float Instead of uint8)

**Symptom:** `ttnn.sparse_matmul` raises a dtype validation error at call time:

```
TTNNException: sparse_matmul: sparsity tensor must have dtype uint8, got bfloat16
```

(Exact error message format may vary by TTNN version.)

**Root cause:** The mask tensor was constructed with a floating-point dtype. Common causes:

- `torch.zeros(...)` or `torch.ones(...)` called without `dtype=torch.uint8` — defaults to `torch.float32`
- A boolean mask (`torch.bool`) passed directly to `ttnn.from_torch` without casting — TTNN may interpret it as a different dtype
- `ttnn.from_torch` called without specifying `dtype=ttnn.uint8`, allowing TTNN to infer the dtype from the PyTorch tensor (which may be `bfloat16` if the rest of the model uses bf16)

**Detection:** This pitfall produces a hard error, not a silent bug. However, if your `ttnn.from_torch` call specifies `dtype=ttnn.bfloat16` for the mask (matching the rest of the model's dtype), you will get the error. The detection is the error itself.

**Fix:** Explicitly specify `dtype=ttnn.uint8` at every point where the mask is created or converted:

```python
# Step 1: PyTorch mask — use torch.uint8
mask_torch = tile_active.unsqueeze(1).expand(E_d * M_t, K_t).to(torch.uint8)
# Do NOT use: .to(torch.float32) or .to(torch.bool)

# Step 2: TTNN conversion — explicitly set dtype
mask_ttnn = ttnn.from_torch(
    mask_torch,
    dtype=ttnn.uint8,     # required — do not omit or use ttnn.bfloat16
    layout=ttnn.TILE_LAYOUT,
    device=mesh_device,
    memory_config=ttnn.L1_MEMORY_CONFIG,
)
```

See `sparsity_tensor_format.md`, Section 3.

---

## P6 — Static Mask Shape Forced to Change When B or S Changes (Recompilation)

**Symptom:** The first call to `sparse_matmul` after a change in batch size $B$ or sequence length $S$ is significantly slower than subsequent calls (e.g., hundreds of milliseconds vs. microseconds). Profiling shows JIT kernel compilation on the first call. Subsequent calls with the same shapes are fast. If $B$ or $S$ changes frequently (e.g., dynamic batching), the system effectively recompiles on every call.

**Root cause:** The mask shape $[E_d \times M_t, K_t]$ depends on $M_t = \lceil C / 32 \rceil$ where $C = \lceil k \times B \times S / E \rceil$. Different values of $B$ or $S$ produce different $C$ values, which produce different $M_t$ values, which produce different mask shapes. TTNN's program cache is keyed on the full set of tensor shapes and configs; a new mask shape means a cache miss and forced recompilation.

For Qwen3.5-35B:

| $B$ | $S$ | $C$ | $M_t$ | Mask shape |
|-----|-----|-----|--------|------------|
| 1 | 1 | 1 | 1 | $[32, 224]$ |
| 8 | 1 | 1 | 1 | $[32, 224]$ |
| 32 | 1 | 1 | 1 | $[32, 224]$ |
| 4 | 2048 | 256 | 8 | $[256, 224]$ |
| 8 | 2048 | 512 | 16 | $[512, 224]$ |

Even within "decode mode" (S=1), different batch sizes can produce different $C$ values once $B$ is large enough that $\lceil k \times B / E \rceil > 1$. For $E=256$, $k=8$: $C > 1$ when $B > 32$.

**Detection:**

```python
# Log mask shape changes to detect recompilation triggers:
last_mask_shape = None
for step in range(max_new_tokens):
    new_mask_shape = (E_d * M_t, K_t)
    if last_mask_shape is not None and new_mask_shape != last_mask_shape:
        print(f"WARNING: Mask shape changed from {last_mask_shape} to {new_mask_shape} — recompilation will occur")
    last_mask_shape = new_mask_shape
```

**Fix:** Define a small set of canonical $(B, S, C)$ shapes that cover the expected operating range. Pad the activation tensor and mask to the nearest canonical shape. Padding rows in the mask should be all-zero (inactive).

```python
# Canonical shapes for Qwen3.5-35B (examples)
CANONICAL_CONFIGS = [
    # (M_t, mask_shape_0)  -- K_t=224 is always fixed
    (1,  32),    # decode: C=1, M_t=1, E_d*M_t=32
    (8,  256),   # prefill 2K: C=256, M_t=8, E_d*M_t=256
    (16, 512),   # prefill 4K: C=512, M_t=16, E_d*M_t=512
]

def get_canonical_mask_rows(actual_mask_rows: int) -> int:
    """Return the smallest canonical mask row count >= actual_mask_rows."""
    for m_t, canonical_rows in CANONICAL_CONFIGS:
        if canonical_rows >= actual_mask_rows:
            return canonical_rows
    raise ValueError(f"No canonical shape for {actual_mask_rows} mask rows — add a new canonical config")

# Pad mask to canonical size
actual_rows = E_d * M_t
canonical_rows = get_canonical_mask_rows(actual_rows)
if actual_rows < canonical_rows:
    padding = torch.zeros(canonical_rows - actual_rows, K_t, dtype=torch.uint8)
    mask_torch = torch.cat([mask_torch, padding], dim=0)
    # Padding rows are all-zero (inactive) — correct, no real tokens in padding rows
```

The activation tensor must also be padded to the corresponding canonical height before calling `sparse_matmul`.

> **Warning:** Failing to pad the *activation tensor* to match the padded mask shape will produce a shape mismatch error (P1). Always pad both together.

See `sparsity_tensor_format.md`, Section 9.

---

## Summary Table

| # | Pitfall | Is it silent? | Primary detection | Fix location |
|---|---------|--------------|-------------------|-------------|
| P1 | Shape mismatch | Sometimes | Assert mask shape before call | Derive $M_t$ from padded activation shape |
| P2 | Partial tile zeroed | Yes (silent) | Validate mask vs. assignment | Use `> 0` not `== 32` for tile-active check |
| P3 | Stale mask in decode loop | Yes (silent) | Check mask changes per step | Recompute mask every decode step |
| P4 | Mask in DRAM | No (performance) | TTNN profiler, latency regression | Use `L1_MEMORY_CONFIG` |
| P5 | Wrong dtype (float) | No (hard error) | TTNN dtype validation error | Use `dtype=ttnn.uint8` |
| P6 | Dynamic mask shape → recompilation | No (latency spike) | Log mask shape changes | Pad to canonical shapes |

---

## References

- `sparsity_tensor_format.md` — shape (P1, P6), partial tiles (P2), dtype (P5), static shape (P6)
- `constructing_from_router_output.md` — tile-active construction logic (P2)
- `sparsity_tensor_placement.md` — L1 placement (P4), TTNN Trace integration (P3)
- Chapter 4, `sparse_matmul_internals.md` — how the mask drives tile-skip decisions (context for P1, P2, P3)
