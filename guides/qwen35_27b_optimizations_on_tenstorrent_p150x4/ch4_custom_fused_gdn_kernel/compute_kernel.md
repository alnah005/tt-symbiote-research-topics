# Compute Kernel: L2 Norm, Gates, and Recurrence

The compute kernel (`gdn_fused.cpp`) is the heart of the fused GDN operation. For each pair, it executes five phases in sequence: L2 normalization of Q (with scaling), L2 normalization of K, K transpose, gate computation, and the full DeltaNet recurrence. It consumes inputs from the reader via circular buffers and produces two outputs for the writer: `cb_out` (`c_16`, Vt=4 output tiles) and `cb_state_out` (`c_8`, 16 updated state tiles).

The kernel runs with `fp32_dest_acc_en=True` and `MathFidelity.HiFi4` (verified at `gdn_kernel_op.py` lines 485–490), ensuring all intermediate matmul accumulations happen in FP32 before being packed back to bfloat16. This is critical for the recurrence, where small updates to a large state matrix can be lost to bfloat16 rounding over many tokens.

## Compile-Time Arguments

The kernel receives three compile-time arguments (line 483 of `gdn_kernel_op.py`):

- `Kt = 4`: key dimension tiles ($D_k / 32 = 128 / 32$)
- `Vt = 4`: value dimension tiles ($D_v / 32 = 128 / 32$)
- `num_pairs`: number of pairs assigned to this core (compile-time specialized per core group)

The derived constant `state_tiles = Kt * Vt = 16` is the number of tiles in a single pair's recurrence state.

## Kernel Structure

The kernel begins by waiting for the five persistent constants (pushed once by the reader), then enters the main per-pair loop:

```cpp
cb_wait_front(cb_norm_w,        Vt);
cb_wait_front(cb_scale,         1);
cb_wait_front(cb_rms_scale,     1);
cb_wait_front(cb_reduce_scaler, 1);
cb_wait_front(cb_rms_eps,       1);

for (uint32_t pair = 0; pair < num_pairs; pair++) {
    cb_wait_front(cb_q_raw,     Kt);
    cb_wait_front(cb_k_raw,     Kt);
    cb_wait_front(cb_v,         Vt);
    cb_wait_front(cb_a,         1);
    cb_wait_front(cb_b,         1);
    cb_wait_front(cb_neg_exp_A, 1);
    cb_wait_front(cb_dt_bias,   1);
    cb_wait_front(cb_state_in,  state_tiles);

    // Phase 1: L2 Norm Q
    // Phase 2: L2 Norm K
    // Phase 3: K Transpose
    // Phase 4: Gates
    // Phase 5: Recurrence
}

cb_pop_front(cb_norm_w,        Vt);
cb_pop_front(cb_scale,         1);
cb_pop_front(cb_rms_scale,     1);
cb_pop_front(cb_reduce_scaler, 1);
cb_pop_front(cb_rms_eps,       1);
```

The persistent constants are only popped after the loop ends.

> **Important — four CBs held but not consumed by compute phases:** `cb_norm_w`, `cb_rms_scale`, `cb_rms_eps`, and `cb_reduce_scaler` are waited for at startup and popped after the loop, but **none of the five compute phases use them**. Inspection of `gdn_fused.cpp` confirms that the kernel body contains no RMS norm phase — the comment at the top of the C++ file lists "6. RMS norm" and "7. SiLU gate" as intended phases, but neither is implemented in the kernel. Both operations run in Python post-kernel: `ttnn.rms_norm` (with `norm_w`) and `ttnn.silu` (for the SiLU gate) are called by `gdn.py` lines 330 and 334 after the kernel returns.
>
> The four CBs must still be allocated and pushed by the reader, because the compute kernel unconditionally calls `cb_wait_front` on them before entering the pair loop — removing them from the reader or CB table would deadlock the kernel. They sit in L1 as dead weight across all pairs and are released only at kernel teardown. `cb_scale` (`c_15`) is the only persistent constant that is actively used inside the compute phases (Phase 1, Step 4).
>
> An implementer extending this kernel to add on-device RMS norm would use these four CBs; until then, they serve as a reservation that prevents deadlock without contributing to computation.

## Phase 1: L2 Normalize Q with Scale

**Goal:** Compute $q = q_{\text{raw}} / \|q_{\text{raw}}\|_2 \cdot D_k^{-0.5}$ and store in `cb_q`.

Rather than using a separate reduce operation, the kernel computes the dot product $q_{\text{raw}} \cdot q_{\text{raw}}^T$ via matmul, which produces $\sum q_i^2$ in a single [1,1] result.

**Step 1 — Transpose $q_{\text{raw}}$** into `cb_sq_acc` for use as the right-hand operand:

```cpp
cb_reserve_back(cb_sq_acc, Kt);
for (uint32_t kt = 0; kt < Kt; kt++) {
    tile_regs_acquire();
    transpose_wh_init_short(cb_q_raw);
    transpose_wh_tile(cb_q_raw, kt, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_sq_acc, kt);
    tile_regs_release();
}
cb_push_back(cb_sq_acc, Kt);
```

**Step 2 — Dot product via matmul** $[1, D_k] \times [D_k, 1] = [1, 1]$:

```cpp
mm_init(cb_q_raw, cb_sq_acc, cb_tmp);
tile_regs_acquire();
for (uint32_t kt = 0; kt < Kt; kt++) {
    matmul_tiles(cb_q_raw, cb_sq_acc, kt, kt, 0);  // accumulates into DST[0]
}
tile_regs_commit();
tile_regs_wait();
pack_tile(0, cb_tmp);
tile_regs_release();
cb_pop_front(cb_sq_acc, Kt);
```

The loop over `kt` accumulates partial products from each tile pair into DST register 0 (FP32 accumulation). This avoids the `reduce_row` operation entirely.

**Step 3 — Inverse square root:**

```cpp
copy_tile(cb_tmp, 0, 0);
rsqrt_tile_init();
rsqrt_tile(0);  // DST[0] = 1 / sqrt(sum_sq)
```

**Step 4 — Multiply by scale** ($D_k^{-0.5}$ from `cb_scale`):

```cpp
mul_tiles_bcast_scalar_init_short(cb_tmp, cb_scale);
mul_tiles_bcast_scalar(cb_tmp, cb_scale, 0, 0, 0);
```

This produces a combined factor $\|q_{\text{raw}}\|_2^{-1} \cdot D_k^{-0.5}$ in DST[0].

**Step 5 — Apply to all Q tiles** via scalar broadcast into `cb_q`:

```cpp
cb_reserve_back(cb_q, Kt);
mul_tiles_bcast_scalar_init_short(cb_q_raw, cb_tmp);
for (uint32_t kt = 0; kt < Kt; kt++) {
    tile_regs_acquire();
    mul_tiles_bcast_scalar(cb_q_raw, cb_tmp, kt, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_q, kt);
    tile_regs_release();
}
cb_push_back(cb_q, Kt);
cb_pop_front(cb_tmp, 1);
cb_pop_front(cb_q_raw, Kt);
```

## Phase 2: L2 Normalize K

**Goal:** Compute $k = k_{\text{raw}} / \|k_{\text{raw}}\|_2$ and store in `cb_k_row`.

This phase is structurally identical to Phase 1 — transpose, dot product, rsqrt, broadcast multiply — but without the scale multiplication in Step 4. The raw inverse norm $\|k_{\text{raw}}\|_2^{-1}$ is broadcast directly onto the K tiles. After this phase, `cb_k_raw` is freed, and `cb_k_row` contains the normalized K row vector.

## Phase 3: K Transpose

**Goal:** Produce $k_{\text{col}} = k_{\text{row}}^T$ for the outer product in the recurrence.

```cpp
cb_wait_front(cb_k_row, Kt);
cb_reserve_back(cb_k_col, Kt);
for (uint32_t kt = 0; kt < Kt; kt++) {
    tile_regs_acquire();
    transpose_wh_init_short(cb_k_row);
    transpose_wh_tile(cb_k_row, kt, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_k_col, kt);
    tile_regs_release();
}
cb_push_back(cb_k_col, Kt);
```

`cb_k_row` is **not** freed here — it is still needed for step 5.3 ($kv_{\text{mem}} = k_{\text{row}} \cdot \text{state}$). It is freed at the end of the recurrence phase.

## Phase 4: Gate Computation

**Goal:** Compute $\beta = \sigma(b)$ and $g = -\exp(A) \cdot \text{softplus}(a + \Delta_{\text{bias}})$.

### Beta gate

```cpp
cb_reserve_back(cb_beta, 1);
tile_regs_acquire();
copy_tile_init(cb_b);
copy_tile(cb_b, 0, 0);    // DST[0] = b
sigmoid_tile_init();
sigmoid_tile(0);           // DST[0] = sigmoid(b)
tile_regs_commit();
tile_regs_wait();
pack_tile(0, cb_beta);
tile_regs_release();
cb_push_back(cb_beta, 1);
cb_pop_front(cb_b, 1);
```

### Decay gate

The softplus $\log(1 + e^x)$ is decomposed into primitives:

```cpp
add_tiles(cb_a, cb_dt_bias, 0, 0, 0);  // DST[0] = a + dt_bias
exp_tile(0);                            // DST[0] = exp(a + dt_bias)
log1p_tile(0);                          // DST[0] = log(1 + exp(a + dt_bias))
pack_tile(0, cb_g);
```

Then the result is multiplied by `neg_exp_A` using an in-place pop-and-repack pattern:

```cpp
cb_wait_front(cb_g, 1);
tile_regs_acquire();
mul_tiles(cb_g, cb_neg_exp_A, 0, 0, 0);
tile_regs_commit();
cb_pop_front(cb_g, 1);
cb_reserve_back(cb_g, 1);
pack_tile(0, cb_g);
tile_regs_release();
cb_push_back(cb_g, 1);
cb_pop_front(cb_neg_exp_A, 1);
```

The final `cb_g` contains $g = -\exp(A) \cdot \text{softplus}(a + \Delta_{\text{bias}})$, a negative value representing the log-space decay rate.

## Phase 5: DeltaNet Recurrence

This phase implements the five recurrence equations. The full math is:

$$\text{state}_b = e^g \cdot \text{state}$$

$$kv_{\text{mem}} = k_{\text{row}} \cdot \text{state}_b$$

$$\delta_s = \beta \cdot (v - kv_{\text{mem}})$$

$$\text{state}_{\text{out}} = \text{state}_b + k_{\text{col}} \otimes \delta_s$$

$$\text{output} = q \cdot \text{state}_{\text{out}}$$

### Step 5.1: Exponential Decay Factor

```cpp
copy_tile(cb_g, 0, 0);
exp_tile(0);              // DST[0] = exp(g), value in (0,1) since g < 0
pack_tile(0, cb_exp_g);
```

### Step 5.2: State Decay

Each of the 16 state tiles is multiplied by the scalar $e^g$ via `mul_tiles_bcast_scalar`:

```cpp
mul_tiles_bcast_scalar_init_short(cb_state_in, cb_exp_g);
for (uint32_t s = 0; s < state_tiles; s++) {
    tile_regs_acquire();
    mul_tiles_bcast_scalar(cb_state_in, cb_exp_g, s, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_state_b, s);
    tile_regs_release();
}
cb_pop_front(cb_state_in, state_tiles);
cb_pop_front(cb_exp_g, 1);
```

### Step 5.3: Key-Value Memory Readout

Computes $kv_{\text{mem}} = k_{\text{row}} \cdot \text{state}_b$ as $[1, D_k] \times [D_k, D_v] = [1, D_v]$:

```cpp
mm_init(cb_k_row, cb_state_b, cb_kv_mem);
for (uint32_t vt = 0; vt < Vt; vt++) {
    tile_regs_acquire();
    for (uint32_t kt = 0; kt < Kt; kt++) {
        matmul_tiles(cb_k_row, cb_state_b, kt, kt * Vt + vt, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_kv_mem, vt);
    tile_regs_release();
}
```

The outer loop iterates over the Vt=4 output tiles. For each output tile, the inner loop accumulates partial products across the Kt=4 input tiles. The state tile index `kt * Vt + vt` maps the 2D $[D_k, D_v]$ layout into the linear tile array.

### Step 5.4: Delta Computation

**Subtraction** $\delta = v - kv_{\text{mem}}$:

```cpp
sub_tiles_init(cb_v, cb_kv_mem);
for (uint32_t vt = 0; vt < Vt; vt++) {
    sub_tiles(cb_v, cb_kv_mem, vt, vt, 0);
    pack_tile(0, cb_delta, vt);
}
cb_pop_front(cb_kv_mem, Vt);
```

**Beta scaling** $\delta_s = \beta \cdot \delta$:

```cpp
mul_tiles_bcast_scalar_init_short(cb_delta, cb_beta);
for (uint32_t vt = 0; vt < Vt; vt++) {
    mul_tiles_bcast_scalar(cb_delta, cb_beta, vt, 0, 0);
    pack_tile(0, cb_delta_s, vt);
}
cb_pop_front(cb_delta, Vt);
```

### Step 5.5: State Update via Outer Product

```cpp
cb_wait_front(cb_delta_s, Vt);
cb_reserve_back(cb_state_out, state_tiles);
for (uint32_t kt = 0; kt < Kt; kt++) {
    for (uint32_t vt = 0; vt < Vt; vt++) {
        uint32_t sidx = kt * Vt + vt;
        tile_regs_acquire();
        copy_tile_to_dst_init_short(cb_state_b);
        copy_tile(cb_state_b, sidx, 0);              // DST[0] = state_b[kt][vt]
        mm_init_short(cb_k_col, cb_delta_s);
        matmul_tiles(cb_k_col, cb_delta_s, kt, vt, 0); // DST[0] += k_col[kt] * delta_s[vt]
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_state_out, sidx);
        tile_regs_release();
    }
}
cb_push_back(cb_state_out, state_tiles);
cb_pop_front(cb_state_b,  state_tiles);
cb_pop_front(cb_delta_s,  Vt);
```

This uses a **copy + matmul accumulate** pattern: `copy_tile` loads `state_b[kt][vt]` into DST[0], then `matmul_tiles` computes $k_{\text{col}}[kt] \times \delta_s[vt]$ and accumulates into DST[0]. Because `matmul_tiles` accumulates rather than overwrites, the sequence produces $\text{state}_b[kt][vt] + k_{\text{col}}[kt] \times \delta_s[vt]$ in a single pass without materializing the outer product as a separate tensor.

### Step 5.6: Query Readout

Computes $\text{output} = q \cdot \text{state}_{\text{out}}$ as $[1, D_k] \times [D_k, D_v] = [1, D_v]$:

```cpp
cb_wait_front(cb_state_out, state_tiles);
cb_reserve_back(cb_out, Vt);
mm_init(cb_q, cb_state_out, cb_out);
for (uint32_t vt = 0; vt < Vt; vt++) {
    tile_regs_acquire();
    for (uint32_t kt = 0; kt < Kt; kt++) {
        matmul_tiles(cb_q, cb_state_out, kt, kt * Vt + vt, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out, vt);
    tile_regs_release();
}
cb_push_back(cb_out, Vt);
```

Result is written directly to `cb_out` (`c_16`), which the writer drains. An earlier version of the kernel used a separate `cb_rec_out` (`c_30`) intermediate buffer; this was removed to save one CB and eliminate a copy step.

### End of Pair

After the recurrence, remaining per-pair CBs are freed:

```cpp
cb_pop_front(cb_q,     Kt);
cb_pop_front(cb_k_row, Kt);
cb_pop_front(cb_k_col, Kt);
cb_pop_front(cb_v,     Vt);
cb_pop_front(cb_g,     1);
cb_pop_front(cb_beta,  1);
```

`cb_out` (Vt tiles) and `cb_state_out` (16 tiles) are now available for the writer. The compute kernel loops back to wait for the next pair's inputs from the reader.

## CB Dataflow Summary

```
Reader fills:     cb_q_raw, cb_k_raw, cb_v, cb_a, cb_b,
                  cb_neg_exp_A, cb_dt_bias, cb_state_in

Compute produces: cb_q         (from cb_q_raw + cb_scale)
                  cb_k_row     (from cb_k_raw)
                  cb_k_col     (from cb_k_row)
                  cb_beta      (from cb_b)
                  cb_g         (from cb_a + cb_dt_bias + cb_neg_exp_A)
                  cb_state_b   (from cb_state_in + cb_exp_g)
                  cb_state_out (from cb_state_b + cb_k_col + cb_delta_s)
                  cb_out       (from cb_q + cb_state_out)

Writer drains:    cb_out, cb_state_out
```

---

**Next:** [`writer_kernel.md`](./writer_kernel.md)
