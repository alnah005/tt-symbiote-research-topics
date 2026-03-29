# Compute Kernel: L2 Norm, Gates, and Recurrence

The compute kernel (`gdn_fused.cpp`) is the heart of the fused GDN operation. For each pair, it executes five phases in sequence: L2 normalization of Q (with scaling), L2 normalization of K, K transpose, gate computation, and the full DeltaNet recurrence. It consumes inputs from the reader via circular buffers and produces two outputs for the writer: `cb_out` (`c_16`, Vt=4 output tiles) and `cb_state_out` (`c_8`, 16 updated state tiles).

The kernel runs with `fp32_dest_acc_en=True` and `MathFidelity.HiFi4`, ensuring that all intermediate matmul accumulations happen in FP32 before being packed back to bfloat16. This is critical for the recurrence, where small updates to a large state matrix can be lost to bfloat16 rounding over many tokens.

## Compile-Time Arguments and CB Assignments

The kernel receives three compile-time arguments:

- `Kt = 4`: key dimension tiles (`Dk / 32 = 128 / 32`)
- `Vt = 4`: value dimension tiles (`Dv / 32 = 128 / 32`)
- `num_pairs`: number of pairs assigned to this core (compile-time specialized per core group)

The derived constant `state_tiles = Kt * Vt = 16` is the number of tiles in a single pair's recurrence state.

The kernel uses 22 distinct CB indices, divided into three categories:

**Reader-filled inputs (per pair):**
- `cb_q_raw` (`c_0`): raw Q head tiles, Kt tiles
- `cb_k_raw` (`c_1`): raw K head tiles, Kt tiles
- `cb_v` (`c_3`): V head tiles, Vt tiles
- `cb_a` (`c_9`): gate input a, 1 tile
- `cb_b` (`c_10`): gate input b, 1 tile
- `cb_neg_exp_A` (`c_12`): -exp(A) constant, 1 tile
- `cb_dt_bias` (`c_13`): dt_bias constant, 1 tile
- `cb_state_in` (`c_6`): recurrence state, 16 tiles

**Persistent constants (read once by reader):**
- `cb_norm_w` (`c_14`): RMS norm weight, Vt tiles
- `cb_scale` (`c_15`): Q scale = Dk^(-0.5), 1 tile
- `cb_rms_scale` (`c_31`): sqrt(Dv), 1 tile
- `cb_reduce_scaler` (`c_19`): all-ones tile, 1 tile
- `cb_rms_eps` (`c_20`): Dv * eps, 1 tile

**Compute intermediates and outputs:**
- `cb_q` (`c_17`): L2-normed and scaled Q, Kt tiles
- `cb_k_row` (`c_18`): L2-normed K, Kt tiles
- `cb_k_col` (`c_2`): K transposed, Kt tiles
- `cb_g` (`c_4`): computed decay gate, 1 tile
- `cb_beta` (`c_5`): sigmoid(b), 1 tile
- `cb_state_b` (`c_7`): decayed state, 16 tiles
- `cb_state_out` (`c_8`): updated state (to writer), 16 tiles
- `cb_out` (`c_16`): final output (to writer), Vt tiles
- `cb_exp_g` (`c_24`): exp(g), 1 tile
- `cb_kv_mem` (`c_25`): k_row @ state, Vt tiles
- `cb_delta` (`c_26`): v - kv_mem, Vt tiles
- `cb_delta_s` (`c_27`): beta * delta, Vt tiles
- `cb_sq_acc` (`c_28`): squared norm accumulator, Kt tiles
- `cb_tmp` (`c_29`): scratch for norm computation, 1 tile

## Kernel Structure

The kernel begins by waiting for the five persistent constants, then enters the main per-pair loop:

```cpp
// Wait for persistent constants (reader pushes once)
cb_wait_front(cb_norm_w, Vt);
cb_wait_front(cb_scale, 1);
cb_wait_front(cb_rms_scale, 1);
cb_wait_front(cb_reduce_scaler, 1);
cb_wait_front(cb_rms_eps, 1);

for (uint32_t pair = 0; pair < num_pairs; pair++) {
    // Wait for all per-pair inputs from reader
    cb_wait_front(cb_q_raw, Kt);
    cb_wait_front(cb_k_raw, Kt);
    cb_wait_front(cb_v, Vt);
    cb_wait_front(cb_a, 1);
    cb_wait_front(cb_b, 1);
    cb_wait_front(cb_neg_exp_A, 1);
    cb_wait_front(cb_dt_bias, 1);
    cb_wait_front(cb_state_in, state_tiles);

    // Phase 1: L2 Norm Q
    // Phase 2: L2 Norm K
    // Phase 3: K Transpose
    // Phase 4: Gates
    // Phase 5: Recurrence
}

// Pop persistent constants
cb_pop_front(cb_norm_w, Vt);
cb_pop_front(cb_scale, 1);
// ...
```

Each phase consumes specific CBs and produces results in other CBs. The persistent constants are only popped after the loop ends.

## Phase 1: L2 Normalize Q with Scale

**Goal:** Compute `q = q_raw / ||q_raw||_2 * Dk^(-0.5)` and store in `cb_q`.

The standard approach to L2 normalization would use element-wise square, row-reduce sum, rsqrt, and broadcast multiply. The fused kernel uses a more efficient approach: compute the dot product `q_raw @ q_raw^T` via matmul, which produces the sum of squares in a single `[1,1]` result without needing a separate reduce operation.

**Step 1 -- Transpose q_raw:**

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

This transposes each of the Kt=4 tiles from `cb_q_raw` into `cb_sq_acc`. The transposed tiles form the right-hand operand of the dot product.

**Step 2 -- Dot product via matmul:**

```cpp
cb_wait_front(cb_sq_acc, Kt);
cb_reserve_back(cb_tmp, 1);
mm_init(cb_q_raw, cb_sq_acc, cb_tmp);
tile_regs_acquire();
for (uint32_t kt = 0; kt < Kt; kt++) {
    matmul_tiles(cb_q_raw, cb_sq_acc, kt, kt, 0);
}
tile_regs_commit();
tile_regs_wait();
pack_tile(0, cb_tmp);
tile_regs_release();
cb_push_back(cb_tmp, 1);
cb_pop_front(cb_sq_acc, Kt);
```

The matmul computes `[1, Dk] x [Dk, 1] = [1, 1]`, which is exactly `sum(q_raw^2)`. The loop over `kt` accumulates partial products from each tile pair into DST register 0 (FP32 accumulation). This avoids the `reduce_row` operation entirely, which is a significant optimization since reduce operations have higher latency than matmul tiles.

**Step 3 -- Inverse square root:**

```cpp
cb_wait_front(cb_tmp, 1);
tile_regs_acquire();
copy_tile_init(cb_tmp);
copy_tile(cb_tmp, 0, 0);
rsqrt_tile_init();
rsqrt_tile(0);
tile_regs_commit();
// ...pack to cb_tmp...
```

Computes `1 / sqrt(sum_sq)` in the DST register.

**Step 4 -- Multiply by scale:**

```cpp
mul_tiles_bcast_scalar_init_short(cb_tmp, cb_scale);
mul_tiles_bcast_scalar(cb_tmp, cb_scale, 0, 0, 0);
```

Multiplies the inverse norm by `Dk^(-0.5)` to produce a combined scaling factor. This is `cb_scale` from the persistent constants (`c_15`).

**Step 5 -- Apply to all Q tiles:**

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

Each Q tile is multiplied by the combined `inv_norm * scale` factor using scalar broadcast. The result goes into `cb_q` (`c_17`), and `cb_q_raw` is freed.

## Phase 2: L2 Normalize K

**Goal:** Compute `k = k_raw / ||k_raw||_2` and store in `cb_k_row`.

This phase is structurally identical to Phase 1 (transpose, dot product, rsqrt, broadcast multiply) but without the scale multiplication in Step 4. The result goes into `cb_k_row` (`c_18`).

The key difference from Phase 1:

```cpp
// Step 4: Normalize K: k[kt] = k_raw[kt] * rsqrt(sum_sq)
cb_wait_front(cb_tmp, 1);
cb_reserve_back(cb_k_row, Kt);
mul_tiles_bcast_scalar_init_short(cb_k_raw, cb_tmp);
for (uint32_t kt = 0; kt < Kt; kt++) {
    // ...broadcast multiply by inv_norm only, no scale...
}
```

After this phase, `cb_k_raw` is freed. Both `cb_q` and `cb_k_row` now contain normalized vectors.

## Phase 3: K Transpose

**Goal:** Produce `k_col = k_row^T` for the outer product in the recurrence.

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

Each of the Kt=4 tiles in `cb_k_row` is transposed via `transpose_wh_tile` and stored in `cb_k_col` (`c_2`). The `k_row` CB is **not** freed here -- it is still needed for step 5.3 (`kv_mem = k_row @ state`). It is freed at the end of the recurrence phase.

## Phase 4: Gate Computation

**Goal:** Compute `beta = sigmoid(b)` and `g = neg_exp_A * softplus(a + dt_bias)`.

### Beta gate

```cpp
cb_reserve_back(cb_beta, 1);
tile_regs_acquire();
copy_tile_init(cb_b);
copy_tile(cb_b, 0, 0);       // DST[0] = b
sigmoid_tile_init();
sigmoid_tile(0);              // DST[0] = sigmoid(b)
tile_regs_commit();
tile_regs_wait();
pack_tile(0, cb_beta);
tile_regs_release();
cb_push_back(cb_beta, 1);
cb_pop_front(cb_b, 1);
```

Simple sigmoid applied to the b scalar tile. Result in `cb_beta` (`c_5`).

### Decay gate

The softplus computation `log(1 + exp(x))` is decomposed into primitive tile operations:

```cpp
cb_reserve_back(cb_g, 1);
tile_regs_acquire();
add_tiles_init(cb_a, cb_dt_bias);
add_tiles(cb_a, cb_dt_bias, 0, 0, 0);  // DST[0] = a + dt_bias
exp_tile_init();
exp_tile(0);                             // DST[0] = exp(a + dt_bias)
log1p_tile_init();
log1p_tile(0);                           // DST[0] = log(1 + exp(a + dt_bias))
tile_regs_commit();
tile_regs_wait();
pack_tile(0, cb_g);
tile_regs_release();
cb_push_back(cb_g, 1);
cb_pop_front(cb_a, 1);
cb_pop_front(cb_dt_bias, 1);
```

Then the result is multiplied by `neg_exp_A`:

```cpp
cb_wait_front(cb_g, 1);
tile_regs_acquire();
mul_tiles_init(cb_g, cb_neg_exp_A);
mul_tiles(cb_g, cb_neg_exp_A, 0, 0, 0);
tile_regs_commit();
tile_regs_wait();
cb_pop_front(cb_g, 1);
cb_reserve_back(cb_g, 1);
pack_tile(0, cb_g);
tile_regs_release();
cb_push_back(cb_g, 1);
cb_pop_front(cb_neg_exp_A, 1);
```

Note the in-place pattern: `cb_g` is popped, the result is packed back into `cb_g` via a fresh `cb_reserve_back`. This reuses the same CB without needing an additional buffer. The final `cb_g` contains `g = -exp(A) * softplus(a + dt_bias)`, which is a negative value representing the decay rate.

## Phase 5: DeltaNet Recurrence

This phase implements the five recurrence equations from Chapter 3 in tile operations.

### Step 5.1: Exponential Decay Factor

```cpp
cb_wait_front(cb_g, 1);
cb_reserve_back(cb_exp_g, 1);
tile_regs_acquire();
copy_tile_init(cb_g);
copy_tile(cb_g, 0, 0);
exp_tile_init();
exp_tile(0);                  // DST[0] = exp(g)
tile_regs_commit();
tile_regs_wait();
pack_tile(0, cb_exp_g);
tile_regs_release();
cb_push_back(cb_exp_g, 1);
```

Since `g` is negative, `exp(g)` produces a value in `(0, 1)` -- the decay factor.

### Step 5.2: State Decay

```cpp
cb_wait_front(cb_exp_g, 1);
cb_reserve_back(cb_state_b, state_tiles);
mul_tiles_bcast_scalar_init_short(cb_state_in, cb_exp_g);
for (uint32_t s = 0; s < state_tiles; s++) {
    tile_regs_acquire();
    mul_tiles_bcast_scalar(cb_state_in, cb_exp_g, s, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_state_b, s);
    tile_regs_release();
}
cb_push_back(cb_state_b, state_tiles);
cb_pop_front(cb_state_in, state_tiles);
cb_pop_front(cb_exp_g, 1);
```

Each of the 16 state tiles is multiplied by the scalar `exp(g)` using `mul_tiles_bcast_scalar`. This is the `state *= exp(g)` decay step. The decayed state goes into `cb_state_b` (`c_7`), and `cb_state_in` is freed.

### Step 5.3: Key-Value Memory Readout

```cpp
cb_wait_front(cb_state_b, state_tiles);
cb_reserve_back(cb_kv_mem, Vt);
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
cb_push_back(cb_kv_mem, Vt);
```

Computes `kv_mem = k_row @ state_b` as `[1, Dk] x [Dk, Dv] = [1, Dv]`. The outer loop iterates over the Vt=4 output tiles. For each output tile, the inner loop accumulates partial products across the Kt=4 input tiles. The state tile index `kt * Vt + vt` maps the 2D `[Dk, Dv]` layout into the linear tile array.

### Step 5.4: Delta Computation

**5.4a -- Subtraction:**

```cpp
cb_wait_front(cb_kv_mem, Vt);
cb_reserve_back(cb_delta, Vt);
sub_tiles_init(cb_v, cb_kv_mem);
for (uint32_t vt = 0; vt < Vt; vt++) {
    tile_regs_acquire();
    sub_tiles(cb_v, cb_kv_mem, vt, vt, 0);  // v - kv_mem
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_delta, vt);
    tile_regs_release();
}
cb_push_back(cb_delta, Vt);
cb_pop_front(cb_kv_mem, Vt);
```

**5.4b -- Beta scaling:**

```cpp
cb_wait_front(cb_beta, 1);
cb_wait_front(cb_delta, Vt);
cb_reserve_back(cb_delta_s, Vt);
mul_tiles_bcast_scalar_init_short(cb_delta, cb_beta);
for (uint32_t vt = 0; vt < Vt; vt++) {
    tile_regs_acquire();
    mul_tiles_bcast_scalar(cb_delta, cb_beta, vt, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_delta_s, vt);
    tile_regs_release();
}
cb_push_back(cb_delta_s, Vt);
cb_pop_front(cb_delta, Vt);
```

This computes `delta_s = beta * (v - kv_mem)` -- the scaled innovation.

### Step 5.5: State Update via Outer Product

```cpp
cb_wait_front(cb_delta_s, Vt);
cb_reserve_back(cb_state_out, state_tiles);
for (uint32_t kt = 0; kt < Kt; kt++) {
    for (uint32_t vt = 0; vt < Vt; vt++) {
        uint32_t sidx = kt * Vt + vt;
        tile_regs_acquire();
        copy_tile_to_dst_init_short(cb_state_b);
        copy_tile(cb_state_b, sidx, 0);            // DST[0] = state_b[kt][vt]
        mm_init_short(cb_k_col, cb_delta_s);
        matmul_tiles(cb_k_col, cb_delta_s, kt, vt, 0);  // DST[0] += k_col[kt] * delta_s[vt]
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_state_out, sidx);
        tile_regs_release();
    }
}
cb_push_back(cb_state_out, state_tiles);
cb_pop_front(cb_state_b, state_tiles);
cb_pop_front(cb_delta_s, Vt);
```

This is the most subtle operation in the kernel. It computes `state_out = state_b + outer(k_col, delta_s)` using a **copy + matmul accumulate** pattern:

1. `copy_tile(cb_state_b, sidx, 0)` loads the decayed state tile into DST[0]
2. `matmul_tiles(cb_k_col, cb_delta_s, kt, vt, 0)` computes `k_col[kt] * delta_s[vt]` and **accumulates** the result into DST[0]

Because `matmul_tiles` accumulates into the destination register, the copy + matmul sequence produces `state_b[kt][vt] + k_col[kt] * delta_s[vt]` in a single pass without materializing the outer product as a separate tensor. This saves 16 tiles of L1 space and eliminates an additional add operation.

The nested loop iterates over all `Kt x Vt = 16` state tiles. For each tile, the `kt` index selects the K column tile and the `vt` index selects the delta tile, producing the correct element of the rank-1 outer product.

### Step 5.6: Query Readout

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

Computes `output = q @ state_out` as `[1, Dk] x [Dk, Dv] = [1, Dv]`. Same matmul pattern as step 5.3 but using the normalized Q vector and the updated state. The result is written directly to `cb_out` (`c_16`), which the writer kernel drains.

Note that in earlier versions of the kernel, there was a separate `cb_rec_out` (`c_30`) intermediate buffer. This was removed -- the recurrence output is now written directly to `cb_out`, saving one CB and eliminating a copy step.

### End of Pair

After the recurrence, the remaining per-pair CBs are freed:

```cpp
cb_pop_front(cb_q, Kt);
cb_pop_front(cb_k_row, Kt);
cb_pop_front(cb_k_col, Kt);
cb_pop_front(cb_v, Vt);
cb_pop_front(cb_g, 1);
cb_pop_front(cb_beta, 1);
```

At this point, `cb_out` (Vt tiles) and `cb_state_out` (16 tiles) are available for the writer. The compute kernel loops back to wait for the next pair's inputs from the reader.

## CB Dataflow Summary

The following diagram shows the CB producer-consumer relationships:

```
Reader fills:     cb_q_raw, cb_k_raw, cb_v, cb_a, cb_b,
                  cb_neg_exp_A, cb_dt_bias, cb_state_in

Compute produces: cb_q (from cb_q_raw + cb_scale)
                  cb_k_row (from cb_k_raw)
                  cb_k_col (from cb_k_row)
                  cb_beta (from cb_b)
                  cb_g (from cb_a + cb_dt_bias + cb_neg_exp_A)
                  cb_state_b (from cb_state_in + cb_exp_g)
                  cb_state_out (from cb_state_b + cb_k_col + cb_delta_s)
                  cb_out (from cb_q + cb_state_out)

Writer drains:    cb_out, cb_state_out
```

The persistent constants (`cb_norm_w`, `cb_scale`, `cb_rms_scale`, `cb_reduce_scaler`, `cb_rms_eps`) are read by the compute kernel but never popped during the per-pair loop, remaining available for all pairs.

---

**Previous:** [`reader_kernel.md`](./reader_kernel.md) | **Next:** [`writer_kernel.md`](./writer_kernel.md)
