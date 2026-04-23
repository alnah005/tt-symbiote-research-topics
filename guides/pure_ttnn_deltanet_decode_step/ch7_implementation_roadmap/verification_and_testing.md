# Verification and Testing

This file defines the complete test matrix for validating the `pure_ttnn_deltanet_decode_step` implementation. It covers unit tests for each task from `task_list_and_priority.md`, an integration test on a representative sub-stack of Qwen3.6-35B-A3B, a trace-specific correctness test, and a performance regression test with latency targets from Chapter 6. All PCC assertions use `pearson_correlation` computed on flattened tensors.

**Notation:** `H = 4` (heads per T3K device), `d_k = d_v = 128`, `B = 1` (decode batch size). All tests run at B=1 unless otherwise noted.

---

## Unit Tests — Per Task

### Task 1 Unit Test: State tensor on-device

**Goal:** Verify that `recurrent_states[layer_idx]` is a `ttnn.Tensor` with the correct shape, layout, and memory config, and that it persists across two sequential decode calls without any host transfer.

**Test procedure:**

```python
def test_task1_state_on_device(model, mesh_device):
    # 1. Assert type and attributes
    S = model.kv_cache.recurrent_states[0]
    assert isinstance(S, ttnn.Tensor), "State must be a ttnn.Tensor"
    assert S.shape == [1, 4, 128, 128], f"Expected [1,4,128,128], got {S.shape}"
    assert S.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    assert S.layout == ttnn.TILE_LAYOUT

    # 2. Run two sequential decode calls with distinct random inputs
    out1 = model.decode_step(random_input_1)
    S_after_step1 = ttnn.to_torch(model.kv_cache.recurrent_states[0]).clone()

    out2 = model.decode_step(random_input_2)
    S_after_step2 = ttnn.to_torch(model.kv_cache.recurrent_states[0]).clone()

    # 3. State must differ between steps (it was updated)
    assert not torch.allclose(S_after_step1, S_after_step2), \
        "State did not update between decode steps"

    # 4. Verify no ttnn.to_torch / ttnn.from_torch was called during step 2
    # (Use the guarded wrapper from trace_integration_checklist.md Step 2)
```

**Pass criteria:** No `AssertionError`; no host-crossing guard exception during step 2.

---

### Task 3 Unit Test: Causal conv1d TTNN sequence

**Goal:** Verify that the TTNN `ttnn.slice + ttnn.concat + ttnn.mul + ttnn.sum` sequence produces numerically identical output to the `causal_conv1d_update` reference for 100 random inputs.

**Test procedure:**

```python
def test_task3_causal_conv1d(num_trials=100):
    for _ in range(num_trials):
        x_new    = torch.randn(1, 1, H, d_inner)
        conv_state = torch.randn(1, H, d_inner, conv_width)
        weight   = torch.randn(H, d_inner, 1, conv_width)

        # Reference
        ref_out, ref_state = causal_conv1d_update(x_new, conv_state, weight)

        # TTNN form
        x_tt    = ttnn.from_torch(x_new, ...)
        cs_tt   = ttnn.from_torch(conv_state, ...)
        w_tt    = ttnn.from_torch(weight, ...)
        tt_out, tt_state = ttnn_causal_conv1d_update(x_tt, cs_tt, w_tt)

        pcc_out   = pearson_correlation(ttnn.to_torch(tt_out).flatten(),   ref_out.flatten())
        pcc_state = pearson_correlation(ttnn.to_torch(tt_state).flatten(), ref_state.flatten())

        assert pcc_out   > 0.9999, f"Output PCC {pcc_out:.6f} < 0.9999"
        assert pcc_state > 0.9999, f"State PCC {pcc_state:.6f} < 0.9999"
```

**Pass criteria:** PCC > 0.9999 for both output and updated conv state across all 100 trials. The conv1d update is exact integer arithmetic (shift and copy, then pointwise multiply) — BF16 rounding applies only to the multiply, which should produce PCC well above 0.9999 on small tensors.

---

### Task 4 Unit Test: Gated RMSNorm TTNN

**Goal:** Verify that `ttnn.rms_norm + ttnn.silu + ttnn.mul` matches `FusedRMSNormSwishGate` within PCC > 0.999.

**Test procedure:**

```python
def test_task4_gated_rmsnorm(num_trials=50):
    rms_norm_ref = FusedRMSNormSwishGate(d_inner, eps=1e-5)

    for _ in range(num_trials):
        x = torch.randn(1, 1, H, d_inner)
        z = torch.randn(1, 1, H, d_inner)

        # Reference
        ref_out = rms_norm_ref(x, z)

        # TTNN form
        x_tt = ttnn.from_torch(x, ...)
        z_tt = ttnn.from_torch(z, ...)
        x_normed = ttnn.rms_norm(x_tt, weight=rms_weight_tt, epsilon=1e-5)
        gate     = ttnn.silu(z_tt)
        tt_out   = ttnn.mul(x_normed, gate)

        pcc = pearson_correlation(ttnn.to_torch(tt_out).flatten(), ref_out.flatten())
        assert pcc > 0.999, f"Gated RMSNorm PCC {pcc:.6f} < 0.999"
```

**Pass criteria:** PCC > 0.999 across all 50 trials.

---

### Task 5 Unit Test: Recurrent delta rule TTNN (composed form)

**Goal:** Run the 6-op TTNN recurrence for 200 decode steps and verify per-step PCC and cumulative state stability against the PyTorch reference.

**Test procedure:**

```python
def test_task5_recurrent_delta_rule(num_steps=200):
    S_ref = torch.zeros(1, H, d_k, d_v)
    S_tt  = ttnn.zeros(shape=[1, H, d_k, d_v], dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=mesh_device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)

    pcc_S_history = []
    pcc_o_history = []
    l2_history    = []

    for step in range(num_steps):
        q, k, v, g, beta = [torch.randn(...) for _ in range(5)]

        # PyTorch reference
        S_ref, o_ref = recurrent_gated_delta_rule(q, k, v, S_ref, g, beta)

        # TTNN composed form
        q_tt, k_tt, v_tt, g_tt, beta_tt = [ttnn.from_torch(t, ...) for t in [q, k, v, g, beta]]
        S_tt, o_tt = ttnn_deltanet_step(q_tt, k_tt, v_tt, S_tt, g_tt, beta_tt)

        S_tt_cpu = ttnn.to_torch(S_tt)
        o_tt_cpu = ttnn.to_torch(o_tt)

        pcc_S = pearson_correlation(S_tt_cpu.flatten(), S_ref.flatten())
        pcc_o = pearson_correlation(o_tt_cpu.flatten(), o_ref.flatten())
        l2    = torch.norm(S_tt_cpu - S_ref).item()

        pcc_S_history.append(pcc_S)
        pcc_o_history.append(pcc_o)
        l2_history.append(l2)

        assert pcc_S > 0.999, f"Step {step}: state PCC {pcc_S:.6f} < 0.999"
        assert pcc_o > 0.999, f"Step {step}: output PCC {pcc_o:.6f} < 0.999"

    # Cumulative state drift must be bounded (not monotonically growing)
    assert l2_history[-1] < 10 * l2_history[0] or l2_history[-1] < 1e-2, \
        f"State L2 drift grew unboundedly: step 0={l2_history[0]:.4f}, step {num_steps-1}={l2_history[-1]:.4f}"

    return pcc_S_history, pcc_o_history, l2_history
```

**Pass criteria:**
- `pcc_S > 0.999` at every step (0–199)
- `pcc_o > 0.999` at every step (0–199)
- L2 norm of state difference is bounded (does not grow monotonically across 200 steps)

---

### Task 6 Unit Test: Fused kernel correctness

**Goal:** Validate the fused `gdn_full_fused_inplace` kernel against both the PyTorch reference and the composed TTNN form.

**Test procedure:** Run the same 200-step test as Task 5, replacing the TTNN composed form with the fused kernel. Additionally, compare fused kernel output against composed TTNN form output (treating composed form as a trusted secondary reference after it passes its own test):

```python
def test_task6_fused_kernel(num_steps=200):
    # Same setup as test_task5_recurrent_delta_rule
    # ... (replace ttnn_deltanet_step with fused_gdn_step)

    # Additional: compare fused vs. composed
    for step in range(num_steps):
        o_fused  = fused_gdn_step(...)
        o_composed = ttnn_deltanet_step(...)
        pcc_fused_vs_composed = pearson_correlation(
            ttnn.to_torch(o_fused).flatten(),
            ttnn.to_torch(o_composed).flatten()
        )
        assert pcc_fused_vs_composed > 0.9999, \
            f"Step {step}: fused vs. composed PCC {pcc_fused_vs_composed:.6f} < 0.9999"
```

**Pass criteria:**
- All Task 5 pass criteria met (PCC > 0.999 vs. PyTorch reference, bounded state drift)
- Fused vs. composed TTNN PCC > 0.9999 at every step (the two on-device forms should agree very closely)

---

## Integration Test: 10-Layer Prefix of Qwen3.6-35B-A3B on T3K

**Goal:** Run a 10-layer decoder prefix (5 DeltaNet layers + 5 full-attention layers, representative slice of the 40-layer Qwen3.6-35B-A3B architecture) on T3K hardware with the full on-device implementation, and verify output logits against the reference (mixed PyTorch/TTNN baseline with host fallbacks).

**Test setup:**
- Load the first 10 layers of Qwen3.6-35B-A3B on T3K (1×8 mesh)
- Reference: the existing model with host fallbacks for DeltaNet ops (Tasks 1–5 not yet applied)
- Test: the model with all Tasks 1–5 applied

**Test procedure:**

```python
def test_integration_10layer(num_steps=20):
    logits_ref_all = []
    logits_tt_all  = []

    for step in range(num_steps):
        input_ids = torch.randint(0, vocab_size, [1, 1])

        logits_ref = model_reference.decode_step(input_ids)     # host fallback
        logits_tt  = model_on_device.decode_step(input_ids)     # on-device implementation

        pcc = pearson_correlation(
            ttnn.to_torch(logits_tt).flatten(),
            logits_ref.flatten()
        )
        assert pcc > 0.99, f"Step {step}: integration logits PCC {pcc:.6f} < 0.99"

        logits_ref_all.append(logits_ref)
        logits_tt_all.append(ttnn.to_torch(logits_tt))
```

**Pass criteria:** PCC > 0.99 between on-device implementation logits and host-fallback reference logits at every step across 20 decode steps.

The 0.99 threshold (vs. 0.999 for unit tests) reflects the additional BF16 rounding through the full attention and MLP layers; the DeltaNet contribution specifically should be well above this threshold based on unit test results.

---

## Trace-Specific Correctness Test

**Goal:** Verify that enabling Metal Trace (`ttnn.execute_trace`) does not change the numerical output of the 10-layer integration test.

**Test procedure:**

```python
def test_trace_integration(num_steps=20):
    # Setup trace (see trace_integration_checklist.md Steps 1–4)
    trace_id = setup_trace(model_on_device)

    for step in range(num_steps):
        input_ids = torch.randint(0, vocab_size, [1, 1])

        # Non-traced reference (on-device implementation, no trace)
        logits_nontrace = model_on_device.decode_step_nontrace(input_ids)

        # Traced execution
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        logits_traced = ttnn.to_torch(output_buffer)

        pcc = pearson_correlation(logits_traced.flatten(), logits_nontrace.flatten())
        assert pcc > 0.99, f"Step {step}: trace vs. non-trace PCC {pcc:.6f} < 0.99"
```

**Pass criteria:** `ttnn.execute_trace` completes without error at every step; output PCC > 0.99 between traced and non-traced on-device implementations across 20 steps.

---

## Performance Regression Test

**Goal:** Assert that the on-device implementation meets the latency targets from Chapter 6 after both the composed TTNN form and the fused kernel form are implemented.

**Measurement procedure:**
- Warm up: 100 decode steps (discard timing)
- Measure: record per-step decode latency for 1000 steps using `time.perf_counter_ns()`
- Isolate DeltaNet contribution: compare total decode step time against a reference run with DeltaNet layers replaced by identity operations (zeroed output, no state update) to isolate the DeltaNet-specific cost

**Latency targets (update after empirical measurement on T3K):**

| Form | DeltaNet contribution target (30 layers) |
|---|---|
| Host CPU fallback (baseline) | 9–21 ms (from Ch6 measurement) |
| Composed TTNN, no trace | < 2 ms |
| Composed TTNN, with trace | < 1 ms |
| Fused kernel, with trace | < 200 µs |

```python
def test_performance_regression(model, use_fused_kernel=False):
    latencies = measure_decode_latencies(model, steps=1000, warmup=100)

    deltanet_p50_ms = latencies["deltanet_p50_us"] / 1000

    if use_fused_kernel:
        assert deltanet_p50_ms < 0.200, \
            f"Fused kernel DeltaNet latency {deltanet_p50_ms:.3f} ms exceeds 200 µs target"
    else:
        assert deltanet_p50_ms < 2.0, \
            f"Composed TTNN DeltaNet latency {deltanet_p50_ms:.3f} ms exceeds 2 ms target"
```

**Pass criteria:** DeltaNet contribution p50 latency meets the applicable target for the form under test. Update the target values in this file after the first empirical measurement on T3K hardware (the analytic estimates from Chapter 6 may differ from actual hardware results by up to 2×).
