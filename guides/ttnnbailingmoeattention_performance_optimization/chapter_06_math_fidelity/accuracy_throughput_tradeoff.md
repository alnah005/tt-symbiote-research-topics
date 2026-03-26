# Accuracy vs Throughput Tradeoff: Configuration Benchmarking

## Configurations to Benchmark

Three configurations should be benchmarked against the current Bailing MoE baseline:

| Config | math_fidelity | fp32_dest_acc_en | packer_l1_acc | Label |
|---|---|---|---|---|
| Baseline (current) | HiFi4 | True | True | A |
| Qwen3-style | HiFi4 | False | False | B |
| HiFi2 conservative | HiFi2 | True | True | C |
| HiFi2 + Qwen3-style | HiFi2 | False | False | D |

Config B is the lowest-risk change — same fidelity as the baseline, but with more dst tiles. Config D is the highest-throughput candidate but requires accuracy validation.

---

## Accuracy Measurement Methodology

To measure accuracy degradation, compare the SDPA output `attn_output` tensor between configurations using representative Bailing MoE inputs:

### Step 1: Run baseline (Config A)

Run `_forward_decode_paged` with the current `HiFi4 / fp32_dest_acc_en=True / packer_l1_acc=True` config on a held-out batch of inputs (B=1 and B=32 are the most important cases). Collect `attn_output` before the `dense` projection by inserting a `ttnn.to_torch` probe after `paged_sdpa_decode`.

### Step 2: Run candidate configs

Run the same inputs with each candidate config (B, C, D) and collect the corresponding `attn_output` tensors.

### Step 3: Compute error metrics

For each candidate config vs baseline:

```python
abs_err = torch.abs(attn_output_candidate - attn_output_baseline)
rel_err = abs_err / (torch.abs(attn_output_baseline) + 1e-6)

max_abs_err = abs_err.max().item()
mean_abs_err = abs_err.mean().item()
max_rel_err = rel_err.max().item()
p99_rel_err = torch.quantile(rel_err.flatten(), 0.99).item()
```

**Thresholds for safe adoption** (rough guidance; to be validated with generation quality tests):
- `max_abs_err < 1e-2` for the attention output in bfloat16 space
- `p99_rel_err < 1%` at B=32 decode

These are conservative thresholds. Models with QK norm tend to have bounded attention logit magnitudes, which typically keeps absolute errors below 1e-2 at HiFi2 fidelity.

### Step 4: End-to-end generation quality

If tensor-level error metrics pass, run a generation quality benchmark on Bailing MoE (e.g., perplexity on a standard eval set). Compare perplexity between the baseline and the candidate config. A perplexity degradation of more than 0.1 points is typically significant for production deployment.

---

## Throughput Measurement Methodology

Measure the SDPA kernel wall time under each configuration using `ttnn.synchronize_device` timing:

```python
ttnn.synchronize_device(device)
t0 = time.perf_counter()
for _ in range(N_REPEATS):
    attn_output = module.paged_sdpa_decode(query_states, layer_idx, ...)
ttnn.synchronize_device(device)
t1 = time.perf_counter()
sdpa_ms = (t1 - t0) / N_REPEATS * 1000
```

Use `N_REPEATS=100` to amortize Python overhead. Run at both B=1 (latency-bound regime) and B=32 (throughput regime, more representative of production).

**Key metrics to report:**
- SDPA kernel wall time (ms) per decode step
- Total `_forward_decode_paged` wall time, to measure the impact of SDPA improvement on the full step

The SDPA kernel is compute-bound during decode for long contexts (KV cache length > 1K tokens) and becomes bandwidth-bound for short contexts. The throughput gain from HiFi2 or `fp32_dest_acc_en=False` will be larger in the compute-bound regime.

---

## Expected Throughput Gains (Rough Estimates)

These are rough order-of-magnitude estimates based on the architectural changes; actual numbers require profiling per the methodology above.

| Config vs Baseline | Expected SDPA speedup | Expected full-step speedup |
|---|---|---|
| Config B (HiFi4, fp32=False, packer=False) | 10–20% | 3–7% |
| Config C (HiFi2, fp32=True, packer=True) | 30–50% | 10–15% |
| Config D (HiFi2, fp32=False, packer=False) | 50–80% | 15–25% |

Full-step speedup is smaller than SDPA speedup because SDPA is one of several ops in the decode path — the all_gather collectives, memory layout transitions, and host-device round-trips identified in Chapters 2–4 are also contributors to total decode latency.

---

## `q_chunk_size` and `k_chunk_size` in Decode Mode

When benchmarking configs B–D against the baseline, pin chunk sizes explicitly (override `q_chunk_size=0` to a fixed value) to isolate the fidelity/accumulator effect from any variation introduced by the kernel's automatic chunk selection. The current chunk sizes are documented in `index.md`'s Current Configuration section.

---

## `exp_approx_mode`: Should It Be Enabled?

Both Bailing MoE and Qwen3 currently use `exp_approx_mode=False`. Enabling this would replace the exact softmax exponential with a faster polynomial approximation.

The risk: in attention, extreme logit differences (e.g., when one token has logit +100 and all others have logit −∞) can cause the softmax to be very sharp, and approximation errors in the exponential can shift the top attention weight significantly. With QK norm applied, the logit magnitudes are bounded — this makes the approximation safer. However, the exact error bound depends on the empirical distribution of attention logits in Bailing MoE, which varies across layers and token positions.

Enabling `exp_approx_mode=True` while also switching to HiFi2 would combine two sources of approximation error. The recommendation is to evaluate these independently: first establish whether `fp32_dest_acc_en=False` (Config B) is safe, then evaluate HiFi2, and only then consider approximate exp.

---

**Next:** [Chapter 7 — Profiling Methodology and Optimization Roadmap](../chapter_07_profiling_and_roadmap/index.md)
