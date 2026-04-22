# GQA TP Constraint

## Overview

This file derives the TP≤2 limit for dots.ocr from first principles. The root cause is `num_key_value_heads=2`: because each device in a tensor-parallel group must hold at least one complete KV head, the maximum number of devices that can participate in attention sharding is exactly 2. Any TP degree above 2 triggers a structural shape assertion failure before any computation occurs.

### GQA Sharding Rule

In tt_transformers, grouped query attention (GQA) is sharded across TP devices as follows:

- Query heads are split evenly: each device handles `num_attention_heads / TP` query heads.
- KV heads are split evenly: each device handles `num_key_value_heads / TP` KV heads.

Both divisions must yield whole numbers. Each device must hold at least one complete KV head because attention computation couples every query head on a device to the KV heads on that same device. A fractional KV head cannot be materialized — the KV head dimension is indivisible within a single attention kernel.

The constraint is therefore:

```
TP must evenly divide num_attention_heads
TP must evenly divide num_key_value_heads
```

The set of valid TP degrees is the set of common divisors of both values, which equals the set of divisors of `gcd(num_attention_heads, num_key_value_heads)`.

### Derivation

For dots.ocr:

- `num_attention_heads = 12`
- `num_key_value_heads = 2`

Step 1: compute the GCD.

```
gcd(12, 2) = 2
```

Step 2: enumerate divisors of 2.

```
divisors(2) = {1, 2}
```

Step 3: verify each candidate divides both head counts.

| TP | 12 / TP | 2 / TP | Valid? |
|----|---------|--------|--------|
| 1  | 12      | 2      | Yes    |
| 2  | 6       | 1      | Yes    |
| 3  | 4       | 0.667  | No     |
| 4  | 3       | 0.5    | No     |
| 8  | 1.5     | 0.25   | No     |

Valid TP degrees: **{1, 2}**.

The TP=4 case illustrates the failure clearly: `2 / 4 = 0.5` KV heads per device. There is no way to represent half a KV head in a TTNN tensor without splitting the head dimension itself, which would break the attention kernel's assumption that each head is a contiguous unit.

### Failure Modes at TP > 2

Attempting to run dots.ocr with `DOTS_T3K_TP` set to any value above 2 will not produce a wrong numerical result — it will fail immediately during model initialization, before any forward pass is attempted.

The failure path is:

1. `DotsModelArgs` computes the per-device KV head count as `num_key_value_heads // TP`.
2. When TP=4, this yields `2 // 4 = 0` (integer division truncates).
3. A shape assertion in the attention layer checks that the KV head count is at least 1 per device.
4. The assertion fires and raises an exception with a message referencing the KV head dimension mismatch.

> **Note:** This is a structural error, not a numerical or memory error. It is caught deterministically at startup on every run and cannot be worked around by adjusting sequence length or batch size.

> **Warning:** Do not confuse this with out-of-memory errors that can occur at large sequence lengths. KV head shape failures appear immediately, before any device memory is allocated for activations.

### Comparison with Qwen 2.5 VL 7B

The TP limit is model-specific and determined entirely by `num_key_value_heads`. For comparison:

| Model | `num_attention_heads` | `num_key_value_heads` | `gcd` | Max TP |
|-------|-----------------------|-----------------------|-------|--------|
| dots.ocr | 12 | 2 | 2 | 2 |
| Qwen2.5-VL-7B | 28 | 4 | 4 | 4 |

Qwen2.5-VL-7B can use TP=4 because it has 4 KV heads; each of 4 devices receives exactly 1 KV head. dots.ocr's choice of 2 KV heads is a parameter efficiency decision: with `num_attention_heads=12` and `num_key_value_heads=2`, the GQA ratio is 6:1. Qwen2.5-VL-7B at 7B parameters runs a 7:1 ratio (28 query heads per 4 KV heads). dots.ocr achieves a comparable compression ratio with far fewer parameters total (~3B), but the lower absolute KV head count is the side effect that constrains TP to 2 on T3K.

This is not a topology decision that can be reversed at inference time. Changing the KV head count would require retraining the model.

**Next:** [T3K Submesh and Env Vars](t3k_submesh_and_env_vars.md)
