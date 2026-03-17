# B Review — Chapter 2: TTNN Quantization API — Pass 1

## Verdict

5 factual errors found. Corrections required before approval.

---

## Error 1 — Incorrect mantissa bit count for bfloat8_b

**File:** `compute_kernel_config.md`
**Approximate line:** 89
**Wrong claim:** "bfloat8_b weights have 8-bit mantissa resolution"
**Correct value:** bfloat8_b encodes 1 sign bit + 7 mantissa bits per element. The mantissa resolution is 7 bits, not 8. The "8" in the name refers to the total bits per element (including the sign bit), not the mantissa width. Stating "8-bit mantissa resolution" overstates the precision by one bit and is inconsistent with the block floating-point encoding defined in Chapter 1.

---

## Error 2 — Incorrect mantissa bit count for bfloat4_b

**File:** `compute_kernel_config.md`
**Approximate line:** 72
**Wrong claim:** "bfloat4_b weights have 4-bit mantissa resolution"
**Correct value:** bfloat4_b encodes 1 sign bit + 3 mantissa bits per element. The mantissa resolution is 3 bits, not 4. The same naming-convention confusion as Error 1: "4" is the total bits per element, not the mantissa width.

---

## Error 3 — MathFidelity levels described by mantissa bits instead of accumulation passes

**File:** `compute_kernel_config.md`
**Approximate lines:** 26–29 (MathFidelity table)
**Wrong claim:** The table describes LoFi as "Uses fewer mantissa bits in the multiply" and omits any characterization of accumulation passes for all four levels.
**Correct value:** The MathFidelity levels are defined by accumulation pass count, not by mantissa-bit truncation in the multiply. LoFi = single-pass accumulation (fastest). HiFi2 = two-pass accumulation. HiFi4 = four-pass accumulation (slowest, full precision). Describing the distinction as a "mantissa bits" difference is factually wrong and will cause readers to misunderstand what LoFi actually trades away.

---

## Error 4 — fp32_dest_acc_en conflated with dequantization target dtype

**File:** `dtype_in_linear_and_matmul.md`
**Approximate line:** 33
**Wrong claim:** "dequantizes each 32x32 tile to bfloat16 (or fp32 if `fp32_dest_acc_en=True`)"
**Correct value:** Dequantization always produces bfloat16 values that enter the FPU. `fp32_dest_acc_en=True` controls the precision of the *accumulation register* (destination register file) — it accumulates partial sums in fp32 before the packer writes the output back to bfloat16. The dequantized tile format is not fp32 under any setting. The parenthetical "(or fp32 if `fp32_dest_acc_en=True`)" incorrectly implies fp32 is the dequantization output dtype rather than the accumulation register dtype.

---

## Error 5 — shard_dims text contradicts the code example in T3K section

**File:** `weight_conversion.md`
**Approximate lines:** 118–136
**Wrong claim:** The prose states "a typical sharding strategy is `shard_dims=(1, 1)`, which shards across both the expert dimension and the inner weight dimension."
**Correct value:** The code example immediately below does not use `shard_dims` at all; it uses `mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0)`, which shards only along dim 0 (the expert dimension). The `shard_dims=(1, 1)` description appears to describe a different API or a different sharding intent (two-dimensional mesh sharding) that is not represented by the code shown. The prose and the code example are inconsistent and the `shard_dims` parameter name does not correspond to any argument in the `ttnn.as_tensor` call as written. This will confuse readers attempting to implement T3K sharding.

---

## Agent A Change Log — B Feedback Pass 1
- compute_kernel_config.md: Fixed bfloat8_b from "8-bit mantissa" to "7 mantissa bits (1 sign + 7 mantissa = 8 total bits)"
- compute_kernel_config.md: Fixed bfloat4_b from "4-bit mantissa" to "3 mantissa bits (1 sign + 3 mantissa = 4 total bits)"
- compute_kernel_config.md: Updated MathFidelity table — LoFi/HiFi2/HiFi4 defined by accumulation passes (single/two/four), not mantissa-bit truncation
- dtype_in_linear_and_matmul.md: Corrected dequantization/accumulation dtype description — dequant always yields bfloat16; fp32_dest_acc_en controls accumulation register dtype
- weight_conversion.md: Aligned prose and code — replaced shard_dims reference with mesh_mapper=ttnn.ShardTensorToMesh API; updated prose to describe sharding dimension

---

# B Review — Chapter 2: TTNN Quantization API — Pass 2

All Pass 1 fixes verified. No feedback — chapter approved.
