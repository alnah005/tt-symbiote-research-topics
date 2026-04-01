# Agent B Review — Chapter 5: Mixture of Experts — Pass 1

## Item 1 — Wrong numerical result: bfp8 expert weight total

**File:** `dram_budget.md`, section "Why bfp4 for Routed Expert Weights"

**Claimed:** "At bfp8, 256 experts × 40 layers at bfp8 would require approximately 256 × 3,145,728 × 1 byte × 40 ≈ 32.2 GB"

**Correct value:** 256 × 3,145,728 × 1 × 40 = 32,212,254,720 bytes. Dividing by 1,073,741,824 (1 GiB) gives **30.0 GB**, not 32.2 GB. The guide overstates this by 2.2 GB. The qualitative conclusion (exceeds 28 GB) remains valid, but the stated figure is wrong.

---

## Item 2 — Inconsistency: shared expert forward uses two separate matmuls, but `architecture_overview.md` fused-weight description could mislead

**File:** `architecture_overview.md`, section "Per-Expert SwiGLU Architecture"

The section says "For the routed experts, the gate and up projections are fused into a single weight matrix." This is accurate as written. However, `qwen35_moe.py` lines 142–143 show the shared expert is computed with **two separate** `ttnn.linear` calls (`shared_w1` and `shared_w3`), not a fused weight. The forward-pass overlap diagram correctly lists "shared_w1 matmul → shared_w3 matmul" as two steps. No explicit error here, but the text does not state that the shared expert retains two separate projection weights while routed experts use a fused one — this asymmetry is never stated and could cause incorrect implementation if a reader assumes the same fusion applies to both. **Flag as an omission that could cause incorrect implementation.**

---

## Item 3 — HF source shape label is ambiguous in `expert_computation.md`

**File:** `expert_computation.md`, section "Expert Weight Layout and bfp4 Storage"

**Claimed comment:** The HuggingFace checkpoint stores gate+up weights as shape `[256, 1024, 2048]` labeled as `[N_exp, 2m, d]`.

**Source:** `qwen35_moe.py` line 74 comment says `[256, 2*intermediate, hidden]` = `[256, 1024, 2048]`. This matches.

**However**, the table directly above states the on-device shape is `[1, 1, 2048, 1024]` (`[1, 1, hidden, 2*intermediate]`). The text immediately following explains `.T` transposes from `[out, in]` to `[in, out]`, correctly accounting for this. No numerical error.

---

## Item 4 — `dram_budget.md` bfp4 "Why bfp4" section repeats a different estimate (16.1 GB) than the derivation section (15.0 GB) without reconciliation

**File:** `dram_budget.md`

The derivation section ("Expert Weight Size Derivation") computes `256 × 3,145,728 × 0.5 × 40 ≈ 15.0 GB` and then says "The reported 12.8 GB accounts for actual tile-aligned bfp4 block format." The "Why bfp4" section later re-states the same formula as `≈ 16.1 GB`. Both calculations use identical inputs: `256 × 3,145,728 × 0.5 × 40 = 16,106,127,360 bytes = 15.003 GB`. The "Why bfp4" section labels this 16.1 GB — that is wrong. The correct result is **15.0 GB** (using GiB) or **16.1 GB** only if dividing by 10^9 (decimal GB). The derivation section uses 15.0 GB (GiB-correct). The "Why bfp4" section uses 16.1 GB (decimal GB). The two sections use different unit conventions for the same arithmetic without disclosure, making one of the stated values wrong depending on convention. Given the DRAM capacity of 28 GB is stated in the guide and matched to the PERF.md which uses the 12.8 GB figure (clearly GiB-based), the consistent convention is GiB, making **15.0 GB** the correct value and **16.1 GB** in the "Why bfp4" section incorrect.

---

## Item 5 — `architecture_overview.md` params-per-expert formula uses ambiguous notation

**File:** `architecture_overview.md`, section "The 256+1 Expert Structure"

**Claimed:**
```
params per expert = d·2m + m·d = 2048·1024 + 512·2048 = 3,145,728
```

The formula `d·2m` is evaluated as `2048·1024`. Since `2m = 2×512 = 1024`, this is arithmetically correct. However the formula also writes `m·d` = `512·2048` for the down projection. The down projection weight shape is `[m, d]` = `[512, 2048]`, so `m × d = 512 × 2048 = 1,048,576`. The sum is `2,097,152 + 1,048,576 = 3,145,728`. The arithmetic is correct. No error.

---

**Summary of actionable errors:**

| # | File | Type | Severity |
|---|------|------|----------|
| 1 | `dram_budget.md` | Wrong number (32.2 GB should be 30.0 GB) | Wrong derivation result |
| 2 | `architecture_overview.md` | Missing statement that shared expert uses two separate weights, not fused | Could cause incorrect implementation |
| 4 | `dram_budget.md` | Inconsistent unit convention: 15.0 GB (GiB) vs 16.1 GB (decimal GB) for same calculation | Wrong number in "Why bfp4" section |

---

# Agent B Review — Chapter 5: Mixture of Experts — Pass 2

## Item 1 — Wrong op count: "3 TTNN ops per expert" undercounts by one

**File:** `expert_computation.md`, section "L1 Accumulation Loop — Full Forward Pass"

**Claimed:** "The loop dispatches 3 TTNN ops per expert (linear, split + fused-mul, linear) plus a scale and an add, for a total of approximately 5 device operations × 8 experts = 40 dispatches"

**Source:** `qwen35_moe.py` lines 177–195 dispatch per expert: (1) `ttnn.linear` gate_up, (2) `ttnn.split`, (3) `ttnn.mul` with fused SILU, (4) `ttnn.linear` down — that is **4** distinct kernel dispatches before the scale and add, not 3. `ttnn.split` and `ttnn.mul` are two separate op dispatches; the bracket groups them as one, which is incorrect. The correct breakdown is "4 TTNN ops per expert (linear, split, fused-mul, linear) plus a scale and an add = 6 ops per expert", or ~48 dispatches for 8 experts in the standard case. The "40 dispatches" total is also therefore wrong (correct upper bound is ~48, lower bound ~40 only if the `w != 1.0` scale multiply is always skipped, which does not happen in the normal 8-expert path where no weight is exactly 1.0).

---

## Item 2 — Wrong formula for "params per expert" counts fused weight, not three separate matrices

**File:** `architecture_overview.md`, section "The 256+1 Expert Structure"

**Claimed:** `params per expert = d·2m + m·d = 2048·1024 + 512·2048 = 3,145,728`

The formula is presented as counting parameters for a SwiGLU expert that has `W_gate`, `W_up`, and `W_down`. The correct three-matrix count is `d×m + d×m + m×d = 3dm = 3 × 2048 × 512 = 3,145,728`, which gives the same number. However the written formula `d·2m + m·d` counts `W_gate+up` (fused, shape `d × 2m`) plus `W_down` (shape `m × d`), implying the expert stores a fused gate+up weight. This is true for the **routed** experts in the TTNN implementation but is **not** true for the shared expert, which stores separate `gate_proj` and `up_proj` weights. The formula is correct numerically but is only valid for the fused (routed expert) layout; presenting it as the generic "per-expert" parameter count immediately after introducing both routed and shared experts is misleading and could cause a reader to assume the shared expert also uses the fused weight layout when implementing the weight-loading logic.

---

## Item 3 — `expert_computation.md` PyTorch reference: `raw_down` HF shape comment conflicts with test file

**File:** `expert_computation.md`, section "Expert Weight Layout and bfp4 Storage"

**Claimed:** "the down projections as `[256, 2048, 512]` (i.e., `[N_exp, d, m]`)"

**Source:** `test_a3b_pcc.py` line 197 comment for the same tensor: `# [256, 2048, 512]`. This matches. However `qwen35_moe.py` line 75 comment says `# [256, hidden, intermediate]` = `[256, 2048, 512]`. No numerical error, but the chapter's label `[N_exp, d, m]` means `[256, 2048, 512]` where `d=hidden=2048` and `m=intermediate=512`. This is consistent and correct.

No error — informational only; this item is not a flag.

---

## Item 4 — `dram_budget.md` shared-expert inline derivation gives 0.12 GB but table says 0.8 GB with no reconciliation

**File:** `dram_budget.md`, section "Why Shared Expert Stays at bfp8"

**Claimed:** "40 layers × 3 × 2048 × 512 × 1 byte (bfp8) = 125,829,120 bytes ≈ 0.12 GB" for the shared expert upgrade cost.

**Source table (same file and PERF.md):** Shared expert weights = **0.8 GB**.

The inline derivation gives ~0.12 GB while the summary table (which matches PERF.md exactly) gives 0.8 GB — a ~7× discrepancy. The derivation uses the raw element count without accounting for tile-alignment padding (tiles are 32×32; both 2048 and 512 are multiples of 32, so no padding) or bfp8 block-header overhead. The discrepancy is too large to be explained by format overhead alone. One of the two numbers is wrong. Given PERF.md is the authoritative source and independently states 0.8 GB, the inline derivation of 0.12 GB is the incorrect value. A correct derivation would need to account for the actual on-disk bfp8 tile format storage size, but the stated 0.12 GB figure as written is wrong.

---

**Summary of new actionable errors (Pass 2):**

| # | File | Type | Impact |
|---|------|------|--------|
| 1 | `expert_computation.md` | Wrong number: "3 TTNN ops per expert" should be 4; "40 dispatches" should be ~48 | Incorrect dispatch count leads to wrong performance model |
| 2 | `architecture_overview.md` | Formula `d·2m + m·d` presented as generic per-expert param count but only valid for routed expert fused-weight layout | Could cause shared expert to be incorrectly implemented with fused weight |
| 4 | `dram_budget.md` | Inline derivation gives 0.12 GB for shared expert weights; table (and PERF.md) gives 0.8 GB — 7× discrepancy, derivation is wrong | Misleading DRAM accounting |

---

# Agent B Review — Chapter 5: Mixture of Experts — Pass 3

## Item 1 — Wrong causal explanation: bfp4 block format increases storage, not decreases it

**File:** `dram_budget.md`, section "Expert Weight Size Derivation"

**Claimed:** "The reported 12.8 GB accounts for the actual tile-aligned bfp4 block format and weight caching overhead."

**Why this is wrong:** The bfp4_b (block float 4) format stores 16 values at 4 bits each (8 bytes) plus a 2-byte bfloat16 shared exponent per block = 10 bytes per 16 elements = 0.625 bytes/element effective. This is *larger* than the naive 0.5 bytes/element used in the derivation. Tile alignment also only adds padding, never reduces storage. The tile-aligned bfp4 block format therefore makes the actual DRAM footprint larger than the naive ~15.0 GB estimate — not smaller. Yet the text uses this explanation to justify why the actual figure (12.8 GB from PERF.md) is *below* the naive estimate. The causal direction is inverted: the explanation as written is factually incorrect and would cause an implementer verifying the DRAM budget to draw the wrong conclusion about bfp4 storage costs.

**Correct statement:** The naive formula `256 × 3,145,728 × 0.5 bytes × 40 ≈ 15.0 GB` is a lower bound. The actual measured 12.8 GB from PERF.md is lower, which cannot be explained by block-format overhead. The most likely explanation is that 12.8 GB is an empirically measured resident footprint (e.g., after weight caching deduplication or with a different byte-counting basis), not a derived figure. The chapter should not attribute the gap to block-format overhead.

---

## Item 2 — Source line number reference is off by two

**File:** `architecture_overview.md`, section "Why Host Top-k"

**Claimed:** "The code comment in `qwen35_moe.py` line 163 notes the tradeoff explicitly" — then shows the `logits_cpu = ttnn.to_torch(...)` line as part of the attributed snippet.

**Source:** In `qwen35_moe.py`, line 163 is the comment `# For batched decode with same prompt, all rows route identically.` The `logits_cpu =` assignment that the chapter quotes is on **line 165**. The chapter attributes the shown code to line 163, but the actual assignment line is 165. A reader trying to navigate to the cited code will land two lines above the expression shown.

---

**Summary of new actionable errors (Pass 3):**

| # | File | Type | Impact |
|---|------|------|--------|
| 1 | `dram_budget.md` | Wrong causal explanation: block-format overhead increases bfp4 storage, not decreases it; the stated rationale for 12.8 GB < 15.0 GB is causally inverted | Incorrect DRAM accounting model; implementer would misunderstand bfp4 storage costs |
| 2 | `architecture_overview.md` | Source line number cited as 163 but the `logits_cpu` assignment is on line 165 | Incorrect cross-reference; reader navigates to wrong line |

---

# Agent B Review — Chapter 5: Mixture of Experts — Pass 4

## Item 1 — Wrong quantity labeled as "sync volume": 1024 bytes is the CPU float32 size, not the DMA transfer size

**Files:** `architecture_overview.md` (section "Why Host Top-k") and `router_and_routing.md` (section "The One Mandatory Sync")

**Claimed:** Both files state that the sync volume / transfer volume is `256 × 4 bytes = 1024 bytes`.

**Why this is wrong:** The device tensor `router_logits` is stored as `bfloat16` (confirmed by `qwen35_moe.py` line 64: `dtype=ttnn.bfloat16` for the router weight, and the matmul output inherits bf16). The DMA transfer from device to host at the `ttnn.to_torch()` call copies the bf16 tensor: `256 × 2 bytes = 512 bytes` of logically relevant data. The `.float()` conversion in `logits_cpu = ttnn.to_torch(router_logits).float()[...]` happens on the CPU after the transfer — it does not affect the transfer size. The correct "sync volume" (DMA transfer size for the 256 logit values) is **512 bytes**, not 1024 bytes. `router_and_routing.md` parenthesizes "(float32 after conversion)" which clarifies the unit but still labels the post-conversion size as the "transfer volume" — a wrong label for the actual transfer quantity.

**Correct statement:** The DMA transfer volume for the 256 bf16 logit values is `256 × 2 bytes = 512 bytes`. After CPU-side `.float()` conversion the values occupy 1024 bytes in CPU memory, but that is not the transfer cost.

---

**Summary of new actionable errors (Pass 4):**

| # | File | Type | Impact |
|---|------|------|--------|
| 1 | `architecture_overview.md`, `router_and_routing.md` | Wrong number: "sync volume = 256 × 4 bytes = 1024 bytes" should be 256 × 2 bytes = 512 bytes (bf16 DMA transfer); 1024 bytes is the post-conversion CPU float32 size | Incorrect transfer-cost figure; stated quantity does not match labeled concept |

---

# Agent B Review — Chapter 5: Mixture of Experts — Pass 5

No feedback — chapter approved.

# Agent B Review — Chapter 5: Mixture of Experts — Pass 6

No feedback — chapter approved.
