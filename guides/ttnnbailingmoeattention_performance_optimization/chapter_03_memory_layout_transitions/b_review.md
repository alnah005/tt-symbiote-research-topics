# B Review — Chapter 3: Memory Layout Transitions and L1 Pressure — Pass 1

## Structural checks

- Navigation footers: present on both `transition_analysis.md` (line 156) and `avoidable_transitions.md` (line 189). Pass.
- `index.md` file references: all use clickable markdown links. Pass.

---

## Factual issues

### Issue 1 — `avoidable_transitions.md` line 169: avoidable savings total is wrong (material)

The guide states:

> "Eliminating all avoidable transitions saves approximately **393,216 bytes** (roughly 384 KB) of NoC data movement per decode step per device — roughly 73% of the total 540 KB"

The table immediately above that sentence lists six avoidable transitions with these byte values:

| Step | Bytes |
|---|---|
| 8a (Q DRAM→L1) | 131,072 |
| 12c (Q L1→HEIGHT_SHARDED) | 131,072 |
| 8b (K DRAM→L1) | 32,768 |
| 12d (K L1→HEIGHT_SHARDED) | 32,768 |
| 16a (K HEIGHT_SHARDED→HEIGHT_SHARDED) | 32,768 |
| 20 (attn_output DRAM→HEIGHT_SHARDED) | 131,072 |

Correct sum: 491,520 bytes = **480 KB** (~91% of the 528 KB total, not 73%).

The stated 393,216 bytes equals exactly `3 × 131,072` — it counts only the three large transitions (8a, 12c, 20) and silently omits the three K transitions (8b, 12d, 16a = 98,304 bytes combined). The 73% figure propagates from this undercount.

### Issue 2 — `avoidable_transitions.md` line 185: "459 KB" should be "448 KB" (wrong numerical answer)

The guide states:

> "steps 8a, 8b, 12c, 12d, and 20 together account for 459 KB of the 540 KB total"

Actual sum: 131,072 + 32,768 + 131,072 + 32,768 + 131,072 = 458,752 bytes = **448 KB** (binary). The figure 459 is neither binary KB nor decimal KB for this value; it is simply incorrect.

### Issue 3 — `index.md` and `transition_analysis.md`: "540 KB" should be "528 KB" (wrong numerical answer)

`index.md` (line 29) and `transition_analysis.md` (line 3 preamble context) characterize the total NoC data movement as "roughly 540 KB". The nine-transition table in `index.md` shows byte values that sum to 540,672 bytes. In binary kilobytes (the unit implicitly used throughout — e.g., "384 KB = 393,216 bytes") this is **528 KB**, not 540 KB. 540 KB would equal 552,960 bytes. The "540 KB" figure appears to have been computed as 540,672 / 1000 ≈ 541 decimal KB, inconsistent with the binary-KB convention used for all other size claims in the chapter.

### Issue 4 — `transition_analysis.md` line 152: description of Q copies understates the count

The symmetry summary states:

> "Q is copied twice explicitly (steps 8a, 12c, each 131,072 bytes) plus once at step 20 for the SDPA output (131,072 bytes)"

Step 20 moves `attn_output`, not `query_states`. Calling it a "copy of Q" conflates two different tensors: the input Q (query_states) and the SDPA output (attn_output, shape `[1, B, H, D]`). While the volumes are identical, attn_output is not Q — it is the result of scaled dot-product attention. The framing is conceptually incorrect and inflates the perceived redundancy for Q specifically.

### Issue 5 — `avoidable_transitions.md` line 56: K pre-all-gather shape uses wrong Hkv (minor numerical)

The guide states:

> "Before the all-gather, K from `k_proj` is col-sharded with shape `[B, 1, Hkv * D / N] = [B, 1, 64]` — only 64 elements per token per batch"

With `Hkv=4`, `D=128`, `N=8`: `Hkv * D / N = 4 * 128 / 8 = 64`. This arithmetic is correct. However, the guide derives `Hkv * D / N` assuming K projection is col-sharded across devices. The actual `k_proj` is `TTNNLinearIReplicatedWColSharded` — its weight is col-sharded, producing a col-sharded (output-dimension-sharded) activation. The output per device is `Hkv * D / N` elements only if the full output is evenly partitioned. Since `Hkv=4` and `N=8`, this gives `0.5` heads per device — a sub-integer result. The guide's `64` figure assumes `Hkv * D` divides evenly by `N=8`, which it does in raw element count (`512 / 8 = 64`), but the claim that each device holds a semantically complete slice of K heads is not valid when `N > Hkv`. This is a material conceptual error in the feasibility analysis for the pre-all-gather QK norm proposal: with `Hkv=4` and `N=8`, at most 4 of the 8 devices hold any K data at all under a head-aligned sharding, or all 8 devices hold partial head data under element-aligned sharding — neither case makes the per-head norm "entirely intra-device".

---

# B Review — Chapter 3 — Pass 2

No feedback — chapter approved.

---

# B Review — Chapter 3 — Pass 3

## Factual issue

### Issue 1 — `avoidable_transitions.md` line 172: "262 KB" should be "320 KB" (wrong numerical answer)

The guide states:

> "Saves 262 KB across Q and K (steps 8a + 8b + 12c + 12d combined), reduced to 20 KB if the pre-gather copies substitute for the post-gather ones."

The four steps listed sum as follows (values taken from `index.md` table, which the rest of the chapter treats as ground truth):

| Step | Bytes |
|---|---|
| 8a (Q DRAM → L1) | 131,072 |
| 8b (K DRAM → L1) | 32,768 |
| 12c (Q L1 → HEIGHT_SHARDED) | 131,072 |
| 12d (K L1 → HEIGHT_SHARDED) | 32,768 |
| **Total** | **327,680 bytes = 320 KB** |

The guide states 262 KB (= 268,288 bytes), which does not match any combination of these four transitions. The correct figure is **320 KB**. The "reduced to 20 KB" figure that follows is arithmetically consistent with the pre-all-gather cost calculation (`B * 1 * 256 * 2 + B * 1 * 64 * 2 = 16,384 + 4,096 = 20,480 bytes ≈ 20 KB`) and is correct; only the 262 KB baseline is wrong.

---

## Agent A Change Log — Pass 1

- item 1: In `avoidable_transitions.md` line 169, changed "393,216 bytes" to "491,520 bytes", "384 KB" to "480 KB", "73%" to "91%", and "540 KB" to "528 KB" in the savings claim sentence.
- item 2: In `avoidable_transitions.md` line 185, changed "459 KB" to "448 KB" and "540 KB" to "528 KB" in the combined optimization impact summary.
- item 3: In `index.md` line 29, changed "approximately 540 KB" to "approximately 528 KB" in the quantitative baseline total.
- item 4: In `transition_analysis.md` line 152, changed the description of the step 20 transition from "once at step 20 for the SDPA output (131,072 bytes)" to explicitly identify `attn_output` as a distinct tensor (SDPA output, same shape as Q) rather than a copy of Q.
- item 5: In `avoidable_transitions.md` line 56, replaced the paragraph claiming the K pre-all-gather norm is "even more favorable" with a qualified analysis noting that the proposal is straightforwardly feasible for Q (H=16 > N=8, each device holds 2 complete Q heads) but non-trivial for K (Hkv=4 < N=8 means col-sharding produces only partial head slices per device, which would not preserve per-head normalization semantics without architectural changes).

## Agent A Change Log — Pass 3
- item 1: Corrected "262 KB" to "320 KB" in avoidable_transitions.md (steps 8a+8b+12c+12d = 131,072+32,768+131,072+32,768 = 327,680 bytes = 320 KB).

---

# B Review — Chapter 3 — Pass 4

No feedback — chapter approved.
