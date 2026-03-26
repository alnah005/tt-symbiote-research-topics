# B Review — Chapter 2: Collective Communication Costs and Sharding Strategy — Pass 1

## Item 1 — `sharding_alternatives.md` line 101: Weight shard shape for `IColShardedWRowSharded` is inverted

**File:** `sharding_alternatives.md`, lines 100–102 (Alternative 1, Weight change section)

**Claim:** "Current (`IColShardedWRowSharded`): weight layout is `[H*D, d_model]`, sharded on dim=-2 (rows), so each device holds `[H*D, d_model/N]`."

**What the code shows:** `TTNNLinearIColShardedWRowSharded` sets `weight_dim=-2` (`linear.py` line 130). For a 2D PyTorch weight matrix of shape `[out_features, in_features]` = `[H*D, d_model]`, dim=-2 is the first (row / output-features) dimension. Sharding N=8 ways on that dimension gives each device `[H*D/N, d_model]`, not `[H*D, d_model/N]`. The listed shard shape has the two dimensions swapped.

**Why it matters:** For Bailing specifically (`H*D = d_model = 2048`), the shard size is numerically 256 either way, so the count is right. But the shape `[H*D, d_model/N]` is factually incorrect and would cause a reader implementing weight sharding for any model where `H*D ≠ d_model` to shard the wrong axis. The description of which axis is split is what determines whether the partial matmul sums are correct.

---

## Item 2 — `all_gather_topology.md` line 38: Q weight shard shape per device stated in transposed form

**File:** `all_gather_topology.md`, line 38 (Collective 2 section)

**Claim:** "Each device holds a row-shard of the Q weight: `W_Q_shard_i` has shape `[d_model/N, H*D]`."

**What the code shows:** Same root cause as Item 1. The Q weight is `[H*D, d_model]` (out × in) sharded on `weight_dim=-2` (the H*D rows). Per-device shape is `[H*D/N, d_model]`. The guide writes `[d_model/N, H*D]`, which is the transposed shape. For Bailing `H*D == d_model` so the numbers are equal, but the label `d_model/N` is attached to the wrong axis. A reader cross-referencing this against `ttnn.shard_tensor_to_mesh_mapper(device, dim=-2)` applied to `[H*D, d_model]` will find a contradiction.

---

## Item 3 — `all_gather_topology.md` line 131: GLM4 `_maybe_all_gather` does not synchronize immediately

**File:** `all_gather_topology.md`, line 131

**Claim:** "`TTNNGlm4MoeLiteAttention` also uses `ttnn.experimental.all_gather_async` (line 1604 of `attention.py`) via its own `_maybe_all_gather` implementation (lines 1600–1613), with the same immediate-synchronize pattern."

**What the code shows:** `TTNNGlm4MoeLiteAttention._maybe_all_gather` (lines 1600–1611 of `attention.py`) calls `all_gather_async` and returns the result directly — there is no `ttnn.synchronize_device` call in that function. This is structurally different from `TTNNQwen3FullAttention._maybe_all_gather` (lines 393–408 of `qwen_attention.py`), which does call `ttnn.synchronize_device(self.device)` immediately after. The claim "same immediate-synchronize pattern" is factually wrong for GLM4.

**Why it matters:** A reader comparing GLM4 and Qwen3 collective behavior based on this text would conclude both are effectively synchronous. In fact GLM4's all-gather is truly async (no barrier in `_maybe_all_gather`), which changes the overlap analysis and any latency comparison.

---

## Agent A Change Log — Pass 1
- item 1: In `sharding_alternatives.md` line 101, corrected the per-device weight shard shape for `IColShardedWRowSharded` from `[H*D, d_model/N]` to `[H*D/N, d_model]` (= `[256, 2048]` for Bailing), reflecting that `weight_dim=-2` shards the row/output-features dimension, not the column dimension.
- item 2: In `all_gather_topology.md` line 38, corrected the per-device Q weight shard shape from `[d_model/N, H*D]` to `[H*D/N, d_model]` (= `[256, 2048]` for Bailing), same root cause as item 1 — the shape was written in transposed form with axis labels swapped.
- item 3: In `all_gather_topology.md` line 131, corrected the description of `TTNNGlm4MoeLiteAttention._maybe_all_gather`: replaced the false claim of "same immediate-synchronize pattern" with an accurate description that GLM4 calls `all_gather_async` and returns directly without any `ttnn.synchronize_device`, making it genuinely async — contrasting explicitly with Qwen3 which does synchronize immediately after the async launch.

## Navigation / Link check

- `op_sequence.md`: navigation footer present.
- `all_gather_topology.md`: navigation footer present (line 163–164).
- `sharding_alternatives.md`: navigation footer present (lines 190–194).
- `index.md`: all file references use clickable markdown links (`[all_gather_topology.md](all_gather_topology.md)`, `[sharding_alternatives.md](sharding_alternatives.md)`). No issues.

---

# B Review — Chapter 2 — Pass 2

## Item 1 — `sharding_alternatives.md` line 102: Proposed `IReplicatedWColSharded` weight shard shape is wrong (survived Pass 1)

**File:** `sharding_alternatives.md`, line 102

**Claim:** "Proposed (`IReplicatedWColSharded`): weight layout is `[H*D, d_model]`, sharded on dim=-1 (columns), so each device holds `[H*D/N, d_model]`."

**What the code shows:** `TTNNLinearIReplicatedWColSharded` sets `weight_dim=-1` (`linear.py` line 260). For a weight matrix of shape `[H*D, d_model]` (= `[out_features, in_features]`), dim=-1 is the `in_features` / d_model dimension. Sharding N=8 ways on that dimension gives each device `[H*D, d_model/N]` — not `[H*D/N, d_model]`. The stated shape is identical to the current `IColShardedWRowSharded` shard shape (line 101, now correctly fixed to `[H*D/N, d_model]`), which makes the two entries look the same and obscures the meaningful difference. Pass 1 corrected only line 101; line 102 retains the wrong shape.

**Why it matters:** For Bailing, `H*D = d_model = 2048`, so both shapes produce a 256-element-wide shard numerically. For any model where `H*D ≠ d_model` the shape governs how `ttnn.shard_tensor_to_mesh_mapper(device, dim=-1)` partitions the weight and determines whether the local matmul output is col-sharded on the output dimension (correct) or on the input dimension (incorrect). The error survives because the two values are equal for this specific model.

---

## Item 2 — `index.md` lines 29–38: Data volume calculations omit bfloat16 factor of 2, understating link traffic by 2×

**File:** `index.md`, lines 29–38 (Total link traffic per decode step section)

**Claim:** "Ops 1, 3: each moves B × d_model × (N−1)/N = B × 2048 × 7/8 ≈ B × 1792 bytes per device received … For B=32: Op 1: 32 × 1792 = 57 344 bytes ≈ 56 KB … Total received per device per decode step ≈ 150 KB."

**What the correct calculation shows:** All tensors are bfloat16, which is 2 bytes per element. `all_gather_topology.md` line 24 correctly applies this: "32 × 2048 × 2 bytes (bfloat16) = 131 072 bytes ≈ 128 KB per device." Applying the same factor to `index.md`:
- Op 1: 32 × 1792 × 2 = 114 688 bytes ≈ 112 KB (not 56 KB)
- Op 3: same, 112 KB (not 56 KB)
- Op 2: 32 × 256 × 2 = 16 384 bytes ≈ 16 KB (not 8 KB)
- Op 4: 32 × 448 × 2 = 28 672 bytes ≈ 28 KB (not 14 KB)
- Op 5: same, 28 KB (not 14 KB)
- Correct total: ≈ 296 KB per decode step (not ≈ 150 KB)

The total is understated by approximately 2×. This contradicts the correctly computed 128 KB figure in `all_gather_topology.md` for Collective 1 alone, which is already larger than the 150 KB total reported in `index.md`.

---

## Item 3 — `index.md` table, lines 20–25: Collective execution order in the table is inverted relative to actual code

**File:** `index.md`, lines 20–25 (decode-step collective operations table)

**Claim:** The table lists Step 1 as "All-gather hidden states for K/V input" and Step 2 as "Reduce-scatter inside Q projection."

**What the code shows:** In `_forward_decode_paged` (attention.py lines 2624–2626), `self.q_proj(hidden_states)` — which contains the reduce-scatter — is dispatched at **line 2624**, before `ttnn.all_gather(hidden_states, dim=-1, num_links=1)` at **line 2626**. The reduce-scatter (listed as Step 2) is actually dispatched first; the hidden-states all-gather (listed as Step 1) is dispatched second.

**Why it matters:** The ordering is not merely cosmetic. The Q projection at line 2624 takes the original col-sharded `hidden_states` as input; the all-gather at line 2626 produces `hidden_states_replicated` for use by `k_proj` and `v_proj`. Presenting the all-gather first implies that Q projection has access to replicated hidden states, which is false — the Q proj uses the col-sharded input. Readers using the table to understand data dependencies will derive an incorrect execution model.

## Agent A Change Log — Pass 2
- item 1: In `sharding_alternatives.md` line 102, corrected the per-device weight shard shape for the proposed `IReplicatedWColSharded` from `[H*D/N, d_model]` to `[H*D, d_model/N]` (= `[2048, 256]` for Bailing), reflecting that `weight_dim=-1` shards the column/in_features dimension, not the row dimension. Also added the explicit `[2048, 256]` example to match the style of the corrected line 101.
- item 2: In `index.md` data volume section (lines 29–38), multiplied all byte figures by 2 to account for bfloat16 (2 bytes/element). Updated the bullet formulas to include `× 2 bytes`, recomputed all per-device byte totals (Op 1: 112 KB, Op 2: 16 KB, Op 3: 112 KB, Op 4: 28 KB, Op 5: 28 KB), and updated the total from `≈ 150 KB` to `≈ 296 KB ≈ 300 KB`. Also updated the "Ops 1, 3" / "Op 2" grouping labels to "Op 1 (reduce-scatter)" / "Ops 2, 3" to match the reordered step numbers from item 3.
- item 3: In `index.md` collective operations table, reordered rows to match actual dispatch order in `attention.py`: Step 1 is now the reduce-scatter inside Q projection (dispatched at line 2624), Step 2 is the hidden-states all-gather for K/V (line 2626), Step 3 is the Q all-gather (line 2631), Step 4 is the K all-gather (line 2632), Step 5 is the V all-gather (line 2633). Added line-number references to each table row. Updated the surrounding prose note from "Op 3 (Q all-gather) and Op 1 (hidden-states all-gather)" to "Op 3 (Q all-gather) and Op 2 (hidden-states all-gather)" to reflect the renumbered steps.

---

# B Review — Chapter 2 — Pass 3

## Item 1 — `sharding_alternatives.md` line 178: Reduce-scatter volume missing bfloat16 factor of 2

**File:** `sharding_alternatives.md`, lines 178 and 182 (Expected Latency Impact section)

**Claim:** "The reduce-scatter volume is B × d\_model bytes total / N = B × 256 bytes per device received. For B=32: 8 192 bytes ≈ 8 KB."

**What is correct:** All tensors in this path are bfloat16 (2 bytes per element). The per-device received volume for the reduce-scatter is B × (d\_model / N) × 2 bytes = B × (2048 / 8) × 2 = B × 512 bytes. For B=32: 32 × 512 = 16 384 bytes ≈ 16 KB — double the stated figure. This is directly confirmed by `index.md` Op 1 (the reduce-scatter), which correctly computes "B × 512 bytes … For B=32: 32 × 512 = 16 384 bytes ≈ 16 KB" after Pass 2 corrections.

The downstream estimate at line 182 — "8 KB takes approximately 0.16 µs to transfer" — is also wrong by the same 2× factor; the correct figure is 16 KB ≈ 0.32 µs at 50 GB/s. This is the same class of error that Pass 2 corrected in `index.md` but did not propagate to `sharding_alternatives.md`.

**Why it matters:** The latency estimate is already hedged as a rough bound, but the data-volume figure in line 178 is stated as a precise formula ("B × 256 bytes") without any hedge, and it directly contradicts the corrected figure in `index.md`. A reader using line 178 to size the reduce-scatter saving would underestimate it by 2×.

## Agent A Change Log — Pass 3
- item 1: Corrected reduce-scatter data volume from B×256 bytes to B×512 bytes (added missing bfloat16 ×2 factor); updated B=32 example from 8 KB to 16 KB and adjusted latency estimate accordingly.

---

# B Review — Chapter 2 — Pass 4

## Item 1 — `index.md` line 36: Parenthetical "(same tensor as Op 2)" is factually wrong

**File:** `index.md`, line 36 (total link traffic bullet list)

**Claim:** "Op 3: 32 × 3584 = 114 688 bytes ≈ 112 KB (same tensor as Op 2)"

**What the code shows:** Op 2 is `ttnn.all_gather(hidden_states, dim=-1, num_links=1)` at line 2626 of `attention.py`, which gathers the col-sharded hidden states into replicated `hidden_states_replicated`. Op 3 is `_maybe_all_gather(query_states)` at line 2631, which gathers the col-sharded output of `q_proj` — a completely different tensor (`query_states`, the output of Q projection). The two operations have the same data volume (both are `d_model = 2048` elements per batch item in bfloat16, gathered across N−1 steps), but they operate on different tensors.

**Why it matters:** A reader encountering "(same tensor as Op 2)" would conclude that the Q all-gather and the hidden-states all-gather can be deduplicated — i.e., that `query_states` is the same object as `hidden_states_replicated`. This is incorrect and directly contradicts the analysis in `sharding_alternatives.md` and `all_gather_topology.md`, which correctly explain that the Q all-gather is a *structurally separate* operation on a different tensor that adds communication cost beyond the hidden-states gather. The parenthetical appears to be a stale artefact introduced when the table was reordered in Pass 2 (at which point the note "same tensor as Op 2" was copied from an earlier draft where the numbering was different). It should read "(same *volume* as Op 2)" or be removed.

No feedback beyond Item 1 — all other numerical claims and implementation descriptions verified against `attention.py` lines 2610–2799 and `linear.py`.

## Agent A Change Log — Pass 4
- item 1: Changed "(same tensor as Op 2)" to "(same *volume* as Op 2)" in index.md Op 3 bullet — Op 2 is hidden_states all-gather, Op 3 is query_states all-gather; different tensors, same data volume.

---

# B Review — Chapter 2 — Pass 5

## Item 1 — `all_gather_topology.md` line 45: Collective 2 data-volume formulas missing bfloat16 ×2 factor

**File:** `all_gather_topology.md`, line 45 (Collective 2 data volume section)

**Claim:** "Each device contributes B × 1 × H*D = B × 2048 bytes to the ring; each device receives back B × (H*D/N) = B × 256 bytes."

**What is correct:** H*D = 16 × 128 = 2048 elements. In bfloat16 (2 bytes/element), 2048 elements = 4096 bytes, not 2048 bytes. The sent formula should be "B × H*D × 2 bytes = B × 4096 bytes." Similarly, H*D/N = 256 elements = 512 bytes; the received formula should be "B × (H*D/N) × 2 bytes = B × 512 bytes."

The B=32 concrete examples that follow ("sent 128 KB, received 16 KB") are correct: 32 × 4096 = 131,072 bytes ≈ 128 KB, and 32 × 512 = 16,384 bytes ≈ 16 KB. However, the formulas "B × 2048 bytes" and "B × 256 bytes" are each off by 2× and directly contradict the examples. A reader applying the formula to any batch size other than 32 will compute a value half the correct size. The same class of error was caught and corrected in `index.md` (Pass 2) and `sharding_alternatives.md` (Pass 3) but was not propagated to this section.

---

## Item 2 — `all_gather_topology.md` line 156 (comparison table): GLM4 Sync API entry contradicts corrected prose

**File:** `all_gather_topology.md`, line 156 (All-Gather Count Comparison table, GLM4 row)

**Claim:** GLM4 Sync API column reads "Async `all_gather_async` + synchronize."

**What the code shows (and what the corrected prose at line 131 now correctly states):** `TTNNGlm4MoeLiteAttention._maybe_all_gather` calls `ttnn.experimental.all_gather_async` and returns the result directly — there is no `ttnn.synchronize_device` call. The "+ synchronize" in the table cell is factually wrong. Pass 1 corrected the prose description at line 131 to accurately describe GLM4 as genuinely async (no barrier), but the comparison table on line 156 was not updated and still carries the false "synchronize" label. The table now internally contradicts the prose two paragraphs above it in the same file.

---

## Item 3 — `all_gather_topology.md` collective numbering contradicts `index.md` step numbering for the same two ops

**File:** `all_gather_topology.md` headings vs `index.md` table

**Claim in `all_gather_topology.md`:** "Collective 1 — Hidden States All-Gather (line 2626)" and "Collective 2 — Reduce-Scatter Inside Q Projection (line 2624)."

**Claim in `index.md`:** Step 1 = reduce-scatter (line 2624); Step 2 = hidden-states all-gather (line 2626). (This ordering was established as correct in Pass 2 to match actual dispatch order in `_forward_decode_paged`.)

**Why it matters:** `all_gather_topology.md` opens by stating it walks through operations "in execution order." But it numbers the hidden-states all-gather (line 2626) as Collective 1 and the Q projection reduce-scatter (line 2624) as Collective 2 — the reverse of actual dispatch order, and the reverse of the ordering that `index.md` was corrected to reflect in Pass 2. A reader using both files together will find that "Collective 1" in `all_gather_topology.md` is "Step 2" in `index.md`, and that `sharding_alternatives.md`'s cross-references to "Collective 1" and "Collective 2" also become ambiguous. This is a material misdirection: the hidden-states all-gather at line 2626 depends on the col-sharded `hidden_states` input that is also consumed by `q_proj` at line 2624; knowing which fires first is essential to understanding the data dependency.

## Agent A Change Log — Pass 5
- item 1: Added ×2 bfloat16 factor to Collective 2 data volume formulas.
- item 2: Changed GLM4 table entry from "+ synchronize" to "(no synchronize)".
- item 3: Renumbered collectives to match index.md dispatch order (reduce-scatter=1, hidden all-gather=2, Q=3, K=4, V=5).

---

# B Review — Chapter 2 — Pass 6

## Item 1 — `all_gather_topology.md` Collectives 2 and 3: "128 KB per device" conflates tensor size held with bytes transferred over the link

**File:** `all_gather_topology.md`, lines 45 and 74

**Claim (Collective 2):** "B × d\_model bytes total on each device after the gather. For B=32: 32 × 2048 × 2 bytes (bfloat16) = 131 072 bytes ≈ 128 KB per device."

**Claim (Collective 3):** "Data volume: identical to Collective 2 (same tensor shape). For B=32: ≈128 KB per device received."

**What is correct:** An all-gather of a col-sharded tensor where each device holds `d_model/N` elements does not transfer `d_model` elements per device over the link — it transfers only the `(N−1)` missing shards. Each device receives `(N−1)/N × B × d_model × 2` bytes = `7/8 × 32 × 2048 × 2 = 114 688 bytes ≈ 112 KB`. The figure 128 KB is the total tensor size *held* by each device after the operation completes, not the bytes transferred. The difference is `d_model/N × B × 2 = 32 × 256 × 2 = 16 KB` (the shard that each device already had before the gather).

**Contradiction with `index.md`:** `index.md` correctly states "Ops 2, 3: each moves B × d\_model × (N−1)/N × 2 bytes = B × 2048 × 7/8 × 2 ≈ B × 3584 bytes per device received … For B=32: 32 × 3584 = 114 688 bytes ≈ 112 KB." The same operations are reported as 128 KB in `all_gather_topology.md` and 112 KB in `index.md`. Collective 3 in `all_gather_topology.md` explicitly uses the word "received," making the 128 KB figure an incorrect byte-transfer count, not just a labeling ambiguity.

## Agent A Change Log — Pass 6
- item 1: Corrected bytes-received-over-link for Collectives 2 and 3 from ≈128 KB to ≈112 KB (= 7/8 × B × d_model × 2 bytes; each device already holds its own 1/8 shard).

---

# B Review — Chapter 2 — Pass 7

## Item 1 — `all_gather_topology.md` line 94: Collective 4 (and by symmetry Collective 5) data volume conflates bytes held with bytes received over the link

**File:** `all_gather_topology.md`, line 94 (Collective 4 section)

**Claim:** "For B=32: 32 × 512 × 2 = 32 768 bytes ≈ 32 KB per device received."

**What is correct:** The tensor being gathered has `Hkv × D = 4 × 128 = 512` elements per batch item (bfloat16). The total tensor size held by each device after the all-gather is `B × 512 × 2 = 32 × 512 × 2 = 32 768 bytes ≈ 32 KB`. However, each device already holds its own `1/N = 1/8` shard before the gather, so the bytes actually transferred over the link are only `(N−1)/N × 32 768 = 7/8 × 32 768 = 28 672 bytes ≈ 28 KB`. The text labels 32 KB as "per device received," which is the wrong quantity — it is the post-gather tensor size, not link traffic.

**Contradiction with `index.md`:** `index.md` line 31 correctly computes "B × Hkv×D × (N−1)/N × 2 bytes = B × 512 × 7/8 × 2 = B × 896 bytes per device received … For B=32: 32 × 896 = 28 672 bytes ≈ 28 KB" for Ops 4 and 5. The same operations are reported as 32 KB in `all_gather_topology.md`. This is the same class of error that Pass 6 fixed for Collectives 2 and 3 (hidden-states and Q all-gathers) but was not applied to Collectives 4 and 5.

No feedback beyond Item 1 — all other numerical values, implementation references, and conceptual claims verified against `attention.py` lines 2610–2799 and `linear.py`.

## Agent A Change Log — Pass 7
- item 1: Corrected bytes-received-over-link for Collectives 4 and 5 from ≈32 KB to ≈28 KB (= 7/8 × B × Hkv*D × 2 bytes).

---

# B Review — Chapter 2 — Pass 8

## Item 1 — `all_gather_topology.md` lines 154–160: Qwen3 all-gather count understated; comparison table and concluding prose are wrong

**File:** `all_gather_topology.md`, lines 154–160 (All-Gather Count Comparison table and note below it)

**Claim:** "TTNNQwen3FullAttention: 4 all-gathers per decode step (1 input + 3 post-proj) … Bailing has 5 inter-device synchronization points versus Qwen3's 4."

**What the code shows:** `TTNNQwen3FullAttention._forward_qkv` (qwen_attention.py lines 509–583) calls `self._maybe_all_gather` in distributed mode at the following points every decode step:
1. Line 510: `hidden_states = self._maybe_all_gather(hidden_states)` — input gather (conditional on input being col-sharded, which it is in the MoE decode path).
2. Line 522: `q_proj_output = self._maybe_all_gather(q_proj_output)` — Q gather.
3. Line 523: `key_states = self._maybe_all_gather(key_states)` — K gather.
4. Line 524: `value_states = self._maybe_all_gather(value_states)` — V gather.
5. Line 582: `cos = self._maybe_all_gather(cos)` — cos (RoPE) gather, inside `if self._is_distributed:`.
6. Line 583: `sin = self._maybe_all_gather(sin)` — sin (RoPE) gather.

That is 6 all-gather calls per decode step in distributed mode, not 4. The table column "1 input + 3 post-proj" accounts for only items 1–4 and omits the two positional-embedding gathers (items 5–6). The concluding sentence "Bailing has 5 inter-device synchronization points versus Qwen3's 4" is therefore also wrong; Qwen3 has at least 6.

**Why it matters:** The comparison table is the primary takeaway of this section. If Qwen3 has 6 all-gathers (plus no reduce-scatter) while Bailing has 5 (4 all-gathers plus 1 reduce-scatter), the framing "Bailing pays an additional reduce-scatter that Qwen3 does not" remains directionally correct for the QKV collectives, but the total-collective comparison changes sign — Qwen3 actually has more collectives per decode step than Bailing, not fewer. This is material misdirection for any reader using the table to motivate optimization work on Bailing relative to Qwen3.

## Agent A Change Log — Pass 8
- item 1: Corrected Qwen3 all-gather count from 4 to 6 after verifying qwen_attention.py; updated comparison table (row now reads "6 (1 input + 3 post-proj + 2 cos/sin)" with Total collectives = 6) and closing summary sentence (now reads "Qwen3 has 6 inter-device synchronization points versus Bailing's 5").

---

# B Review — Chapter 2 — Pass 9

## Item 1 — `sharding_alternatives.md`: Collective numbering is systematically inverted relative to dispatch order established in Passes 2 and 5

**File:** `sharding_alternatives.md`, lines 39, 72, 129, 134, 160–161, 165, 178, 180, 184

**Claim throughout `sharding_alternatives.md`:** "Collective 1 (hidden all-gather for K/V, line 2626)" and "Collective 2 (Q reduce-scatter / reduce-scatter inside `q_proj`)."

**What the corrected guide establishes:** Pass 2 (`index.md`) and Pass 5 (`all_gather_topology.md`) both established that the reduce-scatter at line 2624 is **Collective 1** (it fires first) and the hidden-states all-gather at line 2626 is **Collective 2**. This order reflects actual dispatch order in `_forward_decode_paged`: `self.q_proj(hidden_states)` at line 2624 contains the reduce-scatter and executes before `ttnn.all_gather(hidden_states, dim=-1)` at line 2626.

`sharding_alternatives.md` was never updated to match this renumbering. It consistently calls the hidden-states all-gather "Collective 1" (line 39, 160) and the reduce-scatter "Collective 2" (lines 72, 134, 161, 178, 180, 184) — the reverse of every other file in Chapter 2. A reader cross-referencing `sharding_alternatives.md` against `index.md` or `all_gather_topology.md` will find that "Collective 2" in the former refers to a different operation than "Collective 2" in the latter.

**Secondary numerical error on line 165:** "Both `index.md` and the data volumes show that Collectives 1, 3 each move ≈128 KB per device (at B=32)." Two errors here: (a) under the corrected numbering, the two large all-gathers are Collective 2 (hidden-states) and Collective 3 (Q), not Collectives 1 and 3; (b) Pass 6 corrected those volumes from 128 KB to ≈112 KB (bytes *received* over the link = 7/8 × 32 × 2048 × 2 = 114 688 bytes). The figure 128 KB is the post-gather tensor size, not bytes transferred.

**Why it matters:** `sharding_alternatives.md` is the page where the optimization case is made. The wrong collective labels cause every cross-reference from this page into `index.md` or `all_gather_topology.md` to point at the wrong operation, and the stale 128 KB figure overstates the volume of the operations being discussed by roughly 14%.

## Agent A Change Log — Pass 9
- item 1: Renumbered collectives throughout sharding_alternatives.md to match index.md/all_gather_topology.md (reduce-scatter=1, hidden all-gather=2).
- item 2: Corrected line ~165 from "Collectives 1,3 ≈128 KB" to "Collectives 2 and 3 ≈112 KB per device received".

---

# B Review — Chapter 2 — Pass 10

No feedback — chapter approved.

---

# B Review — Chapter 2 — Pass 11

No feedback — chapter approved.
