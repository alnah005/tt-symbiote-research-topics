# Compression Analysis: Chapter 7 — Profiling and Roadmap — Pass 1

## Summary
- Total files analyzed: 4
- Estimated current line count: ~353 lines
- Estimated post-compression line count: ~295 lines
- Estimated reduction: ~16%

---

## CRUCIAL Suggestions

### CRUCIAL 1 — Bottleneck Rank Descriptions Restate the Index Table (~12 lines of overlap)

`index.md` lines 23–31 contain a full "Relationship to Prior Chapters" table that lists every bottleneck with its chapter source and magnitude. `bottleneck_ranking.md` then re-describes each of these same bottlenecks in full prose (Ranks 1–6). The index table entries are not short pointers — they include specific figures ("9 `to_memory_config` transitions; 528 KB/step; 91% avoidable", "163,840 bytes/step") that are then restated verbatim inside `bottleneck_ranking.md`.

**Specific overlap:**
- Index table row for Chapter 3 ("9 `to_memory_config` transitions; 528 KB/step; 91% avoidable") restates `bottleneck_ranking.md` Rank 2 root-cause sentence almost word-for-word.
- Index table row for Chapter 4 (`cur_pos_tt from_torch each decode step`) maps directly to Rank 3 root-cause description.
- Index table row for Chapter 5 ("QK norm DRAM→L1 transition (163,840 bytes/step)") repeats Rank 2 proposed-fix figures.

**Recommendation:** Strip the "Relationship to Prior Chapters" table in `index.md` down to chapter number, a one-clause summary, and a file pointer only — removing the embedded magnitude figures that are load-bearing in `bottleneck_ranking.md`. Estimated savings: 8–10 lines of duplicated data.

---

### CRUCIAL 2 — Q Projection Strategy Described in Both `bottleneck_ranking.md` and `comparison_to_other_implementations.md` (~10 lines of overlap)

`bottleneck_ranking.md` Rank 1 (lines 7–18) explains the reduce-scatter + all-gather problem for Q, the proposed fix (`TTNNLinearIReplicatedWColSharded`), the Qwen3 precedent, and the head-count compatibility argument. `comparison_to_other_implementations.md` "Key Technique: Q Projection Strategy" section (lines 43–48) then re-explains the same mechanism, the same fix, and the same Qwen3 comparison.

**Specific overlap:**
- Both sections identify `TTNNLinearIReplicatedWColSharded` as the fix.
- Both state that the replicated hidden state from the K/V all-gather eliminates the need for a separate Q gather.
- Both reference the Qwen3 implementation as the validated precedent.
- `bottleneck_ranking.md` Rank 1 already says "No other implementation uses this pattern for Q" — the comparison file's section on this technique adds no new fact.

**Recommendation:** In `comparison_to_other_implementations.md`, collapse "Key Technique: Q Projection Strategy" to 2–3 sentences that point to Rank 1 in `bottleneck_ranking.md` for the full analysis. The side-by-side table row already captures the structural difference; the prose section is largely redundant. Estimated savings: 6–8 lines.

---

### CRUCIAL 3 — SDPA Compute Config Repeated Across Three Locations (~8 lines of overlap)

The `fp32_dest_acc_en` / `packer_l1_acc` configuration difference between Bailing and Qwen3 is stated in:
1. `bottleneck_ranking.md` Rank 4 (lines 51–63) — full treatment with Config B/D labels and speedup estimates.
2. `comparison_to_other_implementations.md` side-by-side table rows for `fp32_dest_acc_en` and `packer_l1_acc`.
3. `comparison_to_other_implementations.md` "Key Technique: SDPA Compute Config" section (lines 61–65) — repeats the Qwen3 attribution and dst-tile mechanism.

The side-by-side table is load-bearing and should stay. The "Key Technique: SDPA Compute Config" prose section restates both the table and Rank 4 without adding new information (the "DeepSeek V3 settings" source comment is the only unique detail).

**Recommendation:** In `comparison_to_other_implementations.md`, condense "Key Technique: SDPA Compute Config" to a single sentence naming the DeepSeek V3 source comment and pointing to `bottleneck_ranking.md` Rank 4. Estimated savings: 6–8 lines.

---

## MINOR Suggestions

### MINOR 1 — `_to_replicated` Elimination Discussion Duplicates Itself

`comparison_to_other_implementations.md` mentions `_to_replicated` in three separate places: the side-by-side table row (line 18), the "Key Technique: `_to_replicated` Elimination" section (lines 28–39), and the "Recommended Adoption Path" list (lines 88–92). The recommendation at line 88 ("Propagate Bailing MoE's `_to_replicated` elimination to Qwen3 and GLM4") repeats the closing sentence of the Key Technique section (lines 38–39). The final two sentences of the Key Technique section can absorb the adoption-path bullet, eliminating the separate repetition.

### MINOR 2 — Bottleneck Ranking Summary Table Restates Risk Column from Prose

`bottleneck_ranking.md` "Optimization Priority Summary" table (lines 97–107) contains a Risk column whose values ("Low", "Moderate", "Very low") are already stated in the **Risk:** paragraph of each individual rank section immediately above. The table is useful for the at-a-glance view, but the Risk column in particular adds no new information. Removing the Risk column from the summary table (keeping Rank, Optimization, and Estimated speedup) would reduce visual noise without data loss.

### MINOR 3 — Async All-Gather Section States GLM4 Fact Already in Table

`comparison_to_other_implementations.md` "Key Technique: Async All-Gather" section (lines 51–57) ends with a sentence noting that GLM4 uses `all_gather_async` for `TTNNDistributedRMSNorm` but retains synchronous all-gather for projection outputs. This specific detail is already captured in the side-by-side table row "Async collective for distributed norm" (line 23). The final sentence of that section (lines 55–56) can be deleted without loss.

### MINOR 4 — Profiling Methodology Repeats Five-Collective Count

`profiling_methodology.md` "Interpreting Results" item 1 (lines 102–103) restates that there are five collective ops in the decode path as the "primary source" of stalls. This count has already been introduced at lines 46–52 in the same file (the enumerated list of five collective positions). A one-line forward reference ("the five collectives enumerated above") would replace the restatement.

---

## Load-Bearing Evidence

### LBE 1 — Five Collective Enumeration with Line Numbers (`profiling_methodology.md` lines 46–52)

```
Each synchronous `ttnn.all_gather` call in the decode path (lines 2626, 2631, 2632, 2633 of
`attention.py`) can be timed with the same bracket pattern. The five collective positions in
the decode path are:

1. Line 2624: `q_proj(hidden_states)` — contains the reduce-scatter inside `TTNNLinearIColShardedWRowSharded`
2. Line 2626: `ttnn.all_gather(hidden_states, dim=-1)` — explicit all_gather for K/V input
3. Line 2631: `_maybe_all_gather(query_states)`
4. Line 2632: `_maybe_all_gather(key_states)`
5. Line 2633: `_maybe_all_gather(value_states)`
```

This is the only place in Chapter 7 that maps each collective to its exact line number and position in the call sequence. It is the primary instrumentation target list and must not be cut or merged.

---

### LBE 2 — Optimization Priority Summary Table (`bottleneck_ranking.md` lines 97–107)

```
| Rank | Optimization | Risk | Estimated full-step speedup |
|---|---|---|---|
| 1 | Switch Q projection to `TTNNLinearIReplicatedWColSharded` | Low | 5–15% (1 collective eliminated) |
| 2 | Pre-all-gather Q norm + step 20 transition elimination | Moderate | 3–8% (NoC bandwidth) |
| 3 | Persistent `cur_pos_tt` device tensor | Low | 1–3% (PCIe stall) |
| 4a | `fp32_dest_acc_en=False`, `packer_l1_acc=False` (Config B) | Low | 3–7% (SDPA throughput) |
| 4b | HiFi2 fidelity (Config D, accuracy required) | Medium | 15–25% (SDPA throughput) |
| 5 | Inline K norm dispatch (direct `ttnn.rms_norm`) | Very low | <1% (Python overhead) |
| 6 | `get_cos_sin_for_decode` refactor | Very low | <1% (conditional path) |
```

This table is the canonical prioritized roadmap and the only place that names Config B vs Config D, gives the 4b dependency on accuracy measurement, and states the dependency of Optimization 2 on Optimization 1. It must be preserved intact.

---

### LBE 3 — Side-by-Side Comparison Table (`comparison_to_other_implementations.md` lines 11–24)

The 13-row side-by-side table is the only location in Chapter 7 that consolidates all three implementations across every performance dimension simultaneously — including line-number citations for each implementation's config choices. Several unique facts appear only here: Qwen3's use of `ttnn.experimental.all_gather_async`, GLM4's configurable `distributed` flag for RMSNorm, and Bailing MoE's separate `decode_program_config`. This table must not be trimmed.

---

## VERDICT
- Crucial updates: yes

---

# Compression Analysis: Chapter 7 — Pass 2

## Summary
- Total files analyzed: 4
- Estimated current line count: ~353 lines
- Estimated post-compression line count: ~280 lines
- Estimated reduction: ~21%

---

## CRUCIAL Suggestions

### CRUCIAL 1 (Pass 1 re-check) — Index Table vs. Bottleneck Ranking Magnitude Figures — RESOLVED

**Status: RESOLVED (Pass 1 overstated the overlap).**

Pass 1 claimed `index.md` lines 23–31 embed magnitude figures ("528 KB/step", "163,840 bytes/step") that duplicate `bottleneck_ranking.md`. On re-examination, the `index.md` table contains only short labels — "9 `to_memory_config` transitions per step", "`cur_pos_tt` `from_torch` each decode step", "QK norm DRAM→L1 transition before `_apply_qk_norm`" — with no embedded numeric magnitudes. The figures cited in Pass 1 do not appear in `index.md` at all; they are exclusive to `bottleneck_ranking.md`. The index table rows are brief pointers, not restatements of the data.

**Evidence:** `index.md` lines 25–30 — each row contains a chapter number, a one-clause label, and a file path. No KB/step or percentage figures present. No action required.

---

### CRUCIAL 2 (Pass 1 re-check) — Q Projection Strategy Prose in `comparison_to_other_implementations.md` — STILL OPEN

**Status: STILL OPEN.**

`bottleneck_ranking.md` Rank 1 (lines 9–17) gives the full analysis: reduce-scatter mechanism, proposed fix (`TTNNLinearIReplicatedWColSharded`), Qwen3 precedent (line 255 of `qwen_attention.py`), head-count compatibility argument (H=16, N=8 → 2 heads/device). `comparison_to_other_implementations.md` "Key Technique: Q Projection Strategy" (lines 43–48) opens with "See `bottleneck_ranking.md` Rank 1 for the full analysis. In brief:" and then restates the same fix and the same Qwen3 comparison across 6 lines.

**Evidence:**
- `bottleneck_ranking.md` line 11: "Switch Q projection to `TTNNLinearIReplicatedWColSharded` (same as K/V at line 2375 and Qwen3 Q at line 255 of `qwen_attention.py`)."
- `comparison_to_other_implementations.md` line 45: "`TTNNQwen3FullAttention` uses `TTNNLinearIReplicatedWColSharded` for Q (line 255 of `qwen_attention.py`), eliminating the reduce-scatter present in Bailing's `TTNNLinearIColShardedWRowSharded` (line 2374 of `attention.py`)."

These two sentences convey the same fact with the same line-number citations. The comparison section adds no unique information beyond the cross-reference opener already on line 45. The side-by-side table row (line 13) is the load-bearing structural statement; the prose section can be collapsed to the opener sentence alone.

**Recommendation:** Reduce "Key Technique: Q Projection Strategy" in `comparison_to_other_implementations.md` to a single sentence: "See `bottleneck_ranking.md` Rank 1 for the full analysis; the table row above captures the structural difference." Estimated savings: 4–5 lines.

---

### CRUCIAL 3 (Pass 1 re-check) — SDPA Compute Config Repeated Across Three Locations — STILL OPEN

**Status: STILL OPEN.**

`bottleneck_ranking.md` Rank 4 (lines 51–64, ~13 lines) is the canonical treatment: root cause (`fp32_dest_acc_en=True` limits to 4 dst tiles), two proposed fixes (Config B and Config D), impact estimates (10–20% and 50–80% SDPA speedup), and risk assessment. The side-by-side table rows in `comparison_to_other_implementations.md` (lines 20–21) record the structural facts. The "Key Technique: SDPA Compute Config" prose section (lines 59–65, ~7 lines) then re-explains the dst-tile mechanism and the Qwen3 attribution — information already covered by Rank 4.

**Evidence:**
- `bottleneck_ranking.md` line 54: "`TTNNQwen3FullAttention` uses `fp32_dest_acc_en=False` and `packer_l1_acc=False` (lines 341–346 of `qwen_attention.py`), enabling 8 dst tiles vs the 4 available under `fp32_dest_acc_en=True`."
- `comparison_to_other_implementations.md` line 61: "`TTNNQwen3FullAttention` uses `fp32_dest_acc_en=False` and `packer_l1_acc=False` (lines 341–346 of `qwen_attention.py`), enabling 8 dst tiles vs the 4 available under `fp32_dest_acc_en=True`."

These two sentences are nearly verbatim identical. The only unique fact in the comparison section is the "DeepSeek V3 settings" comment attribution on line 61. That one clause can be folded into a forward reference.

**Recommendation:** Replace the full "Key Technique: SDPA Compute Config" section in `comparison_to_other_implementations.md` with a single sentence: "The `fp32_dest_acc_en` configuration difference (originally attributed to 'DeepSeek V3 settings for head_dim=256 compatibility' in the Qwen3 source) is analyzed at full depth in `bottleneck_ranking.md` Rank 4." Estimated savings: 5–6 lines.

---

### CRUCIAL 4 (new) — `_to_replicated` Elimination Stated Three Times Within the Same File (~12 lines of internal overlap)

`comparison_to_other_implementations.md` states the `_to_replicated` elimination finding in three separate locations within the file:

1. **Side-by-side table row** (line 18): "`_to_replicated` host round-trip — Not called in decode path (bypassed; lines 2642–2646 reshape approach)" — the structural fact.
2. **"Key Technique: `_to_replicated` Elimination" section** (lines 28–39, ~12 lines): explains the mechanism in full prose with a code block showing the reshape comment, and closes with "These implementations should adopt the reshape-based approach."
3. **"Recommended Adoption Path" section** (lines 84–91): bullet point 3 for Bailing ("Propagate Bailing MoE's `_to_replicated` elimination to Qwen3 and GLM4") and the Qwen3/GLM4 adoption-path bullet ("Adopt the reshape-based Q/K/V construction from `TTNNBailingMoEAttention` to eliminate `_to_replicated`") — both re-state the closing sentence of the Key Technique section.

**Evidence:**
- `comparison_to_other_implementations.md` line 39: "These implementations should adopt the reshape-based approach."
- `comparison_to_other_implementations.md` line 84: "Propagate Bailing MoE's `_to_replicated` elimination to Qwen3 and GLM4."
- `comparison_to_other_implementations.md` line 88: "Adopt the reshape-based Q/K/V construction from `TTNNBailingMoEAttention` to eliminate `_to_replicated`."

The recommendation is stated three times across ~12 combined lines. The code block in the Key Technique section (lines 32–35) reproduces a Python comment, not executable logic, and carries no additional information beyond what the prose already states.

**Recommendation:** Remove the embedded code block (lines 32–35) from the Key Technique section. Collapse the two recommendation restatements in the Adoption Path into the closing sentence of the Key Technique section, then remove the redundant Adoption Path bullets. Estimated savings: 7–9 lines.

---

## MINOR Suggestions

### MINOR 1 — Profiling Methodology "Interpreting Results" Re-States Five-Collective Count

`profiling_methodology.md` lines 102–103 (Interpreting Results item 1) reads: "indicates a synchronization barrier (likely an `all_gather` completing before the next op can dispatch). The five collective ops in the decode path are the primary source." The count "five collective ops" was already introduced with its full enumerated list at lines 46–52 of the same file. Replacing "The five collective ops in the decode path are the primary source" with "the five collectives enumerated above are the primary source" is a one-word change that eliminates the restatement without changing meaning.

---

## Load-Bearing Evidence

### LBE 1 — `_to_replicated` Side-by-Side Table Row and Its Uniqueness (`comparison_to_other_implementations.md` line 18)

```
| **`_to_replicated` host round-trip** | **Not called in decode path** (bypassed; lines 2642–2646 reshape approach) | Called for Q/K/V (lines 766–768 of `qwen_attention.py`) | Called for Q/K/V (lines 1895–1897 of `attention.py`) |
```

This table row is the only location across all four files that captures the three-way structural contrast — Bailing (bypassed), Qwen3 (lines 766–768), GLM4 (lines 1895–1897) — in a single scannable entry with line-number evidence for each. It must not be removed. The prose sections in the Key Technique and Adoption Path sections are redundant relative to this row; the row itself is not.

---

### LBE 2 — Rank 4 Config B vs Config D Speedup Estimates (`bottleneck_ranking.md` lines 59–62)

```
**Impact:** Config B alone: estimated 10–20% SDPA speedup, 3–7% full-step speedup. Config D: estimated 50–80% SDPA speedup if accuracy permits.

**Risk:** Config B is low risk (same fidelity, Qwen3-validated). Config D requires accuracy measurement.
```

These four lines contain the only numeric speedup estimates for the SDPA compute config change in the entire chapter. The comparison file's prose section does not carry these figures. Removing the comparison file's prose section (CRUCIAL 3 recommendation) does not affect these numbers — they stay in `bottleneck_ranking.md` Rank 4 where they belong.

---

## VERDICT
- Crucial updates: yes
