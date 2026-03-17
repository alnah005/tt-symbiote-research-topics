# Compression Analysis — Chapter 7: Implementation and Validation

## Crucial updates: yes

---

### Duplication 1: ShardSpec / MemoryConfig construction block (gate/up WIDTH_SHARDED)

**Source (original):**
- `ch02_dram_sharded_memory_layout/constructing_dram_sharded_config.md` lines 13–35 (Step 1 + Step 2 code block)
- `ch02_dram_sharded_memory_layout/shard_spec_deep_dive.md` lines 92–101 (Worked Example WIDTH_SHARDED)
- `ch06_performance_analysis_and_tradeoffs/shard_setup_overhead.md` lines 77–98 (gate_up_shard_config build)
- `ch03_expert_weight_tensor_structure/tensor_to_shard_grid_mapping.md` lines 160–181 (Worked Example Steps 3–4)

**Duplicate (ch07):**
- `code_patterns.md` lines 95–136 (`make_gate_up_shard_config` function body)
- `code_patterns.md` lines 139–175 (`make_down_shard_config` function body)

**What is duplicated:**
The three-field `ttnn.ShardSpec(grid, shape, orientation)` → `ttnn.MemoryConfig(memory_layout, buffer_type, shard_spec)` construction idiom for Mixtral gate/up `[4096, 1792]` shards is shown verbatim (or near-verbatim with minor variable renaming) in at least four prior files. The assertion checks `d_ff % num_banks == 0`, `shard_h % 32 == 0`, `shard_w % 32 == 0`, and `shard_bytes % 32 == 0` replicate what Ch05 `shard_shape_alignment_rules.md` defines as Rules 1, 2, 3, and 5 and Ch03 `tensor_to_shard_grid_mapping.md` applies in its worked example. The Ch06 `shard_setup_overhead.md` file contains a full `gate_up_shard_config` / `down_shard_config` construction block (lines 77–98) that is structurally identical to Ch07 `code_patterns.md` lines 95–175.

**Recommended action:**
Replace the bodies of `make_gate_up_shard_config` and `make_down_shard_config` in `code_patterns.md` with a forward-reference note:
> "The ShardSpec construction pattern and all four alignment assertions are defined in Ch05 `shard_shape_alignment_rules.md` (Rules 1–5) and demonstrated in Ch03 `tensor_to_shard_grid_mapping.md`. The helper functions here call those rules; do not duplicate the derivation."

Retain the function signatures and the call sites (lines 179–195) because those are integration glue that is unique to Ch07. Remove or dramatically shorten the inline assertion explanations (lines 108–114, 153–156) to cross-references.

---

### Duplication 2: Load-time resharding pattern (`ttnn.from_torch` + `ttnn.to_memory_config` loop)

**Source (original):**
- `ch06_performance_analysis_and_tradeoffs/shard_setup_overhead.md` lines 22–137 (the full `load_expert_weights` function, Option A vs Option B discussion, and the warning against inference-time resharding)
- `ch02_dram_sharded_memory_layout/constructing_dram_sharded_config.md` lines 121–167 (End-to-End Example)

**Duplicate (ch07):**
- `code_patterns.md` lines 33–79 (Step 1: CPU loading + `ttnn.from_torch` loop for gate/up/down)
- `code_patterns.md` lines 207–231 (`reshard_expert_weights` function + call site)

**What is duplicated:**
The loop pattern `ttnn.from_torch(cpu_weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)` and the subsequent `ttnn.to_memory_config(tensor, sharded_config)` with `ttnn.deallocate` of the interleaved copy appear identically in Ch06 `shard_setup_overhead.md` (the `load_expert_weights` production function) and in Ch02 `constructing_dram_sharded_config.md` (the End-to-End Example). The Ch07 `reshard_expert_weights` function (lines 207–224) is a reduced version of Ch06's `load_expert_weights` with the same logic.

**Recommended action:**
In `code_patterns.md` Step 1 (lines 33–79), replace the three separate `ttnn.from_torch` list comprehensions with a single paragraph and a pointer:
> "See Ch06 `shard_setup_overhead.md` — Option A for the canonical load-time resharding pattern, including the warning against inference-time resharding."

In `reshard_expert_weights` (lines 207–231), note that this is an abbreviated variant of Ch06's `load_expert_weights` and cross-reference that file. The unique contribution of Ch07's Step 3 is making the `memory_config()` / `shard_spec()` assertion explicit (lines 234–237) — retain those lines.

---

### Duplication 3: Decision table — regime vs recommended layout

**Source (original):**
- `ch06_performance_analysis_and_tradeoffs/index.md` lines 29–38 (Decision Table: DRAM-Sharded vs Interleaved)
- `ch06_performance_analysis_and_tradeoffs/tradeoff_matrix.md` lines 9–18 (Four-Regime Comparison table)

**Duplicate (ch07):**
- `benchmark_methodology.md` lines 265–269 (Reporting Results — "Expected outcomes" bullet list)
- `index.md` lines 111–112 (Key Constants table — `Decode regime boundary` row)

**What is duplicated:**
The three-bucket expected outcome ranges (`effective_M ≤ 16`: −30 to −50%; `16 < effective_M ≤ 64`: −10 to −25%; `effective_M > 256`: 0 to +5%) are stated verbatim in Ch07 `benchmark_methodology.md` lines 265–269 and are already the defining content of Ch06's trade-off matrix and index decision table.

**Recommended action:**
In `benchmark_methodology.md` lines 265–269, replace the three bullet points with a single sentence:
> "For expected outcome ranges by regime, see Ch06 `tradeoff_matrix.md`. Divergence from those ranges should be investigated via Tracy profiling (see Tracy section above)."

The `index.md` Key Constants table row for `Decode regime boundary` (line 111) is a summary reference, not a duplication of the full table — it can stay.

---

## Load-Bearing Evidence

The following content in Ch07 is original and should not be removed:

- **`code_patterns.md` lines 248–293** (`expert_ffn_forward` function): The first complete, runnable SwiGLU FFN forward pass integrating DRAM-sharded weights with `ttnn.matmul`, including `ttnn.silu` and `ttnn.mul` for the gating, is unique to Ch07. Prior chapters discuss sharding configs but do not show an end-to-end forward function.
- **`code_patterns.md` lines 301–317** (LOFI / HIFI2 compute kernel configs): The `ttnn.WormholeComputeKernelConfig` instantiations are first defined here; they appear nowhere in Ch01–Ch06.
- **`code_patterns.md` lines 321–361** (L1-staging fallback with `expert_ffn_forward_with_l1_stage`): This pattern is unique to Ch07.
- **`correctness_verification.md` lines 46–57** (`compute_pcc` helper using `np.corrcoef`): The PCC computation utility itself is defined only here.
- **`correctness_verification.md` lines 62–153** (step-by-step verification workflow + multi-trial loop): The specific workflow of comparing interleaved vs sharded inference outputs via PCC is unique to Ch07; Ch06 mentions PCC thresholds in a table (`tradeoff_matrix.md`) but does not provide the code.
- **`correctness_verification.md` lines 162–219** (per-projection isolation + PCC failure diagnosis patterns 1–3): All original.
- **`benchmark_methodology.md` lines 30–108** (the `run_benchmark` harness with warmup/timed loop, `ttnn.synchronize_device` timing): Unique to Ch07; no prior chapter provides a timed benchmark harness.
- **`benchmark_methodology.md` lines 155–195** (device memory reporter + Tracy profiler integration): Unique to Ch07.
- **`benchmark_methodology.md` lines 203–247** (`sweep_regimes` multi-regime sweep): Unique to Ch07.

---

## MINOR Suggestions

1. **`code_patterns.md` line 25 (program cache warning):** The warning about `MemoryConfig` affecting program cache keys is stated more fully in Ch06 `shard_setup_overhead.md` lines 183–200. Consider condensing to one sentence with a cross-reference rather than re-explaining the full mechanism.

2. **`index.md` Key Constants table (lines 100–113):** Seven of the nine constants in this table are pulled verbatim from prior chapter index files (Ch04, Ch05, Ch06). The table is useful as a quick-reference summary and its duplication is minor (it is a lookup table, not a code block), but a footnote marking it as "consolidated from prior chapters" would reduce confusion about where values originate.

3. **`correctness_verification.md` lines 225–232 (Expected PCC Ranges table):** The rows for "LOFI vs HIFI2 compute kernel" and "BF16 vs bfloat8_b" essentially restate the PCC threshold table already in `correctness_verification.md` lines 25–32. The two tables can be merged or the second can point to the first.

4. **`benchmark_methodology.md` lines 277–284 (T3K stub):** The T3K comment block (`mesh_device = ttnn.open_mesh_device(...)`) is nearly identical to the T3K mention in Ch06 `tradeoff_matrix.md` (the T3K multi-chip compounding section). This is minor since it is only a comment, not a code block, but it could be removed in favor of a cross-reference.

## Agent A Change Log — C Feedback Pass 1
- code_patterns.md: Condensed ShardSpec function bodies; kept signatures, call sites, and assertion pattern; added ch02/ch05 cross-references
- code_patterns.md: Condensed resharding loop to skeleton (~10 lines); kept memory_config() assertion; added ch06 cross-reference
- benchmark_methodology.md: Replaced verbatim regime delta bullets with ch06 tradeoff_matrix.md cross-reference

## Pass 2 Verification

**Fix 1 — `code_patterns.md`: ShardSpec/MemoryConfig construction blocks**
Confirmed. Both `make_gate_up_shard_config` and `make_down_shard_config` docstrings now carry explicit cross-references to Chapter 2 `constructing_dram_sharded_config.md` (Steps 1–2) and Chapter 5 Rules 1–5. The verbatim ShardSpec construction derivation prose and inline assertion explanations (the former lines 108–114, 153–156) have been removed. The four assertion lines themselves are retained in both functions, clearly marked `# Ch07-original: assert alignment at config-build time` — preserving the unique ch07 contribution. Function signatures and all call sites (Mixtral and Qwen blocks) are intact.

**Fix 2 — `code_patterns.md`: Load-time resharding loop**
Confirmed. The three separate fully-expanded `ttnn.from_torch` list comprehensions (former ~40 lines) are now condensed to three compact single-line comprehensions (~10 lines total), with a comment directing readers to Chapter 6 `shard_setup_overhead.md` Option A and the warning against inference-time resharding. The `reshard_expert_weights` function and its `memory_config()` / `shard_spec()` assertion block (the ch07-original post-reshard check at lines 200–204 in the new file) are fully retained.

**Fix 3 — `benchmark_methodology.md`: Regime expected-outcome ranges**
Confirmed. The three verbatim bullet points (`effective_M ≤ 16: −30 to −50%`, `16 < effective_M ≤ 64: −10 to −25%`, `effective_M > 256: 0 to +5%`) are replaced with the single sentence: "Expected latency deltas per regime are specified in Chapter 6, `tradeoff_matrix.md`." The surrounding result table, "If measured results diverge…" follow-on sentence, benchmark harness code, bandwidth computation, device profiler section, Tracy integration, `sweep_regimes` function, and T3K section are all intact.

### Remaining Crucial Duplications Check

No remaining crucial duplications found. The three duplication sources identified in the original analysis have all been addressed. The remaining content in both files is either ch07-original (assertion-based validation pattern, `reshard_expert_weights` skeleton, `memory_config()` post-reshard check, `expert_ffn_forward`, LOFI/HIFI2 configs, L1-staging fallback, benchmark harness, Tracy integration, `sweep_regimes`) or intentional integration glue (call sites with Mixtral/Qwen parameters) that is unique to ch07 and does not replicate prior chapter content verbatim.

## Crucial updates: no

## Load-Bearing Evidence

All three fixes operated on duplicated content only. The following ch07-original material was identified and confirmed preserved:

- **`code_patterns.md`**: The four `assert` statements in both shard-config helpers (the config-build-time validation pattern) are marked `# Ch07-original` and retained in full. Function signatures and all call sites are intact. The `reshard_expert_weights` loop skeleton with `ttnn.to_memory_config` and the `memory_config()` / `shard_spec()` post-reshard assertion (lines 200–204) are untouched. `expert_ffn_forward`, LOFI/HIFI2 configs, and `expert_ffn_forward_with_l1_stage` are untouched.
- **`benchmark_methodology.md`**: The result table (with `_fill_` placeholders), the "If measured results diverge…" sentence, `run_benchmark` harness, `compute_bandwidth_stats`, device memory reporter block, Tracy profiler integration, `sweep_regimes` multi-regime sweep, and T3K section are all untouched.

## MINOR Suggestions

1. **`code_patterns.md` — `make_gate_up_shard_config` assertion order**: The `d_ff % num_banks == 0` check currently comes after `shard_w` is computed as `d_ff // num_banks`, meaning an assertion failure on the divisibility check would follow a potentially misleading integer-division result. Moving the `assert d_ff % num_banks == 0` check to the top of the function body (before computing `shard_w`) would give a cleaner early-exit error message. This is cosmetic and does not affect correctness.

2. **`code_patterns.md` line 25 (program cache warning)**: Still present verbatim. The original MINOR suggestion from Pass 1 (condense to one sentence with a cross-reference to Ch06 `shard_setup_overhead.md` lines 183–200) was not actioned. This remains a minor item — not a crucial duplication — but could be addressed in a future pass.

3. **`benchmark_methodology.md` — "If measured results diverge…" sentence**: After Fix 3, this sentence reads: "If measured results diverge from these ranges, inspect the Tracy trace…" The phrase "these ranges" now has no local antecedent (the ranges were replaced with a cross-reference). Consider updating to: "If measured results diverge from the Chapter 6 ranges, inspect the Tracy trace…" for referential clarity.
