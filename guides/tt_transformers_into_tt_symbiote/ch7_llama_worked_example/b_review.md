# Agent B Review — Chapter 7: LLaMA Worked Example — Pass 1

## Issues

### Issue 1 — `LlamaAttention.from_torch` incorrectly described as using `TTNNFusedQKVSelfAttention` for LLaMA 3.2-1B (Severity: High)
**File:** step1_module_map.md lines 109 and 112
**What guide said:**
> `TTNNLinear` (created inside `LlamaAttention.from_torch`) … Notes on the attention row: `LlamaAttention.from_torch` internally creates `TTNNFusedQKVSelfAttention` (which wraps a single `TTNNLinear` for fused Q+K+V) and a separate `TTNNLinear` for `o_proj`.

**What source says:**
`attention.py` lines 915–928 (`LlamaAttention.from_torch`):
```python
new_attn.qkv_same_shape = (
    llama_attn.q_proj.weight.shape == llama_attn.k_proj.weight.shape
    and llama_attn.q_proj.weight.shape == llama_attn.v_proj.weight.shape
)
if new_attn.qkv_same_shape:
    new_attn.init_fused_parameters(...)   # creates TTNNFusedQKVSelfAttention
else:
    new_attn.init_parameters()            # creates separate q/k/v/o TTNNLinear
```
`attention.py` lines 909–913 (`init_parameters`):
```python
self.q_proj = TTNNLinear.from_torch(self.torch_layer.q_proj)
self.k_proj = TTNNLinear.from_torch(self.torch_layer.k_proj)
self.v_proj = TTNNLinear.from_torch(self.torch_layer.v_proj)
self.o_proj = TTNNLinear.from_torch(self.torch_layer.o_proj)
```
LLaMA 3.2-1B uses GQA (`num_attention_heads=32`, `num_key_value_heads=8`), so `q_proj` is shape `(2048, 2048)` while `k_proj` and `v_proj` are shape `(512, 2048)`. These are unequal, so `qkv_same_shape=False` and `init_parameters()` is called — producing four separate `TTNNLinear` instances, not `TTNNFusedQKVSelfAttention`.
**Fix:** Replace the claim about `TTNNFusedQKVSelfAttention` with the correct description of the GQA-aware branch. Update both the mapping table row and the notes paragraph.

## Verdict
Issues found requiring another pass

## Change Log — Pass 1 fixes applied

- **step1_module_map.md**: Corrected the attention mapping table row and the notes paragraph to accurately describe the GQA-aware branching logic in `LlamaAttention.from_torch`. The fused path (`TTNNFusedQKVSelfAttention`) applies only when Q/K/V projection shapes are equal (equal-head-count models); for LLaMA 3.2-1B with GQA the `init_parameters()` path is taken, producing four separate `TTNNLinear` instances for `q_proj`, `k_proj`, `v_proj`, and `o_proj`.

---

# Agent B Review — Chapter 7: LLaMA Worked Example — Pass 2

## Pass 1 Fix Verification

The GQA branching fix is correctly applied in `step1_module_map.md`. Both the mapping table row (line 109) and the notes paragraph (line 112) now accurately describe the `qkv_same_shape` check in `LlamaAttention.from_torch`. The corrected text states that for LLaMA 3.2-1B (`num_attention_heads=32`, `num_key_value_heads=8`) `qkv_same_shape` is `False` and `init_parameters()` is called, producing four separate `TTNNLinear` instances (`q_proj`, `k_proj`, `v_proj`, `o_proj`), and that `TTNNFusedQKVSelfAttention` is only used when all three projection shapes are equal. This matches `attention.py` lines 921–928 exactly.

## New issues found

### Issue 1 — `save_stats_to_file` top-30 printout described as indexed by module name, but source indexes by `func_name` (Severity: Medium)

**File:** step3_validation_and_benchmarking.md, section 2 "Identifying Fallback Layers"

**What the guide said:**
> `save_stats_to_file` also prints the top-30 **modules** by `TorchModules` duration to stdout. Modules that appear in this list with large values and zero `TTNN` time are the highest-priority candidates for replacement.

**What the source says (`run_config.py` lines 306–313):**
```python
func_times = df.pivot_table(
    index=["func_name"], columns="backend", values="duration", aggfunc="sum", fill_value=0
)
module_times = func_times[func_times["TorchModules"] != 0]["TorchModules"].sort_values(ascending=False)
print("Top 30 Modules by total duration (s):\n", module_times.head(30))
```
The pivot uses `index=["func_name"]`, not `module_name`. Each row represents a **class name** (e.g., `TTNNLinear`, `LlamaAttention`), not an individual module path (e.g., `model.model.layers.3.mlp.gate_proj`). The top-30 list shows which module **classes** consume the most `TorchModules` time in aggregate, not which specific layer instances do.

**Fix applied:** Updated the description to say "top-30 entries aggregated by `func_name` (class name)" rather than "top-30 modules by module name."

## Verdict

Approved

## Change Log — Pass 2 fixes applied

- **step3_validation_and_benchmarking.md**: Corrected the description of the `save_stats_to_file` top-30 printout. The printout aggregates by `func_name` (class name), not by individual `module_name` paths. Updated wording from "top-30 modules" to "top-30 entries by `func_name` (i.e., by class name)."

---

# Agent B Review — Chapter 7: LLaMA Worked Example — Post-Compression Review

## Review scope

Verified the post-compression state of all four guide files after the two compression passes documented in `compression_analysis.md`. Checked each of the four CRUCIAL cut locations (C1–C4) for clean surrounding prose (no dangling sentences, broken references, or missing context). Spot-checked key factual claims in `step1_module_map.md`, `step2_precision_config.md`, and `step3_validation_and_benchmarking.md` against the source files listed below, and scanned the full text of all four files for any new factual errors introduced or exposed by the cuts.

## Sources checked

- `/localdev/salnahari/testing_dir/tt-symbiote-research-topics/sources/attention.py` — **not found in repository** (absent from filesystem; prior passes document its content via quoted excerpts in `b_review.md` Pass 1 and Pass 2)
- `/localdev/salnahari/testing_dir/tt-symbiote-research-topics/sources/linear.py` — **not found in repository**
- `/localdev/salnahari/testing_dir/tt-symbiote-research-topics/sources/run_config.py` — **not found in repository**

All three source files are absent from the repository at the paths specified in the task instructions. Factual verification in this pass is therefore performed by:
1. Cross-checking internal consistency across the four guide files.
2. Relying on the documented source excerpts from Passes 1 and 2 (quoted verbatim in this file above).
3. Confirming that the Pass 1 and Pass 2 fixes remain correctly in place and were not disturbed by the compression edits.

## Issues found

None.

The two substantive factual errors identified and corrected in prior passes (GQA branching in Pass 1; `save_stats_to_file` `func_name` aggregation in Pass 2) are both still correctly applied in the current files. No new errors were introduced or exposed by any of the four compression cuts.

## Detailed per-location findings

**C1 — step1_module_map.md, post-`LlamaMLP` code block (original lines 66–70)**

Clean. The code block ends at line 64 (`return down_proj`). The section heading `### How the Two-Pass Replacement Works` follows immediately at line 66. There is no dangling sentence, missing antecedent, or broken reference. The code block is self-documenting (`self.gate_proj`, `self.up_proj`, `self.down_proj`, `self.act_fn = nn.SiLU()` are all visible), so no prose context was lost for the reader.

**C2 — step1_module_map.md, post-two-pass-code-block prose (original lines 93–96)**

Clean. The two-pass code block ends at line 86 (`modules = register_module_replacement_dict(...)`). Section `## 3. Complete Mapping Table` begins at line 88. The `## 3` heading provides an immediate anchor; the removed sentences about "After pass 1 all 16 `LlamaMLP` instances…" and "The `test_llama_intelligent` variant follows the same two-pass pattern…" are redundant with the `# Pass 1` / `# Pass 2` comments in the code block and the mapping table in section 3 respectively. No context is missing.

**C3 — step2_precision_config.md, section 1 opening sentences (original lines 3–5)**

Clean. Section 1 now opens at line 5 with "`TTNNLinear.preprocess_weights_impl` stores weights in `ttnn.bfloat16`:" followed directly by the code block. The section heading "Current Symbiote Precision in `test_llama.py`" and the summary table that follows the code block together carry all the orientation the removed sentences provided. No dangling reference.

**C4 — step2_precision_config.md, post-table prose (original lines 101–103)**

Clean. Section 3 ends at the recommended mapping table (line 97). Section `## 4. Verifying Output Quality Using DPL Mode` begins at line 99. The removed sentences described `TTNNLinearLLama`, `TTNNLinearLLamaBFloat16`, and `SmartTTNNLinearLLama` differences — information fully covered by the "Notes" column of the table immediately above. The transition to section 4 is natural and requires no bridging prose.

## Verdict

Approved

## Change Log

None.
