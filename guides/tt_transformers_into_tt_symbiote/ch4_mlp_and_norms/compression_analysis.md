# Compression Analysis: Chapter 4 — MLP Layers and Norms — Pass 1

## Summary

- Total files: 4
- Current lines: ~1067
- Post-compression lines: ~1027
- Reduction: ~3.7%

---

## CRUCIAL Suggestions

### C1 — `tt_transformers_mlp.md` §1.2: Docstring block duplicates the weight-mapping table

**Location:** Lines 38–57. The Markdown table (lines 40–45) maps `w1/w2/w3` to their
HuggingFace names and roles. The docstring block immediately below (lines 46–52) repeats
all three mappings verbatim:

```
w1 -> gate_proj
w2 -> down_proj
w3 -> up_proj
HF reference: self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

The table already documents all three mappings and the HF formula is also shown in
Section 2.4's code block. The docstring block adds no new information.

**Suggestion:** Remove the inline docstring block (the fenced code block at lines 46–52
plus its lead-in sentence "From the docstring at the top of `MLP.forward`..."). Keep the
table; keep the `mlp.py` line reference on line 34.

**Estimated saving:** ~8 lines.

---

### C2 — `tt_transformers_mlp.md` §1.5: Prose restates the two-line sharding code

**Location:** Lines 130–141. The code block (lines 130–133) shows:
```python
w1_dims = (-1, -2) if args.is_galaxy else (-2, -1)
w2_dims = (-2, -1) if args.is_galaxy else (-1, -2)
```
The paragraph that follows (lines 135–141) re-describes these in prose: "w1/w3 are
column-parallel: sharded on `dim=-1`... w2 is row-parallel: sharded on `dim=-2`...
On Galaxy the axes are swapped."

The code is self-explanatory and the terms "column-parallel" and "row-parallel" do not
introduce any new operational fact beyond what the code and its heading ("Sharding
dimensions") already communicate.

**Suggestion:** Remove the three-sentence prose block (lines 135–141). Retain the code
block and source reference.

**Estimated saving:** ~7 lines.

---

### C3 — `tt_transformers_mlp.md` §2.3: Prose preempts the code

**Location:** Lines 204–205. The sentence "There is no fused `ttnn.matmul(w1, w3)` path;
they are two separate `ttnn.linear` calls" is immediately followed by code showing exactly
two separate `ttnn.linear` calls. The prose restates what the code makes obvious.

**Suggestion:** Remove the sentence "There is no fused `ttnn.matmul(w1, w3)` path; they
are two separate `ttnn.linear` calls" (lines 204–205). The code block is sufficient.

**Estimated saving:** ~2 lines.

---

### C4 — `tt_transformers_mlp.md` §2.4: Post-code sentence restates the fused op

**Location:** Line 239. After the `ttnn.mul(...)` code block, the sentence "The fused op
simultaneously applies `silu(w1_out) * w3_out`, matching the SwiGLU formula" restates
what `input_tensor_a_activations=[self.activation_type]` with the default `SILU` already
communicates, plus what the docstring comment at the top of Section 2 already states.

**Suggestion:** Remove the sentence at line 239 ("The fused op simultaneously applies...").
Retain the sentences about deallocation and the optional memory config call (lines 240–242)
which document operational ordering not visible in the code block.

**Estimated saving:** ~1 line.

---

### C5 — `symbiote_mlp.md` §2.1: "The guard ensures..." restates the shown code

**Location:** Lines 45–46. After the `preprocess_weights` code block, the sentence "The
guard ensures `preprocess_weights_impl()` is called at most once." restates the
`if not self._preprocessed_weight: ... else: return` guard that is directly visible in
the code block above it.

**Suggestion:** Remove the sentence "The guard ensures `preprocess_weights_impl()` is
called at most once." (line 45). The guard pattern is self-evident from the code.

**Estimated saving:** ~1 line.

---

### C6 — `symbiote_mlp.md` §2.2: "Again guarded for idempotency." restates the code

**Location:** Line 67. After the `move_weights_to_device` code block (which shows the
identical `if not self._weights_on_device: ... else: return` guard), the sentence "Again
guarded for idempotency." adds no new information.

**Suggestion:** Remove the sentence "Again guarded for idempotency." (line 67).

**Estimated saving:** ~1 line.

---

### C7 — `symbiote_mlp.md` §6.2: Prose restates the code comment

**Location:** Lines 221–222. After the `preprocess_weights_impl` code block that already
contains the comment `# Stored as raw torch.Tensor; sharding deferred to move_weights_to_device_impl`,
the prose "Unlike `TTNNLinear`, no layout conversion happens at preprocessing time. The
raw torch tensor is stored and sharding/layout conversion is deferred to
`move_weights_to_device_impl`." duplicates that comment verbatim.

**Suggestion:** Remove the two-sentence prose block (lines 221–222). The code comment is
sufficient, and the "Unlike `TTNNLinear`" contrast is already evident from the structural
difference between Section 3.2 and Section 6.2.

**Estimated saving:** ~2 lines.

---

### C8 — `symbiote_mlp.md` §6.3: Prose restates dtype and dim already in code

**Location:** Lines 240–241. "The weight is sharded on `dim=-2` across the mesh
(column-parallel weight sharding). Dtype is `bfloat16`." directly restates what is shown
in the code block above: `dtype=ttnn.bfloat16` and `dim=self.weight_dim` where
`weight_dim=-2` is set in the constructor shown in the section header.

**Suggestion:** Remove lines 240–241. The code block already documents both facts.

**Estimated saving:** ~2 lines.

---

### C9 — `integration_gaps.md` §2.1: TT Transformers paragraph restates the table

**Location:** Lines 39–41. "**TT Transformers:** `MLP.__init__` allocates all three
weights and `MLP.forward` executes the complete
`down_proj(silu(gate_proj(x)) * up_proj(x))` pipeline in one call." restates the first
row of the comparison table in §1 ("Three-weight SwiGLU MLP as a single module | Yes —
`MLP` encapsulates `w1`, `w2`, `w3`"). No new information is introduced.

**Suggestion:** Remove the "**TT Transformers:**" paragraph (lines 39–41); keep only the
"**TT Symbiote:**" paragraph (lines 43–46) and the "**What to implement:**" paragraph
(lines 48–49), which both contain information not in the table.

**Estimated saving:** ~3 lines.

---

### C10 — `integration_gaps.md` §2.2: TT Transformers paragraph restates the table and code

**Location:** Lines 53–57. The "**TT Transformers:**" paragraph re-describes the fused
`ttnn.mul` call (same content already in the §1 table row for "Fused SwiGLU activation")
and adds "This is a single kernel that fuses the SiLU activation with the element-wise
multiply" — which restates the `input_tensor_a_activations=[ttnn.UnaryOpType.SILU]`
shown in the code block.

**Suggestion:** Remove the "**TT Transformers:**" paragraph (lines 53–57) and the code
block that follows (which is also shown in `tt_transformers_mlp.md` §2.4). Retain only
the "**TT Symbiote:**" paragraph and "**What to implement:**" paragraph.

**Estimated saving:** ~8 lines (prose + duplicated code block).

---

### C11 — `integration_gaps.md` §2.8: Two sentences state the same absence

**Location:** Lines 148–150. "The `modules/linear.py` file and the rest of the
`tt_symbiote` module tree (as visible from the files provided) contain no normalization
module. There is no `TTNNRMSNorm`, `TTNNLayerNorm`, or equivalent class." Both sentences
assert the same fact (no norm class exists). The second adds only example names, which
are already implied by "no normalization module."

**Suggestion:** Merge into one sentence, e.g., "No normalization module exists in the
`tt_symbiote` module tree — no `TTNNRMSNorm`, `TTNNLayerNorm`, or equivalent."

**Estimated saving:** ~1 line.

---

## MINOR Suggestions

### M1 — `tt_transformers_mlp.md` §2.4: Hedging qualifier in deallocation note

**Location:** Line 240. "Both `w1_out` and `w3_out` are deallocated after the multiply
(`mlp.py` lines 248–249), but only after an optional `ttnn.to_memory_config` call at
line 246 and before the optional Galaxy `all_gather_async` block (lines 251–266) that
operates on `w2_in`." The word "optional" appears twice; the sentence is verbose but
the ordering information it documents is non-obvious and operational.

**Assessment:** Do not remove. The ordering detail (dealloc after memory config move,
before Galaxy all-gather) is load-bearing. Tag as MINOR only; no change recommended.

---

### M2 — `symbiote_mlp.md` §2.4: "device must be set before" is trivially implied

**Location:** Line 93. "device must be set before `move_weights_to_device()` is called."
The `assert self.device is not None` in the code block already enforces this.

**Assessment:** Borderline MINOR. The assert is shown in the code block, but the prose
note makes the developer-facing requirement explicit in a way the assert alone may not.
Recommend leaving as-is.

---

### M3 — `symbiote_mlp.md` §10: Code comments re-describe each step heading

The annotated user-sequence code block (lines 361–381) uses comments like
`# 1. Construct the module`, `# 5. Preprocess (layout conversion, tile conversion — host side)`
alongside surrounding prose that repeats "Steps 5 and 6 are separated to allow
host-side preprocessing to happen in advance." The final prose sentence (lines 383–384)
paraphrases what the code comments already explain.

**Assessment:** MINOR. The code block is pedagogically useful. The trailing two-sentence
prose (lines 383–384) could be dropped without information loss. Estimated saving: ~2 lines.

---

### M4 — `integration_gaps.md` §2.3: TT Symbiote paragraph restates the table cell

**Location:** Lines 73–75. "All classes call `ttnn.to_device(self.tt_weight_host, self.device)`,
which places the tensor in interleaved DRAM." The table row already states "No — all
classes use `ttnn.DRAM_MEMORY_CONFIG` (interleaved)." However, the sentence continues
with "No `MemoryConfig` with `TensorMemoryLayout.WIDTH_SHARDED` and a `ShardSpec` is
ever constructed" which adds specificity not in the table.

**Assessment:** The second sentence is load-bearing (names the missing config type
precisely). Only the first clause ("All classes call `ttnn.to_device`... interleaved
DRAM") is redundant with the table. MINOR trim possible but not critical.

---

## Load-Bearing Evidence

The following facts are non-obvious, operational, and must not be removed:

1. **`tt_transformers_mlp.md` §1.3:** On Galaxy, `w1`/`w3` output activation is always
   `ttnn.bfloat8_b` regardless of optimization mode (`mlp.py` line 148). Weight dtype
   for `w1`/`w3` is still resolved via `decoders_optimizations.get_tensor_dtype` on all
   devices including Galaxy.

2. **`tt_transformers_mlp.md` §1.4:** `create_dram_sharded_mem_config` returns a
   `WIDTH_SHARDED` config; bank counts differ by chip (12 on Wormhole, 8 on P150, 7 on
   P100). On Galaxy, `as_sharded_tensor` uses plain `ttnn.DRAM_MEMORY_CONFIG` instead.

3. **`tt_transformers_mlp.md` §1.6:** The 4D `unsqueeze(0).unsqueeze(0)` is noted as
   critical for the DRAM prefetcher to correctly interpret all weights.

4. **`tt_transformers_mlp.md` §1.8:** Prefetcher registration order is `w1, w3, w2`
   (not `w1, w2, w3`) to match forward execution order.

5. **`tt_transformers_mlp.md` §2.2:** `prefill_len_cutoff` is 1024 on Wormhole, 512 on
   Blackhole — the reshape threshold differs by chip architecture.

6. **`tt_transformers_mlp.md` §2.4:** `self.activation_type` defaults to
   `ttnn.UnaryOpType.SILU` but can be overridden via `args.mlp_activation_type`.

7. **`tt_transformers_mlp.md` §2.7:** In Galaxy (TG), `all_gather_async` on `w2_in`
   occurs after the `ttnn.mul` SwiGLU step and before the `w2` down-projection matmul —
   not between `w1`/`w3` outputs and the element-wise multiply.

8. **`tt_transformers_mlp.md` §3.2:** RMSNorm weight is reshaped from `[dim]` to
   `[1, 1, dim // 32, 32]` before device placement; `fp32_dest_acc_en=False` for
   Qwen2.5-7B specifically.

9. **`tt_transformers_mlp.md` §3.4:** `pre_ff_norm` and `post_ff_norm` are
   conditionally present only if their state-dict keys exist; the residual-add ordering
   differs when `pre_ff_norm` is present (residual add before `pre_ff_norm`, not after
   `ff_norm`).

10. **`symbiote_mlp.md` §3.1:** `from_parameters` calls `preprocess_weights()` immediately
    and then deletes `self.weight` and `self.bias`, while `from_torch` stores weights
    directly and does NOT call `preprocess_weights()` — these behave differently and
    the distinction is only documented here.

11. **`symbiote_mlp.md` §6.3:** `@run_on_devices(DeviceArch.T3K)` checks architecture
    via `os.environ.get("MESH_DEVICE")`, not from the device object directly.

12. **`symbiote_mlp.md` §6.3:** The reduce-scatter uses `cluster_axis=1` (analogous to
    Galaxy's reduce-scatter on `w1_out`/`w3_out`), not `cluster_axis=0` (which is used
    by `tt_all_reduce` after `w2`).

13. **`integration_gaps.md` §2.6:** The user must call `tt_all_reduce` from
    `models.tt_transformers.tt.ccl` manually after the down-projection linear — the
    import path is documented only here.

14. **`integration_gaps.md` §3:** Priority-ordered list of gaps by performance impact is
    a synthesis not present in any other file.

---

## VERDICT

- Crucial updates: **yes**

Eleven CRUCIAL issues found across three files. Total estimated saving: ~35 lines (~3.3%
from the combined 1067-line corpus). The `index.md` file has no redundancy and requires
no changes.

---

## Change Log — Pass 1 CRUCIAL fixes applied

The following changes were applied to the source files.

### `tt_transformers_mlp.md`

**C1 applied:** Removed the lead-in sentence "From the docstring at the top of
`MLP.forward` (`mlp.py` line 121):" and the fenced docstring code block (lines 46–52
in original). The weight-mapping table above it fully captures the same information.

**C2 applied:** Removed the three-sentence prose paragraph (lines 135–141 in original)
that described `w1`/`w3` as column-parallel and `w2` as row-parallel on non-Galaxy and
stated axes are swapped on Galaxy. The two-line code block is self-sufficient.

**C3 applied:** Removed the sentence "There is no fused `ttnn.matmul(w1, w3)` path;
they are two separate `ttnn.linear` calls" (lines 204–205 in original). The code block
immediately following makes this plain.

**C4 applied:** Removed the sentence "The fused op simultaneously applies
`silu(w1_out) * w3_out`, matching the SwiGLU formula." (line 239 in original). The code
block's `input_tensor_a_activations=[self.activation_type]` with the preceding
`# default: ttnn.UnaryOpType.SILU` comment is sufficient.

### `symbiote_mlp.md`

**C5 applied:** Removed the sentence "The guard ensures `preprocess_weights_impl()` is
called at most once." (line 45 in original). The guard is directly visible in the code
block.

**C6 applied:** Removed the sentence "Again guarded for idempotency." (line 67 in
original).

**C7 applied:** Removed the two-sentence prose block "Unlike `TTNNLinear`, no layout
conversion happens at preprocessing time. The raw torch tensor is stored and
sharding/layout conversion is deferred to `move_weights_to_device_impl`." (lines 221–222
in original). The code comment inside the block already states this.

**C8 applied:** Removed the two sentences "The weight is sharded on `dim=-2` across the
mesh (column-parallel weight sharding). Dtype is `bfloat16`." (lines 240–241 in
original). Both facts are directly shown in the `move_weights_to_device_impl` code block.

### `integration_gaps.md`

**C9 applied:** Removed the "**TT Transformers:**" paragraph in §2.1 (lines 39–41 in
original). The table row in §1 already captures the same content. Retained the
"**TT Symbiote:**" and "**What to implement:**" paragraphs.

**C10 applied:** Removed the "**TT Transformers:**" paragraph and duplicated code block
in §2.2 (lines 53–57 and the `ttnn.mul(...)` code block in original). The table row in
§1 documents the TT Transformers behaviour; the code is already shown in
`tt_transformers_mlp.md` §2.4. Retained the "**TT Symbiote:**" and "**What to
implement:**" paragraphs.

**C11 applied:** Merged the two-sentence statement of absence in §2.8 into one: "No
normalization module exists in the `tt_symbiote` module tree — no `TTNNRMSNorm`,
`TTNNLayerNorm`, or equivalent class."

### `index.md`

No changes. File is clean.

---

# Compression Analysis: Chapter 4 — MLP and Norms — Pass 2

## Summary
- Files analyzed: 6 (4 previously compressed + 2 new)
- Current line count: ~1699 lines
- Estimated post-compression: ~1694 lines
- Estimated reduction: ~0.3%

---

## Pass 1 Verification (for original 4 files)

All eleven Pass 1 fixes (C1–C11) are confirmed still in place:

- **`tt_transformers_mlp.md`:** C1 docstring block absent; C2 prose paragraph after `w1_dims`/`w2_dims` code absent; C3 "no fused matmul path" sentence absent; C4 post-mul restatement sentence absent.
- **`symbiote_mlp.md`:** C5 "guard ensures called at most once" sentence absent; C6 "Again guarded for idempotency" sentence absent; C7 "Unlike TTNNLinear, no layout conversion" prose block absent; C8 `dim=-2` / `bfloat16` restatement sentences absent.
- **`integration_gaps.md`:** C9 TT Transformers paragraph in §2.1 absent; C10 TT Transformers paragraph and duplicate code block in §2.2 absent; C11 two-sentence absence statement merged (§2.8 now correctly describes partial coverage, consistent with the new `normalization_comparison.md`).
- **`index.md`:** No changes required; still clean.

---

## CRUCIAL Suggestions

### NC1 — `normalization_comparison.md` §1.1: Compute kernel config prose restates a code block in `tt_transformers_mlp.md` §3.2

**Location:** Lines 34–35. The sentence "**Compute kernel config.** All calls use
`WormholeComputeKernelConfig` with `MathFidelity.HiFi2`, `math_approx_mode=False`,
`fp32_dest_acc_en=True` (default)." restates, in prose, the identical values shown in
the code block at `tt_transformers_mlp.md` §3.2 (lines 351–358), which is the authoritative
reference for `RMSNorm`. `normalization_comparison.md` §1.1 does not even reproduce the
code block — it merely paraphrases the three config values. The `HiFi2` fact is
load-bearing, but it is retained in `tt_transformers_mlp.md` §3.2 and is independently
documented in the gap at `normalization_comparison.md` §4.5. Removing the two-line prose
sentence from §1.1 does not destroy any information.

**Suggestion:** Remove the two-line "**Compute kernel config.**" sentence (lines 34–35 of
`normalization_comparison.md`). The surrounding weight-layout prose and distributed-path
code block in §1.1 are sufficient.

**Estimated saving:** ~2 lines.

---

### NC2 — `normalization_comparison.md` §4.1: TT Transformers paragraph restates §1.3 verbatim

**Location:** Lines 205–208. The "**TT Transformers:**" paragraph in gap §4.1 reads:
"The flag is wired in every `RMSNorm` constructor call in `decoder.py` and `model.py`
via `add_unit_offset=self.args.rms_norm_add_unit_offset`. When `True`,
`torch_weight = torch_weight + 1.0` is applied before upload and before any disk cache
write." This repeats §1.3 (lines 91–98) verbatim: §1.3 already states the wiring call
name, the `torch_weight + 1.0` operation, the "before upload and before caching" timing,
and the Gemma purpose. The gap §4.1 "**TT Symbiote:**" and "**What to add:**" paragraphs
are load-bearing and must be retained.

**Suggestion:** Remove the "**TT Transformers:**" paragraph (lines 205–208) from §4.1.
The reader can find the TT Transformers implementation description in §1.3. Retain the
"**TT Symbiote:**" and "**What to add:**" paragraphs.

**Estimated saving:** ~4 lines (the paragraph itself plus its blank-line separator).

---

## MINOR Suggestions

### NM1 — `normalization_comparison.md` §1.4: Norm positions table near-duplicates `tt_transformers_mlp.md` §3.1

**Location:** Lines 112–124 of `normalization_comparison.md`. The four-row table
(attribute / `weight_key` / used-before / always-present) closely mirrors the table at
`tt_transformers_mlp.md` §3.1. The `normalization_comparison.md` version adds one extra
fact: `ff_norm.enable_all_gather` is set to `False` when `pre_ff_norm` is present.

**Assessment:** MINOR. The `enable_all_gather` detail is load-bearing and appears only
here. The table format in a dedicated comparison document has navigational value for
readers focused on normalization. Recommend leaving as-is.

---

### NM2 — `decoder_block_assembly.md` §2.3: "No `DistributedNorm` wrapper" appears in three files

**Location:** `decoder_block_assembly.md` lines 148–150; `normalization_comparison.md`
§4.4; `integration_gaps.md` §2.8. All three mention the absence of a `DistributedNorm`
wrapper in Symbiote, but each does so in a different context (assembly recipe, norm
comparison, and MLP gap summary respectively). No file is purely redundant; each usage
is brief and serves a different audience entry point.

**Assessment:** MINOR. The brief mentions are orienting context rather than full
descriptions. Recommend leaving as-is.

---

## VERDICT
- Crucial updates: **yes**

Two CRUCIAL issues found in `normalization_comparison.md`. Total estimated saving: ~6 lines
(~0.35% from the combined 1699-line corpus). The two previously uncompressed files
(`normalization_comparison.md`, `decoder_block_assembly.md`) are otherwise clean.
`decoder_block_assembly.md` has no CRUCIAL issues.

---

## Change Log — Pass 2 CRUCIAL fixes applied

### `normalization_comparison.md`

**NC1 applied:** Removed the two-line "**Compute kernel config.**" sentence from §1.1
(original lines 34–35: "**Compute kernel config.** All calls use
`WormholeComputeKernelConfig` with `MathFidelity.HiFi2`, `math_approx_mode=False`,
`fp32_dest_acc_en=True` (default)."). The authoritative code block in
`tt_transformers_mlp.md` §3.2 and the gap description in §4.5 of this file both retain
the `HiFi2` detail.

**NC2 applied:** Removed the "**TT Transformers:**" paragraph from §4.1 (original lines
205–208). Section 1.3 of this file already fully documents the `add_unit_offset` flag
wiring and implementation. The "**TT Symbiote:**" and "**What to add:**" paragraphs are
retained.

### `decoder_block_assembly.md`

No changes. File is clean.

### `index.md`

No changes. File is clean.

---

# Compression Analysis: Chapter 4 — MLP and Norms — Pass 3

## Summary

- Files re-analyzed: 6
- Pass 2 fixes verified: yes (NC1, NC2 both confirmed in place)
- New CRUCIAL redundancies found: 0
- Guide files modified: 0

---

## Pass 2 Verification

Both Pass 2 fixes are confirmed still in place in `normalization_comparison.md`:

- **NC1:** The "**Compute kernel config.**" sentence (original lines 34–35) is absent from §1.1. The section goes directly from the `forward` signature block to the distributed path description.
- **NC2:** The "**TT Transformers:**" paragraph (original lines 205–208) is absent from §4.1. The section opens immediately with "**TT Symbiote:**" followed by "**What to add:**".

---

## CRUCIAL Suggestions

None.

---

## MINOR Suggestions

No new MINOR issues beyond those already catalogued in Pass 1 (M1–M4) and Pass 2 (NM1–NM2).

One borderline observation for the record:

### PM1 — `decoder_block_assembly.md` §2.3 partially anticipates §4.2

**Location:** `decoder_block_assembly.md` lines 145–148. The two sentences "Residual additions remain PyTorch... No TTNN memory config is specified; the output lands wherever TTNN defaults to (typically interleaved DRAM)." are a brief coverage-inventory statement. Section 4.2 covers the same mechanism in detail, with `TorchTTNNTensor.__add__` dispatch context, a code example, and a quantified performance impact table.

**Assessment:** MINOR. The §2.3 mention is a brief orientation note in a coverage inventory; §4 is a full performance analysis. The two serve different reader purposes. No change recommended.

---

## VERDICT

- Crucial updates: **no**

All six files are clean after Passes 1 and 2. No CRUCIAL redundancies remain.

---

## Change Log — Pass 3 fixes applied

None.
