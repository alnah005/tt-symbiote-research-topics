# Chapter 5 -- Change Log

## 2026-04-09 -- Module Catalog Update (Agent B Feedback Round 1)

**File modified:** `module_catalog.md`

**Changes applied:**

1. **Added 3 missing classes from `modules/linear_intelligent.py` to the Linear Modules table:**
   - `SmartTTNNLinear` -- extends `TTNNLinear` with automatic prefill/decode dispatch based on sequence length (threshold: 32 tokens). Caches `MatmulMultiCoreReuseMultiCastProgramConfig` per sequence length for the prefill path.
   - `SmartTTNNLinearLLama` -- bfloat8_b precision variant with `@deallocate_weights_after` decorator for memory efficiency.
   - `SmartTTNNLinearLLamaBFloat16` -- bfloat16 precision variant with `@deallocate_weights_after` decorator.

2. **Added descriptive paragraph** explaining the Smart linear family's prefill/decode dispatch pattern and per-class differences.

3. **Updated summary counts:**
   - Linear module count: 12 -> 15
   - Total class count: ~50 -> ~53
   - Pain-point variant count: 8 -> 11 (added "dispatch-mode" as a cross-cutting concern dimension)

## 2026-04-09 -- Agent B Pass 2 Feedback

**Files modified:** `module_catalog.md`, `ttnn_module_lifecycle.md`

**Changes applied:**

1. **`module_catalog.md` -- Added missing `TTNNLinearInputReplicatedWeightSharded`:** This class (defined at line 252 of `modules/linear.py`) is the parent of `TTNNLinearIReplicatedWColSharded`. It accepts a `weight_dim` parameter and shards weights via `ttnn.shard_tensor_to_mesh_mapper`. Added to the Linear Modules table with precision bfloat16, sharding I:replicated/W:sharded (dim=-1), no CCL, trace enabled. Updated linear module count 15 -> 16, total class count ~53 -> ~54, and pain-point variant count 11 -> 12.

2. **`ttnn_module_lifecycle.md` -- Fixed dataclass attribution:** `DistributedTensorConfig`, `DistributedConfig`, and `CCLManagerConfig` are defined in `core/run_config.py`, not `core/module.py`. Added an explicit **Source:** `core/run_config.py` header to the Distributed Configuration section to correct the attribution.

## 2026-04-09 -- Compression Pass 1 (Agent C)

### Summary

Chapter 5 (4 files, ~460 lines total) documents TT-Symbiote's architecture across module lifecycle, dispatch system, and module catalog. The prose is generally well-structured and information-dense. However, there is substantial **cross-file repetition** of pain-point descriptions and TT-Lang opportunity framing, redundant enumeration of the same handler boilerplate pattern, and hedging/motivational language that restates rather than adds information.

### CRUCIAL Suggestions

Crucial updates: no

### MINOR Suggestions

1. **Repeated pain-point framing across all four files.** The observation that "linear variants exist because sharding/precision/CCL/trace are cross-cutting concerns" is stated in `index.md` (line 72, bullet 2; line 76, bullet 5), `ttnn_module_lifecycle.md` (lines 122-133, "The Boilerplate Burden"), and `module_catalog.md` (lines 217-222, "Cross-Cutting Pain Points" items 1-2). Consolidate into one authoritative location (recommend `module_catalog.md` since it has the concrete data) and cross-reference from the others.

2. **Dispatch handler anatomy explained twice.** `dispatch_system.md` describes the prepare/call/wrap/cleanup pattern at lines 122-148 ("Anatomy of a Handler") and then restates the identical pattern at lines 192-196 ("Pain Point 2: Repeated Boilerplate Across Handlers"), including re-listing the same helper functions. The pain-point section could simply reference the earlier anatomy section rather than re-enumerating the steps.

3. **"TT-Lang opportunity" blocks are formulaic and repetitive.** Five consecutive pain-point sections in `dispatch_system.md` (lines 173-216) each end with a "TT-Lang opportunity:" paragraph. These share overlapping suggestions (declarative mappings, auto-generation, compile-time validation). A single consolidated "TT-Lang Opportunities" subsection would eliminate ~40% of the text in that span without losing information.

4. **`index.md` "Key Takeaways" largely duplicate the sub-file content.** The six bullets at lines 70-80 preview points that are then developed fully in the sub-files. Since this is a chapter index, the bullets could be shortened to one-line pointers rather than multi-clause sentences that restate details available one click away.

5. **Hedging/motivational prose.** Several passages include phrasing that motivates rather than documents: "Understanding its lifecycle is essential for identifying where TT-Lang can reduce friction" (`ttnn_module_lifecycle.md`, line 5), "Understanding the full breadth of modules reveals where boilerplate concentrates" (`module_catalog.md`, line 5). These could be cut without information loss.

6. **Shared helpers listed twice in `dispatch_system.md`.** The four helpers (`_prepare_tensor_input`, `_prepare_binary_inputs`, `ensure_tile_layout`, `_cleanup_tensors`) are listed at lines 154-160 ("Shared Helpers") and again at line 194 in Pain Point 2. Deduplicate.

7. **`module_catalog.md` normalization "Pattern" paragraph** (lines 70-72) restates the generic lifecycle from `ttnn_module_lifecycle.md` verbatim ("from_torch() extracts weights, preprocess_weights_impl() converts to TTNN, move_weights_to_device_impl() transfers to device, forward() calls the TTNN op"). Since the lifecycle is already fully documented in the prior file, a brief "Follows the standard 3-phase lifecycle" with a cross-reference would suffice.

8. **Verbose inline list in dispatch table description.** `dispatch_system.md` lines 118 contains a ~90-word parenthetical enumerating every op category already visible in the code block above it. This could be removed or reduced to "covering arithmetic, activations, shape ops, comparisons, reductions, memory, and advanced operations."

### Load-Bearing Evidence

- **`index.md`**: "The 3-phase module lifecycle (preprocess_weights / move_weights_to_device / forward + deallocate_weights) is powerful but imposes 3--4 method overrides per module, creating significant boilerplate that scales with the number of module variants." (line 72) -- restated at `ttnn_module_lifecycle.md` lines 124-131 and `module_catalog.md` lines 219-222.
- **`ttnn_module_lifecycle.md`**: "Understanding its lifecycle is essential for identifying where TT-Lang can reduce friction." (line 5) -- motivational hedging with no documentary content.
- **`dispatch_system.md`**: "The deferred import of TorchTTNNTensor, the _prepare_binary_inputs / _prepare_tensor_input calls, ensure_tile_layout, and _cleanup_tensors pattern appear in almost every handler." (line 194) -- duplicates the "Shared Helpers" section at lines 154-160.
- **`module_catalog.md`**: "Each follows the same lifecycle --- from_torch() extracts weights, preprocess_weights_impl() converts to TTNN, move_weights_to_device_impl() transfers to device, forward() calls the TTNN op." (lines 70-71) -- verbatim restatement of the lifecycle documented in `ttnn_module_lifecycle.md`.

### VERDICT

No crucial changes. Eight minor compression opportunities identified, primarily cross-file duplication of pain-point descriptions and TT-Lang opportunity framing. Estimated recoverable space: ~60-80 lines (~15% of total) through deduplication and consolidation. Content is otherwise well-organized and information-dense.
