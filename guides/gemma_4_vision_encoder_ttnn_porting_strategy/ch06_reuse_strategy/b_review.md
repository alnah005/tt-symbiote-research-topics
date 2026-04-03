# Chapter 6 — Critic (Agent B) Review

## Issue 1: Missing `gemma4_vision_pooler.py` from the reuse scorecard

**Location:** `index.md` scorecard table (lines 38-51) and `new_implementation_modules.md`

Chapter 2's module mapping (`ch02_siglip_vs_gemma4_comparison/module_mapping.md`, line 21) lists `gemma4_vision_pooler.py` as a **New** module requiring 2-3 days of effort. Chapter 6's `modification_required_modules.md` (line 163) also references it as a separate module ("complex enough to warrant its own module (`gemma4_vision_pooler.py`)"). However, the pooler is entirely absent from the Chapter 6 scorecard table, the aggregate effort summary, and the dependency order. This means the total module count should be 12 (not 11), the "New implementation" category should be 4 modules (not 3), and the aggregate effort of 10-17 days underestimates by 2-3 days.

## Issue 2: Inconsistent classification of `gemma_image_block.py` between Chapter 2 and Chapter 6

**Location:** `index.md` line 42 and `direct_reuse_modules.md` lines 106-159

Chapter 6 classifies the encoder block as **Direct reuse** at 1 day effort. Chapter 2's module mapping (`module_mapping.md`, line 13) classifies the same module as **Modify** at 1-2 days. Given that Chapter 6's own detailed analysis lists four required changes (swap LayerNorm to RMSNorm, add two post-norms, change the residual pattern, and thread RoPE cos/sin arguments), the "Modify" classification from Chapter 2 is more accurate. At minimum, the two chapters should agree.

## Issue 3: Inconsistent classification of `model_config.py` and `load_checkpoints.py` between Chapter 2 and Chapter 6

**Location:** `index.md` lines 43-44

Chapter 6 classifies both `model_config.py` and `load_checkpoints.py` as **Direct reuse**. Chapter 2's module mapping (`module_mapping.md`, lines 17-18) classifies both as **Modify** at 1 day and 1-2 days respectively. The Chapter 6 detail pages themselves describe non-trivial work: `model_config.py` requires adding seven new parameters and potentially new memory configs, while `load_checkpoints.py` requires rewriting the entire weight key mapping dictionary with at least ten changed key patterns. These are better described as "Modify" to be consistent with Chapter 2.

## Issue 4: Dependency order omits `gemma4_vision_pooler.py`

**Location:** `index.md` dependency order (lines 67-77)

Because the pooler module is missing from the scorecard (Issue 1), it is also absent from the dependency ordering. The pooler depends on position IDs (for grid cell assignment) and should appear between step 7 and step 8, since the multimodal embedder calls the pooler. Without it, the dependency chain from encoder output to language model input is incomplete.

## Pass 2

All four Pass 1 issues have been addressed:
- The pooler is now in the scorecard (index.md line 51), aggregate summary (4 new modules), and dependency order (step 8).
- The encoder block is reclassified as **Modify** in both the scorecard and in `modification_required_modules.md`, with a reclassification note in `direct_reuse_modules.md` (line 5).
- `model_config.py` and `load_checkpoints.py` are reclassified as **Modify** in the scorecard and detailed in `modification_required_modules.md`.
- The dependency order now includes the pooler at step 8, before the multimodal embedder at step 9.

### Issue 1 (Pass 2): Inconsistent "share of total codebase" for new implementation modules

**Location:** `new_implementation_modules.md` line 3 vs. `index.md` line 59

`new_implementation_modules.md` states the four new modules account for "approximately 25% of the codebase." The authoritative scorecard in `index.md` line 59 states "~35%." These should agree. Given that 4 out of 12 modules are new and they carry 6-10 of the 14-23 total effort days (roughly 40%), ~35% is the more defensible figure. Update line 3 of `new_implementation_modules.md` to say "approximately 35%."

### Issue 2 (Pass 2): Dependency order has pooler before its stated dependency

**Location:** `index.md` lines 76-78

Step 8 (`gemma4_vision_pooler.py`) states "depends on: position IDs from preprocessor," but the preprocessor (`gemma4_variable_resolution.py`) does not appear until step 10. If the pooler genuinely depends on the preprocessor, the preprocessor should precede it. Alternatively, if the intent is that position IDs can be trivially hand-constructed for unit testing (which is true), the dependency annotation on step 8 should be revised to say "depends on: position IDs (trivially constructed or from preprocessor)" to avoid implying a strict build-order dependency on step 10.

No other factual issues found. With these two minor fixes applied, the chapter is ready for approval.
