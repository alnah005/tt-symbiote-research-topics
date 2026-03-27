## Agent A Change Log — B Review Pass 1
- Fixed T3K chip count (8 N300 modules = 16 chips)
- Added TTModelRunner definition to index.md and serving_stack.md
- Clarified block size 64 as TT hardware default, configurable via ModelSpec
- Added VLLM_USE_V1 import-time warning to serving_stack.md

## Agent A Change Log — B Review Pass 2
- Corrected n=1 enforcement attribution from TTSamplingParams to TTPlatform.check_and_update_config()
- Aligned trace capture timing: TTNN traces captured during initialize_vllm_model() at startup
- Verified request_lifecycle.md navigation footer format

## Agent A Change Log — B Review Pass 3
- Standardized block_table → block_tables throughout request_lifecycle.md
- Added TTWorker to subsystem table in index.md and to serving_stack.md

## Agent A Change Log — B Review Pass 4
- Standardized TTModelInput builder to TTWorker across all files
- Corrected mesh device opener: TTWorker opens mesh device, TTModelLoader receives it

## Agent A Change Log — B Review Pass 5
- Standardized initialize_vllm_model() caller to TTModelLoader across all files
- Clarified TTSamplingParams can vary per-sequence within a batch

# Compression Analysis: Ch1 Architecture Overview — Pass 1

## Summary
- Total files analyzed: 3
- Estimated current line count: ~195 lines (index.md ~36, serving_stack.md ~167, request_lifecycle.md ~124)
- Estimated post-compression line count: ~170 lines
- Estimated reduction: ~13%

## CRUCIAL Suggestions

### [index.md] ~lines 24–27
**Issue:** The `TTWorker`, `TTModelRunner`, `TTModelLoader`, and `TTPlatform` rows in the subsystem table contain detailed prose descriptions that are then repeated — often word-for-word — in the dedicated `## TTWorker`, `## TTModelRunner`, and `## TTModelLoader` sections of `serving_stack.md`. Notably, both `index.md` line 26 and `serving_stack.md` line 97 contain the identical parenthetical `"TTModelLoader (not TTWorker directly) calls initialize_vllm_model()"`. The index table's "Role" column re-narrates what `serving_stack.md` owns as its primary content.
**Suggestion:** Trim the "Role" column entries for TTWorker, TTModelRunner, TTModelLoader, and TTPlatform in `index.md` to one short phrase each (e.g., "Opens mesh device, manages KV cache pool, dispatches forward calls"). Remove the parenthetical `"(not TTWorker directly)"` clarification from `index.md` line 26 entirely — it belongs only in `serving_stack.md` where the distinction is explained in context.

### [serving_stack.md] ~line 7
**Issue:** "Each layer calls downward; no layer reaches upward." is stated in prose immediately before the ASCII stack diagram (lines 9–26) which visually encodes the same directional constraint. The sentence adds zero information beyond what the diagram already shows.
**Suggestion:** Delete the sentence "Each layer calls downward; no layer reaches upward." The diagram is self-documenting.

## MINOR Suggestions

### [request_lifecycle.md] ~lines 57–58
**Issue:** The prose before the `TTSamplingParams` dataclass says "sequences within the same batch may carry different values, so model authors must not assume uniform sampling parameters across the batch. Always iterate over the list rather than applying a single shared configuration" — this same point is already established by the table entry for `sampling_params` at line 53 which reads "sequences in the same batch may have different temperatures, top_k, and top_p values (see below)". The concept is stated twice in close proximity.
**Suggestion:** Remove the redundant sentence in the prose block ("Sequences within the same batch may carry different values, so model authors must not assume uniform sampling parameters across the batch.") and keep only the actionable instruction: "Always iterate over the list rather than applying a single shared configuration."

### [serving_stack.md] ~lines 30–37
**Issue:** The `sys.argv` injection is shown once in prose description (lines 37–38) and then immediately repeated as a code block (lines 41–47) that re-demonstrates the same single-line prepend. The prose sentence "Before this call, the workflow orchestrator reads the active `DeviceModelSpec` and prepends its `vllm_args` list to `sys.argv` so that vLLM's argument parser sees the device-specific flags (e.g., `--block-size`, `--max-model-len`, `--tensor-parallel-size`) as if they had been typed on the command line" restates what the code block already shows.
**Suggestion:** Drop the prose sentence (lines 37–38) and keep only the code block with its lead-in "A representative `sys.argv` injection looks like:". The examples in the inline comment (`["--block-size", "64", ...]`) make the mechanism self-evident without the preceding paragraph.

### [request_lifecycle.md] ~lines 79 and 87
**Issue:** Both the Prefill and Decode subsections contain a closing sentence characterising the performance sensitivity of the operation ("Because prefill is compute-bound and not latency-sensitive to the same degree as decode, it does not need to be captured in a TTNN trace." / "Decode is repeated hundreds or thousands of times per request and is extremely latency-sensitive."). The decode latency point is already implied by the TTNN trace section that follows (lines 89–119), which exists precisely because decode is latency-sensitive.
**Suggestion:** The prefill sentence (line 79) is load-bearing because it explains the trace decision; keep it. Remove the redundant closing sentence from the Decode subsection (line 87: "Decode is repeated hundreds or thousands of times...extremely latency-sensitive.") — the trace section immediately following makes this point through mechanism rather than assertion.

## Load-Bearing Evidence
- `index.md` line ~14: "Read `serving_stack.md` first to understand the static architecture — which components exist and how they are wired together. Then read `request_lifecycle.md` to understand the dynamic flow..." — load-bearing because this reading-order guidance is the primary purpose of the index and appears nowhere else.
- `serving_stack.md` line ~59: "do **not** set it via `os.environ["VLLM_USE_V1"] = "1"` inside a script after any `vllm` module has been imported. vLLM reads this variable at import time; if any `vllm` module is imported before the variable is set, the wrong engine is silently selected and no error is raised." — load-bearing operational warning not restated in any other file.
- `request_lifecycle.md` line ~79: "Because prefill is compute-bound and not latency-sensitive to the same degree as decode, it does not need to be captured in a TTNN trace." — load-bearing because it directly explains the architectural decision to omit trace capture for prefill, which would otherwise appear as an inconsistency.

## VERDICT
- Crucial updates: yes

## Agent A Change Log — C Compression Pass 1
- Trimmed index.md subsystem table Role column to one short phrase per row
- Deleted redundant "Each layer calls downward" sentence from serving_stack.md

## Agent A Change Log — B Review Pass 7
- Fixed subsystem count heading: "seven" → "eight"

# Compression Analysis: Ch1 Architecture Overview — Pass 2

## Summary
- Total files analyzed: 3
- Estimated current line count: ~325 lines (index.md ~36, serving_stack.md ~165, request_lifecycle.md ~124)
- Estimated post-compression line count: ~315 lines
- Estimated reduction: ~3%

## CRUCIAL Suggestions

None — Pass 1 CRUCIAL items resolved.

- Pass 1 item 1 (index.md subsystem table Role column verbosity): confirmed resolved — all eight Role cells are now short phrases.
- Pass 1 item 2 (serving_stack.md "Each layer calls downward; no layer reaches upward." sentence): confirmed deleted — line 7 now reads "The stack is composed of three layers stacked vertically." and the offending sentence is absent.

## MINOR Suggestions

### [request_lifecycle.md] ~lines 53–58
**Issue:** The `sampling_params` row in the `TTModelInput` table (line 53) already states "sequences in the same batch may have different temperatures, top_k, and top_p values (see below)". The prose block introducing `TTSamplingParams` (lines 57–58) then restates the same constraint: "Sequences within the same batch may carry different values, so model authors must not assume uniform sampling parameters across the batch." This is a direct duplicate flagged in Pass 1 MINOR that has not yet been addressed.
**Suggestion:** Remove the first sentence of that prose block ("Sequences within the same batch may carry different values, so model authors must not assume uniform sampling parameters across the batch.") and keep only the actionable instruction: "Always iterate over the list rather than applying a single shared configuration."

### [request_lifecycle.md] ~line 87
**Issue:** The Decode subsection ends with "Decode is repeated hundreds or thousands of times per request and is extremely latency-sensitive." The TTNN trace section that follows (lines 89–119) exists precisely because decode is latency-sensitive and explains the mechanism in full. The assertion on line 87 is made redundant by the mechanism immediately below it. Flagged in Pass 1 MINOR, not yet addressed.
**Suggestion:** Delete the sentence "Decode is repeated hundreds or thousands of times per request and is extremely latency-sensitive." The trace section's opening sentence already implies the latency constraint.

### [request_lifecycle.md] ~line 35
**Issue:** "For TT hardware, the TT hardware default block size is **64 tokens**" — the phrase "TT hardware" appears twice in five words ("For TT hardware, the TT hardware default block size"). This is a copy-edit redundancy.
**Suggestion:** Rewrite as "The TT hardware default block size is **64 tokens**" (drop the leading "For TT hardware,").

### [serving_stack.md] ~lines 37–47
**Issue:** The prose sentence at line 37 ("Before this call, the workflow orchestrator reads the active `DeviceModelSpec` and prepends its `vllm_args` list to `sys.argv` so that vLLM's argument parser sees the device-specific flags…as if they had been typed on the command line") fully describes the mechanism that the code block on lines 41–47 already shows. The mechanism is explained twice — once in prose, once in code. Flagged in Pass 1 MINOR, not yet addressed.
**Suggestion:** Drop the prose sentence at lines 37–38 and keep only the code block with its lead-in "A representative `sys.argv` injection looks like:". The inline comment in the code block (`# e.g. ["--block-size", "64", "--max-model-len", "131072"]`) makes the mechanism self-evident.

## Load-Bearing Evidence
- `index.md` line ~14: "Read `serving_stack.md` first to understand the static architecture…Then read `request_lifecycle.md` to understand the dynamic flow…" — load-bearing because it is the sole statement of reading order and appears in no other file.
- `serving_stack.md` line ~59: "do **not** set it via `os.environ[\"VLLM_USE_V1\"] = \"1\"` inside a script after any `vllm` module has been imported. vLLM reads this variable at import time; if any `vllm` module is imported before the variable is set, the wrong engine is silently selected and no error is raised." — load-bearing operational warning that appears only in this file.
- `request_lifecycle.md` line ~79: "Because prefill is compute-bound and not latency-sensitive to the same degree as decode, it does not need to be captured in a TTNN trace." — load-bearing because it directly explains the architectural decision to omit trace capture for prefill.

## VERDICT
- Crucial updates: no
