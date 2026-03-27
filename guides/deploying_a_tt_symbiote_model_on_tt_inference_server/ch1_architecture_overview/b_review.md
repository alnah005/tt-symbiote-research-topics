## Pass 1

**File:** `serving_stack.md`, line 140
**Issue:** The T3K hardware description is factually wrong. The document states "Setting `MESH_DEVICE=T3K` gives a `(1, 8)` mesh across eight Wormhole N300 devices." A T3K system has eight N300 *modules*, but each N300 module contains two Wormhole ASIC chips — the physical chip count is 16, not 8. Calling the N300 a "device" conflates the module with a chip. A reader sizing KV cache memory or reasoning about per-chip sharding from this description will compute incorrect per-device capacities.
**Fix:** Change to: "Setting `MESH_DEVICE=T3K` gives a `(1, 8)` mesh; note that T3K comprises eight N300 modules, each containing two Wormhole chips, for 16 physical chips total. The `(1, 8)` mesh shape represents eight logical mesh positions across these chips as configured by the driver." Align the mesh-shape semantics with the actual `ttnn.open_mesh_device` behaviour for T3K.

---

**File:** `request_lifecycle.md` / `index.md` — `TTModelRunner` undefined
**Issue:** `request_lifecycle.md` (line 23–24) introduces `TTModelRunner` (`tt_model_runner.py`) as the component that populates `TTModelInput` and dispatches `prefill_forward`/`decode_forward`. Neither `index.md`'s six-subsystem table nor `serving_stack.md` mentions `TTModelRunner` at all. A reader finishing `serving_stack.md` has no definition of this component before encountering it in the lifecycle diagram. This is a critical coherence gap: the reader cannot distinguish `TTWorker` from `TTModelRunner` or understand why both exist.
**Fix:** Add `TTModelRunner` as a seventh row in the subsystem table in `index.md` (e.g., "Translates the vLLM scheduler output into a `TTModelInput` and dispatches `prefill_forward()` or `decode_forward()`"), and add a short section to `serving_stack.md` explaining its role and relationship to `TTWorker`.

---

**File:** `request_lifecycle.md`, line 35
**Issue:** The document states "the default block size is **64 tokens**" as if it is a fixed architectural constant. `serving_stack.md` (line 44) and `index.md` (line 21) both show that `--block-size` is a `vllm_args` entry inside `ModelSpec` — it is a per-model, per-device configuration value, not a fixed default. A reader implementing a new model for a different device will not know this value is configurable via `ModelSpec` and may hard-code 64.
**Fix:** Restate as: "The block size is set per model via `ModelSpec.vllm_args` (e.g., `--block-size 64`). The value 64 is typical for Tenstorrent hardware because tile-based compute prefers 64-token-aligned access, but it must be explicitly configured — it is not a compile-time constant." Add a forward reference to Chapter 2 for how to set it.

---

**File:** `serving_stack.md`, line 57
**Issue:** The document says the `VLLM_USE_V1=1` environment variable must be "set in the environment before the vLLM process starts." However, the same document (lines 32–46) explains that the workflow orchestrator launches vLLM via `runpy.run_module(...)` *within the same Python process*. If `VLLM_USE_V1` is read by vLLM only at import time (which it is — it gates the engine class selection at module initialisation), and the module has already been partially imported before `runpy` is called, then setting the env var inside the orchestrator script may be too late. Conversely, if the process is fresh, the statement "before the vLLM process starts" is redundant and misleading given in-process launch. The text does not resolve this ambiguity, and a reader could set the variable too late and silently run the wrong engine.
**Fix:** Clarify explicitly: "`VLLM_USE_V1=1` must be present in `os.environ` before any `vllm` module is imported. Because `tt-inference-server` launches vLLM in-process via `runpy.run_module`, the safest approach is to set this variable in the shell environment (or in the Docker `ENV` directive) before the orchestrator process starts, not inside the orchestrator script itself."

---

## Pass 2

**File:** `serving_stack.md`, line 85 — `n > 1` enforcement attributed to `TTSamplingParams`, but `TTSamplingParams` has no `n` field
**Issue:** The capability constraints table states "`TTSamplingParams` enforces `n=1`". The `TTSamplingParams` dataclass defined in `request_lifecycle.md` (lines 60–65) has three fields — `temperature`, `top_k`, `top_p` — and no `n` field. A reader implementing a custom model class that inspects `TTSamplingParams` for multi-completion behaviour will find no `n` field and will not know where the `n=1` constraint is actually checked. Worse, they may conclude `n` is unconstrained at the model level and add their own handling.
**Fix:** Either add `n: int` to the `TTSamplingParams` dataclass in `request_lifecycle.md` if it truly carries that field, or correct the table entry to name the actual enforcement point (e.g., "rejected at request validation in `TTPlatform`'s request validator, before `TTSamplingParams` is constructed").

---

**File:** `request_lifecycle.md`, line 91 vs. line 107 — contradictory placement of trace capture
**Issue:** Line 91 states traces are captured "before the first decode step." The code comment at line 107 says "done once at model initialisation." These are different points in the lifecycle. Model initialisation happens inside `initialize_vllm_model()` at startup, before any request arrives. "Before the first decode step" implies it happens lazily during the first request's warm-up, which is an entirely different implementation strategy. A model author who follows the prose will implement trace capture in the wrong place (inside a warmup call during the first request rather than inside `initialize_vllm_model()`), causing either a missing trace error on the first actual decode or a re-capture penalty on every new batch.
**Fix:** Resolve the contradiction. If capture is at initialisation (as the code comment states), change line 91 to "before inference begins, during model initialisation." If capture is deferred to first-decode warm-up, update the code comment accordingly.

---

**File:** `request_lifecycle.md`, line 123 — broken forward navigation link
**Issue:** The "Next" footer links to `../ch2_model_spec/index.md`. This file is referenced as a clickable navigation link that a reader following the guide sequentially will click. If Chapter 2 does not yet exist at that path, the link is broken and blocks navigation. Per the review scope, non-clickable file references are flaggable structural gaps.
**Fix:** Verify that `ch2_model_spec/index.md` exists at the relative path `../ch2_model_spec/index.md`. If it does not yet exist, replace the footer with a placeholder such as "Chapter 2 (coming soon)" or a link to the guide root, so the chapter is not left with a dead navigation link.

---

**Pass 1 correction — item 2 (`TTModelRunner` missing from `index.md`)**
**Issue:** Pass 1 flagged `TTModelRunner` as absent from the subsystem table in `index.md`. Reviewing the current file, `TTModelRunner` is present as the seventh row of the subsystem table (line 26). Pass 1's flag is factually incorrect with respect to the current document state. No action needed on this item.

---

## Pass 3

**File:** `request_lifecycle.md`, line 39 vs. line 50 — `block_table` / `block_tables` field name inconsistency
**Issue:** The prose at line 39 refers to the field as `block_table` (singular): "The model implementation receives the `block_table` as part of `TTModelInput`." The `TTModelInput` field table at line 50 names the field `block_tables` (plural). A model author implementing `prefill_forward()` or `decode_forward()` will write code that accesses `tt_model_input.block_table` or `tt_model_input.block_tables` — whichever name appears in these docs — and will get an `AttributeError` at runtime if they pick the wrong one. This is a direct implementation error.
**Fix:** Standardise on one spelling throughout both documents. Match whichever name the actual `TTModelInput` dataclass uses in code.

---

**File:** `request_lifecycle.md` / `index.md` — `TTWorker` never defined
**Issue:** The end-to-end flow diagram in `request_lifecycle.md` (lines 20–21) shows `TTWorker (tt_worker.py)` as a distinct box responsible for "build TTModelInput for this step." `TTWorker` is not listed anywhere in `index.md`'s seven-subsystem table, and `serving_stack.md` mentions it only incidentally as the container that holds `TTModelRunner`. A reader following the guide sequentially has no definition of what `TTWorker` is, what it owns, or how it differs from `TTModelRunner`, before the lifecycle diagram requires them to understand both. The split matters: `TTWorker` owns the scheduling-to-input translation step, `TTModelRunner` owns the dispatch-to-hardware step. Conflating them or leaving one undefined causes an incorrect mental model of where to place custom pre-processing code.
**Fix:** Add `TTWorker` as an eighth row in the subsystem table in `index.md`, and add a sentence to `serving_stack.md` clarifying the division of responsibility between `TTWorker` (builds `TTModelInput`) and `TTModelRunner` (dispatches to model).

---

**Pass 2 correction — item 2 (contradictory trace capture placement)**
**Issue:** Pass 2 claimed line 91 says "before the first decode step," contradicting the code comment at line 107. Reading `request_lifecycle.md` line 91 directly: it says "both are captured as TTNN traces during `initialize_vllm_model()` at model startup." The phrase "before the first decode step" does not appear in the document. The prose and the code comment are consistent. Pass 2's flag on this item is incorrect with respect to the current document text. No action needed.

---

**Pass 2 correction — item 1 (`n > 1` attributed to `TTSamplingParams`)**
**Issue:** Pass 2 claimed the capability constraints table attributes `n=1` enforcement to `TTSamplingParams`. Reading `serving_stack.md` line 85 directly: the table entry credits `TTPlatform.check_and_update_config()`, not `TTSamplingParams`. Pass 2's attribution is a misread. No action needed on this item.

---

## Pass 4

**1. `TTModelInput` builder — direct contradiction across three files**
`index.md` (line 27) says `TTModelRunner` "builds `TTModelInput`." `serving_stack.md` (line 97) says `TTWorker` "builds the `TTModelInput` for each step." The `request_lifecycle.md` flow diagram (line 21) also places "build TTModelInput" under `TTWorker`. However, `request_lifecycle.md` (line 43) says "`TTModelInput` is the data-transfer object that `TTModelRunner` populates before each forward call." A model author deciding where to insert custom per-step pre-processing will get this wrong depending on which sentence they read last. This is a critical implementation error risk.
**Fix:** Establish one canonical owner. If `TTWorker` allocates and populates the struct (as the majority of references suggest), correct `index.md` line 27 and `request_lifecycle.md` line 43 to say `TTWorker`, and clarify that `TTModelRunner` receives the already-built `TTModelInput` and dispatches it.

---

**2. Mesh device opener — contradiction between `index.md` and `serving_stack.md`**
`index.md` (line 26) assigns `open_mesh_device()` to `TTWorker`. `serving_stack.md` (line 108) assigns it to `TTModelLoader` step 2: "Open the mesh device by calling the device-setup function in `tt_worker.py`." These cannot both be true as the primary opener; a reader reasoning about initialization order or trying to inject device configuration will target the wrong class.
**Fix:** Identify which component calls `open_mesh_device()` first and update the other reference. If `TTModelLoader` delegates to `tt_worker.py`'s helper but the call site is in `TTWorker`, make that indirection explicit rather than attributing device-open to `TTModelLoader` directly.

---

**3. Broken forward navigation link in `request_lifecycle.md`**
The footer (line 123) links to `../ch2_model_spec/index.md`. The directory `ch2_model_spec/` does not exist in the repository. A reader following the guide sequentially clicks a dead link. This blocks navigation out of Chapter 1.
**Fix:** Replace with a placeholder ("Chapter 2 — coming soon") or a link to the guide root until Chapter 2 is published.

---

**Pass 3 correction — item 2 (`TTWorker` missing from `index.md`)**
Pass 3 claimed `TTWorker` is absent from the seven-subsystem table in `index.md`. Reading `index.md` line 26 directly: `TTWorker` is present as the seventh row. Pass 3's flag is a misread. No action needed.

---

**Pass 3 correction — item 1 (`block_table` vs `block_tables` spelling inconsistency)**
Pass 3 claimed line 39 of `request_lifecycle.md` uses `block_table` (singular). Reading the file directly: line 39 uses `block_tables` (plural), consistent with the field table at line 50. Pass 3's flag is a misread. No action needed.

## Pass 5

**Pass 4 correction — items 1 and 2 (TTModelInput builder and open_mesh_device() contradictions)**
Pass 4 flagged both issues as open contradictions. Reading the current files: `index.md` line 27 now says `TTModelRunner` "receives the `TTModelInput` built by `TTWorker`" — consistent with `serving_stack.md` line 97 and `request_lifecycle.md` line 43. For `open_mesh_device()`, `index.md` line 26 and `serving_stack.md` line 108 both assign it to `TTWorker`. Both items are resolved. No action needed.

---

**1. `initialize_vllm_model()` caller — contradiction between `index.md` and `serving_stack.md`**
`index.md` line 26 states that `TTWorker` "calls `initialize_vllm_model()` on the resolved model class" directly. `serving_stack.md` line 97 states that `TTWorker` "delegates to `TTModelLoader` to call `initialize_vllm_model()`", and `serving_stack.md` lines 109–119 list calling `initialize_vllm_model()` as step 3 of `TTModelLoader`'s own responsibilities. A model author debugging initialization failures or injecting custom startup logic will find contradictory answers for which component is the actual call site of `initialize_vllm_model()`. This directly affects where they would add error handling or logging.
**Fix:** Reconcile to one description. If `TTModelLoader` is the component that calls `initialize_vllm_model()` (as `serving_stack.md` specifies in detail), correct `index.md` line 26 to say `TTWorker` "delegates model initialization to `TTModelLoader`, which calls `initialize_vllm_model()`."

---

**2. `TTSamplingParams` typed as scalar but described as "per-request" in a batched context**
`TTModelInput.sampling_params` (line 53, `request_lifecycle.md`) is typed as a single `TTSamplingParams` object. The field description says "per-request sampling configuration." The struct has `batch_size: int`, meaning multiple sequences with potentially different temperatures, top-k, and top-p values can be active simultaneously. A single scalar `TTSamplingParams` cannot represent different sampling parameters for different sequences in the same batch. A model author implementing batched decode will apply one temperature to all sequences rather than per-sequence parameters, producing incorrect sampling for any batch where requests differ in temperature or top-p.
**Fix:** Either clarify that `sampling_params` is applied uniformly to all sequences in the batch (i.e., sampling parameters are homogeneous per scheduling step), or change the type to `List[TTSamplingParams]` with length `batch_size` and update the description accordingly.

---

**3. Broken forward navigation link in `request_lifecycle.md` (re-confirmed open)**
The footer at line 123 of `request_lifecycle.md` links to `../ch2_model_spec/index.md`. The directory `ch2_model_spec/` does not exist. This was first flagged in Pass 2 and confirmed open in Pass 4. It remains unresolved. The link is dead and blocks sequential navigation out of Chapter 1.
**Fix:** Replace with a placeholder ("Chapter 2 — coming soon") or a link to the guide root until `ch2_model_spec/index.md` is published.

## Pass 6

**Pass 5 correction — items 1 and 2**
Pass 5 item 1 flagged a contradiction over which component calls `initialize_vllm_model()`. Reading `index.md` line 26 in the current file: it explicitly states "`TTModelLoader` (not `TTWorker` directly) calls `initialize_vllm_model()`" — consistent with `serving_stack.md`. Resolved; no action needed.
Pass 5 item 2 flagged `TTModelInput.sampling_params` as a scalar `TTSamplingParams`. Reading `request_lifecycle.md` line 53: the field is typed `list[TTSamplingParams]` and the description states "one entry per sequence in the batch." Pass 5's flag was a misread. Resolved; no action needed.

---

**1. Broken forward navigation link in `request_lifecycle.md` (re-confirmed open)**
The footer at line 123 of `request_lifecycle.md` links to `../ch2_model_spec/index.md`. The directory `ch2_model_spec/` does not exist in the repository — confirmed by directory listing. This issue has been open since Pass 2 and remains unresolved. A reader following the guide sequentially clicks a dead link and cannot proceed to Chapter 2.
**Fix:** Replace the footer with a placeholder ("Chapter 2 — coming soon") or a link to the guide root until `ch2_model_spec/index.md` is published.

## Pass 7

**Pass 6 correction — item 1 (broken forward navigation link)**
The footer at line 123 of `request_lifecycle.md` still links to `../ch2_model_spec/index.md`. The directory `ch2_model_spec/` was confirmed absent (only `ch1_architecture_overview/` and `plan.md` exist under the guide root). This issue remains open and unresolved entering Pass 7.

---

**1. Subsystem count in `index.md` heading is wrong — eight rows listed, heading says seven**
`index.md` line 16 reads "## The seven major subsystems." The table that follows (lines 18–27) contains eight rows: CLI entry point, `ModelSpec` registry, Workflow orchestrator, `tt-vllm-plugin`, `TTPlatform`, `TTModelLoader`, `TTWorker`, `TTModelRunner`. A reader using this heading to estimate coverage, write tests, or verify a deployment checklist will work from a count of seven and miss one subsystem entirely. This is a factual error with a concrete wrong numerical answer.
**Fix:** Change the heading to "## The eight major subsystems."

---

**2. Broken forward navigation link in `request_lifecycle.md` (re-confirmed open, fifth consecutive pass)**
The footer at line 123 links to `../ch2_model_spec/index.md`. The file does not exist. A reader following the guide sequentially is stranded at the end of Chapter 1 with a dead link. This is a critical structural gap under the review scope.
**Fix:** Replace with a placeholder ("Chapter 2 — coming soon") or a link to the guide root until `ch2_model_spec/index.md` is published.

## Pass 8

**Pass 7 correction — item 1 (subsystem count heading)**
`index.md` line 16 now reads "## The eight major subsystems." The table contains eight rows. Resolved; no action needed.

---

**1. Broken forward navigation link in `request_lifecycle.md` (re-confirmed open, sixth consecutive pass)**
The footer at line 123 of `request_lifecycle.md` links to `../ch2_model_spec/index.md`. The directory `ch2_model_spec/` does not exist in the repository. A reader following the guide sequentially clicks a dead link and cannot navigate to Chapter 2. This is a critical structural gap under the review scope.
**Fix:** Replace the footer with a placeholder ("Chapter 2 — coming soon") or a link to the guide root until `ch2_model_spec/index.md` is published.
