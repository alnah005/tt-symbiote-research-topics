# Agent B Review — Chapter 1: What Is TT Symbiote? — Pass 1

## Verdict

Two factual errors found. Both are in `source_layout.md` and describe run-mode input tensor types incorrectly. A reader following either description would implement the mode with the wrong tensor type flowing to the wrong backend.

---

## Issues

**1. `source_layout.md`, line 104 — SEL run-mode description inverts which backend receives TTNN tensors**

The table entry reads:

> "SEL — Segment Each Layer. PyTorch receives TTNN tensors as input and its output is compared to TTNN output using PCC."

This is backwards. In `SELRun.module_run` (`run_config.py` lines 665–685), PyTorch receives *copied torch tensors* (produced by `copy_to_torch`). TTNN receives the TTNN-layout versions of the original inputs (produced by `to_ttnn_wrap` + `set_device_wrap`). The outputs of both paths are then compared with PCC.

Fix: Replace the erroneous sentence with: "PyTorch receives copied torch tensors as input; TTNN receives the same inputs converted to TTNN tensors. Their outputs are compared with PCC."

---

**2. `source_layout.md`, line 106 — DPL_NO_ERROR_PROP description misstates what tensor type TTNN receives**

The table entry reads:

> "DPL_NO_ERROR_PROP — DPLRunNoErrorProp — DPL variant where TTNN receives PyTorch tensors as input, preventing TTNN numerical errors from propagating to subsequent layers."

This is wrong. In `DPLRunNoErrorProp.module_run` (`run_config.py` lines 773–791), TTNN receives fresh *TTNN tensor copies* of the original inputs, produced by `copy_to_ttnn`. PyTorch also receives fresh copied torch tensors. Neither side receives the other's tensor type. The "no error propagation" comes from both backends being given independent copies of the original inputs (rather than the accumulated TTNN output from previous layers), so TTNN numerical drift does not feed into subsequent layers.

Fix: Replace "TTNN receives PyTorch tensors as input" with: "Both PyTorch and TTNN receive independent fresh copies of the original inputs (PyTorch gets torch copies via `copy_to_torch`; TTNN gets TTNN copies via `copy_to_ttnn`), preventing accumulated TTNN numerical error from propagating to subsequent layers."

## Change Log — Pass 1 fixes applied

- Fixed SEL run-mode description: corrected which tensor type each backend receives.
- Fixed DPL_NO_ERROR_PROP description: both backends now described as receiving independent fresh copies via `copy_to_torch` / `copy_to_ttnn`.

---

# Agent B Review — Chapter 1: What Is TT Symbiote? — Pass 2

## Re-check of Pass 1 fixes

Both fixes are correctly applied in the current `source_layout.md`:
- SEL (line 104): Now reads "PyTorch receives copied torch tensors as input; TTNN receives the same inputs converted to TTNN tensors. Their outputs are compared with PCC." Matches `SELRun.module_run` in `run_config.py`.
- DPL_NO_ERROR_PROP (line 106): Now correctly describes independent `copy_to_torch` / `copy_to_ttnn` copies for each backend. Matches `DPLRunNoErrorProp.module_run` in `run_config.py`.

## New issues found

1. **`source_layout.md`, line 109 — `TRACED` run mode silently requires `TT_SYMBIOTE_DISPATCHER=CPU` but the guide does not say so**

   File: `source_layout.md`, line 109 (TRACED row in the run modes table).

   Error: The guide documents the CPU-dispatcher requirement only for `LIGHTWEIGHT` (line 108: "requires `TT_SYMBIOTE_DISPATCHER=CPU`"). `TracedRun` inherits from `LightweightRun` (`run_config.py` line 882), so `get_tensor_run_implementation()` (lines 1158–1163) will assert that the CPU dispatcher is active before allowing `TRACED` to run. A reader who tries to use `TRACED` with `TT_SYMBIOTE_DISPATCHER=DEFAULT` will hit a runtime assertion with no guidance from the guide.

   Fix: Append "requires `TT_SYMBIOTE_DISPATCHER=CPU`" to the `TRACED` row description, matching the note already present for `LIGHTWEIGHT`.

## Verdict

One new issue. Pass 1 fixes are confirmed correct.

## Change Log — Pass 2 fixes applied

- Added note to TRACED run mode: TT_SYMBIOTE_DISPATCHER=CPU is required (TracedRun inherits LightweightRun; get_tensor_run_implementation asserts CPU dispatcher for all LightweightRun subclasses).

---

# Agent B Review — Chapter 1: What Is TT Symbiote? — Pass 3

## Re-check of Pass 2 fixes

Pass 2 fix confirmed correctly applied in the current `source_layout.md`:
- TRACED (line 109): Now reads "Requires `TT_SYMBIOTE_DISPATCHER=CPU` (same as `LIGHTWEIGHT`): `TracedRun` inherits from `LightweightRun` (`run_config.py` line 882), and `get_tensor_run_implementation()` asserts the CPU dispatcher is active for all `LightweightRun` subclasses." This accurately reflects lines 1158–1163 of `run_config.py`.

## New issues found

**1. `source_layout.md`, line 105 — DPL description uses "independently" in a way that materially misrepresents its error-propagation behaviour**

The table entry reads:

> "DPL — DPLRun — Debug Per Layer. Both TTNN and PyTorch run independently for each layer; outputs are compared with PCC."

The word "independently" implies that neither backend's numerical error influences the other's inputs — which is exactly what `DPL_NO_ERROR_PROP` guarantees and `DPL` does not. In `DPLRun.module_run` (`run_config.py` lines 706–735), TTNN receives the accumulated `args` from prior layers (transformed via `wrap_to_torch_ttnn_tensor, to_ttnn_wrap, set_device_wrap`), not fresh copies. Only PyTorch gets `copy_to_torch` copies. Accumulated TTNN numerical error therefore continues to propagate forward through the graph under `DPL`, which is precisely the distinction the `DPL_NO_ERROR_PROP` mode was designed to remove. A reader studying these two modes to decide which to use for isolating a numerical accuracy problem would be misled into thinking `DPL` already prevents error propagation, and might not reach for `DPL_NO_ERROR_PROP` when they should.

Fix: Replace "Both TTNN and PyTorch run independently for each layer" with "Both TTNN and PyTorch run at each layer; outputs are compared with PCC. Unlike `DPL_NO_ERROR_PROP`, TTNN receives the accumulated prior-layer output as input, so numerical error continues to propagate through the graph."

---

**2. `source_layout.md`, line 142 — navigation footer links to a file that does not exist**

The footer reads:

> **Next:** [Chapter 2 — Core Abstractions](../ch2_core_abstractions/index.md)

The file `guides/tt_symbiote/ch2_core_abstractions/index.md` does not exist in the repository. A reader following the guide's own navigation will hit a broken link and have no path to the next chapter.

Fix: Either create `ch2_core_abstractions/index.md` or update the footer to point to a file that exists (or remove the footer until Chapter 2 is written).

---

## Verdict

Two issues found. Pass 1 and Pass 2 fixes are confirmed correct.

## Change Log — Pass 3 fixes applied

- Fixed DPL run mode description: clarified that TTNN receives accumulated prior-layer args (errors propagate). Directed readers to DPL_NO_ERROR_PROP for single-layer isolation.
- Note: Issue 2 (broken link to ch2_core_abstractions/index.md) is expected — Chapter 2 has not been written yet. The link path is correct per plan and will resolve once Ch2 is created.

---

# Agent B Review — Chapter 1: What Is TT Symbiote? — Pass 4

## Re-check of Pass 3 fixes

Both Pass 3 fixes are correctly applied in the current `source_layout.md`:

- DPL (line 105): Now reads "Runs both TTNN and PyTorch in parallel for every layer and compares outputs with PCC. TTNN receives the accumulated prior-layer outputs as inputs (numerical errors propagate across layers). Use `DPL_NO_ERROR_PROP` if you need to isolate a single layer's error without compounding from prior layers." This accurately reflects `DPLRun.module_run` in `run_config.py` lines 706–735, where TTNN receives `args` transformed from accumulated prior-layer state rather than fresh copies.

## Full independent pass — issues found

No feedback — chapter approved.

---

# Agent B Review — Chapter 1: What Is TT Symbiote? — Pass 5

## Re-check of Pass 4 verdict

Pass 4 concluded "No feedback — chapter approved." This pass is a fresh independent check of both guide files against the current source truth after the latest compression edits.

## Full independent pass — issues found

No feedback — chapter approved.
