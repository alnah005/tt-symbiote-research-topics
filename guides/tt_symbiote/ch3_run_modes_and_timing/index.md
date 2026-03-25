# Chapter 3: Run Modes and the Dispatch Manager — choosing an execution strategy and understanding timing data.

This chapter explains how TT Symbiote selects an execution backend at runtime, what each of the eight run modes does mechanically, and how the `DispatchManager` records and surfaces per-operation timing data.

## Contents

| File | What it covers |
|------|---------------|
| [run_modes.md](run_modes.md) | All eight run modes (`NORMAL`, `LIGHTWEIGHT`, `NORMAL_WITH_FALLBACK`, `SEL`, `DPL`, `DPL_NO_ERROR_PROP`, `CPU`, `TRACED`), the `@trace_enabled` / `@trace_disabled` decorators, the `no_dispatch()` context manager, and the suite of helper transform functions. |
| [dispatch_manager.md](dispatch_manager.md) | `DispatchManager` internals: the module-name stack, `dispatch_to_ttnn_wrapper`, `dispatch_to_torch_wrapper`, timing recording, and how to read the CSV output to find bottleneck layers. |

## Quick-reference: when to use which mode

For `NORMAL`, `LIGHTWEIGHT`, `CPU`, and `TRACED` the mode name is self-describing; see [run_modes.md](run_modes.md) for full guidance. `NORMAL_WITH_FALLBACK` is like `NORMAL` but wraps every TTNN op dispatch in a `try/except` and falls back to PyTorch on error — use it when some ops may not yet be supported in TTNN and silent per-op fallback is acceptable. The three modes below have non-obvious distinctions:

| Goal | Recommended mode |
|------|-----------------|
| Debugging a single layer — confirm TTNN output matches PyTorch; result carries TTNN data | `SEL` |
| Full parallel TTNN + PyTorch run; TTNN errors can propagate forward through the graph | `DPL` |
| Full parallel run where each TTNN op receives fresh PyTorch-derived tensors, preventing error accumulation | `DPL_NO_ERROR_PROP` |

## Environment variables at a glance

| Variable | Effect |
|----------|--------|
| `TT_SYMBIOTE_RUN_MODE` | Selects the run mode by name; overrides any `set_run_mode()` call. |
| `TT_SYMBIOTE_SIGNPOST_MODE` | When set, emits Tracy signpost markers at each module invocation. |
| `TT_SYMBIOTE_DISPATCHER` | Selects the active dispatcher (e.g., `CPU` is required for `LIGHTWEIGHT` and `TRACED`). |

## Navigation

- [run_modes.md](run_modes.md)
- [dispatch_manager.md](dispatch_manager.md)
