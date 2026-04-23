# Symbiote-Wide Audit: Other synchronize_device Calls That Block Trace

Removing `ttnn.synchronize_device` from `_maybe_all_gather` in `TTNNQwen3FullAttention` and `TTNNQwen3LinearAttention` is a necessary step, but it may not be sufficient to enable full-stack Metal Trace capture for the hybrid decoder. Other forward-path code in tt-symbiote may contain additional `ttnn.synchronize_device` calls that would break trace capture the moment the trace bracket is extended to cover them. This chapter provides the methodology for a full codebase audit, presents the known call sites derived from domain context, and establishes the scope of work required before a complete end-to-end trace is achievable.

## Scope

This chapter surveys only `synchronize_device` calls inside **forward-path code** — that is, methods reachable from `Module.forward` or `TracedRun.__call__` during normal inference. Those calls are left out of this analysis. See [`audit_methodology.md` Section 2](./audit_methodology.md) for the full classification of non-forward-path call sites.

## Summary Table

Run the search commands in [`audit_methodology.md`](./audit_methodology.md) to fill in TODO fields.

| Module | File path | Method | Approx. line | Probable purpose | `@trace_enabled`? | Trace-blocking? |
|---|---|---|---|---|---|---|
| `TTNNQwen3FullAttention` | `models/tt_symbiote/nn/attention/qwen3_full_attention.py` | `_maybe_all_gather` | TODO | ordering / debugging artifact | Yes (target of trace) | Yes — primary subject of this guide |
| `TTNNQwen3LinearAttention` | `models/tt_symbiote/nn/attention/qwen3_linear_attention.py` | `_maybe_all_gather` | TODO | same shared method or copy | Yes (target of trace) | Yes |
| *(other modules — fill in after running audit)* | TODO | TODO | TODO | TODO | TODO | TODO |

`TTNNQwen3FullAttention` and `TTNNQwen3LinearAttention` are the primary subjects of this guide. The audit described in the following files may reveal additional blocking calls in other modules that must be addressed as the trace-enablement project progresses. Any such calls should be appended to the table above once confirmed.

## What's Next

- `audit_methodology.md` — Search commands and classification procedure for identifying every forward-path `synchronize_device` call in tt-symbiote
- `audit_results.md` — Known and expected call sites, with `# TODO: verify` annotations for all source locations that require confirmation against the actual repository
- `prioritization.md` — (Chapter 5, forthcoming) Ordering the fixes: which calls must be removed first, and which can be deferred
