# Audit Results: synchronize_device Calls in tt-symbiote Forward-Path Code

This file catalogues known and expected `ttnn.synchronize_device` calls in the tt-symbiote forward path, together with their trace-blocking status and recommended remedy type.

> **Note:** The tt-symbiote source repository was not accessible at a local path during the writing of this guide. The results below are derived from the plan specification, domain context, and cross-references in related guides. `# TODO: verify` markers appear wherever source confirmation was not possible. The implementing engineer must run the search commands in `audit_methodology.md` to obtain exact line numbers and verify the completeness of this list.

## Section 1: Confirmed Forward-Path Calls

The following call sites are known to exist based on the scope and subject matter of this guide. Both are the primary targets of the remediation described in Chapters 1–3.

---

### Call 1 — `TTNNQwen3FullAttention._maybe_all_gather`

- **File:** `models/tt_symbiote/nn/attention/qwen3_full_attention.py` (TODO: verify)
- **Method:** `_maybe_all_gather`
- **Approximate line:** TODO
- **Trigger condition:** `if self.num_devices > 1` or equivalent (TODO: verify whether the call is unconditional or guarded by a multi-device check)
- **Preceding op:** `ttnn.all_gather` (synchronous variant) — confirmed by domain context. `ttnn.experimental.all_gather_async` is not the current op at this call site.
- **`@trace_enabled` status:** Yes — `TTNNQwen3FullAttention` is in the trace-enabled attention stack (TODO: verify decorator on the class definition)
- **Trace-blocking:** Yes
- **Remedy:** The preceding op is confirmed synchronous `ttnn.all_gather`, so Type B1 does not apply. Two cases remain: (1) synchronous `ttnn.all_gather` and intent is synchronous dispatch → Type A (delete `synchronize_device` only); (2) synchronous `ttnn.all_gather` but intent is async dispatch → Type B2 (first replace `ttnn.all_gather` with `ttnn.experimental.all_gather_async`, then delete `synchronize_device` and add TT_CCL cycling semaphore management). The choice between Type A and Type B2 depends on dispatch intent — this remains a source-access TODO. See Chapter 3 and `audit_methodology.md` Section 3 Question 4 for the full verdict and exact code changes.
- **Notes:** Primary subject of this guide. Fix this call first.

---

### Call 2 — `TTNNQwen3LinearAttention._maybe_all_gather`

- **File:** `models/tt_symbiote/nn/attention/qwen3_linear_attention.py` (TODO: verify)
- **Method:** `_maybe_all_gather`
- **Approximate line:** TODO
- **Trigger condition:** Same multi-device conditional as the full-attention variant (TODO: verify)
- **Preceding op:** `ttnn.all_gather` (synchronous variant) — same as Call 1, confirmed by domain context (TODO: verify whether the method is shared via a base class or duplicated — see the note below).
- **`@trace_enabled` status:** Yes — `TTNNQwen3LinearAttention` is also in the trace-enabled stack (TODO: verify decorator)
- **Trace-blocking:** Yes
- **Remedy:** Identical to Call 1 (same preceding op, same dispatch-intent uncertainty). See Call 1 remedy above and `audit_methodology.md` Section 3, Question 4. Choice between Type A and Type B2 depends on dispatch intent — same open TODO.
- **Notes:** If `_maybe_all_gather` is a shared base-class method, fixing Call 1 also fixes Call 2 and no separate change is needed here. If the method is duplicated across the two classes, both implementations must be updated independently. Check whether a common base class (e.g., `TTNNAttentionBase`) owns the method before deciding how many edits to make.

---

> **Key Finding:** Based on the available domain context, at least two trace-blocking `synchronize_device` calls exist in the forward path of the hybrid attention stack: one in `TTNNQwen3FullAttention._maybe_all_gather` and one (or the same shared method) in `TTNNQwen3LinearAttention._maybe_all_gather`. Both must be removed before full-stack Metal Trace capture is possible for the hybrid decoder. Additional calls may exist in other modules — the engineer must run the audit search commands in `audit_methodology.md` to confirm the complete list.

## Section 2: Expected Non-Forward-Path Calls (Not Trace-Blocking)

The following call sites are expected (based on domain knowledge) to be in non-forward-path code and are therefore **not** trace-blocking. They are listed here so the engineer can quickly dismiss them when they appear in grep output, rather than spending time tracing their call chains.

- **Calls inside `move_weights_to_device_impl` or weight-loading routines:** These run at initialization, before any trace capture. Even if the call is present, it does not affect the trace bracket.
- **Calls inside `TracedRun._capture_trace` only when they appear before `begin_trace_capture` (the warm-up compile run):** The warm-up run executes the full forward pass to compile ops before the trace bracket opens. A `synchronize_device` in this position does not block trace capture — it only adds latency to warm-up. However, any call that appears after `begin_trace_capture` in the same method — i.e., inside the actual capture bracket — is trace-blocking and must not be dismissed. Do not dismiss a grep hit in `_capture_trace` without first confirming which side of `begin_trace_capture` it falls on.
- **Calls inside test harnesses, profiling scripts, or `__main__` blocks:** These are outside the module forward path and will be excluded by the `grep -v "test" | grep -v "benchmark" | grep -v "profil"` filter in the search command.

## Section 3: Unknown Call Sites (Requires Source Access)

The following module types may contain additional forward-path `synchronize_device` calls that could not be confirmed without source access. The engineer must verify each of these when running the audit.

- **`TTNNQwen3MoELayer` or other MoE routing modules:** If these modules perform an all_gather for expert-dispatch routing and include a `synchronize_device` call after the gather, that call would be trace-blocking for any decoder layer that uses mixture-of-experts attention. (TODO: verify whether MoE modules are `@trace_enabled` and whether they contain any synchronize calls)
- **Any shared base class (`TTNNAttentionBase` or equivalent) that implements `_maybe_all_gather` and is inherited by multiple module types:** If the method lives in a base class, a single fix covers all subclasses. If it is duplicated, each copy must be updated. (TODO: confirm class hierarchy for `TTNNQwen3FullAttention` and `TTNNQwen3LinearAttention`)
- **`TTNNRotaryPositionEmbedding.forward`:** Unlikely to contain a `synchronize_device` call, since rotary embedding is a local compute op with no CCL dependency. However, it should appear in the grep output (or its absence confirmed) before it is ruled out. (TODO: verify)

> **TODO:** Run the audit commands in `audit_methodology.md` against the actual tt-symbiote source to fill in exact line numbers, discover additional call sites, and confirm the `@trace_enabled` status of each enclosing module. Update the summary table in `index.md` with the complete results before proceeding to Chapter 5.
