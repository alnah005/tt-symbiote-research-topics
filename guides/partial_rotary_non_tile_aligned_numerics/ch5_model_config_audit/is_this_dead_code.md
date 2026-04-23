# Is the Non-Tile-Aligned rotary_dim Path Dead Code?

This file evaluates the practical severity of the PCC ~0.71 bug documented in Chapters 1–3 by determining whether any currently supported tt-symbiote model exercises the non-tile-aligned `rotary_dim` code path. It then draws conclusions about fix urgency and strategy selection.

---

> **Key Finding:** For all currently supported Qwen3-family models in tt-symbiote, `rotary_dim % 64 == 0` holds. The non-tile-aligned `rotary_dim` zero-padding branch is not exercised by any production model. The tile-aligned production path (`rotary_dim=64, head_dim=128`) likely avoids the bug, but this has not been verified against `rope.py` — the conclusion that no production model is affected should be confirmed by running a PCC check before treating this as established. It remains a correctness hazard because any new model with `partial_rotary_factor` that yields a non-tile-aligned `rotary_dim` will trigger silent numerical corruption without warning.

---

## What "Latent Dead Code" Means Here

A latent bug is one where the defective code path exists and contains incorrect logic, but the inputs that would exercise it do not occur in the current production workload. The code runs without error because it is not reached; but if it were reached, it would produce incorrect output.

In this case:
- `TTNNRotaryPositionEmbedding.__init__` always precomputes cos/sin of shape `[1, 1, max_seq_len, rotary_dim]`.
- The zero-padding branch (`ttnn.pad` from `rotary_dim` to `nearest_32(rotary_dim)`) executes only when `rotary_dim % 32 != 0`.
- For all currently supported models, `rotary_dim=64` and `64 % 32 == 0`. The padding branch (`ttnn.pad` from `rotary_dim` to `nearest_32(rotary_dim)`) executes only when `rotary_dim % 32 != 0`, so for `rotary_dim=64` that branch is skipped.

However, this creates an unresolved question: if `cos.shape[-1]=64` is then passed to `ttnn.experimental.rotary_embedding` with `head_dim=128`, the same `TT_FATAL` ("Cos dims must match input dims") documented in Chapter 2 should also fire. The exact mechanism by which the `rotary_dim=64, head_dim=128` production path avoids this error has not been verified against `rope.py`. The tile-aligned case likely has additional padding logic (padding from 64 to `head_dim=128`) or takes a separate code route, but this has not been confirmed.

> **[UNVERIFIED]** The tile-aligned `rotary_dim=64, head_dim=128` code path in the current tt-symbiote implementation has not been traced against `rope.py`. The chapter's central conclusion — that the bug is latent dead code for all production models — depends on confirming that production-model execution does NOT encounter the `TT_FATAL` or the autoformat-silent-corruption path. The verification procedure: run `TTNNRotaryPositionEmbedding` with `rotary_dim=64, head_dim=128` in a non-traced forward pass and confirm PCC > 0.9999 against a PyTorch reference. Until this is confirmed, the "latent dead code" conclusion is an assertion to be verified, not an established fact. What is confirmed by the audit is that no currently supported model exercises `rotary_dim=48` or any other non-tile-aligned value.

---

## Implications of Latent Dead Code

### Implication 1: No currently shipped model is broken

No Qwen3-family model in tt-symbiote today produces PCC ~0.71 due to this bug. Users are not experiencing incorrect outputs from a non-tile-aligned `rotary_dim` configuration.

### Implication 2: The next model addition is the risk point

If a future model uses `partial_rotary_factor` such that `rotary_dim % 32 != 0` (for example, `partial_rotary_factor=0.375` → `rotary_dim=48`), it will enter the buggy zero-padding branch. Depending on the TTNN version and autoformat behavior, it will either:

- **Path A:** Crash immediately with `TT_FATAL: Cos dims must match input dims` — detectable but disruptive.
- **Path B:** Run to completion with PCC ~0.71 — potentially undetected if PCC checks are not part of the bring-up validation suite.

Path B is the more dangerous scenario. It produces plausible-looking but numerically wrong output, and the corruption is spread across the entire head rather than being a simple offset or scale error.

> **[SILENT FAILURE]** If a new model with non-tile-aligned `rotary_dim` is brought up without running PCC validation against a PyTorch reference, Path B (autoformat-extended cos/sin with PCC ~0.71) will produce wrong model outputs silently. The activation statistics will look reasonable because ~60% of elements are correct; only a direct numerical comparison against the reference will catch the regression.

### Implication 3: Strategy B is adequate as an immediate mitigation

Because no currently supported model is affected, the risk of the latent bug is low in the short term. Strategy B — enforcing `rotary_dim % 64 == 0` as a hard precondition in `TTNNRotaryPositionEmbedding.__init__` — is adequate as an immediate fix. It converts Path B (silent corruption) into an explicit error, preventing incorrect outputs from reaching users.

Strategy B does not fix the underlying limitation; it makes the failure loud rather than silent.

### Implication 4: Strategy C should be implemented before any non-tile-aligned model is brought up

Strategy C (identity-filled precomputed cos/sin table of shape `[max_seq_len, head_dim]`) eliminates the bug entirely for all `rotary_dim <= head_dim` configurations, tile-aligned or not. It is trace-compatible and adds no runtime overhead. See Chapter 4 ([`../ch4_implementation_strategies/strategy_c_precomputed_full_head_cos_sin.md`](../ch4_implementation_strategies/strategy_c_precomputed_full_head_cos_sin.md)) for the construction details.

The recommended timeline:
1. **Now (if desired):** Deploy Strategy B as a one-line guard in `__init__`. Zero risk, zero effort.
2. **Before the next model with non-tile-aligned rotary_dim:** Implement Strategy C, remove the Strategy B guard, run the verification checklist in Chapter 6.

---

## Why Fixing Latent Dead Code Is Still Warranted

Latent bugs in correctness-critical paths share a common failure mode: they are discovered at the worst possible time — when a new model is being brought up under deadline pressure, and the silent corruption is attributed to a data issue, a quantization problem, or an unrelated software change rather than the actual root cause.

The root cause analysis in Chapters 1–3 is already complete. The fix (Strategy C) is well-understood and straightforward to implement. The cost of fixing it now is low; the cost of diagnosing it under pressure when a new model is being brought up is high.

---

## What's Next

The next file traces the exact sequence of operations that would occur if the `rotary_dim=48` test case were run today, illustrating both failure paths concretely.

**Next:** [`the_rotary_dim_48_test_case.md`](./the_rotary_dim_48_test_case.md)
