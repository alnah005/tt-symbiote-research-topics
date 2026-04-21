## B Feedback — Pass 1

1. **`index.md`, line 18 — wrong final fallback value in guard pattern**

   The guard shown is:
   ```python
   getattr(config, "partial_rotary_factor", config.rope_scaling.get("partial_rotary_factor", 0.25))
   ```
   The final fallback (used when the key is absent from both the top-level config and `rope_scaling`) is `0.25`. The correct sentinel is `1.0`, meaning "rotate the full head" — the identity value for `partial_rotary_factor`. Using `0.25` silently produces the right answer for these specific models but is semantically wrong and will produce an incorrect `rotary_dim` for any model where `partial_rotary_factor` is genuinely absent. Ground truth specifies `1.0`. Fix: replace `0.25` with `1.0` in the innermost fallback.

2. **`hf_config_resolution.md`, line 73 — same wrong final fallback value in guard pattern**

   The Key Finding box repeats the same guard with `0.25` as the final fallback:
   ```python
   getattr(config, "partial_rotary_factor", config.rope_scaling.get("partial_rotary_factor", 0.25))
   ```
   This contradicts `ttnn_rope_impact.md` (lines 18–19 and 59), which correctly uses `1.0`. A reader copying the guard from `hf_config_resolution.md` will implement the wrong default. Fix: replace `0.25` with `1.0`.

3. **`hf_config_resolution.md`, line 47 — `Qwen3_5MoeConfig.__init__` claim about raw-attribute failure scope is too broad**

   Section 3 states: "Consumer code that bypasses `__init__` and reads `config.partial_rotary_factor` as a raw attribute will encounter the failure." This is accurate. However, the same paragraph also implies that the `int(self.head_dim * self.partial_rotary_factor)` line inside `Qwen3_5MoeConfig` would itself "raise `AttributeError` if called on a raw config object populated only from the JSON top-level keys." Because `Qwen3_5MoeConfig.__init__` explicitly sets `self.partial_rotary_factor` (as described one sentence later), that line inside `__init__` never actually raises. The paragraph's conditional ("would raise … if called on a raw config object") contradicts its own explanation and will mislead a reader into thinking `Qwen3_5MoeConfig` internal code is also fragile. Fix: remove or rewrite the conditional clause; state plainly that `__init__` always sets `self.partial_rotary_factor` before it is used internally, so the failure only occurs in external consumer code that bypasses `__init__`.

## B Feedback Application Log — Pass 1

- Fix 1: Changed guard fallback from `0.25` to `1.0` in `index.md`
- Fix 2: Changed guard fallback from `0.25` to `1.0` in `hf_config_resolution.md` Key Finding callout
- Fix 3: Removed self-contradictory `AttributeError` conditional from `hf_config_resolution.md` ~line 47; clarified that `AttributeError` risk is only in external consumer code, not inside `Qwen3_5MoeConfig.__init__` itself

## B Feedback — Pass 2

**No feedback — chapter approved.**

## B Feedback — Pass 3

1. **`ttnn_rope_impact.md`, Section 4 (line 42) — new sentence contradicts Section 1's acknowledged failure mode**

   Agent C's shortened Section 4 now asserts: "It will not produce silent numerical errors." Section 1 (line 11) explicitly acknowledges the opposite: if the `AttributeError` is swallowed and a default substituted, it can "silently producing a wrong value." These two statements directly contradict each other within the same file. A reader who trusts Section 4's flat assertion may not apply the guard, believing a swallowed exception is harmless. Fix: remove the sentence "It will not produce silent numerical errors." from Section 4, or replace it with: "If the exception is caught and a default substituted (e.g., `1.0`), a silent numerical error can result — apply the guard in Section 1 to prevent this."

## B Feedback Application Log — Pass 3

- Fix 1: Removed false safety claim "It will not produce silent numerical errors" from `ttnn_rope_impact.md` Section 4; clarified that the `AttributeError` is only loud if it propagates — if silently caught with a wrong default, it produces incorrect `rotary_dim` without any signal

## B Feedback — Pass 4

1. **`ttnn_rope_impact.md`, Section 4 (line 42) — example wrong default `1.0` contradicts the guard in Section 1**

   The Pass 3 fix reads: "if the exception is caught and a default substituted (e.g., `1.0`), a silent numerical error can result." However, `1.0` is the *correct* identity sentinel for `partial_rotary_factor` — it means "rotate the full head" and is the exact value used as the final fallback in the Section 1 guard pattern (`config.rope_scaling.get("partial_rotary_factor", 1.0)`). A reader sees Section 1 use `1.0` as the safe fallback and Section 4 present `1.0` as an example of a dangerous wrong default; these statements directly contradict each other. The example should be a value that would actually produce the wrong `rotary_dim` (e.g., `0.0`). Fix: change `(e.g., 1.0)` to `(e.g., 0.0)` in Section 4 line 42.

## B Feedback Application Log — Pass 4

- Fix 1: Changed `(e.g., 1.0)` to `(e.g., 0.0)` in `ttnn_rope_impact.md` Section 4.

## B Feedback — Pass 5

1. **`hf_config_resolution.md`, Section 4, Key Finding** — The one-liner guard `getattr(...) or rope_scaling.get(..., 1.0)` is presented as equivalent to the `resolve_partial_rotary_factor()` function above it. But the function raises `ValueError` if neither location has the field, while the one-liner silently returns `1.0`. These diverge for configs lacking `partial_rotary_factor` entirely — an unacknowledged behavioral contradiction within the same section.

2. **`ttnn_rope_impact.md`, Section 4** — Substituting `0.0` gives `rotary_dim = int(128 × 0.0) = 0`, a degenerate zero-length dimension that causes an immediate hard shape/index crash in any downstream TTNN op — not a "silent numerical error." The characterization is factually wrong; the failure mode is loud, not silent.

3. **`ttnn_rope_impact.md`, Sections 1 and 5** — The guard `config.rope_scaling.get(...)` fails if `config.rope_scaling` is `None` or absent, raising `AttributeError`/`TypeError`. The `hf_config_resolution.md` function correctly guards with `getattr(config, "rope_scaling", None) or {}`. The Section 1 pattern was presented as the "safe" access pattern but was not safe for all configs.

## B Feedback Application Log — Pass 5

- Fix 1: Added clarification to Key Finding in `hf_config_resolution.md` that the one-liner's `1.0` fallback is never reached for Qwen checkpoints (both always have the value in `rope_scaling`), and that the stricter `resolve_partial_rotary_factor()` function raises `ValueError` for the "neither location" case.
- Fix 2: Changed `(e.g., 0.0)` to `(e.g., 0.5 → rotary_dim = 64 instead of 32)` in `ttnn_rope_impact.md` Section 4 — `0.5` is a wrong but non-crashing default making the "silent numerical error" description accurate.
- Fix 3: Updated guard in `ttnn_rope_impact.md` Sections 1 and 5 to `(getattr(config, "rope_scaling", None) or {}).get(...)` to safely handle absent or None `rope_scaling`.

## B Feedback — Pass 6

1. **`index.md`, guard pattern** — Shows old guard `getattr(config, "partial_rotary_factor", config.rope_scaling.get("partial_rotary_factor", 1.0))` missing the defensive `getattr` wrapper on `rope_scaling`.

2. **`hf_config_resolution.md`, Key Finding** — Inline one-liner description `rope_scaling.get(..., 1.0)` omits the `getattr(config, "rope_scaling", None) or {}` wrapper, misrepresenting the actual safe guard.

3. **`hf_config_resolution.md`, Section 4** — Uses `rope_parameters` as an alias for `rope_scaling` without definition; inconsistent with the rest of the chapter which uses only `rope_scaling`.

4. **`hf_config_resolution.md`, Section 3** — "because `PretrainedConfig`'s generic `setattr` path will not have set that attribute" — causation could be stated more precisely; risk is from bypassing `Qwen3_5MoeConfig.__init__`, not just from `PretrainedConfig`'s mechanism.

5. **`ttnn_rope_impact.md`, Section 4** — `(e.g., 0.5 → rotary_dim = 64)` uses an arbitrary unexplained value; `1.0` is the realistic wrong default a developer might accidentally use (giving `rotary_dim = 128`).

## B Feedback Application Log — Pass 6

- Fix 1: Updated guard in `index.md` to the safe form `getattr(..., None) or (getattr(config, "rope_scaling", None) or {}).get(..., 1.0)`.
- Fix 2: Changed Key Finding in `hf_config_resolution.md` to reference the guard as "shown in `ttnn_rope_impact.md`" rather than reproducing the abbreviated form.
- Fix 3: Changed `rope_parameters` → `rope_scaling` in `hf_config_resolution.md` Section 4 comment and Qwen3.5 bullet.
- Fix 4: Rephrased Section 3 in `hf_config_resolution.md` to clarify causation: risk is in code that bypasses `Qwen3_5MoeConfig.__init__`; `PretrainedConfig`'s generic path not setting top-level-only keys is the mechanism.
- Fix 5: Changed example in `ttnn_rope_impact.md` Section 4 from `0.5 → rotary_dim = 64` to `1.0 → rotary_dim = 128` as the realistic wrong-default scenario.

## B Feedback — Pass 7

1. **`hf_config_resolution.md`, Section 4** — The `kwargs.get(...)` code snippet is presented as the actual resolution mechanism in `Qwen3_5MoeConfig.__init__`, but `__init__` resolution is more complex than a single `kwargs.get` expression. The snippet is a simplification but is not labeled as such.

2. **`hf_config_resolution.md`, Key Finding** — References "`resolve_partial_rotary_factor()` function above" — no such function is defined anywhere in this chapter. The reference is a dangling citation.

3. **`hf_config_resolution.md`, Section 1** — "PretrainedConfig.__init__ stores any unrecognised kwargs as instance attributes using `setattr(self, key, value)`" misrepresents the mechanism: `AutoConfig` resolves the model-specific class (e.g., `Qwen3_5MoeConfig`) whose `__init__` handles attributes explicitly, not generically via `PretrainedConfig`'s setattr path.

4. **`hf_config_resolution.md`, Section 1 and Section 3** — The `AttributeError` risk is framed as unconditional for Qwen3.5, but per ground truth, `Qwen3_5MoeConfig.__init__` always sets `self.partial_rotary_factor`; the risk only applies to code bypassing `__init__`.

5. **`hf_config_resolution.md`, Section 2 table** — `config.partial_rotary_factor` for Qwen3.5 listed as `AttributeError`; this is false for configs properly loaded via `AutoConfig.from_pretrained` (which returns `Qwen3_5MoeConfig`, whose `__init__` sets the attribute).

## B Feedback Application Log — Pass 7

- Fix 1: Added "(simplified illustration)" qualifier to the Section 4 `kwargs.get` code snippet in `hf_config_resolution.md`.
- Fix 2: Removed undefined `resolve_partial_rotary_factor()` reference from Key Finding; replaced with a note that the guard's `1.0` fallback is never reached for Qwen checkpoints.
- Fix 3: Rewrote Section 1 opening paragraph to correctly describe that `AutoConfig` resolves the model-specific config class whose `__init__` handles attributes explicitly.
- Fix 4: Changed Section 1 Qwen3.5 code block from `AttributeError` to `0.25 (set by Qwen3_5MoeConfig.__init__)` and added a note that `AttributeError` only occurs when bypassing `__init__`. Updated Section 3 "PretrainedConfig setattr path" language to correctly explain the mechanism.
- Fix 5: Updated Section 2 table — `config.partial_rotary_factor` for Qwen3.5 changed from `AttributeError` to `0.25 (via __init__)`.
- Proactive: Applied same `AttributeError` qualification to `index.md` "The One Actionable Risk" section and `ttnn_rope_impact.md` Sections 1 and 4, for consistency with the hf_config_resolution.md fixes.

## B Feedback — Pass 8

1. **`hf_config_resolution.md`, Section 3** — "(or `rope_parameters`)" introduces a dict name not in the ground truth; only `rope_scaling` is authoritative.

2. **`ttnn_rope_impact.md`, Key Finding** — "the same code will raise `AttributeError` on **Qwen3.5** configs" is too broad; should be qualified to config objects bypassing `Qwen3_5MoeConfig.__init__`.

3. **`ttnn_rope_impact.md`, Section 2 table** — Qwen3.5 cell "inside `rope_scaling` only" misrepresents the loaded object state; after `AutoConfig.from_pretrained`, `partial_rotary_factor` is also a top-level attribute via `__init__`.

4. **`hf_config_resolution.md`, Section 4 code** — `rope_parameters_value` variable uses the non-ground-truth name; should be `rope_scaling_value`.

5. **`index.md`, "The One Actionable Risk"** — "Any code that must handle both checkpoints… requires the guard pattern" overstates the requirement; for properly loaded `AutoConfig` objects the guard is not strictly needed.

## B Feedback Application Log — Pass 8

- Fix 1: Removed "(or `rope_parameters`)" from Section 3 of `hf_config_resolution.md`.
- Fix 2: Changed Key Finding in `ttnn_rope_impact.md` from "on **Qwen3.5** configs" to "on **Qwen3.5** config objects that bypass `Qwen3_5MoeConfig.__init__`".
- Fix 3: Updated Section 2 table Qwen3.5 cell to "0.25 (JSON: rope_scaling only; loaded object: also top-level via __init__)".
- Fix 4: Changed `rope_parameters_value` → `rope_scaling_value` in Section 4 code of `hf_config_resolution.md`.
- Fix 5: Narrowed `index.md` guard requirement: "Code working with raw config objects… requires the guard pattern (code using properly loaded AutoConfig objects does not strictly need it)".
