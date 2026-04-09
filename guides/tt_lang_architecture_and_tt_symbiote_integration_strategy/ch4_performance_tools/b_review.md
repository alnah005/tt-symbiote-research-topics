# Agent B Review: Chapter 4 — Pass 1

**Verdict: 1 issue found.**

## Issue 1 — `generate_signpost_name()` described as canonical but is dead code (auto_profile.md)

**File:** `auto_profile.md`, section "1. Compile-Time Instrumentation"

The text says:

> The helper `generate_signpost_name()` produces the canonical form:
> ```python
> def generate_signpost_name(operation: str, lineno: int, col: int) -> str:
>     return f"{operation}_L{lineno}_C{col}"
> ```

This function exists in the source (`auto_profile.py:116`) but is never called anywhere in the codebase. The actual signpost names are constructed inline in `ttl_ast.py`:

- **Line-level signposts** (line 232): `f"{self.name}_L{file_lineno}"` — no `_C{col}` suffix.
- **Op-level signposts** (line 264): `f"{self.name}_L{file_lineno}_{prefix}{op_name}"` — also no `_C{col}`.

The doc's prose examples (`compute_L52`, `dm_read_L52_cb_wait`) are correct and match the actual code. But presenting an unused function as "the canonical form" is misleading. Either remove the `generate_signpost_name()` block and its surrounding text, or note that it is unused/vestigial and that the actual naming happens in `ttl_ast.py`.

---

No other factual errors, coherence issues, structural gaps, or missing navigation elements found. All navigation footers are present on content files. All index.md links are clickable.

# Agent B Review: Chapter 4 — Pass 2

**No feedback — chapter approved.**

Pass 1 issue (`generate_signpost_name()` presented as canonical) has been fixed. The function is now correctly identified as dead code in `auto_profile.md` (line 38). All factual claims across the four files were re-verified against source:

- Execution order in `index.md` matches `ttl_api.py` lines 1442-1474.
- Signpost naming patterns (`<kernel>_L<lineno>`, `<kernel>_L<lineno>_{prefix}{op_name}`) match `ttl_ast.py`.
- `_USER_PREFIX = "ttl_"` filtering in `signpost_profile.py` confirmed.
- `TT_METAL_PROFILER_MID_RUN_DUMP` warning in `_run_signpost_profile()` confirmed at `ttl_api.py:304`.
- Wrapper zone filtering list in `perf_trace_server.py` matches doc.
- NOC event classification logic in `perf_summary.py` matches doc table.
- `_get_container_ip()` Docker detection and `_find_free_port()` confirmed.
- Standalone CLI args (`--path`, `--port`, `--json`, `--names`) confirmed for both modules.
- All navigation footers present. All index.md links are clickable relative paths.
