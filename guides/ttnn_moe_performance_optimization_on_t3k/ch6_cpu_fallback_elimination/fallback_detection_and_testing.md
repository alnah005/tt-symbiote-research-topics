# Fallback Detection and Testing

## Context

This file addresses **Q6**: systematic detection of silent CPU fallbacks across `moe.py`, and automated testing infrastructure to prevent regressions.

Source ranges: `moe.py:L542–L556`, `moe.py:L559–L613`, `moe.py:L817–L852`.

---

## 1. Source-Level Detection

Before running any model, perform a static audit of `moe.py` with the following grep patterns. Each command targets a distinct category of silent CPU fallback.

```bash
# Find hardcoded flag that disables the TTNN path
grep -n "ttnn = False\|ttnn=False" moe.py

# Find explicit CPU tensor moves
grep -n '\.cpu()\|\.to("cpu")\|\.to('"'"'cpu'"'"')\|device("cpu")\|device('"'"'cpu'"'"')' moe.py

# Find conditional TTNN disables
grep -n "if not ttnn\|if ttnn is False\|if ttnn == False" moe.py

# Find torch-native ops that execute on CPU and may bypass TTNN
grep -n "nn\.functional\.\|torch\.nn\." moe.py
```

**Expected findings at the time of writing:**

- `ttnn = False` grep: one match at `moe.py:L569`. This is the only hardcoded disable flag. No other `ttnn = False` patterns exist in the file.

- CPU tensor move grep: no matches. The file contains no explicit `.cpu()` or `.to("cpu")` calls.

- Conditional TTNN disable grep: no matches. The only TTNN guard is the hardcoded flag; it does not appear in an `if not ttnn` form.

- `nn.functional.` grep: matches in `Glm4MoeExpertLayersTorch.forward` (`moe.py:L551–L556`) and `Glm4MoeNaiveMoe.forward` (`moe.py:L357–L359`). These are expected — both are the intentional CPU reference implementations, not hidden fallbacks elsewhere in the file.

If the grep for `ttnn = False` returns a result and the loaded `moe.py` is the one in use, the `Glm4MoeNaiveMoeHybrid` class will execute on CPU. This is the only silent fallback. All other CPU execution in `moe.py` is confined to explicit reference classes (`Glm4MoeNaiveMoe`, `Glm4MoeExpertLayersTorch`) that should not appear in a production `TTNNGlm4MoeMoE` deployment.

Re-run this audit whenever `moe.py` is modified. The `ttnn = False` grep in particular should be included in CI as a lint rule or pre-commit hook.

---

## 2. Runtime Detection: Host-Device Transfer Hook

Source-level grep catches static patterns but cannot detect dynamic CPU fallbacks triggered by control flow at runtime. This section provides a test harness that instruments `ttnn.to_torch` and `ttnn.from_torch` to surface unexpected host-device round-trips during inference.

```python
import unittest.mock as mock
import traceback
import ttnn

_original_to_torch = ttnn.to_torch
_original_from_torch = ttnn.from_torch

def patched_to_torch(tensor, *args, **kwargs):
    print("WARNING: ttnn.to_torch called from:")
    traceback.print_stack(limit=5)
    return _original_to_torch(tensor, *args, **kwargs)

def patched_from_torch(tensor, *args, **kwargs):
    print("WARNING: ttnn.from_torch called from:")
    traceback.print_stack(limit=5)
    return _original_from_torch(tensor, *args, **kwargs)

# --- Usage in a test or profiling script ---

# Step 1: Run one warmup forward pass WITHOUT the patches.
# This allows initialization-time transfers (weight loading, CCL semaphore
# handle creation) to complete before monitoring begins.
_ = model(inputs)

# Step 2: Enable patches and run the measured forward pass.
with mock.patch.object(ttnn, 'to_torch', patched_to_torch), \
     mock.patch.object(ttnn, 'from_torch', patched_from_torch):
    output = model(inputs)

# If no warnings are printed, no host-device transfers occurred during inference.
```

**What to expect:**

- With a correctly configured `TTNNMoE` or `TTNNBailingMoE`: zero patched calls during the measured forward. All tensor movement happens at initialization time, which is excluded by the warmup step.

- With `Glm4MoeNaiveMoeHybrid` as the expert module: expect approximately `top_k` calls (typically 8 at decode time) to `ttnn.from_torch` or equivalent conversion points as the TTNN router output is transferred back to host tensors before being passed to the CPU expert loop.

- During a TTNN forward pass, `ttnn.from_torch` is legitimately called at initialization time for CCL semaphore handles and weight tensors. The warmup step excludes these. If the patch still fires during the measured forward, inspect the printed stack trace to identify the call site.

To convert the hook from a warning to a hard assertion — useful in CI — replace the `print` with a `raise`:

```python
def patched_to_torch_strict(tensor, *args, **kwargs):
    stack = "".join(traceback.format_stack(limit=5))
    raise AssertionError(f"Unexpected ttnn.to_torch during inference:\n{stack}")
```

---

## 3. Runtime Detection: Module-Level Introspection

The module audit function walks `model.named_modules()` and classifies each MoE-related module as TTNN-accelerated, CPU-executing, or unknown. Run this before any profiling session.

```python
from moe import (
    Glm4MoeNaiveMoeHybrid,
    Glm4MoeExpertLayersTorch,
    Glm4MoeNaiveMoe,
    TTNNMoE,
    TTNNBailingMoE,
    TTNNExperts,
)

def audit_moe_modules(model):
    """
    Walk model.named_modules() and classify MoE-related modules.

    Returns a dict with keys:
      "cpu"   — modules that execute on the host CPU
      "ttnn"  — modules that execute on the T3K device via TTNN
      "unknown" — modules not classified (empty by design; extend as needed)

    A clean TTNNMoE deployment returns findings["cpu"] == [].
    """
    CPU_CLASSES = (Glm4MoeNaiveMoeHybrid, Glm4MoeExpertLayersTorch, Glm4MoeNaiveMoe)
    TTNN_CLASSES = (TTNNMoE, TTNNBailingMoE, TTNNExperts)
    findings = {"cpu": [], "ttnn": [], "unknown": []}

    for name, module in model.named_modules():
        if isinstance(module, CPU_CLASSES):
            findings["cpu"].append((name, type(module).__name__))
        elif isinstance(module, TTNN_CLASSES):
            findings["ttnn"].append((name, type(module).__name__))

    return findings
```

**Interpreting the output:**

```python
findings = audit_moe_modules(model)

# Print a summary
print("TTNN modules:", findings["ttnn"])
print("CPU  modules:", findings["cpu"])

# Expected for a clean deployment:
# TTNN modules: [('model.layers.0.mlp', 'TTNNMoE'), ...]
# CPU  modules: []
```

If `findings["cpu"]` contains any `Glm4MoeNaiveMoeHybrid` entries, the model was initialized via `TTNNGlm4MoeMoE.from_torch` with `ttnn = False` still set. If it contains `Glm4MoeExpertLayersTorch` entries without an enclosing `Glm4MoeNaiveMoeHybrid`, the expert layers were constructed directly from a Torch path and bypassed the hybrid wrapper entirely.

`Glm4MoeNaiveMoe` appearing in `findings["cpu"]` indicates the original non-hybrid class is still in use — this is the pure-PyTorch baseline (`moe.py:L324–L363`) and should never appear in a TTNN deployment.

The audit only classifies modules from the explicit class lists. If new CPU classes are introduced to `moe.py`, add them to `CPU_CLASSES`. The `"unknown"` bucket is intentionally unused in the current implementation but available for extensions.

---

## 4. Pytest Integration

The following fixture wraps the module audit in a pytest assertion. Attach it to any test that constructs the full model to catch CPU fallback regressions automatically.

```python
import pytest
from moe import (
    Glm4MoeNaiveMoeHybrid,
    Glm4MoeExpertLayersTorch,
    Glm4MoeNaiveMoe,
)

@pytest.fixture
def assert_no_cpu_fallback(model):
    """
    Pytest fixture: assert that no CPU-executing MoE modules are present in model.

    Usage:
        def test_moe_forward(model, assert_no_cpu_fallback):
            output = model(inputs)
            ...
    """
    findings = audit_moe_modules(model)
    assert findings["cpu"] == [], (
        f"CPU fallback modules found: {findings['cpu']}. "
        "Check for ttnn=False flags in moe.py or incorrect model initialization."
    )
```

The fixture is designed to be used as a parameter in test function signatures. It runs before the test body, so a CPU fallback causes a test failure immediately at setup rather than producing a slow but numerically correct result that passes silently.

For a more granular assertion, assert on specific class names rather than the full list:

```python
cpu_class_names = [name for _, cls_name in findings["cpu"] for name in [cls_name]]
assert "Glm4MoeNaiveMoeHybrid" not in cpu_class_names, (
    "Glm4MoeNaiveMoeHybrid found — ttnn=False is set at moe.py:L569"
)
```

This is particularly useful if `Glm4MoeNaiveMoe` is intentionally present in a reference test but `Glm4MoeNaiveMoeHybrid` must never be.

---

## 5. Confirming Correct Inference Path

This checklist supplements the six-item checklist in Ch1's `cpu_fallback_paths.md` (which covers class verification, the `ttnn=False` grep, device arch confirmation, weight-on-device verification, CCL semaphore initialization, and TTNN op trace inspection). The items below are specific to `TTNNGlm4MoeMoE` and the GLM-4 migration path.

**Before any profiling session:**

1. Run the module audit: `audit_moe_modules(model)` must return `findings["cpu"] == []`. This catches any `Glm4MoeNaiveMoeHybrid` present due to the `ttnn=False` flag. Cross-reference Ch1 checklist item 1 for the broader class verification approach.

2. If using `TTNNGlm4MoeMoE` as the entry point (via `TTNNGlm4MoeMoE.from_torch`), confirm that `model.layers[i].mlp.experts` is NOT an instance of `Glm4MoeNaiveMoeHybrid`. The audit above catches this, but a direct `isinstance` check is faster in a quick diagnostic session:

   ```python
   from moe import Glm4MoeNaiveMoeHybrid
   for i, layer in enumerate(model.layers):
       experts = layer.mlp.experts
       if isinstance(experts, Glm4MoeNaiveMoeHybrid):
           print(f"Layer {i}: CPU fallback active (Glm4MoeNaiveMoeHybrid)")
   ```

3. Run the host-device transfer hook with at least one warmup forward before enabling the patches. Zero patched calls during the measured forward confirms no runtime transfers are occurring.

4. If using `TTNNMoE` or `TTNNBailingMoE` directly (not via `TTNNGlm4MoeMoE`), the `Glm4MoeNaiveMoeHybrid` CPU fallback path does not apply — those classes use `TTNNExperts` for routed expert computation. Cross-reference Ch1 checklist item 1 to confirm the outer MoE class before applying the GLM-4-specific checks in this file.

5. As a final sanity check before collecting latency numbers, re-run the §1 source grep (`grep -n "ttnn = False" moe.py`). See §1 for expected output and interpretation.

---

**Previous:** [GLM-4 CPU Path Audit](glm4_cpu_path_audit.md) | **Next:** [Chapter 7 Index](../ch7_end_to_end_optimization_summary/index.md)
