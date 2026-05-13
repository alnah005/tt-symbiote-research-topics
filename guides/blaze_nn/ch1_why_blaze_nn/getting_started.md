# Getting started: install, environment, three test tiers

Goal of this page: a clean checkout, a working `import blaze_nn`, and a green test run on whichever of the three tiers your machine supports.

## Install

blaze-nn is a pure-Python package with **no required dependencies** (`pyproject.toml:11` — `dependencies = []`). The `[dev]` extra adds `pytest` and `torch`, the latter only for golden-reference comparisons in tests (`pyproject.toml:13-14`).

```bash
git clone git@github.com:tenstorrent/blaze-nn.git
cd blaze-nn
pip install -e ".[dev]"
```

Sanity check that the package imports and exposes its surface:

```python
import blaze_nn
print(dir(blaze_nn))
# Includes: 'Module', 'Parameter', 'Sequential', 'ModuleList',
# 'ModuleDict', 'compose', 'functional', 'F', 'modules', 'ops'
```

The exported names are declared in `__all__` at `blaze_nn/__init__.py:53-64`.

> **Note:** `import blaze_nn` is safe on a machine **without** tt-blaze installed. All `blaze` and `ttnn` imports are deferred to the point a `Module` is actually executed — for example, `BlazeCompiler` is imported inside `_call_graph` at `blaze_nn/modules/base.py:98`, not at package load. This is the basis for the framework-only test tier below.

## The three test tiers

The same `pytest` invocation runs all three, with the upper tiers self-skipping when their prerequisites are missing. Pick the lowest tier your environment supports first; each one is a strict superset of the previous.

### Tier A — framework-only

**Prerequisites:** Python, `pytest`. No tt-blaze, no ttnn, no device.

**What it covers:** the `Module` attribute protocol, `Parameter` semantics, container behavior, `OpModule` construction, state-dict roundtrips, and the closure structure of the functional dispatcher.

**How it stays self-contained:** these tests use `object()` sentinels in place of `ttnn.Tensor`. Because the framework treats parameter tensors as opaque `Any` (`blaze_nn/parameter.py:18`), the test does not need a real tensor type to verify routing.

```bash
python -m pytest tests/test_module.py tests/test_parameter.py \
                 tests/test_containers.py tests/test_state_dict.py \
                 tests/test_op_module.py tests/test_functional.py -v
```

This is the tier you run on every commit.

### Tier B — dispatch integration

**Prerequisites:** Tier A plus tt-blaze importable (`import blaze` must succeed). No device required.

**What it covers:** the full `F.<op>` → `_dispatch` → `ctx.dispatch` → `BlazeGraph` chain. The test constructs a graph and asserts that the expected op nodes appear, without ever running on hardware.

**Gate:** the file opens with `pytest.importorskip("blaze")`, so it auto-skips when tt-blaze is not on `PYTHONPATH` and the rest of the test suite still runs (`README.md:46-49`).

```bash
python -m pytest tests/test_dispatch_integration.py -v
```

### Tier C — parity on hardware

**Prerequisites:** Tier B plus ttnn and an attached Tenstorrent device.

**What it covers:** end-to-end module execution compared against a torch reference at 0.99 PCC threshold (the `comp_pcc` helper in `tests/torch_reference.py`). This is where `tests/test_pytorch_parity.py` lives.

```bash
python -m pytest tests/test_pytorch_parity.py -v
```

> **Note:** Chapter 7 (`testing_strategy.md`) is a reverse index — it lists every test file by tier and links back to the chapter section whose claims that file backs. Use it when adding a new framework feature.

## tt-blaze environment

For Tiers B and C, tt-blaze must be importable. The standard recipe (`README.md:30-34`):

```bash
source /path/to/tt-blaze/env.sh
```

If your tt-blaze checkout has no built `tt-metal/` submodule but a built tt-metal lives elsewhere, point at it explicitly (`README.md:36-44`):

```bash
source /path/to/built/tt-metal/python_env/bin/activate
export TT_METAL_HOME=/path/to/built/tt-metal
export PYTHONPATH=/path/to/tt-blaze:/path/to/built/tt-metal:$PYTHONPATH
```

After sourcing, verify both imports resolve before running the upper tiers:

```bash
python -c "import blaze, ttnn; print('ok')"
```

> **Warning:** for Tier C, `module.to(device)` does **not** move parameter tensors onto the device — it only stores a `DeviceConfig` (covered in Chapter 2 `device_binding.md`). You must construct each `ttnn.Tensor` with the desired `memory_config` and `shard_spec` *before* calling `load_state_dict`. Mismatches surface as runtime errors during compile, not at load.

## Where to go next

You now have the mental model (Ch1 `what_it_is.md`), the invariant (Ch1 `ttnn_native_contract.md`), and a green test run. Chapter 2 opens the two foundational classes: `Module`'s attribute protocol, `Parameter`'s lifecycle, `state_dict` / `load_state_dict`, `module.to(device)`, and the `interop` helpers that build the dict you feed to `load_state_dict`.

_Previous: [The ttnn-native contract](ttnn_native_contract.md) · Next: [Chapter 2 — Module, Parameter, and the device boundary](../ch2_module_and_parameter/index.md) · [Up](index.md)_
