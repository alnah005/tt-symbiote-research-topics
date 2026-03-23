# Operational Concerns

A trace that works correctly on the day it was captured can silently produce wrong outputs after a seemingly unrelated code change, a model update, or a device restart.

---

## Re-Capture Triggers: When a Trace Becomes Invalid

A captured trace is a snapshot of a specific command sequence with specific buffer addresses. Any change that alters either the command sequence or the buffer addresses invalidates the trace. The categories below are exhaustive for standard transformer decode workloads; models with custom ops may have additional triggers specific to their dispatch structure.

### Trigger 1: Tensor Shape Change

**What changes:** Any tensor in the traced region has a different shape than it had at capture time.

**Why it invalidates the trace:** The trace encodes the DRAM addresses of every tensor it accesses. A tensor's DRAM address is assigned by the allocator based on the tensor's size in bytes. A shape change (new batch size, new hidden dimension, new max sequence length) changes the size, which changes the address. The trace replays to the old address, which either belongs to a different tensor now or is unallocated, producing incorrect output or a device fault.

**Common sources:**
- Changing `batch_size` to serve a different concurrency level.
- Changing `max_seq_len` (KV-cache capacity) to support longer contexts.
- Changing `hidden_dim`, `num_heads`, or `ffn_dim` when switching to a different model size.
- Changing the model precision from `bfloat16` to `float32` or `float16` (element size changes, buffer size changes, address changes).
- Adding or removing a model component (layer, projection, normalization) that contributes tensors to the traced region.

**Detection:** Shape changes almost always cause an immediate observable failure — either a device fault (accessing an unmapped address) or obviously wrong output values. They are among the easier trace invalidation causes to catch in testing.

**Resolution:** Release the existing trace, reallocate all tensors at the new shapes, warm up, and re-capture.

---

### Trigger 2: Weight Update or In-Place Weight Modification

**What changes:** A weight tensor's values are modified after capture while its buffer address stays the same.

**Why it invalidates the trace:** The trace does not record weight values — it records the address of the weight buffer and the kernel that uses it. If you update a weight in-place (for example, applying a LoRA adapter by adding a delta to the base weight), the trace will use the updated weight values on replay, which is usually the desired behavior. However, if you replace a weight buffer with a new tensor (different allocation), the trace still points to the old address. The new weight is invisible to the replay.

More subtly: if the weight modification changes the tensor's `dtype` or `layout` (e.g., re-quantizing from `bfloat16` to `uint8`), a new buffer will be allocated regardless of whether you intended an in-place update, because the buffer size changes. The old address encoded in the trace is now stale.

**Common sources:**
- Applying a fine-tuning delta to a production model while keeping the same server instance running.
- Switching quantization levels between requests.
- Hot-swapping LoRA adapters that are implemented by reassigning weight tensors rather than by in-place addition.
- Model version upgrade in a long-running inference server.

**Detection:** Weight-update invalidation is the most dangerous failure mode because the trace executes successfully from the device's perspective — no device fault, no error. Output is numerically wrong, and there is no signal from the trace machinery pointing to the update as the cause.

**Resolution pattern for hot LoRA swap:**

```python
class TracedModelWithLoRA:
    def __init__(self, device, config):
        self.device = device
        self.config = config
        self.trace_id = None
        self._capture_done = False

    def load_base_model(self, checkpoint_path: str):
        """Load base weights and pre-allocate tensors. Does not capture trace."""
        self.weights = load_weights_to_device(self.config, self.device)
        self.input_tensor, self.kv_cache, self.position_tensor, self.output_tensor = \
            preallocate_tensors(self.config, self.device)
        self._capture_done = False

    def apply_lora_inplace(self, lora_delta: dict):
        """
        Apply LoRA delta by in-place addition to existing weight buffers.

        This is SAFE for trace because the buffer addresses do not change:
        the delta is added into the existing allocation.

        After this call the existing trace is still valid — the kernels now
        operate on the updated weight values, which is the intended behavior.
        """
        for key, delta_torch in lora_delta.items():
            if key in self.weights:
                delta_device = ttnn.from_torch(
                    delta_torch, dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT, device=self.device
                )
                # ttnn.add_ is an in-place addition: same buffer address, new values.
                ttnn.add_(self.weights[key], delta_device)
        # NOTE: trace remains valid. No re-capture needed.

    def swap_lora_by_replacement(self, new_weights: dict):
        """
        Replace weights by assigning new tensors. This INVALIDATES the trace.

        If you use this pattern, you must release and re-capture.
        """
        if self.trace_id is not None:
            ttnn.release_trace(self.device, self.trace_id)
            self.trace_id = None
            self._capture_done = False

        for key, tensor in new_weights.items():
            self.weights[key] = tensor  # new allocation, new address

        # Re-capture is required before the next decode call.
        self._ensure_trace_captured()

    def _ensure_trace_captured(self):
        if not self._capture_done:
            self.trace_id, _ = capture_trace(
                self.device, self.weights, self.input_tensor,
                self.kv_cache, self.position_tensor, self.output_tensor, self.config
            )
            self._capture_done = True
```

---

### Trigger 3: Device Reset or Re-Open

**What changes:** The device is closed (`ttnn.close_device`) and re-opened, or the device firmware is reset.

**Why it invalidates the trace:** All device DRAM contents are lost on device reset. The trace buffer stored in device DRAM is gone. All tensor buffers allocated before the reset are gone. Every address encoded in the trace no longer refers to any valid allocation.

**What to do:** After any device reset, perform full re-initialization: re-open the device, re-enable async mode, re-load all weights, re-allocate all tensors, and re-capture the trace. There is no partial recovery; the entire setup sequence must be repeated.

**Detection:** Calling `ttnn.execute_trace` with a `trace_id` that was captured before a device reset raises a runtime error (see the Error Handling section below). If the device was reset and re-opened under the hood (for example, by a fault recovery mechanism in a long-running server), ensure the recovery path includes trace re-capture.

```python
def reinitialize_after_device_reset(config: ModelConfig):
    """
    Full re-initialization sequence after a device reset.

    This is the only valid recovery path after a device reset.
    """
    device = setup_device()                                # re-open + enable_async
    weights = load_weights_to_device(config, device)      # reload all weights
    tensors = preallocate_tensors(config, device)         # reallocate all buffers
    input_t, kv_cache, position_t, output_t = tensors
    trace_id, _ = capture_trace(                          # re-capture
        device, weights, input_t, kv_cache, position_t, output_t, config
    )
    return device, weights, tensors, trace_id
```

---

### Trigger 4: Change in Device Configuration

**What changes:** The device is opened with a different `num_hw_cqs` value, or with different core grid or firmware settings.

**Why it invalidates the trace:** The trace encodes CQ-specific command formats and kernel configurations. A trace captured on `num_hw_cqs=1` (single CQ) was encoded for CQ0's command buffer format. If the device is re-opened with `num_hw_cqs=2`, the CQ layout changes and the trace buffer format may be incompatible.

**Detection:** Using a trace captured under one `num_hw_cqs` setting after re-opening with a different setting raises a runtime error on `execute_trace`.

**Prevention:** Store the device configuration used at capture time alongside the `trace_id`. Validate the current device configuration matches before calling `execute_trace`.

---

### Trigger 5: Op Sequence Change in the Traced Region

**What changes:** A code modification adds, removes, or reorders ops inside the `decode_core` function (or whatever function was called during capture).

**Why it invalidates the trace:** The trace records the exact sequence of commands that were dispatched during capture. If the sequence is different — because an op was added, removed, or its arguments changed — the trace replays the old sequence. The output tensor contains the result of the old computation, not the new one.

**Detection:** Op sequence changes are the primary target of regression tests. The test pattern in Chapter 5 (`profiling_workflow.md`, Stage 6) detects this by comparing trace replay output to a live dispatch run on every CI commit. Any divergence indicates an op sequence change that the trace has not yet been updated to reflect.

---

## Detecting a Stale Trace at Runtime

The two reliable signals that a trace has become stale are numerical divergence and device-level errors. Structural validation (comparing the current model configuration to the configuration used at capture time) is the preferred early-warning approach.

### Approach 1: Configuration Hash Guard

Store a hash of the model configuration at the time of capture. Before each `execute_trace` call, recompute the hash and compare. If they differ, refuse to replay and trigger re-capture.

```python
import hashlib
import json

def compute_config_hash(
    config: ModelConfig,
    weight_shapes: Dict[str, Tuple[int, ...]],
    num_hw_cqs: int,
) -> str:
    """
    Compute a stable hash over all configuration that affects trace validity:
    model dimensions, weight shapes, and device CQ count.

    Weight shapes (not values) are included because a shape change implies
    a new allocation and therefore a new address. Weight value changes that
    use in-place modification do not invalidate the trace and are NOT included.
    """
    config_dict = {
        "hidden_dim":  config.hidden_dim,
        "num_heads":   config.num_heads,
        "head_dim":    config.head_dim,
        "ffn_dim":     config.ffn_dim,
        "num_layers":  config.num_layers,
        "max_seq_len": config.max_seq_len,
        "batch_size":  config.batch_size,
        "vocab_size":  config.vocab_size,
        "num_hw_cqs":  num_hw_cqs,
        "weight_shapes": {k: list(v) for k, v in sorted(weight_shapes.items())},
    }
    blob = json.dumps(config_dict, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class TracedDecoder:
    """
    Wrapper that guards execute_trace with a config hash check.
    Re-captures automatically when the hash changes.
    """

    def __init__(self, device: ttnn.Device, config: ModelConfig):
        self.device = device
        self.config = config
        self.trace_id: Optional[int] = None
        self._capture_hash: Optional[str] = None
        self.weights: Dict[str, ttnn.Tensor] = {}
        self.input_tensor: Optional[ttnn.Tensor] = None
        self.kv_cache: Dict[str, ttnn.Tensor] = {}
        self.position_tensor: Optional[ttnn.Tensor] = None
        self.output_tensor: Optional[ttnn.Tensor] = None

    def _current_config_hash(self) -> str:
        weight_shapes = {k: tuple(v.shape) for k, v in self.weights.items()}
        return compute_config_hash(self.config, weight_shapes, num_hw_cqs=1)

    def _ensure_valid_trace(self):
        current_hash = self._current_config_hash()

        if self._capture_hash != current_hash:
            # Config has changed since last capture (or no capture has been done).
            # Release any existing trace before re-capturing.
            if self.trace_id is not None:
                ttnn.release_trace(self.device, self.trace_id)
                self.trace_id = None

            self.trace_id, _ = capture_trace(
                self.device,
                self.weights,
                self.input_tensor,
                self.kv_cache,
                self.position_tensor,
                self.output_tensor,
                self.config,
            )
            self._capture_hash = current_hash

    def decode(
        self,
        current_token: int,
        step: int,
        embedding_table: torch.Tensor,
    ) -> int:
        """One traced decode step with automatic re-capture if config changed."""
        self._ensure_valid_trace()

        embedding = embedding_table[current_token].view(
            self.config.batch_size, 1, self.config.hidden_dim
        )
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(embedding, dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT),
            self.input_tensor,
        )

        position = step
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(torch.tensor([[position]], dtype=torch.int32),
                            dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
            self.position_tensor,
        )

        safe_execute_trace(self.device, self.trace_id, cq_id=0)
        ttnn.synchronize_device(self.device)

        logits_host = ttnn.to_torch(self.output_tensor)
        return int(logits_host[0, 0, :].argmax(-1).item())
```

> **Note:** `_ensure_valid_trace` is called at the start of every decode step. For a production model that runs thousands of steps without configuration changes, the hash comparison costs fewer than 5 us per step — negligible relative to the 1 300 us step latency. The re-capture path is taken only when the hash changes, which is a rare event in stable production deployments.

---

### Approach 2: Numerical Spot Check

Periodically (for example, once every 1000 steps or at the start of each request) re-run the decode core in live dispatch mode on the same input used by the most recent trace replay and compare outputs. If the outputs diverge beyond the expected bfloat16 tolerance, the trace is stale.

```python
def spot_check_trace_correctness(
    device: ttnn.Device,
    trace_id: int,
    weights: Dict[str, ttnn.Tensor],
    input_tensor: ttnn.Tensor,
    kv_cache: Dict[str, ttnn.Tensor],
    position_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    config: ModelConfig,
    check_input_host: torch.Tensor,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> bool:
    """
    Run one live dispatch step and one trace replay step on the same input.
    Return True if outputs agree within tolerance, False if the trace is stale.

    This check is not free: it runs one full live dispatch, which costs the
    full dispatch overhead that trace was designed to eliminate. Use it
    sparingly — at the start of a request, or after any configuration change.
    """
    # Set up the same input for both runs.
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(check_input_host, dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT),
        input_tensor,
    )

    # Live dispatch reference.
    live_logits = decode_core(input_tensor, weights, kv_cache, position_tensor, config)
    ttnn.synchronize_device(device)
    live_out = ttnn.to_torch(live_logits)

    # Restore the same input (live dispatch may have written into kv_cache state).
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(check_input_host, dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT),
        input_tensor,
    )

    # Trace replay.
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    ttnn.synchronize_device(device)
    traced_out = ttnn.to_torch(output_tensor)

    abs_diff = (live_out - traced_out).abs().max().item()
    rel_diff = (abs_diff / (live_out.abs().max().item() + 1e-8))

    if abs_diff > atol or rel_diff > rtol:
        print(
            f"Stale trace detected: max_abs={abs_diff:.6f}, max_rel={rel_diff:.6f} "
            f"(thresholds: atol={atol}, rtol={rtol})"
        )
        return False

    return True
```

---

## Error Handling: Trace Replay Exceptions

`ttnn.execute_trace` can raise exceptions in specific failure conditions. The table below lists each condition, its observable exception, and the correct recovery action.

| Condition | Exception raised | Recovery |
|---|---|---|
| `trace_id` was never returned by `end_trace_capture` (e.g., programmer error: wrong ID) | `RuntimeError: invalid trace_id` | Do not retry; fix the code to use the correct `trace_id` |
| `ttnn.release_trace` was called before `execute_trace` | `RuntimeError: trace buffer has been released` | Re-capture; do not call `execute_trace` after `release_trace` |
| Device reset between capture and replay | `RuntimeError: device state invalid` or a hardware fault | Full reinitialize (`reinitialize_after_device_reset`) |
| `cq_id` does not match the CQ used at capture | `RuntimeError: CQ mismatch` | Use the same `cq_id` as passed to `begin_trace_capture` and `end_trace_capture` |
| Device DRAM address conflict (another allocation overwrote the trace buffer) | Silent numerical incorrectness; no exception | Avoid allocating tensors after capture that could collide with trace buffer addresses; use the config hash guard |
| Concurrent capture: `begin_trace_capture` called while a session is open | `RuntimeError: trace capture already in progress` | Always pair `begin_trace_capture` with exactly one `end_trace_capture` before starting a new session |

### Structured recovery wrapper

```python
class TraceReplayError(Exception):
    """Raised when trace replay fails in a way that requires re-capture."""
    pass


class UnrecoverableTraceError(Exception):
    """Raised when trace replay fails in a way that requires device restart."""
    pass


def safe_execute_trace(
    device: ttnn.Device,
    trace_id: int,
    cq_id: int = 0,
) -> None:
    """
    Execute a trace replay with structured error handling.

    Translates TTNN runtime errors into typed exceptions that the caller
    can handle with specific recovery actions.

    Raises:
        TraceReplayError: trace is invalid but device is healthy; re-capture required.
        UnrecoverableTraceError: device is in a bad state; full reinitialize required.
    """
    try:
        ttnn.execute_trace(device, trace_id, cq_id=cq_id, blocking=False)
    except RuntimeError as exc:
        msg = str(exc).lower()

        if "invalid trace_id" in msg or "trace buffer has been released" in msg:
            raise TraceReplayError(
                f"Trace is no longer valid (trace_id={trace_id}): {exc}. "
                "Release the existing trace, re-capture, and retry."
            ) from exc

        if "cq mismatch" in msg:
            raise TraceReplayError(
                f"CQ mismatch on execute_trace (cq_id={cq_id}): {exc}. "
                "Ensure cq_id matches the value used during capture."
            ) from exc

        if "device state invalid" in msg or "hardware fault" in msg:
            raise UnrecoverableTraceError(
                f"Device is in an unrecoverable state: {exc}. "
                "Call reinitialize_after_device_reset() and re-capture the trace."
            ) from exc

        # Unknown RuntimeError: re-raise for investigation.
        raise


def decode_with_recovery(
    decoder: TracedDecoder,
    current_token: int,
    step: int,
    embedding_table: torch.Tensor,
    max_recapture_attempts: int = 2,
) -> int:
    """
    Decode one step with automatic re-capture on recoverable trace errors.

    On TraceReplayError, releases the stale trace, re-captures, and retries.
    On UnrecoverableTraceError, propagates to the caller (requires full restart).
    """
    for attempt in range(max_recapture_attempts + 1):
        try:
            return decoder.decode(current_token, step, embedding_table)

        except TraceReplayError as exc:
            if attempt < max_recapture_attempts:
                print(
                    f"Recoverable trace error on attempt {attempt + 1}: {exc}. "
                    "Forcing re-capture and retrying."
                )
                # Force re-capture by invalidating the stored hash.
                decoder._capture_hash = None
            else:
                raise RuntimeError(
                    f"Trace replay failed after {max_recapture_attempts} re-capture "
                    f"attempts. Giving up."
                ) from exc
```

---

## CI Integration

### Strategy 1: Capture-at-Test-Time (Recommended)

The most reliable CI strategy is to re-capture the trace at the start of every test run. This ensures that every test always reflects the current model code, without relying on a stored trace artifact that may be stale.

```python
# tests/conftest.py
import pytest
import ttnn
import torch


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0, num_hw_cqs=1)
    dev.enable_async(True)
    yield dev
    ttnn.close_device(dev)


@pytest.fixture(scope="module")
def traced_model(device):
    """
    Fixture that sets up the traced model once per test module.

    The trace is captured fresh at fixture setup time, ensuring it always
    reflects the current code. No stored trace artifact is used.
    """
    config = ModelConfig()
    weights = load_weights_to_device(config, device)
    input_tensor, kv_cache, position_tensor, output_tensor = \
        preallocate_tensors(config, device)

    # Capture.
    trace_id, capture_logits = capture_trace(
        device, weights, input_tensor, kv_cache,
        position_tensor, output_tensor, config
    )

    yield {
        "config":          config,
        "weights":         weights,
        "input_tensor":    input_tensor,
        "kv_cache":        kv_cache,
        "position_tensor": position_tensor,
        "output_tensor":   output_tensor,
        "trace_id":        trace_id,
        "capture_logits":  capture_logits,
    }

    # Teardown: release the trace after all tests in the module have run.
    ttnn.release_trace(device, trace_id)
```

### Correctness regression test

```python
# tests/test_trace_correctness.py
import pytest
import ttnn
import torch


def test_trace_replay_matches_live_dispatch(device, traced_model):
    """
    Verify that one trace replay step produces output numerically identical
    to one live dispatch step given the same input.

    This test catches any code change that modifies the op sequence inside
    decode_core without updating the captured trace.
    """
    cfg = traced_model["config"]
    weights = traced_model["weights"]
    input_tensor = traced_model["input_tensor"]
    kv_cache = traced_model["kv_cache"]
    position_tensor = traced_model["position_tensor"]
    output_tensor = traced_model["output_tensor"]
    trace_id = traced_model["trace_id"]

    # Use a fixed input for reproducibility.
    test_input = torch.randn(cfg.batch_size, 1, cfg.hidden_dim, dtype=torch.bfloat16)

    # ── Live dispatch reference ───────────────────────────────────────────────
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(test_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
        input_tensor,
    )
    live_out_tensor = decode_core(
        input_tensor, weights, kv_cache, position_tensor, cfg
    )
    ttnn.synchronize_device(device)
    live_out = ttnn.to_torch(live_out_tensor)

    # Restore the same input before the trace replay.
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(test_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
        input_tensor,
    )

    # ── Trace replay ──────────────────────────────────────────────────────────
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    ttnn.synchronize_device(device)
    traced_out = ttnn.to_torch(output_tensor)

    # ── Assert numerical agreement ────────────────────────────────────────────
    torch.testing.assert_close(
        traced_out,
        live_out,
        atol=1e-2,
        rtol=1e-2,
        msg=(
            "Trace replay output does not match live dispatch output. "
            "A code change may have altered the op sequence in decode_core "
            "without updating the captured trace."
        ),
    )


def test_trace_speedup_above_minimum(device, traced_model):
    """
    Verify that the traced decode step is measurably faster than the untraced baseline.

    Fails if speedup falls below 1.05x (5%), which would indicate that
    the trace is not eliminating dispatch overhead as expected — for example,
    because the model is too small or the batch size is too large for
    dispatch to be a significant fraction.

    The threshold is intentionally conservative (5%) to avoid flakiness from
    measurement noise while still catching regressions where the trace provides
    zero benefit.
    """
    import time, statistics

    cfg = traced_model["config"]
    weights = traced_model["weights"]
    input_tensor = traced_model["input_tensor"]
    kv_cache = traced_model["kv_cache"]
    position_tensor = traced_model["position_tensor"]
    output_tensor = traced_model["output_tensor"]
    trace_id = traced_model["trace_id"]

    WARMUP = 5
    NUM_STEPS = 30
    MIN_SPEEDUP = 1.05

    # Warm-up to establish steady-state kernel cache.
    for _ in range(WARMUP):
        decode_core(input_tensor, weights, kv_cache, position_tensor, cfg)
    ttnn.synchronize_device(device)

    # Baseline timing (live dispatch, no trace).
    baseline_times_us = []
    for _ in range(NUM_STEPS):
        t0 = time.perf_counter_ns()
        decode_core(input_tensor, weights, kv_cache, position_tensor, cfg)
        ttnn.synchronize_device(device)
        baseline_times_us.append((time.perf_counter_ns() - t0) / 1_000.0)

    # Traced timing (execute_trace).
    traced_times_us = []
    for _ in range(NUM_STEPS):
        t0 = time.perf_counter_ns()
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        traced_times_us.append((time.perf_counter_ns() - t0) / 1_000.0)

    measured_speedup = (
        statistics.mean(baseline_times_us) / statistics.mean(traced_times_us)
    )

    assert measured_speedup >= MIN_SPEEDUP, (
        f"Trace speedup {measured_speedup:.3f}x is below the minimum threshold "
        f"{MIN_SPEEDUP}x. "
        f"Baseline mean: {statistics.mean(baseline_times_us):.1f} us, "
        f"Traced mean: {statistics.mean(traced_times_us):.1f} us. "
        "Check that decode_core dispatches enough ops for trace overhead "
        "to be measurable at this batch size."
    )
```

### Strategy 2: Capture-on-First-Run with Stored Artifact (Advanced)

For very large models where re-capture takes more than a few minutes, the alternative is to capture once, serialize a capture validation artifact (not the trace buffer itself — that cannot be serialized to disk), and use the artifact to verify on subsequent runs that the stored trace is still valid.

```python
import json
import os

TRACE_ARTIFACT_PATH = "./trace_capture_artifact.json"


def save_capture_artifact(
    config: ModelConfig,
    weight_shapes: Dict[str, Tuple[int, ...]],
    capture_output_stats: Dict[str, float],
    num_hw_cqs: int,
):
    """
    Save a lightweight artifact that captures the model state at capture time.

    The artifact contains:
    - The config hash (all model dimensions and device config).
    - Summary statistics of the capture output (mean, max, min).
      These are used to detect silent numerical drift in re-validation.

    The trace buffer itself is NOT serialized — it lives on device DRAM and
    cannot be meaningfully saved to disk in a way that survives device resets.
    """
    artifact = {
        "config_hash": compute_config_hash(config, weight_shapes, num_hw_cqs),
        "capture_output_mean": capture_output_stats["mean"],
        "capture_output_max":  capture_output_stats["max"],
        "capture_output_min":  capture_output_stats["min"],
    }
    with open(TRACE_ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2)


def validate_against_artifact(
    config: ModelConfig,
    weight_shapes: Dict[str, Tuple[int, ...]],
    num_hw_cqs: int,
) -> bool:
    """
    Return True if the current config matches the stored artifact.
    Return False if the artifact is missing or if config has changed.
    """
    if not os.path.exists(TRACE_ARTIFACT_PATH):
        return False

    with open(TRACE_ARTIFACT_PATH) as f:
        artifact = json.load(f)

    current_hash = compute_config_hash(config, weight_shapes, num_hw_cqs)
    return artifact["config_hash"] == current_hash
```

> **Note:** Strategy 2 requires that the process does not restart between the original capture and subsequent uses — the trace lives on device DRAM and is not persistent across device resets. Strategy 2 is therefore only useful within a single long-running process (for example, a server that captures once at startup and runs thousands of requests without restarting). For CI, Strategy 1 (capture at test time) is always the correct choice because CI jobs start with a fresh process and device state.

### What to test in CI: summary

| Test | Trigger condition | Catches |
|---|---|---|
| `test_trace_replay_matches_live_dispatch` | Every commit that touches `decode_core` or any weight allocation | Op sequence changes, tensor address changes |
| `test_trace_speedup_above_minimum` | Every commit that touches the decode path | Regressions that accidentally disable or bypass the trace |
| Config hash validation | Every commit | Shape changes, layer count changes, dtype changes |
| Numerical spot check (optional, slow) | Pre-merge only; not on every commit | Subtle weight-handling errors not caught by shape hash |

---

**End of guide.** Return to [Guide Index](../index.md)
