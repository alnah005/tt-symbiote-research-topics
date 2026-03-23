# Trace Constraints

This file defines the categories of operations and model behaviors that prevent successful tracing, explains the underlying mechanism that disqualifies each category, covers the prefill/decode asymmetry (why prefill almost never benefits from trace while decode almost always does), and gives concrete guidance on how to restructure a model to maximize the region that can safely be traced. By the end you will be able to read a model's forward pass, identify every construct that would break a trace, and decide where to draw the boundary between traced and untraced execution.

---

The trace cannot encode "do what the current call would do"; it can only encode "do what the capture call did."

---

## Category 1: Dynamic Shapes

**Definition:** An operation has a dynamic shape if the size of any dimension of its input or output tensors can vary from one call to the next.

**Why it breaks trace:** Buffer addresses are allocated based on tensor size. A tensor with shape `[1, S, 512]` where S varies between calls will be allocated a different number of bytes for different values of S. A larger S means a larger buffer, which the allocator places at a different DRAM address. Since the trace encodes the specific buffer address from the capture run, a replay with a different S will point the kernel to the wrong address — or to an address that was not allocated at all.

**Common sources of dynamic shapes in transformer inference:**

- **Prefill:** The prompt sequence length S varies per request. The key and query projections produce tensors of shape `[B, heads, S, head_dim]` where S is the prompt length. The attention computation's intermediate tensors are `[B, heads, S, S]`, which grows quadratically with S. None of these can be traced across requests with different prompt lengths.
- **Speculative decoding verification:** The verification step processes a batch of draft tokens whose count varies per step (depending on how many draft tokens were accepted in the previous step).
- **Dynamic batching:** Variable batch sizes between calls produce different-sized input tensors.
- **Output-length-dependent operations:** Ops whose output shape is a function of the input values (e.g., `torch.nonzero`, `torch.unique`) produce dynamically sized outputs. TTNN equivalents of these are disqualified.

> **Example:** A decode step with a fixed sequence length of 1 (autoregressive generation of one token at a time) and a fixed batch size has no dynamic shapes in the attention projection, matmul, or FFN ops. Every tensor's shape is known statically from the model configuration. This is why single-token decode is the canonical traceable workload.

---

## Category 2: Data-Dependent Dispatch (Branching and Value-Dependent Ops)

**Definition:** Any dispatch path whose outcome depends on runtime conditions — whether a Python control flow construct whose condition involves the value of a device tensor, or an op whose internal behavior or output shape depends on the numeric content of its input data rather than just the input shape.

**Why it breaks trace:** The trace records only the specific sequence of commands dispatched during capture. If a different branch would be taken, or different commands would be generated, on a subsequent call — because input values differ or a device-computed predicate evaluates differently — the trace will replay the wrong command sequence.

**Examples — host-side branching on device outputs:**

```python
# UNTRACEABLE: Python branch on a device tensor value.
# The trace will record only one branch, whichever ran at capture time.
logits = ttnn.matmul(hidden, self.lm_head)
logits_host = ttnn.to_torch(logits)
if logits_host.max() > threshold:              # host-side read of device value
    output = ttnn.softmax(logits, dim=-1)
else:
    output = ttnn.relu(logits)
```

```python
# UNTRACEABLE: loop count determined by a device output.
while not ttnn.to_torch(is_done_tensor).item():   # host-side read each iter
    hidden = step(hidden)
```

```python
# TRACEABLE: Python branch that does NOT depend on device tensor values.
# The branch is resolved at Python import/config time, not at op dispatch time.
if config.use_gated_ffn:
    output = self.gated_ffn(hidden)     # always takes this branch — traceable
else:
    output = self.dense_ffn(hidden)     # the other branch is never dispatched
```

The key distinction is whether the branching condition can be resolved before any op is dispatched, or whether it requires reading a device value. Compile-time or config-time conditions are fine; device-value-dependent conditions are not.

**Examples — value-dependent ops:**

- **EOS detection inside the model** — if the model's forward pass detects an end-of-sequence token and modifies its computation path based on that detection, the path taken during capture will differ from the path taken when EOS is not present.
- **Padding mask short-circuits** — ops that early-exit based on input content (e.g., skipping computation for fully-padded positions in a way that changes which kernels run).
- **Data-dependent gather/scatter with variable output size** — operations where the number of non-zero elements in an output is determined by a threshold applied to input data.

**What is NOT data-dependent (and is therefore traceable):**

Standard attention, FFN, normalization, and embedding lookup ops are data-independent: given the same input shapes and the same weight addresses, they always issue the same kernel with the same buffer accesses regardless of input values. The compute result varies (as intended), but the dispatch commands do not.

---

## Category 3: Ops That Require a Host Readback Mid-Sequence

**Definition:** Any operation that, as part of its dispatch or execution, requires the host to read a value from device memory before the next op can be dispatched.

**Why it breaks trace:** Readbacks are synchronization points (as described in Chapter 2). A readback mid-sequence means the host pauses dispatch, waits for the device, reads a value, and uses that value to decide what to dispatch next. This decision cannot be pre-recorded in the trace because the decision depends on the specific device output at runtime.

In TTNN specifically, ops that trigger host readbacks mid-dispatch typically include:

- `ttnn.to_torch()` called on a tensor whose value is needed to determine the next op's arguments.
- Device-to-host transfers used to implement dynamic loop bounds.
- Profiling callbacks registered on specific ops that inspect tensor values.

> **Warning:** If your model calls `ttnn.to_torch()` anywhere between `ttnn.begin_trace_capture` and `ttnn.end_trace_capture`, the capture will complete — the readback call executes and returns a value — but the host-side decision that follows the readback will be baked into the capture as a specific sequence of ops. On replay, the trace will always execute the ops that corresponded to the capture-time readback value, regardless of what the readback would return on the current step.

---

## Category 4: Ops That Modify Their Own Dispatch Configuration Based on Prior Results

**Definition:** Ops that inspect a prior op's output to decide how to configure themselves — for example, selecting a different kernel variant based on the measured sparsity of an intermediate result.

**Why it breaks trace:** The configuration chosen during capture is baked into the recorded command. If a different configuration would be chosen on a subsequent call (because the prior result has different sparsity), the trace replays the wrong configuration.

This category is less common in standard transformer architectures but appears in custom ops, pruning-aware models, and models with runtime-adaptive compute strategies.

---

## The Prefill / Decode Asymmetry

The single most important structural fact about tracing in LLM inference is that **prefill and decode have opposite traceability profiles**:

| | Prefill | Decode |
|---|---|---|
| Sequence length | Variable per request (S = prompt length) | Fixed: 1 token per step |
| Tensor shapes | Dynamic (depend on S) | Static (model-configuration constants) |
| Execution count | Once per request | Thousands of times per request |
| Dispatch overhead fraction | Low (one execution, long kernels dominate) | High (repeated, short step amortizes encoding) |
| Traceable? | Rarely, and only for fixed-length prompts | Almost always |
| Benefit from trace? | Minimal (trace overhead > gain for a single execution) | Significant (encoding eliminated from thousands of steps) |

**Prefill is usually untraceable** because prompt sequence lengths vary per request. Even if you pad all prompts to a fixed maximum length, doing so changes the computation (the attention mask and KV-cache fill length vary) and typically does not match standard prefill implementations. More fundamentally, the gain from tracing prefill is small: prefill runs once per request, and the dispatch overhead for one execution is a tiny fraction of the total prefill time (which is dominated by the O(S^2) attention computation).

**Decode is almost always traceable** for the standard single-token autoregressive case: each step generates exactly one token, all tensors have shapes that are constants of the model configuration (`[1, 1, hidden]`, `[1, heads, 1, head_dim]`, etc.), and the same op sequence repeats thousands of times. Eliminating per-step encoding overhead produces a compounding benefit across all decode steps.

> **Note:** "Almost always traceable" is not "always traceable." Decode steps that include speculative decoding, model-parallel conditional routing, or external control flow driven by streaming output are subject to the constraints above and may require restructuring. Evaluate each model's decode implementation against the categories in this file.

---

## How to Structure a Model to Maximize the Traceable Region

When a model's forward pass contains a mix of traceable and untraceable operations, the standard approach is to identify the boundary between the two and restructure the code so that the untraceable operations happen outside the trace boundary.

### Step 1: Identify the untraceable operations

Walk through the model's decode-step forward pass and apply the constraint categories above. Mark every op or Python statement that falls into one of the four categories. Common findings:

- Token sampling (argmax, top-k, nucleus sampling) — often involves a host readback (to select the next token in Python) or device-side ops with data-dependent outputs. Move to after `end_trace_capture`.
- EOS detection — typically a host-side check on the sampled token. Move to after `end_trace_capture`.
- KV-cache append with dynamic position index — if the position index is a Python variable that changes each step, and it is passed as a runtime argument that changes the command encoding, this may be untraceable. A common solution is to pass the position index as a pre-allocated device scalar tensor that is written before each replay step (like the input tensor) rather than as a Python int that becomes a kernel argument at encode time.
- Attention mask updates for incremental context — if the mask is generated dynamically in Python and differs each step, precompute the full mask at a fixed size and use a device-side slice or mask-application op instead.

### Step 2: Separate the model into a traced inner function and an untraced outer wrapper

```python
# BEFORE: monolithic forward pass, partially untraceable
class MyModel:
    def forward(self, token_ids, position):
        hidden = self.embedding(token_ids)
        for layer in self.layers:
            hidden = layer(hidden, position)     # position changes each step
        logits = self.lm_head(hidden)
        next_token = ttnn.to_torch(logits).argmax(-1).item()  # host readback
        return next_token

# AFTER: separate the traceable core from the untraceable wrapper
class MyModel:
    def decode_core(self, hidden_in, kv_cache):
        # Only fixed-shape, traceable ops here.
        # position is encoded in kv_cache state; not passed as a changing arg.
        for layer in self.layers:
            hidden_in = layer.forward_fixed(hidden_in, kv_cache)
        logits = self.lm_head(hidden_in)
        return logits   # device tensor, no readback

    def decode_step(self, token_ids, kv_cache, position):
        # Untraceable parts happen outside the traced region.
        hidden = self.embedding(token_ids)   # can be traced if shape fixed
        logits = self.decode_core(hidden, kv_cache)   # this is what we trace
        # Host-side sampling happens after trace replay completes.
        next_token = ttnn.to_torch(logits).argmax(-1).item()
        return next_token
```

Capture only `decode_core`. The outer `decode_step` wrapper handles embedding lookup (which may be traceable if the input shape is fixed) and token sampling (which is not traced). The host readback in `decode_step` happens once per step, after the replay of `decode_core` completes, and does not occur inside the trace boundary.

### Step 3: Use pre-allocated device tensors for per-step variables

Any value that changes each step but whose shape is fixed can be passed through a pre-allocated device tensor written before each replay:

```python
# Instead of: position as a Python int (changes each step, would vary kernel arg)
# Use: position as a fixed-size device tensor, updated in-place before replay

position_tensor = ttnn.from_torch(
    torch.tensor([0], dtype=torch.int32),
    dtype=ttnn.uint32,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    device=device,
)

# In the capture, position_tensor is read by ops as a device tensor.
# Its address is fixed; its value changes via in-place write before replay.
ttnn.begin_trace_capture(device, cq_id=0)
logits = model.decode_core(hidden, kv_cache, position_tensor)
trace_id = ttnn.end_trace_capture(device, cq_id=0)

# In the replay loop:
for step in range(num_steps):
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(torch.tensor([step], dtype=torch.int32), ...),
        position_tensor,   # in-place write to the same buffer address
    )
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
```

This pattern converts a per-step varying Python value into a per-step varying device tensor value. The buffer address stays constant (satisfying address fixity); only the content changes.

### Step 4: Validate replay outputs match live dispatch outputs

Before replacing your live dispatch loop with a trace replay loop, run both versions on identical inputs and compare outputs:

```python
# Save KV-cache state before capture so both runs start from identical state
kv_cache_backup = {k: ttnn.clone(v) for k, v in kv_cache.items()}

# Capture
ttnn.begin_trace_capture(device, cq_id=0)
out_trace = model.decode_core(input_tensor, kv_cache)
trace_id = ttnn.end_trace_capture(device, cq_id=0)

# Restore KV-cache so the live run starts from the same pre-capture state
for k in kv_cache:
    ttnn.copy_(kv_cache[k], kv_cache_backup[k])

# Live dispatch (re-run with same inputs, no trace)
out_live = model.decode_core(input_tensor, kv_cache)

# Compare
ttnn.synchronize_device(device)
assert ttnn.allclose(out_trace, out_live, atol=1e-2), \
    "Trace output does not match live dispatch output"
```

If outputs differ, the capture contains an incorrect trace — likely due to a constraint violation that was not caught by the runtime. Diagnose by narrowing down which op in the sequence produces the divergence.

---

## Quick Reference: Traceable vs Untraceable

| Construct | Traceable? | Notes |
|---|---|---|
| Fixed-shape matmul, softmax, layernorm, ReLU | Yes | Standard transformer ops; shapes from model config |
| Embedding lookup with fixed input shape | Yes | Shape is `[batch, seq=1]` in decode |
| Attention with fixed sequence length | Yes | Decode single-token case |
| Prefill attention with variable S | No | Shape changes per request |
| `ttnn.to_torch()` mid-sequence | No | Host readback mid-dispatch |
| Python `if` on device tensor value | No | Host-side branch on device output |
| `ttnn.argmax` returning a device tensor (no readback) | Yes | Device tensor in, device tensor out |
| Top-k sampling with fixed k | Yes | If output shape is fixed (`[1, k]`) |
| Top-k sampling with data-dependent output | No | Variable number of qualifying candidates |
| KV-cache read/write with fixed cache size | Yes | Cache buffer address is fixed |
| Dynamic positional index as Python int argument | No* | Kernel argument changes each step |
| Dynamic positional index as device tensor | Yes* | Buffer address fixed; value updated in-place |
| Model branching on config flag (resolved at import) | Yes | Branch is deterministic at capture time |
| Model branching on device-computed value | No | Branch outcome varies per step |

\* See the pre-allocated device tensor pattern in Step 3 above.

---

**Next:** [Chapter 4 — When to Use Trace](../ch4_when_to_use_trace/index.md)
