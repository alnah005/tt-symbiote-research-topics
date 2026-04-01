# Agent B Review — Chapter 4: Decoder Block — Pass 1

## Finding 1 — `forward_signature.md`: hidden_dim derivation is internally inconsistent and contradicts `mlp_dispatch.md`

**File:** `forward_signature.md`, section "L1 CB Clash Workaround for hidden\_dim=17408"

**Claim:**
> The 27B model has `hidden_size = 7168` but the MLP intermediate dimension is
> $d_{\text{ff}} = 4 \times 7168 \times \tfrac{2}{3} \approx 19{,}131$, rounded up to a
> multiple of 256, giving `hidden_dim = 17408`

**Problem — the arithmetic does not produce 17408:**

$4 \times 7168 \times \tfrac{2}{3} = \tfrac{28672}{3} \times 2 \approx 19114.7$

Rounding 19114.7 up to the nearest multiple of 256:
$\lceil 19114.7 / 256 \rceil = 75$, so $75 \times 256 = 19200$

Rounding down gives $74 \times 256 = 18944$.

Neither rounding produces 17408. The value 17408 equals $68 \times 256$ and does not follow from the formula shown.

**Contradiction with `mlp_dispatch.md`:** That file states the 27B intermediate dimension is $d_{\text{ff}} = 18944$, which is consistent with rounding 19114.7 down to the nearest multiple of 256. The two files name different values (17408 vs 18944) for the same model's MLP intermediate dimension without explanation.

**Impact:** A reader implementing the model who relies on the `forward_signature.md` derivation to determine the intermediate dimension will compute the wrong tensor shape. The value 18944 stated in `mlp_dispatch.md` is consistent with the SwiGLU formula; the value 17408 in `forward_signature.md` is not.

**Fix:** Either (a) replace 17408 with 18944 and correct the rounding step in `forward_signature.md`, or (b) if 17408 refers to a different quantity (e.g. a hardware-aligned slice of the intermediate buffer rather than the full $d_{\text{ff}}$), explain what quantity it actually measures and remove the misleading derivation from $4 \times 7168 \times \tfrac{2}{3}$.

---

No other correctness issues were found. The layer counts (48 DeltaNet + 16 full-attention for 27B; 30 DeltaNet + 10 full-attention for A3B), constructor signatures, `hasattr` dispatch logic, MoE formula parameters (256 experts, top-8), and residual data-flow equations all match the source files.

---

# Agent B Review — Chapter 4: Decoder Block — Pass 2

## Finding 1 — `block_structure.md` vs `forward_signature.md`: Contradictory shape for the DeltaNet recurrent state tensor

**Files:** `block_structure.md` line 68; `forward_signature.md` line 65

**Claim in `block_structure.md`:**
> allocates the recurrent state tensor $S \in \mathbb{R}^{B \times H \times d_k \times d_v}$

**Claim in `forward_signature.md`:**
> the recurrence accumulates a key-value memory matrix $S \in \mathbb{R}^{H \times d_k \times d_v}$

**Problem:** The two files disagree on whether the state tensor carries the batch dimension B. Both describe the same tensor initialized by `self.attention.initialize_states(batch_size=batch)`. Exactly one of the two shapes is wrong. A reader implementing or inspecting the recurrent state will get conflicting information depending on which file they read.

**Fix:** Decide the canonical shape and make both files consistent. If `initialize_states` allocates a batched tensor `[B, H, d_k, d_v]`, the shape in `forward_signature.md` must include B. If it allocates `[H, d_k, d_v]` and replicates or broadcasts across B at runtime, the shape in `block_structure.md` must drop B.

---

## Finding 2 — `forward_signature.md`: `page_table` and `kv_cache` are described as "used by GatedAttention" but the block never forwards them

**File:** `forward_signature.md`, lines 62–63

**Claim:**
> `page_table` and `kv_cache` are used by `GatedAttention` for paged KV-cache lookups.

**Problem:** The verified source (`qwen35_decoder.py`, line 167) shows the GatedAttention call as:

```python
attn_out = self.attention.forward(
    attn_in,
    current_pos=current_pos,
    rot_mats=rot_mats_global,
    mode=mode,
)
```

Neither `page_table` nor `kv_cache` is forwarded. Both arguments are accepted by `DeltaNetDecoderBlock.forward` but silently dropped — regardless of whether the inner attention is DeltaNet or GatedAttention. The chapter's claim that GatedAttention uses them for paged KV-cache lookups is incorrect for this block's implementation; they are no-ops at the `DeltaNetDecoderBlock` level.

**Impact:** A reader relying on this description to understand paged-attention behavior in A3B hybrid layers will incorrectly believe paged KV-cache is active for the full-attention layers routed through `DeltaNetDecoderBlock`.

**Fix:** Change the description to state that `page_table` and `kv_cache` are accepted for signature parity but are not forwarded by `DeltaNetDecoderBlock` in the current implementation.

---

No further correctness issues found in Pass 2. The constructor signatures, `hasattr` sentinel logic, residual data-flow formulas, layer counts, and MoE formula parameters all verify correctly against the source.

---

# Agent B Review — Chapter 4: Decoder Block — Pass 3

## Finding 1 — `mlp_dispatch.md`: MoE expert weight keys are wrong

**File:** `mlp_dispatch.md`, section "State Dict Prefix Isolation"

**Claim:**
```
model.layers.{i}.feed_forward.experts.{e}.w1.weight  (gate+up fused)
model.layers.{i}.feed_forward.experts.{e}.w2.weight  (down)
```

**Problem:** The source (`qwen35_moe.py` lines 74–75) accesses expert weights as two packed 3D tensors — not per-expert indexed keys:

```python
raw_gate_up = state_dict[f"{prefix}.experts.gate_up_proj"]  # [256, 2*intermediate, hidden]
raw_down    = state_dict[f"{prefix}.experts.down_proj"]     # [256, hidden, intermediate]
```

This is also confirmed by `qwen35_utils.py` line 26–27: `"experts.gate_up_proj [256,1024,2048], experts.down_proj [256,2048,512]"`. There are no per-expert keys of the form `experts.{e}.w1.weight`. A developer using the key names as written would fail to find any tensor in the state dict.

**Fix:** Replace the per-expert key pattern with the actual fused keys: `feed_forward.experts.gate_up_proj` and `feed_forward.experts.down_proj`.

---

## Finding 2 — `mlp_dispatch.md`: Router weight key is wrong

**File:** `mlp_dispatch.md`, section "State Dict Prefix Isolation"

**Claim:**
```
model.layers.{i}.feed_forward.router.weight
```

**Problem:** The source (`qwen35_moe.py` line 63) accesses the router as:

```python
router_w = state_dict[f"{prefix}.gate.weight"]
```

The key segment is `.gate.weight`, not `.router.weight`. Using the name stated in the guide to look up the router weight in the state dict will raise a `KeyError`.

**Fix:** Replace `feed_forward.router.weight` with `feed_forward.gate.weight`.

---

No other correctness issues found in Pass 3. The MoE formula parameters (256 experts, top-8, expert $d_{\text{ff}} = 512$), layer counts, constructor signatures, residual data-flow, and `hasattr` dispatch all verify correctly against the source.

---

# Agent B Review — Chapter 4: Decoder Block — Pass 4

## Finding 1 — `mlp_dispatch.md`: Cache size estimate of ~12.8 GB is wrong

**File:** `mlp_dispatch.md`, section "State Dict Prefix Isolation"

**Claim:**
> 256 experts × 40 layers × 2 matrices per expert at `bfp4` consume approximately 12.8 GB of cache files

**Problem:** The arithmetic does not produce 12.8 GB. From `qwen35_moe.py`:

- `gate_up_proj` per expert after transpose: shape `[1, 1, 2048, 1024]` → 2,097,152 elements
- `down_proj` per expert after transpose: shape `[1, 1, 512, 2048]` → 1,048,576 elements
- bfp4 = 4 bits = 0.5 bytes per element

Per-expert cost: (2,097,152 + 1,048,576) × 0.5 = 1,572,864 bytes ≈ 1.5 MB

Total: 256 × 40 × 1.5 MB = 15,360 MB ≈ 15 GB

The stated 12.8 GB is wrong by approximately 2.2 GB (17% low). A reader using this figure to plan DRAM capacity will underestimate storage requirements.

**Fix:** Replace "approximately 12.8 GB" with "approximately 15 GB".

---

## Finding 2 — `mlp_dispatch.md`: MoE formula omits the sigmoid gate on the shared expert

**File:** `mlp_dispatch.md`, section "MoE Substitution: 35B-A3B Layers"

**Claim:**
$$\text{MoE}(x) = \text{SharedExpert}(x) + \sum_{e \in \text{TopK}(r(x),\,8)} w_e \cdot \text{Expert}_e(x)$$

**Problem:** The source (`qwen35_moe.py`, lines 157–159) applies a learned sigmoid gate to the shared expert output before accumulation:

```python
gate = ttnn.linear(x, self.shared_gate_weight_tt, memory_config=L1)
gate = ttnn.sigmoid(gate, memory_config=L1)
shared_out = ttnn.mul(shared_out, gate, memory_config=L1)
```

The actual computation is:

$$\text{MoE}(x) = \sigma(x W_{\text{sg}}) \cdot \text{SharedExpert}(x) + \sum_{e \in \text{TopK}(r(x),\,8)} w_e \cdot \text{Expert}_e(x)$$

where $W_{\text{sg}}$ is `shared_expert_gate.weight` and $\sigma$ is the sigmoid function. The formula as written in the guide treats the shared expert as unscaled, which would produce incorrect output if used as a reference implementation.

**Fix:** Add the $\sigma(x W_{\text{sg}})$ factor to the shared expert term in the formula.

---

No other correctness issues found in Pass 4. The layer counts, constructor signatures, `hasattr` dispatch logic, residual data-flow formulas, router logit count (256), top-k value (8), expert intermediate dimension (512), and state dict key names all verify correctly against the source.

---

# Agent B Review — Chapter 4: Decoder Block — Pass 5

No feedback — chapter approved.

All verifiable claims were checked against `qwen35_decoder.py` and `qwen35_moe.py`:

- Constructor signature: exact match.
- `hasattr(self.attention, "initialize_states")` sentinel and both dispatch branches: exact match.
- Layer counts (48 DeltaNet + 16 full-attention for 27B; 30 DeltaNet + 10 full-attention for 40-layer A3B): consistent with source comments and build-loop structure.
- Residual data-flow formulas ($x_1 = x_0 + a$, $\text{out} = x_1 + f$): match source lines 171–193.
- `intermediate_size = 17408` described as read from HuggingFace config (not derived): no contradicting formula present in current text.
- MoE formula including $\sigma(x W_{\text{sg}})$ on the shared expert: matches source lines 157–159.
- State dict keys `experts.gate_up_proj` [256, 1024, 2048] and `experts.down_proj` [256, 2048, 512]: match source lines 74–75 (with `intermediate=512`, `hidden=2048`).
- Router key `gate.weight`: matches source line 63.
- Cache size estimate ~15 GB: consistent with per-expert calculation (256 × 40 × 1.5 MB ≈ 15 GB).
- `Qwen35MoE` constructor accepts all nine kwargs listed in the guide; extra optional `state_dict_prefix=None` does not make the parity claim wrong for the calling convention described.

---

# Agent B Review — Chapter 4: Decoder Block — Pass 6

No feedback — chapter approved.

All verifiable claims were rechecked:

- `hidden_size = 5120` and `intermediate_size = 17408` for the 27B model (`forward_signature.md`): stated as read from HuggingFace config, no conflicting derivation present.
- $W_1, W_3 \in \mathbb{R}^{5120 \times 17408}$ and $W_2 \in \mathbb{R}^{17408 \times 5120}$ (`forward_signature.md`): consistent with the stated dimensions.
- A3B layer counts (30 DeltaNet + 10 full-attention = 40 total) and 27B counts (48 DeltaNet + 16 full-attention = 64 total): internally consistent within `block_structure.md`.
- MoE expert weight shapes [256, 1024, 2048] and [256, 2048, 512] (`mlp_dispatch.md`): consistent with expert intermediate = 512 and model hidden = 2048 for the A3B experts.
- MoE formula with $\sigma(x W_{\text{sg}})$ on the shared expert, 256 logits, top-8: all match `mlp_dispatch.md` current text.
- Cache size estimate ~15 GB (256 × 40 × 1.5 MB): arithmetic confirmed correct.
- State dict keys `experts.gate_up_proj`, `experts.down_proj`, `gate.weight`, `shared_expert.*`: consistent with `mlp_dispatch.md` current text.
- Residual data-flow formulas ($x_1 = x_0 + a$, $\text{out} = x_1 + f$): match code in `forward_signature.md`.
- `hasattr(self.attention, "initialize_states")` dispatch and both branch call signatures: consistent across `block_structure.md` and `forward_signature.md`.
- No cross-file contradictions on any numeric value found in this pass.
