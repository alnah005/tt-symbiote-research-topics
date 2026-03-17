# Agent B Review — Chapter 1: MoE Architecture Fundamentals — Pass 1

**No feedback — chapter approved.**

Rationale (not for downstream readers — internal audit trail only):

All formulas were verified:

- Capacity factor formula `expert_capacity = C * T * top_k / num_experts` and the derived `sparsity_ratio = 1 - 1/C` are algebraically correct; the tile-ceiling caveat is explicitly noted.
- Switch Transformer load balancing loss `num_experts * sum(f_i * P_i) = 1.0` at uniform routing is correct; the omitted alpha coefficient is explicitly flagged inline.
- Mixtral 8x7B active parameter decomposition (2 × 32 layers × 176M per SwiGLU expert ≈ 11.3B + ~1.5B non-MoE ≈ 12.8B) is consistent with the published 12.9B figure (arXiv:2401.04088); the ~0.1B rounding gap is immaterial.
- Per-expert SwiGLU parameter count (3 × 4096 × 14336 ≈ 176M) checks out.
- Naive loop dispatch counts (3 × 8 × 32 = 768 for Mixtral 8x7B; 3 × 8 × 56 = 1,344 for 8x22B) are arithmetically correct.
- DeepSeek-MoE 16B and DeepSeek-V2 parameters are consistent with their respective arXiv papers.
- The zero-contribution convention for dropped tokens is defined precisely and applied consistently across all files.
- The `sparsity_ratio = 1 - 1/C` derivation is correct under the stated assumptions (no drops, pre-rounding formula); deviations from tile-ceiling rounding are noted.
- Switch Transformer CF = 1.0 for top-1 routing is correctly attributed to Section 2.1 of arXiv:2101.03961.

No item met the threshold of (a) producing a wrong numerical answer, (b) causing an incorrect implementation, or (c) materially misleading a reader into a wrong conceptual model.

---

# Agent B Review — Chapter 1: MoE Architecture Fundamentals — Pass 2

1. **`router_weight.T` is a broken code example that will not run (`routing_and_sparsity.md`, `compute_routing` function, line `logits = flat_x @ router_weight.T`).** `router_weight` is defined two lines above the function as `torch.nn.Linear(d_model, num_experts, bias=False)` — a Python module object. Calling `.T` on an `nn.Linear` instance raises an `AttributeError`; you cannot matrix-multiply a tensor with a module. A reader implementing this verbatim will get a runtime crash. The correct expression is either `router_weight(flat_x)` (call the module) or `flat_x @ router_weight.weight.T` (access the underlying weight tensor). Because this is presented as a concrete, runnable `compute_routing` function — not a pseudocode sketch — the broken call is an implementation error, not a style issue.

2. **Switch Transformer CF=1.0 default claim is factually incorrect (`moe_overview.md`, line 80).** The text states "Switch Transformer typically uses `CF = 1.0` for top-1 routing (Fedus et al., arXiv:2101.03961, Section 2.1)." Section 2.1 of the Switch Transformer paper introduces the capacity factor and uses `CF = 1.25` as the operational default for Switch-Base (Table 5 and surrounding text); `CF = 1.0` is discussed only as the lower bound, and the paper explicitly warns that it causes frequent token dropping. Attributing `CF = 1.0` as Switch Transformer's typical choice is wrong and will mislead readers who tune capacity factors by reference to this model family.

---

# Agent B Review — Chapter 1: MoE Architecture Fundamentals — Pass 3

**No feedback — chapter approved.**

Rationale (not for downstream readers — internal audit trail only):

Both Pass 2 fixes were verified as correctly applied:

- `routing_and_sparsity.md` `compute_routing`: `logits = router_weight(flat_x)` — calling `nn.Linear` as a callable is correct; the prior `flat_x @ router_weight.T` AttributeError is resolved.
- `moe_overview.md` CF note: now correctly states `CF = 1.25` as Switch Transformer's operational default and characterizes `CF = 1.0` as the theoretical minimum that causes drops — consistent with arXiv:2101.03961 Table 5.

All other content re-checked:

- Dispatch counts: 3 × 8 = 24 (unfused SwiGLU, Mixtral), 2 × 8 = 16 (fused gate+up), 32 × 24 = 768 (Mixtral 8x7B forward pass), 56 × 24 = 1,344 (Mixtral 8x22B) — all arithmetically correct.
- Sparsity ratio derivation `1 - 1/C` and the decode-mode example (T=1, top_k=2, num_experts=8, expert_capacity=32 → 1 - 2/256 ≈ 0.992) — correct.
- DeepSeek-MoE 16B active fraction (6/64 = 9.375%) and DeepSeek-V2 (6/160 = 3.75%) — correct per respective arXiv papers.
- Switch Transformer load balancing loss: at uniform routing N × Σ(f_i × P_i) = N × N × (1/N)² = 1.0 before alpha — correct; alpha omission is explicitly flagged inline.
- Mixtral 8x7B active parameter decomposition (2 × 32 × 176M ≈ 11.3B + ~1.5B ≈ 12.8B) consistent with arXiv:2401.04088.
- Per-expert SwiGLU weight count (3 × 4096 × 14,336 ≈ 176M) — correct.
- Zero-contribution dropped-token convention is defined precisely and applied consistently across all files.

No item met the threshold of (a) producing a wrong numerical answer, (b) causing an incorrect implementation, or (c) materially misleading a reader into a wrong conceptual model.
