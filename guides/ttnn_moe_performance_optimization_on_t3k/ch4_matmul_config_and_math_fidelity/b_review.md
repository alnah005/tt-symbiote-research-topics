# B Review — Chapter 4 — Pass 1

1. **math_fidelity_evaluation.md — reference computation omits the w3 (up projection) and the element-wise gate×up multiply, so all accuracy metrics are measured against the wrong reference for GLM-4-MoE.**

   GLM-4-MoE experts use SwiGLU: the hidden activation fed into w2 is `silu(x @ w1) * (x @ w3)`. The Step 1 harness computes:

   ```python
   ref_gate_up = (x_f32 @ w1_f32)
   ref_down = (torch.silu(ref_gate_up) @ w2_f32)
   ```

   This omits `w3` entirely and never forms the element-wise product. The `evaluate()` function repeats the same structure. Cosine similarity and max absolute error are therefore measured against a reference that does not correspond to any computation the model actually performs. A reader running this harness could pass or fail the go/no-go cosine threshold (> 0.9999) based on agreement with the wrong quantity, and could incorrectly adopt or reject LoFi as a result.

   Fix: introduce `w3_f32 = torch.randn(hidden_size, intermediate_size)`, compute the correct reference as `ref_act = torch.silu(x_f32 @ w1_f32) * (x_f32 @ w3_f32)` and `ref_down = ref_act @ w2_f32`, and mirror the same gate×up multiply on the device-side path inside `evaluate()` before passing to the w2 matmul.

# B Review — Chapter 4 — Pass 2

No feedback — chapter approved.

# B Review — Chapter 4 — Pass 3

No feedback — chapter approved.
