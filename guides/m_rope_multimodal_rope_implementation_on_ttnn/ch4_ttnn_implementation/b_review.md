## B Feedback — Pass 1

1. **`existing_ttnn_rope_gap_analysis.md`, Section 4, DRAM placement bullet** — States "The same `[max_seq_len, 64]` half-table works for both standard RoPE and M-RoPE." The half-table shape is `[max_seq_len, 32]` (= `[max_seq_len, rotary_dim/2]`). Writing `64` (= `rotary_dim`) for a table labeled "half" is factually wrong.

2. **`extension_approach.md`, Section 3, forward signature comments** — `q` and `k` annotated as `# [..., rotary_dim]` implies shape `[..., 64]`. The inputs are full Q/K vectors of shape `[..., head_dim]` = `[..., 128]`. The partial rotation is applied internally; inputs must include the pass-through dims.

3. **`existing_ttnn_rope_gap_analysis.md`, Section 1, rotate-half variable name** — Variable named `q_pass` holds the rotated recombination `q1*sin + q2*cos`, not a passthrough. "Pass-through" in this chapter specifically means the last 64 of 128 dims (outside `rotary_dim`). Using `q_pass` for a rotated half creates a direct terminology conflict.

4. **`existing_ttnn_rope_gap_analysis.md`, Section 3 (Gap 2) pseudocode** — `cos_table[position_ids[0], :s_t]` presents the gather and column-slice as a single indexing operation, misrepresenting the two-step nature of the M-RoPE access. The actual operation is (1) full-row gather via `ttnn.embedding`, then (2) column slice. The pseudocode obscures this distinction.

## B Feedback Application Log — Pass 1

- Fix 1: Changed `[max_seq_len, 64]` to `[max_seq_len, 32]` in DRAM placement bullet in `existing_ttnn_rope_gap_analysis.md` Section 4.
- Fix 2: Changed `# [..., rotary_dim]` to `# [..., head_dim]` for q and k in `extension_approach.md` Section 3 forward signature.
- Fix 3: Renamed `q_pass` → `q_rot2` (and `q_rot` → `q_rot1`) in rotate-half block in `existing_ttnn_rope_gap_analysis.md` Section 1. Added clarifying note that both halves are rotated; the true pass-through dims are separate.
- Fix 4: Updated Section 3 (Gap 2) pseudocode in `existing_ttnn_rope_gap_analysis.md` to show the two-step gather-then-slice using `ttnn.embedding`, with intermediate variables separating full-row gather from column-slice.

## B Feedback — Pass 2

1. **`gather_operation_on_ttnn.md`, Section 2** — States "Index tensor (position IDs): `[batch, seq_len]`" as an unqualified description. The full M-RoPE position_ids tensor is `[3, batch, seq_len]`; the `[batch, seq_len]` shape is only the per-axis slice passed to each individual `ttnn.embedding` call. A reader consulting Section 2 in isolation would conclude the full position_ids input is `[batch, seq_len]`, directly contradicting the ground-truth shape.

## B Feedback Application Log — Pass 2

- Fix 1: Clarified in Section 2 of `gather_operation_on_ttnn.md` that `[batch, seq_len]` is the per-axis index shape — one of three slices from the `[3, batch, seq_len]` M-RoPE position_ids tensor.
