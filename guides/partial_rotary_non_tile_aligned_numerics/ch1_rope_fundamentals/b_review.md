# B Review — Pass 1

1. [`tile_alignment_in_ttnn.md`, ~line 31, factual error, fix shape] The cos/sin tensor shape is written as `[batch, num_heads, seq_len, rotary_dim]`. Standard RoPE cos/sin tensors are **not** per-head — they are shared across all heads and have shape `[seq_len, rotary_dim]` (or broadcast-compatible variants such as `[1, 1, seq_len, rotary_dim]`). Listing `num_heads` as an explicit dimension implies a separate cos/sin per head, which is incorrect and would cause a reader to allocate or expect the wrong tensor shape. Fix: replace the shape with `[seq_len, rotary_dim]` (or `[1, 1, seq_len, rotary_dim]` if a 4-D broadcast form is preferred), and add a note that the tensor is broadcast across the head dimension.

# B Review — Pass 2

No feedback — chapter approved.
