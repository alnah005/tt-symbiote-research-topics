## Pass 1

1. **`thinking_preservation_mechanism.md`, line 74** — The KV cache tensor shape is stated as `[B, T, num_heads, head_dim]`. This is wrong for a KV cache. Keys and values are computed using the key/value heads, not the query heads. The correct shape is `[B, T, num_key_value_heads, head_dim]`, where `num_key_value_heads = 2` (per the model config), not `num_attention_heads = 16`. An implementer following the stated shape would allocate a KV cache 8× too large (16 heads × 256 head_dim instead of 2 heads × 256 head_dim per layer). Fix: replace `num_heads` with `num_key_value_heads` in the shape and clarify the value is 2.

## Pass 2

**No feedback — chapter approved.**
