# Agent B Review: Chapter 5 — Pass 1

1. **TTNN Porting Note #2 — incorrect description of K-equals-V buffer stage** (`index.md`, TTNN Porting Considerations, item 2): The note says the value path should "share the key buffer (post k_norm, pre-RoPE)." This is factually wrong. In the source (`modeling_gemma4.py` lines 1204-1211), when `v_proj is None`, `value_states` is assigned from `key_states` **before** `k_norm` and RoPE are applied to `key_states`. The value tensor then goes through `v_norm` independently. The correct description is "pre-k_norm, pre-RoPE" (i.e., the raw `k_proj` output). Fix the note to say the value path shares the key buffer at the k_proj output stage, before k_norm and RoPE.

No other factual, coherence, or structural issues found.

# Agent B Review: Chapter 5 — Pass 2

The k_eq_v fix from Pass 1 has been correctly applied. Both the K-Equals-V Mode section (Section 5.3) and TTNN Porting Consideration #2 now correctly state "pre-k_norm, pre-RoPE" matching the source at `modeling_gemma4.py` line 1205.

No feedback — chapter approved.
