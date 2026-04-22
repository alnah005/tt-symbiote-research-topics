# Agent B Review — Chapter 4 — Pass 1

## Issues Found: 0

All numerical claims, derivations, table values, and technical assertions in the four Chapter 4 files were checked against the verified facts. Every item checked out:

- `num_attention_heads=12`, `num_key_value_heads=2`, `gcd(12,2)=2`, valid TP={1,2}: correct in `gqa_tp_constraint.md` and `index.md`.
- TP failure table (TP=3→0.667, TP=4→0.5, TP=8→0.25): correct in `gqa_tp_constraint.md`.
- Qwen2.5-VL-7B comparison row (28 heads, 4 KV heads, gcd=4, max TP=4, 7:1 ratio): correct.
- LM head math at TP=1: `ceil(151936/2048)=75`. Correct.
- LM head math at TP=2: `ceil(151936/2)=75968` columns per device, `ceil(75968/2048)=38` ops. Correct.
- T3K topology: 1×8 mesh, 8 Wormhole N300 devices, Galaxy interconnect. Correct.
- Submesh lifecycle (open full 1×8 parent first, carve 1×2 or 1×1 submesh, release submesh before parent): correct.
- Idle device count at TP=2 (6 devices held idle): correct.
- Env var defaults and descriptions (`DOTS_T3K_OPEN_FULL_MESH` default=1, `DOTS_LM_HEAD_MAX_COLUMNS_PER_DEVICE` default=2048, `DOTS_MAX_SEQ_LEN` and `DOTS_MAX_SEQ_LEN_WH_LB` both listed with no default): correct.
- Vision token count for 896×1344 image: 1536 tokens. Correct.
- `num_hidden_layers=28` referenced in chunked prefill: correct.
- `max_position_embeddings=131072` cited in chunked prefill overview: correct.

## VERDICT: Approved
