## B Feedback — Pass 1

No feedback — chapter approved.

All claims verified:
- MTP gate conditions (`labels is not None AND self.training is True`) — consistent with MTP guide Ch05
- 9 weight key patterns under `model.future_prediction[0].*` — consistent with MTP guide Ch02
- ~160M params / ~304.6 MiB BF16 — consistent with MTP guide Ch02 (304.6 MiB × 1048576 / 2 ≈ 159.7M params ✓)
- H=4096, `num_key_value_heads=8`, `head_dim=128` — consistent with MTP guide Ch04 (C_decode ≈ 243 ms derivation)
- `lm_head` tied to `embed_tokens.weight`, not duplicated — consistent with established MTP architecture
- filter_mtp_keys prefix `"model.future_prediction"` covers all 9 key patterns — correct
- Validation assertion checks `"future_prediction"` in loaded TTNN keys — actionable and correct
- `vocab_size = 151,936` in lm_head / embed_tokens shape — consistent with Ch4 (bos_token_id=248044 > 151936)
