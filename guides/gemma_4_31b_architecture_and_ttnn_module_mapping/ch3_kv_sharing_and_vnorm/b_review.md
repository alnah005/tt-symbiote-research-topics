# Agent B Review: Chapter 3

## Pass 1

No feedback — chapter approved.

All key claims were verified against the HuggingFace `transformers` source code for `Gemma4TextAttention` and `Gemma4RMSNorm` (from `modeling_gemma4.py` on the `main` branch):

- **K=V activation condition.** `self.use_alternative_attention = config.attention_k_eq_v and not self.is_sliding` matches the source exactly.
- **v_proj set to None for global layers.** The conditional `self.v_proj = nn.Linear(...) if not self.use_alternative_attention else None` and the forward-pass fallback `value_states = ... if self.v_proj is not None else key_states` both match.
- **V-norm with_scale=False.** The `Gemma4RMSNorm` class implementation (constructor, `_norm`, `forward`) reproduced in `vnorm_implementation.md` is character-accurate against the source. The `with_scale=False` instantiation for `v_norm` and `with_scale=True` (default) for `k_norm` and `q_norm` are correct.
- **Divergent post-processing order.** The chapter's stated order (k_proj -> view -> k_norm -> RoPE -> transpose for K; same shared tensor -> v_norm -> transpose for V) matches the HuggingFace forward method exactly, including `unsqueeze_dim=2` for `apply_rotary_pos_emb`.
- **Mathematical formulation.** The V-norm formula `v / sqrt(mean(v^2) + eps)` correctly reflects the code `mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps; return hidden_states * torch.pow(mean_squared, -0.5)`.
- **Tensor shapes and parameter counts.** The K projection shape `[5376, 2048]` (4 KV heads x 512 head_dim), the per-layer savings of 11,010,048 parameters, and the total ~220 MB savings across 10 global layers are arithmetically correct.
- **V-norm presence in all layers.** Confirmed: `self.v_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps, with_scale=False)` is instantiated unconditionally in `__init__`, active for all 60 layers.
