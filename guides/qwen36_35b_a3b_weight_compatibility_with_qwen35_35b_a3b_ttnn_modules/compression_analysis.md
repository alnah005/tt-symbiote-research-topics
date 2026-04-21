## Guide-Level C Analysis — Pass 1

No fixes applied from guide-level B pass 1 (guide approved with no feedback).

Cross-chapter consistency verified at guide level:
- vocab_size = 151,936 thread: Ch1 → Ch4 (out-of-range) → Ch5 (embed_tokens shape) → Ch6 (CI assertion) — correct and consistent
- partial_rotary_factor = 0.25 → rotary_dim = 32: Ch3 → Ch6 defensive fallback — correct
- H=4096 hidden dim: Ch5 MTP block (verified against MTP guide Ch02/Ch04) → Ch6 — correct
- MTP gate training-only: Ch3→Ch5 (architecture) → Ch6 (verdict, Steps 3+4) — correct
- filter_mtp_keys prefix: Ch5 → Ch6 Step 3 — correct and sufficient (no other Qwen3.6 keys share "model.future_prediction" prefix)
- bos_token_id=248044 > 151936 risk chain: Ch4 analysis → Ch6 medium risk + Steps 4+5 mitigations — correct
- Weight tensor shapes identical (backbone): Ch2 → Ch6 "none" risk row — correct
- 86 ms/token target: Ch4 (Qwen3.5 baseline) → Ch6 Step 7 — correct
- 7 migration steps map correctly to 4 master table items (bos_token_id gets 2 steps: suppress + CI check) — correct
- Overall verdict "compatible, low-to-medium effort" justified by per-chapter findings — correct

No crucial inaccuracies found at guide level. Chapter-level compression analyses are all "Crucial updates: no" for their second pass.

**Crucial updates: no**

---

## Guide-Level C Analysis — Pass 2

**Crucial updates: no**
