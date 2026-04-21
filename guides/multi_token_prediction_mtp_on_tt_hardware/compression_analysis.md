## Guide-Level C Analysis — Pass 1

No fixes applied from guide-level B pass 1 (guide approved with no feedback).

Cross-chapter consistency verified at guide level:
- H=4096 thread: Ch02 weight inventory → Ch04 C_decode ≈ 243 ms derivation → Ch05 memory placement (1.06 ms / 243 ms ≈ 0.4%) — correct chain
- labels=None gate: Ch03 HuggingFace analysis → Ch05 use_mtp flag and scope note — correct
- No-resampling constraint: Ch04 derivation (resampling collapses E=2, speedup=1.0) → Ch05 CRITICAL blockquote — correct and consistent
- C_decode ≈ 243 ms (P150): Ch04 throughput analysis → Ch05 (fixed from 50-100 ms in chapter B pass 1) — correct after fix
- verify_logits position comment: Ch04 derivation → Ch05 (fixed "for position t+2" comment in chapter B pass 1) — correct after fix
- Speedup < 1 at batch=1 BW-bound: Ch04 key finding → Ch05 testing/validation framing — consistent; guide does not overclaim MTP speedup
- MTP KV cache 128 MiB (1.1% of backbone): Ch04 → Ch05 memory_placement_for_mtp.md — correct
- Weight keys model.future_prediction[0].*: Ch01 → Ch02 → Ch03 → Ch05 — consistent throughout

No crucial inaccuracies found at guide level. All chapter-level compression analyses are "Crucial updates: no" for their second pass.

**Crucial updates: no**

---

## Guide-Level C Analysis — Pass 2

**Crucial updates: no**
