## C Analysis — Pass 1

Two fixes applied from B pass 1:
- `speculative_decode_loop_integration.md` Step 3 comments: "at position t+1/t+2" corrected to "for position t+2/t+3". Verification: when backbone processes [x_t1, x_hat_t2], output index 0 gives the distribution for the token AFTER x_t1 (position t+2); the code `softmax(verify_logits[..., 0, :])[x_hat_t2]` correctly evaluates p(x_hat_t2) — both comment and code are now consistent.
- `memory_placement_for_mtp.md` Key Finding: "50–100 ms backbone decode step latency" replaced with the Ch04 P150 baseline (C_decode ≈ 243 ms); overhead recomputed as 1.06 ms / 243 ms ≈ 0.4% (not "less than 2%").

Verified all key claims across all five files after fixes:
- `TTNNMTPHead` inputs: `backbone_hidden_state` and `x_t1_embedding` both `[batch, 1, H=4096]` — consistent with MTP guide Ch02
- MTP head: dense FFN (not MoE), shares `lm_head` with backbone — consistent with Ch01/Ch03
- Weight key pattern `model.future_prediction[0].*` — consistent with Ch02
- Total new DRAM allocation ~304.6 MiB — consistent with Ch02
- No-resampling constraint on rejection path: `[CRITICAL]` blockquote present and correctly explained — consistent with Ch04 derivation
- KV cache advance: 1 or 2 per cycle; accepted path sets current_token to x_hat_t2, rejected path sets to x_t1 — correct
- MTP KV cache sizing: 2 × 8 × 32768 × 128 × 2 = 128 MiB → 1/94 ≈ 1.1% of backbone KV cache — correct
- MTP head cost ≈ 0.4% of C_decode ≈ 0 in BW-bound regime — now consistent with Ch04 P150 figure
- Throughput benchmark entries all marked `[placeholder — to be filled during bring-up]` — correct

No remaining crucial inaccuracies found.

**Crucial updates: yes** (two fixes above applied)

---

## C Analysis — Pass 2

All crucial inaccuracies from Pass 1 resolved. Verified:
- verify_logits index 0 = "for position t+2" comment consistent with code — correct
- Memory placement latency reference uses Ch04 P150 baseline (243 ms, 0.4%) — correct and consistent
- No-resample constraint preserved in Step 5 — correct
- All placeholder entries in throughput benchmark table — correct
- No further crucial inaccuracies found across all five files

**Crucial updates: no**
