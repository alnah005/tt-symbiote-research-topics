## B Feedback — Pass 1

1. **`speculative_decode_loop_integration.md`, Step 3 comments** — The inline comments read "verify_logits[..., 0, :] is the backbone's distribution at position t+1" and "verify_logits[..., 1, :] is the backbone's distribution at position t+2." When the backbone processes the 2-token verification sequence [x_t1, x_hat_t2], output index 0 is the distribution for what comes AFTER x_t1 — i.e., for position t+2 (the predicted token at t+2, which is x_hat_t2). Output index 1 is for what comes after x_hat_t2 — i.e., position t+3. The comments mislabel these as "at t+1" and "at t+2," which could lead a reader to extract the wrong index when implementing the acceptance check. The code itself is correct (uses verify_logits[..., 0, :][x_hat_t2] to evaluate p(x_hat_t2), which is correct), but the comments describe the wrong positions. Fix: change "backbone's distribution at position t+1" to "backbone's distribution for position t+2 (token after x_t1)" and "at position t+2" to "for position t+3 (token after x_hat_t2)".

2. **`memory_placement_for_mtp.md`, Key Finding** — States "At a typical 50–100 ms backbone decode step latency, this is less than 2% overhead." The Chapter 4 cost model (`throughput_analysis_on_tt_hardware.md`) establishes C_decode ≈ 243 ms for P150 (single-chip, bandwidth-bound). Using that figure: 1.06 ms / 243 ms ≈ 0.4%, not "less than 2%". The 50–100 ms range is consistent with T3K (4 chips, ~4× bandwidth) but the file does not specify the hardware. This inconsistency with the Ch04 baseline may confuse readers comparing the two analyses. Fix: either reference Ch04's P150 figure (1.06 ms / 243 ms ≈ 0.4%) or explicitly state the target hardware (e.g., "on T3K with reduced per-chip decode latency") and add the numeric estimate.

## B Feedback Application Log — Pass 1

- Fix 1: Changed Step 3 inline comments in `speculative_decode_loop_integration.md`:
  - "backbone's distribution at position t+1" → "backbone's distribution for position t+2 (the token after x_t1)"
  - "backbone's distribution at position t+2" → "backbone's distribution for position t+3 (the token after x_hat_t2)"
- Fix 2: Updated Key Finding in `memory_placement_for_mtp.md` to use Ch04's P150 estimate (243 ms): changed "At a typical 50–100 ms backbone decode step latency, this is less than 2% overhead" to "At the Ch04 P150 baseline of C_decode ≈ 243 ms: 1.06 ms / 243 ms ≈ 0.4% overhead — consistent with the Ch04 assessment that MTP head cost ≈ 0 in the BW-bound regime."

## B Feedback — Pass 2

No feedback — chapter approved.
