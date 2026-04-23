# B Review — Pass 1

1. [`forward_pass_walkthrough.md`, ~lines 150 and 158, **transpose inconsistency in delta rule recurrence — implementation error**: The file defines S with shape [B, num_v_heads, d_k, d_v] (line 160), meaning S is a [d_k × d_v] matrix per head. The retrieval formula on line 150 is written correctly for this convention: `retrieval = (S_decayed^(h))^T k_tilde_t^(h)` — S^T has shape [d_v, d_k], multiplied by k [d_k] gives [d_v]. But the output formula on line 158, `o_t^(h) = S_t^(h) q_tilde_t^(h)`, drops the transpose: S [d_k, d_v] multiplied by q [d_k] contracts along the d_v axis (semantically wrong) and yields a [d_k]-shaped vector, not a [d_v]-shaped one. Because d_k == d_v == 128 for this model config, the shape check passes silently but the values are wrong — the matmul sums over the wrong axis. A reader implementing the Chapter 2 six-op TTNN decomposition from this formula would produce numerically incorrect attention output. Fix: write `o_t^(h) = S_t^{(h)\top} \tilde{q}_t^{(h)}` (add the transpose), consistent with how retrieval is written.]

2. [`device_state_persistence.md`, ~lines 129 and 153, **wrong MB totals for DRAM estimates**: Line 129 states "30 × 128 KB = 3.84 MB" and line 153 states "30 × 64 KB = 1.92 MB". Both are arithmetic errors caused by dividing KB by 1000 instead of 1024: 30 × 128 KB = 3,840 KB = 3.75 MB (not 3.84 MB) and 30 × 64 KB = 1,920 KB = 1.875 MB (not 1.92 MB). The context is a "negligible DRAM budget" argument, so no implementation decision turns on these numbers, but the stated figures are factually wrong. Fix: replace "3.84 MB" with "3.75 MB" and "1.92 MB" with "1.875 MB".]

# B Review — Pass 2

1. [`forward_pass_walkthrough.md`, line 150, **retrieval formula uses S_decayed instead of S_{t-1} — contradicts domain key**: The retrieval step in the Step 4 recurrence is written as `retrieval = (S_decayed^(h))^T k_tilde_t^(h)`, where `S_decayed = g_t * S_{t-1}`. This expands to `retrieval = g_t * S_{t-1}^T k_tilde_t`, which is factually inconsistent with the domain key update formula `S_t = g_t * S_{t-1} + k_tilde_t ⊗ (β_t * (v_t - S_{t-1}^T k_tilde_t))`. The domain key uses S_{t-1} (the pre-decay state) in the retrieval term, not the decayed state. A reader implementing the Chapter 2 TTNN decomposition from the walkthrough formula would compute a numerically different result (the decay scalar g_t is incorrectly folded into the retrieval inner product). Fix: replace line 150 with `retrieval = (S_{t-1}^{(h)})^\top \tilde{k}_t^{(h)}` and update line 148 to keep `S_decayed` only for the state base, so the five steps read: `S_decayed = g_t * S_{t-1}`; `retrieval = S_{t-1}^T k_tilde`; `error = beta * (v - retrieval)`; `write = k_tilde ⊗ error`; `S_t = S_decayed + write`.

2. [`host_crossing_summary_table.md`, line 41, **wrong binary-to-MB conversion for P2 total bandwidth**: The P2 rationale states "160 KB per layer × 30 layers = 4.8 MB". 160 × 30 = 4,800 KB. Dividing by 1,024 (binary, as required by the domain rules: 1 MB = 1,024 KB) gives 4,800 / 1,024 = 4.6875 MB, not 4.8 MB. The 4.8 MB figure is the result of dividing by 1,000 (decimal), which is incorrect per the stated convention. Fix: replace "4.8 MB" with "≈4.69 MB" (or the exact value 4.6875 MB) in the P2 rationale paragraph.

# B Review — Pass 3 (Change Log)

Changes applied in response to Pass 2:
1. `forward_pass_walkthrough.md` ~line 150: changed retrieval formula from `(S_decayed^(h))^T` to `(S_{t-1}^(h))^T` — retrieval uses pre-decay state per the DeltaNet recurrence
2. `host_crossing_summary_table.md` ~line 41: changed "4.8 MB" to "≈4.69 MB" — 4,800 KB ÷ 1,024 = 4.6875 MB

# B Review — Pass 3

1. [`forward_pass_walkthrough.md`, line 199, **wrong `ttnn.to_torch` count in Step 4 — off by one**: The text states "`TO_TORCH`: five `ttnn.to_torch` calls, each forcing a device sync." The code block immediately above (lines 166–171) shows six `ttnn.to_torch` calls: `q_tilde`, `k_tilde`, `v`, `g_t`, `beta_t`, and `S_prev`. The comments on `g_t` and `beta_t` note they may already be on host from Step 3, but those comments are conditional ("or already on host") and both calls are still present in the shown code; the Step 4 boundary table also lists all six tensors as device→host transfers. "Five" is therefore factually incorrect as written. Fix: replace "five `ttnn.to_torch` calls" with "six `ttnn.to_torch` calls".]

2. [`host_crossing_summary_table.md`, line 29 (P3 rationale for Step 3), **wrong round-trip byte count for Steps 3 decay gate**: The P3 priority description states "128 B round-trip". The same file's P2 rationale uses "round-trip" to mean total bidirectional PCIe traffic (device→host plus host→device: 80 KB + 80 KB = 160 KB). By that same convention, Step 3's round-trip is 128 B device→host (`a_t` 64 B + `b_t` 64 B) plus 128 B host→device (`g_t` 64 B + `beta_t` 64 B) = 256 B total. "128 B round-trip" counts only one direction. Fix: replace "128 B round-trip" with "256 B round-trip".]

# B Review — Pass 4 (Change Log)

Changes applied in response to Pass 3:
1. `forward_pass_walkthrough.md` ~line 199: changed "five `ttnn.to_torch` calls" to "six" — code block shows 6 calls (q_tilde, k_tilde, v, g_t, beta_t, S_prev)
2. `host_crossing_summary_table.md` ~line 29: changed "128 B round-trip" to "256 B round-trip" — bidirectional: 128 B device→host + 128 B host→device

# B Review — Pass 4

1. [`forward_pass_walkthrough.md`, lines 129–132 and 136, and `host_crossing_summary_table.md`, line 29 — **Step 3 host→device rows and FROM_TORCH classification are factually wrong**: The Step 3 boundary table (forward_pass_walkthrough.md lines 131–132) lists `g_t` [1,1,32] = 64 B and `beta_t` [1,1,32] = 64 B as "host → device" transfers, and the summary table (host_crossing_summary_table.md line 29) lists `FROM_TORCH` as a trace-break mechanism for Step 3 and lists `g_t` and `beta_t` under "Tensors written to device". However, the Step 4 code block in `forward_pass_walkthrough.md` (lines 169–170) explicitly annotates both tensors with "(or already on host from Step 3)" — meaning in the current implementation `g_t` and `beta_t` stay as host-side PyTorch tensors and flow directly into the `recurrent_gated_delta_rule` host kernel. They are never transferred back to the Wormhole device via `ttnn.from_torch` during Step 3; the `FROM_TORCH` (if any) for these tensors occurs later in the target on-device implementation, not in the current one. Showing them as "host → device" in the Step 3 current-implementation table and listing `FROM_TORCH` as a current trace-break mechanism for Step 3 misrepresents the actual code path. Fix: (a) remove the two "host → device" rows for `g_t` and `beta_t` from the Step 3 boundary table in `forward_pass_walkthrough.md` (or add a note that in the current implementation these stay on host and feed Step 4 directly); (b) remove `FROM_TORCH` from the Step 3 entry in the summary table in `host_crossing_summary_table.md`; (c) update the Step 3 round-trip figure accordingly — the actual one-way transfer is 128 B (device→host only: `a_t` + `b_t`), not 256 B bidirectional, since `g_t` and `beta_t` never return to device in the current flow.]

# B Review — Pass 5 (Change Log)

Changes applied in response to Pass 4:
1. `forward_pass_walkthrough.md` Step 3 table: removed g_t and beta_t "host→device" rows — they stay as host tensors, no from_torch call
2. `host_crossing_summary_table.md` Step 3: removed FROM_TORCH classification; changed "256 B round-trip" to "128 B device→host" (a_t + b_t only; g_t and beta_t remain on host)

# B Review — Pass 5

No feedback — chapter approved.
