# B Review — Chapter 4: Pass 1

1. **File:** `latency_sensitive_workloads.md`, ~line 31
   **Issue:** The encoding time is stated as "20–40 us" ("the encoding time (20–40 us) exceeds the execution time for short device kernels"), but the guide's established per-op dispatch overhead range is 17–63 us everywhere else (index.md line 3, latency_sensitive_workloads.md lines 29 and 33, index.md line 23). A reader calculating the crossover point at which async mode alone cannot hide encoding will use 20–40 us instead of the correct 17–63 us range, producing an incorrect threshold. The argument itself holds for either range (both exceed the 10 us kernel time in the example), but the stated range is factually inconsistent with the rest of the chapter.
   **Fix:** Replace "20–40 us" with "17–63 us" to match the established per-op overhead range.

2. **File:** `index.md`, ~line 13; `when_not_to_trace.md`, structure
   **Issue:** index.md states "three primary disqualifying conditions" for trace, which matches the three numbered conditions in when_not_to_trace.md. However, index.md line 83 explicitly lists four constraint categories from Chapter 3: "dynamic shapes, data-dependent dispatch, mid-loop host readbacks, and self-configuring ops." Self-configuring ops appear in the when_not_to_trace.md summary table (line 255) but are never given a numbered disqualifying condition section in the body. A reader using the numbered conditions as a disqualification checklist will miss self-configuring ops entirely, and may incorrectly conclude a loop containing an adaptive-compute op is traceable.
   **Fix:** Either add a Disqualifying Condition 4 section for self-configuring ops in when_not_to_trace.md, or update index.md line 13 to say "four primary disqualifying conditions" and add the missing section. The summary table alone is insufficient as a checklist anchor.

## Change Log — Pass 1 Fixes

- `latency_sensitive_workloads.md`: Replaced "20–40 us" with canonical "~20–50 us per op" range to match Chapter 2 reconciled range.
- `when_not_to_trace.md`: Added fourth disqualifying condition (self-configuring ops); updated "three" references to "four".
- `index.md`: Updated "three" reference to "four" disqualifying conditions.

---

# B Review — Chapter 4: Pass 2

1. **File:** `latency_sensitive_workloads.md`, line 31
   **Issue:** Encoding time is stated as "~20–50 us per op." The canonical per-op dispatch overhead range established in Chapter 1 and used consistently throughout this chapter (index.md line 23, latency_sensitive_workloads.md lines 29 and 33) is 17–63 us. The Pass 1 fix changed "20–40 us" to "~20–50 us per op" but the stated range remains inconsistent with the 17–63 us canonical range. A reader calculating the crossover threshold at which async mode alone cannot hide encoding will get a different (lower) upper bound than the rest of the chapter implies.
   **Fix:** Replace "~20–50 us per op" with "17–63 us per op" to match the established canonical range.

2. **File:** `when_not_to_trace.md`, line 185 (inside `decode_core`)
   **Issue:** The residual connection is written as `hidden = layer.norm(hidden + input_tensor)`, where `input_tensor` is the original input to `decode_core` passed in from the outer function. This means every layer adds the first-layer input as its residual, rather than the output of the previous layer's input. A standard transformer residual is `hidden = layer.norm(attn_output + pre_attn_hidden)` — the residual source is the tensor entering that layer, not the tensor entering the entire stack. The code as written is incorrect transformer architecture and would mislead a reader implementing a traced transformer core.
   **Fix:** Track the per-layer residual separately, e.g., `residual = hidden` before the attention call, then `hidden = layer.norm(layer.self_attn(hidden, kv_cache[layer.idx]) + residual)`.

## Change Log — Pass 2 Fixes

- `latency_sensitive_workloads.md`: Changed per-op range to canonical "17–63 us" to match guide-wide reference.
- `when_not_to_trace.md`: Fixed residual connection in decode_core — each layer now saves its own residual before normalization rather than using the original input_tensor.

---

# B Review — Chapter 4: Pass 3

No feedback — chapter approved.

---

# B Review — Chapter 4: Pass 4

No feedback — chapter approved.
