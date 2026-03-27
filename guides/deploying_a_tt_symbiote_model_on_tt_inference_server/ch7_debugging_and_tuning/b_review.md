## Pass 1

### Issue 1 — Error 1 root cause is factually wrong (common_errors.md, lines 16–19)

The document states that if `initialize_vllm_model` is defined as an instance method instead of a classmethod, "Python will not find it when called on the class, and will instead report that the attribute is missing." This is incorrect. In Python, instance methods are accessible at the class level as plain function objects. Calling `TTMyModel.initialize_vllm_model(hf_config, ...)` on an instance method does not produce an `AttributeError` for the method being absent — it produces a `TypeError` because `hf_config` is consumed as `self` and subsequent required arguments shift (which is exactly the symptom described in Error 2). An `AttributeError: 'TTMyModel' object has no attribute 'initialize_vllm_model'` with this exact message text would only occur if the method is completely absent from the class, not because it is the wrong type. A reader following the root cause explanation will look for a classmethod decoration problem when the actual issue may be a missing or misspelled method definition. The root cause and the stated symptom describe two different bugs.

### Issue 2 — Error 2 root cause partly duplicates Error 1 incorrectly (common_errors.md, lines 45–48)

Error 2's root cause correctly identifies the argument-shift mechanism (`hf_config` consumed as `self`, remaining args shift). However it then says "This is the same underlying mistake as error 1 — wrong method type." Given that Error 1's root cause is itself misdescribed (see Issue 1 above), linking back to it as authoritative compounds the confusion. More concretely: the `TypeError` in Error 2 and the `AttributeError` in Error 1 have different triggers. Treating them as "the same underlying mistake" leads a reader to apply the same fix (`@classmethod`) to an `AttributeError` that may actually indicate a missing method, leaving the real problem unfound.

### Issue 3 — `max-num-batched-tokens` decode formula uses block_size, but block_size is a KV cache unit, not the natural decode batch token count (performance_tuning.md, lines 23–24)

The document recommends setting `max-num-batched-tokens` to `max_concurrency * block_size` for decode-focused workloads, with the example `32 × 64 = 2048`, and states this "matches the natural batch shape the decode kernel is compiled for." The KV cache block size (64 tokens per block) is a memory allocation granularity, not a determinant of decode batch width. In a standard decode step each active sequence contributes exactly one token regardless of block size; the natural batch width for 32 concurrent sequences is 32 tokens, not 2048. Setting `max-num-batched-tokens = max_concurrency * block_size` would allow up to 2048 tokens per forward pass, which accommodates mixed prefill+decode scheduling but is not derived from the decode kernel's compiled shape in the way the text implies. A reader implementing this formula will set a value 64× larger than the pure-decode batch width; this may be the intended operating point, but the stated rationale (matching the decode kernel's compiled shape) is incorrect and would cause a reader to misunderstand why the value was chosen.

No further qualifying issues found.

## Pass 2

### Issue 1 — Error 4 logit-shape guidance is factually wrong for prefill (common_errors.md, lines 116)

Error 4's fix section states: "A common secondary issue is returning the full `[batch, seq, vocab]` tensor when the sampler expects only the last-token logit slice for each sequence." For a prefill step in standard vLLM operation the engine requires logits for **all** prefill token positions — not just the last one. vLLM uses the full sequence of prefill logits to compute prompt log-probabilities (used for best-of-N, beam search scoring, and log-prob API responses). Returning only the last-token slice during prefill will silently produce incorrect log-probabilities for every prompt token except the final one. A reader following this guidance literally will implement a prefill path that drops all but the last token's logits, causing wrong results for any caller that uses `logprobs` or `prompt_logprobs`.

### Issue 2 — `accuracy` mode description says "No trace capture" but this contradicts Error 8's root cause (performance_tuning.md, line 54; common_errors.md, lines 193–196)

`performance_tuning.md` states that `optimizations="accuracy"` gives "No trace capture: every decode step dispatches TTNN operations through the Python call stack." Error 8 identifies the absence of trace capture as the root cause of slow decode (> 100 ms per step) and frames it as a misconfiguration to fix. Taken together, a reader is told that deliberately selecting `accuracy` mode puts the system into the exact broken state described in Error 8 — but with no warning in either location that `accuracy` mode will consistently reproduce the > 100 ms symptom. This is a critical coherence gap: a reader who hits Error 8 while in `accuracy` mode during debugging will follow Error 8's fix (add trace capture), which directly conflicts with `accuracy` mode's defined behavior (no trace). The document must clarify that `accuracy` mode intentionally operates without trace and that Error 8 applies only when `performance` mode is intended but trace was not captured correctly.

No further qualifying issues found.
