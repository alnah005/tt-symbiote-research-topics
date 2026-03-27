## Agent A Change Log — B Review Pass 1
- Corrected Error 1 root cause: missing/misspelled method, not wrong callable type
- Corrected Error 2 to stand alone: @classmethod missing, not cross-ref to Error 1
- Fixed max-num-batched-tokens rationale: decode width is max_concurrency, not block_size

## Agent A Change Log — B Review Pass 2
- Corrected Error 4 logit shape: prefill returns all-position logits, decode returns last-position only
- Added accuracy mode caveat to Error 8 and performance_tuning.md accuracy description
