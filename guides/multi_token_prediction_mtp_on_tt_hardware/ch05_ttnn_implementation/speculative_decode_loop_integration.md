# Speculative Decode Loop Integration

## Modified Generation Loop Structure

The three-pass speculative decode cycle replaces the standard single-pass autoregressive decode loop. Each cycle produces either one or two new tokens. The steps below correspond directly to the algorithm defined in Chapter 4 (`ch04_speculative_decoding_with_mtp/`).

### Step 1 — Primary Backbone Pass (always-accepted token)

Run the backbone forward pass over the current token `input_ids[-1]` (the last token in the sequence or the last confirmed token from the previous cycle):

```python
primary_logits, backbone_hidden_state = backbone_forward(
    input_ids=current_token,
    kv_cache=kv_cache,
    kv_cache_len=kv_cache_len,
)
x_t1 = sample(primary_logits)   # argmax or temperature sampling
```

`x_t1` is a sample from the target backbone distribution `p`. There is **no accept/reject decision for `x_t1`** — it is always appended to the output sequence. `backbone_hidden_state` (shape `[batch, 1, H]`) is retained for Step 2.

### Step 2 — MTP Head Pass (draft token)

Run `TTNNMTPHead.forward()` using the backbone hidden state from Step 1 and the embedding of `x_t1`:

```python
x_t1_emb = embed_tokens(x_t1)           # [batch, 1, H]
draft_logits = mtp_head.forward(
    backbone_hidden=backbone_hidden_state,
    x_t1_emb=x_t1_emb,
    mtp_kv_cache=mtp_kv_cache,
)
x_hat_t2 = sample(draft_logits)          # draft candidate for position t+2
```

`draft_logits` (distribution `q`) will be used in the acceptance check in Step 4.

### Step 3 — Verification Backbone Pass (2-token prefill)

Run the backbone over the two-token sequence `[x_t1, x_hat_t2]`, extending the KV cache:

```python
verify_logits = backbone_forward(
    input_ids=[x_t1, x_hat_t2],
    kv_cache=kv_cache,
    kv_cache_len=kv_cache_len,
)
# verify_logits[..., 0, :] is the backbone's distribution for position t+2 (the token after x_t1)
# verify_logits[..., 1, :] is the backbone's distribution for position t+3 (the token after x_hat_t2)
p_xhat = softmax(verify_logits[..., 0, :])[x_hat_t2]
q_xhat = softmax(draft_logits)[x_hat_t2]
```

The verification pass is a standard 2-token prefill reusing the same backbone TTNN module with a 2-token input tensor rather than a 1-token tensor. No new backbone code path is needed.

### Step 4 — Acceptance Check

```python
u = uniform_random(0.0, 1.0)
accepted = (u < min(1.0, p_xhat / q_xhat))
```

This is the standard speculative sampling acceptance criterion. `x_t1` is never part of this check; only the draft token `x_hat_t2` is evaluated.

### Step 5 — Advance (CRITICAL: NO RESAMPLING)

> **[CRITICAL]** The accepted and rejected paths differ only in how many tokens are appended and by how much the KV cache advances. On rejection, **do not resample position t+2**. Resampling would add an extra backbone call per rejection, collapsing E[tokens/cycle] to 2 regardless of α, which gives speedup = `2 / 2 = 1.0` — no benefit. The speedup formula `(1+α)/2` from Chapter 4 (`ch04_speculative_decoding_with_mtp/`) is only valid when the rejected path advances by exactly 1 token with no additional sampling. See Chapter 4 for the derivation.

**Accepted path** (`accepted == True`):

```python
output_ids.extend([x_t1, x_hat_t2])
kv_cache_len += 2
current_token = x_hat_t2   # next cycle starts from the second new token
```

**Rejected path** (`accepted == False`):

```python
output_ids.extend([x_t1])
kv_cache_len += 1
current_token = x_t1       # next cycle starts from the first new token
# DO NOT append x_hat_t2
# DO NOT call backbone_forward again to resample position t+2
```

> **[SILENT FAILURE]** If `x_hat_t2` is appended on rejection (without going through an accept/reject check), the output distribution is no longer equivalent to the target distribution `p`. This is a correctness bug that will not raise an exception — it silently produces slightly biased token sequences. Validate via the correctness test in `testing_and_validation.md`.

## KV Cache Management

### Variable Advance Length

The generation loop must handle KV cache advances of either 1 or 2 tokens per cycle. The simplest implementation tracks a scalar `kv_cache_len` per sequence and increments it by the advance amount after each cycle.

For the verification pass (Step 3), the backbone processes 2 tokens against the existing KV cache at positions `[kv_cache_len, kv_cache_len+1]`. The KV entries written during the verification pass must **not** be committed until after the acceptance decision: if rejected, the KV entry at position `kv_cache_len+1` (corresponding to `x_hat_t2`) was written but should not be used in subsequent cycles (it will be overwritten by the next cycle's primary pass at that position). In practice, the simplest approach is to let the verification pass write both entries and then advance `kv_cache_len` by 1 on rejection — the stale entry at `kv_cache_len+1` will be overwritten in the next cycle's Step 3 before it is ever read.

### Paged KV Cache

If the backbone uses a paged KV cache (page table indexing), each cycle must update 1 or 2 page table entries:

- **Accepted**: mark both positions `kv_cache_len` and `kv_cache_len+1` as valid; advance `kv_cache_len` by 2.
- **Rejected**: mark only position `kv_cache_len` as valid; advance `kv_cache_len` by 1. The entry at `kv_cache_len+1` written during the verification pass is effectively abandoned and will be reclaimed when that slot is next allocated.

The MTP head's own KV cache (one additional layer slot, see `memory_placement_for_mtp.md`) follows the same advance logic.

### Maximum Context Length Guard

```python
if kv_cache_len + 2 > max_seq_len:
    # Cannot run the full 3-pass cycle; fall back to single-pass AR
    # or terminate generation
    ...
```

When fewer than 2 positions remain in the KV cache, the verification pass cannot write 2 entries. The loop must detect this before Step 3 and either fall back to single-pass AR for the final token or halt generation. See the edge case in `testing_and_validation.md`.

## Batch Size > 1

For batch size `B > 1`, each sequence in the batch may have a different acceptance outcome in Step 4. The implementations diverge per sequence:

- Each sequence `b` independently computes `accepted[b]`.
- `kv_cache_len[b]` is a per-sequence scalar; advance by 1 or 2 independently.
- `current_token[b]` is set to `x_hat_t2[b]` if accepted, or `x_t1[b]` if rejected.

The backbone and MTP head forward passes process all `B` sequences in parallel throughout Steps 1–3. Only the advance logic in Step 5 is per-sequence. The verification pass (Step 3) processes `B × 2` tokens simultaneously — the 2-token prefill is a fully batched operation.

> **Key Finding:** At batch=1, the three-pass cycle costs approximately `2 × C_decode` (primary pass + verification pass; MTP head cost ≈ 0 in BW-bound regime) and yields `E[tokens/cycle] = 1 + α`. Speedup = `(1+α)/2 < 1` — always slower than standard AR at batch=1. At larger batch sizes the compute-to-memory ratio improves and speedup > 1 becomes achievable. This is expected behavior; see Chapter 4 (`ch04_speculative_decoding_with_mtp/`) for the full cost analysis.

## Integration Point in tt-transformers

The primary integration points in the tt-transformers codebase are:

- **Generation loop entry** (`generate.py` or equivalent in `models/`): the three-pass cycle replaces the single-step decode call. `TTNNMTPHead` is instantiated alongside the backbone model during model loading.
- **Backbone forward call**: Steps 1 and 3 both call the same backbone TTNN module. Step 1 passes a 1-token input; Step 3 passes a 2-token input. The backbone module must support variable-length prefill inputs — this is already required for the initial prompt prefill and should not require changes.
- **`embed_tokens` access**: Step 2 requires `embed_tokens(x_t1)` to be callable independently of a full backbone forward pass. If the embedding lookup is currently embedded inside the backbone `forward()` call, it must be exposed as a separate callable for the MTP head path.

The `TTNNMTPHead` instance is created with `use_mtp=True` for speculative decode runs and `use_mtp=False` for standard AR baseline runs. When `use_mtp=False`, the loop executes Step 1 only (standard single-pass AR) and skips Steps 2–5.

## Fallback to Standard Autoregressive Decode

When `use_mtp=False` (or when `TTNNMTPHead.forward()` returns `None`):

```python
primary_logits, _ = backbone_forward(current_token, kv_cache, kv_cache_len)
x_t1 = sample(primary_logits)
output_ids.append(x_t1)
kv_cache_len += 1
current_token = x_t1
```

This is identical to the pre-MTP generation loop. The fallback path must be exercised in CI to guard against regressions introduced by loop restructuring.

## References

- Chapter 4: `ch04_speculative_decoding_with_mtp/` — Algorithm definition, E[tokens/cycle] derivation, speedup formula, rejection mechanics
- Chapter 5: `mtp_head_ttnn_module.md` — `TTNNMTPHead` module interface and `use_mtp` flag
- Chapter 5: `memory_placement_for_mtp.md` — KV cache sizing and DRAM placement for the MTP attention layer
- Chapter 5: `testing_and_validation.md` — Acceptance rate harness, edge case tests for rejection path and context overflow
