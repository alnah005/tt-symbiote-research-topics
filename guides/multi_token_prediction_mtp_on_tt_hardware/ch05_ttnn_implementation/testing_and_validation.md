# Testing and Validation

## Correctness Check for TTNNMTPHead

**Goal**: Verify that `TTNNMTPHead.forward()` on TT hardware produces logits numerically consistent with the HuggingFace `model.future_prediction[0]` module on CPU.

**Procedure**:

1. Run the HuggingFace model on CPU with a fixed prompt. Extract `backbone_hidden_state` and `x_t1_embedding` at a known token position by instrumenting the HF model's forward pass.
2. Feed these tensors (converted to BF16) through `TTNNMTPHead.forward()` on TT hardware.
3. Compare the TTNN `draft_logits` output against the HF `draft_logits` reference on CPU.

**Acceptance criteria**:

- Pearson Correlation Coefficient (PCC) between TTNN and HF `draft_logits` tensors: **PCC > 0.99** in BF16.
- Use `torch.testing.assert_close` on the CPU reference (HF vs. HF re-run) to confirm the HF baseline is deterministic before comparing to TTNN.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def run_hf_reference(model_path, prompt, position):
    """Extract HF draft_logits at a given token position."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # Extract backbone hidden state and x_t1 embedding via hooks
    backbone_hidden_state = None
    def hook_fn(module, input, output):
        nonlocal backbone_hidden_state
        backbone_hidden_state = output[0][:, position:position+1, :].detach()

    handle = model.model.layers[-1].register_forward_hook(hook_fn)
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    handle.remove()

    x_t1 = input_ids[0, position + 1]
    x_t1_emb = model.model.embed_tokens(x_t1.unsqueeze(0).unsqueeze(0))

    # HF MTP head forward (requires training mode guard bypass for direct call)
    mtp_input = backbone_hidden_state + x_t1_emb
    hf_draft_logits = model.future_prediction[0](mtp_input)   # direct call, not via model.generate()
    return hf_draft_logits, backbone_hidden_state, x_t1_emb

def compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    a_centered = a_flat - a_flat.mean()
    b_centered = b_flat - b_flat.mean()
    pcc = (a_centered * b_centered).sum() / (
        torch.norm(a_centered) * torch.norm(b_centered) + 1e-8
    )
    return pcc.item()

# Compare
hf_logits, hf_hidden, hf_emb = run_hf_reference(MODEL_PATH, PROMPT, POSITION)
ttnn_logits = mtp_head_ttnn.forward(to_ttnn(hf_hidden), to_ttnn(hf_emb), mtp_kv_cache)
ttnn_logits_cpu = to_torch(ttnn_logits)

pcc = compute_pcc(hf_logits, ttnn_logits_cpu)
assert pcc > 0.99, f"PCC {pcc:.4f} below threshold 0.99"
print(f"TTNNMTPHead PCC: {pcc:.4f} — PASS")
```

## Backbone Non-Regression Test

**Goal**: Confirm that enabling `TTNNMTPHead` (`use_mtp=True`) does not alter the backbone's `primary_logits` output compared to the `use_mtp=False` baseline.

**Procedure**:

1. Run the full generation loop with `use_mtp=False` for N decode steps. Record `primary_logits` at each step.
2. Run the same loop with `use_mtp=True`. Record `primary_logits` at each step.
3. Assert that the two `primary_logits` sequences are **bit-for-bit identical**.

Bit-for-bit identity (not just PCC) is required here because any numerical difference in `primary_logits` indicates unintended coupling between the MTP head and the backbone computational graph — shared mutable state, inadvertent KV cache writes from the MTP path being read by the backbone, or similar bugs.

> **[CRITICAL]** If `primary_logits` differ between `use_mtp=True` and `use_mtp=False`, the MTP head is silently corrupting backbone state. Do not proceed to throughput benchmarking until this test passes. Check for shared KV cache buffer references, shared norm weight buffers, or any point where the MTP path can write to a tensor read by the backbone.

```python
def backbone_non_regression_test(model_path, prompt, n_steps=20):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.tolist()[0]

    logits_no_mtp = []
    logits_with_mtp = []

    for use_mtp in [False, True]:
        backbone, mtp_head = load_ttnn_model(model_path, use_mtp=use_mtp)
        ids = list(input_ids)
        kv_len = len(ids) - 1
        for step in range(n_steps):
            primary_logits, hidden = backbone_forward(ids[-1], kv_cache, kv_len)
            if use_mtp:
                x_t1 = sample(primary_logits)
                mtp_head.forward(hidden, embed(x_t1), mtp_kv_cache)   # run but ignore output
            (logits_no_mtp if not use_mtp else logits_with_mtp).append(
                primary_logits.cpu().clone()
            )
            x_t1 = sample(primary_logits)
            ids.append(x_t1)
            kv_len += 1

    for step in range(n_steps):
        assert torch.equal(logits_no_mtp[step], logits_with_mtp[step]), (
            f"Backbone regression at step {step}: primary_logits differ with use_mtp=True"
        )
    print("Backbone non-regression: PASS")
```

## Acceptance Rate Measurement Harness

**Goal**: Measure the empirical acceptance rate α = P(draft accepted) across a representative set of prompts. This value is needed to predict practical speedup from the formula `speedup = (1+α)/2` (see Chapter 4, `ch04_speculative_decoding_with_mtp/`).

```python
def measure_acceptance_rate(model_ttnn, prompts, n_steps=50):
    """
    Returns empirical acceptance rate alpha = accepts / total_draft_tokens.
    Each prompt contributes n_steps draft tokens (one per speculative cycle).
    """
    accepts, total = 0, 0

    for prompt in prompts:
        input_ids = tokenize(prompt)
        kv_cache_len = len(input_ids) - 1   # after prefill

        for _ in range(n_steps):
            # Step 1: primary backbone pass
            primary_logits, hidden = backbone_forward(input_ids[-1], kv_cache, kv_cache_len)
            x_t1 = sample(primary_logits)

            # Step 2: MTP head draft
            draft_logits = mtp_head_forward(hidden, embed(x_t1), mtp_kv_cache)
            x_hat = sample(draft_logits)

            # Step 3: verification backbone pass
            verify_logits = backbone_forward([x_t1, x_hat], kv_cache, kv_cache_len)

            # Step 4: acceptance check
            p_xhat = softmax(verify_logits[..., 0, :])[x_hat].item()
            q_xhat = softmax(draft_logits)[x_hat].item()
            accepted = (random() < min(1.0, p_xhat / q_xhat))

            accepts += int(accepted)
            total += 1

            # Step 5: advance (NO RESAMPLING on rejection)
            input_ids = advance(input_ids, x_t1, x_hat if accepted else None)
            kv_cache_len += 2 if accepted else 1

    alpha = accepts / total if total > 0 else 0.0
    print(f"Empirical acceptance rate alpha = {alpha:.4f} ({accepts}/{total})")
    print(f"Predicted speedup at batch=1: {(1 + alpha) / 2:.4f}x")
    return alpha
```

Run this harness on a mix of prompt types (code, prose, structured data) to get a distribution of α values. Log per-prompt α alongside the aggregate.

## Throughput Benchmark

**Goal**: Measure tokens/second with `use_mtp=True` vs. `use_mtp=False` across batch sizes to identify the crossover point where speculative decode provides a net throughput gain.

| Batch Size | `use_mtp=False` (tok/s) | `use_mtp=True` (tok/s) | Speedup | Empirical α |
|---|---|---|---|---|
| 1 | `[placeholder — to be filled during bring-up]` | `[placeholder — to be filled during bring-up]` | `[placeholder — to be filled during bring-up]` | `[placeholder — to be filled during bring-up]` |
| 4 | `[placeholder — to be filled during bring-up]` | `[placeholder — to be filled during bring-up]` | `[placeholder — to be filled during bring-up]` | `[placeholder — to be filled during bring-up]` |
| 8 | `[placeholder — to be filled during bring-up]` | `[placeholder — to be filled during bring-up]` | `[placeholder — to be filled during bring-up]` | `[placeholder — to be filled during bring-up]` |
| 32 | `[placeholder — to be filled during bring-up]` | `[placeholder — to be filled during bring-up]` | `[placeholder — to be filled during bring-up]` | `[placeholder — to be filled during bring-up]` |

**Expected outcomes based on Chapter 4 analysis:**

- Batch=1: `use_mtp=True` will be SLOWER. Expected speedup ≈ `(1+α)/2 < 1` due to the 2× decode cycle cost (primary + verification passes) at memory-bandwidth-bound batch=1. This is not a regression — it is the predicted result.
- Batch=4, 8: Transition region. Speedup may approach or exceed 1.0 depending on empirical α and the degree to which larger batch sizes alleviate BW-bound conditions.
- Batch=32: Speculative decode benefit expected to be clearly positive if α ≥ 0.5.

> **Key Finding:** A speedup of < 1.0 at batch=1 is expected and correct behavior. Do not interpret this as a bug. The value of implementing the MTP loop now is to enable future throughput gains at production batch sizes and with K > 1 draft tokens.

**Benchmark procedure**:

```bash
# Standard AR baseline
python benchmark.py --model qwen3-35b-a3b --use-mtp=false \
    --batch-sizes 1,4,8,32 --n-tokens 200 --output results_no_mtp.json

# MTP speculative decode
python benchmark.py --model qwen3-35b-a3b --use-mtp=true \
    --batch-sizes 1,4,8,32 --n-tokens 200 --output results_with_mtp.json
```

Record hardware configuration (number of chips, mesh topology), software versions (tt-metal, tt-transformers commit), and prompt type alongside throughput numbers.

## Edge Cases

### Empty Prompt (Single BOS Token)

The MTP head receives `backbone_hidden_state` from position 0 (the BOS embedding). This is a valid hidden state; the MTP block should process it without error. Verify:

- No crash or NaN in `draft_logits`
- `x_t1_embedding` is the embedding of the token sampled from the BOS position logits — a valid vocab index

### Maximum Context Length

When `kv_cache_len + 2 > max_seq_len`, the verification pass would overflow the KV cache buffer. Verify:

- The guard in the generation loop (`speculative_decode_loop_integration.md`) correctly detects this condition before Step 3
- The loop either falls back to single-pass AR for the final token or terminates cleanly — no out-of-bounds memory access
- No silent buffer wrap-around that produces garbled output

### α ≈ 0 (Every Draft Rejected)

When every draft token is rejected, the loop degenerates to standard single-pass AR: one confirmed token per cycle, `kv_cache_len` advances by 1 per cycle. Verify:

- Output sequence is identical to `use_mtp=False` baseline (up to sampling randomness — use fixed seed)
- No correctness regression from the extra (discarded) MTP head + verification pass computation
- KV cache does not accumulate stale entries from repeatedly-rejected verification passes

### α ≈ 1 (Every Draft Accepted)

When every draft token is accepted, the loop advances by 2 tokens per cycle. Verify:

- `kv_cache_len` increments by 2 each cycle without overflow (given sufficient headroom)
- Output sequence is coherent (no duplicated or skipped token positions)
- The accepted `x_hat_t2` token is correctly used as `current_token` for the next cycle's Step 1

> **[SILENT FAILURE]** The α ≈ 1 case is the most likely to expose KV cache off-by-one errors. If `current_token` is incorrectly set to `x_t1` instead of `x_hat_t2` on an accepted step, generation will re-process position t+1 in the next cycle, producing repeated tokens. This does not raise an exception. Validate by inspecting output token sequences directly with a fixed-seed run at high temperature (to maximize α).

## References

- Chapter 4: `ch04_speculative_decoding_with_mtp/` — Acceptance rate formula, speedup analysis, E[tokens/cycle] = 1 + α derivation
- Chapter 5: `mtp_head_ttnn_module.md` — `TTNNMTPHead` module interface, `use_mtp` flag
- Chapter 5: `speculative_decode_loop_integration.md` — Loop structure, rejection path (no resampling), KV cache advance, context length guard
- Chapter 5: `memory_placement_for_mtp.md` — MTP KV cache sizing (128 MiB), DRAM placement
