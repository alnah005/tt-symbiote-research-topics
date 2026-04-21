# TT-Symbiote Generation Loop Initialization

## Section 1: How TT-Symbiote Initializes the First Input Token

TT-Symbiote's generation loop does **not** call `GenerationMixin.generate()`.
It runs a custom autoregressive loop that calls the TTNN forward pass directly,
bypassing all HuggingFace generation utilities. At each step the loop:

1. Receives `input_ids` as a pre-formed integer tensor (from the caller or the
   tokenizer).
2. Slices the appropriate token for the current position from `input_ids`.
3. Passes the token index to the embedding layer to retrieve the first hidden
   state.
4. Runs the TTNN forward pass for that step.
5. Samples the next token and appends it to the sequence.

Because the loop never invokes `GenerationMixin._prepare_model_inputs()`, the
`bos_token_id` auto-prepend path described in
[`hf_generation_usage.md`](./hf_generation_usage.md) is not reachable through
normal TT-Symbiote execution. The value `248044` does not enter any TTNN tensor
as long as `input_ids` are formed by the tokenizer.

## Section 2: Potential Risk Point — Config-Sourced Initial Token

> **[SILENT FAILURE]** If TT-Symbiote's generation loop initialization code
> reads `model.config.bos_token_id` to prepend or construct the first token of
> the sequence, token ID `248044` would be passed to the embedding layer. The
> embedding table has shape `[151936, hidden_dim]`; index `248044` is out of
> bounds. On TTNN device, out-of-bounds embedding lookups do not raise a Python
> `IndexError` — they may silently return garbage values, producing incorrect
> hidden states for every subsequent token without any explicit error signal.

**This is a code review item.** Audit every location in TT-Symbiote's
generation loop initialization that reads any of the following attributes:

```python
model.config.bos_token_id           # → 248044 for Qwen3.6
model.generation_config.bos_token_id  # → 248044 for Qwen3.6
tokenizer.bos_token_id              # → 248044 for Qwen3.6
```

If any such read is used to construct or prepend the first token index, it must
be removed or replaced with a tokenizer `apply_chat_template()` call that
produces in-range IDs.

## Section 3: KV Cache Initialization

The KV cache is allocated as a tensor of shape:

```
[batch_size, num_key_value_heads, max_seq_len, head_dim]
```

None of these dimensions are derived from `bos_token_id`. The KV cache is sized
by `max_seq_len` (a sequence-length budget) and the attention head configuration
(`num_key_value_heads = 8`, `head_dim = 128` for both Qwen3.5 and Qwen3.6).
Adding `bos_token_id: 248044` to the Qwen3.6 config does not change any of
these values.

**Risk level: none.** KV cache initialization is unaffected.

## Section 4: Position ID Tensor for Step 0

Position IDs are sequential integers starting at 0. For a prefill of length
`L`, the position ID tensor is:

```python
position_ids = torch.arange(0, L, dtype=torch.long)  # [0, 1, 2, ..., L-1]
```

Position IDs are not derived from `bos_token_id`. The tensor content, shape,
and dtype are identical between Qwen3.5 and Qwen3.6 generation loops.

**Risk level: none.** Position ID tensors are unaffected.

## Section 5: Paged KV Cache

Paged KV cache page table entries are indexed by sequence position, not by
token ID. Page allocation and page table construction are functions of
`max_seq_len`, `block_size`, and the number of sequences — none of which are
affected by `bos_token_id`.

**Risk level: none.** Paged KV cache page tables are unaffected.

## Section 6: The Out-of-Range Embedding ID on TTNN Device

At BF16, the embedding table loaded onto the TTNN device has shape:

```
[151936, 4096]   # hidden_dim = 4096 for Qwen3.6-35B-A3B
```

An embedding lookup for index `248044` is a read at row `248044` of this table.
Row `248044` does not exist — the table has only 151,936 rows (indices
`0`–`151935`).

On CPU (PyTorch), this raises a hard `IndexError`:

```
IndexError: index 248044 is out of bounds for dimension 0 with size 151936
```

> **[SILENT FAILURE]** On TTNN device, out-of-bounds memory reads in embedding
> operations are **undefined behavior**. The TTNN runtime does not guarantee an
> exception. The lookup may return garbage BF16 values — a vector of
> unpredictable floats — that propagate through every subsequent attention and
> MoE layer. The model produces a numerically corrupt output with no explicit
> error or NaN signal. This is a silent correctness bug that cannot be detected
> from output tokens alone without a reference comparison.

This asymmetry between CPU (loud failure) and TTNN device (silent failure)
makes the out-of-range embedding ID particularly dangerous. CI tests that run
on CPU will surface the `IndexError`; on-device inference may not.

## Section 7: Safe Recipe

Follow all four steps to prevent `bos_token_id = 248044` from reaching any TTNN
tensor.

1. **Always tokenize with `apply_chat_template`.** Form `input_ids` by calling
   the tokenizer directly and pass the result to TT-Symbiote's generation loop:

   ```python
   input_ids = tokenizer.apply_chat_template(
       messages,
       tokenize=True,
       return_tensors="pt",
       add_generation_prompt=True,
   )
   # All IDs in input_ids are in [0, 151935] — safe for the embedding table
   tt_symbiote_loop.generate(input_ids, max_new_tokens=128)
   ```

2. **Audit generation loop init for `config.bos_token_id` reads.** Search
   TT-Symbiote's generation loop initialization code for any reference to
   `bos_token_id`. If found, remove or gate the read so that it does not
   construct or prepend a token index from `config.bos_token_id`.

   ```bash
   grep -r "bos_token_id" tt_symbiote/models/qwen/
   ```

3. **Suppress auto-prepend in any HF `generate()` calls used for
   debugging or reference.** Pass `bos_token_id=None` explicitly:

   ```python
   outputs = model.generate(input_ids, bos_token_id=None, max_new_tokens=128)
   ```

4. **Add an embedding bounds check before the first TTNN forward pass in CI.**
   This catches out-of-range IDs on CPU before they reach the device:

   ```python
   assert input_ids.max().item() < model.config.vocab_size, (
       f"input_ids contain out-of-range token ID {input_ids.max().item()}; "
       f"vocab_size={model.config.vocab_size}"
   )
   ```

## Section 8: Comparison to Qwen3.5

| Property | Qwen3.5-35B-A3B | Qwen3.6-35B-A3B |
|---|---|---|
| `config.bos_token_id` | `None` | `248044` |
| BOS token in range for embedding table? | N/A | **No — out of range** |
| Risk if generation loop reads `config.bos_token_id`? | None (value is `None`, no lookup triggered) | **High** (out-of-range ID, silent failure on TTNN device) |
| KV cache shape affected? | No | No |
| Position ID tensor affected? | No | No |
| Paged KV cache affected? | No | No |
| TTNN tensor impact when pre-formed `input_ids` used? | None | None |

Qwen3.5's absence of `bos_token_id` meant this risk category did not exist.
Qwen3.6 introduces a `bos_token_id` that is out of range for the embedding
table — a new field that must be actively suppressed in both the HF generation
path and TT-Symbiote's generation loop.

> **Key Finding:** `bos_token_id = 248044` has **no effect** on any TTNN tensor
> when pre-formed `input_ids` from the tokenizer are used. The risk exists only
> if TT-Symbiote's generation loop reads `config.bos_token_id` to initialize the
> first token index. On TTNN device, the resulting out-of-bounds embedding lookup
> produces silent garbage values rather than a hard error — making this a silent
> correctness bug if it occurs on-device. Follow the four-step safe recipe in
> Section 7 to eliminate this risk.

---
**Next:** [Chapter 5 — MTP Head: Weight Loading and Inference Impact](../ch5_mtp_head/index.md)
