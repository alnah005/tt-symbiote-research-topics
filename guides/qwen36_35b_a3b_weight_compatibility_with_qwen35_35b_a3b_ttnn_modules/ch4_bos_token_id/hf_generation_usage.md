# HuggingFace `GenerationMixin.generate()` Usage of `bos_token_id`

## Section 1: `bos_token_id` in `GenerationConfig`

`GenerationConfig` carries a `bos_token_id` field that controls whether a
beginning-of-sequence token is synthesized when no `input_ids` are provided to
`generate()`. The resolution order during a `generate()` call is:

1. Explicit `GenerationConfig` object passed by the caller — if
   `generation_config.bos_token_id` is set, it is used directly.
2. Fall back to `model.config.bos_token_id` — if no explicit `GenerationConfig`
   is passed (or its `bos_token_id` is `None`), `generate()` reads
   `model.config.bos_token_id`.

For **Qwen3.6**, both locations resolve to `248044`:

```python
# After AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-35B-A3B")
model.config.bos_token_id          # → 248044
model.generation_config.bos_token_id  # → 248044  (populated from model.config)
```

For **Qwen3.5**, both locations are `None`:

```python
# After AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-35B-A3B")
model.config.bos_token_id          # → None
model.generation_config.bos_token_id  # → None
```

## Section 2: BOS Auto-Prepend Condition

`GenerationMixin._prepare_model_inputs()` constructs the initial `input_ids`
tensor passed to the first forward step. The relevant simplified logic from
`transformers/generation/utils.py`:

```python
# Simplified from GenerationMixin._prepare_model_inputs
if input_ids is None:
    # No input_ids provided: build a BOS-only sequence from bos_token_id
    input_ids = torch.full(
        (batch_size, 1),
        self.generation_config.bos_token_id,
        dtype=torch.long,
        device=device,
    )
```

The auto-prepend path is triggered **only** when `input_ids is None` at the
`generate()` call site. When the caller passes pre-formed `input_ids` (produced
by the tokenizer), this branch is never taken and `bos_token_id` has no effect
on the `input_ids` tensor.

`forced_bos_token_id` is a separate field — also `None` for both Qwen3.5 and
Qwen3.6 unless explicitly set — and is not affected by the `bos_token_id`
change analyzed here.

**Risk window:** The only dangerous call pattern is:

```python
# Dangerous: triggers bos_token_id = 248044 → out-of-range embedding lookup
outputs = model.generate(max_new_tokens=50)  # input_ids omitted
```

This constructs a `[batch, 1]` tensor containing `248044` and passes it to
`model.embed_tokens`, which indexes a `[151936, hidden_dim]` table. Index
`248044` is out of bounds.

## Section 3: Tokenizer Side

For Qwen3.6, `AutoTokenizer.from_pretrained` sets:

```python
tokenizer.bos_token_id  # → 248044
tokenizer.bos_token     # → the string representation of token 248044
```

However, Qwen-series tokenizers use a chat template and tiktoken-based encoding.
The `apply_chat_template()` and `encode()` paths in the Qwen3.6 tokenizer do
**not** automatically prepend token `248044` to their output. All token IDs
produced by `apply_chat_template()` fall within the normal vocab range
`[0, 151935]`. The `bos_token_id` attribute exists on the tokenizer object and
is accessible, but the tokenizer's encoding logic does not insert it.

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-35B-A3B")

ids = tok.apply_chat_template(
    [{"role": "user", "content": "Hello"}],
    tokenize=True,
    add_generation_prompt=True,
)
assert max(ids) < tok.vocab_size  # True: all IDs in [0, 151935]
assert tok.bos_token_id == 248044  # True: attribute set, but not inserted
```

The risk is not the tokenizer's `encode()` path — it is `model.generate()`
reading `model.config.bos_token_id` to synthesize `input_ids` when none are
provided.

## Section 4: Concrete Scenarios

### Scenario A — Safe: tokenizer-produced `input_ids` passed explicitly

```python
inputs = tokenizer.apply_chat_template(
    messages, tokenize=True, return_tensors="pt", add_generation_prompt=True
)
outputs = model.generate(inputs, max_new_tokens=128)
# input_ids is not None → auto-prepend branch is skipped
# bos_token_id = 248044 has no effect on any tensor
```

**Risk level: none.** This is the standard Qwen usage pattern.

### Scenario B — Unsafe: `generate()` called without `input_ids`

```python
outputs = model.generate(max_new_tokens=128)
# input_ids is None → generate() constructs [[248044]]
# embed_tokens(248044) on a [151936, hidden_dim] table → IndexError on CPU/GPU
# On TTNN device: undefined behavior (silent garbage values possible)
```

**Risk level: high.** This is the failure mode introduced by adding
`bos_token_id: 248044` to Qwen3.6. The identical call on Qwen3.5 would raise
a different error (no `bos_token_id` to construct from), but Qwen3.6 silently
proceeds to an out-of-range lookup.

### A Note on Pipeline Objects

HuggingFace `Pipeline` objects (e.g., `pipeline("text-generation", model=...)`) accept raw text strings and call the tokenizer internally before invoking `model.generate()`. They are not affected by the Scenario B failure mode because they always produce `input_ids` before calling `generate()`. However, when using a pipeline with a Qwen3.6 model, ensure the pipeline does not pass `bos_token_id` directly to `generate()` — use Option B from Section 6 to suppress it at the model level. Direct `model.generate()` calls always require a pre-formed `input_ids` tensor; they do not accept raw text strings.

## Section 5: Comparison to Qwen3.5

| Property | Qwen3.5-35B-A3B | Qwen3.6-35B-A3B |
|---|---|---|
| `model.config.bos_token_id` | `None` | `248044` |
| `model.generation_config.bos_token_id` | `None` | `248044` |
| `tokenizer.bos_token_id` | `None` or absent | `248044` |
| Auto-prepend triggers when `input_ids=None`? | No (no BOS token to prepend) | Yes — out-of-range ID |
| Effect when pre-formed `input_ids` provided | None | None |
| Embedding table size (`vocab_size`) | `151,936` | `151,936` |
| BOS token `248044` in-range for embedding? | N/A | **No — out of range** |

Qwen3.5 had `bos_token_id = None`, so `generate()` never attempted BOS
auto-prepend. Qwen3.6 introduces a `bos_token_id` that is **out of range** for
the embedding table, creating a new failure mode that does not exist in Qwen3.5.

## Section 6: Mitigation

Suppress BOS auto-prepend for Qwen3.6 by one of the following approaches:

**Option A — Pass `bos_token_id=None` at the call site (preferred for one-off use):**

```python
outputs = model.generate(input_ids, bos_token_id=None, max_new_tokens=128)
```

**Option B — Clear `model.generation_config.bos_token_id` after loading (preferred for module-level use):**

```python
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-35B-A3B")
model.generation_config.bos_token_id = None  # suppress out-of-range auto-prepend
```

**Option C — Always pass pre-formed `input_ids` (always required regardless of the above):**

```python
input_ids = tokenizer.apply_chat_template(
    messages, tokenize=True, return_tensors="pt", add_generation_prompt=True
)
outputs = model.generate(input_ids, max_new_tokens=128)
```

Options A and B address the `generate(input_ids=None)` failure path. Option C
is the primary defense and should always be followed. Using all three together
is the safest configuration.

> **Key Finding:** `bos_token_id = 248044` in Qwen3.6 activates `generate()`'s
> BOS auto-prepend path in a way that was never possible with Qwen3.5 (which had
> no `bos_token_id`). Because `248044` exceeds `vocab_size = 151,936`, the
> auto-prepend path leads to an out-of-range embedding index. The failure only
> occurs when `input_ids=None` at the `generate()` call site. Standard usage
> with pre-formed `input_ids` is unaffected. **Action required:** always pass
> pre-formed `input_ids`; additionally set `model.generation_config.bos_token_id
> = None` to close the failure path entirely.

---
**Next:** [`tt_symbiote_generation_loop.md`](./tt_symbiote_generation_loop.md)
