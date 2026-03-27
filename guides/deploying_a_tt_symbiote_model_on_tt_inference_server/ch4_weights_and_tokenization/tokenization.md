# Tokenization

## vLLM Owns Tokenization

vLLM owns tokenization; your model receives integer token IDs only. No tokenizer code is required inside a TT Symbiote model implementation — there is no tokenizer object passed to `initialize_vllm_model`, no `encode` or `decode` call to implement, and no vocabulary file to manage from within the model class.

## HuggingFace Tokenizer Discovery

vLLM resolves the tokenizer from the same path passed to `--model` when the server starts. For a standard TT Symbiote model backed by a HuggingFace repository, the tokenizer files (`tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, and optionally `tokenizer.model` for SentencePiece-based models) are already present in the checkpoint directory after `setup_host.py` runs `huggingface-cli download`.

```
<checkpoint_dir>/
  config.json
  tokenizer_config.json
  tokenizer.json
  special_tokens_map.json
  model-00001-of-00004.safetensors
  ...
```

vLLM constructs the tokenizer like this (simplified):

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
token_ids = tokenizer.encode("Hello, world!")
# token_ids is a list[int] — this is what the model receives
```

Because the checkpoint already contains all tokenizer files, no additional configuration is needed for models that use a standard HuggingFace vocabulary.

## Custom Tokenizer Scenario

If the TT Symbiote model uses a vocabulary that does not match any existing HuggingFace tokenizer, or if the tokenizer has been modified (e.g., added special tokens, different byte-pair merges), custom tokenizer files must be placed in the checkpoint directory:

- `tokenizer_config.json` — specifies the tokenizer class and special token mappings.
- `tokenizer.json` — the full fast-tokenizer definition (for `tokenizers`-backed models).
- `vocab.json` and `merges.txt` — for BPE tokenizers that use the slow path.
- `tokenizer.model` — for SentencePiece-backed tokenizers (e.g., LLaMA-style).

To point vLLM at a tokenizer that lives in a different directory from the model weights, add `--tokenizer <path>` to `DeviceModelSpec.vllm_args`:

```python
class MyCustomModelSpec(DeviceModelSpec):
    vllm_args: list[str] = [
        "--tokenizer", "/path/to/custom/tokenizer/",
        "--tokenizer-mode", "slow",   # use slow tokenizer if fast is unavailable
    ]
```

`--tokenizer` accepts either a local directory path or a HuggingFace repo ID. This is useful when the model's weight checkpoint and its tokenizer are maintained in separate repositories.

## Special Tokens

vLLM reads `eos_token_id`, `pad_token_id`, and `bos_token_id` from `config.json` or `tokenizer_config.json` in the checkpoint directory. These values control generation termination and padding behavior:

- **`eos_token_id`** — generation stops when this token is produced. If this is wrong or missing, the model may generate indefinitely or truncate too early.
- **`pad_token_id`** — used when batching sequences of unequal length. Many decoder-only models set this equal to `eos_token_id`.
- **`bos_token_id`** — prepended to the input sequence for models that require it.

Ensure these are set correctly in `config.json`:

```json
{
  "eos_token_id": 128009,
  "bos_token_id": 128000,
  "pad_token_id": 128009,
  ...
}
```

If the model uses multiple EOS tokens (common in instruction-tuned models where both `<|eot_id|>` and `<|end_of_text|>` should stop generation), pass a list:

```json
{
  "eos_token_id": [128001, 128009]
}
```

Incorrect special token IDs are a common source of generation bugs — outputs that never terminate, outputs missing the final token, or padding that bleeds into the generated text. Verify these values against the original model card before deploying.

---

**Next:** [Chapter 5 — Hardware Initialization and Device Ownership](../ch5_hardware_init/index.md)
