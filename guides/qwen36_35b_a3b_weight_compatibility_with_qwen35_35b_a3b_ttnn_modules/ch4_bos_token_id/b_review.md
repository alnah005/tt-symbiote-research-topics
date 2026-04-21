## B Feedback — Pass 1

1. **`tt_symbiote_generation_loop.md`, Section 6** — States "the embedding table loaded onto the TTNN device has shape `[151936, hidden_dim]` where `hidden_dim = 7168` for Qwen3.6-35B-A3B." The correct value is 4096, not 7168. Cross-check: the MTP guide (`ch02_mtp_weights_and_memory/`) established the Qwen3.6-35B-A3B MTP head contains ~160M parameters with H=4096; at H=7168, attention alone ≈ 4 × 7168² ≈ 205M params, which already exceeds the accepted 160M total. At H=4096: attention ≈ 67M + dense FFN ≈ 90M ≈ 157M ≈ 160M (consistent). Fix: change `hidden_dim = 7168` to `hidden_dim = 4096`; update the inline embedding table shape to `[151936, 4096]`.

2. **`hf_generation_usage.md`, Scenario C** — Reads "Some `generate()` call patterns accept raw text strings and use the tokenizer internally." The standard `model.generate()` method does NOT accept raw text strings; it requires an `input_ids` tensor. Raw-text input is only supported by HuggingFace `Pipeline` objects (which internally call the tokenizer before `generate()`). Describing this as a `generate()` call pattern misrepresents the API boundary. Fix: remove Scenario C or reframe it as a pipeline-based scenario distinct from direct `model.generate()` calls; make clear that direct `model.generate()` always requires an `input_ids` tensor.

## B Feedback Application Log — Pass 1

- Fix 1: Changed `hidden_dim = 7168` to `hidden_dim = 4096` in `tt_symbiote_generation_loop.md` Section 6; updated the embedding table shape comment from `[151936, hidden_dim]   # hidden_dim = 7168 for Qwen3.6-35B-A3B` to `[151936, 4096]`.
- Fix 2: Removed Scenario C from `hf_generation_usage.md`; added a note that direct `model.generate()` always requires pre-formed `input_ids` — raw text input is handled only by HuggingFace Pipeline objects, not by `model.generate()` itself.

## B Feedback — Pass 2

No feedback — chapter approved.
