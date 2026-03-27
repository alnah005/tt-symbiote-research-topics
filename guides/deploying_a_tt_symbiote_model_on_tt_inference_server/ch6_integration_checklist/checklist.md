# Integration Checklist

Use this checklist in order. Each phase must be fully complete before beginning the next. Items in italics are failure modes that are easy to overlook; pay special attention to them.

---

## Phase 1 — Repository & Dependencies

**1. Fork/clone the tt-symbiote repo and confirm the model class exists.**

   Clone or fork the `tt-symbiote` repository that contains your model implementation. Navigate to the module directory and confirm the model class file is present and importable:

   ```bash
   git clone https://github.com/tenstorrent/tt-symbiote.git
   cd tt-symbiote
   python -c "from models.my_model.my_model import TTMySymbioteModel; print('OK')"
   ```

   If the import raises a `ModuleNotFoundError`, the `PYTHONPATH` is not set correctly or the model file does not exist at the expected path. Fix this before proceeding.

**2. Install `tt-vllm-plugin` and its dependencies.**

   From the root of the `tt-inference-server` checkout, locate the plugin directory and install it in editable mode:

   ```bash
   cd tt-inference-server/tt-vllm-plugin
   pip install -e .
   ```

   Editable installation (`-e`) is required so that changes to the plugin source are immediately visible without reinstalling. Verify the installation:

   ```bash
   python -c "import tt_vllm_plugin; print(tt_vllm_plugin.__file__)"
   ```

**3. Set `VLLM_USE_V1=1` in the shell environment.**

   The `tt-vllm-plugin` targets the vLLM V1 executor architecture. The env variable `VLLM_USE_V1` must be set to `1` before any Python process imports `vllm`. Setting it after import has no effect.

   ```bash
   export VLLM_USE_V1=1
   ```

   Add this line to your shell profile (`~/.bashrc`, `~/.zshrc`) or to the container entrypoint so it is always present. *Do not rely on setting it inline only for the server launch command; model registration hooks run at import time and require it.*

**4. Confirm `HF_TOKEN` is set and the target HuggingFace model repo is accessible.**

   ```bash
   export HF_TOKEN=hf_...
   huggingface-cli whoami          # confirm token is valid
   huggingface-cli download <org>/<model-repo> --repo-type model --dry-run
   ```

   If the `--dry-run` command returns a list of files without error, the token and repo access are correct. If you receive a 401 or 403, request access to the gated repo on the HuggingFace website before proceeding.

---

## Phase 2 — ModelSpec Registration

**5. Add an `ImplSpec` pointing to the tt-symbiote repo and model module path. (→ Ch2)**

   In `workflows/model_spec.py` (inside `tt-inference-server`), define an `ImplSpec` that tells the server where to find the model source:

   ```python
   from model_spec import ImplSpec

   MY_MODEL_IMPL = ImplSpec(
       repo="https://github.com/tenstorrent/tt-symbiote.git",
       module_path="models.my_symbiote_model.my_symbiote_model",
   )
   ```

   The `module_path` must be a fully-qualified Python module path resolvable from the root of the cloned repo. See Chapter 2 for the full `ImplSpec` field reference.

**6. Define `DeviceModelSpec` entries for each target device with `max_concurrency`, `max_context`, `vllm_args`, and `override_tt_config`. (→ Ch2)**

   Each `DeviceModelSpec` captures one (device, configuration) pair. At minimum you must set:

   - `max_concurrency`: maximum simultaneous in-flight sequences
   - `max_context`: maximum total token context window (prompt + generation)
   - `vllm_args`: list of additional CLI flags passed to vLLM at server startup
   - `override_tt_config`: dict of TT-Metalium config keys to override for this device

   ```python
   from model_spec import DeviceModelSpec

   N300_SPEC = DeviceModelSpec(
       device="N300",
       max_concurrency=32,
       max_context=8192,
       vllm_args=["--block-size", "64", "--max-num-seqs", "32"],
       override_tt_config={"ENABLE_ELTWISE_UNARY_FUSED_WITH_MATMUL": True},
   )

   T3K_SPEC = DeviceModelSpec(
       device="T3K",
       max_concurrency=64,
       max_context=32768,
       vllm_args=["--block-size", "64", "--max-num-seqs", "64", "--tensor-parallel-size", "8"],
       override_tt_config={},
   )
   ```

**7. Create a `ModelSpecTemplate` and append to `spec_templates`; confirm `expand_to_specs()` produces non-empty `MODEL_SPECS`. (→ Ch2)**

   Wrap the `ImplSpec` and all `DeviceModelSpec` entries in a `ModelSpecTemplate`, then register it:

   ```python
   from model_spec import ModelSpecTemplate

   MY_MODEL_TEMPLATE = ModelSpecTemplate(
       hf_model_repo="<org>/<model-repo>",
       impl=MY_MODEL_IMPL,
       device_specs=[N300_SPEC, T3K_SPEC],
   )

   spec_templates.append(MY_MODEL_TEMPLATE)
   ```

   After saving, confirm expansion is non-empty:

   ```bash
   python -c "
   from workflows.model_spec import MODEL_SPECS
   print(len(MODEL_SPECS), 'specs registered')
   for s in MODEL_SPECS:
       print(s.device, s.hf_model_repo)
   "
   ```

   If `MODEL_SPECS` is empty, `expand_to_specs()` found no matching template-device pairs. Check that `spec_templates` was mutated (not shadowed by a local rebind) and that `device_specs` is non-empty.

---

## Phase 3 — Model Class Contract

**8. Register the model class in `register_tt_models()` under key `"TT<ArchName>"`. (→ Ch3)**

   Inside `tt_vllm_plugin`, locate the `register_tt_models()` function. Add your class under a key that matches the architecture name vLLM will look up from `hf_config.architectures[0]`:

   ```python
   from models.my_symbiote_model.my_symbiote_model import TTMySymbioteModel

   def register_tt_models():
       registry = {
           # ... existing entries ...
           "TTMySymbioteModel": TTMySymbioteModel,
       }
       return registry
   ```

   The key must match exactly what appears in the model's `config.json` under `"architectures"`. A mismatch causes vLLM to raise an `UnknownModelError` at startup.

**9. Implement `@classmethod initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size, max_seq_len, ...)` returning a fully-initialized instance. (→ Ch3)**

   This is the primary constructor path. It receives a pre-opened `mesh_device` handle and must return a ready-to-run `cls` instance. See Chapter 3 for the full expected signature and Chapter 5 for device lifecycle rules.

   Key invariants:
   - Do NOT call `ttnn.open_mesh_device()` inside this method (the device is already open).
   - Load all weights before returning; the caller will not call any setup method after this.
   - The method must be decorated with `@classmethod`.

**10. Implement `prefill_forward(tokens, prompt_lens, page_table, kv_cache)` returning CPU logits of shape `(batch, vocab_size)`. (→ Ch3)**

    `tokens` is a 2-D integer tensor of shape `(batch, max_prompt_len)`. `prompt_lens` is a 1-D integer tensor containing the unpadded length of each sequence. The returned logits tensor must be on CPU (`.cpu()` must be called before returning). Returning a device tensor causes a silent hang in the scheduler.

**11. Implement `decode_forward(tokens, page_table, kv_cache)` returning logits of shape `(batch, vocab_size)`. (→ Ch3)**

    During decode, `tokens` has shape `(batch, 1)` — one token per sequence per step. The returned logits tensor must again be on CPU.

**12. Implement `allocate_kv_cache(num_blocks, block_size, num_kv_heads, head_dim, dtype)` returning per-layer `(k_cache, v_cache)` tensors on device. (→ Ch3)**

    The return value must be a list of length equal to the number of transformer layers. Each element is a `(k_cache, v_cache)` tuple where each tensor has shape `(num_blocks, block_size, num_kv_heads, head_dim)` and lives on the `mesh_device`. Do NOT allocate KV cache inside `initialize_vllm_model`; the block manager calls this method separately after it has determined the number of blocks.

---

## Phase 4 — Weight & Tokenization Setup

**13. Confirm HF checkpoint is downloadable via `huggingface-cli download <repo>` with `HF_TOKEN`. (→ Ch4)**

    ```bash
    HF_TOKEN=<token> huggingface-cli download <org>/<model-repo> \
        --local-dir /path/to/checkpoints/<model-repo>
    ```

    After downloading, verify the directory contains at least one `.safetensors` shard and a `config.json`.

**14. Verify `hf_config._name_or_path` resolves to the checkpoint directory inside `initialize_vllm_model`. (→ Ch4)**

    Log or assert the path at the top of `initialize_vllm_model`:

    ```python
    import os
    assert os.path.isdir(hf_config._name_or_path), (
        f"Checkpoint directory not found: {hf_config._name_or_path}"
    )
    ```

    If the assertion fails, set the `HF_HOME` env variable or pass `--download-dir` to the vLLM server so the path resolves correctly.

**15. If using a custom tokenizer: place `tokenizer_config.json` in the checkpoint dir and add `"--tokenizer <path>"` to `vllm_args`. (→ Ch4)**

    Some Symbiote models use a tokenizer that differs from the upstream HuggingFace repo. In that case:

    - Copy or symlink `tokenizer_config.json`, `tokenizer.model`, and any vocabulary files into the local checkpoint directory.
    - Add `"--tokenizer", "/path/to/checkpoint"` to the `vllm_args` list in the `DeviceModelSpec`.

**16. Confirm `eos_token_id`, `pad_token_id`, and `bos_token_id` are set in `config.json`. (→ Ch4)**

    Open `config.json` in the checkpoint directory and verify all three fields are present and non-null. Missing token IDs cause vLLM's stopping criteria to malfunction, producing runaway generation or immediate EOS at the first token.

---

## Phase 5 — Hardware & Environment

**17. Set `MESH_DEVICE` to the target device string (`"N300"`, `"T3K"`, etc.). (→ Ch5)**

    ```bash
    export MESH_DEVICE=N300   # or T3K, N150, etc.
    ```

    The plugin reads this variable to select the correct `DeviceModelSpec` at startup. An unset or misspelled value causes the plugin to fall back to a default or raise a `KeyError`.

**18. Confirm `TT_VISIBLE_DEVICES` is set correctly for the target hardware. (→ Ch5)**

    ```bash
    export TT_VISIBLE_DEVICES=0        # single-chip N300
    export TT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # eight-chip T3K
    ```

    If `TT_VISIBLE_DEVICES` includes indices for chips that are not physically present or not in the expected topology, `ttnn.open_mesh_device()` will raise a device initialization error.

**19. Confirm `initialize_vllm_model` does NOT call `ttnn.open_mesh_device()` or `ttnn.close_mesh_device()`. (→ Ch3, Ch5)**

    Search your model file for both calls:

    ```bash
    grep -n "open_mesh_device\|close_mesh_device" models/my_model/my_model.py
    ```

    Either call inside the model class is a hard error. The device lifecycle is managed exclusively by `tt-vllm-plugin`. Calling `open_mesh_device` creates a second conflicting device handle; calling `close_mesh_device` destroys the shared handle and hangs the server.

**20. If the model uses on-device TTNN traces: set `has_builtin_warmup=True` in `ModelSpecTemplate`. (→ Ch4, Ch5)**

    Models that compile and capture TTNN traces during the first forward pass must signal this to the plugin so the scheduler does not time out during the warmup period:

    ```python
    MY_MODEL_TEMPLATE = ModelSpecTemplate(
        hf_model_repo="<org>/<model-repo>",
        impl=MY_MODEL_IMPL,
        device_specs=[N300_SPEC, T3K_SPEC],
        has_builtin_warmup=True,
    )
    ```

---

## Phase 6 — Smoke Test

**21. Start the vLLM server directly (no Docker).**

    ```bash
    VLLM_USE_V1=1 \
    MESH_DEVICE=N300 \
    HF_TOKEN=<token> \
    HF_MODEL=<org>/<model-repo> \
    python -m vllm.entrypoints.openai.api_server \
        --model <org>/<model-repo> \
        --max-model-len <N> \
        --block-size 64 \
        --max-num-seqs <B> \
        --dtype bfloat16
    ```

    Replace `<N>` with the `max_context` value from your `DeviceModelSpec` and `<B>` with `max_concurrency`. Wait for the line `INFO:     Application startup complete.` before sending any request.

**22. Send a test request.**

    ```bash
    curl -s http://localhost:8000/v1/completions \
        -H 'Content-Type: application/json' \
        -d '{"model":"<org>/<model-repo>","prompt":"Hello","max_tokens":5}'
    ```

**23. Verify the response contains valid text and no error fields.**

    A successful response looks like:

    ```json
    {
      "id": "cmpl-...",
      "object": "text_completion",
      "choices": [
        {"text": " world! How can", "index": 0, "finish_reason": "length"}
      ]
    }
    ```

    If the response contains `"error"` at the top level, check the server stderr for the Python traceback. Common causes at this stage are a missing key in `register_tt_models()`, an incorrect `hf_config._name_or_path`, or a device tensor returned from `prefill_forward`.

**24. Check that `decode_forward` is called after prefill completes.**

    Add a temporary `print("decode_forward called")` at the top of your `decode_forward` method and confirm it appears in the server log after the prefill log lines. If the server hangs after prefill and the decode print never appears, the most common cause is `prefill_forward` returning a device tensor instead of a CPU tensor.

---

**Next:** [`worked_example.md`](./worked_example.md)
