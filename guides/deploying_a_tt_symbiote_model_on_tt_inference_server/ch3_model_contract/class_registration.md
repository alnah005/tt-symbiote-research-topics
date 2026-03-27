# Class Registration

## How `TTPlatform.get_model_cls()` Resolves the Class Name

When `TTModelLoader` is asked to load a model, it delegates class resolution to `TTPlatform.get_model_cls()`. This method reads the `"architectures"` field from `config.json` in the HuggingFace model repository — for example, `"LlamaForCausalLM"` — and prepends the literal string `"TT"` to produce the registry key it looks up: `"TTLlamaForCausalLM"`. If no entry exists in `ModelRegistry` under that key, the loader raises an error and refuses to proceed.

The transformation is purely string-based: whatever string appears in `config.json`'s `"architectures"` list becomes the suffix. There is no fuzzy matching or fallback to the vLLM built-in registry. Your registration key must match exactly.

## Where to Register

Registration belongs in the `register_tt_models()` function located in:

```
tt-vllm-plugin/tt_vllm_plugin/model_loader/tt_loader.py
```

This function is called once at worker startup, before any model is loaded. Add a call to `ModelRegistry.register_model()` for each TT Symbiote model your plugin supports:

```python
from vllm.model_executor.models import ModelRegistry
from my_symbiote_package.models.my_symbiote_model import MySymbioteModel

def register_tt_models() -> None:
    ModelRegistry.register_model(
        "TTMySymbioteModel",   # must match "TT" + config.json "architectures" entry
        MySymbioteModel,
    )
    # Register additional models here as needed.
```

The `ModelRegistry.register_model()` call accepts the string key as its first argument and the class object (not an instance) as its second. vLLM stores this mapping internally; `TTPlatform.get_model_cls()` retrieves it by key at load time.

## Why the `"TT"` Prefix Is Required

The `"TT"` prefix serves two distinct purposes:

1. **Collision avoidance.** vLLM ships with its own built-in registry entries for standard architectures. `"LlamaForCausalLM"` is already registered there. Prepending `"TT"` produces a key that cannot collide with any vLLM-native entry, so `TTModelLoader` can safely call `ModelRegistry.register_model()` without overwriting the upstream implementation.

2. **Loader signaling.** `TTModelLoader` uses the presence of the `"TT"` prefix as a lightweight signal that the class was explicitly provided for TT hardware and therefore satisfies the `initialize_vllm_model()` contract described in the next section. Classes retrieved under a `"TT"`-prefixed key are assumed to expose that classmethod; the loader will call it unconditionally.

---

**Next:** [initialize_vllm_model.md](./initialize_vllm_model.md)
