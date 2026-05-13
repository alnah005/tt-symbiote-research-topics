# Chapter 2 — Module, Parameter, and the device boundary

This chapter opens blaze-nn's two foundational classes — `Module` and `Parameter` — and walks the contract that surrounds them: how attributes are routed into the parameter and submodule registries, how `state_dict` / `load_state_dict` preserve tensor identity verbatim, what `module.to(device)` actually does (and does not) move, and the `blaze_nn.interop` helpers that model authors use at the torch ↔ ttnn boundary to build the dict they hand to `load_state_dict`. Everything here is part of the public model-author surface; tracing internals stay in Chapter 5.

1. [Parameter — the trivial-looking class](parameter.md)
2. [Module attribute protocol — how `__setattr__` routes](module_attribute_protocol.md)
3. [Traversal and the state-dict contract](traversal_and_state_dict.md)
4. [Device binding — what `module.to(device)` does and doesn't](device_binding.md)
5. [Interop at the boundary — torch ↔ ttnn for `load_state_dict`](interop_at_the_boundary.md)
