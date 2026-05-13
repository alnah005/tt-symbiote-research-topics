# Chapter 1 — Why blaze-nn and how it fits together

This chapter positions blaze-nn against the three layers it lives between — PyTorch (the API shape it borrows), tt-blaze (the dataflow compiler it traces into), and ttnn (the tensor library it carries) — and gets you to a running test suite. By the end you will have one mental model (user code → blaze-nn tracing → tt-blaze graph → tt-metal kernels), one invariant (every tensor crossing a `Module` boundary is a `ttnn.Tensor`), and an installed checkout with the three test tiers green.

1. [What blaze-nn is (and is not)](what_it_is.md)
2. [The ttnn-native contract](ttnn_native_contract.md)
3. [Getting started: install, environment, three test tiers](getting_started.md)
