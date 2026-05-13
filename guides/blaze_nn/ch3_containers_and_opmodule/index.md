# Chapter 3 — Containers, OpModule, and pre-built ops

Chapter 2 introduced the two foundational classes — `Module` and `Parameter` — and the device boundary that surrounds them. This chapter fills in the rest of the public model-authoring surface: the three containers that compose modules into trees, the workhorse `OpModule` (in both of its forms) that turns any tt-blaze op into a one-line module, and the two pre-built ops — `blaze_nn.Linear` and `blaze_nn.ops.RMSNorm` — that every model in the repo reaches for first.

By the end of the chapter you should be able to read every line of the Qwen3 walkthrough in Chapter 4 without encountering a `blaze_nn` symbol you do not recognize.

1. [Sequential — the one callable container](sequential.md)
2. [ModuleList and ModuleDict — the non-callable containers](modulelist_and_moduledict.md)
3. [OpModule without subclassing](opmodule_no_subclass.md)
4. [OpModule as a base class](opmodule_subclass.md)
5. [User-allocated output tensors](output_tensors.md)
6. [Pre-built modules: Linear and RMSNorm](prebuilt_modules.md)
