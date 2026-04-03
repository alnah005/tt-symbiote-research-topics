# Compression Analysis: Chapter 3 — K=V Sharing and V-Norm — Pass 1

## Summary
- Total files analyzed: 3
- Estimated current line count: ~752
- Estimated post-compression line count: ~620
- Estimated reduction: ~18%

## CRUCIAL Suggestions
None.

## MINOR Suggestions

1. **index.md lines 10-23: Overview duplicates sub-file content.**
   The overview paragraph on K=V sharing (lines 10-16) repeats the projection shape, the divergent K/V paths, normalization details, and memory savings figure (~220 MB) — all of which are covered in full detail in `k_eq_v_mechanism.md` (lines 29-36, 63-88, 169-188). Similarly, the V-norm paragraph (lines 18-23) restates the `with_scale=False` semantics and the `TTNNDistributedRMSNorm` concern, both fully developed in `vnorm_implementation.md` (lines 9-67, 148-228). Recommend trimming each overview paragraph to 1-2 sentences and deferring specifics to the sub-files. Estimated savings: ~15 lines.

2. **k_eq_v_mechanism.md lines 146-167: "Why K=V Sharing Works / Intuition" restates the dataflow.**
   The intuition section re-explains that K gets RoPE (position-dependent) and V does not (position-invariant), and that K-norm has a learned scale while V-norm does not. This is already stated in the Step 3 divergent post-processing section (lines 58-88) and the dataflow diagram (lines 92-119). The two descriptions differ only in framing ("why" vs. "what") but carry the same information. Recommend collapsing the intuition into 2-3 sentences appended to Step 3 rather than a separate subsection. Estimated savings: ~15 lines.

3. **k_eq_v_mechanism.md lines 306-328: "Key Implementation Considerations" items 1 and 2 repeat earlier material.**
   Item 1 ("Tensor aliasing safety") re-explains that `k_norm` returns a new tensor leaving the original intact for V-norm — this is already stated in Step 2 (lines 49-53) and the Reference Code commentary (lines 141-144). Item 2 ("KV cache writes") restates that K and V are different tensors at cache-write time despite sharing a source, which is evident from the dataflow diagram (lines 110, 118). Recommend removing items 1 and 2, keeping only items 3 and 4 which contain non-redundant weight-loading and fused-QK guidance. Estimated savings: ~15 lines.

4. **vnorm_implementation.md lines 29-61: "Contrast With Standard RMSNorm" section is verbose.**
   The full `Gemma4RMSNorm` class listing (lines 44-61) is 18 lines of code that primarily serves to show the `with_scale` flag. The mathematical contrast (standard RMSNorm formula with gamma) is already implied by the V-norm definition in lines 12-16. Recommend replacing the full class listing with a brief note (e.g., "The `Gemma4RMSNorm` class controls this via `with_scale=False`, which skips `self.weight` registration and the gamma multiply — see the HuggingFace source for the full class."). Estimated savings: ~20 lines.

5. **vnorm_implementation.md lines 126-131: "Gradient Flow" subsection is tangential.**
   This section discusses training-time gradient behavior (no gamma gradient to accumulate, reduced optimizer state). The guide's scope is TTNN inference mapping, not training. Recommend removing or reducing to a single-line note. Estimated savings: ~8 lines.

6. **vnorm_implementation.md: Option C description (lines 230-266) and its performance table entry are disproportionate.**
   Option C is explicitly "not recommended for production" and serves only as a debugging reference, yet it receives 37 lines of description plus a column in the 14-line performance table. Recommend condensing to ~10 lines: the code snippet and a brief note on why it is suboptimal (multiple dispatches, no fusion). Estimated savings: ~25 lines.

7. **k_eq_v_mechanism.md lines 259-286: Fused QKV section partially overlaps with index.md.**
   The index.md "Why These Features Matter" item 1 (lines 27-31) already notes that fused QKV must adapt and that the fused weight packs Q and K only. The sub-file then re-introduces this before expanding on it. The introductory framing (lines 259-262) can be cut. Estimated savings: ~4 lines.

## Load-Bearing Evidence

- **index.md**: Lines 10-16 contain the sentence "A single `k_proj` linear produces a shared tensor that is assigned to both the K and V paths *before* any normalization or RoPE" and the ~220 MB savings figure — both repeated verbatim in concept in `k_eq_v_mechanism.md` lines 29-36 and 183-185.
- **k_eq_v_mechanism.md**: Lines 49-53 state "all subsequent operations on `key_states` are **not in-place** --- they produce new tensors via functional operations (RMSNorm, RoPE, transpose). This means the original shared tensor remains intact for the V path." Lines 308-312 restate this: "In PyTorch, `value_states = key_states` creates an alias. The subsequent `self.k_norm(key_states)` returns a new tensor, leaving the original intact for V-norm."
- **vnorm_implementation.md**: Lines 126-131 discuss training-time gradient flow and optimizer state reduction — content that falls outside the guide's TTNN inference mapping scope.

## VERDICT
- Crucial updates: no
