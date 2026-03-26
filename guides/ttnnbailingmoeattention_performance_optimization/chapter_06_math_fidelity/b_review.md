# B Review — Chapter 6: Math Fidelity — Pass 1

1. **`index.md` line 95 — wrong line range for Qwen3 config block.**
   The text says `TTNNQwen3FullAttention.move_weights_to_device_impl` sets the config at "lines 333–347 of `qwen_attention.py`". In the source file, `self.sdpa.program_config = ttnn.SDPAProgramConfig(` begins at line **332** (the `if self.sdpa.program_config is None:` guard is at line 331). The correct range is lines 332–347, not 333–347. The same off-by-one error means the code block shown in the document (`self.sdpa.program_config = ttnn.SDPAProgramConfig(` as the first line) does not correspond to line 333 in source.

2. **`hifi4_vs_hifi2.md` line 68 — wrong bit-field breakdown for BFP16/bfloat16.**
   The text states: "The BFP16 accumulator uses 4-bit exponent + 11-bit mantissa (approximately)." Bfloat16 uses **1 sign + 8 exponent + 7 mantissa** bits (it is a truncated IEEE 754 float32, sharing the same 8-bit exponent). Neither bfloat16 (8e+7m) nor float16 (5e+10m) has a 4-bit exponent or 11-bit mantissa. This is a factual error in the description of the accumulator format when `fp32_dest_acc_en=False`.

3. **`index.md` line 71 — `fp32_dest_acc_en` description inverts the size direction.**
   The text says: "When `False`, the accumulator uses the smaller `dst` format (16-bit BFP), which frees 2× more destination tiles." The source comment at `qwen_attention.py` line 339 states `fp32_dest_acc_en=False increases dst_size from 4 to 8`. Going from 4 to 8 tiles is a doubling of tile capacity, which happens *because the per-tile memory cost is halved* by using 16-bit instead of 32-bit — not because the format is "smaller." The current wording is internally consistent but the phrase "frees 2× more destination tiles" is misleading: setting `fp32_dest_acc_en=False` doubles the number of available dst tiles (from 4 to 8), not merely "frees 2× more." More critically, the parameter description earlier (line 68 of `index.md`) says `fp32_dest_acc_en=True` gives "smaller dst" and `fp32_dest_acc_en=False` gives "larger dst" — this word choice ("larger dst") is the opposite of the natural reading. The source says `fp32_dest_acc_en=False` *increases* the number of dst *tiles*, but only because the per-tile format is *smaller* (16-bit vs 32-bit). Calling the 16-bit format "larger dst" (line 74: "larger dst, more throughput") is backwards: 16-bit dst tiles are individually smaller but there are more of them. This inconsistency could cause reader confusion and is a factual framing error.

4. **`hifi4_vs_hifi2.md` line 52 — `_apply_qk_norm` line number not verifiable from the cited location.**
   The text says "`_apply_qk_norm` normalizes Q and K to unit scale before the attention score computation (line 2659 of `attention.py`)". In the source, line 2659 is a *call site* (`query_states, key_states = self._apply_qk_norm(query_states, key_states)`) inside `_forward_decode_paged`, while the *definition* of `_apply_qk_norm` is at line 2454. The citation says "line 2659" as if it were the definition or the canonical reference for the feature, but line 2659 is just one of two call sites (the other is at line 2559 in the prefill path). This is misleading: a reader who navigates to line 2659 finds a call, not the implementation. The cite should point to the definition at line 2454 or describe it as a call site explicitly.

5. **`index.md` line 24 — tile size claim for `q_chunk_size=256` at `head_dim=128` is incorrect.**
   The text states: "At `head_dim=128`, 256 tokens fit in two standard TTNN tiles (128 tokens per tile)." A standard TTNN tile is 32×32, giving 32 tokens (rows) per tile at any head dimension — the tile row count is 32, not 128. If a tile holds 32 tokens, then 256 tokens span 8 tiles, not 2. The claim of "128 tokens per tile" contradicts the standard TTNN 32×32 tile geometry.

## Agent A Change Log — Pass 1

- item 1: `index.md` — corrected Qwen3 config line range from "lines 333–347" to "lines 332–347" (the `sdpa.program_config =` assignment opens at line 332).
- item 2: `hifi4_vs_hifi2.md` — corrected BFP16 bit-field from "4-bit exponent + 11-bit mantissa" to "1 sign + 8-bit exponent + 7-bit mantissa" (bfloat16 format).
- item 3: `index.md` — corrected the `fp32_dest_acc_en=False` labeling. Changed "larger dst" to "smaller per-tile dst format freed up — 8 tiles available" and "larger dst, more throughput" to reflect that it is the reduced per-tile size that enables more tiles, not a larger dst.
- item 4: `hifi4_vs_hifi2.md` — corrected `_apply_qk_norm` citation from "(line 2659)" to "(defined at line 2454, called at line 2659)" to point readers to the definition rather than the call site.
- item 5: `index.md` — corrected tile geometry: "256 tokens fit in two standard TTNN tiles (128 tokens per tile)" → "256 tokens span 8 tiles (32 tokens per tile)" (TTNN tiles are 32×32 = 32 tokens per tile).

## Agent A Change Log — Pass 1 (C compression)

- C1: Removed the 3-level HiFi4/HiFi2/LoFi bullet list from `hifi4_vs_hifi2.md` (which duplicated index.md's math_fidelity table). Replaced with a cross-reference sentence pointing to index.md's Parameter Reference.
- C2: Removed the standalone `fp32_dest_acc_en` mechanism paragraph from `hifi4_vs_hifi2.md` (lines 15–19, duplicating index.md lines 69–74). Kept the combined-effects table (which is new) and preserved the fp32 detail in the cross-reference.
- C3: Replaced the `q_chunk_size` and `k_chunk_size` section in `accuracy_throughput_tradeoff.md` with a short cross-reference to index.md and a net-new profiling note about pinning chunk sizes across configs.

---

# B Review — Chapter 6: Math Fidelity — Pass 2

1. **`index.md` line 73 — parenthetical for `fp32_dest_acc_en=True` states the wrong size direction.**
   The current text reads: "Current Bailing MoE setting: `True` (high accuracy, **smaller** per-tile dst format — 4 tiles available)." `fp32_dest_acc_en=True` widens the accumulator to 32-bit FP — the **larger** per-tile format. Having only 4 dst tiles available is the *consequence* of each tile occupying more memory (32 bits vs 16 bits). The parenthetical should say "larger per-tile dst format — 4 tiles available." The adjacent line 74 correctly describes `False` as "smaller per-tile dst format freed up — 8 tiles available," so the two lines are now internally inconsistent: True and False are both labeled "smaller," which is a direct contradiction. Source evidence: `qwen_attention.py` line 339 comment states `fp32_dest_acc_en=False increases dst_size from 4 to 8`, confirming that True corresponds to the 4-tile (FP32, larger) case.

2. **`index.md` line 24 — description of what `q_chunk_size=256` and `k_chunk_size=256` do contains a residual inaccuracy.**
   The text states: "Standard TTNN tiles are 32×32, so 256 tokens span 8 tiles (32 tokens per tile). `exp_approx_mode=False` disables the polynomial approximation of the exponential in softmax, using the exact hardware exponential instead." The tile count arithmetic (256 / 32 = 8 tiles) is now correct after the Pass 1 fix. However, the sentence mixes two unrelated facts in a single sentence without a logical connector, and — more critically — it identifies the tile span as "8 tiles" but then immediately introduces `exp_approx_mode` without completing the thought about why the chunk size matters for performance. This is a minor structural issue, but the factual claim about 8 tiles spanning 256 tokens at 32 tokens per tile is correct and matches the source. No further correction needed on the tile arithmetic itself.
   **Verdict: no error — item raised for completeness only; the tile arithmetic is correct.**

3. **`hifi4_vs_hifi2.md` line 44 — GQA group ratio stated as a fact without a verifiable source in the provided files.**
   The text asserts: "each Q head attends to KV pairs shared with 4 Q heads (H/Hkv = 16/4 = 4)." The `TTNNBailingMoEAttention.from_torch` method (line 2328–2329 of `attention.py`) reads `num_heads` and `num_kv_heads` from the model config at runtime — neither value is hardcoded in the source files provided as the source of truth. No Bailing MoE config JSON or fixture file in the provided source confirms H=16 or Hkv=4. If the actual deployed model uses different values (e.g., H=32, Hkv=8), the GQA ratio claim and the associated reasoning about robustness would be incorrect. The claim should either be sourced to a config file or qualified as an example.

4. **`index.md` line 67 — `math_fidelity` table labels the mantissa bits for LoFi as 1, which is inconsistent with the prose description.**
   The table at lines 61–65 lists LoFi as "1" mantissa bit. The prose at line 67 states the fidelity controls "the number of mantissa bits used when multiplying two BFP8 values in the inner product." With 1 mantissa bit, the product is effectively a sign-magnitude integer multiply. This is factually plausible for LoFi but the table does not cite a source for these exact bit counts. The source files (attention.py, qwen_attention.py) do not define or document the HiFi4/HiFi2/LoFi mantissa bit counts anywhere in the provided lines; the values come from TTNN hardware documentation not included in the source of truth. The bit counts (4/2/1) are widely cited in Tenstorrent documentation and are unlikely to be wrong, but they are not directly verifiable from the cited source files.
   **Verdict: unverifiable from provided source, not a confirmed error.**

5. **`accuracy_throughput_tradeoff.md` line 96 — cross-reference to `index.md`'s decode chunk size section is accurate, but the claim that the auto chunk size "is typically a single tile (32 or 64 tokens depending on the KV cache configuration)" originally appeared in `index.md` line 37 and is not sourced to the provided files.**
   `index.md` line 37 states: "the kernel default is typically a single tile (32 or 64 tokens depending on the KV cache configuration)." Neither `attention.py` lines 2413–2440 nor `qwen_attention.py` lines 332–347 document what value the kernel selects when `q_chunk_size=0`. This is an internal claim about kernel behavior that is not verifiable from the specified source-of-truth files. If the default is actually different (e.g., always 32, or dependent on a different parameter), the description would be wrong.
   **Verdict: unverifiable from provided source of truth; flag for confirmation against TTNN kernel documentation.**

## Agent A Change Log — Pass 2

- item 1: `index.md` — fixed the `fp32_dest_acc_en=True` description from "smaller per-tile dst format" to "FP32 accumulator, larger per-tile format — 4 dst tiles available". The FP32 format is the larger format (uses more per-tile memory), and having only 4 tiles is the consequence. Also corrected `False` line to say "BFP16 accumulator, smaller per-tile format — 8 dst tiles available."
- item 3: `hifi4_vs_hifi2.md` — added "(using the H=16 and Hkv=4 parameters established in Chapter 1)" to qualify the H/Hkv values.
- item 4: `hifi4_vs_hifi2.md` — added "per Tenstorrent Wormhole hardware documentation" qualification to the mantissa bit counts for HiFi4/HiFi2/LoFi.
- item 5: `index.md` — removed the speculative "typically a single tile (32 or 64 tokens depending on the KV cache configuration)" claim. Changed to the factual "the kernel picks chunk sizes automatically based on the input dimensions."
- C CRUCIAL-3: Trimmed the `q_chunk_size` section in `accuracy_throughput_tradeoff.md` to 2 sentences — one profiling recommendation (pin chunk sizes for fair comparison) and one back-reference to index.md.

---

# B Review — Chapter 6: Math Fidelity — Pass 3

No feedback — chapter approved.

---

# B Review — Chapter 6: Math Fidelity — Pass 3

No feedback — chapter approved.
