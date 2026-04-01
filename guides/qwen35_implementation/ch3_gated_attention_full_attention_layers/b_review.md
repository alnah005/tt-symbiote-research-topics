# Agent B Review — Chapter 3: GatedAttention — Pass 1

1. **output_gate.md, ~line 9 — Wrong shape in gate formula.** The formula states $W_\text{gate} \in \mathbb{R}^{d_\text{hidden} \times (n_\text{heads} \cdot \text{head\_dim})}$ and then computes $\sigma(x \, W_\text{gate}^\top)$. With that shape, $W_\text{gate}^\top$ has shape $(n_\text{heads} \cdot \text{head\_dim}) \times d_\text{hidden}$, so $x \, W_\text{gate}^\top$ would be $(B, d_\text{hidden}) \times (n_\text{heads} \cdot \text{head\_dim}, d_\text{hidden})$ — dimension mismatch. In code, `gate_w` is loaded as the raw weight of shape `(n_heads * head_dim, hidden_size)`, then transposed via `.T` to `(hidden_size, n_heads * head_dim)`, and `ttnn.linear(x, gate_weight)` computes `x @ gate_weight` = $(B, d_\text{hidden}) \times (d_\text{hidden}, n_\text{heads} \cdot \text{head\_dim})$. The formula must be corrected to either: (a) $\sigma(x \, W_\text{gate})$ with $W_\text{gate} \in \mathbb{R}^{d_\text{hidden} \times (n_\text{heads} \cdot \text{head\_dim})}$ (no transpose), or (b) $\sigma(x \, W_\text{gate}^\top)$ with $W_\text{gate} \in \mathbb{R}^{(n_\text{heads} \cdot \text{head\_dim}) \times d_\text{hidden}}$ (matching the raw HF weight shape before transpose). The same error appears in the inline formula at line 109 (`_apply_gate` section).

# Agent B Review — Chapter 3: GatedAttention — Pass 2

1. **partial_rope.md, line 87 — Wrong variable name in pairing-distance sentence.** The sentence reads: "Qwen3.5 needs `head_dim/2 = 32` separation within the rotated block only." This is factually wrong. For Qwen3.5, `head_dim = 256`, so `head_dim/2 = 128`, not 32. The correct quantity is `rotary_dim/2 = 32`. The sentence contradicts the correct statement one paragraph above it (line 83: "the correct pairing is $j$ with $j + \text{rotary\_dim}/2 = j + 32$"). A reader implementing this would use 128 or 32 unpredictably depending on which line they follow.

2. **partial_rope.md, lines 83 and 177 — Contradictory descriptions of `rotary_embedding_llama` pairing.** Line 83 describes the standard (Llama) op as pairing "dimension $j$ with dimension $j + \text{head\_dim}/2 = j + 128$" — the split-half (HF-style) convention. Line 177 (table) correctly labels `RotarySetup` as "interleaved (dim $2i$ with $2i+1$)" — the Meta-style convention. These are opposite pairing schemes; only one can be correct for the same op. Meta-style `rotary_embedding_llama` uses interleaved pairing ($2i$ with $2i+1$), not split-half ($j$ with $j + \text{head\_dim}/2$). Line 83's description of the Llama pairing distance is wrong: the interleaved format pairs consecutive odd/even indices, not indices 128 apart. A reader relying on line 83 alone would misunderstand what Failure 2 actually is.

# Agent B Review — Chapter 3: GatedAttention — Pass 3

1. **partial_rope.md, line 165 — Incorrect parenthetical about HfRotarySetup pairing range.** The sentence reads: "Within the first 64 dims, this means dim 0 pairs with dim 128, dim 1 with dim 129, etc." Dim 128 is not within the first 64 dims — it is in the second half of the 256-dim head vector. The `HfRotarySetup` op pairs dim $j$ with dim $j + \text{head\_dim}/2 = j + 128$ across the entire head, not within the first 64 dims. The parenthetical inverts the actual geometry: dim 0 (from the rotary block) pairs with dim 128 (from the pass-through region of the unpatched layout), not with another dim in the first 64. A reader implementing a custom op from this description would incorrectly scope the pairing to within the 64-dim rotary block.

2. **partial_rope.md, line 208 — Wrong sync count.** States "eliminates all 2+ host-device syncs per attention layer." The source comment at `gated_attention.py` line 92 reads "Eliminates 5 host-device syncs per attention layer." The number 2 (or 2+) contradicts the source. The correct figure from the implementation is 5.

3. **forward_flow.md, lines 190–195 — `compute_pcc` snippet omits variable definitions.** The shown snippet uses `x_flat` and `y_flat` as if they are already in scope, but the actual function in `test_pcc.py` lines 65–66 defines them as `x_flat = x.flatten().float()` and `y_flat = y.flatten().float()`. The guide omitted both lines. The snippet as written would raise `NameError: name 'x_flat' is not defined` if executed, and misrepresents the reference implementation.

# Agent B Review — Chapter 3: GatedAttention — Pass 4

1. **partial_rope.md, lines 77–79 — Wrong numerical value and wrong ratio in Failure 1 example.** The guide states that at `i=16`, the incorrect frequency is `1 / (10^6)^{0.125} ≈ 0.0316` and that "the incorrect value is 31.6x too large." Both figures are wrong. `(10^6)^{0.125} = 10^{6 × 0.125} = 10^{0.75} ≈ 5.623`, giving `1 / 5.623 ≈ 0.1778`, not 0.0316. The value 0.0316 would correspond to `(10^6)^{0.25} = 10^{1.5} = 31.6`, i.e., the exponent 32/128 rather than 32/256 — a factor-of-2 error in the exponent. The correct ratio of incorrect to correct is `0.1778 / 0.001 ≈ 178x`, not 31.6x. An implementer using this example to sanity-check frequency values at i=16 would accept a computed value that is 5.6x too large as correct.

# Agent B Review — Chapter 3: GatedAttention — Pass 5

1. **partial_rope.md, line 177 (table) — Wrong pairing-distance formula for `HfRotarySetup`.** The table entry for `HfRotarySetup` lists dimension pairing as "non-interleaved (dim $j$ with $j + \text{head\_dim}/2$)". With `head_dim=256` that resolves to $j + 128$. This is wrong for Qwen3.5's partial-RoPE configuration. The actual pairing within the rotated block is dim $j$ with $j + \text{rotary\_dim}/2 = j + 32$ for $j \in [0, 31]$, as stated correctly at line 165 of the same file. The table uses `head_dim/2` where it must be `rotary_dim/2`. An implementer reading the table as a quick reference would code pairing at distance 128 instead of 32, rotating dimensions entirely outside the 64-dim rotary block and producing incorrect Q/K vectors at every non-zero position.

# Agent B Review — Chapter 3: GatedAttention — Pass 6

No feedback — chapter approved.

# Agent B Review — Chapter 3: GatedAttention — Pass 7

1. **partial_rope.md, line 147 — Wrong pairing distance claimed for `HfRotarySetup`.** The bullet states "`ttnn.experimental.rotary_embedding` applies split-half pairing within the rotary block: dim $j$ pairs with dim $j + \text{rotary\_dim}/2 = j + 32$ for $j \in [0, 31]$" and concludes "dim 0 pairs with dim 32, ... dim 31 with dim 63." This is directly contradicted by the patch code in the same section: Step 3 writes corrected cos/sin into `[:half_rotary]` (indices 0–31) and `[half_head : half_head + half_rotary]` (indices 128–159). The two slice offsets are separated by `half_head = head_dim // 2 = 128`, meaning the actual pairing used by `ttnn.experimental.rotary_embedding` is dim $j$ with dim $j + 128$ — distance `head_dim/2`, not `rotary_dim/2`. An implementer reading line 147 would place paired dimensions 32 apart within the first 64 dims, producing a custom RoPE op that mismatches what the device kernel actually does.

2. **partial_rope.md, line 159 (table) — Same wrong pairing distance in `HfRotarySetup` table entry.** The table lists dimension pairing for `HfRotarySetup` as "non-interleaved (dim $j$ with $j + \text{rotary\_dim}/2 = j + 32$ for $j \in [0, 31]$)". As shown above, the actual pairing distance is `head_dim/2 = 128`. The correct entry is "non-interleaved (dim $j$ with $j + \text{head\_dim}/2 = j + 128$)". Line 163 in the same section partially acknowledges this by stating "the same `half_head` offset applies" for `HfRotarySetup`, but the table and the Failure 2 bullet both assert distance 32, creating a direct contradiction within the chapter.

3. **forward_flow.md, line 157 — Wrong constant name for test layer variable.** States "Tests layer index `ATTENTION_LAYER = 3`." In the actual source file `test_attention_pcc.py`, the variable is named `LAYER` (not `ATTENTION_LAYER`) and is not hardcoded to 3; it is determined dynamically as `LAYER = next(i for i, t in enumerate(layer_types) if t == "full_attention")`. No constant named `ATTENTION_LAYER` exists anywhere in that file. An implementer searching for or defining `ATTENTION_LAYER = 3` would be working from a fabricated identifier.

# Agent B Review — Chapter 3: GatedAttention — Pass 8

No feedback — chapter approved.

# Agent B Review — Chapter 3: GatedAttention — Pass 9

1. **partial_rope.md — Failure 3 is referenced but never defined.** The section header "The Three Failure Modes of Standard `rotary_embedding_llama`" introduces `### Failure 1` and `### Failure 2` subsections, but there is no `### Failure 3` subsection anywhere in the file. The resolution table in "Why the Patch Addresses All Three Failure Modes" references "Failure 3 (format mismatch): Resolved by using `HfRotarySetup` (HF format) instead of `RotarySetup` (Meta interleaved format)" — but this failure mode is never explained. The `gated_attention.py` docstring (lines 14–16) lists it explicitly as the third reason the standard op cannot be used: "cos/sin are in Meta interleaved format but Q/K are in HF format." An implementer reading the chapter has no basis to identify or avoid this failure mode, as it is mentioned only in the resolution summary and never described as a failure mode in its own right. A `### Failure 3 — cos/sin Format Mismatch` subsection is required for the chapter to be correct and complete on this point.

# Agent B Review — Chapter 3: GatedAttention — Pass 10

1. **forward_flow.md, line 157 — Wrong test file for dynamic layer determination.** The 27B PCC Validation section states: "Tests the first `full_attention` layer, whose index is determined dynamically as `LAYER = next(i for i, t in enumerate(layer_types) if t == "full_attention")` (read from the model's `config.json`)." This dynamic lookup is in `test_attention_pcc.py` (the reference script), not `test_pcc.py`. In `test_pcc.py` line 35, the layer index is hardcoded: `ATTENTION_LAYER = 3`. An implementer extending `test_pcc.py` following this description would write a dynamic config-reading lookup, when the actual test uses a compile-time constant. The sentence also uses the wrong variable name `LAYER`; the constant in `test_pcc.py` is named `ATTENTION_LAYER`.

# Agent B Review — Chapter 3: GatedAttention — Pass 11

No feedback — chapter approved.
