## B Feedback — Pass 1

1. [index.md, ~line 17] GQA prerequisite table states "64 Q heads and **4 KV heads**". Ground truth (Chapter 1): `num_key_value_heads=8`. This wrong number seeds the `k_proj`/`v_proj` shape errors throughout both content files. Fix: change "4 KV heads" to "8 KV heads".

2. [shared_weight_shapes.md, ~lines 18–19 and ~lines 114–115] `k_proj` and `v_proj` shapes are stated as `[512, 7168]` with derivation `num_key_value_heads × head_dim = 4 × 128 = 512`. Ground truth: `num_key_value_heads=8`, so the correct shape is `[1024, 7168]`. A developer implementing weight loading from this table would allocate a tensor of the wrong size. Fix: replace every occurrence of `[512, 7168]` (for `k_proj`/`v_proj`) with `[1024, 7168]` and update the derivation to `8 × 128 = 1024`.

3. [shared_weight_shapes.md, ~lines 44–45 and ~line 57] States `num_experts = 256` and derives the router gate shape as `[256, 7168]`. Ground truth (Chapter 1): `num_experts=128`. Fix: replace 256 with 128 in the prose and in the router gate row (`[128, 7168]`).

4. [extra_weight_keys.md, ~lines 14–15] MTP head `k_proj` and `v_proj` rows repeat the `[512, 7168]` / `4 × 128 = 512` error from issue 2. Fix: same as issue 2 — change to `[1024, 7168]` / `8 × 128 = 1024`.

5. [extra_weight_keys.md, ~lines 29–47] The MTP parameter count math uses `2 × 512 × 7168 = 7,340,032` for k_proj + v_proj, which is based on the wrong `num_key_value_heads=4`. With the correct `num_key_value_heads=8`, k_proj + v_proj = `2 × 1024 × 7168 = 14,680,064`. The total therefore changes from `433,090,560` to `440,430,592` (≈ 440M, not ≈ 433M). Fix: update the k_proj/v_proj line to `2 × (1024 × 7168)`, update the subtotal to `14,680,064`, and update the grand total to `440,430,592 ≈ 440M`. Also update the prose reference "≈ 433M parameters" on line 50 accordingly. (Note: the Chapter 1 ground-truth value of ≈ 433M is itself derived from the same wrong `num_key_value_heads=4` assumption and should be revisited.)

## B Feedback — Pass 2

1. [shared_weight_shapes.md, ~line 13] "94 transformer layers (indices 0–93)" is wrong. Ground truth: `num_hidden_layers = 40`, so there are 40 layers with indices 0–39. A developer writing layer-dispatch code from this line would iterate over 94 layers and produce out-of-range index accesses. Fix: replace "94 transformer layers (indices 0–93)" with "40 transformer layers (indices 0–39)" and apply the same correction to the identically wrong statement on ~line 30 (Layer Norm section).

2. [index.md, ~line 25] Summary Finding states "These keys total approximately 433 million parameters." Pass 1 corrected the arithmetic in `extra_weight_keys.md` to 440,430,592 ≈ 440M, but the index summary was not updated. A reader who only scans the index will carry the wrong 433M figure. Fix: change "approximately 433 million parameters" to "approximately 440 million parameters".

## B Feedback Application Log — Pass 1

- Fix 1: Changed "4 KV heads" to "8 KV heads" in index.md prerequisites table
- Fix 2: Corrected k_proj/v_proj shapes from [512, 7168] to [1024, 7168] (num_kv_heads 4→8) in shared_weight_shapes.md
- Fix 3: Corrected num_experts from 256 to 128, router gate shape from [256, 7168] to [128, 7168] in shared_weight_shapes.md
- Fix 4: Corrected k_proj/v_proj shapes from [512, 7168] to [1024, 7168] in extra_weight_keys.md MTP weight table
- Fix 5: Updated MTP parameter count arithmetic: k+v subtotal 7,340,032→14,680,064; grand total 433,090,560→440,430,592 (≈433M→≈440M) in extra_weight_keys.md

## B Feedback Application Log — Pass 2

- Fix 1: Corrected backbone layer count from 94/0–93 to 40/0–39 in shared_weight_shapes.md
- Fix 2: Updated MTP parameter count in index.md summary from 433M to 440M

## B Feedback — Pass 3

1. [shared_weight_shapes.md, ~lines 13–24] The Attention Projection Weights section states these shapes apply "uniformly to all 40 transformer layers (indices 0–39)" with no qualification. Ground truth: layers 0–29 use linear attention and layers 30–39 use full attention — two distinct mechanisms. A developer reading this section would apply the same full-attention TTNN module dispatch to all 40 layers, which is incorrect for layers 0–29. Fix: add a caveat stating that while the listed weight key names and shapes are present in all 40 layers, the attention mechanism differs between layers 0–29 (linear attention) and layers 30–39 (full attention); layer-dispatch code must select the correct TTNN module per layer index.

## B Feedback Application Log — Pass 3

- Fix 1: Added caveat in Attention Projection Weights section of shared_weight_shapes.md clarifying that q/k/v/o_proj shapes apply to full-attention layers (30–39) only; linear attention layers (0–29) use different projection keys via TTNNQwen3LinearAttention

## B Feedback — Pass 4

1. [shared_weight_shapes.md, ~line 13] The section opening sentence still reads "These shapes apply uniformly to all 40 transformer layers (indices 0–39)" — directly contradicting the Pass 3 caveat note immediately below it, which states these shapes apply to full-attention layers (30–39) only. A developer reading the opening sentence before reaching the note would implement layer dispatch across all 40 layers using full-attention projections, which is wrong for layers 0–29. Fix: change the opening sentence to "These shapes apply to the full-attention layers (indices 30–39). Linear attention layers (indices 0–29), handled by `TTNNQwen3LinearAttention`, use a different projection structure — see the caveat note below."

## B Feedback Application Log — Pass 4

- Fix 1: Updated Attention Projection Weights opening sentence in shared_weight_shapes.md to scope shapes to full-attention layers (30–39) only, matching the caveat note already present

## B Feedback — Pass 5

**No feedback — chapter approved.**

## B Feedback — Pass 6

**No feedback — chapter approved.**
