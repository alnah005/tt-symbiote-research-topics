# B Review — Chapter 3 — Pass 1

## Item 1 — KV cache head-count contradiction between Section 6 and Section 8 (decode)

**File:** `gated_attention_formulation.md`

**Location:** Section 6 (KV Cache Update) vs. Section 8 (SDPA, Decode block)

**Problem:** Section 6 states the KV cache is stored with `n_kv_h = 2` heads:

```
K cache: [B, n_kv_h, T_max, d_h] = [B, 2, T_max, 256]
V cache: [B, n_kv_h, T_max, d_h] = [B, 2, T_max, 256]
```

Section 8's decode block then reads those same cache tensors as:

```
K_cache:  [B, 16, S, 256]
V_cache:  [B, 16, S, 256]
```

This is a direct contradiction. The cache is written with 2 heads and read back as 16. A reader implementing decode attention would not know whether to allocate the KV cache buffer for 2 heads or 16 heads. The correct picture is: the cache stores 2-head K and V; the GQA 8× expansion is applied after the cache read (not before the cache write). The decode SDPA block should show `K_cache [B, 2, S, 256]` and `V_cache [B, 2, S, 256]`, with an explicit repeat step to `[B, 16, S, 256]` before the dot product.

---

## Item 2 — Decode SDPA is missing the `/ sqrt(d_h)` scale factor

**File:** `gated_attention_formulation.md`

**Location:** Section 8, Decode block

**Problem:** The prefill block explicitly writes:

```
scores = (Q_rope @ K_exp^T) / sqrt(d_h)
```

The decode block writes:

```
scores = Q_rope @ K_cache^T  →  [B, 16, 1, S]
```

No division by `sqrt(d_h)` appears anywhere in the decode formulation. An implementer following the decode path literally would omit the scale factor, producing raw dot-product scores that are 16× too large (`sqrt(256) = 16`), which would cause the softmax to be far too peaked and produce incorrect attention distributions. The scale factor `1/sqrt(d_h) = 1/16` must be applied identically in both prefill and decode.

---

## Agent A Change Log — Pass 1

**File edited:** `gated_attention_formulation.md`

**Item 1 applied — KV cache head-count contradiction (Section 8, decode block):**
The decode SDPA block previously showed `K_cache [B, 16, S, 256]` and `V_cache [B, 16, S, 256]`, contradicting Section 6's allocation of the cache at `n_kv_h = 2` heads. The block now reads the cache as `[B, 2, S, 256]` for both K and V, adds an explicit `repeat 8×` step to expand to `[B, 16, S, 256]`, and carries a clarifying note that the cache is always allocated and written with 2 heads — the GQA expansion occurs after the cache read, not before the cache write.

**Item 2 applied — Decode SDPA missing scale factor (Section 8, decode block):**
The scores line `Q_rope @ K_cache^T` was missing the `/ sqrt(d_h)` scale factor. Updated to `(Q_rope @ K_cache^T) / sqrt(d_h)` = `(Q_rope @ K_cache^T) / 16`, matching the prefill block.

**Item 3 — NOT applied:**
`index.md` is explicitly exempt from the navigation footer requirement. No change made.

---

## Item 3 — `index.md` is missing the required navigation footer

**File:** `index.md`

**Location:** End of file (line 40)

**Problem:** The chapter requirement states that every content file must end with a navigation footer. `index.md` ends abruptly after the symbols table with no footer linking to the previous chapter or to the first section. Both `gated_attention_formulation.md` and `gated_vs_vanilla_attention_shapes.md` have correct `**Next:**` footers. `index.md` does not, leaving a reader with no navigational anchor from the chapter entry point.

---

# B Review — Chapter 3 — Pass 2

## Item 1 — Decode SDPA block reuses `K_cache` / `V_cache` as both 2-head and 16-head tensors

**File:** `gated_attention_formulation.md`

**Location:** Section 8, Decode block — GQA expansion step and scores line

**Problem:** After the Pass 1 fix, the decode block now reads:

```
  -- read from KV cache (stored with n_kv_h = 2 heads) --
  K_cache:  [B, 2, S, 256]
  V_cache:  [B, 2, S, 256]

  -- GQA 8× expansion (after cache read, before SDPA) --
  K_cache:  [B, 2, S, 256]  → repeat 8×  →  [B, 16, S, 256]
  V_cache:  [B, 2, S, 256]  → repeat 8×  →  [B, 16, S, 256]

  scores = (Q_rope @ K_cache^T) / sqrt(d_h)
         = (Q_rope @ K_cache^T) / 16        →  [B, 16, 1, S]
```

The identifier `K_cache` is used for both the 2-head cache buffer (`[B, 2, S, 256]`) and the 16-head expanded tensor (`[B, 16, S, 256]`). The GQA expansion line shows `K_cache [B, 2, S, 256] → repeat 8× → [B, 16, S, 256]` but does not introduce a distinct name for the expanded result. The scores line then references `K_cache^T`; a reader implementing this literally would be uncertain whether to transpose the 2-head buffer or the 16-head expanded tensor.

An implementer who substitutes the 2-head `K_cache [B, 2, S, 256]` directly into the dot product `[B, 16, 1, 256] @ [B, 2, 256, S]` would produce an incorrect result or a shape error. The expanded tensor must be given a distinct name (e.g., `K_exp` / `V_exp`, matching the prefill convention already used in Section 7) so that the scores line unambiguously references the 16-head tensor.

---

## No further items.

All other numerical claims verified: layer count (indices 3,7,…,39 = 10 layers), shape arithmetic throughout Sections 2–9, `sqrt(256) = 16`, GQA expand factors (8× for Gated Attention, 4× for the GQA example), KV cache byte calculations (2×256×2×2 = 2,048 bytes/token, 4× ratio vs. MHA example), gate memory table (T=512: 4 MB, T=2048: 16 MB, T=8192: 64 MB), and Gated Delta Net comparison values (`H_v × d_v = 32 × 128 = 4,096`) are all correct.

---

## Agent A Change Log — Pass 2

**File edited:** `gated_attention_formulation.md`

**Item 1 applied — Ambiguous `K_cache`/`V_cache` names in decode SDPA block (Section 8):**
The GQA expansion step in the decode block previously wrote `K_cache [B, 2, S, 256] → repeat 8× → [B, 16, S, 256]` and `V_cache [B, 2, S, 256] → repeat 8× → [B, 16, S, 256]`, leaving both the 2-head cache buffer and the 16-head expanded result under the same `K_cache`/`V_cache` names. The scores, weights, and output lines then referenced `K_cache^T` and `V_cache`, creating ambiguity about which shape was being used. The expansion step now assigns distinct names `K_exp: [B, 16, S, 256]` and `V_exp: [B, 16, S, 256]` for the post-expansion tensors, consistent with the naming already used in Section 7 (prefill). The scores line reads `(Q_rope @ K_exp^T) / sqrt(d_h)` and the output line reads `weights @ V_exp`, unambiguously referencing the 16-head expanded tensors throughout.

---

# B Review — Chapter 3 — Pass 3

## Item 1 — Weight shape definition contradicts the `weight^T` notation in every projection

**File:** `gated_attention_formulation.md`

**Location:** Sections 2.1, 2.2, and 9 — all three linear projection blocks

**Problem:** Each projection defines the weight with dimensions `[in, out]` and then writes `weight^T [in, out]` — the same shape — in the matrix multiplication. Concretely, Section 2.1 states:

```
q_gate_proj weight: [H, n_q_h × d_h × 2] = [2048, 8192]

hidden_states [B, T, 2048] × weight^T [2048, 8192]  →  q_gate_raw [B, T, 8192]
```

If `weight` is `[2048, 8192]` then `weight^T` is `[8192, 2048]`, not `[2048, 8192]`. The matmul `[B, T, 2048] @ [8192, 2048]` does not produce `[B, T, 8192]`; it would either fail on shape or produce `[B, T, 2048]` if broadcast rules applied.

For the arithmetic to be correct, the weight must be stored as `[out, in]` = `[8192, 2048]` (the PyTorch `nn.Linear` convention), so that `weight^T = [2048, 8192]` and `[B, T, 2048] @ [2048, 8192] = [B, T, 8192]`. The document states the weight shape the wrong way around in all three places. A reader who allocates a weight buffer of shape `[2048, 8192]` and then transposes it before the matmul will get a `[8192, 2048]` operand, causing a shape error or silently wrong output. The same error appears in:

- Section 2.2: `kv_proj weight: [H, n_kv_h × d_h × 2] = [2048, 1024]` — should be `[1024, 2048]`
- Section 9: `o_proj weight: [n_q_h × d_h, H] = [4096, 2048]` — should be `[2048, 4096]`

---

## No further items.

All other claims verified independently: layer indices (3, 7, 11, …, 39 = 10 layers correct), all shape splits and reshape products, `sqrt(256) = 16`, GQA expand factors (8× Gated Attention, 4× GQA example), KV cache byte calculations, gate memory table entries, rotary fraction (64/256 = 25%), and the Gated Delta Net comparison values are all numerically correct.

---

## Agent A Change Log — Pass 3

**File edited:** `gated_attention_formulation.md`

**Item 1 applied — Weight shape/transpose notation inconsistency (Sections 2.1, 2.2, and 9):**
All three projection blocks previously declared the weight in `[in, out]` order while writing `weight^T [in, out]` in the matmul — a contradiction, because transposing an `[in, out]` weight yields `[out, in]`, not `[in, out]`. The fix adopts Option A (PyTorch `nn.Linear` convention) throughout: weight is stored as `[out, in]` so that `weight^T = [in, out]`, and the matmul `x @ weight^T` is dimensionally consistent.

Specific changes:
- Section 2.1: `q_gate_proj weight` declaration changed from `[H, n_q_h × d_h × 2] = [2048, 8192]` to `[n_q_h × d_h × 2, H] = [8192, 2048]`. The matmul annotation `weight^T [2048, 8192]` is now correct (transpose of `[8192, 2048]` = `[2048, 8192]`).
- Section 2.2: `kv_proj weight` declaration changed from `[H, n_kv_h × d_h × 2] = [2048, 1024]` to `[n_kv_h × d_h × 2, H] = [1024, 2048]`. The matmul annotation `weight^T [2048, 1024]` is now correct (transpose of `[1024, 2048]` = `[2048, 1024]`).
- Section 9: `o_proj weight` declaration changed from `[n_q_h × d_h, H] = [4096, 2048]` to `[H, n_q_h × d_h] = [2048, 4096]`. The matmul annotation `weight^T [4096, 2048]` is now correct (transpose of `[2048, 4096]` = `[4096, 2048]`).

---

# B Review — Chapter 3 — Pass 4

## Item 1 — Weight shape convention in `gated_vs_vanilla_attention_shapes.md` contradicts the corrected `gated_attention_formulation.md`

**File:** `gated_vs_vanilla_attention_shapes.md`

**Location:** Section 2, Difference 1 (lines 76–77)

**Problem:** After Pass 3 fixed all weight shape declarations in `gated_attention_formulation.md` to use `[out, in]` (PyTorch `nn.Linear`) convention, `gated_vs_vanilla_attention_shapes.md` still states the same weights in `[in, out]` order:

```
Vanilla Q proj weight:       [H, n_q_h × d_h]     = [2048, 4096]
Gated Attention Q+gate proj: [H, n_q_h × d_h × 2] = [2048, 8192]
```

The corrected canonical source now declares:

```
q_gate_proj weight: [n_q_h × d_h × 2, H] = [8192, 2048]
```

A reader implementing from `gated_vs_vanilla_attention_shapes.md` would allocate a weight buffer of `[2048, 8192]` (or `[2048, 4096]` for vanilla Q), which is the transpose of the correct shape. Multiplying `hidden_states [B, T, 2048] @ weight^T` with a `[2048, 8192]` weight would produce `weight^T = [8192, 2048]`, causing a shape mismatch or silently wrong output (`[B, T, 2048] @ [8192, 2048]` is invalid for standard matmul). The two files in this chapter now give contradictory shapes for the same physical parameter.

---

## No further items.

All other claims verified: shape arithmetic throughout both files, `sqrt(256) = 16`, layer count (10 layers at indices 3, 7, 11, …, 39), GQA expand factors (8× Gated Attention, 4× GQA example), KV cache byte totals (2 × 256 × 2 × 2 = 2,048 bytes/token per layer, 20 KB across 10 layers), gate memory table entries (T=512: 4 MB, T=2048: 16 MB, T=8192: 64 MB), KV cache 4× ratio vs. MHA example, rotary fraction (64/256 = 25%), and Gated Delta Net comparison values (32 × 128 = 4,096) are all numerically correct. The decode SDPA block in `gated_attention_formulation.md` is consistent with the KV cache allocation and the GQA expansion naming after Passes 1–3.

---

## Agent A Change Log — Pass 4

**File edited:** `gated_vs_vanilla_attention_shapes.md`

**Item 1 applied — Weight shape convention mismatch in Section 2, Difference 1:**
The weight shape code block in Difference 1 previously declared both weights in `[in, out]` order:

```
Vanilla Q proj weight:       [H, n_q_h × d_h]     = [2048, 4096]
Gated Attention Q+gate proj: [H, n_q_h × d_h × 2] = [2048, 8192]
```

This contradicted `gated_attention_formulation.md` after its Pass 3 correction, which adopted `[out, in]` (PyTorch `nn.Linear`) convention throughout. The code block now reads:

```
Vanilla Q proj weight:       [n_q_h × d_h, H]     = [4096, 2048]
Gated Attention Q+gate proj: [n_q_h × d_h × 2, H] = [8192, 2048]
```

A convention note was also added immediately after the code block, clarifying that all weight shapes in this chapter follow the PyTorch `nn.Linear` convention (`[out_features, in_features]`) so that `x @ weight^T` is dimensionally consistent.

---

# B Review — Chapter 3 — Pass 5

No feedback — chapter approved.

# B Review — Chapter 3 — Pass 6

No feedback — chapter approved.
