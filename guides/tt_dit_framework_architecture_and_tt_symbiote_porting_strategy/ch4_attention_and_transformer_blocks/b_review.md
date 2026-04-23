# B Review -- Chapter 4: Attention and Transformer Blocks (Pass 1)

## Issue 1: Incorrect claim about when TTNNGR00TSelfAttention applies Q/K RMSNorm

**File:** `comparison_with_symbiote_attention.md`, Section "2. Per-Head QKV Normalization"

**Claim (lines 127-128):**
> "However, these norms are `TTNNRMSNorm` instances mapped from the source model's `q_norm` and `k_norm` modules, and they operate on the full projected tensor before head reshaping. The normalization happens at a different point in the computation graph and with different tensor shapes."

**Problem:** The source code in `modules/attention.py` (TTNNGR00TSelfAttention.forward, around lines 1363-1372) shows the opposite order. `prepare_heads_on_device` reshapes from `(B, seq, H)` to `(B, num_heads, seq, d_head)` first, and only then are the norms applied on the resulting 4D tensor:

```python
q_4d, b, q_len, h, d_head, q_pad_s, q_pad_d = prepare_heads_on_device(q_w, self.num_heads, apply_pad=False)
k_4d, _, kv_len, _, _, kv_pad_s, kv_pad_d = prepare_heads_on_device(k_w, self.num_kv_heads, apply_pad=False)
...
if self.tt_q_norm is not None:
    q_4d = self._rms_norm_on_device(q_4d, self.tt_q_norm, hw_dev)
if self.tt_k_norm is not None:
    k_4d = self._rms_norm_on_device(k_4d, self.tt_k_norm, hw_dev)
```

The norm is applied **after** head reshaping, on the 4D tensor -- the same ordering as TT-DiT. The chapter should state that TTNNGR00TSelfAttention actually applies Q/K norms after head splitting (on 4D tensors), which is closer to TT-DiT's pattern than the text suggests. The claimed difference ("different point in the computation graph and with different tensor shapes") is incorrect for this class.

**Suggested fix:** Replace the claim with an accurate description: the norms in TTNNGR00TSelfAttention are applied after head reshaping on 4D tensors, similar to TT-DiT. The actual differences are: (a) TTNNGR00TSelfAttention only applies these norms in the self-attention path (not cross-attention), and (b) the norm weights come from the source PyTorch model rather than being part of a fused QKV pipeline.

---

No other factual errors, incorrect implementations, or material misconceptions were found. The remaining content in all four chapter files accurately reflects the source code in `blocks/attention.py`, `blocks/transformer_block.py`, and `modules/attention.py`.

---

# B Review -- Chapter 4: Attention and Transformer Blocks (Pass 2)

## Issue 1: Incorrect claim that per-head RMSNorm replaces 1/sqrt(d) scaling

**Files:** `joint_attention.md` (Per-Head RMSNorm section, line ~109; Key Takeaways item 2, line ~382), `index.md` (Key Takeaways item 2, line ~110)

**Claim:**
> "Replaces 1/sqrt(d) scaling: by normalizing Q and K to unit RMS, the dot products are naturally bounded, making the traditional 1/sqrt(d_k) scaling factor unnecessary."

**Problem:** TT-DiT does not pass an explicit `scale` parameter to `ttnn.transformer.joint_scaled_dot_product_attention` (see `attention.py` lines 305-315). TTNN's SDPA kernels, like every standard SDPA implementation (PyTorch, Flash Attention, etc.), apply 1/sqrt(d_k) scaling by default when no explicit scale is provided. The per-head RMSNorm operates **in addition to** the standard SDPA scaling, not as a replacement. Claiming that RMSNorm makes the scaling "unnecessary" is incorrect -- both are active simultaneously. Compare with `TTNNGR00TSelfAttention` (line 1528), which explicitly passes `scale=float(d_head**-0.5)` to the non-joint SDPA, confirming the kernel expects a scale parameter and has a default.

**Suggested fix:** State that per-head RMSNorm is applied to bound Q/K magnitudes for training stability, and note that the standard 1/sqrt(d_k) scaling is still applied by the SDPA kernel. Remove the claim that one replaces the other.

---

## Issue 2: Index.md claims per-head Q/K norm is absent from all TT-Symbiote attention modules

**File:** `index.md`, Key Takeaways item 2 (line ~110) and the comparison table row "Head normalization" (line ~29)

**Claims:**
> Key Takeaway: "This is a normalization strategy not present in any TT-Symbiote attention module"
> Table row: "Head normalization | None (or pre-norm on full hidden state) | Per-head RMSNorm on Q and K separately"

**Problem:** `TTNNGR00TSelfAttention` has optional per-head Q/K RMSNorm applied after head reshaping on 4D tensors (lines 1369-1372 of `modules/attention.py`). The Pass 1 fix correctly updated `comparison_with_symbiote_attention.md` to acknowledge this, but `index.md` still says the feature is "not present in any TT-Symbiote attention module" and the table says "None" for TT-Symbiote head normalization. These statements now contradict the corrected comparison file.

**Suggested fix:** Update the table to say "Optional per-head Q/K RMSNorm in TTNNGR00TSelfAttention; none in other classes" and update the Key Takeaway to say the pattern exists in TTNNGR00TSelfAttention but not in the other attention classes.

---

No other factual errors, incorrect implementations, or material misconceptions were found in this pass. The Pass 1 issue (GR00T norm ordering) has been correctly fixed in `comparison_with_symbiote_attention.md`.

---

# B Review -- Chapter 4: Attention and Transformer Blocks (Pass 3)

**No feedback -- chapter approved.**

All four chapter files were verified against the source code in `blocks/attention.py`, `blocks/transformer_block.py`, and `modules/attention.py`. Specific claims cross-checked:

- Fused QKV weight interleaving algorithm in `_reshape_and_merge_qkv` (joint_attention.md) matches source exactly.
- Per-head RMSNorm ordering (after QKV split, before RoPE) is accurately described and the Pass 2 fix correctly states both RMSNorm and 1/sqrt(d) scaling are active simultaneously.
- `context_pre_only` chunk ordering (scale-then-shift for 2 chunks vs. shift-then-scale for 6 chunks) in transformer_block.md matches the source code at lines 241-255.
- All-gather count (4 for context_pre_only, 6 otherwise) in the placement analysis table is correct.
- The `add_attention_to_output` flag behavior and feedforward norm input (`spatial_plus_attn` regardless of flag) are accurately described.
- Comparison table in `comparison_with_symbiote_attention.md` correctly reflects TTNNGR00TSelfAttention's per-head Q/K norm capability (after Pass 1 fix) and the two SDPA execution paths.
- Ring attention parameters (persistent ping-pong buffers, logical_n, CCL config) match source.
- UnregisteredModule weight-sharing mechanism is correctly described.
