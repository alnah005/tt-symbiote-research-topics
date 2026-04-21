# Mathematical Equivalence Proof: M-RoPE Reduces to Standard RoPE for Text-Only Inputs

## 1. Formal Statement

Let the cos/sin frequency table have shape `[max_seq_len, 32]`, constructed with `rope_theta = 1000000.0` and `rotary_dim = 64` (32 pairs), as specified in the Qwen3.6-35B-A3B configuration. Let `mrope_section = [11, 11, 10]` with `s_t=11, s_h=11, s_w=10`.

**Proposition.** For any batch index `b` and sequence position `s`, if:

```
position_ids[0, b, s] == position_ids[1, b, s] == position_ids[2, b, s] == t
```

then the assembled M-RoPE cos vector satisfies:

```
cos_assembled[b, s, :] == [cos(t·θ_0), ..., cos(t·θ_31), cos(t·θ_0), ..., cos(t·θ_31)]
```

which equals `cat([cos_table[t, :], cos_table[t, :]])` — identical to the standard 1D partial RoPE full cos vector for position `t`.

## 2. Proof by Substitution

The three-gather + duplication construction (derived in Ch1 `section_dimension_assignment.md`) is:

```python
cos_assembled = torch.cat([
    cos_table[position_ids[0], :s_t],          # temporal: [B, S, 11]
    cos_table[position_ids[1], s_t:s_t+s_h],   # height:   [B, S, 11]
    cos_table[position_ids[2], s_t+s_h:],       # width:    [B, S, 10]
], dim=-1)                                      # → [B, S, 32]
cos_full = torch.cat([cos_assembled, cos_assembled], dim=-1)  # → [B, S, 64]
```

Substitute `position_ids[0] = position_ids[1] = position_ids[2] = t_seq` (the `[B, S]` tensor of sequential positions):

- **Temporal gather:** `cos_table[t_seq, :11]` — retrieves columns 0–10 of the table row for each position `t_seq`.
- **Height gather:** `cos_table[t_seq, 11:22]` — retrieves columns 11–21 of the same row `t_seq`.
- **Width gather:** `cos_table[t_seq, 22:32]` — retrieves columns 22–31 of the same row `t_seq`.

After concatenation:

```
cat([cos_table[t_seq, :11], cos_table[t_seq, 11:22], cos_table[t_seq, 22:32]])
  = cos_table[t_seq, :32]
  = cos_1d[t_seq]          # the standard 1D partial RoPE half-vector
```

After duplication:

```
cat([cos_table[t_seq, :32], cos_table[t_seq, :32]])
  = cat([cos_1d[t_seq], cos_1d[t_seq]])
  = standard 1D RoPE full cos vector for positions t_seq
```

This equals the output of a standard `cos_table[t_seq, :]` lookup followed by the usual `cat([cos_half, cos_half], dim=-1)` duplication. **QED.**

The following Python snippet confirms this numerically, validating both the 32-wide half-vector and the full 64-wide duplication:

```python
import torch

B, S = 2, 16
T = 128  # max_seq_len
rotary_dim = 64
s_t, s_h, s_w = 11, 11, 10

# Build a representative cos_table
rope_theta = 1_000_000.0
inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
t = torch.arange(T).float()
freqs = torch.outer(t, inv_freq)         # [T, 32]
cos_table = freqs.cos()                  # [T, 32]

# Sequential text position IDs: [B, S]
position_ids_text = torch.arange(S).unsqueeze(0).expand(B, -1)  # [B, S]

# M-RoPE assembled (text-only: all three axes identical)
cos_mrope = torch.cat([
    cos_table[position_ids_text, :s_t],
    cos_table[position_ids_text, s_t:s_t+s_h],
    cos_table[position_ids_text, s_t+s_h:],
], dim=-1)  # [B, S, 32]

# Standard 1D RoPE
cos_1d = cos_table[position_ids_text, :]  # [B, S, 32]

assert torch.allclose(cos_mrope, cos_1d), "Must be identical for text-only inputs"

# Duplication step — validates the full 64-wide proposition from Section 1
cos_1d_full = torch.cat([cos_1d, cos_1d], dim=-1)            # [B, S, 64]
cos_mrope_full = torch.cat([cos_mrope, cos_mrope], dim=-1)    # [B, S, 64]
assert torch.allclose(cos_mrope_full, cos_1d_full), "Full 64-wide cos must also be identical"
```

## 3. Coverage Argument

The three sections `[0, 11)`, `[11, 22)`, and `[22, 32)` partition the half-index range `[0, 32)` exactly:

- No gaps: `0 → 11 → 22 → 32` — every pair index in `[0, 32)` belongs to exactly one section.
- No overlaps: the sections are defined by half-open intervals with abutting boundaries; `s_t + s_h + s_w = 11 + 11 + 10 = 32 = rotary_dim/2` by construction.

Consequently, `cat([temporal_slice, height_slice, width_slice])` reassembles the entire half-frequency range `[0, 32)` in order. No frequency pair is counted twice; none is omitted. The union of the three sections equals the full standard partial RoPE coverage over `rotary_dim = 64` real dimensions.

## 4. The Silent Failure Caveat

The proof in Section 2 holds **if and only if** all three rows of `position_ids` are identical for every token. If they are not — for example, due to a bug that sets the height and width rows to zero while leaving the temporal row as sequential values — then the substitution does not apply and the assembled cos diverges from standard RoPE:

```
# Bug: temporal = sequential, height = 0, width = 0
cos_temporal = cos_table[t_seq, :11]        # correct temporal values
cos_height   = cos_table[zeros, 11:22]      # columns 11-21 of row 0 for every token
cos_width    = cos_table[zeros, 22:32]      # columns 22-31 of row 0 for every token
```

The resulting `cos_assembled` encodes position `t` only in the first 11 columns and encodes position `0` in the remaining 21 columns. Attention scores that depend on those 21 columns will be numerically wrong relative to standard RoPE. The model will not raise an error. Outputs will be silently incorrect.

This is the most dangerous failure mode in an M-RoPE TTNN implementation: a text-only test suite will pass (because text-only position IDs are always equal across axes), but any downstream vision evaluation will fail in a way that does not immediately point to RoPE as the root cause.

> **[SILENT FAILURE]** If any row of `position_ids` is incorrectly set for text tokens (e.g., height or width row left at zero), the assembled cos/sin diverges from standard RoPE across 21 of 32 frequency pairs. No error is raised. Text-only benchmarks remain unaffected. Validate position IDs explicitly in tests.

---
**Next:** [`practical_implications_for_text_inference.md`](./practical_implications_for_text_inference.md)
