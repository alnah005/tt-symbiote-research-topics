# Chapter 3: GDN Layer Decode Pipeline

Gated DeltaNet (GDN) layers make up 48 of the 64 layers in Qwen3.5-27B and replace the KV cache with a fixed-size recurrence state tensor, introducing a trace-compatible causal conv1d shift register, a multi-step DeltaNet recurrence, and a head expansion from 4 key heads to 12 value heads per device.

## Files

| File | Description |
|------|-------------|
| [`gdn_decode_flow.md`](./gdn_decode_flow.md) | End-to-end decode dataflow for a single GDN layer, covering both fused and unfused paths |
| [`conv1d_shift_register.md`](./conv1d_shift_register.md) | The 4-tap causal conv1d implemented as a trace-compatible shift register |
| [`recurrence_math.md`](./recurrence_math.md) | DeltaNet recurrence equations and their mapping to tensor operations |

## Process Files

The following files are internal pipeline artifacts and are not part of the reading sequence:

- [`b_review.md`](./b_review.md) — Editorial review notes for this chapter
- [`compression_analysis.md`](./compression_analysis.md) — Content compression analysis

---

**Next:** [`gdn_decode_flow.md`](./gdn_decode_flow.md)
