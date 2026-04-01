# Chapter 6: L1 State Management and Rolling Window (WIP)

GDN layers consume 85% of total decode time, and the dominant cost within each layer is DRAM bandwidth for reading and writing the recurrence state. This chapter covers the work-in-progress effort to move GDN recurrence states from DRAM into L1 using a rolling window of 3 layers and a HEIGHT_SHARDED memory layout that eliminates NOC transfers entirely.

## Files

| File | Description |
|------|-------------|
| [`l1_state_design.md`](./l1_state_design.md) | Rolling window L1 state approach: `enable_l1_state()`, `_swap_l1_state()`, and forward pass hooks |
| [`height_sharded_kernel.md`](./height_sharded_kernel.md) | HEIGHT_SHARDED L1 state support in the custom fused GDN kernel |
| [`sdpa_l1_conflict.md`](./sdpa_l1_conflict.md) | The SDPA circular buffer conflict that blocks full L1 state deployment |

## Process Files

- [`b_review.md`](./b_review.md) — Correctness review identifying writer kernel compile-time arg index, output tensor shape, and reader line range issues
- [`compression_analysis.md`](./compression_analysis.md) — Compression analysis pass 1

---

**Next:** [`l1_state_design.md`](./l1_state_design.md)
