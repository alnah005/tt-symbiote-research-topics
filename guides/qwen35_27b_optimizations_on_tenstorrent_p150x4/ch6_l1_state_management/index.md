# Chapter 6: L1 State Management and Rolling Window (WIP)

The profiler breakdown for a non-traced decode step reveals that GDN layers consume **85% of total decode time**: 469.6 ms across 48 layers (9.78 ms/layer average) compared to 69.2 ms across 16 attention layers (4.33 ms/layer). The remaining 15.7 ms (3%) is overhead. The dominant cost within each GDN layer is not computation but **DRAM bandwidth** for reading and writing the recurrence state.

Each GDN layer maintains a recurrence state of shape `[B*Nv_TP, Dk, Dv]` = `[32*12, 128, 128]` per device. At bfloat16 precision, this is **12 MB per layer**. Every decode step reads the full state from DRAM, updates it through the fused kernel's recurrence phases, and writes it back. Across 48 GDN layers, this amounts to approximately **1.2 GB of DRAM bandwidth consumed per decode step** just for state I/O -- a round-trip read and write for each layer.

The L1 state optimization targets this bottleneck by moving recurrence states from DRAM into L1 (on-chip SRAM), where they can be accessed with significantly lower latency and higher bandwidth. The approach uses a **rolling window** of 3 layers to fit within L1 capacity constraints, and a **HEIGHT_SHARDED** memory layout to place state tiles directly on the compute cores that need them -- eliminating NOC transfers entirely for state reads and writes.

This optimization is a work in progress. L1 INTERLEAVED state has been validated with 4 layers producing correct output. HEIGHT_SHARDED state has been validated for 1-2 layers with correct "Paris" output. The remaining challenge is a conflict between SDPA circular buffer usage during attention layers and the L1 address space occupied by GDN states.

## Learning Objectives

After reading this chapter you will understand:

- Why DRAM bandwidth for GDN recurrence state is the primary decode bottleneck, and how 12 MB per layer across 48 layers produces ~1.2 GB of state I/O per step
- How `enable_l1_state()` loads a rolling window of 3 GDN layers' states into L1 and how `_swap_l1_state()` rotates groups around attention layer boundaries
- The forward pass hook mechanism that injects swap logic without modifying the parent `Transformer` class
- How the custom kernel's `STATE_IS_SHARDED` compile-time flag switches between NOC-based DRAM reads and direct L1 memcpy for state access
- The SDPA circular buffer conflict that currently prevents full HEIGHT_SHARDED deployment

## Files

| File | Description |
|------|-------------|
| [`l1_state_design.md`](./l1_state_design.md) | Rolling window L1 state approach: `enable_l1_state()`, `_swap_l1_state()`, and forward pass hooks |
| [`height_sharded_kernel.md`](./height_sharded_kernel.md) | HEIGHT_SHARDED L1 state support in the custom fused GDN kernel |
| [`sdpa_l1_conflict.md`](./sdpa_l1_conflict.md) | The SDPA circular buffer conflict that blocks full L1 state deployment |

See [`l1_state_design.md`](./l1_state_design.md) to begin.

---

**Next:** [`l1_state_design.md`](./l1_state_design.md)
