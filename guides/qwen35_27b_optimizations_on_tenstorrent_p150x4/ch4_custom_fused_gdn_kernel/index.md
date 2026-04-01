# Chapter 4: Custom Fused GDN Kernel

The fused GDN kernel replaces dozens of separate `ttnn` op dispatches — L2 normalization, gate computation, and the full DeltaNet recurrence — with a single `ttnn.generic_op` call executing a custom reader/compute/writer kernel triplet on device.

The kernel follows the standard tt-metal three-kernel architecture: a **reader** dataflow kernel fetches inputs from DRAM (or L1) into circular buffers, a **compute** kernel processes data through L2 normalization, gating, and recurrence phases, and a **writer** dataflow kernel drains results back to memory. All three run concurrently on each assigned core, synchronized through circular buffer semaphores.

The unit of work is a **pair** — a (batch, value_head) combination. With $B=32$ and $N_{v,TP}=12$ value heads per device, each device processes $\text{num pairs} = 384$ pairs per decode step. These are distributed across up to 40 compute cores using a `pairs_per_core + remainder` assignment pattern.

## Files

| File | Description |
|------|-------------|
| [`kernel_dispatch.md`](./kernel_dispatch.md) | Python-side dispatch via `gdn_kernel_op.py` and `ttnn.generic_op` |
| [`reader_kernel.md`](./reader_kernel.md) | Reader dataflow kernel: batched NOC reads, sub-tile extraction, scratch buffer layout |
| [`compute_kernel.md`](./compute_kernel.md) | Compute kernel: L2 norm, gates, recurrence phases with CB flow |
| [`writer_kernel.md`](./writer_kernel.md) | Writer kernel: output and state writeback for DRAM and L1 paths |

## Process Files

| File | Description |
|------|-------------|
| [`b_review.md`](./b_review.md) | Correctness review (Agent B) |
| [`compression_analysis.md`](./compression_analysis.md) | Compression analysis — cross-file duplication suggestions |

---

**Next:** [`kernel_dispatch.md`](./kernel_dispatch.md)
