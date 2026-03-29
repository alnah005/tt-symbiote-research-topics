# Chapter 4: Custom Fused GDN Kernel

The fused GDN kernel is the single most important optimization in the Qwen3.5-27B deployment. It replaces what would otherwise be dozens of separate `ttnn` op dispatches -- L2 normalization, gate computation, and the full DeltaNet recurrence -- with a single `ttnn.generic_op` call that executes a custom reader/compute/writer kernel triplet on device. The unfused path requires approximately 30 separate kernel launches per GDN layer per decode step; the fused kernel reduces this to one, eliminating host-device dispatch overhead and enabling fine-grained control over data movement and circular buffer scheduling.

The kernel follows the standard tt-metal three-kernel architecture: a **reader** dataflow kernel that fetches inputs from DRAM (or L1) into circular buffers, a **compute** kernel that processes the data through L2 normalization, gating, and recurrence phases, and a **writer** dataflow kernel that drains the results back to memory. All three run concurrently on each assigned core, synchronized through circular buffer semaphores: the reader pushes tiles into CBs, the compute kernel waits for them, produces output tiles, and the writer drains them.

The unit of work is a **pair** -- a (batch, value_head) combination. With `B=32` and `Nv_TP=12` value heads per device, each device processes `num_pairs = 384` pairs per decode step. These pairs are distributed across up to 40 compute cores using a `pairs_per_core + remainder` assignment pattern.

## Learning Objectives

After reading this chapter you will understand:

- How the Python-side dispatch in `gdn_kernel_op.py` constructs `ProgramDescriptor` and `MeshProgramDescriptor` objects for single-device and multi-device execution
- The 28 circular buffer descriptors and their roles in the data pipeline
- How the reader kernel uses batched NOC reads with a single barrier to extract Q/K/V rows and scalar gates from packed tensors via sub-tile addressing
- The five compute phases: L2 norm Q, L2 norm K, K transpose, gate computation, and DeltaNet recurrence
- How the writer kernel handles both DRAM-interleaved and HEIGHT_SHARDED L1 state writeback paths
- The compile-time and runtime argument conventions that parameterize all three kernels

## Files

| File | Description |
|------|-------------|
| [`kernel_dispatch.md`](./kernel_dispatch.md) | Python-side dispatch via `gdn_kernel_op.py` and `ttnn.generic_op` |
| [`reader_kernel.md`](./reader_kernel.md) | Reader dataflow kernel: batched NOC reads, sub-tile extraction, scratch buffer layout |
| [`compute_kernel.md`](./compute_kernel.md) | Compute kernel: L2 norm, gates, recurrence phases with CB flow |
| [`writer_kernel.md`](./writer_kernel.md) | Writer kernel: output and state writeback for DRAM and L1 paths |

See [`kernel_dispatch.md`](./kernel_dispatch.md) to begin.

---

**Next:** [`kernel_dispatch.md`](./kernel_dispatch.md)
