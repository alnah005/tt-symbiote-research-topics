# Causes of Low FPU Utilization

This file explains the seven root causes of low `FPU UTIL` on Wormhole B0. Each section covers the mechanism, how it manifests in measured data, and the targeted fix. Exact CSV column patterns for each cause are in [`csv_signatures.md`](./csv_signatures.md). Configuration parameters for each fix are in [`remediation_levers.md`](./remediation_levers.md).

(See Chapter 4 for the `FPU UTIL` definition, classification thresholds, and ridge point derivation.)

---

## Cause 1 — Insufficient Tile Count (Small M, K, or N)

### Mechanism

All work is distributed across Tensix cores as tiles. A matmul with shape (M, K, N) produces `M_t × N_t` output tiles, where `M_t = ⌈M/32⌉` and `N_t = ⌈N/32⌉`. Each core is responsible for a rectangular sub-block of the output tile grid. If the total number of output tiles is smaller than the number of active cores, some cores receive no tiles at all and execute zero FMA operations for that call. The cores that do receive work may carry a fractional tile count, leaving the FPU pipeline only partially filled.

The result: `CORE COUNT` is high (reflecting how many cores were allocated), but the arithmetic work is spread so thinly that each core's FPU is idle for most of its active kernel duration.

### Observable Effect

- `FPU UTIL < 0.2` despite the op being a matmul with theoretically high arithmetic intensity.
- `CORE COUNT` is large — often the full default grid (e.g., 8×8 = 64 cores) — relative to `M_t × N_t`.
- `TRISC1 KERNEL DURATION` is short because there is genuinely little work per core.
- `DEVICE KERNEL DURATION` is dominated by synchronization overhead rather than computation.

### Fix

Reduce the number of cores assigned to the op so that every active core has enough tiles to keep its FPU pipeline full. The guideline:

```
M_t × N_t / core_count ≥ 4
```

Use `compute_with_storage_grid_size` in the `ttnn.matmul` program config to specify a smaller grid. See [`remediation_levers.md`](./remediation_levers.md) for the parameter syntax.

An alternative when the operation sequence is fixed: pack multiple ops into a single kernel so the combined tile count is large enough to saturate the allocated cores. This is an advanced technique and typically requires custom kernel authoring.

> **Warning:** Reducing core count increases per-core compute time but decreases synchronization overhead. There is an optimal grid size for each (M, K, N) shape. Choosing a grid that is too small re-introduces throughput loss from under-parallelism; the goal is to find the crossover point where `M_t × N_t / core_count ≈ 4–8`.

---

## Cause 2 — Sub-Optimal Data Format (FP32 Where BF16 Suffices)

### Mechanism

On Wormhole B0, the FPU's native high-throughput mode is designed for 16-bit formats (BF16 and FP16). When operands are in FP32, the tile-based dataflow engine must perform wider accumulation and more internal FPU passes per 32×32 tile than it does for BF16. The matmul engine therefore delivers fewer effective FMA operations per cycle in FP32 mode than in BF16 mode — approximately half the throughput, because each FP32 tile requires more processing passes through the internal FPU pipeline.

The consequence: for an identical (M, K, N) shape, a FP32 matmul has the same tile count but each tile takes roughly twice as many cycles, while `PM IDEAL` (which is derived from a fixed FMA rate) does not scale up to reflect the slower FP32 FMA rate. The ratio `PM_IDEAL / TRISC1_DURATION` therefore appears lower than expected.

### Observable Effect

- `DATA FORMAT` column shows `"FLOAT32"`.
- `FPU UTIL` is approximately half of what a BF16 run of the same shape would produce.
- `TRISC1 KERNEL DURATION` is roughly twice the BF16 equivalent.

### Fix

Convert inputs and weights to BF16 (or BF8 where model accuracy permits) before the matmul:

```python
x_bf16 = ttnn.typecast(x, ttnn.bfloat16)
w_bf16 = ttnn.typecast(w, ttnn.bfloat16)
```

Or specify the output dtype directly in the op config to let ttnn handle the conversion internally. Evaluate whether your model's numerical precision requirements actually need FP32 — most inference workloads are unaffected by BF16 conversion, and many training workloads use BF16 for activations with FP32 only for the optimizer state.

> **Tip:** `ttnn.bfloat8_b` (BF8 block-float format) can provide an additional throughput benefit over BF16 for weight tensors in certain ops. Check [`remediation_levers.md`](./remediation_levers.md) for format selection guidance.

---

## Cause 3 — Math Fidelity Mismatch (HiFi4 Where LoFi Is Tolerable)

### Mechanism

Math fidelity controls how many internal FMA loop iterations the TRISC1 math engine performs per tile. Higher fidelity improves numerical precision at the cost of throughput: `HiFi4` runs ~4× more FMA iterations per tile than `LoFi`, and `HiFi2` runs ~2× more than `LoFi` (authoritative throughput table in [`remediation_levers.md`](./remediation_levers.md)).

Crucially, `PM IDEAL` accounts for fidelity proportionally. As shown in Ch3 (`pm_ideal_and_fpu_util.md`), the effective FLOPs/cycle used in PM IDEAL computation scales with fidelity (256 for HiFi4, 512 for HiFi2, 1024 for LoFi at BF8/LoFi). This means that switching from `LoFi` to `HiFi4` increases both `TRISC1_KERNEL_DURATION` and `PM_IDEAL_cycles` by the same factor. Consequently, **`FPU UTIL` (= PM_IDEAL / TRISC1_KERNEL_DURATION) remains roughly constant across fidelity modes** for a well-tuned op — it does not drop to 0.25× simply because `HiFi4` is in use.

The actual consequence of using `HiFi4` where `LoFi` suffices is that the **absolute kernel time is ~4× longer** (lower throughput), but this is fully reflected in both the numerator and denominator of `FPU UTIL`. A drop in `FPU UTIL` attributable to fidelity only occurs when the extra overhead from higher fidelity (pipeline stalls, register pressure, increased loop trip count) exceeds the proportional increase that PM IDEAL already models — for example, if `HiFi4` introduces micro-architectural stalls on top of the 4× iteration count that PM IDEAL does not capture.

### Observable Effect

- `MATH FIDELITY` column shows `"HiFi4"`.
- `TRISC1 KERNEL DURATION` is approximately 4× the duration of an equivalent `LoFi` run (slower absolute throughput).
- `FPU UTIL` is **not** necessarily lower than a LoFi run — PM IDEAL is also scaled up for HiFi4, so the ratio stays similar. A significant FPU UTIL drop relative to a LoFi baseline indicates overhead beyond what PM IDEAL models.

### Fix

Specify the fidelity in the op's program config:

```python
matmul_config = ttnn.MatmulMultiCoreReuseProgramConfig(
    ...,
    math_fidelity=ttnn.MathFidelity.LoFi,
)
```

Choose fidelity based on the required precision:

- `LoFi`: suitable for most BF16 inference; negligible accuracy loss versus `HiFi4` for typical transformer workloads.
- `HiFi2`: a middle ground for cases where `LoFi` produces unacceptable accuracy loss.
- `HiFi4`: reserved for FP32 accumulation or operations that are numerically sensitive (e.g., softmax in long-context attention, certain normalization layers).

> **Warning:** Reducing fidelity below what the operation requires will silently degrade model accuracy. Always validate outputs against a reference when changing math fidelity. Use `HiFi4` for any op where precision has not been verified with lower settings.

---

## Cause 4 — TRISC0/TRISC2 Pipeline Stalls (Unpacker Latency)

### Mechanism

Each Tensix core runs **five** independent RISC-V processors: BRISC, NCRISC, TRISC0, TRISC1, TRISC2.

- **BRISC (broadcast RISC):** the primary host-facing processor; dispatches work and coordinates the other cores.
- **NCRISC (NoC RISC):** manages NoC/DMA transfers, moving tiles between DRAM and L1.
- **TRISC0 (unpacker):** reads tiles from L1 and reformats them for the math engine.
- **TRISC1 (math):** executes FMA operations on the prepared tiles.
- **TRISC2 (packer):** takes math results and writes them back to L1.

The math engine (TRISC1) can only run when TRISC0 has finished unpacking the next pair of input tiles. If TRISC0 is slow — because NCRISC has not yet delivered tiles to L1 (NCRISC performs the DRAM-to-L1 DMA; TRISC0 reads only from L1) — TRISC1 stalls and waits. The FPU is physically idle during these stalls, but `TRISC1 KERNEL DURATION` continues to accumulate wall-clock time.

The result: `PM_IDEAL / TRISC1_DURATION` drops below the compute ceiling even though the FPU is occasionally running at full speed between stalls.

### Observable Effect

- `TRISC0 KERNEL DURATION [ns] > 1.2 × TRISC1 KERNEL DURATION [ns]`.
- `FPU UTIL` is below 0.5 for an op that should be compute-bound.
- `NOC BW UTIL` may be moderate (0.4–0.7), reflecting intermittent DRAM-to-L1 DMA traffic that NCRISC is issuing, with TRISC0 stalled waiting for those tiles to arrive in L1.

### Fix

Two complementary approaches:

1. **Increase L1 double-buffering:** configure the op to prefetch the next batch of tiles into a second L1 buffer while TRISC1 is computing on the current buffer. This hides DRAM fetch latency. Controlled via `in0_block_w` in `ttnn.MatmulMultiCoreReuseProgramConfig` (see [`remediation_levers.md`](./remediation_levers.md)).

2. **Use sharded memory layouts:** place weight tensors in L1-sharded memory so that tiles are already resident in L1 before TRISC0 needs them, eliminating the need for NCRISC to DMA them from DRAM at compute time. `ttnn.MemoryConfig` with `TensorMemoryLayout.HEIGHT_SHARDED` or `BLOCK_SHARDED` keeps the tiles resident on the core, eliminating DRAM fetch latency entirely for stationary operands.

> **Note:** Double-buffering trades L1 capacity for throughput. If L1 is already near capacity, increasing `in0_block_w` can cause an L1 overflow, which is a hard error. Check total L1 usage before increasing buffer depth.

---

## Cause 5 — NoC Contention (Too Many Active Cores Saturating the NoC)

### Mechanism

The NoC is a shared interconnect. When many cores are active simultaneously and NCRISC on each core is issuing DRAM-to-L1 DMA read requests, the aggregate read traffic can saturate the NoC's available bandwidth. The result is that individual DRAM reads take much longer than expected (the NoC queues requests), NCRISC is delayed delivering tiles to L1, and TRISC0 stalls waiting for those tiles to arrive.

This is distinct from Cause 4 in that the bottleneck is not a single core's unpacker latency but rather system-wide bandwidth saturation. The symptom appears at the CSV level as simultaneously high `NOC BW UTIL` and low `FPU UTIL`, indicating the NoC is busy but the FPU is not benefiting from that traffic.

### Observable Effect

- `NOC BW UTIL > 0.8`.
- `FPU UTIL < 0.3`.
- Both conditions hold at the same time — the NoC is genuinely saturated, not just occasionally busy.
- Reducing `CORE COUNT` for the same op reduces `NOC BW UTIL` and increases `FPU UTIL`.

### Fix

Reduce the number of active cores so that the combined DRAM read traffic stays within the NoC's sustainable bandwidth. As with Cause 1, use `compute_with_storage_grid_size` to reduce the grid.

An alternative approach is to interleave the NCRISC DMA access pattern across cores so that DRAM reads are staggered in time rather than issued simultaneously by all NCRISCs. This is an advanced technique that requires modifying the kernel's DMA scheduling and is typically not configurable via the ttnn API without custom kernels.

> **Tip:** Causes 1 and 5 can coexist. A large grid with a small problem means each core has few tiles (Cause 1) and NCRISC on each core is also issuing DRAM-to-L1 DMA reads concurrently with every other core (Cause 5). Reducing `compute_with_storage_grid_size` addresses both simultaneously.

---

## Cause 6 — Program Cache Miss (Recompilation on First Call)

### Mechanism

When a ttnn op is called for the first time with a given combination of shapes, data formats, program config parameters, and device configuration, the runtime must compile the kernel (TRISC0/TRISC1/TRISC2 programs) from scratch. Compilation happens on the host CPU and takes on the order of tens to hundreds of milliseconds. During this time, the device may be idle or running the kernel at reduced efficiency because the compiled binary is being transferred to the device.

This manifests as an anomalously long `DEVICE KERNEL DURATION` on the first call to an op with a new shape, followed by normal durations on subsequent calls once the compiled kernel is cached.

> **Note:** "First call" here means the first call with a specific set of parameters that the program cache has not seen before. If your model runs the same shapes repeatedly (as in inference), only the very first forward pass incurs this overhead. If shapes vary (as in dynamic-shape inference), cache misses can recur on every new shape encountered.

### Observable Effect

- First-call `DEVICE KERNEL DURATION` is more than 10× the steady-state duration for subsequent identical calls.
- Subsequent calls with identical parameters show normal `DEVICE KERNEL DURATION`.

### Fix

Enable the program cache before the first call:

```python
ttnn.enable_program_cache(device)
```

This instructs the runtime to compile once and reuse the binary for all subsequent calls with the same parameters. The cache persists for the lifetime of the device handle.

Additionally, warm up the cache explicitly before beginning any performance measurement:

```python
# Warmup pass — not timed
_ = ttnn.matmul(a, b, program_config=config)

# Timed pass
with tracy.region("matmul_timed"):
    out = ttnn.matmul(a, b, program_config=config)
```

> **Warning:** Always exclude first-call durations from performance benchmarks. A single cache-miss call can inflate aggregate timing statistics by an order of magnitude if the warmup step is omitted.

---

## Cause 7 — Incorrect Loop Count in Kernel (Padded Shapes)

### Mechanism

Tensix kernels are generated with loop bounds derived from tile counts. In some cases, when a tensor dimension is not a multiple of 32, the logical tile count differs from the padded tile count. If the kernel's inner loop is generated using padded tile counts rather than logical tile counts, the math engine performs more FMA iterations than the op's logical shape requires — it over-iterates, running past the logical boundary onto zero-padded tiles. The FPU is physically running (TRISC1 is active), but a fraction of those cycles are wasted on multiply-accumulate with zero values rather than contributing useful work.

Because the kernel does run to completion without error, none of the other CSV signals are anomalous: there are no stalls (ruling out Cause 4), no format or fidelity issues (ruling out Causes 2 and 3), no contention (ruling out Cause 5), no cache miss (ruling out Cause 6), and sufficient tile count (ruling out Cause 1). The only signal is a `FPU UTIL` that is stable across calls but consistently below 0.4 for a compute-bound shape, with no other cause identified.

Diagnosing Cause 7 requires inspecting the generated kernel source (TRISC1 program) to verify that the inner loop bounds match the logical tile counts from the op's input shapes.

> **Note:** For dispatch overhead introduced by host-side call patterns (including mesh trace capture via `ttnn.execute_trace`), see Chapter 6. Cause 7 is strictly a device-side issue: the kernel executes on the device with the correct duration budget, but wastes a fraction of its FLOPs on zero-padded tiles rather than useful data. Chapter 6 covers the separate case where the host is unable to dispatch work to the device fast enough to keep it busy.

### Observable Effect

- `FPU UTIL` is stable (consistent across multiple calls with identical shapes).
- `FPU UTIL` is consistently below 0.4 for a matmul shape where arithmetic intensity is well above the ridge point.
- No other cause (1–6) can be identified.
- The discrepancy between measured `FPU UTIL` and expected `FPU UTIL` scales with the fraction of padding in the shape.

### Fix

This cause requires kernel-level investigation:

1. Print the generated TRISC1 kernel source for the op (available in the ttnn kernel build output directory).
2. Identify the inner loop bounds and compare them against `M_t`, `K_t`, `N_t` computed from the logical input shapes.
3. If the loop bounds use padded tile counts, file a bug against the program config or use input shapes that are already multiples of 32 as a workaround.

> **Tip:** The simplest mitigation without kernel modification is to pad input tensors to multiples of 32 before the op and slice the output afterward. This changes the logical tile counts to match the padded tile counts, resolving the loop bound mismatch.

---

**Next:** [`csv_signatures.md`](./csv_signatures.md)
