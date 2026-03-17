# Device Performance Counters

This file describes the hardware performance counters available on Wormhole B0, explains how to
access and read them via the TTNN and tt-metal APIs, and shows how to map counter readings to
specific MoE operations. Counter data is used in Step 4 of the profiling workflow to confirm
which hardware resource is the binding constraint.

---

## Quick Reference: Counter API Symbols

| Symbol | Type | Description |
|---|---|---|
| Ethernet bytes-transferred counter | HW counter | Per Ethernet port; bytes sent or received per interval |
| DRAM read / write bandwidth counter | HW counter | Bytes per DRAM channel per interval |
| NOC tile-transfer counter | HW counter | Tiles routed on-chip Network-on-Chip per interval |
| Tensix active-cycle counter | HW counter | Cycles with active compute per core per interval |
| `tt_metal::perf_counter::read_ethernet_utilization()` | C++ API | [VERIFY exact API name in tt-metal] |
| `tt_metal::perf_counter::read_dram_bandwidth()` | C++ API | [VERIFY exact API name in tt-metal] |
| TTNN debug API equivalent | Python API | [VERIFY Python wrapper exists] |

---

## Section 1: Available Counters on Wormhole B0

Wormhole B0 exposes four categories of hardware performance counters relevant to MoE workloads.

### Ethernet Link Utilization

Counts bytes transferred per Ethernet port per measurement interval. T3K has
~12.5 GB/s of bandwidth per Ethernet link (see `ch01_t3k_topology/ethernet_link_bandwidth.md`).
Utilization is reported as a fraction of peak: a value of 1.0 means the link is fully saturated.

Each T3K device participates in a 1×8 linear mesh; interior devices (IDs 1–6) have two active
Ethernet ports (left and right neighbor). Devices 0 and 7 have one active port each. The counter
is available **per port**; sum across all active ports on a device for the aggregate device-level
bandwidth consumption.

### DRAM Bandwidth

Counts bytes read and bytes written per DRAM channel per interval. Wormhole B0 has multiple DRAM
channels per chip; aggregate read bandwidth is approximately 300 GB/s per chip [UNVERIFIED].
Report reads and writes separately: expert weight reads during FFN matmul are the dominant read
source; all-to-all buffer writes are the dominant write source in the dispatch phase.

### NOC Traffic

Counts tiles transferred on the on-chip Network-on-Chip (NOC) per interval. NOC traffic is
elevated during operations that move data between Tensix cores (e.g., reductions, tensor
permutations). In MoE workloads, NOC traffic spikes during the top-$k$ scatter/gather and during
expert output accumulation. High NOC traffic with low Ethernet and low DRAM utilization indicates
an on-chip communication bottleneck rather than off-chip.

### Tensix Core Utilization

Counts cycles in which a Tensix core's compute engine is actively executing a tile operation,
as a fraction of total elapsed cycles. A value above 80% indicates the workload is
compute-bound on that device. A value below 30% during the expert FFN phase suggests either
memory-bandwidth-bound or communication-bound behavior.

---

## Section 2: How to Enable and Read Counters

### API Access

Hardware counters are accessible via the tt-metal `perf_counter` API [VERIFY exact API in
tt-metal source]. A Python wrapper may also be available through TTNN debug utilities [VERIFY].

```python
# Illustrative usage — verify API names in tt-metal before use
import tt_metal  # [VERIFY import path]

device = mesh_device.get_device(device_id=0)

# Start counter collection window
tt_metal.perf_counter.start(device)   # [VERIFY]

# Run the operation to measure
output = ttnn.all_to_all(dispatch_tensor, mesh_device, num_links=1)

# Stop and read counters
counters = tt_metal.perf_counter.stop_and_read(device)  # [VERIFY]

eth_bytes_sent     = counters["ethernet_bytes_sent"]     # [VERIFY key name]
dram_bytes_read    = counters["dram_bytes_read"]         # [VERIFY key name]
tensix_active_frac = counters["tensix_active_cycles_frac"]  # [VERIFY key name]
```

> **Warning:** Counter API names above are illustrative. Verify them against the tt-metal source
> before use. Look in `tt_metal/impl/debug/` or `tt_metal/perf_counter/` [VERIFY path].

### Counter Granularity

All counters are reported **per device**. The Ethernet counter is further broken down per port.
There is no per-Tensix-core counter accessible from the Python layer without a custom kernel
instrumentation pass. Use Tracy (see `ttnn_profiler.md` §4) for per-core timeline visibility.

### Measurement Window

To isolate a single operation, wrap the start/stop calls tightly around the TTNN dispatch. Be
aware that TTNN dispatch is asynchronous by default: the op may not have completed on the device
when the Python call returns. Use `ttnn.synchronize_device()` [VERIFY] after the op dispatch and
before reading counters to ensure the measurement window is closed correctly.

```python
ttnn.synchronize_device(mesh_device)   # [VERIFY API name]
counters = tt_metal.perf_counter.stop_and_read(device)  # [VERIFY]
```

---

## Section 3: Mapping Counters to Operations

The table below shows which counters are most informative for each phase of the MoE layer.

| MoE Phase | Ethernet Utilization | DRAM BW (Read) | DRAM BW (Write) | Tensix Utilization | NOC Traffic |
|---|---|---|---|---|---|
| All-to-all dispatch | High | Moderate (dispatch buffer read) | Low | Low (idle) | Low |
| Expert FFN matmul | Low | High (weight streaming) | Low | High | Moderate |
| All-to-all combine | High | Low | Moderate (combine buffer write) | Low (idle) | Low |
| Expert output accumulation | Low | Low | Low | Moderate | High |
| Router top-$k$ | Low | Low | Low | Moderate | Moderate |

### Interpretation Notes

- **All-to-all dispatch:** Ethernet utilization spikes as each device sends token activations to
  the devices that own the selected experts. DRAM BW is moderate because the dispatch buffer is
  read from DRAM (or L1 if small enough — see `ch04_memory_config/decode_memory_strategy.md`).
  Tensix cores are idle during the transfer.

- **Expert FFN matmul:** Ethernet utilization drops to near zero. DRAM read BW is high because
  expert weight tensors ($[H, D]$ per expert, where $H=7168$) cannot fit in L1 and must be
  streamed from DRAM. Tensix utilization is the key indicator: if it is high (> 80%), the
  workload is compute-bound; if moderate (30–80%) with high DRAM BW, it is memory-bandwidth-bound.

- **Expert output accumulation:** NOC traffic spikes as partial results from each expert are
  gathered and reduced. Ethernet and DRAM are quiet; this phase is typically fast relative to
  the others.

---

## Section 4: Detecting Link Saturation

### Saturation Indicator

Ethernet link utilization ≥ 90% during the all-to-all dispatch or combine phase indicates that
the link is fully saturated: the payload is large enough to fully utilize the available Ethernet
bandwidth.

### Interpreting Saturation

| Situation | Interpretation | Recommended Action |
|---|---|---|
| Saturated AND latency acceptable | Links are being used efficiently | No change required |
| Saturated AND latency too high | Payload exceeds single-link capacity | Increase `num_links` if hardware maximum not reached; or reduce payload via expert placement (ch05) |
| Below 50% utilization | Links are underutilized; setup overhead dominates | Consider reducing `num_links` or increasing batch size to amortize overhead |
| Oscillating between saturated and idle | Pipeline bubbles between dispatch and compute | Pipeline dispatch+compute+combine (see `bottleneck_diagnosis_guide.md` §2) |

### Counter-Intuitive Case: Reducing num_links

For small payloads ($C=1$, $B=1$, dispatch volume ≈ 3.2 MB per device), link setup and
synchronization overhead can exceed transfer time. In this regime, using two links instead of
one **increases** total all-to-all latency because the overhead doubles while the payload is not
large enough to benefit from parallel transfer. The counter evidence: Ethernet utilization is
low (< 50%) despite `num_links=2`. Reducing to `num_links=1` in this case removes the setup
overhead. See `ch03_all_to_all_num_links/num_links_parameter.md` for the full analysis.

---

## Section 5: Detecting L1 Pressure

### Symptom

L1 pressure manifests in two ways:

1. **Compile-time:** `ttnn.exceptions.MemoryAllocationError` during model compilation, indicating
   a circular buffer (CB) allocation failure. The op could not fit its input and output tensors
   in the L1 budget of the assigned cores.

2. **Runtime:** Unexpectedly high DRAM read bandwidth during operations that are expected to be
   L1-resident. The kernel has fallen back to streaming from DRAM instead of reading from L1.

### Diagnosis via DRAM Bandwidth Counter

Expected DRAM read BW during the expert FFN matmul is dominated by weight streaming:

$$\text{Expected BW} = \frac{\text{weight tiles per expert} \times \text{tile size}}{\text{matmul time}}$$

If the measured DRAM read BW significantly exceeds this estimate, activation tensors that were
expected to be L1-resident are spilling to DRAM. Expert weights (~205 MB per expert [UNVERIFIED])
always stream from DRAM; their bandwidth is expected, not a spill indicator. A DRAM BW spike
**above** the weight-streaming baseline means activations are spilling.

For the full CB footprint table (activation sizes, weight streaming classification, L1 fit
analysis) and the remediation procedure, see `bottleneck_diagnosis_guide.md` Section 5.

---

## References

- Wormhole B0 architecture specification (internal) — DRAM bandwidth [UNVERIFIED: ~300 GB/s aggregate]
- `tt-metal` repository: `tt_metal/impl/debug/` and `tt_metal/perf_counter/` [VERIFY paths]
- `ch01_t3k_topology/ethernet_link_bandwidth.md` — Ethernet link bandwidth baseline (~12.5 GB/s)
- `ch03_all_to_all_num_links/num_links_parameter.md` — `num_links` tuning and overhead model
- `ch04_memory_config/decode_memory_strategy.md` — L1 budget estimation and placement decisions
- `ttnn_profiler.md` — TTNN profiler setup (prerequisite for counter interpretation)
