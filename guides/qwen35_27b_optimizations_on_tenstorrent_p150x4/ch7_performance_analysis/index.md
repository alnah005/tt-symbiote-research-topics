# Chapter 7: Performance Analysis and Remaining Bottlenecks

This chapter consolidates the measured performance of the Qwen3.5-27B deployment on P150x4, compares current numbers against the baseline, and identifies where time is spent and what optimization opportunities remain.

The profiler breakdown across all prior chapters consistently pointed to GDN layers as the dominant cost. This chapter quantifies that cost in detail, traces it to its root cause in DRAM bandwidth for recurrence state I/O, and catalogs the remaining optimization paths -- from the L1 state work in progress (Chapter 6) to further kernel fusion and prefill improvements.

## Learning Objectives

After reading this chapter you will understand:

- The current decode throughput and TTFT numbers, and how they compare to the baseline
- The per-layer state size arithmetic that drives the DRAM bandwidth bottleneck (12 MB per GDN layer)
- The profiler breakdown showing GDN at 85% of decode time vs 12% for attention
- The expected impact of L1 state optimization and the remaining challenges
- Additional optimization opportunities beyond L1 state: kernel fusion, conv1d fusion, and prefill parallelism

## Files

| File | Description |
|------|-------------|
| [`performance_summary.md`](./performance_summary.md) | Current performance numbers, comparison to baseline, and test commands for reproduction |
| [`bottleneck_analysis.md`](./bottleneck_analysis.md) | Profiler breakdown, root cause analysis, and catalog of remaining optimization opportunities |

See [`performance_summary.md`](./performance_summary.md) to begin.

---

**Next:** [`performance_summary.md`](./performance_summary.md)
