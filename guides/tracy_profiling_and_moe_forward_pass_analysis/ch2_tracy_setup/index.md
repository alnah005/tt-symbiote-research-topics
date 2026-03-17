# Chapter 2 — Setting Up Tracy Profiling

This chapter walks through the complete setup required to capture a Tracy trace from a tt-metal workload. It covers the CMake build flags that activate the profiler, the three-component launch sequence, the environment variables that control runtime behavior, and the output files produced. By the end of this chapter you will have a working capture pipeline and know how to verify it at each step.

**Prerequisite:** Complete Chapter 1 (`ch1_tracy_overview/`) before reading this chapter. The terminology used here — zones, the two-process model, `TRACY_ENABLE`, the capture server — is defined there and is not redefined below.

---

## Learning Objectives

After completing this chapter, you will be able to:

1. State the CMake flag needed to enable Tracy and the on-device cycle-counter profiler in a tt-metal build (`ENABLE_TRACY=ON`).
2. Identify the three components that must run together for a successful capture: the profiled process, `tracy-capture`, and optionally the Tracy GUI (`tracy-profiler`).
3. Explain the purpose of `TRACY_NO_EXIT=1` and identify the class of workloads (Python scripts, short-lived processes) where it is required.
4. Describe the two output artifacts produced by a combined profiling run: the `.tracy` binary and `profile_log_device.csv`.
5. Identify the most common failure mode — client/server version mismatch — and state the fix: rebuild `tracy-capture` from the same `third_party/tracy` submodule commit.
6. Explain how to use `tracy-csvexport` to extract a flat, human-readable zone CSV from a `.tracy` file and compute inter-zone gaps with pandas.

---

## Setup Checklist

Follow these five steps in order to go from a clean tt-metal checkout to your first `.tracy` file.

1. Build tt-metal with `-DENABLE_TRACY=ON` (see `build_flags.md`).
2. Build `tracy-capture` from `third_party/tracy` at the same submodule commit (see `build_flags.md`).
3. Start `tracy-capture` in one terminal before launching your workload (see `capture_workflow.md`).
4. Launch your profiled process with `TT_METAL_DEVICE_PROFILER=1` and `TRACY_NO_EXIT=1` set (see `capture_workflow.md`).
5. Inspect the `.tracy` file with `tracy-csvexport` or open it in the Tracy GUI (see `output_format.md`).

---

## Chapter Contents

| File | Topic |
|---|---|
| [`build_flags.md`](./build_flags.md) | CMake flags, Tracy submodule pinning, build verification, and `TRACY_ON_DEMAND` |
| [`capture_workflow.md`](./capture_workflow.md) | Launch order, environment variables, complete two-terminal example, and common failure modes |
| [`output_format.md`](./output_format.md) | The `.tracy` binary, `tracy-csvexport` CSV columns, `profile_log_device.csv` post-processing, and cross-tool timestamp correlation |

---

## Reading Order

Read the files in the order listed above. `build_flags.md` is a prerequisite for `capture_workflow.md` because the workflow assumes you already have a profiler-enabled binary. `output_format.md` references artifact names introduced in `capture_workflow.md`.

---

## Next Steps

Begin with [`build_flags.md`](build_flags.md) to configure and build a profiler-enabled tt-metal binary.
