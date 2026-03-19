## Pass 1

**3 correctness issues found.**

---

**1. `index.md` line 7 vs. `measuring_dispatch_vs_kernel.md` line 137 — Inconsistent crossover value (wrong numerical answer)**

`index.md` states the dispatch/kernel crossover "falls around a device kernel duration of ~10 µs for a warm-cache TTNN op call." `measuring_dispatch_vs_kernel.md` line 137 states the crossover occurs at `device_kernel_time ≈ 100 µs` (justified as 10× a ~10 µs dispatch cost). These differ by 10×. A reader comparing the two will get a wrong numerical threshold. The body derivation (100 µs = 10× the ~10 µs dispatch baseline) is self-consistent; the index value of ~10 µs is the error.

Fix: Change `index.md` line 7 from "~10 µs" to "~100 µs".

---

**2. `eliminating_dispatch_overhead.md` lines 33 and 36 — `trace_id` used before assignment (implementation error)**

The code block calls `ttnn.end_trace_capture(device, trace_id, cq_id=0)` passing `trace_id` as an argument, but `trace_id` has not been assigned at that point. In the tt-metal API, `end_trace_capture` *returns* the trace_id — it is an output, not an input. The variable is then reused in `ttnn.execute_trace(device, trace_id, cq_id=0)`. A reader implementing this literally will get a `NameError` at `end_trace_capture`.

Fix: Change line 33 to `trace_id = ttnn.end_trace_capture(device, cq_id=0)` (capturing the return value).

---

**3. `measuring_dispatch_vs_kernel.md` lines 73–74 — Previously flagged claim is not present in current file (Pass 1 finding retracted)**

The current file text at lines 73–75 correctly states: "the Tracy zone for op N accurately records the host-side enqueue cost — it ends when enqueue returns, exactly as in the serialized case. The zone itself is not distorted by pipelining." No error is present. Pass 1 item 3 does not apply to the current file.

---

## Pass 2

**2 correctness issues found.**

---

**1. `index.md` line 7 — Crossover value is 10× too small (wrong numerical answer)**

The index states the crossover "falls around a device kernel duration of ~10 µs for a warm-cache TTNN op call." The body (`measuring_dispatch_vs_kernel.md` line 137) derives the crossover as `device_kernel_time ≈ 100 µs` (10× the ~10 µs baseline dispatch cost). The body derivation is internally consistent and matches the worked data table (e.g., large matmul at ~200 µs kernel / ~12 µs dispatch is not dispatch-bound; small matmul at ~0.5 µs kernel / ~8 µs dispatch clearly is). The index value of ~10 µs is a factor-of-10 error.

Confirmed: Pass 1 item 1 stands. Current `index.md` line 7 still reads "~10 µs."

Fix: Change `index.md` line 7 from `~10 µs` to `~100 µs`.

---

**2. `what_is_dispatch_overhead.md` line 108 — "one or two tiles" claim is wrong for the stated shape (wrong conceptual understanding)**

The file states: "Device kernel time: ~0.3 µs. The compute is trivially small — one or two tiles spread across a handful of cores." The tensor shape given is `[32, 4096]`. In TTNN tile layout (32×32 tiles), a [32, 4096] tensor contains 4096/32 = 128 tiles — not one or two. A reader who internalizes "one or two tiles" as the explanation for ~0.3 µs kernel time will have a fundamentally wrong model of how tile counts map to kernel duration for small elementwise ops. The correct explanation is that the compute per tile is minimal (a single elementwise add), and even across 128 tiles on a few cores the kernel finishes in ~0.3 µs because the memory-bandwidth cost per tile is negligible at this size. Claiming one or two tiles is the cause is numerically false for this shape.

Fix: Replace "one or two tiles spread across a handful of cores" with something accurate for [32, 4096], e.g. "128 tiles spread across a handful of cores, each tile requiring only a trivial elementwise addition."

---

**Pass 1 item 2 status:** The current `eliminating_dispatch_overhead.md` line 33 correctly reads `trace_id = ttnn.end_trace_capture(device, cq_id=0)` — the return value is captured and no extra argument is passed. Pass 1 item 2 does not apply to the current file.

## Pass 3

**No feedback — chapter approved.**

All issues raised in Pass 1 and Pass 2 have been resolved in the current file versions:

- `index.md` line 7 now correctly states the crossover at "~100 µs" (not "~10 µs"). Consistent with the derivation in `measuring_dispatch_vs_kernel.md` line 137.
- `what_is_dispatch_overhead.md` line 108 now correctly computes 128 tiles for a [32, 4096] tensor and does not use the previously flagged "one or two tiles" language.
- `eliminating_dispatch_overhead.md` line 33 correctly assigns `trace_id = ttnn.end_trace_capture(device, cq_id=0)`.

No new issues found that would cause a reader to obtain a wrong numerical answer, implement something incorrectly, or arrive at a materially wrong conceptual understanding.

## Pass 4

**No feedback — chapter approved.**

All previously raised issues remain resolved. Specific checks performed against current file versions:

- `total_op_latency = host_dispatch_time + device_kernel_time + sync_overhead` is stated correctly in `index.md` (line 12) and `what_is_dispatch_overhead.md` (lines 102–103).
- `TT_METAL_PROFILER_SYNC=1` is named and used correctly throughout `measuring_dispatch_vs_kernel.md`.
- `ttnn.end_trace_capture` correctly returns `trace_id` via assignment (`trace_id = ttnn.end_trace_capture(device, cq_id=0)`) in `eliminating_dispatch_overhead.md` line 21.
- Navigation footers are present in all three content files; `eliminating_dispatch_overhead.md` ends with "End of guide."
- Crossover value in `index.md` line 7 reads "~100 µs," consistent with the body derivation in `measuring_dispatch_vs_kernel.md`.
- Tile count for [32, 4096] in `what_is_dispatch_overhead.md` line 94 correctly computes 128 tiles.
- All worked data table dispatch/kernel ratios are arithmetically correct (16×, 0.06×, 20×, 3.5×).

No new issues found.

## Pass 5

**2 correctness issues found.**

---

**1. `eliminating_dispatch_overhead.md` lines 15–19 — Ops inside trace capture ARE executed, not merely recorded (wrong conceptual understanding)**

The comment inside the code block states: "All TTNN op calls inside this block are recorded but not executed yet." This is factually incorrect. In the tt-metal trace API, the op calls between `begin_trace_capture` and `end_trace_capture` execute normally during the capture pass — they run once on the device so the command buffer is populated with real, executed commands. They are not deferred or skipped. A reader who internalizes "not executed yet" will incorrectly believe that tensor outputs from the capture pass are uninitialized, may not pre-allocate output buffers correctly (thinking first execution only happens at `execute_trace`), and will have a wrong mental model of when side effects occur.

Fix: Change the comment to something accurate, e.g.: "All TTNN op calls inside this block execute once on the device AND are recorded into the trace command buffer."

---

**2. `measuring_dispatch_vs_kernel.md` line 94 — Subtracting device kernel time from the Tracy zone duration is conceptually incoherent (implementation error)**

The file states: "find the dispatch zone matching this invocation. Subtract `DEVICE KERNEL DURATION [ns] / 1000` from the Tracy zone duration to isolate the non-kernel host overhead." However, the same file established at line 7 that the Tracy zone "ends when the enqueue command is written to the device command queue and the enqueue function returns." The Tracy zone therefore already measures only `host_dispatch_time`; it contains no device kernel time. Subtracting `DEVICE KERNEL DURATION / 1000` from it yields a meaningless negative or near-zero number, not a useful decomposition. A reader following this instruction literally will compute an incorrect quantity and may conclude dispatch overhead is negligible when it is not.

Fix: Replace the subtraction instruction with an accurate statement, e.g.: "The Tracy zone duration for that invocation directly equals `host_dispatch_time` — no subtraction is needed, because the zone ends at enqueue and contains no device execution time. Read `DEVICE KERNEL DURATION [ns] / 1000` separately from the CSV as `device_kernel_time`."

---

## Pass 6

**No feedback — chapter approved.**

All issues from Pass 5 are resolved in the current file versions:

- `eliminating_dispatch_overhead.md` lines 15–19: the comment now correctly states that ops inside the capture block "execute normally on the device, producing real outputs, while the device command sequence is simultaneously recorded." The "not executed yet" error is gone.
- `measuring_dispatch_vs_kernel.md` line 94: the file now correctly states "The Tracy zone duration directly equals `host_dispatch_time` — the zone spans only the host-side enqueue work and contains no device kernel time. With `TT_METAL_PROFILER_SYNC=1` active, simply read the Tracy zone duration to obtain `host_dispatch_time`; no subtraction is needed." The spurious subtraction instruction is gone.

Arithmetic and consistency checks performed this pass:

- `what_is_dispatch_overhead.md` lines 102–108: 6 + 0.3 + 0.5 = 6.8 µs; 6/6.8 = 88.2% — stated as "> 88%." Correct.
- `measuring_dispatch_vs_kernel.md` worked table ratios: 8/0.5 = 16×, 12/200 = 0.06×, 6/0.3 = 20×, 7/2 = 3.5×. All correct.
- `eliminating_dispatch_overhead.md` lines 38–42: ~1 µs / 50 ops = ~0.02 µs per op. Correct.
- `measuring_dispatch_vs_kernel.md` line 121: 10× × ~10 µs dispatch = ~100 µs crossover — consistent with `index.md` line 7 (~100 µs). Consistent.

No new issues found that would cause a reader to obtain a wrong numerical answer, implement something incorrectly, or arrive at a materially wrong conceptual understanding.
