# Compression Analysis — Chapters 5, 6, and 7: Latency Analysis, Implementation, and Validation

## Pass 1

### Crucial issues found: 0

No pairs of files across Chapters 5, 6, and 7 contain substantially identical content blocks of 5 or more near-verbatim lines that serve no purpose.

Candidates investigated and ruled out:

**ch6/index.md "Before/After Overview" vs. ch6/structural_changes.md "Change Group B2-2b"**
Both files show the same `ttnn.all_gather` → `ttnn.experimental.all_gather_async` code transition, sharing ~10 near-verbatim lines in the BEFORE block and ~10 in the AFTER (Type B2) block. This is not flagged as crucial because the overlap is intentional and functional: `ch6/index.md` is an orientation layer that previews the key code change so readers know what to expect before reading the detail file. The index explicitly labels its section "Before/After Overview" and closes with a "What's Next" pointer to `structural_changes.md` as the authoritative source. The two files serve distinct roles (chapter navigation vs. implementation specification) and the preview in the index reduces the cognitive load of opening the detail file cold. One referencing the other would eliminate the useful at-a-glance overview without gaining compression.

**ch5/measuring_the_cost.md Tracy env block vs. ch7/latency_measurement.md Tracy env block**
Both files include `export TT_METAL_DEVICE_PROFILER=1` and `export TT_METAL_PROFILER_TRACE_TRACKING=1`. This is only 2 lines — below the 5-line threshold — and the surrounding Tracy commands target different tests with different `-k` flags (`test_maybe_all_gather_latency` vs. `test_decode_step_latency`). Not flagged.

**ch7/multi_replay_stability.md references to ch6 wrapper checklist step numbers**
`multi_replay_stability.md` references "steps 1–4", "steps 6–7", and "steps 8–11" from the ch6 wrapper checklist by name, not by reproducing the content. This is cross-referencing, not duplication.

**Cross-chapter concept repetition (e.g., "0.1–0.5 ms", "PCC > 0.999", "K × T_sync")**
The latency range and PCC threshold appear in multiple files, but always in single-sentence or single-table-row form — never as a block of 5+ near-verbatim lines. Each appearance is contextualized differently (model, measurement, validation) and is necessary for the file to be self-contained.

---

### VERDICT

Crucial updates: no

Chapters 5, 6, and 7 compression approved.
