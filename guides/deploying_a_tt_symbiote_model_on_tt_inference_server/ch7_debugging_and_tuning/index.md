# Chapter 7 — Debugging, Constraints, and Performance Tuning

This chapter covers the practical work of getting a TT Symbiote integration over the finish line: diagnosing the most common integration failures, understanding the full set of platform constraints that shape what your model code can and cannot do, and applying the tuning knobs available in `DeviceModelSpec`, `vllm_args`, and `override_tt_config` to hit your throughput and latency targets.

By the end of this chapter you will be able to read an integration error, identify its root cause from the Symbiote/vLLM call path, apply a targeted fix, and then profile and tune the resulting deployment.

## What This Chapter Covers

This chapter does not cover model architecture or training. It assumes you have a working `TTMyModel` class that passes offline inference tests and that you are now trying to run it inside `tt-inference-server`.

## Reading Order

Work through the files in order. The error reference is most useful when you already understand the architecture from earlier chapters; the tuning guide builds on the same mental model.

1. [common_errors.md](./common_errors.md) — Catalog of the ten most common integration failures, each with the exact error message or symptom, root cause analysis, and a concrete fix.
2. [performance_tuning.md](./performance_tuning.md) — How to use the throughput, latency, and memory levers available in the platform configuration to hit your target performance envelope.

## What's Next

This is the final chapter. Return to the [guide index](../index.md) for an overview of all chapters.

---

**Next:** [common_errors.md](./common_errors.md)
