# Chapter 2 — TTNN Decomposition of the Recurrent Delta Rule Step

By the end of this chapter, you will understand how the DeltaNet gated recurrence — currently implemented via host-side NumPy/PyTorch in `TTNNQwen3LinearAttention.forward` — can be expressed as a sequential composition of existing TTNN primitives, entirely on-device, without any host readback. You will see the exact tensor shapes and memory configurations at every step, understand why each choice is correct, and be positioned to wire the composed form into the existing forward pass.

The central question this chapter answers is: **can TTNN primitives express the DeltaNet recurrence without host readback?** The answer is yes. All 12 operations required for a complete decode step are available in TTNN today. No new kernel development is required for the composed form.

> **Note:** This chapter derives the minimal correct implementation — an ordered composition of existing ops with no operator fusion. The latency analysis comparing this composed form against the current host-roundtrip path is deferred to Chapter 4. A fused custom kernel for the full recurrence step is discussed in Chapter 5.

## Learning Objectives

1. State the six mathematical operations of the gated DeltaNet recurrence and identify which TTNN primitive maps to each.
2. Explain why `retrieval` must use `S_{t-1}` (the pre-decay state) and why both `retrieval` and `o_t` use `S^T` rather than `S`.
3. Specify the tensor shapes and memory configurations (DRAM vs. L1, tile layout) required to keep the recurrent state `S` alive across decode steps under trace.
4. Enumerate all 12 TTNN operations for a single complete decode step and confirm their availability in the current TTNN API.

## Operation Flow

The six core recurrence operations, in execution order:

```
Input: S_{t-1} [B, nH, d_k, d_v], k̃_t [B, nH, d_k, 1],
       q̃_t [B, nH, d_k, 1], v_t [B, nH, d_v, 1],
       g_t [B, nH, 1, 1], β_t [B, nH, 1, 1]

  S_{t-1}
  │
  ├─── (1) DECAY ──────────────────────────────────────────────────┐
  │         S_decayed = g_t * S_{t-1}                              │
  │         [B, nH, d_k, d_v]                                      │
  │                                                                 │
  └─── (2) RETRIEVAL ──────────────────────────────────────────────┤
            retrieval = S_{t-1}^T @ k̃_t                           │
            [B, nH, d_v, 1]                                        │
            │                                                       │
            (3) ERROR                                              │
            error = β_t * (v_t − retrieval)                       │
            [B, nH, d_v, 1]                                        │
            │                                                       │
            (4) WRITE (outer product)                              │
            write = k̃_t ⊗ error                                   │
            [B, nH, d_k, d_v]                                      │
            │                                                       │
            └───────────────── (5) NEW STATE ────────────────────┘
                                S_t = S_decayed + write
                                [B, nH, d_k, d_v]
                                │
                               (6) OUTPUT
                                o_t = S_t^T @ q̃_t
                                [B, nH, d_v, 1]

Output: S_t (persisted in DRAM), o_t (consumed by downstream projection)
```

Steps (1) and (2) both read `S_{t-1}` and can be treated as independent reads. Step (5) merges the results of (1) and (4). Step (6) reads the result of (5). This ordering is not arbitrary — using `S_{t-1}` for retrieval rather than `S_decayed` is a correctness requirement of the delta rule formulation.

## Files in Reading Order

1. [recurrence_math_and_tensor_ops.md](recurrence_math_and_tensor_ops.md) — Mathematical derivation of each operation with single-head and batched multi-head shapes, TTNN primitives, and correctness notes.
2. [ttnn_ops_per_step.md](ttnn_ops_per_step.md) — Full table of all 12 TTNN operations for one complete decode step, including input/output shapes, memory configs, and API availability.
3. [state_tensor_memory_config.md](state_tensor_memory_config.md) — Memory layout of the recurrent state tensor `S` for T3K, including DRAM persistence rationale, tile alignment analysis, and sizing for all 30 DeltaNet layers.
