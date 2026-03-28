# Chapter 2 — KV Cache Management During Decode

Chapter 1 established the core vocabulary: window size `w`, total sequence
length `T`, head dimension `d`, number of heads `H`, batch size `B`, and the
definition of the windowed attention set
`A_win(t) = [max(0, t-w+1), t]`.
This chapter takes that vocabulary and works out its concrete implication for
device memory: how the KV cache is allocated, how it evolves token by token,
why it reaches a fixed steady-state size, and how the eviction policy is
implemented as a circular buffer in DRAM.

## Reading Order

1. [`kv_cache_lifecycle.md`](./kv_cache_lifecycle.md) — how the KV cache grows
   during the fill phase, transitions to steady-state once it holds `w` entries,
   the exact formula for windowed versus full-attention cache size at generation
   step T, and the memory saving factor as a function of T and w.

2. [`circular_buffer_layout.md`](./circular_buffer_layout.md) — the circular
   (ring) buffer data structure that implements the eviction policy in device
   DRAM, the write-pointer and wrap-around arithmetic, how the buffer is exposed
   as a fixed-shape TTNN tensor `[B, H, w, d]` with a companion position-offset
   scalar, and the contrast with the grow-in-place tensor used by full-attention
   KV caches.

## Connection to Chapter 1

Chapter 2 derives the size formulae stated in Chapter 1's decode table and explains the data structure that keeps the windowed cache size constant.

## What Comes Next

Chapter 3 — Data Dependencies and Memory Access Patterns — characterises the
read pattern that the circular buffer imposes on each decode step: which cache
slots are accessed, in what order, and what the resulting DRAM access footprint
looks like for a T3K mesh.
