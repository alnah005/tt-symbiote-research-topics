# Chapter 5: Multi-Device Scaling

## Overview

This chapter covers how TT-Transformers scales transformer inference and training across multiple Tenstorrent Wormhole devices using tensor parallelism (TP) and collective communication operations (CCL). The two preceding chapters established the single-device memory hierarchy and sharding primitives (Ch4) and the matmul program configurations that drive compute efficiency (Ch3). Chapter 5 builds on those foundations to explain how weight shards are distributed across devices and how collective operations reconnect those shards into coherent outputs.

The chapter is organized as two focused sections:

| File | Topic |
|---|---|
| `tensor_parallelism.md` | Column-parallel, row-parallel, and vocab-parallel linear patterns; GQA KV head replication; N300 vs T3K configurations |
| `ccl_and_ethernet.md` | AllGather, ReduceScatter, AllReduce semantics; Ethernet bandwidth budget; CCL pipelining; when TP helps vs hurts throughput |

## Reading Order

Read this index first, then proceed to `tensor_parallelism.md`, then `ccl_and_ethernet.md`. Readers who are already familiar with the Megatron-style column/row-parallel split can skim the first section of `tensor_parallelism.md` and jump directly to the GQA subsection, which covers behavior specific to grouped-query attention under TP.

## Key Concepts

| Concept | Short Definition | Where Covered |
|---|---|---|
| Tensor parallelism (TP) | Splitting weight matrices across devices along a chosen dimension so that each device performs a fraction of the total compute | `tensor_parallelism.md` |
| Column-parallel linear | Weight shard is `[K, N/TP]`; activations are broadcast; output reconstructed with `ttnn.all_gather` | `tensor_parallelism.md` |
| Row-parallel linear | Weight shard is `[K/TP, N]`; activations are pre-sharded along K; result reduced with `ttnn.reduce_scatter` | `tensor_parallelism.md` |
| Vocab-parallel LM head | Embedding shard is `[vocab_size/TP, hidden]`; result collected with `ttnn.all_reduce` | `tensor_parallelism.md` |
| GQA KV replication | When TP exceeds the number of KV heads, KV weights cannot be sharded and must be replicated on every device | `tensor_parallelism.md` |
| AllGather | Each device contributes its shard; every device receives the full concatenated tensor | `ccl_and_ethernet.md` |
| ReduceScatter | Each device contributes partial sums; the summed result is distributed so each device holds one shard | `ccl_and_ethernet.md` |
| AllReduce | AllGather followed by elementwise reduction; every device receives the full reduced tensor | `ccl_and_ethernet.md` |
| CCL pipelining | Overlapping a collective communication with the next matmul computation to hide latency | `ccl_and_ethernet.md` |
| N300 | 2-chip Wormhole board connected via on-board Ethernet; max TP=2 | `tensor_parallelism.md` |
| T3K | 8-chip Wormhole Ethernet ring; max TP=8 | `tensor_parallelism.md` |

## Hardware Context

Tenstorrent multi-device configurations covered in this guide use Wormhole chips connected by Ethernet, not PCIe. Full hardware specifications for N300 (2-chip, max TP=2) and T3K (8-chip Ethernet ring, max TP=8) are in `tensor_parallelism.md` § Hardware Configurations. The ring topology detail — and why it drives CCL hop-count cost — is covered in `ccl_and_ethernet.md` § Ethernet Bandwidth Budget.

## Relationship to Prior Chapters

- **Ch2** established GQA: the ratio of Q heads to KV heads (the GQA group size G) determines whether KV weights can be sharded or must be replicated at a given TP degree. That connection is made explicit in `tensor_parallelism.md`.
- **Ch3** established weight quantization and bandwidth reduction factors. The same bandwidth multipliers (BFP8: 1.88x lower than BF16; BFP4: 3.56x lower than BF16) apply to the per-device weight shards under TP, and their interaction with CCL cost is discussed in `ccl_and_ethernet.md`.
- **Ch4** established the L1 sharding primitives and the Ethernet topology that underpin multi-device collectives. The height-sharded and block-sharded tensors described in Ch4 are the activation layouts that feed into the TP patterns described here.
