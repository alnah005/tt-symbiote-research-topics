# Chapter 3: GQA Tensor Layout Requirements

## Overview

This chapter covers the tensor shapes and axis conventions that `paged_sdpa_decode` expects for Grouped-Query Attention (GQA) decode. Getting these shapes right is a prerequisite for correctness — several failure modes in this chapter produce wrong outputs silently, with no error raised by TTNN.

---

## Learning Objectives

By the end of this chapter, you should be able to:

1. State the TTNN decode tensor shapes for Q, K, and V with exact dimension ordering.
2. Explain why Q has a leading `1` dimension.
3. Identify the silent bug that occurs when nkv padding changes the Q/KV head ratio.
4. State the paged KV cache shape and describe what each dimension represents.
5. Avoid the common mistake of writing expanded Q-head-count data into the paged KV cache.

---

## Prerequisites

Before reading this chapter, you should be comfortable with:

- **Chapter 1** — GQA fundamentals, the group_size invariant, and the paged layout concept.
- **Chapter 2** — The `paged_sdpa_decode` function signature and its tensor shape requirements.

---

## Conceptual Diagram: Head Grouping

The head-grouping formula and padding collision table are covered in full in `gqa_grouping_in_kernel.md`.

---

## Contents

| File | Topic |
|------|-------|
| [`head_axis_conventions.md`](head_axis_conventions.md) | TTNN tensor dimension ordering for Q, K, V at decode time; the legacy KV cache layout change and its silent failure mode |
| [`gqa_grouping_in_kernel.md`](gqa_grouping_in_kernel.md) | How the Flash-Decode kernel maps Q heads to KV heads; padding silent correctness bug |
| [`paged_layout_for_gqa.md`](paged_layout_for_gqa.md) | Paged KV cache shape, what each dimension represents, and the expanded-head-count write mistake |

---

## Next Steps

Start with [`head_axis_conventions.md`](head_axis_conventions.md) to establish the exact axis ordering before examining how the kernel uses those axes.
