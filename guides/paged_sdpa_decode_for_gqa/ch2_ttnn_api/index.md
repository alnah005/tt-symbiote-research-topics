# Chapter 2: TTNN paged_sdpa_decode API

## Description

This chapter documents the complete public interface of
`ttnn.transformer.scaled_dot_product_attention_decode`. It covers every
parameter, all tensor shapes for both paged and non-paged modes, the
`SDPAProgramConfig` fields, and the grid-size constraint that causes silent
correctness failures when violated.

---

## Prerequisites

You must have read Chapter 1 (`ch1_gqa_paged_fundamentals/`) before this
chapter. The terminology below is used without re-definition here:

- `nh`, `nkv`, `group_size`, `dh`, `b`, `s`, `block_size`
- `max_num_blocks`, `max_num_blocks_per_seq`, `cur_pos[i]`
- `pnh` (nh padded to nearest 32 for tile alignment)

If any of those terms are unfamiliar, return to Chapter 1 first.

---

## Learning Objectives

After reading this chapter you will be able to:

1. State the complete function signature of
   `ttnn.transformer.scaled_dot_product_attention_decode` including every
   parameter name, type, and default value.
2. Identify the Q, K, and V tensor shapes for both paged and non-paged modes
   and explain which axis carries which dimension.
3. Explain the output shape `[1 x b x pnh x dh]`, why `pnh` may differ from
   `nh`, and how to recover the real `nh` heads from the output tensor.
4. Describe what `cur_pos` and `cur_pos_tensor` represent (the count of valid
   KV tokens, not the write index), state the recompilation trade-off between
   them, and identify the silent failure caused by passing a scalar instead of
   a length-`b` list.
5. Explain the `SDPAProgramConfig` fields (`compute_with_storage_grid_size`,
   `q_chunk_size`, `k_chunk_size`) and how grid size controls parallelization
   over batch × KV head pairs.
6. Identify the constraint `num_cores >= b x nkv`, describe the symptom when
   it is violated (wrong or zero output for some batch elements or KV heads),
   and state the fix.

---

## Chapter Contents

| File | What it covers |
|------|----------------|
| [`function_signature.md`](./function_signature.md) | Full Python signature; per-parameter type, purpose, and gotchas |
| [`tensor_shape_reference.md`](./tensor_shape_reference.md) | Master shape table; layout and dtype requirements; silent failure catalog |
| [`sdpa_program_config.md`](./sdpa_program_config.md) | `SDPAProgramConfig` fields; parallelization strategy; worked example |

---

## Reading Order

Read the files in the order listed in the table above. `function_signature.md`
introduces every parameter name that `tensor_shape_reference.md` and
`sdpa_program_config.md` then elaborate on. Skipping ahead to the config file
without reading the signature first will leave the field names without context.
