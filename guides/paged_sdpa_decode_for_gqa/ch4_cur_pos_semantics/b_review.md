# B Review — Chapter 4: cur_pos Semantics — Pass 1

No errors found — chapter approved.

All key facts verified against the provided reference:

- `cur_pos[i]` is correctly defined as the length of the valid KV prefix (not the index of the token being written) across all files.
- After the first decode step, `cur_pos[i] = 1` is stated correctly in `cur_pos_definition.md`.
- The `-1` sentinel is correctly documented: skips computation for that batch element; output left undefined.
- Both passing modes are correctly described: Python list (CPU tensor, recompiles per unique tuple) and `cur_pos_tensor` (device tensor, avoids recompilation).
- Per-user semantics are correctly documented: each batch element has an independent `cur_pos[i]`.
- Common Mistake 2 (off-by-one) correctly identifies using the token index being written instead of the cache length after writing.
- `num_active_blocks[i] = ceil(cur_pos[i] / block_size)` is stated correctly in `cur_pos_in_paged_mode.md`.
- Issue #30362 is correctly described as sporadic PCC failures near block boundaries, suspected block-boundary arithmetic, open as of early 2026.
- Tensor shapes are all correct: paged KV cache `[max_num_blocks x nkv x block_size x dh]`, `paged_update_cache` input `[b x nkv x 1 x dh]`, Q tensor `[1 x b x nh x dh]`.
- GQA mapping formula `kv_head_idx = q_head_idx // group_size` is correct.
- Block boundary examples are correct: `cur_pos=64` yields exactly 2 full blocks; `cur_pos=65` yields 3 blocks with the third partially valid.
