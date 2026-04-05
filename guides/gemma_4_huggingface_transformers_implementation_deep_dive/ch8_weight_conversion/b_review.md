# Agent B Review: Chapter 8 — Pass 1

## Issue 1: Special token count is wrong (Section 8.8.2)
The chapter states "The `extra_special_tokens` dict defines 16 special tokens for multimodal delimiters and tool-calling." The table that follows lists 18 tokens (image_token, boi_token, eoi_token, audio_token, boa_token, eoa_token, sot_token, eot_token, soc_token, eoc_token, think_token, escape_token, str_token, etr_token, stc_token, etc_token, std_token, etd_token). The source code at lines 1142-1161 confirms 18 entries. The count should be 18, not 16.

## Issue 2: Missing transpose for mm_input_projection and audio_input_projection (Section 8.4)
The dispatcher table in Section 8.4 lists `...mm_input_projection` and `...audio_input_projection` as "Direct mapping." However, the source code applies `.transpose()` to both values before storing them (lines 1062 and 1065: `value.transpose()`). These are not direct copies; they are transposed. The table should note the transpose.

## Issue 3: Misleading parenthetical for sliding K proj shape (Section 8.9)
In the 31B weight shape table, the k_proj row says: "`(4096, 5376)` (full) or `(1024, 5376)` (sliding, 4 gkv heads)". The parenthetical "4 gkv heads" is misleading. Sliding attention layers use `kv_einsum` with `num_key_value_heads=16` and `head_dim=64`, yielding 16 * 64 = 1024. The number 4 refers to `num_global_key_value_heads`, which is used only for the separate `k_einsum` path (applicable to full attention layers when `attention_k_eq_v=True`), not for sliding layers. The parenthetical should say "16 kv_heads * 64 head_dim" to match what actually produces the 1024 dimension.

## Issue 4: Code snippet in Section 8.3 references undefined variable
The `_restore_checkpoint()` code block ends with `return checkpointer.restore(checkpoint_path, args=restore)` but the variable `restore` is never defined in the shown snippet. The source code defines it on line 1020 as `restore = obc_args.PyTreeRestore(item=target, restore_args=restore_args_tree)`. This missing line makes the code snippet non-functional and could confuse readers trying to understand the restore args construction.

# Agent B Review: Chapter 8 — Pass 2
No feedback — chapter approved.
