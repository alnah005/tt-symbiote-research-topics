# cur_pos Validation

`cur_pos` controls which KV positions are visible to each decode step. Errors here
produce subtly wrong outputs that pass shape checks and have misleading PCC values.

## Semantic Definition

`cur_pos[i]` is the **post-write KV length** for batch element `i`. It is the number
of tokens already present in the KV cache after the current write, not the zero-based
write index.

- If the sequence has 5 tokens and you are writing the 6th: `cur_pos[i] = 6`.
- The kernel uses `cur_pos[i]` as the exclusive upper bound for the attention mask.

## 1. Logging cur_pos

Log `cur_pos` immediately before every call to the decode op:

```python
print(f"[step {step}] cur_pos = {cur_pos}")
out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
    q, k, v, cur_pos=cur_pos, ...
)
```

Confirm the values match your expected sequence lengths. A single transposition or
off-by-one here can cause the model to attend to future tokens or mask valid context.

## 2. Off-by-One Test

Run exactly two consecutive decode steps on a minimal input and verify the increment:

```python
step_0_pos = cur_pos_after_prefill   # e.g., [10] for a batch of 1
# decode step 1
out_1 = decode(q1, ..., cur_pos=step_0_pos)
step_1_pos = [p + 1 for p in step_0_pos]
# decode step 2
out_2 = decode(q2, ..., cur_pos=step_1_pos)

assert step_1_pos[0] == step_0_pos[0] + 1, \
    f"expected increment of 1, got {step_1_pos[0] - step_0_pos[0]}"
```

Common mistakes: incrementing `cur_pos` before the write instead of after, or
incrementing by the number of heads instead of 1.

## 3. Batch Consistency

For `b=1`, `cur_pos` must be `[42]`, not `42`; passing a bare scalar may silently broadcast or raise inside the op. For `b > 1` with divergent lengths, every element must have its own value — for the correct pattern, see Chapter 4, `per_user_vs_shared.md`.

## 4. Paged Mode Block Count Test

In paged mode, `cur_pos[i]` must be consistent with the number of fully-written blocks
in the page table for batch element `i`:

```python
def expected_full_blocks(cur_pos_i, block_size):
    return cur_pos_i // block_size

for i in range(b):
    full_blocks = expected_full_blocks(cur_pos[i], block_size)
    # count non-sentinel entries in row i of the page table
    written = (page_table[i] != SENTINEL).sum().item()
    assert written >= full_blocks, (
        f"batch[{i}]: cur_pos={cur_pos[i]} implies {full_blocks} full blocks, "
        f"but only {written} are allocated in the page table"
    )
```

A mismatch here means either `cur_pos` is wrong or the page table was not updated
before the decode call.

## 5. Testing the `-1` Skip Behavior

Passing `-1` for a batch element signals the op to skip that element; the corresponding output slice is undefined and must not be consumed by downstream layers without an explicit mask or zero-out. For the `-1` sentinel behavior and masking requirements, see Chapter 4, `cur_pos_definition.md`.

## 6. Validation Summary

| # | Check | Command / assertion |
|---|-------|---------------------|
| 1 | Log before every call | `print(f"cur_pos = {cur_pos}")` |
| 2 | Increment by exactly 1 per step | `assert new[i] == old[i] + 1` |
| 3 | Length-1 list for b=1 | `assert isinstance(cur_pos, list) and len(cur_pos) == b` |
| 4 | Block count consistent with page table | `cur_pos[i] // block_size <= allocated_blocks[i]` |
| 5 | `-1` outputs explicitly ignored | code review / assert downstream mask |

---

**Next:** [`pcc_comparison_workflow.md`](./pcc_comparison_workflow.md)
