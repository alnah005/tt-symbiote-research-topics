# Root Cause Isolation

Use this flowchart after shape validation and `cur_pos` validation pass but the PCC
comparison still fails. The goal is to determine whether the bug lives in the paging
logic, the GQA group-size logic, or the underlying attention kernel.

## Isolation Flowchart

```
PCC fails
│
├─ Shape check fails?
│  └─ YES → fix shape (see shape_validation_checklist.md), re-run
│
├─ cur_pos inconsistent?
│  └─ YES → fix cur_pos (see cur_pos_validation.md), re-run
│
├─ Disable paging → PCC recovers?
│  ├─ YES → bug is in paging logic (page table, block addressing, boundary)
│  │         check issue #30362; verify block_size arithmetic
│  └─ NO  → bug is independent of paging
│
├─ Set nkv = nh (non-GQA) → PCC recovers?
│  ├─ YES → bug is in GQA group-size logic (group_size computation, nkv padding)
│  │         verify nkv_padded = nh_padded / original_group_size
│  └─ NO  → bug is independent of GQA
│
├─ Switch cur_pos_tensor ↔ cur_pos list → behavior changes?
│  ├─ YES → compilation/caching artifact; clear op cache and recompile
│  └─ NO  → not a caching issue
│
└─ All above pass → likely kernel bug
   → file issue with minimal reproducer (see "Escalation" below)
```

## Step 1: Disable Paging (Contiguous K/V)

Replace the paged K/V tensor with a standard contiguous tensor of shape
`[b, nkv, s, dh]`. Keep all other parameters identical.

```python
# Paged path (original):
out_paged = ttnn.transformer.paged_scaled_dot_product_attention_decode(
    q, k_paged, v_paged, cur_pos=cur_pos, page_table=page_table, ...
)

# Contiguous path (test):
out_contig = ttnn.transformer.scaled_dot_product_attention_decode(
    q, k_contig, v_contig, cur_pos=cur_pos, ...
)

if pcc(out_contig, ref) > 0.99 and pcc(out_paged, ref) < 0.99:
    print("Bug is in paging logic, not attention kernel")
```

If the contiguous path also fails, paging is not the cause.

## Step 2: Disable GQA (Set nkv = nh)

Replicate KV heads so `nkv == nh` (group_size = 1). This removes all GQA code paths.

```python
k_full = k_contig.repeat_interleave(group_size, dim=1)   # [b, nh, s, dh]
v_full = v_contig.repeat_interleave(group_size, dim=1)

out_mha = ttnn.transformer.scaled_dot_product_attention_decode(
    q, k_full, v_full, cur_pos=cur_pos, ...
)

if pcc(out_mha, ref_mha) > 0.99 and pcc(out_gqa, ref_gqa) < 0.99:
    print("Bug is in GQA group-size logic")
```

Check: is `nkv_padded` computed correctly? The correct formula is:

```python
nkv_padded = nh_padded // original_group_size
```

Padding `nkv` to a different value (e.g., next power of two independently) silently
changes the group size.

## Step 3: cur_pos_tensor vs cur_pos List

Some bugs arise from op compilation being cached with a different `cur_pos` type. If one form succeeds and the other fails, clear the op program cache with `device.clear_compiled_program_cache()` and rerun. For the cur_pos format reference, see Chapter 4, `cur_pos_definition.md`.

## Step 4: Escalation — Filing a Kernel Bug

Escalate to a kernel bug report only after confirming all of the following:

- Contiguous K/V path still fails (paging is not the cause).
- nkv = nh still fails (GQA group size is not the cause).
- Shapes are correct.
- `cur_pos` is correct.
- Both `cur_pos` list and tensor forms fail.

**Minimal reproducer checklist for issue filing:**

1. Hardware: Wormhole B0 (6 DRAM controllers, not 8).
2. Exact `nh`, `nkv`, `dh`, `b`, `s`, `cur_pos` values that trigger the failure.
3. PCC value observed vs expected (>0.99).
4. Whether the failure is position-dependent (binary search result from
   `pcc_comparison_workflow.md`).
5. Reference to issue #30362 if the failure is at a block boundary.
6. Self-contained Python script that reproduces the failure on a single device.

File the issue with the tt-metal repo and tag the attention kernel owners.
