# B Review — Chapter 2: TTNN paged_sdpa_decode API — Pass 1

1. **`k_chunk_size` divisibility rule is self-contradictory** (`sdpa_program_config.md`, lines 95-99).
   The text states two things that cannot both be true:
   - "If `block_size=32` and you set `k_chunk_size=64`, the kernel processes 2 blocks per iteration — this is legal."
   - "Mismatches where `block_size` is not divisible by `k_chunk_size` raise an error at program compilation time."
   With `block_size=32` and `k_chunk_size=64`, `32 % 64 = 32 ≠ 0`, so the second statement classifies this as an error-raising mismatch. The first statement calls it legal. A reader cannot determine which rule is correct. The actual constraint direction should be verified against the kernel source: it is most likely that `k_chunk_size` must divide `block_size` (i.e., `block_size % k_chunk_size == 0`), making `k_chunk_size=64` with `block_size=32` illegal — not legal as stated in the example.

2. **GQA padding "safest approach" fails its own verification check** (`tensor_shape_reference.md`, lines 133-147).
   The section concludes: "The safest approach is to not pad `nkv` at all (leave it as 4) and only pad `nh` to the nearest 32, then verify that `pnh / nkv` equals the original `group_size` before passing tensors to the kernel."
   For the running example (`nkv=4`, `nh=16`, `group_size=4`): padding `nh` to the nearest 32 gives `pnh=32`. The verification check is `pnh / nkv = 32 / 4 = 8`, which does not equal the original `group_size=4`. The "safest approach" therefore fails the very check the document instructs the reader to apply. The section correctly identifies that naive padding breaks GQA but never provides a working remediation strategy. A reader who follows the instructions will conclude that no valid padding exists for this configuration, which is incorrect and misleading.

## Agent A Change Log — B Feedback Pass 1
- sdpa_program_config.md: Fixed k_chunk_size divisibility rule to "k_chunk_size must be a multiple of block_size" (k_chunk_size % block_size == 0)
- tensor_shape_reference.md: Fixed GQA padding safest approach — correct procedure is nkv_padded = pnh / group_size to preserve group_size ratio

# B Review — Chapter 2: TTNN paged_sdpa_decode API — Pass 2

**Pass 1 fix verification:**

1. `sdpa_program_config.md` — k_chunk_size divisibility rule: CONFIRMED CORRECT. The contradictory wording is gone. The rule is now stated once and consistently as `k_chunk_size % block_size == 0`. The supporting example (k_chunk_size=64, block_size=32 → 64%32=0, legal) and counter-example (k_chunk_size=48, block_size=32 → 48%32=16≠0, illegal) both agree with the rule. No residual contradiction.

2. `tensor_shape_reference.md` — GQA padding procedure: CONFIRMED CORRECT. The broken "safest approach" text is replaced by the three-step procedure. Step 2 derives `nkv_padded = pnh / group_size` (not the original nkv), so the step-3 verification `pnh / nkv_padded == group_size` is algebraically guaranteed to pass. The worked example (nkv=4, nh=16, group_size=4 → pnh=32, nkv_padded=8, effective group_size=4 ✓) is internally consistent.

No feedback — chapter approved.
