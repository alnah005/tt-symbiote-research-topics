# B Review — Chapter 1 — Pass 1

1. **ttnn_moe_forward.md, line 59** — The `topology` row in the all-gather parameter table reads: "Chain gather; each device forwards to the next along a **linear ring**." Linear topology is not a ring. A ring has wrap-around (last device connects back to first); a Linear chain does not. A reader implementing a custom CCL or reasoning about tail-device behavior would model the wrong physical graph. Fix: change "along a linear ring" to "along a linear chain (no wrap-around)."

2. **cpu_fallback_paths.md, line 21 (section header) and line 112 (checklist grep output)** — The document consistently cites the hardcoded flag as `moe.py:L571`, but the ground-truth source places the `ttnn = False` assignment at lines 569–570. A reader running `grep -n "ttnn = False" moe.py` and comparing against line 571 will see the expected line in the wrong location, which can cause confusion when navigating the file or applying a patch. Fix: update both references from `L571` to `L569–570`.

No further correctness issues found. All CCL parameters (topology, num_links, chunks_per_sync, num_workers_per_link, cluster_axis, dim), precision configs (HiFi4 gate linear, HiFi2 expert matmuls, fp32_dest_acc_en, packer_l1_acc, math_approx_mode), alignment constants (SPARSITY_BLOCK_SIZE=32, TOPK_MIN_WIDTH=64), weight application shape transforms (repeat_dims=(hidden_size,1,1,1), permute (3,1,2,0)), and navigation footers are all correct. index.md uses clickable links throughout.

# B Review — Chapter 1 — Pass 2

No feedback — chapter approved.

# B Review — Chapter 1 — Pass 3

No feedback — chapter approved.
