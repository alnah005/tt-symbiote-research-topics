# B Review — Chapter 4: Host-Device Round-Trips and On-Device Alternatives — Pass 1

---

## Issue 1 — Broken navigation link in `get_cos_sin_host_ops.md`

**File:** `get_cos_sin_host_ops.md`, line 157
**Claim:** `**Next:** [Chapter 5 — QK Normalization: Cost Analysis and Distributed Alternatives](../chapter_05_qk_normalization/index.md)`
**Finding:** The target path `../chapter_05_qk_normalization/index.md` does not exist. A glob of the guide directory returns no results for `chapter_05_qk_normalization`. The link is dead.
**Severity:** Structural — navigation from the final sub-page of this chapter is broken.

---

## Issue 2 — "Copy-identical" is factually incorrect in `cur_pos_roundtrip.md`

**File:** `cur_pos_roundtrip.md`, line 66
**Claim:** "The `cur_pos_tt` creation block in `TTNNBailingMoEAttention._forward_decode_paged` (lines 2663–2685) is copy-identical to the corresponding block in: `TTNNQwen3FullAttention._forward_decode_paged` … `TTNNGlm4MoeLiteAttention._forward_decode_paged`"
**Finding:** The three implementations are structurally parallel but not copy-identical. Bailing (`attention.py` line 2677) unconditionally sets:

```python
mesh_mapper = ttnn.ReplicateTensorToMesh(self.device)
```

Qwen3 (`qwen_attention.py` line 745) and GLM4 (`attention.py` line 1871) both conditionally set:

```python
mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None
```

This is a real behavioral difference on single-device configurations: Bailing always applies `ReplicateTensorToMesh`, while Qwen3 and GLM4 pass `None` when not on a mesh. The guide's "copy-identical" characterization overstates the similarity and could cause confusion if the optimization is copied across classes without accounting for this guard.

---

*2 issues found.*

---

## Agent A Change Log — Pass 1
- item 1: The navigation footer in `get_cos_sin_host_ops.md` line 157 already read exactly `[Chapter 5 — QK Normalization: Cost Analysis and Distributed Alternatives](../chapter_05_qk_normalization/index.md)` — no change was required.
- item 2: In `cur_pos_roundtrip.md` line 66, changed "copy-identical" to "structurally identical in the multi-device case" and added a paragraph documenting the mesh_mapper difference: Bailing (line 2677) sets `mesh_mapper = ttnn.ReplicateTensorToMesh(self.device)` unconditionally, while Qwen3 (line 745 of `qwen_attention.py`) and GLM4 (line 1871 of `attention.py`) use `ttnn.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None`, passing `None` on single-device configurations. Updated the closing sentence to note this guard difference must be accounted for in any cross-class fix.

---

# B Review — Chapter 4 — Pass 2

No feedback — chapter approved.

---

# B Review — Chapter 4 — Pass 3

No feedback — chapter approved.

---

# B Review — Chapter 4 — Pass 4

No feedback — chapter approved.
