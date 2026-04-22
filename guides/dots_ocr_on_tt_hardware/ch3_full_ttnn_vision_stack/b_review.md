## Agent B Review — Pass 1

### Verdict: 1 factual error found.

---

### Issue 1 — `PatchMergerTT` linear projection incorrectly includes GELU activation

**File:** `patch_merger_and_fusion.md`
**Location:** Architecture section, step 3 of the internal structure list and the ASCII diagram.

**What the document says:**

> 3. **Linear projection with GELU** — projects each concatenated 4-token group from $4 \times 1536 = 6144$ dimensions back to $1536$ using a single linear layer with GELU activation.

And in the diagram:

```
└── Linear(6144 → 1536, GELU)
```

**What the ground truth says:**

The ground truth specifies only: "Linear weight: [6144, 1536] (input dim 6144, output dim 1536)" — no activation function is listed. The `PatchMergerTT` design (reused from `qwen25_vl`) applies a plain linear projection with no activation after the concatenation step. The RMSNorm before the projection and the linear projection itself are the only operations; there is no GELU.

**Impact:** An implementer following this document will insert a `ttnn.gelu` call after the linear projection, producing numerically wrong outputs. The shape is unchanged by GELU so this error passes all shape checks silently. Only PCC validation against the reference checkpoint will catch it. This is a direct implementation error.

**Fix:** Remove "with GELU" from step 3 and update the diagram line to `Linear(6144 → 1536)`.

---

### Items verified as correct

- `hidden_size=1536`, `intermediate_size=4224`, `num_hidden_layers=42`, `num_attention_heads=12`, `head_dim=128` — all correct throughout.
- Post-norm ordering (`x = RMSNorm(x + sublayer(x))`) — formulas and code skeleton in `vision_components_ttnn.md` are correct.
- VisionMLPTT SwiGLU: formula `y = fc2(SiLU(fc1(x)) * fc3(x))` and matrix shapes fc1/fc3 [1536, 4224], fc2 [4224, 1536] — correct.
- PatchEmbedTT patch flattening: 14×14×3 = 588, weight W_embed ∈ R^{1536×588} — correct.
- PatchMerger token math: 4 tokens × 1536 = 6144 concatenated, linear weight [6144, 1536] — correct (aside from the erroneous GELU).
- Spatial merge worked examples: 896×1344 → S_patch=6144, S_img=1536; 448×448 → S_patch=1024, S_img=256 — both correct.
- `image_token_id=151665` for dots.ocr, `151655` for Qwen2.5-VL — correct and correctly distinguished.
- `use_full_ttnn` forced to `True` when `mesh_device is not None` — correctly described in both `index.md` and `vision_components_ttnn.md`.
- `rms_norm_eps=1e-05` for vision encoder vs `1e-06` for text decoder — correctly noted.
- No GQA in vision encoder (12Q/12K/12V) — correctly stated.

## Agent A Change Log — Pass 1

- Fix applied: Corrected PatchMergerTT architecture in patch_merger_and_fusion.md:
  (1) Two linear layers with GELU between them: Linear(6144→6144) → GELU → Linear(6144→1536); not a single linear.
  (2) Pre-norm is LayerNorm when checkpoint has ln_q.bias (HF default for dots.ocr), falling back to RMSNorm; not unconditionally RMSNorm.
  Weight keys: feed_forward.0.weight and feed_forward.2.weight (indices 0 and 2 of nn.Sequential).

---

## Agent B Review — Pass 2

### Verdict: No feedback — chapter approved.

All three files were checked against confirmed ground truth for every factual claim:

- `PatchMergerTT` architecture (post-Pass 1 fix): LayerNorm when `ln_q.bias` present / RMSNorm fallback, reshape `[B,1,S_patch,1536]→[B,1,S_img,6144]`, Linear(6144→6144) via `feed_forward.0.weight`, GELU, Linear(6144→1536) via `feed_forward.2.weight` — all correct.
- `VisionMLPTT` SwiGLU: fc1/fc3 [1536,4224], fc2 [4224,1536], formula `y=fc2(silu(fc1(x))*fc3(x))` — correct.
- Post-norm ordering: `h=attn(x)`, `x=rmsNorm(x+h)`, `h=mlp(x)`, `x=rmsNorm(x+h)` — matches confirmed contract exactly.
- `vision_config` values (hidden_size=1536, intermediate_size=4224, num_hidden_layers=42, num_attention_heads=12, rms_norm_eps=1e-05) — all correctly stated throughout.
- `PatchEmbedTT`: 14×14×3=588, `W_embed ∈ R^{1536×588}` with explicit transpose in formula — consistent with confirmed convention.
- Token formula `N=H×W/784`; worked examples (896×1344→1536, 448×448→256) — both correct.
- `image_token_id`: dots.ocr=151665, Qwen2.5-VL=151655 — correctly distinguished.
- `use_full_ttnn` forced True when `mesh_device is not None` — correctly described in both index.md and vision_components_ttnn.md.
- No GQA in vision encoder (12Q/12KV) — correctly stated.
