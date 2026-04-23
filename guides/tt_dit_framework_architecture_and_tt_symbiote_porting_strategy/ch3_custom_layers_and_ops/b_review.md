# Chapter 3 -- Agent B Review

## Issue 1: Embedding sub-category count is wrong

**File:** `index.md`, line 74

The text says "TT-DiT's embedding layers are model-specific and fall into **three** sub-categories" but then enumerates four numbered items: (1) Sinusoidal timestep embeddings, (2) Patch embeddings, (3) Combined embeddings, (4) Token embedding. Either change "three" to "four" or merge two of the categories.

**Severity:** Wrong numerical answer.

**Status:** Fixed (confirmed "four" in current text).

---

No other issues. All TTNN op names, class names, weight shapes, parallelism modes, API comparisons, code snippets, navigation footers, and cross-chapter links were verified against the source code in `models/tt_dit/layers/normalization.py`, `models/tt_dit/layers/conv2d.py`, `models/tt_dit/layers/conv3d.py`, and `models/experimental/tt_symbiote/modules/normalization.py` / `conv.py`. Factual claims about missing TT-Symbiote equivalents (DistributedLayerNorm, GroupNorm, Conv3d) are correct. Statistics dtype difference (float32 vs bfloat16 in distributed RMSNorm) is correctly reported. The Welford algorithm reference, weight interleaving formula, and Conv3d weight permutation are accurate.

---

## Pass 2 Review

### Issue 2: Embedding layer count is 10, not 7

**Files:** `index.md`, lines 22-23 (table) and line 72 (section heading)

The category table lists 7 embedding layers and the section heading says "Embeddings (7 layers)". The actual `models/tt_dit/layers/embeddings.py` file contains 10 `Module` subclasses:

1. `Timesteps`
2. `TimestepEmbedding`
3. `PixArtAlphaTextProjection`
4. `SD35CombinedTimestepTextProjEmbeddings`
5. `CombinedTimestepGuidanceTextProjEmbeddings`
6. `PatchEmbed`
7. `MochiPatchEmbed`
8. `WanPatchEmbed`
9. `WanTimeTextImageEmbedding`
10. `Embedding`

The table and count omit `PixArtAlphaTextProjection`, `SD35CombinedTimestepTextProjEmbeddings`, and `CombinedTimestepGuidanceTextProjEmbeddings`. Note that the sub-category text (lines 77-79) does mention the latter two under "Combined embeddings", making the count inconsistent with the text's own content.

**Severity:** Wrong numerical answer.

---

No other issues found in Pass 2. Verified against source code:

- Normalization: all five layer implementations, weight shapes, TTNN op names, `mesh_axes` declarations, compute kernel configs, and TT-Symbiote equivalents are accurately described. The `DistributedRMSNorm` statistics dtype (`float32`) and TT-Symbiote's (`bfloat16`) are correctly reported.
- Experimental ops catalog: all 15 ops in the summary table were cross-checked. Op names, parameter signatures, and TT-Symbiote equivalences (or lack thereof) are correct. The three shared attention ops (`nlp_create_qkv_heads`, `rotary_embedding_llama`, `nlp_concat_heads`) are indeed used by both frameworks.
- Convolution: Conv2d parallelism modes, weight/bias shapes, `assert dilation == (1, 1)`, lack of groups parameter, `WormholeComputeKernelConfig` with `HiFi4`, slice config pattern, and OOM error handler are all accurate. Conv3d weight permutation sequence, `HiFi2` compute config, causal padding logic, and blocking table match source. TT-Symbiote's `TTNNConv2dNHWC` builder pattern, caching, and fused variants are correctly described.
- Feedforward: `FeedForward` uses two `Linear` layers; `ParallelFeedForward` uses `ColParallelLinear` + `RowParallelLinear`. Both confirmed.

---

## Pass 3 Review

**No feedback -- chapter approved.**

Independent verification against source code confirms all factual claims are accurate after Pass 1 and Pass 2 fixes:

- **Numerical counts**: 5 normalization layers, 3 linear layers, 2 feedforward layers, 2 convolution layers, 10 embedding layers in 4 sub-categories. All match source.
- **TTNN op names and namespaces**: All 15 experimental ops in the catalog table verified. `wan_fused_rmsnorm_pre/post_allgather`, `dit_layernorm_pre/post_allgather`, `minimal_matmul`, `minimal_matmul_split`, `conv3d`, `nlp_create_qkv_heads`, `rotary_embedding_llama`, `nlp_concat_heads`, `dit_minimal_matmul_addcmul_fused`, `all_gather_async`, `reduce_scatter_minimal_async`, `neighbor_pad_async`, `slice_reshard_async` -- all match actual usage in the TT-DiT codebase.
- **TT-Symbiote equivalents**: `TTNNRMSNorm`, `TTNNLayerNorm`, `TTNNLocalRMSNorm`, `TTNNDistributedRMSNorm`, `TTNNConv2dNHWC`, `TTNNConv2dBNNHWC`, `TTNNConv2dBNActivationNHWC`, `TTNNConv2dNHWCInputMultipleOf16`, `TTNNBottleneck`, `TTNNPatchEmbedding`, `TTNNLinearIColShardedWRowSharded`, `TTNNLinearIReplicatedWColSharded` -- all exist in source and are described correctly.
- **Implementation details**: Conv2d `HiFi4` compute config, Conv3d `HiFi2` compute config, `DistributedRMSNorm` float32 statistics vs TT-Symbiote bfloat16, weight shapes (`[1, dim]` vs `[32, dim]`), GroupNorm `[B, 1, H*W, C]` reshape, Conv3d 16-byte alignment padding, DistributedLayerNorm Welford reciprocal caching, and PatchEmbed unfolded linear decomposition -- all confirmed against source.
- **Gap analysis**: No DistributedLayerNorm, GroupNorm, or Conv3d in TT-Symbiote -- confirmed by absence in `modules/normalization.py` and `modules/conv.py`.
