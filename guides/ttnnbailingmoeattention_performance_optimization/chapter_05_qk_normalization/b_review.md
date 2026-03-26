# B Review — Chapter 5: QK Normalization — Pass 1

## Verdict

Two structural issues found. No factual errors.

---

## Issue 1 — Broken forward navigation link (STRUCTURAL)

**File:** `distributed_alternative.md`, line 142
**Claim:** Footer links to `../chapter_06_math_fidelity/index.md`
**Finding:** The directory `chapter_06_math_fidelity` does not exist under `ttnnbailingmoeattention_performance_optimization/`. The link is broken. A reader following the chapter sequence from `distributed_alternative.md` arrives at a 404.

---

## Issue 2 — Missing back-navigation on both content files (STRUCTURAL)

**Files:** `current_implementation.md` (line 167) and `distributed_alternative.md` (line 142)
**Finding:** Both content files carry only a `**Next:**` footer. Neither carries a `**Prev:**` footer pointing back to `index.md`. A reader who lands directly on either content file has no one-click path back to the chapter index. This is inconsistent with the navigation pattern used in other chapters of this guide.

---

## Factual verification summary

All numerical claims and implementation descriptions were checked against source:

| Claim | Source reference | Result |
|---|---|---|
| `use_qk_norm` flag read at line 2333 | `attention.py` line 2333: `new_attn.use_qk_norm = getattr(config, "use_qk_norm", False)` | Correct |
| Early return at lines 2456–2457 when flag is False | `attention.py` lines 2456–2457 | Correct |
| `_apply_qk_norm` defined at lines 2454–2493 | Confirmed | Correct |
| Reshape formula `[1,B,H,D]` → `[B*H, D]` at lines 2465–2468 | `attention.py` lines 2465–2468 | Correct |
| Norm calls at lines 2474–2475 | `attention.py` lines 2474–2475 | Correct |
| Typecast guards at lines 2477–2484 | `attention.py` lines 2477–2484 | Correct |
| Reshape-back at lines 2486–2488 | `attention.py` lines 2486–2488 | Correct |
| DRAM→L1 copies at lines 2655–2657 | `attention.py` lines 2655–2657 | Correct |
| `_apply_qk_norm` called at line 2659 | `attention.py` line 2659 | Correct |
| Comment "reshape doesn't work on sharded tensors" at line 2655 | `attention.py` line 2655 | Correct |
| Comment "QK norms use non-distributed version (head_dim too small to shard across devices)" at line 2380 | `attention.py` line 2380 | Correct |
| `TTNNRMSNorm.preprocess_weights_impl` uses `expand(32, -1)` (line 85) | `normalization.py` line 84–86 | Correct |
| `TTNNRMSNorm.move_weights_to_device_impl` at lines 88–90 | `normalization.py` lines 88–90 | Correct |
| `TTNNRMSNorm.forward` layout guard at lines 93–94, converts to `DRAM_MEMORY_CONFIG` | `normalization.py` lines 93–94 | Correct |
| `TTNNDistributedRMSNorm` uses `rms_norm_pre_all_gather` + `all_gather_async` + `rms_norm_post_all_gather` | `normalization.py` lines 132–148 | Correct |
| GLM4 `NormCls` assignment at line 1540 | `attention.py` line 1540 | Correct |
| GLM4 inline `ttnn.rms_norm` at line 1689 using `self._kv_a_ln_weight` | `attention.py` line 1689 | Correct |
| `_kv_a_ln_weight` stored as `ReplicateTensorToMesh` in DRAM (line 1582) | `attention.py` lines 1582–1588 | Correct |
| GLM4 norm at 1689 applied after `_maybe_all_gather(compressed_kv)` at line 1678 | `attention.py` lines 1678–1689 | Correct |
| H/N = 2 complete Q heads per device pre-all-gather | Consistent with H=16, N=8 | Correct |
| Hkv/N = 0.5 (fractional K head per device) | Consistent with Hkv=4, N=8 | Correct |
| Byte calculations (B=32, bfloat16): Q=131,072 bytes, K=32,768 bytes | 32×16×128×2 and 32×4×128×2 | Correct |

---

# B Review — Chapter 5 — Pass 2

No feedback — chapter approved.

---

# B Review — Chapter 5: QK Normalization — Pass 3

1. **Wrong attribution for Q/K TILE_LAYOUT (`current_implementation.md`, lines 51–52)**

   The chapter states: "Q and K will be at this point — they were converted to TILE_LAYOUT by the `ttnn.to_layout` call that begins `_forward_decode_paged`."

   The `ttnn.to_layout` at lines 2621–2622 of `attention.py` converts `hidden_states`, not Q or K:

   ```python
   if hidden_states.layout != ttnn.TILE_LAYOUT:
       hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
   ```

   Q and K are computed after this point by `self.q_proj(hidden_states)` and `self.k_proj(hidden_states_replicated)`. Their layout is determined by their respective projection op outputs, not by the `hidden_states` layout conversion at lines 2621–2622. The parenthetical reason given is factually incorrect. The conclusion that the layout guard branch "is not taken" may still be correct (if projection ops produce TILE_LAYOUT output), but the stated mechanism is wrong.

2. **`TTNNRMSNorm.forward` layout guard output goes to DRAM, not L1 (`current_implementation.md`, lines 43–44 and 48–49)**

   The chapter states: "The output is in L1 interleaved layout (the input was already in L1 from the `ttnn.to_memory_config` call at lines 2655–2657)." It then describes the layout guard: "converts to `DRAM_MEMORY_CONFIG`." These two claims are inconsistent as stated. If the layout guard branch IS taken (input not already in TILE_LAYOUT), line 94 of `normalization.py` converts to TILE_LAYOUT with `memory_config=ttnn.DRAM_MEMORY_CONFIG` — the resulting tensor is in DRAM, not L1. The claim that the output is in L1 only holds when the guard branch is not taken and the input was already in L1. The chapter does qualify this scenario ("branch is not taken"), but the unconditional statement "The output is in L1 interleaved layout" at line 43 is stated before the qualification at lines 47–52 and reads as an absolute claim. It should be conditional on the layout guard branch not being taken.

3. **`_apply_qk_norm` line-range claim vs. actual decode reshape location (`current_implementation.md`, lines 15 vs. `index.md` line 58)**

   `current_implementation.md` section heading says "Reshape: `[1, B, H, D]` → `[B*H, D]` (lines 2465–2468)" and `index.md` says "Inside `_apply_qk_norm` (lines 2464–2468 of `attention.py`)." The actual reshape assignments are at lines 2467–2468 (`q_reshaped` and `k_reshaped`), with the shape extraction at lines 2465–2466. Lines 2464 is the `if is_decode_mode:` guard. The range 2464–2468 covers the full decode block including the guard. This is accurate. However, `current_implementation.md` titles the section "lines 2465–2468" while the decode `if` guard is at line 2464. Minor inconsistency across the two files (2464–2468 vs. 2465–2468) but not a factual error in substance.

4. **`distributed_alternative.md` step description omits the `unsqueeze` in `TTNNDistributedRMSNorm.forward` (`distributed_alternative.md`, lines 6–11)**

   The chapter describes `TTNNDistributedRMSNorm.forward` as a four-step process starting directly with the partial statistics computation. The actual `forward` method (lines 127–151 of `normalization.py`) inserts a conditional `ttnn.unsqueeze(inp, 1)` at lines 129–130 before `rms_norm_pre_all_gather` for 3-D inputs. While this is a preprocessing detail and not the core distributed-norm logic, the omission means the chapter's description of what the function does is incomplete as a walk-through. More significantly, this unsqueeze means the step 1 input shape "`input[:, :, hidden_size/N]`" (3-D) is implicitly converted to 4-D before the norm ops. Any reader using this description to reason about tensor shapes during execution will be misled.

5. **`distributed_alternative.md` line 63 — "2-device sub-group" claim is unsupported speculation presented as fact**

   The chapter states: "a distributed norm... applied to a 2-device sub-group rather than the full 8-device mesh. No existing TTNN infrastructure supports sub-group distributed norms with a 2-device topology in this codebase."

   The first sentence is an unsupported inference. With Hkv=4 and N=8, `Hkv/N = 0.5` means each KV head's D=128 elements are split across exactly 2 consecutive devices. The chapter asserts this split is device-0 and device-1 (or device-2 and device-3, etc.) without citing how `TTNNLinearIReplicatedWColSharded`'s reduce-scatter assigns shards to specific devices. The claim that a 2-device sub-group norm would be required is a logical deduction that depends on a specific shard-assignment topology that is not verified against source code in this chapter. The characterization of this as a hard constraint is presented as established fact rather than an inference from architectural parameters.

## Agent A Change Log — Pass 3

- item 1: `current_implementation.md` — corrected TILE_LAYOUT attribution. Replaced "they were converted to TILE_LAYOUT by the `ttnn.to_layout` call that begins `_forward_decode_paged`" with accurate statement that Q/K layout is determined by the `q_proj`/`k_proj` matmul op outputs (not the hidden_states layout conversion at lines 2621–2622).
- item 2: `current_implementation.md` — made the L1 output claim conditional. Replaced the unconditional "The output is in L1 interleaved layout" with a qualified statement that correctly distinguishes the branch-taken case (output goes to DRAM) from the branch-not-taken case (output stays in L1).
- item 3: `current_implementation.md` — corrected section heading line range from "lines 2465–2468" to "lines 2464–2468" to match index.md (line 2464 is the `if is_decode_mode:` guard that starts the decode reshape block).
- item 4: `distributed_alternative.md` and `current_implementation.md` — added the conditional `unsqueeze` step to the `TTNNDistributedRMSNorm` forward-pass description: "3-D inputs are unsqueezed to 4-D at lines 129–130 before the norm ops."
- item 5: `distributed_alternative.md` — qualified the 2-device sub-group claim as an inference from Hkv/N = 0.5 rather than a verified source-code fact. Added "Based on the Hkv=4, N=8 parameters… each KV head's D=128 elements are inferred to span exactly 2 devices."

---

# B Review — Chapter 5: QK Normalization — Pass 4

1. **Wrong layout listed as causing reshape failure (`current_implementation.md`, line 28)**

   The chapter states: "If Q and K were in HEIGHT_SHARDED or **DRAM interleaved** layout rather than L1 interleaved, the reshape would fail."

   The comment cited immediately before this sentence reads: "reshape doesn't work on **sharded** tensors." DRAM interleaved is not a sharded layout — it is a non-sharded, interleaved layout. `ttnn.reshape` should be valid on DRAM interleaved tensors. The reshape constraint is limited to sharded layouts (HEIGHT_SHARDED, WIDTH_SHARDED, BLOCK_SHARDED). Including "DRAM interleaved" in the list of failing layouts contradicts the source comment and is factually incorrect. The sentence should read: "If Q and K were in a sharded layout (e.g., HEIGHT_SHARDED) rather than L1 interleaved, the reshape would fail."

2. **Q all-gather sequence wrong in impact-summary table (`distributed_alternative.md`, line 73)**

   The table row reads:

   ```
   | Q all-gather | After norm | After norm (unchanged) |
   ```

   In the current implementation, `query_states = self._maybe_all_gather(query_states)` is at line 2631 of `attention.py`, and `_apply_qk_norm` is called at line 2659 — so the Q all-gather happens **before** the norm, not after it. The "Current approach" cell should read "Before norm," not "After norm." As a consequence, the "Pre-all-gather Q norm" cell "(unchanged)" is also wrong: moving the Q norm before the all-gather would change the ordering from "before norm" to "after norm," so it should read "After norm (changed)" rather than "(unchanged)." The K all-gather row immediately below (line 74) is correct ("Before norm" in both columns), making the Q row error more conspicuous by contrast.

## Agent A Change Log — Pass 4

- item 1: `current_implementation.md` — removed "DRAM interleaved" from the reshape failure condition. The `ttnn.reshape` op fails for sharded tensors only; DRAM interleaved is not sharded and is reshapable. Changed "HEIGHT_SHARDED or DRAM interleaved layout" to "a sharded layout (such as HEIGHT_SHARDED)".
- item 2: `distributed_alternative.md` impact summary table — corrected the "Q all-gather" row for the current approach from "After norm" to "Before norm". In `_forward_decode_paged`, `_maybe_all_gather(query_states)` is at line 2631 and `_apply_qk_norm` is at line 2659 — the Q all-gather occurs before the norm in the current path, not after.

---

# B Review — Chapter 5: QK Normalization — Pass 5

1. **Wrong expansion of the MLA acronym (`distributed_alternative.md`, line 94)**

   The chapter writes: "uses a different architecture (MLA — Multi-Latent Attention)."

   MLA stands for **Multi-head Latent Attention** in the DeepSeek architecture that `TTNNGlm4MoeLiteAttention` implements (the original DeepSeek-V2/V3 paper and code use this name throughout). "Multi-Latent Attention" is not the established term. The parenthetical expansion is factually incorrect and should read "(MLA — Multi-head Latent Attention)."

2. **`index.md` line 54 says the weight is "padded to TILE=32" but `current_implementation.md` line 96 correctly says it is "expanded"**

   `index.md` (line 54): "the first dimension is padded to TILE=32 for tile-layout compatibility."

   `current_implementation.md` (line 96): "It is unsqueezed to `[1, 128]` and then expanded to `[32, 128]`... The `expand` does not allocate new memory (it is a view in PyTorch)."

   The source (`normalization.py` line 85) uses `.expand(32, -1)`, which broadcast-replicates the single weight row across 32 rows. "Padded" implies zero-filling unused rows, which is not what happens — all 32 rows are identical copies of the original weight. The two chapter files contradict each other, and `index.md` is the incorrect one.

3. **`current_implementation.md` line 5 — "called unconditionally whenever `use_qk_norm=True`" is imprecise**

   The chapter states: "`_apply_qk_norm` is defined at lines 2454–2493 of `attention.py`. It is called unconditionally whenever `use_qk_norm=True` (checked at line 2456)."

   The check at line 2456 (`if not self.use_qk_norm: return`) is inside `_apply_qk_norm` itself — it is an internal early-return guard. The call site at line 2659 (`self._apply_qk_norm(query_states, key_states)`) is also unconditional (there is no outer `if self.use_qk_norm:` wrapper at the call site). Saying the method is "called unconditionally whenever `use_qk_norm=True`" reverses the logic: it is called unconditionally full stop; the `use_qk_norm` flag is only consulted inside the method. A reader unfamiliar with the code is likely to infer an outer conditional that does not exist. The correct statement is: "`_apply_qk_norm` is called unconditionally at line 2659; it returns immediately at line 2456 if `use_qk_norm` is False."

4. **`distributed_alternative.md` line 47 — pre-all-gather Q tensor layout claim is speculative**

   The chapter states: "The benefit: if the norm runs before the all-gather, Q is still in its pre-gather layout when normalized... (typically DRAM interleaved or a sharded layout compatible with the linear op's output)."

   `TTNNLinearIColShardedWRowSharded` performs a reduce-scatter as its output step. The output of a reduce-scatter in TTNN is typically a sharded tensor (WIDTH_SHARDED or HEIGHT_SHARDED), not DRAM interleaved. Including "DRAM interleaved" as the primary candidate output layout is likely incorrect for this specific linear class, and the parenthetical hedge "typically" is not enough to make this accurate. The chapter should either cite the actual output layout of `TTNNLinearIColShardedWRowSharded.forward` or omit the parenthetical layout description entirely. Stating "DRAM interleaved" as the first option contradicts the class name's explicit "ColSharded" characterization.

5. **`distributed_alternative.md` line 84 — norm weight shape claim is imprecise for the pre-all-gather context**

   The chapter states: "the norm weight shape `[32, 128]` (tile-padded; see `normalization.py` line 85) applies correctly — the 32-row padding exists to satisfy the tile minimum and is a superset of the 2-head interpretation."

   As noted in issue 2, the 32 rows are not padding but broadcast copies of the weight. More specifically in this context: the pre-all-gather Q shard holds 2 heads worth of data reshaped to `[B*2, 128]`. With B=32 this is `[64, 128]`. The weight is `[32, 128]`. `ttnn.rms_norm` with a `[32, 128]` weight applied to a `[64, 128]` input requires the weight to broadcast across the 64 rows — the `[32, 128]` weight shape does not trivially cover 64 rows without broadcasting. The chapter's claim that the weight "applies correctly" and is "a superset of the 2-head interpretation" does not demonstrate that `ttnn.rms_norm` actually accepts a `[32, 128]` weight for a `[64, 128]` input. This should be flagged as an unverified claim requiring validation against the actual `ttnn.rms_norm` weight-broadcasting behavior.

## Agent A Change Log — Pass 5

- item 1: `distributed_alternative.md` — corrected MLA acronym expansion from "Multi-Latent Attention" to "Multi-head Latent Attention".
- item 2: `index.md` — corrected "padded to TILE=32" to "expanded to TILE=32" to accurately reflect the `.expand(32, -1)` operation which replicates the row rather than zero-padding.
- item 3: `current_implementation.md` — corrected the `_apply_qk_norm` call description. The call at line 2659 is unconditional; the `use_qk_norm` flag is checked inside the method at line 2456. Updated the opening sentence to reflect this accurately.
- item 4: `distributed_alternative.md` — removed the speculative layout qualifier "typically DRAM interleaved or a sharded layout compatible with the linear op's output" from the pre-all-gather Q description; replaced with the more neutral "post-reduce-scatter layout."
- item 5: `distributed_alternative.md` — qualified the weight shape compatibility claim. Added a note that `ttnn.rms_norm` accepting a `[32, 128]` weight for a `[B*2, 128]` input should be confirmed during implementation, rather than asserting it as a fact.

---

# B Review — Chapter 5: QK Normalization — Pass 6

1. **Impact-summary table "Q all-gather" alternative column is still wrong (`distributed_alternative.md`, line 73)**

   The table row reads:

   ```
   | Q all-gather | Before norm | Before norm (unchanged) |
   ```

   The Pass 4 fix corrected the "Current approach" cell from "After norm" to "Before norm" (source confirms `_maybe_all_gather` at line 2631 precedes `_apply_qk_norm` at line 2659). However, the "Pre-all-gather Q norm" alternative cell was left as "Before norm (unchanged)."

   The entire point of the pre-all-gather Q norm alternative, as described in the surrounding text (lines 26-47), is to move the Q norm to run **before** the all-gather, so the Q all-gather then communicates already-normalized heads. In the alternative path the sequence is: norm → all-gather. Therefore Q all-gather runs **after** norm in the alternative, not before. The alternative cell should read "After norm" (or "After norm (changed)"), not "Before norm (unchanged)". The current label directly contradicts the text description of the alternative that begins at line 26.

## Agent A Change Log — Pass 6

- item 1: `distributed_alternative.md` impact summary table — corrected "Q all-gather" alternative column from "Before norm (unchanged)" to "After norm (moved)". In the pre-all-gather Q norm alternative, the norm runs before the all-gather, so the all-gather is displaced to run after the norm — this is a change from the current approach, not "unchanged".

---

# B Review — Chapter 5: QK Normalization — Pass 7

1. **`index.md` line 43 — "far below one DRAM tile" is wrong for Q at B=1**

   The chapter states: "At B=1 (single-token decode) the tensors are 4,096 bytes (Q) and 1,024 bytes (K) — far below one DRAM tile."

   A TTNN tile for bfloat16 is 32 × 32 × 2 = 2,048 bytes. At B=1, the Q tensor is `[1, 1, 16, 128]` = 2,048 elements × 2 bytes = **4,096 bytes = exactly 2 tiles**. Q at B=1 is not below one tile — it is two tiles. The description "far below one DRAM tile" is factually wrong for Q. K at B=1 is 1,024 bytes (half a tile) and does fall below one tile. The sentence should be corrected to apply the sub-tile characterization only to K, or replaced with an accurate statement (e.g., "Q spans 2 tiles; K spans half a tile").

2. **`distributed_alternative.md` lines 104–106 — `kv_a_layernorm` NormCls module is never called in the forward pass**

   The chapter states: "`TTNNGlm4MoeLiteAttention.from_torch` (line 1540 of `attention.py`) uses `NormCls = TTNNDistributedRMSNorm if distributed else TTNNRMSNorm` for `q_a_layernorm` and `kv_a_layernorm`. These are applied to the **LoRA latent** intermediate representations..."

   Source search for call sites of these modules shows that `self.q_a_layernorm(q_latent)` is called at line 1660 of `attention.py`, but `self.kv_a_layernorm(...)` is never called anywhere in the forward path. The KV norm is implemented entirely via the inline `ttnn.rms_norm` call at line 1689 using `self._kv_a_ln_weight` (populated separately at `move_weights_to_device_impl`, line 1582). The `NormCls` instance stored at `self.kv_a_layernorm` is dead code in the forward pass. The chapter's statement that "these are applied to the LoRA latent intermediate representations" is correct for `q_a_layernorm` but factually wrong for `kv_a_layernorm`: that module is never invoked.


## Agent A Change Log — Pass 7

- item 1: `index.md` — corrected the B=1 tile-size characterization. Q at B=1 is 4,096 bytes = 2 bfloat16 tiles (32×32×2=2048 bytes each), not "far below one DRAM tile". K at B=1 is 1,024 bytes < 1 tile. Updated to "4,096 bytes (Q, 2 tiles) and 1,024 bytes (K, below one tile)".
- item 2: `distributed_alternative.md` — corrected the `kv_a_layernorm` module description. The `kv_a_layernorm` NormCls module object is constructed in `from_torch` but its `forward` is not called in the decode path; the KV latent normalization runs entirely via the inline `ttnn.rms_norm` at line 1689. Updated to accurately reflect that only `q_a_layernorm` is called as a module, and the inline `ttnn.rms_norm` is the functional parallel to `TTNNBailingMoEAttention`'s QK norm.

---

# B Review — Chapter 5: QK Normalization — Pass 8

No feedback — chapter approved.
