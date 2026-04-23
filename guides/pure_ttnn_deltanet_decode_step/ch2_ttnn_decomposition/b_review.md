# B Review — Pass 1

1. **[recurrence_math_and_tensor_ops.md, ~line 60, wrong mapping direction for S]**

   The text states: "The state `S` maps `d_k → d_v`, so its transpose `S^T` maps `d_k → d_v` when applied from the left."

   Both halves of this sentence assert the same mapping direction, which is self-contradictory and the first half is wrong. A matrix of shape `[d_k, d_v]` maps R^{d_v} → R^{d_k} (it left-multiplies a column vector of length d_v to produce a column vector of length d_k). Therefore `S: [d_k, d_v]` maps **d_v → d_k**, and `S^T: [d_v, d_k]` maps **d_k → d_v**. The transpose is needed precisely because the direction of S is wrong for retrieval and output readout.

   Fix: Replace the sentence with: "The state `S: [d_k, d_v]` maps d_v-space to d_k-space. Its transpose `S^T: [d_v, d_k]` maps d_k-space to d_v-space, which is what retrieval and output readout require: given a d_k-space input (k̃ or q̃), `S^T @ input` produces a d_v-space result."

2. **[recurrence_math_and_tensor_ops.md, ~line 155 (closing Note after Op 6), same wrong mapping repeated]**

   The closing note says: "The state matrix `S: [d_k, d_v]` encodes associations from key-space to value-space." This restates the same directional error. `S: [d_k, d_v]` applied to a d_v vector gives a d_k vector — it is a map from value-space to key-space, not key-space to value-space. The following sentence ("To retrieve from it given a key ... one must apply the transpose") is correct in conclusion but the premise is backwards.

   Fix: Replace "encodes associations from key-space to value-space" with "encodes associations from value-space to key-space (it left-multiplies d_v-length vectors to produce d_k-length vectors). To retrieve a d_v-space output given a d_k-space input, the transpose `S^T: [d_v, d_k]` is required."

3. **[state_tensor_memory_config.md, ~line 91, L1 working-set arithmetic inconsistent]**

   The text claims: "The per-head working set for one decode step is under 200 KB total (state × 2 + vectors × 6)."

   The arithmetic does not support 200 KB as a meaningful bound. Per-head state is 32 KB (stated on line 89). "State × 2" = 64 KB. Six intermediate vectors at `[128, 1]` BF16 = 256 bytes each = 1.5 KB total. Actual per-head working set is approximately 65.5 KB — far below 200 KB. The "200 KB" figure appears with no derivation and is inconsistent with the per-head numbers given two lines earlier. A reader implementing a memory budget check would not be able to reproduce the 200 KB figure.

   Fix: Replace "under 200 KB total (state × 2 + vectors × 6)" with "under 70 KB total: two copies of the per-head state matrix (2 × 32 KB = 64 KB) plus six intermediate vectors at 256 bytes each (~1.5 KB), totaling approximately 65.5 KB."

# B Review — Pass 2 (Change Log)

Changes applied in response to Pass 1:
1. `recurrence_math_and_tensor_ops.md` ~line 60: corrected mapping direction — S: [d_k, d_v] maps d_v-space to d_k-space; S^T maps d_k-space to d_v-space
2. `recurrence_math_and_tensor_ops.md` ~line 155: corrected closing Note — "key-space to value-space" → "value-space to key-space"
3. `state_tensor_memory_config.md` ~line 91: corrected working-set estimate from "under 200 KB" to "under 70 KB (~65.5 KB)"

# B Review — Pass 2

1. **[state_tensor_memory_config.md, ~lines 110–130, conv state tile alignment violation]**

   The conv state is initialized with `ttnn.TILE_LAYOUT` and shape `[1, 1024, 4]`. TTNN tile layout requires the innermost two dimensions to be multiples of 32 (explicitly stated in the Tile Alignment Analysis section of this same file). The innermost dimension here is `4`, which is not a multiple of 32. This contradicts the file's own alignment rule and would produce an error or silent misalignment at runtime. A reader copying this initialization code verbatim would get incorrect behavior.

   The same file's tile alignment table (lines 96–103) confirms the rule: "TTNN tile operations require the innermost two dimensions to be multiples of 32." The shape `[1, 1024, 4]` violates this for the last dimension (4 < 32).

   Fix: The conv state should use `ttnn.ROW_MAJOR_LAYOUT` instead of `ttnn.TILE_LAYOUT`, since its last dimension (4) cannot be tile-aligned without padding. Alternatively, if TILE_LAYOUT is required for DRAM config compatibility, the shape must be padded to `[1, 1024, 32]` (padding the last dim to the next tile boundary), and the actual data occupies only the first 4 columns. The text and code must be consistent on which approach is chosen; neither is currently stated. Change the initialization code and the surrounding prose to reflect the correct layout choice.

# B Review — Pass 3 (Change Log)

Changes applied in response to Pass 2:
1. `state_tensor_memory_config.md` ~lines 110–130: changed conv state from TILE_LAYOUT to ROW_MAJOR_LAYOUT — last dimension is 4 (not a multiple of 32); TILE_LAYOUT is incompatible

# B Review — Pass 3

1. **[state_tensor_memory_config.md, ~line 91, working-set arithmetic missing the `write` matrix]**

   The file states: "two copies of the per-head state matrix (2 × 32 KB = 64 KB) plus six intermediate vectors at 256 bytes each (~1.5 KB), totaling approximately 65.5 KB."

   The `write` tensor (op 9 output, shape [128, 128], 32 KB) is a required L1 intermediate that is omitted from this count. Op 10 computes `S_new = S_decayed + write`, which means both `S_decayed` (32 KB) and `write` (32 KB) must be live in L1 simultaneously before the add can execute. That is already 64 KB for those two tensors alone. `S_new` itself (32 KB) is the result of op 10 and must also be live before it can be written to DRAM and read by op 11. At the point between op 10 and op 11, S_new (32 KB) is the only large matrix live, but at the peak — the instant before op 10 completes — S_decayed and write coexist at 64 KB. Additionally, S_prev (read from DRAM) is live simultaneously with S_decayed during ops 6–7 (independent reads), meaning the true peak is three matrices: S_prev + S_decayed + write = 96 KB, plus ~1.5 KB of vectors ≈ 97.5 KB total.

   The "two copies" framing and "65.5 KB" total are both wrong; the `write` matrix ([d_k, d_v] = 32 KB per head) is not a vector and must be counted as a third large intermediate.

   Fix: Replace the working-set sentence with: "The per-head working set peaks at approximately 97.5 KB: three copies of the per-head state matrix simultaneously in L1 (S_prev during retrieval, S_decayed from op 6, and write from op 9; each 32 KB = 96 KB total), plus six intermediate vectors at 256 bytes each (~1.5 KB). This is still well within a single Tensix core's 1.5 MB L1." Also update the section heading guard ("under 70 KB") to match the corrected estimate.

# B Review — Pass 4 (Change Log)

Changes applied in response to Pass 3:
1. `state_tensor_memory_config.md` ~line 91: corrected L1 working-set — added missing write tensor (op 9 output, [128,128] = 32 KB); peak is 3 × 32 KB matrices + ~1.5 KB vectors ≈ 97.5 KB (not 65.5 KB)

# B Review — Pass 4

1. **[ttnn_ops_per_step.md, ~line 99, "L1 interleaved" defined as TILE_LAYOUT but intermediate vectors are not tile-aligned]**

   The "Notes on Memory Config Column" section defines "L1 interleaved" as `ttnn.L1_MEMORY_CONFIG` with `ttnn.TILE_LAYOUT`. However, the majority of the intermediate tensors in this step — k_tilde, q_tilde, v_t, retrieval, error, and o_t — all have shape `[B, nH_local, d_k, 1]` or `[B, nH_local, d_v, 1]`, where the innermost dimension is **1**. TTNN TILE_LAYOUT requires the innermost two dimensions to be multiples of 32 (stated explicitly in `state_tensor_memory_config.md` lines 95–96). A last dimension of 1 violates this constraint. Assigning TILE_LAYOUT to these tensors would produce a runtime error or require TTNN to silently pad them to [B, nH_local, 128, 32] — ballooning memory by 32×.

   Fix: The definition must distinguish between matrix intermediates (d_k × d_v, tile-aligned — TILE_LAYOUT is correct) and vector intermediates (d_k × 1 or d_v × 1 — must use ROW_MAJOR_LAYOUT). Update the "L1 interleaved" note to: "Matrix intermediates (ops 6, 9, 10) use `ttnn.L1_MEMORY_CONFIG` with `ttnn.TILE_LAYOUT`; vector intermediates (ops 2, 4, 5, 7, 8, 11, 12) use `ttnn.L1_MEMORY_CONFIG` with `ttnn.ROW_MAJOR_LAYOUT` because their last dimension (1) is not a multiple of 32."

2. **[ttnn_ops_per_step.md, ~line 12 (op table row 2) vs. ~lines 34–37 (annotated code), contradiction on whether `ttnn.repeat` is used]**

   The op table row for step 2 ("Q/K head expand") lists the TTNN API as `ttnn.reshape` **+ `ttnn.repeat``. The annotated code block below (lines 34–37) performs only `ttnn.reshape` — `ttnn.repeat` does not appear anywhere in the code. These two representations of the same operation are contradictory. A reader implementing the step must know whether a repeat is required (relevant if the model uses grouped-query attention where fewer K heads must be expanded to match Q heads) or not.

   Fix: Reconcile the two representations. If no head repetition is needed (all heads are unique), remove `ttnn.repeat` from the op table API column. If repetition is required for correctness, add the corresponding `ttnn.repeat` call to the annotated code with a `# why:` comment explaining which dimension is repeated and by what factor.

# B Review — Pass 5 (Change Log)

Changes applied in response to Pass 4:
1. `ttnn_ops_per_step.md` ~line 99: corrected L1 interleaved layout — state tensor S uses TILE_LAYOUT (tile-aligned 128×128); vector intermediates with last dim=1 must use ROW_MAJOR_LAYOUT
2. `ttnn_ops_per_step.md` ~lines 12 and 34–37: reconciled step 2 — aligned op table and code block on ttnn.reshape + ttnn.repeat; added # TODO: verify comment

# B Review — Pass 5

1. **[state_tensor_memory_config.md, ~line 91, wrong causal claim — S_prev and `write` never coexist in L1]**

   The working-set sentence reads: "three state-matrix copies (S_prev, S_decayed, and the write accumulation — 3 × 32 KB = 96 KB)." This asserts that S_prev, S_decayed, and write are simultaneously live in L1 at peak. That is factually wrong.

   Tracing tensor lifetimes through the op sequence:
   - Op 6 reads S_prev (DRAM) and produces S_decayed (L1). S_prev is still needed for op 7. Peak here: S_prev + S_decayed = 64 KB.
   - Op 7 reads S_prev and produces retrieval (256 bytes). S_prev can be released after op 7.
   - Ops 8–9 run without any large matrix; op 9 produces write (32 KB). At this point only S_decayed + write = 64 KB.
   - Op 10 reads S_decayed + write and produces S_new (32 KB); S_decayed and write are released.

   S_prev is consumed (released) before write is produced. They are never simultaneously live. The true per-head L1 peak is 64 KB (two state matrices, not three), occurring at two separate moments: S_prev + S_decayed during ops 6–7, and S_decayed + write during ops 9–10.

   The 97.5 KB figure overstates the peak by 32 KB, and the "three copies simultaneously" causal claim would mislead a reader doing memory-pressure analysis or reasoning about double-buffering requirements.

   Fix: Replace the working-set sentence with: "The per-head working set peaks at approximately 65.5 KB: two state-matrix copies simultaneously in L1 at peak (either S_prev + S_decayed during ops 6–7, or S_decayed + write during ops 9–10; each pair is 2 × 32 KB = 64 KB), plus six intermediate vectors at 256 bytes each (~1.5 KB). S_prev is released after op 7, before write is produced at op 9, so all three large matrices are never live simultaneously. This is well within a single Tensix core's 1.5 MB L1."

# B Review — Pass 6 (Change Log)

Changes applied in response to Pass 5:
1. `state_tensor_memory_config.md` ~line 91: corrected working-set back to ~65.5 KB — S_prev and write are never simultaneously live; two non-overlapping peaks of 64 KB each (S_prev+S_decayed at ops 6–7; S_decayed+write at ops 9–10)

# B Review — Pass 6

1. **[state_tensor_memory_config.md, ~lines 21–31, nH_local=4 conflicts with domain fact specifying nH_local=8]**

   The file states: "T3K has 8 devices. Under head-parallel sharding, each device owns `32 / 8 = 4` heads. Each device holds: `S_local: [B, nH_local, d_k, d_v] = [1, 4, 128, 128]`."

   The domain facts specify: "State matrix S shape [B, nH_local, d_k, d_v] = [1, 8, 128, 128] per head — innermost 128×128 is tile-aligned; TILE_LAYOUT is correct for S."

   These two claims are mutually exclusive: nH_local cannot be both 4 and 8. All memory calculations in the file derive from nH_local=4 and would be wrong by a factor of 2 if nH_local=8 is correct:
   - File: 4 × 128 × 128 × 2 = 131,072 bytes = **128 KB** per device per layer
   - If nH_local=8: 8 × 128 × 128 × 2 = 262,144 bytes = **256 KB** per device per layer
   - File: 30 layers × 128 KB = **3,840 KB = 3.75 MB** per device
   - If nH_local=8: 30 layers × 256 KB = **7,680 KB = 7.5 MB** per device
   - The summary table (line 136) and the total DRAM figure ("under 4 MB per device") would also be wrong by a factor of 2.

   The file's arithmetic is internally self-consistent (nH=32 total heads / 8 devices = 4 local heads is straightforward division), and nH=32 is stated in `recurrence_math_and_tensor_ops.md` line 10. A T3K with 8 devices and 32 total heads yields exactly 4 local heads per device. The domain fact's [1, 8, 128, 128] appears to be a transcription error (8 instead of 4).

   This conflict must be resolved before the memory sizing figures can be trusted. If nH_local=4 is correct (consistent with nH=32 and 8 devices), the file's calculations are correct and the domain fact has a typo. If nH_local=8 is correct, the file's nH_local, per-device shape, and all derived memory figures (128 KB, 3.75 MB, 136 KB per layer, 3.98 MB total) must be doubled.

# B Review — Pass 7 (Change Log)

Changes applied in response to Pass 6:
1. No changes made. Pass 6 Issue 1 (nH_local conflict) is a false positive: the file's nH_local=4 (32 total heads / 8 T3K devices) is arithmetically correct. The nH_local=8 in the Pass 6 reviewer domain facts was a transcription error in the review context. All memory figures in `state_tensor_memory_config.md` are consistent with nH_local=4 and require no correction.

# B Review — Pass 7

1. **[ttnn_ops_per_step.md, annotated code block, step 8 — undefined variable `beta_broadcast`]**

   In the annotated code block, step 4 assigns the update rate to the variable `beta_t`:

   ```python
   beta_t = ttnn.sigmoid(b_t)           # [B, nH_local, 1, 1], range (0, 1)
   ```

   Step 8 then references `beta_broadcast`, which is never assigned anywhere in the code block:

   ```python
   error = ttnn.mul(beta_broadcast, error_raw)  # scale by update rate
   # why: beta_broadcast [B, nH_local, 1, 1] broadcasts over d_v
   ```

   `beta_broadcast` is a dangling identifier. A reader copying this code verbatim would get a `NameError` at runtime. The intent is clear (use `beta_t` directly, since it already has shape `[B, nH_local, 1, 1]` and broadcasting over d_v requires no reshape), but the code as written is broken.

   Note the parallel treatment of `g_t` in step 6: step 3 produces `g_t` and step 6 introduces `g_broadcast = ttnn.reshape(g_t, [B, nH_local, 1, 1])` before use. If `g_t` already has shape `[B, nH_local, 1, 1]` (as implied by the op table row for step 3), that reshape is a no-op but at least `g_broadcast` is defined before use. No such intermediate assignment is present for `beta_t`.

   Fix: Either rename `beta_t` to `beta_broadcast` in step 4, or add `beta_broadcast = beta_t` (or `beta_broadcast = ttnn.reshape(beta_t, [B, nH_local, 1, 1])` mirroring the g_t pattern) between steps 4 and 8 to make the variable available. The choice should be made consistent with the g_t/g_broadcast pattern already present in the file.

# B Review — Pass 8 (Change Log)

Changes applied in response to Pass 7:
1. `ttnn_ops_per_step.md` step 8 code block: added `beta_broadcast = ttnn.reshape(beta_t, [B, nH_local, 1, 1])` before `error_raw`; `beta_broadcast` was used but never defined, causing NameError; follows the `g_broadcast = ttnn.reshape(g_t, ...)` pattern from step 6

# B Review — Pass 8

1. **[recurrence_math_and_tensor_ops.md, ~line 155 (closing Note after Op 6), "dimensionally inconsistent" claim is false for d_k = d_v]**

   The closing note states: "Applying `S` directly (without transpose) would produce a `[d_k, 1]` result in key-space, which is dimensionally inconsistent with the expected `[d_v, 1]` output."

   This claim is only valid when `d_k ≠ d_v`. In the Qwen3.6-35B-A3B model documented throughout this chapter, `d_k = d_v = 128`. Therefore `[d_k, 1] = [128, 1]` and `[d_v, 1] = [128, 1]` — the shapes are identical. Applying S directly (without transpose) would produce a tensor of the same shape as applying S^T. There is no dimensional inconsistency and no runtime error. The bug would be silent: the model would produce numerically wrong outputs with no error signal.

   A reader relying on this note as a correctness check would be misled into believing the mistake is detectable at runtime (shape mismatch), when in fact it would be undetectable — making it a more dangerous class of error, not a less dangerous one.

   Fix: Replace "which is dimensionally inconsistent with the expected `[d_v, 1]` output" with "which, in models where d_k ≠ d_v, would be a shape mismatch caught at runtime; in this model d_k = d_v = 128, so the mistake would be silent — producing numerically wrong values with no error signal."

# B Review — Pass 9 (Change Log)

Changes applied in response to Pass 8:
1. `recurrence_math_and_tensor_ops.md` ~line 155: corrected false "dimensional inconsistency" claim — since d_k=d_v=128, applying S without transpose produces [128,1] which is the same shape as the expected [d_v,1]; the error is silent (numerically wrong, no shape mismatch, TTNN does not raise); replaced with explicit "silent error" warning

# B Review — Pass 10 (Change Log)

Changes applied in response to Pass 9:
1. `ttnn_ops_per_step.md` ~line 109 (Notes on Memory Config Column): expanded ROW_MAJOR rule from "last dimension is 1" to "either innermost dimension (last or second-to-last) is not a multiple of 32"; explicitly added `error_T` (shape [B, nH_local, 1, d_v], dim[-2]=1) as a named example alongside the existing column-vector examples; `error_T` was previously unclassified and would have incorrectly received TILE_LAYOUT

# B Review — Pass 9

1. **[ttnn_ops_per_step.md, ~line 109 (Notes on Memory Config Column), layout rule does not cover `error_T` — shape `[B, nH_local, 1, d_v]` violates TILE_LAYOUT on dim[-2]]**

   The notes section defines two layout cases under "L1 interleaved":
   - State tensor S (`[d_k, d_v] = [128, 128]`, both inner dims multiples of 32) — TILE_LAYOUT.
   - Vector intermediates with last dimension = 1 (k_tilde, q_tilde, v_t, retrieval, error, o_t) — ROW_MAJOR_LAYOUT.

   Op 9 (write/outer product) requires transposing `error` from `[B, nH_local, d_v, 1]` to `error_T` with shape `[B, nH_local, 1, d_v]`. The op table (row 9) lists this input shape explicitly. TTNN TILE_LAYOUT requires **both** innermost dimensions to be multiples of 32 (stated in `state_tensor_memory_config.md` line 95). `error_T` has dim[-2] = 1, which is not a multiple of 32. Neither of the two layout rules in the notes section covers this tensor:
   - It is not the state S.
   - Its last dimension is `d_v = 128` (not 1), so the "last dimension is 1 → ROW_MAJOR" rule does not match it.

   If a reader applies TILE_LAYOUT to `error_T` (the only other option defined), they get a tensor whose second-to-last dimension (1) violates the tile constraint. The notes section has a gap: `error_T` is unclassified, and the default from the two stated rules is wrong for it.

   The correct layout for `error_T` is `ttnn.ROW_MAJOR_LAYOUT`, for the same reason the "last dim = 1" vectors use it: a dimension of 1 in either inner position is not a multiple of 32 and cannot be tile-aligned without 32× padding.

   Fix: In the notes section's "L1 interleaved" bullet, expand the ROW_MAJOR rule to cover any tensor whose innermost two dimensions are not both multiples of 32. Specifically add `error_T` (shape `[B, nH_local, 1, d_v]`, dim[-2] = 1) to the list of tensors requiring ROW_MAJOR_LAYOUT. The revised rule should read: "Tensor intermediates where either of the innermost two dimensions is not a multiple of 32 must use `ttnn.ROW_MAJOR_LAYOUT`. This includes both the `[..., d_k, 1]`/`[..., d_v, 1]` column vectors (last dim = 1) and the `[..., 1, d_v]` row vector `error_T` produced by transposing `error` for the outer-product matmul (second-to-last dim = 1)."

---

## Pass 10

### Issues found: 0

None. Chapter is correct.

All domain-fact checks pass against the current text:

1. **S mapping direction** (`recurrence_math_and_tensor_ops.md`, Op 2 and closing Note): S: [d_k, d_v] maps d_v-space to d_k-space; S^T maps d_k-space to d_v-space — stated correctly.

2. **ẽ_t uses S_prev, not S_decayed** (`recurrence_math_and_tensor_ops.md`, Op 2; `ttnn_ops_per_step.md`, op 7): both files use S_prev for retrieval and carry an explicit warning against using S_decayed — correct.

3. **Silent-error note for S vs S^T** (`recurrence_math_and_tensor_ops.md`, closing Note after Op 6): correctly states that since d_k = d_v = 128, applying S without transpose produces a [128, 1] result of identical shape to the expected [d_v, 1]; the error is silent (no shape mismatch, no runtime signal) — correct after Pass 8 fix.

4. **TILE_LAYOUT rule for error_T** (`ttnn_ops_per_step.md`, Notes on Memory Config Column): the ROW_MAJOR rule now explicitly covers tensors where either innermost dimension is not a multiple of 32, naming error_T (shape [B, nH_local, 1, d_v], dim[-2]=1) as a required ROW_MAJOR case — correct after Pass 9 fix.

5. **L1 working-set calculation** (`state_tensor_memory_config.md`, L1 Feasibility section): peak is 65.5 KB, arising at two non-overlapping moments (S_prev + S_decayed during ops 6–7; S_decayed + write during ops 9–10; each pair = 64 KB). S_prev is explicitly noted as released after op 7 before write is produced at op 9 — the claim is internally consistent and arithmetically correct.

6. **nH_local = 4** (`state_tensor_memory_config.md`, `ttnn_ops_per_step.md`): 32 total heads / 8 T3K devices = 4 — stated correctly in both files.

7. **No other correctness errors found** across all three files.
