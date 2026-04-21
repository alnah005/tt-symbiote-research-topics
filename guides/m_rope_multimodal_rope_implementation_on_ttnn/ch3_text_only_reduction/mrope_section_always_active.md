# M-RoPE Section Split: Always Structurally Active, Values-Dependent on Position IDs

## 1. The Misconception

Reading `config.rope_scaling.type = "mrope"` alongside `config.rope_scaling.mrope_section = [11, 11, 10]`, one might assume the model has two modes of operation: an "M-RoPE mode" activated for vision inputs and a "standard RoPE mode" activated for text inputs, with a config flag or input check selecting between them. This is incorrect.

There is no mode switch. The `Qwen2_5_VLRotaryEmbedding` class always performs the three-gather + duplication construction on every forward call, regardless of input type.

## 2. What "Always Active" Means

In HuggingFace, `Qwen2_5_VLRotaryEmbedding.forward()` executes the following unconditionally:

```python
cos = torch.cat([
    cos_table[position_ids[0], :s_t],
    cos_table[position_ids[1], s_t:s_t+s_h],
    cos_table[position_ids[2], s_t+s_h:],
], dim=-1)
cos = torch.cat([cos, cos], dim=-1)
```

No branch inspects whether `position_ids[0] == position_ids[1] == position_ids[2]` before deciding how to compute. The section split is structurally permanent in the forward pass — it cannot be disabled by config.

However, the *output values* depend entirely on what the position ID tensor contains:

- When all three rows are identical (text-only), the output equals standard 1D RoPE values (proved in `mathematical_equivalence_proof.md`). The code path does not change; only the numbers happen to match.
- When the rows differ (vision input), the output encodes independent spatial coordinates per section — this is the designed M-RoPE behavior.

The section split is a structural property of the computation graph. Whether it has any observable effect on the output is a property of the data.

## 3. Implication for TTNN

At the TTNN level, the decision between the two paths does not follow from any config flag. It follows from the content of the position ID tensor:

- If position IDs are always the same 1D tensor broadcast across all three axes (text-only batch), a standard 1D cos/sin table lookup is sufficient. The three-gather construction is computationally equivalent and can be skipped.
- If position IDs differ across axes (vision input), the three-gather construction must be used. Falling back to the standard path would silently encode the wrong positions for vision tokens.

| Input type | `position_ids[0] == [1] == [2]`? | TTNN path |
|---|---|---|
| Text-only | Yes | Standard 1D table lookup |
| Image input | No | Three-gather M-RoPE construction |
| Video input | No | Three-gather M-RoPE construction |

The routing logic in a TTNN implementation should therefore inspect whether vision tokens are present in the batch — not query any config flag — to decide which path to take.

## 4. Why This Matters for Testing

A test that exercises only text-only inference will pass even if the M-RoPE three-gather construction is completely broken, because the standard path produces numerically correct output for that input type. The section split has no observable effect when all three rows of `position_ids` are identical.

Vision-input tests are required to validate the three-gather construction. Specifically:

- A test with a single image patch (temporal != height != width coordinates) exercises the three-gather path with divergent position IDs.
- A test that asserts numerical identity against the HuggingFace reference for a mixed text+image batch confirms the gather indices are correct.
- A text-only test that passes is insufficient evidence that M-RoPE is correctly implemented.

Chapter 4 specifies the required test cases for each path, and Chapter 6 `correctness_validation.md` defines the concrete test matrix including both text-only and vision-input cases.

---
**Next:** [Chapter 4 — M-RoPE TTNN Implementation Strategy](../ch4_ttnn_implementation/index.md)
