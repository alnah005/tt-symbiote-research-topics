# Thinking Preservation — Mechanism, Cost, and Implementation Impact

## What Thinking Preservation Is

Chain-of-thought models like Qwen3.6 emit explicit reasoning traces — sequences of tokens representing intermediate reasoning steps — before producing a final answer. In a single-turn exchange this is straightforward: the model reasons, then answers.

In a multi-turn conversation the question is what to do with those reasoning traces from prior turns. The default behavior in most inference systems is to discard or truncate the prior-turn reasoning: only the final answers and user messages are kept in the context window, since reasoning traces can be thousands of tokens long. This reduces context consumption but means the model cannot reference its own prior reasoning when answering follow-up questions.

**Thinking Preservation** is the practice of retaining the model's reasoning traces from prior conversation turns in the active context window, so that the model has access to its earlier chain-of-thought when generating subsequent responses.

## Mechanical Operation

### Standard Multi-Turn Inference (Without Thinking Preservation)

In a standard multi-turn conversation the context window at turn $n$ contains:

- System prompt
- User message (turn 1)
- Final answer (turn 1) — reasoning trace stripped
- User message (turn 2)
- Final answer (turn 2) — reasoning trace stripped
- ...
- User message (turn $n$)

The reasoning traces $r_1, r_2, \ldots, r_{n-1}$ are discarded before they enter the context for the next turn. This keeps the context compact at the cost of the model losing access to its own prior reasoning.

### With Thinking Preservation

With Thinking Preservation enabled, the context window at turn $n$ contains:

- System prompt
- User message (turn 1)
- Reasoning trace (turn 1) — **retained**
- Final answer (turn 1)
- User message (turn 2)
- Reasoning trace (turn 2) — **retained**
- Final answer (turn 2)
- ...
- User message (turn $n$)

The model attends over the preserved reasoning traces when generating the response to turn $n$. It can reference its prior chain-of-thought, which improves consistency and depth of reasoning across turns — particularly for tasks where earlier reasoning steps are prerequisite to answering a follow-up question correctly.

## Implementation: Prompting Layer Only

Thinking Preservation is implemented entirely in the conversation template and context management layer:

1. The serving layer constructs the prompt by including reasoning trace tokens from prior turns alongside final answers and user messages.
2. The assembled token sequence is passed to the model's forward pass as a single flat input — no special handling.
3. The model processes every token in the sequence identically through the decoder stack.

There are no changes to:
- Model architecture or layer structure
- Forward pass computation
- Attention patterns or masking logic
- Weight values or fine-tuning
- Any TTNN kernel or operation

All implementation work is in the application and serving layer:
- Conversation template construction: include or exclude prior reasoning traces based on configuration
- Context length tracking: monitor total token count and apply management strategies before the 262K limit is reached
- Paged KV cache sizing: allocate sufficient pages for Gated Attention layers given the expected token count with preserved reasoning

## KV Cache Implications

Qwen3.6 uses a hybrid architecture (30 Gated DeltaNet + 10 Gated Attention layers; see [Chapter 1](../ch1_architecture_overview/index.md)) whose two layer types respond very differently to increased token counts from preserved reasoning.

### Gated DeltaNet Layers (30 of 40 layers)

Gated DeltaNet is a recurrent architecture. Each head maintains a fixed-size state matrix $S \in \mathbb{R}^{128 \times 128}$. This state is updated token-by-token as the sequence is processed, but its size does not change with sequence length.

Consequence: **Thinking Preservation has zero memory impact on the 30 Gated DeltaNet layers.** Whether the input sequence is 1K tokens or 100K tokens — with or without preserved reasoning traces — the recurrent state size per head is constant.

### Gated Attention Layers (10 of 40 layers)

Gated Attention uses a paged KV cache. For a sequence of $T$ total tokens, each Gated Attention layer stores key and value tensors with shape `[B, T, num_key_value_heads, head_dim]` where `num_key_value_heads = 2` (GQA). Memory consumption grows linearly with $T$.

When Thinking Preservation is active, $T$ includes reasoning trace tokens from all prior turns in addition to user messages and final answers. Reasoning traces from a single turn can span thousands of tokens. Across multiple turns the cumulative reasoning token count can be substantial.

Consequence: **The KV cache for the 10 Gated Attention layers is the primary memory cost of Thinking Preservation.** At long conversations with preserved thinking, these layers become the memory bottleneck. The 30 Gated DeltaNet layers are unaffected.

The total KV cache size scales as:

$$\text{KV memory} \propto N_{\text{attn}} \times B \times T_{\text{total}} \times n_{kv} \times d_h \times 2$$

where $N_{\text{attn}} = 10$ is the number of Gated Attention layers, $B$ is batch size, $T_{\text{total}}$ is the total token count including preserved reasoning, $n_{kv} = 2$ is the number of key-value heads (GQA), $d_h = 256$ is the head dimension, and the factor of 2 accounts for keys and values.

## Interaction with the 262K Context Window

Qwen3.6 supports a maximum context length of 262,144 tokens (`max_position_embeddings = 262144`). This is the hard ceiling on $T_{\text{total}}$.

Without Thinking Preservation, a conversation turn adds roughly:

$$T_{\text{turn}} \approx T_{\text{user}} + T_{\text{answer}}$$

With Thinking Preservation, each turn also contributes its reasoning trace:

$$T_{\text{turn}} \approx T_{\text{user}} + T_{\text{reasoning}} + T_{\text{answer}}$$

Since $T_{\text{reasoning}}$ can be orders of magnitude larger than $T_{\text{answer}}$ for complex reasoning tasks, the context window fills significantly faster. The number of conversation turns before hitting the 262K limit is reduced by a factor roughly proportional to the average reasoning trace length relative to the answer length.

For very long conversations, context management strategies become necessary:

- **Selective preservation**: retain reasoning traces only from the most recent $k$ turns
- **Summarization**: compress old reasoning traces into a shorter summary
- **Sliding window**: evict oldest turns (with their reasoning) once the context approaches the limit

These strategies are all implemented at the serving/application layer. The model itself has no awareness of which management strategy is in use.

---

**Next:** [Chapter 7 — MoE Architecture and Cross-Model Comparison](../ch7_moe_comparison/index.md)
