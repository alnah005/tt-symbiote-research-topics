# Guide Generation Prompt

This file instructs you to generate a structured, multi-chapter markdown guide for a given topic. Follow the process below exactly.

---

## Shared Research Topics — Trigger Behavior

This flow is **one-shot**. It is triggered manually; it does not poll or loop on its own.

### On startup

```bash
# First time: clone
git clone https://github.com/alnah005/tt-symbiote-research-topics.git /home/ttuser/salnahari/research-topics/tt-symbiote-research-topics

# Every time: pull latest
cd /home/ttuser/salnahari/research-topics/tt-symbiote-research-topics && git pull
```

Read `/home/ttuser/salnahari/research-topics/tt-symbiote-research-topics/research_topics.md` and collect all topics with `Status: Pending`.

- If there are **no Pending topics**: exit immediately. Nothing to do.
- If there are **Pending topics**: you (the orchestrator — the Claude Code instance reading this prompt) directly manage all topics. There are no coordinator sub-agents. You spawn Agent A, Agent B, and Agent C yourself, directly, for each topic.

### Orchestrator state

For each Pending topic, initialize and track the following state yourself:

```
topic_state = {
    topic_name,
    output_dir,          # <RESEARCH_OUTPUT_BASE_DIR>/<snake_case_topic_name>/
    skip,                # true if output_dir already exists
    current_chapter,     # index of chapter currently being processed (0-based)
    phase,               # "plan" | "write" | "review" | "compress" | "finalpass" | "done"
    pending_b_feedback,  # feedback from last Agent B invocation, or null
    pending_c_feedback,  # CRUCIAL suggestions from last Agent C invocation, or null
}
```

If `output_dir` already exists for a topic, set `skip = true` and advance it directly to the git-push step.

### Dispatch loop

At each iteration of the dispatch loop, you:

1. For every topic that is not `done` and not `skip`, determine which agent to spawn next based on that topic's current `phase`.
2. Spawn one agent per active topic **in parallel** (all as background agents in the same message). Each agent is a fresh Agent A, B, or C invocation — never reused across spawns.
3. Wait for all spawned agents to complete.
4. For each completed agent, read its result and advance that topic's state:
   - Agent A finishes writing → advance phase to `"review"`
   - Agent B returns "No feedback — chapter approved" → advance phase to `"compress"`
   - Agent B returns feedback → set `pending_b_feedback`, keep phase as `"review"`, re-spawn Agent A next
   - Agent C returns `Crucial updates: no` (validated) → if last chapter, advance to `"finalpass"`; else advance `current_chapter` and reset phase to `"write"`
   - Agent C returns `Crucial updates: yes` → set `pending_c_feedback`, advance phase back to `"write"` (re-spawn A with CRUCIAL suggestions), then B, then C again
5. Repeat until all topics are `done`.

### Per-topic context passing

You pass each agent only what it needs — no shared context between agents:
- Agent A receives: the plan, the chapter/files to write, and any `pending_b_feedback` or `pending_c_feedback` to apply.
- Agent B receives: the plan and the file paths of the chapter to review.
- Agent C receives: the chapter file paths and the current pass number.
- Never let B or C see A's reasoning. Never let A pre-read B's or C's prompts.

### Completion per topic

When a topic's guide satisfies the completion condition, you directly:

1. Update `research_topics.md` for that topic:
   - Set `**Status:**` to `Completed`
   - Add a `**Guide:**` field pointing to the output directory

   Example:
   ```
   **Status:** Completed
   **Guide:** <RESEARCH_OUTPUT_BASE_DIR>/<snake_case_topic_name>/
   ```

2. Push immediately — do not wait for other topics:

```bash
cd /home/ttuser/salnahari/research-topics/tt-symbiote-research-topics
git add research_topics.md
git commit -m "research: completed topic <topic-name>"
git push
```

If push fails due to a concurrent push, rebase and retry:

```bash
git pull --rebase && git push
```

### Parallelism rules

- **Across topics**: at each dispatch step, one agent per active topic runs in parallel. ✓
- **Within a topic, across chapters**: chapters are processed in order (Chapter 1 first). Do NOT parallelize chapters within a topic.
- **Within a chapter, across agents**: A → B → A(fix) → C → A(fix) is strictly sequential within each topic. The phase state enforces this.
- **Agent A, B, and C are always spawned by you (the orchestrator) directly.** No coordinator sub-agents exist.

### Completion

The dispatch loop ends when all topics are `done`. Report back:
- How many topics were researched.
- Topic name, output directory, and one-line finding summary for each.

---

## Inputs

- **Topic file**: The topic's `**Why Needed:**` and `**Questions:**` fields from `research_topics.md`, treated as the guide scope and key concepts.
- **Output directory**: `<RESEARCH_OUTPUT_BASE_DIR>/<snake_case_topic_name>/`

Read both fields before doing anything else.

---

## Phase 0 — Initial Plan

Before any writing begins, the orchestrator spawns **Agent A** to generate an **initial plan** and write it to `<output_dir>/plan.md`.

The plan must contain:

1. **Audience** — who is this guide for, what do they already know.
2. **Chapter list** — typically 5–8 chapters, ordered from foundational to advanced. For each chapter:
   - Chapter number and title
   - One-sentence description of what it covers
   - List of files inside the chapter directory (e.g., `index.md`, `topic_a.md`, `topic_b.md`)
   - Bullet points describing the content of each file
3. **Conventions** — terminology, notation, and formatting rules that must be consistent across all chapters.
4. **Cross-chapter dependencies** — which chapters reference concepts introduced in earlier chapters.

The plan is the source of truth for structure. Agents may propose amendments to the plan, but all amendments must be written back to `plan.md` before content is changed.

---

## Agents

**Critical rule**: Each agent invocation is a fresh sub-agent with no memory of prior invocations. The orchestrator (you) passes context explicitly via the prompt. Agents B and C have never seen the content they are reviewing — they approach it with zero attachment.

### Agent A — Generator
- The **only agent that writes or modifies guide content**.
- Writes chapter files, `index.md`, and any cross-chapter reference material.
- When revising, applies feedback verbatim — does not silently skip or partially apply suggestions.
- After applying feedback, writes a brief **change log** note at the bottom of the relevant `compression_analysis.md` confirming what was done.
- Receives from orchestrator: the plan, the list of files to write, and any feedback to apply.

### Agent B — Critic
- A **separate agent invocation** that has never seen this content before. It reads as a cold, independent reviewer.
- **Reads** chapter files and evaluates them. Never modifies content files.
- Checks:
  1. **Correctness** — are all technical facts accurate? Flag any statement that is wrong or unverifiable.
  2. **Coherence** — does the text flow logically? Are concepts introduced before they are used?
  3. **Structural alignment** — does the chapter match the plan? Are all planned files present and covering the intended content?
  4. **Cross-chapter consistency** — do terminology, notation, and examples match across chapters (relevant once multiple chapters exist)?
- Writes feedback as a numbered list. Each item must state: the file, approximate line or section, the issue, and a concrete fix.
- When there are no issues, explicitly writes: **"No feedback — chapter approved."**
- Receives from orchestrator: the plan, the list of chapter files to read (by path), and any prior feedback from previous passes.

### Agent C — Compressor
- A **separate agent invocation** that has never seen this content before. It reads as an adversarial editor whose sole job is to find and eliminate bloat.
- **Reads** chapter files and identifies redundancy and bloat. Never modifies content files.
- **Must find something to compress.** Agent C is not done until it has read every line and made an honest attempt to identify redundancy. A verdict of `Crucial updates: no` is only valid if Agent C has populated both a `## Load-Bearing Evidence` section (see format below) AND at least one `## MINOR Suggestions` item. A compression_analysis.md with an empty MINOR section and no Load-Bearing Evidence section will be **rejected by the orchestrator** and Agent C will be re-spawned.
- If a chapter truly has zero CRUCIAL bloat (uncommon), Agent C must:
  1. Populate `## Load-Bearing Evidence` with a bullet per file quoting a specific line or passage and explaining why it cannot be cut.
  2. Provide at least one MINOR suggestion — verbose phrasing, redundant adjectives, over-long code comments, restated variable names in prose. There is always something.
  3. Only then may VERDICT be `Crucial updates: no`.
- Writes its output to `<chapter_dir>/compression_analysis.md`, structured as:

```
# Compression Analysis: <Chapter Title> — Pass <N>

## Summary
- Total files analyzed: <N>
- Estimated current line count: ~<X> lines
- Estimated post-compression line count: ~<Y> lines
- Estimated reduction: ~<Z>%

## CRUCIAL Suggestions
### [<filename>] ~lines <range>
**Issue:** <description of redundancy or bloat>
**Suggestion:** <specific action to take>

## MINOR Suggestions
### [<filename>] ~lines <range>
**Issue:** ...
**Suggestion:** ...

## Load-Bearing Evidence
(Required when VERDICT is "Crucial updates: no". One bullet per file quoting a specific line/passage and explaining why it cannot be cut. Omitting this section invalidates the verdict.)
- `<filename>` line ~<N>: "<quoted text>" — load-bearing because <reason>

## VERDICT
- Crucial updates: yes | no
```

- **CRUCIAL**: Significant redundancy (duplicate sections, restated tables, over-explained concepts already covered elsewhere in the same or prior chapters). Agent A must address all CRUCIAL items.
- **MINOR**: Verbose phrasing, low-value elaboration, repeated code boilerplate, hedging language, redundant inline comments. Agent A may address MINOR items at discretion.
- When `Crucial updates: no`, Agent C's pass is complete for that scope — but only if the format rules above are satisfied.
- On each subsequent pass, Agent C increments the pass number and only re-checks items from the previous pass that were flagged as CRUCIAL.
- Receives from orchestrator: the list of chapter files to read (by path), the current pass number, and (on re-check passes) the prior compression_analysis.md.

**Orchestrator enforcement**: After receiving Agent C's output, the orchestrator must verify:
1. The `## Summary` section has non-zero line count estimates.
2. If VERDICT is `Crucial updates: no`: `## Load-Bearing Evidence` is present and non-empty, AND `## MINOR Suggestions` is non-empty.
3. If either check fails, re-spawn Agent C with the instruction: "Your previous compression_analysis.md was rejected. Reason: [specific missing section]. Re-read all files and produce a valid analysis."

---

## Per-Chapter Loop

Process chapters in order: Chapter 1 first, last chapter last.

For each chapter, the orchestrator runs the following loop by spawning agents sequentially:

```
repeat:
    [Orchestrator spawns Agent A]
        Agent A writes (or revises) the chapter files per the plan (and any open feedback)
    repeat:
        [Orchestrator spawns Agent B — fresh invocation, no shared context with A]
            Agent B reads the chapter and produces feedback
        if feedback is "No feedback — chapter approved":
            break
        [Orchestrator spawns Agent A — fresh invocation, passes B's feedback explicitly]
            Agent A applies all feedback
    [Orchestrator spawns Agent C — fresh invocation, no shared context with A or B]
        Agent C reads the chapter and produces compression_analysis.md (or updates it)
    if compression_analysis VERDICT is "Crucial updates: no":
        break
    [Orchestrator spawns Agent A — fresh invocation, passes C's CRUCIAL suggestions explicitly]
        Agent A applies all CRUCIAL compression suggestions
until Agent B approves AND Agent C verdict is "Crucial updates: no"
```

A chapter is **done** only when both conditions hold simultaneously in the same iteration.

**Before accepting Agent C's verdict**, the orchestrator must check:
- `## Summary` has non-zero line counts.
- If VERDICT is `Crucial updates: no`: `## Load-Bearing Evidence` is present and non-empty, AND `## MINOR Suggestions` has at least one item.
- If either check fails, re-spawn Agent C (directly, as orchestrator) with explicit rejection reason. Do not accept a `Crucial updates: no` verdict without this evidence.

---

## Final Pass — Index and Cross-Chapter Coherence

After all chapters are complete:

1. **Orchestrator spawns Agent A** to write `<output_dir>/index.md`. The index must contain:
   - A 1–2 sentence description of the guide (scope and audience).
   - A **"How to Use This Guide"** table: common reader goals mapped to recommended chapter paths and direct deep links.
   - A **Chapter Index** table: chapter number, title, one-line description, key operations or concepts.
   - A **Quick Reference** table: the most-used API calls or concepts, what each does, and where to learn more.
   - A **Prerequisites** section.
   - A **Source Code Location** section (if applicable).

2. Run the full A→B→C loop on the index and all chapters together:
   - **Orchestrator spawns Agent B** (fresh) to check cross-chapter consistency (terminology, notation, forward/back references).
   - **Orchestrator spawns Agent C** (fresh) to write `<output_dir>/compression_analysis.md` covering cross-chapter redundancy (duplicate tables, concepts defined in multiple chapters verbatim, etc.).
   - **Orchestrator spawns Agent A** (fresh) to apply all feedback and CRUCIAL compression suggestions.
   - Repeat until Agent B produces **"No feedback — guide approved."** and Agent C produces **"Crucial updates: no"** for both the cross-chapter analysis and each chapter's own analysis.

---

## Completion Condition

The process is complete when **all** of the following are true:

- Every chapter's `compression_analysis.md` ends with `Crucial updates: no`.
- `<output_dir>/compression_analysis.md` (cross-chapter) ends with `Crucial updates: no`.
- Agent B's last pass over the full guide produces **"No feedback — guide approved."**

At that point, report back to the user with:
- The output directory path.
- A summary of what was generated (chapter count, file count, rough line count).
- Any plan amendments that were made during the process and why.

---

## File Layout Convention

```
<output_dir>/
├── plan.md                        # Initial plan (updated if amended)
├── index.md                       # Top-level guide index (written in final pass)
├── compression_analysis.md        # Cross-chapter compression analysis
├── ch1_<title>/
│   ├── index.md
│   ├── <topic_a>.md
│   ├── <topic_b>.md
│   └── compression_analysis.md
├── ch2_<title>/
│   └── ...
└── ...
```

Chapter directory names use the format `ch<N>_<short_snake_case_title>`.

---

## Rules

1. Agent A is the only agent that writes files. Agents B and C only read and produce feedback.
2. Every CRUCIAL suggestion from Agent B or Agent C must be applied before moving forward.
3. The plan in `plan.md` is the authority on structure. If content contradicts the plan, Agent B flags it; Agent A fixes the content or proposes a plan amendment.
4. Do not move to the next chapter until the current chapter satisfies the completion condition.
5. Do not write the index until all chapters are done.
6. Compression analysis files are append-only across passes (each pass adds a new dated section). Never delete a previous pass's analysis.
7. Keep `index.md` and chapter `index.md` files as pure navigation — no content that belongs in a section file.
8. All cross-chapter references use relative markdown links.
9. **Agent B and Agent C are always separate agent invocations.** The orchestrator must never allow B or C to run in the same context as A. This isolation is what makes the review adversarial and trustworthy.
10. **Agent C must produce a Summary section with actual line count estimates.** A verdict of `Crucial updates: no` without a populated Summary section is invalid.
11. **A `Crucial updates: no` verdict requires both `## Load-Bearing Evidence` (non-empty, one bullet per file with a quoted line) and at least one `## MINOR Suggestions` item.** The orchestrator must reject and re-spawn Agent C if either is absent. There is no such thing as a chapter with zero MINOR issues.
12. **The orchestrator must never rubber-stamp Agent C's verdict.** It must explicitly verify the two conditions in rule 11 before accepting the verdict and advancing.
