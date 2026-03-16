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
- If there are **Pending topics**: spawn one independent researcher agent per topic **in parallel**. Each agent runs the full guide generation process below (Phase 0 through Final Pass) for its assigned topic.

### Per-topic researcher agent

Each agent is responsible for exactly one topic. It:

1. Derives an output directory name from the topic: `<RESEARCH_OUTPUT_BASE_DIR>/<snake_case_topic_name>/`
2. **If the directory already exists**: the guide has been generated before. Skip generation. Go directly to step 4.
3. **If the directory does not exist**: run the full guide generation process below (starting at Phase 0) using the topic's `**Why Needed:**` and `**Questions:**` fields as the topic file inputs.
4. After the guide satisfies the completion condition, update `research_topics.md` for this topic:
   - Set `**Status:**` to `Completed`
   - Add a `**Guide:**` field pointing to the output directory

   Example:
   ```
   **Status:** Completed
   **Guide:** <RESEARCH_OUTPUT_BASE_DIR>/<snake_case_topic_name>/
   ```

5. Push immediately — do not wait for other agents:

```bash
cd /home/ttuser/salnahari/research-topics/tt-symbiote-research-topics
git add research_topics.md
git commit -m "research: completed topic <topic-name>"
git push
```

If push fails due to a concurrent push from another agent, rebase and retry:

```bash
git pull --rebase && git push
```

### Completion

All agents are done when each has pushed. Report back:
- How many topics were researched.
- Topic name, output directory, and one-line finding summary for each.

---

## Inputs

- **Topic file**: The topic's `**Why Needed:**` and `**Questions:**` fields from `research_topics.md`, treated as the guide scope and key concepts.
- **Output directory**: `<RESEARCH_OUTPUT_BASE_DIR>/<snake_case_topic_name>/`

Read both fields before doing anything else.

---

## Phase 0 — Initial Plan

Before any writing begins, generate an **initial plan** and write it to `<output_dir>/plan.md`.

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

### Agent A — Generator
- The **only agent that writes or modifies guide content**.
- Writes chapter files, `index.md`, and any cross-chapter reference material.
- When revising, applies feedback verbatim — does not silently skip or partially apply suggestions.
- After applying feedback, writes a brief **change log** note at the bottom of the relevant `compression_analysis.md` confirming what was done.

### Agent B — Critic
- **Reads** chapter files and evaluates them. Never modifies content files.
- Checks:
  1. **Correctness** — are all technical facts accurate? Flag any statement that is wrong or unverifiable.
  2. **Coherence** — does the text flow logically? Are concepts introduced before they are used?
  3. **Structural alignment** — does the chapter match the plan? Are all planned files present and covering the intended content?
  4. **Cross-chapter consistency** — do terminology, notation, and examples match across chapters (relevant once multiple chapters exist)?
- Writes feedback as a numbered list. Each item must state: the file, approximate line or section, the issue, and a concrete fix.
- When there are no issues, explicitly writes: **"No feedback — chapter approved."**

### Agent C — Compressor
- **Reads** chapter files and identifies redundancy and bloat. Never modifies content files.
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

## VERDICT
- Crucial updates: yes | no
```

- **CRUCIAL**: Significant redundancy (duplicate sections, restated tables, over-explained concepts already covered elsewhere). Agent A must address all CRUCIAL items.
- **MINOR**: Verbose phrasing, low-value elaboration. Agent A may address MINOR items at discretion.
- When `Crucial updates: no`, Agent C's pass is complete for that scope.
- On each subsequent pass, Agent C increments the pass number and only re-checks items from the previous pass that were flagged as CRUCIAL.

---

## Per-Chapter Loop

Process chapters in order: Chapter 1 first, last chapter last.

For each chapter:

```
repeat:
    Agent A writes (or revises) the chapter files per the plan (and any open feedback)
    repeat:
        Agent B reads the chapter and produces feedback
        if feedback is "No feedback — chapter approved":
            break
        Agent A applies all feedback
    Agent C reads the chapter and produces compression_analysis.md (or updates it)
    if compression_analysis VERDICT is "Crucial updates: no":
        break
    Agent A applies all CRUCIAL compression suggestions
until Agent B approves AND Agent C verdict is "Crucial updates: no"
```

A chapter is **done** only when both conditions hold simultaneously in the same iteration.

---

## Final Pass — Index and Cross-Chapter Coherence

After all chapters are complete:

1. **Agent A** writes `<output_dir>/index.md`. The index must contain:
   - A 1–2 sentence description of the guide (scope and audience).
   - A **"How to Use This Guide"** table: common reader goals mapped to recommended chapter paths and direct deep links.
   - A **Chapter Index** table: chapter number, title, one-line description, key operations or concepts.
   - A **Quick Reference** table: the most-used API calls or concepts, what each does, and where to learn more.
   - A **Prerequisites** section.
   - A **Source Code Location** section (if applicable).

2. Run the full A→B→C loop on the index and on all chapters together:
   - Agent B checks cross-chapter consistency (terminology, notation, forward/back references).
   - Agent C writes `<output_dir>/compression_analysis.md` covering cross-chapter redundancy (duplicate tables, concepts defined in multiple chapters verbatim, etc.).
   - Agent A applies all feedback and CRUCIAL compression suggestions.
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
