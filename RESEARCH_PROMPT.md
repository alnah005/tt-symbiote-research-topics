# Guide Generation Prompt

Generate a structured, multi-chapter markdown guide for each pending research topic.

---

## Startup

```bash
cd tt-symbiote-research-topics && git pull
```

Read `research_topics.md`. Collect all topics with `Status: Pending`.

- **None pending** → exit.
- **Pending** → process each topic directly. No coordinator sub-agents.

---

## State

Per topic, the orchestrator tracks:

```
topic_state = {
    topic_name,
    output_dir,          # <RESEARCH_OUTPUT_BASE_DIR>/<snake_case_topic_name>/
    skip,                # true if output_dir exists → jump to git-push
    current_chapter,     # 0-based index
    phase,               # plan | write | review | compress | finalpass | done
    pending_b_feedback,  # from Agent B, or null
    pending_c_feedback,  # CRUCIAL items from Agent C, or null
}
```

---

## Inputs

Per topic, read from `research_topics.md`:

- **`Why Needed:`** — guide scope
- **`Questions:`** — key concepts to cover

---

## Agents

Every invocation is a **fresh sub-agent** with zero memory. The orchestrator passes all context explicitly. Agents never share context.

### Agent A — Generator

The **only agent that writes or modifies files.**

- Writes chapter files, `index.md`, cross-chapter references.
- Applies feedback verbatim — never skips or partially applies.
- Appends a change log to the relevant `compression_analysis.md` after applying feedback.
- **Receives:** plan, target files, `pending_b_feedback`, `pending_c_feedback`.

### Agent B — Critic

**Read-only.** Writes to `<chapter_dir>/b_review.md` (append with pass number heading).

Flags **only** these categories:

| Category | Flag when... |
|---|---|
| Factual correctness | A fact, formula, or derivation would produce incorrect code or results |
| Critical coherence | A concept is used before it is defined, blocking comprehension |
| Critical structural gaps | A planned file is missing, or a file omits content a later chapter depends on |

**Out of scope:** style, formatting, wordiness, abbreviations — anything a reader can resolve with basic context.

- **Max 5 items per pass.** Each item: file, ~line, the error, a concrete fix.
- Zero issues → write exactly: **"No feedback — chapter approved."**
- **Receives:** plan, chapter file paths, prior feedback.

### Agent C — Compressor

**Read-only.** Writes to `<chapter_dir>/compression_analysis.md`.

Flags **only:** duplicate explanations, restated tables, verbose prose, over-long code comments, repeated examples, hedging language. Does NOT flag factual errors or missing content (Agent B's job).

**Output format:**

```
# Compression Analysis: <Chapter Title> — Pass <N>

## Summary
- Total files analyzed: <N>
- Estimated current line count: ~<X>
- Estimated post-compression line count: ~<Y>
- Estimated reduction: ~<Z>%

## CRUCIAL Suggestions
### [<filename>] ~lines <range>
**Issue:** <redundancy or bloat>
**Suggestion:** <specific action>

## MINOR Suggestions
### [<filename>] ~lines <range>
**Issue:** ...
**Suggestion:** ...

## Load-Bearing Evidence
- `<filename>` line ~<N>: "<quoted text>" — load-bearing because <reason>

## VERDICT
- Crucial updates: yes | no
```

- **CRUCIAL** = significant redundancy Agent A must fix. **MINOR** = verbose phrasing Agent A may fix.
- Re-check passes: increment pass number, only re-check prior CRUCIAL items.
- **Receives:** chapter file paths, pass number, prior `compression_analysis.md` (on re-checks).

### Agent C Verdict Validation

The orchestrator **must** enforce this. A `Crucial updates: no` verdict is valid **only if all three hold:**

1. `## Summary` has non-zero line count estimates.
2. `## Load-Bearing Evidence` is present and non-empty (one bullet per file with a quoted line).
3. `## MINOR Suggestions` has at least one item.

Any check fails → re-spawn Agent C: *"Rejected. Reason: [missing section]. Re-read all files and produce a valid analysis."*

---

## Phase 0 — Plan

Spawn Agent A to write `<output_dir>/plan.md`:

1. **Audience** — who they are, what they already know.
2. **Chapter list** (1–8 chapters, foundational → advanced). Per chapter:
   - Number, title, one-sentence scope
   - Files in the chapter directory (`index.md`, `topic_a.md`, ...)
   - Bullet points per file
3. **Conventions** — terminology, notation, formatting rules.
4. **Cross-chapter dependencies** — which chapters reference earlier concepts.

`plan.md` is the structural source of truth. Amend it before making content changes that deviate from it.

---

## Per-Chapter Loop

Chapters are processed **in order**. For each chapter:

```
loop:
    Agent A writes/revises chapter files (applying any open feedback)

    loop:
        Agent B reviews → feedback or "No feedback — chapter approved"
        if approved: break
        Agent A applies B's feedback
    end

    Agent C analyzes → compression_analysis.md
    Orchestrator validates verdict
    if VERDICT is "Crucial updates: no": break
    Agent A applies all CRUCIAL suggestions
end
```

A chapter is **done** when both hold in the same iteration:
- Agent B: approved
- Agent C: validated `Crucial updates: no`

---

## Multi-Topic Dispatch

Topics run independently. The dispatch loop is **event-driven per topic**, not wave-synchronized.

| Event | Action |
|---|---|
| Agent A finishes writing | → spawn Agent B |
| Agent B approves | → spawn Agent C |
| Agent B returns feedback | → set `pending_b_feedback`, spawn Agent A |
| Agent C verdict `no` (validated), not last chapter | → advance `current_chapter`, spawn Agent A |
| Agent C verdict `no` (validated), last chapter | → advance to `finalpass` |
| Agent C verdict `yes` | → set `pending_c_feedback`, spawn Agent A |

One agent per topic at a time. Spawn the next immediately when the previous completes.

Print state on every dispatch:

```
**State:**
<topic>: phase=<phase>, chapter=<N> → <Agent X> running
```

---

## Final Pass — Index & Cross-Chapter Coherence

After all chapters complete:

1. **Agent A** writes `<output_dir>/index.md`:
   - 1–2 sentence guide description (scope + audience)
   - **How to Use This Guide** table: reader goals → chapter paths with deep links
   - **Chapter Index** table: number, title, description, key concepts — every entry links to `<chapter>/index.md`
   - **Quick Reference** table: most-used API calls/concepts, what each does, where to learn more
   - **Prerequisites** section
   - **Source Code Location** section (if applicable)

2. Run A→B→C loop on index + all chapters together:
   - Agent B checks cross-chapter consistency (terminology, notation, references)
   - Agent C writes `<output_dir>/compression_analysis.md` for cross-chapter redundancy
   - Agent A applies all feedback and CRUCIAL suggestions
   - Repeat until B produces **"No feedback — guide approved."** and C produces validated **"Crucial updates: no"**

---

## Completion

Done when **all three hold:**

1. Every chapter's `compression_analysis.md` → validated `Crucial updates: no`
2. Cross-chapter `compression_analysis.md` → validated `Crucial updates: no`
3. Agent B's last full-guide pass → **"No feedback — guide approved."**

### Per-topic completion

1. Update `research_topics.md`:
   ```
   **Status:** Completed
   **Guide:** <RESEARCH_OUTPUT_BASE_DIR>/<snake_case_topic_name>/
   ```

2. Push immediately (don't wait for other topics):
   ```bash
   cd tt-symbiote-research-topics
   git add research_topics.md
   git commit -m "research: completed topic <topic-name>"
   git push
   ```
   On conflict: `git pull --rebase && git push`

### Final report

- Topics researched (count)
- Per topic: name, output directory, one-line finding summary

---

## File Layout

```
<output_dir>/
├── plan.md
├── index.md
├── compression_analysis.md          # cross-chapter
├── ch1_<title>/
│   ├── index.md
│   ├── <topic_a>.md
│   ├── <topic_b>.md
│   ├── b_review.md
│   └── compression_analysis.md
├── ch2_<title>/
│   └── ...
└── ...
```

Directory names: `ch<N>_<short_snake_case_title>`

---

## Rules

1. **Agent A alone writes files.** B and C only read and produce feedback.
2. **Every CRUCIAL item must be applied** before advancing.
3. **`plan.md` is the structural authority.** Content contradictions → B flags, A fixes or amends the plan.
4. **Chapters are sequential.** Don't start chapter N+1 until N is done.
5. **Index comes last.** Don't write it until all chapters are done.
6. **`compression_analysis.md` is append-only.** Each pass adds a section. Never delete prior passes.
7. **Agent isolation is mandatory.** B and C never run in the same context as A.
8. **All file references must be clickable links.**
   - Chapter `index.md`: `[`filename.md`](./filename.md)`
   - Guide `index.md`: links to each chapter's `index.md`
9. **Every content file ends with a navigation footer** (excludes `index.md`, `b_review.md`, `compression_analysis.md`, `plan.md`):

   | Position | Footer |
   |---|---|
   | Not last file in chapter | `---\n\n**Next:** [\`next_file.md\`](./next_file.md)` |
   | Last in chapter, not last chapter | `---\n\n**Next:** [Chapter N+1 — Title](../chN+1_title/index.md)` |
   | Last file of last chapter | `---\n\n**End of guide.** Return to [Guide Index](../index.md)` |

10. **`index.md` files are pure navigation.** No content that belongs in a section file.
11. **LaTeX formatting:**

    | Context | Format |
    |---|---|
    | Display equations (default) | `$$...$$` on own lines |
    | Display with `\texttt` + underscores or `\!` | ` ```math ` fenced block |
    | Inline expressions | `$...$` |
    | Shape annotations, arithmetic, pseudocode | Code blocks or plain text |

    **Forbidden in `$$...$$` blocks:** underscores inside `\text{...}`, `\texttt{name_with_underscores}`, `\!`.
