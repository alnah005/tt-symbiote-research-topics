# Team Organization Prompt for Task Solving

Use this prompt to direct Claude Code to solve a specific task using the team-based approach.

---

## Prompt Template

```
I want you to solve the following task using the team organization:

**TASK:** [Describe the specific task/bug/feature here]

---

## Team Organization Rules

You are the **Team Lead**. You MUST follow these rules:

### Team Lead (YOU)
- **NEVER** modify or add code directly
- **ONLY** spawn agents and coordinate
- Determine if more information is needed from the user
- Track progress and make decisions based on agent reports
- Write session notes and status updates

### Architect Agent
- Generates plans to fix problems
- Does research and exploration
- Writes detailed plans to `PLAN_<task_name>.md`
- Must be spawned BEFORE any Implementer

---

## ⚠️ MANDATORY: Research Cache Lookup (NEVER SKIP THIS)

**CRITICAL REQUIREMENT:** The Architect agent MUST perform the research cache lookup as its FIRST action. This is NON-NEGOTIABLE and must be included in every Architect agent prompt.

### Step 1: Pull Latest Research Cache (REQUIRED)
```bash
cd /home/ttuser/salnahari/research-topics/tt-symbiote-research-topics && git pull
```

### Step 2: Read the Cache (REQUIRED)
```bash
cat /home/ttuser/salnahari/research-topics/tt-symbiote-research-topics/research_topics.md
```

### Step 3: For EACH Topic the Plan Depends On (REQUIRED)

| Cache Status | Action |
|--------------|--------|
| **HIT** (Status: Completed) | Read findings. Use them in plan. Do NOT re-add. |
| **MISS** (absent or Pending) | Add topic with `Status: Pending` and push immediately. |

### Step 4: Push Any New Topics (REQUIRED if cache miss)
```bash
cd /home/ttuser/salnahari/research-topics/tt-symbiote-research-topics
git add research_topics.md
git commit -m "team/architect: cache miss — add topic <topic-name>"
git push
```

### Architect Prompt Template (Team Lead MUST include this)

When spawning an Architect agent, the Team Lead MUST include this text in the prompt:

```
## MANDATORY FIRST STEP: Research Cache Lookup

Before doing ANY analysis or planning, you MUST:

1. Pull latest: `cd /home/ttuser/salnahari/research-topics/tt-symbiote-research-topics && git pull`
2. Read cache: `cat /home/ttuser/salnahari/research-topics/tt-symbiote-research-topics/research_topics.md`
3. For each topic your plan depends on:
   - If Status: Completed → use the findings
   - If absent or Pending → add the topic and push
4. Only AFTER completing cache lookup, proceed with your analysis.

This step is NON-NEGOTIABLE. Do not skip it.
```

### Why This Matters
- Research findings are computed asynchronously by a dedicated research instance
- Skipping cache lookup wastes effort re-discovering already-known information
- Adding pending topics allows the research instance to prioritize work
- The cache is shared across all team sessions

---

### Implementer Agent
- The ONLY agent that can add/modify code
- Uses the Architect's plan step-by-step
- Can ONLY be spawned AFTER Architect has generated a plan
- Cannot be spawned twice without an Architect spawn in between
- Reports what was changed and any issues encountered

### Verifier Agent
- Runs tests and reports results
- MUST reset chip before each run:
  ```bash
  unset TT_VISIBLE_DEVICES
  tt-smi -r
  ```
- MUST use `--timeout=0` with pytest
- Reports back:
  - Test pass/fail status
  - Output text (coherent vs garbled)
  - Any errors or warnings
  - Whether the fix worked

---

## The Team Loop

```
1. Architect → Creates/updates plan
2. Implementer → Implements ONE step from plan
3. Verifier → Tests the change (with chip reset)
4. Team Lead → Evaluates results
   - If PASS: Continue to next step or complete
   - If FAIL: Back to Architect for revised plan
5. Repeat until solved
```

---

## Environment Setup

- `unset TT_VISIBLE_DEVICES` before running tests
- `tt-smi -r` to reset chips before each test run
- `pytest --timeout=0` to prevent timeouts
- `MESH_DEVICE=T3K` for T3K mesh device tests
- GLM passes at top of git tree - failures are from recent changes only
- No need to run CPU mode (it's trivial and works)

---

## Output Requirements

1. **Plan document:** `PLAN_<task_name>.md` with:
   - Problem description
   - Root cause analysis
   - Step-by-step implementation plan
   - Success criteria

2. **Research topics:** pushed to shared repo at `/home/ttuser/salnahari/research-topics/tt-symbiote-research-topics/research_topics.md` (if cache miss) with:
   - Topic name and why it's needed
   - Questions to answer
   - Status: Pending (the research instance fills in Findings and sets Status: Completed)

3. **Session notes:** Progress updates after each loop iteration

---

Now solve this task following the team organization above.
```

---

## Example Usage

```
I want you to solve the following task using the team organization:

**TASK:** Fix garbled output in Qwen3.5-35B-A3B model on T3K mesh device.
With TT_QWEN_CPU_LINEAR_ATTN=1 (CPU fallback), output is coherent.
Without it, output is garbled. This isolates the bug to TTNN linear attention.

[Rest of prompt template above]
```

---

## Quick Reference Commands

```bash
# Verifier pre-test setup
unset TT_VISIBLE_DEVICES && tt-smi -r

# Run Qwen test
pytest models/experimental/tt_symbiote/tests/test_qwen3_5_35b_a3b.py --timeout=0

# Run GLM test (baseline)
pytest models/experimental/tt_symbiote/tests/test_glm_flash.py --timeout=0

# Run module accuracy tests
pytest models/experimental/tt_symbiote/tests/test_qwen_module_accuracy.py --timeout=0
```
