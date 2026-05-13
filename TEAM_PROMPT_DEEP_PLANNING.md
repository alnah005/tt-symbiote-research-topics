# Team Organization Prompt for Task Solving (Deep Planning Mode)

Use this prompt to direct Claude Code to solve a specific task using the deep-planning team-based approach. Deep planning mode spawns multiple independent Architects in parallel, evaluates their plans independently, synthesizes the best, and verifies it before proceeding — looping until a solid plan is confirmed.

---

## Prompt Template

```
I want you to solve the following task using the deep-planning team organization:

**TASK:** [Describe the specific task/bug/feature here]

---

## Team Organization Rules

You are the **Team Lead**. You MUST follow these rules:

### ⚠️ Git Push Constraints (STRICTLY ENFORCED)
- **research-topics repo:** ONLY `research_topics.md` may be pushed
- **tt-metal repo:** NO pushes allowed — strictly prohibited

### Team Lead (YOU)
- **NEVER** modify or add code directly
- **ONLY** spawn agents and coordinate
- Determine if more information is needed from the user
- Track progress and make decisions based on agent reports
- Write session notes and status updates
- **Assess problem difficulty and decide N (number of parallel Architects)**

---

## ⚠️ MANDATORY: Research Cache Lookup (NEVER SKIP THIS)

**CRITICAL REQUIREMENT:** Every Architect agent (planning, evaluating, or synthesizing) MUST perform the research cache lookup as its FIRST action. This is NON-NEGOTIABLE.

### Step 1: Pull Latest Research Cache (REQUIRED)
```bash
cd tt-symbiote-research-topics && git pull
```

### Step 2: Read the Cache (REQUIRED)
```bash
cat research_topics.md
```

### Step 3: For EACH Topic the Plan Depends On (REQUIRED)

> **What makes a good research topic?**
> Topics must be **generic and reusable** — broadly applicable across different tasks, not tied to one specific bug or feature. Ask: *"Would another task in a different context ever need this answer?"* If not, do not add it.

- **Cache HIT** — topic exists with `Status: Completed`: read the findings and use them. Do not re-add.
- **Cache MISS** — topic is absent OR has `Status: Pending`: proceed using best-effort knowledge. Report to Team Lead.

### Architect Prompt Template (Team Lead MUST include this in EVERY Architect spawn)

```
## MANDATORY FIRST STEP: Research Cache Lookup (EXECUTE THESE COMMANDS)

You MUST execute the following commands using the Bash tool and SHOW their output before doing ANY other work.

### Step 1: Pull latest (USE BASH TOOL)
```bash
cd tt-symbiote-research-topics && git pull
```

### Step 2: Read cache (USE BASH TOOL)
```bash
cat research_topics.md
```

### Step 3: Report cache status
After reading the cache, you MUST report in this format:

**CACHE LOOKUP RESULTS:**
| Topic | Status | Action |
|-------|--------|--------|
| [topic name] | Completed/Pending/Missing | Using findings / Best-effort / Adding |

### Step 4: For research questions with Status: Completed
Copy the **Findings** section into your analysis. Do NOT re-research completed questions.

### Step 5: For missing research questions (cache miss)
Report the missing topic to the Team Lead (DO NOT ADD PLAN FILES or ADD RESEARCH QUESTIONS THAT HAVE BEEN ANSWERED BY YOU). The Team Lead handles all writes and pushes to the research-topics repo.

## VERIFICATION CHECKLIST (Must complete before proceeding)
- [ ] Executed `git pull` and showed output
- [ ] Executed `cat research_topics.md` and showed output
- [ ] Listed all relevant research questions with their Status
- [ ] Used findings from Completed topics
- [ ] Reported any missing topics to the Team Lead

**FAILURE TO SHOW COMMAND OUTPUTS = INVALID RESPONSE. START OVER.**
```

---

## Deep Planning Loop

```
STEP 0:  Team Lead assesses difficulty → sets N
STEP 1:  N Planning Architects spawned in parallel → N independent plans
STEP 2:  N Evaluator Architects spawned in parallel → N independent evaluations
STEP 3:  1 Synthesis Architect → picks the best plan
STEP 4:  1 Verification Architect → checks if the plan is solid
         - PASS → proceed to Implementer → Verifier loop (Steps 5–7)
         - FAIL → return to STEP 1 with failure notes injected into prompts
STEP 5:  Implementer → implements ONE step from the verified plan
STEP 6:  Verifier → tests the change (with chip reset)
STEP 7:  Team Lead evaluates results
         - PASS → continue to next step or complete
         - FAIL → return to STEP 1 for revised plans
```

---

## Step 0: Team Lead Difficulty Assessment

Before spawning any Architects, the Team Lead MUST assess the difficulty of the task and choose N:

| Difficulty | Criteria | N |
|------------|----------|---|
| **Trivial** | Straightforward, well-understood change (e.g. rename, 1-line config fix) | 2 |
| **Low** | Small feature or bug with clear root cause | 3 |
| **Medium** | Multi-file change, unclear root cause, or moderate algorithmic complexity | 4 |
| **High** | Cross-system change, novel approach required, or significant uncertainty | 5 |
| **Critical** | Architecture-level change, high risk of breakage, or foundational component | 6+ |

The Team Lead MUST report the chosen N and its justification before spawning any agents.

---

## Step 1: N Planning Architects (Parallel)

Spawn N Architect agents **simultaneously** (all in a single parallel batch). Each receives the same task description plus the Research Cache Lookup mandate, but MUST produce an **independent** plan without being aware of the others.

### ⚠️ File Location Rule (STRICTLY ENFORCED)
Planning Architects MUST write their plans to `/tmp`, NOT the current working directory:
```
/tmp/PLAN_<task_name>_v<1..N>.md
```

Each plan MUST include:
- Problem description
- Root cause analysis
- Step-by-step implementation plan
- Success criteria
- Risks and trade-offs
- Any cache-miss research topics to report

The Team Lead collects all N plans before proceeding.

---

## Step 2: N Evaluator Architects (Parallel)

Spawn N Evaluator Architect agents **simultaneously**, one per proposed plan. Each Evaluator receives:
- The task description
- **All N plans** (read from `/tmp/PLAN_<task_name>_v*.md`)
- Their assigned plan index (1..N)
- The Research Cache Lookup mandate

### ⚠️ File Location Rule (STRICTLY ENFORCED)
Evaluator Architects MUST write their evaluations to `/tmp`, NOT the current working directory:
```
/tmp/EVAL_<task_name>_v<1..N>.md
```

Each evaluation MUST include:
- **Assigned plan summary** (one paragraph)
- **Strengths** of the assigned plan
- **Weaknesses / risks** of the assigned plan
- **Comparison to other plans** (brief — which aspects of other plans are better or worse)
- **Recommendation:** Accept / Accept with modifications / Reject (with reason)

The Team Lead collects all N evaluations before proceeding.

---

## Step 3: 1 Synthesis Architect

Spawn **one** Synthesis Architect with:
- The task description
- All N plans (read from `/tmp/PLAN_<task_name>_v*.md`)
- All N evaluations (read from `/tmp/EVAL_<task_name>_v*.md`)
- The Research Cache Lookup mandate

### ⚠️ File Location Rule (STRICTLY ENFORCED)
The Synthesis Architect MUST write the candidate final plan to `/tmp`, NOT the current working directory:
```
/tmp/PLAN_<task_name>_final.md
```

The Synthesis Architect MUST:
1. Review all plans and evaluations
2. Select the single best plan (or construct a hybrid)
3. Write the candidate final plan to `/tmp/PLAN_<task_name>_final.md`
4. Include a **Selection Rationale** section explaining why this plan was chosen over the others
5. Report any cache-miss research topics to the Team Lead

The Team Lead collects the candidate final plan before proceeding.

---

## Step 4: 1 Verification Architect

Spawn **one** Verification Architect with:
- The task description
- The candidate final plan (read from `/tmp/PLAN_<task_name>_final.md`)
- The Research Cache Lookup mandate
- (On retry) Notes from previous failed verifications

The Verification Architect MUST:
1. Critically assess the final plan against the task requirements
2. Check for: logical gaps, missing edge cases, incorrect assumptions, unresolved dependencies
3. Return a verdict:

**VERDICT: PASS** — the plan is solid and ready for implementation. Provide a brief rationale.

**VERDICT: FAIL** — the plan has unresolved issues. List each issue precisely:
```
ISSUE 1: [Description of gap or flaw]
ISSUE 2: ...
```

The Team Lead acts on the verdict:
- **PASS** → copy `/tmp/PLAN_<task_name>_final.md` to `PLAN_<task_name>_final.md` in the current working directory, then proceed to Step 5 (Implementer)
- **FAIL** → inject the failure notes into the next round of Planning Architect prompts and return to Step 1

### ⚠️ Only the Team Lead may write to the current working directory
No Architect agent (planning, evaluating, or synthesizing) may write files to the current working directory. The Team Lead is the sole agent that copies the verified final plan from `/tmp` to the project directory on PASS.

---

## Step 5–7: Implementation Loop

### Implementer Agent
- The ONLY agent that can add/modify code
- Implements the verified `PLAN_<task_name>_final.md` step-by-step
- Can ONLY be spawned AFTER Verification Architect returns PASS
- Reports what was changed and any issues encountered
- **⚠️ CANNOT push to tt-metal — all code changes remain local**

### Verifier Agent
- Runs tests and reports results
- MUST reset chip before each run:
  ```bash
  unset TT_VISIBLE_DEVICES
  tt-smi -r
  ```
- MUST use `--timeout=0` with pytest
- Reports:
  - Test pass/fail status
  - Output text (coherent vs garbled)
  - Any errors or warnings
  - Whether the fix worked

### Team Lead Evaluation (after each implementation step)
- **PASS** → continue to next plan step or mark complete
- **FAIL** → return to Step 1 with failure context injected into Planning Architect prompts

---

## Environment Setup

- `unset TT_VISIBLE_DEVICES` before running tests
- `tt-smi -r` to reset chips before each test run
- `pytest --timeout=0` to prevent timeouts
- `MESH_DEVICE=QB2` for QB2 mesh device tests
- tt-symbiote by default shards the output of all TTNN modules on the last dimension (check run_config.py). Expect to add an all gather if shape mismatches.
- No need to run CPU mode (it's trivial and works)

---

## Output Requirements

1. **Planning plans:** `/tmp/PLAN_<task_name>_v<1..N>.md` — one per Planning Architect (temporary, never in project dir)
2. **Evaluations:** `/tmp/EVAL_<task_name>_v<1..N>.md` — one per Evaluator Architect (temporary, never in project dir)
3. **Candidate final plan:** `/tmp/PLAN_<task_name>_final.md` — written by Synthesis Architect (temporary)
4. **Verified final plan:** `PLAN_<task_name>_final.md` in the **current working directory** — copied here by the Team Lead ONLY after Verification Architect returns PASS
5. **Research topics:** pushed to `research_topics.md` by Team Lead (if cache miss)
   - Topic name (generic and reusable)
   - Questions to answer (Architect poses, does NOT fill in)
   - Findings: TBD / Status: Pending
5. **Session notes:** N chosen, round number, verdict per round, and overall progress

---

Now solve this task following the deep-planning team organization above.
```
