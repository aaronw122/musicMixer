---
name: execute-plan
description: Orchestrate multi-agent plan execution with worktrees, review, testing, and PRs
argument-hint: "<path/to/plan.md> [--dry-run]"
---

# Execute Plan: Multi-Agent Orchestrated Implementation

Read a plan document, identify parallelizable work, and deploy subagents in isolated worktrees to implement, test, review, and PR each piece.

**Usage:**
- `/execute-plan docs/impl/day-2-the-real-remix.md` — Execute remaining steps
- `/execute-plan docs/impl/day-2-the-real-remix.md --dry-run` — Preview execution plan without deploying agents

**Arguments:** `$ARGUMENTS`

---

## Phase 1: Analyze the Plan

1. Read the plan document specified in the arguments
2. Parse all tasks/steps, identifying:
   - **Completed** items (checked boxes `[x]`, status markers)
   - **Remaining** items (unchecked boxes `[ ]`, pending status)
   - **Dependencies** between remaining items (what must finish before what)
3. Cross-reference with git history (merged PRs, recent commits) to catch items that are done but not checked off
4. Group remaining items into **parallelizable buckets** — items with no inter-dependencies can run concurrently

## Phase 2: Present Execution Plan

Before deploying any agents, present a summary to the user:

```
EXECUTION PLAN
══════════════
Plan: [filename]
Completed: X of Y items
Remaining: Z items in N parallel buckets

Bucket 1 (parallel):
  - [task A] → agent in worktree
  - [task B] → agent in worktree

Bucket 2 (after bucket 1):
  - [task C, depends on A] → agent in worktree

Estimated agents: N
```

If `--dry-run` was specified, stop here and exit.

Otherwise, ask for confirmation: "Deploy N agents? (yes/no)"

## Phase 3: Pre-Flight Checks

Before deploying agents:

1. **Kill zombie agents** — check for orphaned claude/codex/gemini processes:
   ```bash
   pgrep -lf 'claude -p|codex exec|gemini -m' 2>/dev/null
   ```
   If found, warn and kill (with confirmation).

2. **Check port availability** — verify 8000 and 5173 are free if the plan involves running servers.

3. **Verify clean git state** — ensure no uncommitted changes in the target repo(s) that could cause merge conflicts.

## Phase 4: Deploy Agents

For each bucket (sequential between buckets, parallel within a bucket):

Deploy each task as a subagent with `isolation: "worktree"`. Each agent receives this prompt template:

```
You are implementing a specific task from the musicMixer project plan.

## Your Task
[task description from plan]

## Instructions
1. Implement the task in your worktree
2. Write or update tests for your changes
3. Run tests and fix any failures
4. When complete, commit your changes with a conventional commit message
5. Push to a new branch and create a PR using `gh pr create`
6. The PR title should be descriptive, the body should summarize changes and include test results

## Context
- Read the full plan at: [plan path] for broader context
- Read the relevant CLAUDE.md files for conventions
- Your work is isolated in a worktree — commit freely
- Another agent will review your PR after you create it

## Quality Bar
- Tests must pass
- Code must follow existing patterns (read similar files first)
- No TODO comments or placeholder implementations
- Conventional commit messages (feat/fix/refactor)
```

## Phase 5: Monitor and Review

As agents complete:

1. Track which agents have finished and their PR URLs
2. For each completed PR, deploy a **review agent** to check the PR:
   ```
   Review PR #{number} for:
   - Correctness (does it implement the task?)
   - Code quality (follows project patterns?)
   - Test coverage (are changes tested?)
   - No regressions introduced

   Leave a review comment via `gh pr review`.
   ```
3. If a review agent finds critical issues, notify the user

## Phase 6: Report Results

When all agents complete, present a summary:

```
EXECUTION COMPLETE
══════════════════
Plan: [filename]
Agents deployed: N
PRs created: [list with URLs]
Reviews: [pass/fail status]
Items completed: X (was Y, now Z of total)

Next steps:
- [any remaining items or follow-ups]
```

Update the plan document to check off completed items.

---

## Key Rules

- **Orchestrator stays lean.** Never read implementation files or agent output directly. Use consolidation subagents for result gathering.
- **One task per agent.** Don't bundle unrelated tasks into a single agent.
- **Worktree isolation is mandatory.** Every implementation agent gets its own worktree.
- **Review before merge.** No PR merges without agent review.
- **Respect dependencies.** Only deploy bucket N+1 after bucket N completes.
- **Keep the user informed.** Report agent status as they complete, don't wait until the end.
