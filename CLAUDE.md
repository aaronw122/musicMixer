# musicMixer Workspace

## Repos

This workspace contains two repos as child directories:

| Repo | Purpose | Status |
|------|---------|--------|
| `frontend/` | Web client (browser-based audio mixing UI) | **Active** |
| `backend/` | Server-side API and audio processing pipeline | **Active** |

Each repo should have its own CLAUDE.md with detailed context. When you run from a child repo, both that repo's CLAUDE.md and this parent file are loaded — child instructions take precedence.

## Workspace Git Repository

The workspace root is a git repo that tracks shared configuration and tooling. Child directories (`frontend/`, `backend/`) are **independent git repos** — they are gitignored by the workspace repo.

**Scoping rule for agents:** Git commands run from a child repo (e.g., `frontend/`) scope to that child repo. Git commands run from the workspace root scope to the workspace repo. Be careful not to accidentally commit workspace-level changes to a child repo or vice versa.

### What Is Tracked vs Gitignored

| Tracked (shared) | Gitignored (workspace-specific) |
|---|---|
| `CLAUDE.md` | `notes/` |
| `docs/` | `.mcp.json` (secrets) |
| `blueprints/` | `.claude/settings.local.json` |
| `.claude/commands/` | Child repos (`frontend/`, `backend/`) |
| `.gitignore` | `node_modules/` |
| `.mcp.json.example` | |

### Setting Up `.mcp.json`

`.mcp.json` is gitignored (contains API keys). To set up in a new workspace:

1. Copy the template: `cp .mcp.json.example .mcp.json`
2. Fill in your actual API keys

## About musicMixer

A web app that lets anyone create a music remix by prompting AI. Upload two songs, type what you want ("Hendrix guitar with MF Doom rapping over it"), and the AI splits the songs into stems, combines them, auto-matches tempo and key, and plays it back. Anyone can be a DJ.

Every remix is ephemeral — replayable for up to 3 hours or until you create a new one, then gone forever.

### Core Flow

1. Upload two songs (local MP3/WAV files)
2. Type a prompt describing the mashup
3. AI separates stems, selects the right ones, matches tempo/key
4. Transparent progress updates as it works
5. Remix plays back
6. Remix expires after 3 hours or when a new one is created

## Context

Early-stage personal project. Prioritize simplicity — avoid over-engineering. Ship quality core software efficiently.

**Push back like a senior engineer.** If a request could cause bugs, side effects, technical debt, or architectural problems — say so directly. Don't just execute questionable instructions; flag concerns and propose better alternatives.

## Package Manager

**This project uses Bun, not npm.** Use `bun` for all package management and script execution:

```bash
# Package management
bun install          # Install dependencies
bun add <package>    # Add a dependency
bun add -d <package> # Add a dev dependency
bun remove <package> # Remove a dependency

# Running scripts
bun run dev          # Start dev server
bun run build        # Build for production
bun run test         # Run tests
bun run <script>     # Run any package.json script

# Running files directly
bun run file.ts      # Execute a TypeScript file directly
```

**NEVER use npm, npx, yarn, or pnpm.** If you see a command in docs or Stack Overflow that uses npm, translate it to the bun equivalent before running.

## Full-Stack Awareness

When working on features, consider both sides. Client changes often need corresponding API work, and API changes may affect the client. Check the other repo when relevant — you have visibility into both.

- The `backend/` repo is available for researching API endpoints, processing pipeline, or validating integration points.
- The `frontend/` repo is available for understanding how the API is consumed and how audio is rendered in-browser.

## Documentation

### Three-Tier Documentation Hierarchy

1. **Blueprints** (`blueprints/`): Human‑written, gold‑standard architectural and system documentation. This is the canonical reference and should be updated when system‑level truths change.
2. **Docs** (`docs/`): Persistent supporting documentation that should remain accurate over time, but is not at the same criticality as Blueprints (for example, user guides, runbooks, or stable component details).
3. **Notes** (`notes/`): Temporary, work‑in‑progress material. By default, new decisions, discoveries, and conversation summaries land here first. Promote content upward to Docs or Blueprints once it is stable and broadly useful.

**Reviews go in `notes/`.** When running subagent reviews (plan reviews, code reviews, etc.), always write output files to `notes/`, not `/tmp` or other locations. Use descriptive filenames like `notes/2026-02-23-mvp-plan-review-agent1.md`.

**Session documentation is mandatory.** During any session that involves implementation, debugging, investigation, or design work:
- Write findings, decisions, and changes to `notes/` as you go — don't wait until the end or until asked.
- Use the format `notes/YYYY-MM-DD-<topic>.md` (e.g., `notes/2026-02-25-volume-dip-investigation.md`).
- Include: what was investigated, what changed, key decisions and their rationale, and any open items for future sessions.
- If subagents write scratch output to `/tmp`, the orchestrator must ensure the useful content gets captured in `notes/` before the session ends.
- Trivial sessions (quick questions, single-line fixes) don't need notes.

## Safety Rules

**NEVER execute these commands without explicit user approval:**

```bash
# File deletion
rm -rf, rm -f, find . -delete

# Git destructive operations
git push --force, git reset --hard

# Database destructive commands
DROP TABLE, DELETE FROM (without WHERE), TRUNCATE, migration resets
```

**How to get approval:** Before running any destructive command, STOP and ask the user explicitly: "This command will [describe impact]. Do you want me to proceed?" Wait for a clear "yes" before executing.

## Testing

**Unit test fixes:** When asked to fix failing unit tests, first understand why they failed. Treat failures as strong signals of incorrect logic, not just brittle tests. If you conclude the root cause is production/business logic rather than the test itself, stop and bring it to the operator's attention before changing tests.

**Test runs:** Whenever you run tests, always report the number of failing tests in your final output.

## Self-Improvement

If you notice a pattern, convention, or piece of knowledge that would help future sessions, suggest adding it to the relevant CLAUDE.md file. A 30-second addition here saves hours across future sessions.

### When to Add a Lesson

Any time you:
- Spend significant time debugging something non-obvious
- Discover a pattern, convention, or constraint that isn't documented
- Hit a gotcha that would trip up the next agent
- Find that existing documentation is wrong or misleading
- Learn something about how frontend and backend interact

### Where to Put It

| Scope | File |
|-------|------|
| Workspace-wide (cross-repo, tooling, workflow) | This file (`CLAUDE.md` at workspace root) |
| Frontend-specific | `frontend/CLAUDE.md` → Lessons Learned section |
| Backend-specific | `backend/CLAUDE.md` → Lessons Learned section |

## Implementation Schedule

4-day sprint. Demo: **Saturday 2026-03-01**.

| Day | Milestone | Key Deliverable |
|-----|-----------|----------------|
| Day 1 | Hardcoded Frankenmix | E2E pipeline works: upload 2 songs → stems separated → hardcoded mix → playback |
| Day 2 | The Real Remix | Prompt-driven stem selection, tempo/key matching, async processing + SSE progress |
| Day 3 | Intelligence | LLM-powered mix decisions, energy profiling, smart volume/timing |
| Day 4 | Demo Ready | Polish, error handling, test suite, deployment |

Implementation plans: `docs/impl/day-{1,2,3,4}-*.md`

## Subagent Implementation Workflow

When deploying subagents for parallel implementation work, follow this standard pattern:

1. **Worktree isolation** — every implementation agent gets its own worktree (`isolation: "worktree"`)
2. **Implement + test** — agent implements the task, writes/runs tests, fixes failures
3. **Commit + PR** — agent commits with conventional message, pushes to branch, creates PR via `gh pr create`
4. **Agent review** — a separate review agent checks the PR for correctness, patterns, and test coverage
5. **Orchestrator tracks status** — orchestrator monitors PR URLs and review results, reports to user

**Orchestrator discipline:** The orchestrator's ONLY job is to spawn agents, monitor status, and report results. Never read implementation files or agent output directly in the orchestrator. Use consolidation subagents for result gathering. Even small tasks should be delegated if multi-agent coordination follows.

## Test Data

Example songs for testing are in `examples/`:

| File | Role |
|------|------|
| `The Notorious B.I.G. - Hypnotize (Official Audio).mp3` | Song A (vocals source) |
| `Althea (2013 Remaster).mp3` | Song B (instrumentals source) |

## AGENTS.md Symlinks

`AGENTS.md` files in `frontend/` and `backend/` are symlinks to their respective `CLAUDE.md`. Edit `CLAUDE.md` only — both files point to the same content.

## Lessons Learned

- **Kill background agents before dev servers.** "Clear" agents can outlive tasks and trigger `--reload` loops.
  - **Symptoms:** "changes detected" messages, server restarting endlessly, pages stop loading, port conflicts
  - **Diagnose:** `pgrep -lf 'claude -p|codex exec|gemini -m'`
  - **Kill:** `pkill -f 'claude -p'; pkill -f 'codex exec'; pkill -f 'gemini -m'`
  - **Verify ports free:** `lsof -i :8000` and `lsof -i :5173`
  - **Root cause:** Background agents edit files in the project dir, triggering uvicorn's `--reload` file watcher into restart loops. Each restart drops active connections.
