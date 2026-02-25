---
name: dev
description: Start dev environment with pre-flight checks (kills zombies, frees ports, starts servers)
argument-hint: "[backend|frontend|all] [--stop] [--cleanup-only]"
---

# Dev Environment Manager

Start the musicMixer development environment safely. Kills zombie agents, frees ports, starts servers, verifies health.

**Usage:**
- `/dev` — Start backend + frontend (default: all)
- `/dev backend` — Start backend only
- `/dev frontend` — Start frontend only
- `/dev --stop` — Stop all dev servers and free ports
- `/dev --cleanup-only` — Kill zombies and free ports without starting servers

**Arguments:** `$ARGUMENTS`

---

## Step 1: Kill Zombie Agents

Check for orphaned agent processes that could trigger file watcher reload loops:

```bash
# Find rogue agents
pgrep -lf 'claude -p|codex exec|gemini -m' 2>/dev/null
```

If found:
- List each process with PID and command
- Kill them: `pkill -f 'claude -p'; pkill -f 'codex exec'; pkill -f 'gemini -m'`
- Report: "Killed N zombie agent(s)"

If none found: "No zombie agents detected."

## Step 2: Check and Free Ports

Check if target ports are occupied:

```bash
# Backend port
lsof -i :8000 -t 2>/dev/null
# Frontend port
lsof -i :5173 -t 2>/dev/null
```

If occupied:
- Show what's using the port: `lsof -i :PORT`
- Kill the process: `kill $(lsof -i :PORT -t)`
- Report: "Freed port PORT (was held by PROCESS)"

If `--cleanup-only` was specified, stop here and report clean state.
If `--stop` was specified, also stop here after killing any server processes.

## Step 3: Start Servers

Based on the argument (default: all):

### Backend (port 8000)
```bash
cd /Users/aaron/Projects/musicMixer/backend
uv run uvicorn musicmixer.main:app --reload --port 8000 &
```

Wait up to 10 seconds for health check:
```bash
curl -s http://localhost:8000/health
```

Expected: `{"status":"ok"}`

### Frontend (port 5173)
```bash
cd /Users/aaron/Projects/musicMixer/frontend
bun run dev &
```

Wait up to 10 seconds for the dev server to start.

## Step 4: Report Status

```
DEV ENVIRONMENT
═══════════════
Zombies killed: N
Ports freed: [list or "none"]

Backend:  ✓ http://localhost:8000 (health: ok)
Frontend: ✓ http://localhost:5173

Tip: Run `/dev --stop` to shut down cleanly
```

If any server failed to start, report the error and suggest next steps.

---

## Important Notes

- **File watcher warning:** The backend runs with `--reload`. Any file changes in `backend/` will restart the server. If agents are editing backend files, use `uv run uvicorn musicmixer.main:app --port 8000` (without `--reload`) instead.
- **Log output:** Server logs go to the terminal where they were started. Use `uv run dev 2>&1 | tee /tmp/backend.log` to capture logs for debugging.
- **Frontend requires backend:** The frontend expects the backend API at localhost:8000. Start backend first or start both.
