---
name: test-remix
description: Run end-to-end remix test using Chrome MCP with example songs
argument-hint: "[--curl] [--prompt 'custom prompt']"
---

# Test Remix: End-to-End Validation

Run a full remix test to verify the pipeline works end-to-end.

**Usage:**
- `/test-remix` — Test via Chrome MCP (browser-based, visual verification)
- `/test-remix --curl` — Test via curl (API-only, no browser)
- `/test-remix --prompt "vocals from A with drums from B"` — Custom remix prompt

**Arguments:** `$ARGUMENTS`

## Test Data

| Song | File | Role |
|------|------|------|
| The Notorious B.I.G. - Hypnotize | `examples/The Notorious B.I.G. - Hypnotize (Official Audio).mp3` | Song A (vocals source) |
| Grateful Dead - Althea | `examples/Althea (2013 Remaster).mp3` | Song B (instrumentals source) |

Default prompt: `"Biggie vocals over Althea instrumentals"`

---

## Step 1: Pre-Flight Check

Verify dev servers are running:

```bash
curl -s http://localhost:8000/health
```

If not running, suggest: "Servers aren't up. Run `/dev` first."

Verify test files exist:
```bash
ls "examples/The Notorious B.I.G. - Hypnotize (Official Audio).mp3"
ls "examples/Althea (2013 Remaster).mp3"
```

## Step 2: Run Test

### Option A: Chrome MCP (default)

Use Chrome MCP to test the full browser flow:

1. Navigate to `http://localhost:5173` (or `http://localhost:8000` if using static UI)
2. Upload Song A (Hypnotize) to the first upload slot
3. Upload Song B (Althea) to the second upload slot
4. Enter the remix prompt
5. Click submit/create remix
6. Wait for processing to complete (expect 3-15 minutes depending on backend)
7. Verify audio player appears with playable remix
8. Take a screenshot of the final state

### Option B: curl (`--curl` flag)

Test the API directly without a browser:

```bash
# Submit remix
SESSION=$(curl -s -X POST http://localhost:8000/api/remix \
  -F "song_a=@examples/The Notorious B.I.G. - Hypnotize (Official Audio).mp3" \
  -F "song_b=@examples/Althea (2013 Remaster).mp3" \
  -F "prompt=Biggie vocals over Althea instrumentals" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['session_id'])")

echo "Session ID: $SESSION"

# Download the remix
curl -s "http://localhost:8000/api/remix/${SESSION}/audio" --output /tmp/test-remix.mp3

# Verify it's a valid MP3
file /tmp/test-remix.mp3
ffprobe -v quiet -show_format /tmp/test-remix.mp3 2>&1 | grep -E "duration|format_name|bit_rate"
```

## Step 3: Report Results

```
TEST REMIX RESULTS
══════════════════
Method: [Chrome MCP | curl]
Songs: Hypnotize (A) + Althea (B)
Prompt: [prompt used]
Session ID: [id]
Status: [SUCCESS | FAILED]

Duration: [processing time]
Output: [file path or "playing in browser"]
Audio: [duration]s, [bitrate] kbps, [format]

[If failed: error details and suggested fix]
```

## Expected Processing Times

| Backend | Model | Expected Time |
|---------|-------|--------------|
| Modal (cloud GPU) | BS-RoFormer 6-stem | 3-5 min (first run: +60-90s cold start) |
| Local (CPU) | htdemucs_ft 4-stem | 10-20 min |

Day 1 separates both songs sequentially, so double the single-song time.
