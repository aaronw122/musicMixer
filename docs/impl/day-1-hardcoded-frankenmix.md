# Day 1: "Hardcoded Frankenmix" -- Implementation Plan

**Date:** 2026-02-24
**Theme:** End-to-end plumbing proof-of-concept
**Exit criteria:** Upload two songs in a browser, hear vocals from Song A over instrumentals from Song B, press play.

---

## What We Are Building

A synchronous pipeline: upload 2 MP3/WAV files via a web form, separate each into 6 stems on Modal cloud GPU (BS-RoFormer SW), take the vocal stem from Song A and recombine the 5 instrumental stems (drums + bass + guitar + piano + other) from Song B, overlay them with pydub, export an MP3, and serve it back to a single HTML page with an `<audio>` tag.

**What we are NOT building today:** No async processing, no SSE, no ThreadPoolExecutor, no processing lock, no LLM, no tempo matching, no key matching, no LUFS normalization, no React frontend, no state machine. Synchronous. Hardcoded. Proof-of-concept.

---

## Files to Create

### Backend (`backend/`)

| # | File | Purpose |
|---|------|---------|
| 1 | `pyproject.toml` | Python project config, dependencies, Python version pin |
| 2 | `.python-version` | Pin Python 3.11 for uv |
| 3 | `.env` | Local environment variables (gitignored) |
| 4 | `.gitignore` | Ignore `data/`, `.env`, `__pycache__`, `.venv`, etc. |
| 5 | `CLAUDE.md` | Backend-specific conventions for agents |
| 6 | `src/musicmixer/__init__.py` | Package marker |
| 7 | `src/musicmixer/main.py` | FastAPI app, CORS, lifespan, static HTML serving, mounts API router |
| 8 | `src/musicmixer/config.py` | Pydantic `BaseSettings` (stripped for Day 1) |
| 9 | `src/musicmixer/api/__init__.py` | Package marker |
| 10 | `src/musicmixer/api/health.py` | `GET /health` endpoint |
| 11 | `src/musicmixer/api/remix.py` | `POST /api/remix` (upload + synchronous pipeline), `GET /api/remix/{id}/audio` |
| 12 | `src/musicmixer/services/__init__.py` | Package marker |
| 13 | `src/musicmixer/services/separation.py` | Backend-agnostic `separate_stems()` dispatcher |
| 14 | `src/musicmixer/services/separation_modal.py` | Modal app definition + `separate_stems_remote()` function |
| 15 | `src/musicmixer/services/separation_local.py` | Local `htdemucs_ft` fallback (4-stem) |
| 16 | `src/musicmixer/services/mixer.py` | Hardcoded overlay: vocals + instrumentals, export MP3 |
| 17 | `static/index.html` | Single HTML page: 2 file inputs, prompt input, submit, `<audio>` playback |
| 18 | `data/` directory structure | `data/uploads/`, `data/stems/`, `data/remixes/` (gitignored, created at startup) |

### Frontend

No frontend repo on Day 1. The single HTML page is served directly by FastAPI from `backend/static/index.html`.

---

## Implementation Order

The steps below are sequenced by dependency. Each step states its prerequisite.

---

### Step 1: Backend Project Scaffolding

**Prereq:** None
**Time estimate:** 15-20 minutes

#### 1a. Initialize Python project with uv

```bash
cd /Users/aaron/Projects/musicMixer
mkdir -p backend
cd backend
uv init --name musicmixer --python 3.11
```

If `uv init` does not produce a `src/` layout, restructure manually:

```
backend/
  pyproject.toml
  .python-version
  .env
  .gitignore
  CLAUDE.md
  src/
    musicmixer/
      __init__.py
      main.py
      config.py
      api/
        __init__.py
        health.py
        remix.py
      services/
        __init__.py
        separation.py
        separation_modal.py
        separation_local.py
        mixer.py
  static/
    index.html
  data/
    uploads/     (gitignored, created at runtime)
    stems/       (gitignored, created at runtime)
    remixes/     (gitignored, created at runtime)
```

#### 1b. Install Day 1 dependencies

```bash
cd /Users/aaron/Projects/musicMixer/backend
uv add fastapi 'uvicorn[standard]' python-multipart pydantic-settings
uv add modal soundfile pydub numpy
```

**Do NOT install yet (not needed Day 1):** `librosa`, `essentia`, `pyrubberband`, `pyloudnorm`, `structlog`, `anthropic`, `scipy`, `soxr`, `audio-separator` (locally -- it runs inside the Modal container).

**System dependencies (must already be installed):**
- `ffmpeg` -- required by pydub for MP3 export. Verify: `ffmpeg -version`
- `libsndfile` -- required by soundfile. Verify: `python -c "import soundfile"`

Check both before proceeding. Install via Homebrew if missing:
```bash
brew install ffmpeg libsndfile
```

#### 1c. Create `.gitignore`

```
# Python
__pycache__/
*.pyc
*.pyo
.venv/
*.egg-info/
dist/
build/

# Environment
.env
.env.*

# Runtime data
data/

# IDE
.idea/
.vscode/
*.swp
```

#### 1d. Create `.env`

```
# Modal auth is handled by `modal token new` (writes to ~/.modal.toml)
# No API keys needed for Day 1
```

#### 1e. Create `CLAUDE.md`

Backend-specific conventions: Python 3.11, FastAPI, uv package manager, project structure, how to run the dev server (`uv run uvicorn musicmixer.main:app --reload --port 8000`).

#### 1f. Create `config.py`

Stripped-down `Settings` for Day 1:

```python
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["http://localhost:5173"]

    # File limits
    max_file_size_mb: int = 50
    allowed_extensions: set[str] = {".mp3", ".wav"}

    # Storage
    data_dir: Path = Path("data")

    # Stem separation
    stem_backend: str = "modal"  # "modal" or "local"

    # Output
    output_format: str = "mp3"
    output_bitrate: str = "320k"

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()
```

**Key decision:** `stem_backend` defaults to `"modal"`. Can be overridden in `.env` to `"local"` for development without cloud GPU.

#### 1g. Create `main.py`

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from musicmixer.config import settings
from musicmixer.api import health, remix
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create data directories
    for subdir in ("uploads", "stems", "remixes"):
        (settings.data_dir / subdir).mkdir(parents=True, exist_ok=True)
    logger.info("musicMixer backend started")
    yield
    logger.info("musicMixer backend shutting down")

app = FastAPI(title="musicMixer", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(remix.router, prefix="/api")

# Serve static HTML page
app.mount("/", StaticFiles(directory="static", html=True), name="static")
```

**Gotcha:** The `StaticFiles` mount must come LAST (after all API routes). Otherwise it swallows `/api/*` requests.

#### 1h. Create `health.py`

```python
from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
async def health_check():
    return {"status": "ok"}
```

#### 1i. Verify server starts

```bash
cd /Users/aaron/Projects/musicMixer/backend
uv run uvicorn musicmixer.main:app --reload --port 8000
# Test: curl http://localhost:8000/health -> {"status":"ok"}
```

**Exit:** Server responds to `/health`.

---

### Step 2: Modal + BS-RoFormer Setup

**Prereq:** Step 1 complete (project exists, `modal` package installed)
**Time estimate:** 1-2 hours (time-boxed -- see risk items)

This is the highest-risk item on Day 1. The plan specifies a 2-hour time-box: if Modal setup blocks progress for >2 hours, pivot to local `htdemucs_ft`.

#### 2a. Modal account and CLI auth

```bash
# If not already done:
modal token new
# Follow the browser auth flow
# Verify:
modal profile current
```

#### 2b. Create `services/separation_modal.py`

This file defines a Modal app with a GPU container that runs BS-RoFormer stem separation.

```python
import modal
import io

app = modal.App("musicmixer-separation")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install("audio-separator[gpu]", "torch", "soundfile")
    # Bake model weights into image to eliminate cold-start download
    .run_commands(
        "python -c \"from audio_separator.separator import Separator; "
        "s = Separator(); s.load_model('BS-Roformer-Viperx-1297.ckpt')\""
    )
)

@app.function(image=image, gpu="A10G", timeout=180)
def separate_stems_remote(audio_bytes: bytes, filename: str = "input.wav") -> dict[str, bytes]:
    """Run BS-RoFormer 6-stem separation on cloud GPU.

    Accepts raw audio bytes, returns dict mapping stem name to WAV bytes.
    """
    import tempfile
    import soundfile as sf
    from pathlib import Path
    from audio_separator.separator import Separator

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        input_path = tmpdir / filename
        output_dir = tmpdir / "stems"
        output_dir.mkdir()

        # Write input file
        input_path.write_bytes(audio_bytes)

        # Run separation
        separator = Separator(output_dir=str(output_dir))
        separator.load_model("BS-Roformer-Viperx-1297.ckpt")
        separator.separate(str(input_path))

        # Collect output stems
        stems = {}
        expected_stems = ["vocals", "drums", "bass", "guitar", "piano", "other"]
        for stem_file in output_dir.iterdir():
            if not stem_file.suffix == ".wav":
                continue
            # Map filename to stem name
            name_lower = stem_file.stem.lower()
            for stem_name in expected_stems:
                if stem_name in name_lower:
                    # Re-encode as float32 WAV to preserve precision
                    audio_data, sr = sf.read(str(stem_file), dtype="float32")
                    buf = io.BytesIO()
                    sf.write(buf, audio_data, sr, format="WAV", subtype="FLOAT")
                    stems[stem_name] = buf.getvalue()
                    break

        return stems
```

**Critical gotchas from the plan:**
1. **Verify the correct 6-stem checkpoint.** The `12.9755` checkpoint is 2-stem only. The correct checkpoint for 6-stem is the `Viperx-1297` variant. If `audio-separator`'s model name has changed, check their docs.
2. **Validate float32 WAV output.** The plan warns that default `sf.write` may produce PCM_16. We explicitly use `subtype='FLOAT'`.
3. **Validate 6 stems returned.** The plan says to assert stem count after receiving results.

#### 2c. Create `services/separation.py` (dispatcher)

```python
from pathlib import Path
from typing import Callable
import logging
import io
import soundfile as sf

from musicmixer.config import settings

logger = logging.getLogger(__name__)

def separate_stems(
    audio_path: Path,
    output_dir: Path,
    progress_callback: Callable | None = None,
) -> dict[str, Path]:
    """Separate audio into stems. Returns mapping of stem name to WAV file path.

    Dispatches to Modal (cloud GPU) or local backend based on settings.stem_backend.
    """
    if settings.stem_backend == "modal":
        return _separate_modal(audio_path, output_dir, progress_callback)
    else:
        return _separate_local(audio_path, output_dir, progress_callback)


def _separate_modal(
    audio_path: Path,
    output_dir: Path,
    progress_callback: Callable | None = None,
) -> dict[str, Path]:
    """Separate stems via Modal cloud GPU (BS-RoFormer SW, 6-stem)."""
    from musicmixer.services.separation_modal import separate_stems_remote

    if progress_callback:
        progress_callback("Uploading to cloud GPU...")

    # Read audio file as bytes
    audio_bytes = audio_path.read_bytes()

    # Call Modal function
    stem_bytes_map = separate_stems_remote.remote(
        audio_bytes=audio_bytes,
        filename=audio_path.name,
    )

    if progress_callback:
        progress_callback("Stems received, saving...")

    # Validate stem count
    expected = {"vocals", "drums", "bass", "guitar", "piano", "other"}
    received = set(stem_bytes_map.keys())
    if received != expected:
        missing = expected - received
        extra = received - expected
        raise RuntimeError(
            f"Expected 6 stems ({expected}), got {len(received)} ({received}). "
            f"Missing: {missing}. Extra: {extra}. Check BS-RoFormer checkpoint."
        )

    # Save stems to disk as WAV, validate float32
    output_dir.mkdir(parents=True, exist_ok=True)
    stem_paths = {}
    for stem_name, wav_bytes in stem_bytes_map.items():
        # Validate float32
        info = sf.info(io.BytesIO(wav_bytes))
        if info.subtype != "FLOAT":
            logger.warning(
                f"Stem {stem_name} is {info.subtype}, expected FLOAT. "
                "Precision may be lost."
            )

        out_path = output_dir / f"{stem_name}.wav"
        out_path.write_bytes(wav_bytes)
        stem_paths[stem_name] = out_path

    return stem_paths


def _separate_local(
    audio_path: Path,
    output_dir: Path,
    progress_callback: Callable | None = None,
) -> dict[str, Path]:
    """Separate stems locally using htdemucs_ft (4-stem fallback)."""
    # Placeholder -- implemented in Step 5
    raise NotImplementedError(
        "Local separation not yet implemented. Set stem_backend='modal' or see Step 5."
    )
```

**Key design decision:** The interface returns `dict[str, Path]`. All downstream code works with this mapping. Swapping backends is a one-line config change.

#### 2d. Test Modal separation with one song

```bash
cd /Users/aaron/Projects/musicMixer/backend

# Deploy the Modal app first (builds the container image)
uv run modal deploy src/musicmixer/services/separation_modal.py

# Then test with a real song (replace with an actual file path)
uv run python -c "
from musicmixer.services.separation import separate_stems
from pathlib import Path

stems = separate_stems(
    audio_path=Path('path/to/test_song.mp3'),
    output_dir=Path('data/stems/test'),
)
print(f'Got {len(stems)} stems:')
for name, path in sorted(stems.items()):
    print(f'  {name}: {path} ({path.stat().st_size / 1024:.0f} KB)')
"
```

**Validate:**
- Exactly 6 stems returned (vocals, drums, bass, guitar, piano, other)
- Each stem is a valid WAV file
- Each stem has `subtype='FLOAT'` (32-bit float)
- Total processing time is within ~30-60s for a 3-4 minute song

**If Modal fails (account issues, GPU capacity, wrong checkpoint):** Immediately pivot to local fallback (Step 5). Do not spend more than 2 hours on Modal setup.

**Exit:** One song successfully separated into 6 stems via Modal.

---

### Step 3: Upload Endpoint

**Prereq:** Step 1 complete
**Time estimate:** 20-30 minutes

#### 3a. Create `api/remix.py`

Day 1 version: synchronous. Accepts two files, validates extension only, saves to disk, runs pipeline inline, returns session ID. No processing lock, no ThreadPoolExecutor, no SSE.

```python
import uuid
import logging
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse

from musicmixer.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/remix")
async def create_remix(
    song_a: UploadFile = File(...),
    song_b: UploadFile = File(...),
    prompt: str = Form(""),
):
    """Accept two songs, separate stems, overlay, return session ID.

    Day 1: Synchronous. Blocks until remix is complete.
    """
    # Validate extensions
    for label, file in [("song_a", song_a), ("song_b", song_b)]:
        ext = Path(file.filename or "").suffix.lower()
        if ext not in settings.allowed_extensions:
            raise HTTPException(
                422,
                f"Invalid file type for {label}: '{ext}'. "
                f"Allowed: {settings.allowed_extensions}"
            )

    # Generate session ID
    session_id = str(uuid.uuid4())

    # Save uploaded files
    upload_dir = settings.data_dir / "uploads" / session_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    song_a_ext = Path(song_a.filename or "song_a.mp3").suffix.lower()
    song_b_ext = Path(song_b.filename or "song_b.mp3").suffix.lower()

    song_a_path = upload_dir / f"song_a{song_a_ext}"
    song_b_path = upload_dir / f"song_b{song_b_ext}"

    song_a_path.write_bytes(await song_a.read())
    song_b_path.write_bytes(await song_b.read())

    logger.info(f"Session {session_id}: saved uploads ({song_a_path.name}, {song_b_path.name})")

    # Run pipeline synchronously (Day 1 -- no background processing)
    try:
        from musicmixer.services.pipeline_day1 import run_pipeline_sync
        remix_path = run_pipeline_sync(session_id, song_a_path, song_b_path)
        logger.info(f"Session {session_id}: remix complete at {remix_path}")
    except Exception as e:
        logger.exception(f"Session {session_id}: pipeline failed")
        raise HTTPException(500, f"Remix failed: {str(e)}")

    return {"session_id": session_id}


@router.get("/remix/{session_id}/audio")
async def get_audio(session_id: str):
    """Serve the rendered remix MP3."""
    remix_path = settings.data_dir / "remixes" / session_id / "remix.mp3"
    if not remix_path.exists():
        raise HTTPException(404, "Remix not found")
    return FileResponse(remix_path, media_type="audio/mpeg", filename="remix.mp3")
```

**Key decisions:**
- **Synchronous for Day 1.** The POST blocks until the remix is ready. The HTML page handles this with a loading spinner.
- **Extension-only validation.** Skip magic bytes, MIME check, duration check, file size streaming check. These come on Day 4.
- **`pipeline_day1.py`** is a simplified synchronous pipeline (Step 4). It will be replaced by the full async pipeline on Day 2.
- **Path traversal defense** is not critical for Day 1 (single user, no adversarial input) but we use uuid4 session IDs which are not user-controlled.

**Exit:** Upload endpoint saves files and calls pipeline.

---

### Step 4: Synchronous Pipeline + Mixer

**Prereq:** Steps 2 and 3 complete
**Time estimate:** 30-45 minutes

#### 4a. Create `services/pipeline_day1.py`

The Day 1 pipeline: separate both songs, overlay vocals + instrumentals, export MP3. No analysis, no LLM, no tempo/key matching.

```python
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from musicmixer.config import settings
from musicmixer.services.separation import separate_stems
from musicmixer.services.mixer import overlay_and_export

logger = logging.getLogger(__name__)


def run_pipeline_sync(
    session_id: str,
    song_a_path: Path,
    song_b_path: Path,
) -> Path:
    """Day 1 synchronous pipeline.

    1. Separate both songs into stems (parallel via Modal)
    2. Overlay Song A vocals + Song B instrumentals
    3. Export MP3
    """
    stems_dir = settings.data_dir / "stems" / session_id
    remix_dir = settings.data_dir / "remixes" / session_id
    remix_dir.mkdir(parents=True, exist_ok=True)
    output_path = remix_dir / "remix.mp3"

    # Step 1: Separate both songs (in parallel)
    logger.info(f"[{session_id}] Separating stems...")

    song_a_stems_dir = stems_dir / "song_a"
    song_b_stems_dir = stems_dir / "song_b"

    # Run both separations concurrently -- Modal supports parallel calls
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(
            separate_stems, song_a_path, song_a_stems_dir
        )
        future_b = executor.submit(
            separate_stems, song_b_path, song_b_stems_dir
        )
        song_a_stems = future_a.result(timeout=300)
        song_b_stems = future_b.result(timeout=300)

    logger.info(
        f"[{session_id}] Separation complete. "
        f"Song A: {list(song_a_stems.keys())}, Song B: {list(song_b_stems.keys())}"
    )

    # Step 2+3: Overlay and export
    logger.info(f"[{session_id}] Mixing and exporting...")
    overlay_and_export(
        vocal_stems={"vocals": song_a_stems["vocals"]},
        instrumental_stems={
            k: song_b_stems[k]
            for k in ["drums", "bass", "guitar", "piano", "other"]
        },
        output_path=output_path,
    )

    logger.info(f"[{session_id}] Remix exported to {output_path}")
    return output_path
```

**Key decisions:**
- **Parallel separation** via `ThreadPoolExecutor(max_workers=2)`. The plan explicitly calls this out: "Song A and Song B stem separation should be run concurrently... This cuts separation time roughly in half."
- **Hardcoded assignment:** vocals from Song A, all 5 instrumental stems from Song B. No LLM, no user prompt interpretation.
- **300-second timeout** on each separation future (5 minutes, well above the expected 30-60s, to account for cold start + slow uploads).

#### 4b. Create `services/mixer.py`

Hardcoded overlay: sum vocals + instrumentals, export MP3 via ffmpeg.

```python
import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def overlay_and_export(
    vocal_stems: dict[str, Path],
    instrumental_stems: dict[str, Path],
    output_path: Path,
    target_sr: int = 44100,
) -> Path:
    """Overlay vocal and instrumental stems, export as MP3.

    Day 1: No tempo matching, no key matching, no LUFS normalization.
    Just load, standardize to same length, sum, and export.
    """
    # Load all stems as float32
    all_audio = []
    max_length = 0

    for stem_name, stem_path in {**vocal_stems, **instrumental_stems}.items():
        audio, sr = sf.read(str(stem_path), dtype="float32")

        # Resample if needed (should be 44.1kHz from BS-RoFormer, but verify)
        if sr != target_sr:
            import librosa
            logger.warning(f"Stem {stem_name} at {sr}Hz, resampling to {target_sr}Hz")
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

        # Ensure stereo
        if audio.ndim == 1:
            audio = np.column_stack([audio, audio])

        all_audio.append((stem_name, audio))
        max_length = max(max_length, len(audio))

    # Pad shorter stems to match longest
    padded = []
    for stem_name, audio in all_audio:
        if len(audio) < max_length:
            pad_length = max_length - len(audio)
            audio = np.pad(audio, ((0, pad_length), (0, 0)), mode="constant")
        padded.append(audio)

    # Sum all stems (float32 -- no clipping during summation)
    mixed = np.sum(padded, axis=0)

    # Basic peak normalization to prevent clipping (no LUFS -- Day 1)
    peak = np.max(np.abs(mixed))
    if peak > 0.95:
        mixed = mixed * (0.95 / peak)

    # Export via ffmpeg (float32 WAV -> MP3 320kbps)
    # DO NOT use pydub for export -- it quantizes to 16-bit internally
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = Path(tmp.name)

    sf.write(str(tmp_wav), mixed, target_sr, subtype="FLOAT")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(tmp_wav),
        "-codec:a", "libmp3lame",
        "-b:a", "320k",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=120)
    if result.returncode != 0:
        logger.error(f"ffmpeg failed: {result.stderr.decode()}")
        raise RuntimeError(f"ffmpeg export failed: {result.stderr.decode()[:500]}")

    # Clean up temp WAV
    tmp_wav.unlink(missing_ok=True)

    logger.info(f"Exported remix: {output_path} ({output_path.stat().st_size / 1024:.0f} KB)")
    return output_path
```

**Key decisions from the plan:**
- **Sum in numpy (float32), not pydub overlay.** The plan explicitly warns: "Sum all processed stems in numpy (32-bit float) rather than pydub overlay -- avoids pydub's internal 16-bit clipping during summation."
- **Export via ffmpeg subprocess, not pydub.** The plan warns: "Do NOT use pydub for export -- pydub's AudioSegment quantizes to 16-bit integers internally, destroying the 32-bit float headroom."
- **Basic peak normalization** (not LUFS) as a stopgap. If the sum peaks above 0.95, scale down. Proper LUFS normalization comes on Day 2.
- **Pad shorter stems** to match the longest. Since there is no tempo matching, the vocal track and instrumental track may be different lengths. Padding prevents index errors; the shorter track fades to silence naturally.

**Exit:** Pipeline produces an MP3 file from separated stems.

---

### Step 5: Local Fallback (htdemucs_ft)

**Prereq:** Step 1 complete. **Only implement if Modal is blocked.**
**Time estimate:** 30-45 minutes (if needed)

#### 5a. Install local dependencies

```bash
cd /Users/aaron/Projects/musicMixer/backend

# audio-separator requires torch
uv add audio-separator torch --extra-index-url https://download.pytorch.org/whl/cpu
```

**Warning:** This pulls in PyTorch (~2 GB). Only install if Modal is not working.

#### 5b. Create `services/separation_local.py`

```python
import logging
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


def separate_stems_local(
    audio_path: Path,
    output_dir: Path,
    progress_callback: Callable | None = None,
) -> dict[str, Path]:
    """Separate stems locally using htdemucs_ft (4-stem).

    Returns dict with keys: vocals, drums, bass, other.
    guitar and piano are None (htdemucs_ft only produces 4 stems).
    """
    from audio_separator.separator import Separator

    output_dir.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback("Running local stem separation (htdemucs_ft)...")

    separator = Separator(output_dir=str(output_dir))
    separator.load_model("htdemucs_ft")
    separator.separate(str(audio_path))

    # Map output files to stem names
    stems = {}
    expected_4stem = ["vocals", "drums", "bass", "other"]
    for stem_file in output_dir.iterdir():
        if not stem_file.suffix == ".wav":
            continue
        name_lower = stem_file.stem.lower()
        for stem_name in expected_4stem:
            if stem_name in name_lower:
                stems[stem_name] = stem_file
                break

    # 4-stem model: guitar and piano are not separated
    # Return None for these -- downstream code treats None as silent
    stems.setdefault("guitar", None)
    stems.setdefault("piano", None)

    logger.info(f"Local separation complete: {list(stems.keys())}")
    return stems
```

Then update `separation.py`'s `_separate_local()` to call `separate_stems_local()` instead of raising `NotImplementedError`.

**Key from the plan:** "When using 4-stem fallback, return `{'vocals': Path, 'drums': Path, 'bass': Path, 'guitar': None, 'piano': None, 'other': Path}` -- downstream code handles `None` stems gracefully (treated as silent)."

Update `mixer.py` to skip `None` stems:
```python
# In overlay_and_export, when iterating stems:
for stem_name, stem_path in {**vocal_stems, **instrumental_stems}.items():
    if stem_path is None:
        continue  # Skip missing stems (4-stem fallback)
    ...
```

Also set `stem_backend` in `.env`:
```
STEM_BACKEND=local
```

**Exit:** Local fallback produces 4 stems; mixer handles missing guitar/piano gracefully.

---

### Step 6: Static HTML Test Page

**Prereq:** Steps 3 and 4 complete
**Time estimate:** 15-20 minutes

#### 6a. Create `static/index.html`

A single HTML page with:
- Two file inputs (Song A, Song B)
- A text input for the prompt (unused on Day 1, but present for the form contract)
- A submit button
- A loading spinner / status text
- An `<audio>` element for playback

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>musicMixer - Day 1</title>
    <style>
        /* Minimal styling -- replaced by React + Tailwind on Day 3 */
        body { font-family: system-ui, sans-serif; max-width: 600px; margin: 40px auto; padding: 0 20px; }
        h1 { margin-bottom: 4px; }
        .subtitle { color: #666; margin-top: 0; }
        label { display: block; margin-top: 16px; font-weight: 600; }
        input[type="file"], input[type="text"] { margin-top: 4px; }
        button { margin-top: 20px; padding: 10px 24px; font-size: 16px; cursor: pointer; }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
        #status { margin-top: 16px; color: #444; }
        #player { margin-top: 20px; display: none; }
        audio { width: 100%; }
        .error { color: #c00; }
    </style>
</head>
<body>
    <h1>musicMixer</h1>
    <p class="subtitle">Day 1: Hardcoded Frankenmix</p>

    <form id="remix-form">
        <label for="song_a">Song A (vocals source):</label>
        <input type="file" id="song_a" name="song_a" accept=".mp3,.wav" required>

        <label for="song_b">Song B (instrumentals source):</label>
        <input type="file" id="song_b" name="song_b" accept=".mp3,.wav" required>

        <label for="prompt">Prompt (not used yet):</label>
        <input type="text" id="prompt" name="prompt" placeholder="Describe your remix..."
               style="width: 100%; padding: 8px;">

        <button type="submit" id="submit-btn">Create Remix</button>
    </form>

    <div id="status"></div>

    <div id="player">
        <h3>Your Remix</h3>
        <audio id="audio" controls></audio>
    </div>

    <script>
        const form = document.getElementById('remix-form');
        const status = document.getElementById('status');
        const player = document.getElementById('player');
        const audio = document.getElementById('audio');
        const submitBtn = document.getElementById('submit-btn');

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            player.style.display = 'none';
            status.className = '';
            status.textContent = 'Uploading and processing... (this takes 1-2 minutes)';
            submitBtn.disabled = true;

            const formData = new FormData();
            formData.append('song_a', document.getElementById('song_a').files[0]);
            formData.append('song_b', document.getElementById('song_b').files[0]);
            formData.append('prompt', document.getElementById('prompt').value || '');

            try {
                const response = await fetch('/api/remix', {
                    method: 'POST',
                    body: formData,
                });

                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.detail || `HTTP ${response.status}`);
                }

                const data = await response.json();
                const sessionId = data.session_id;

                status.textContent = 'Remix complete! Loading audio...';

                // Set audio source
                audio.src = `/api/remix/${sessionId}/audio`;
                player.style.display = 'block';
                status.textContent = 'Press play to hear your remix.';
            } catch (err) {
                status.textContent = `Error: ${err.message}`;
                status.className = 'error';
            } finally {
                submitBtn.disabled = false;
            }
        });
    </script>
</body>
</html>
```

**Key decisions:**
- **No polling.** Since the Day 1 pipeline is synchronous, the `fetch('/api/remix')` call blocks until the remix is done. The loading state is handled by the button disable + status text.
- **No React, no Vite, no build step.** Just a single HTML file served by FastAPI.
- **Prompt input is present** but unused. Maintains the form contract for Day 2+.

**Exit:** HTML page loads at `http://localhost:8000/`.

---

### Step 7: End-to-End Test

**Prereq:** All previous steps complete
**Time estimate:** 30-60 minutes (includes debugging)

#### 7a. Start the server

```bash
cd /Users/aaron/Projects/musicMixer/backend
uv run uvicorn musicmixer.main:app --reload --port 8000
```

#### 7b. Manual test via browser

1. Open `http://localhost:8000/`
2. Select two MP3/WAV files (use real songs, 3-4 minutes each)
3. Click "Create Remix"
4. Wait 1-2 minutes (stem separation on Modal)
5. Audio player should appear
6. Press play -- you should hear Song A's vocals over Song B's instrumentals

#### 7c. Manual test via curl

```bash
curl -X POST http://localhost:8000/api/remix \
  -F "song_a=@/path/to/song_a.mp3" \
  -F "song_b=@/path/to/song_b.mp3" \
  -F "prompt=test"

# Returns: {"session_id":"<uuid>"}
# Then:
curl http://localhost:8000/api/remix/<uuid>/audio --output remix.mp3
# Play remix.mp3 in any audio player
```

#### 7d. Verify data directories

```bash
ls -la data/uploads/*/
ls -la data/stems/*/
ls -la data/remixes/*/
```

Each session should have:
- `data/uploads/{session_id}/song_a.mp3` and `song_b.mp3`
- `data/stems/{session_id}/song_a/` with 6 WAV stems
- `data/stems/{session_id}/song_b/` with 6 WAV stems
- `data/remixes/{session_id}/remix.mp3`

---

### Step 8: Install rubberband (Prep for Day 2)

**Prereq:** None (can run in parallel with other steps)
**Time estimate:** 5-10 minutes

The plan says to verify rubberband on Day 1 afternoon so it is ready for Day 2.

```bash
brew install rubberband

# Verify version is 3.x (needed for -3 / --fine flag)
rubberband --version

# Quick test: stretch a WAV file to 90% speed
rubberband -t 1.1 data/stems/test/song_a/vocals.wav /tmp/stretched_test.wav
# Play /tmp/stretched_test.wav to verify it sounds reasonable
```

If rubberband is v2.x instead of v3.x, note this for Day 2: use `--crisp 5` instead of `-3`.

---

## Exit Criteria Checklist

When Day 1 is done, all of these must be true:

- [ ] `curl http://localhost:8000/health` returns `{"status":"ok"}`
- [ ] `POST /api/remix` with two MP3 files returns a `session_id`
- [ ] Both songs are separated into 6 stems via Modal (or 4 stems via local fallback)
- [ ] Stems are validated: correct count, float32 WAV format
- [ ] `GET /api/remix/{id}/audio` serves an MP3 file
- [ ] The MP3 file plays back in a browser and you can hear:
  - Vocals from Song A
  - Instrumentals (drums, bass, guitar, piano, other) from Song B
- [ ] The HTML page at `http://localhost:8000/` allows upload and playback
- [ ] `rubberband --version` returns 3.x (ready for Day 2)
- [ ] Switching `STEM_BACKEND=local` in `.env` uses htdemucs_ft (4-stem) fallback (if local fallback was implemented)

---

## Risk Items

### Risk 1: Modal Setup Fails (HIGH)

**What could go wrong:**
- Modal account approval delays
- GPU capacity issues (A10G not available in your region)
- `audio-separator` model name mismatch -- the BS-RoFormer checkpoint name may differ from what the plan assumes (`BS-Roformer-Viperx-1297.ckpt`)
- Wrong checkpoint produces 2 stems instead of 6 (the plan warns about this explicitly)
- Container image build fails (dependency conflicts between torch, audio-separator, CUDA)
- Cold start exceeds timeout (first run may take 60-90s)

**Mitigation:**
- **2-hour time-box.** If Modal is not working after 2 hours, switch to local `htdemucs_ft` (Step 5). You get 4 stems instead of 6, but the pipeline still works.
- Test with one song before wiring into the full pipeline.
- Check `audio-separator` docs for the exact model name. Run `python -c "from audio_separator.separator import Separator; print(Separator.list_models())"` inside the Modal container to see available models.
- If 6-stem is problematic, fall back to the standard BS-RoFormer 2-stem (vocals + accompaniment) as an intermediate step. You can split "accompaniment" into individual instruments later.

### Risk 2: Long Synchronous Request Timeout (MEDIUM)

**What could go wrong:**
- Stem separation takes 60-120s. The synchronous POST may hit browser or proxy timeouts.
- `fetch()` has no default timeout, but the browser may close the connection after ~2-5 minutes of no data.

**Mitigation:**
- This is a Day 1 limitation. The async pipeline (Day 2) solves this properly.
- If timeouts occur, add a simple polling pattern: POST returns immediately with session ID, HTML page polls `GET /api/remix/{id}/audio` until it returns 200.

### Risk 3: Disk Space (LOW)

**What could go wrong:**
- Each song produces ~240MB of stems (6 stems x 40MB each). Two songs = ~480MB stems + ~10MB uploads + ~10MB remix = ~500MB per session.
- Repeated testing without cleanup fills the disk.

**Mitigation:**
- Manually clean `data/` between test runs: `rm -rf data/stems/* data/uploads/* data/remixes/*`
- TTL cleanup comes on Day 4. For now, manual cleanup is fine.

### Risk 4: ffmpeg Not Installed (LOW)

**What could go wrong:**
- `pydub` and the MP3 export subprocess both require `ffmpeg`. If it is not on PATH, the mixer silently fails or throws a cryptic error.

**Mitigation:**
- Check `ffmpeg -version` before starting. Install via `brew install ffmpeg` if missing.
- The mixer script raises a clear error if the ffmpeg subprocess fails.

---

## Dependencies Summary

### Python packages (via `uv add`)

```bash
# Day 1 core
uv add fastapi 'uvicorn[standard]' python-multipart pydantic-settings
uv add modal soundfile pydub numpy

# Only if using local fallback (Step 5):
uv add audio-separator torch --extra-index-url https://download.pytorch.org/whl/cpu
```

### System dependencies (via `brew install`)

```bash
brew install ffmpeg libsndfile rubberband
```

### Modal setup

```bash
modal token new     # One-time auth
modal deploy src/musicmixer/services/separation_modal.py  # Build + deploy container
```

---

## What Day 2 Builds On

Day 2 ("The Real Remix") picks up from here and adds:

1. **Async processing:** `ThreadPoolExecutor` + `threading.Queue` -- move pipeline to background thread
2. **SSE progress:** `GET /api/remix/{id}/progress` streaming events
3. **Processing lock:** One remix at a time, 429 on busy
4. **BPM detection:** `librosa.beat.beat_track` + reconciliation
5. **Tempo matching:** `pyrubberband` / rubberband CLI (already installed from Step 8)
6. **LUFS normalization:** `pyloudnorm` on final mix
7. **Section-based arrangement:** Deterministic fallback plan (3 sections: intro/main/outro)
8. **Updated HTML page:** `EventSource` for SSE + progress bar

The Day 1 code (separation, mixer, upload endpoint) carries forward directly. `pipeline_day1.py` is replaced by the full `pipeline.py` with async support.
