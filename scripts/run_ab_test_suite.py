"""A/B test suite for sound quality enhancement flags.

Downloads YouTube audio pairs, runs the remix pipeline with different AB flag
configurations, measures LUFS and true-peak on outputs, and saves labeled
results to mashupTests/.

Usage (from workspace root or backend dir):
  cd backend && uv run python ../scripts/run_ab_test_suite.py
  cd backend && uv run python ../scripts/run_ab_test_suite.py --mode sweep
  cd backend && uv run python ../scripts/run_ab_test_suite.py --pairs 1,3
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — must happen before any musicmixer imports
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ab-test-suite")


# ---------------------------------------------------------------------------
# Step 0: Preflight validation
# ---------------------------------------------------------------------------

def preflight() -> None:
    """Assert prerequisites exist before starting the suite."""
    # Check AB flags exist on settings
    from musicmixer.config import settings

    required_flags = [
        "ab_control_day3",
        "ab_autolvl_tune_v1",
        "ab_vocal_makeup_v1",
        "ab_mp3_export_path_v1",
        "ab_per_stem_eq_v1",
        "ab_resonance_detection_v1",
        "ab_multiband_comp_v1",
        "ab_static_mastering_v1",
    ]
    missing = [f for f in required_flags if not hasattr(settings, f)]
    if missing:
        log.error(
            "Missing AB flags on settings: %s. "
            "Implement sound-quality-enhancement-plan prerequisites first.",
            ", ".join(missing),
        )
        sys.exit(1)

    # Check yt-dlp is available
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        log.error(
            "yt_dlp not importable. "
            "Implement the YouTube input feature first (see youtube-input-plan.md)."
        )
        sys.exit(1)

    log.info("Preflight OK — all flags present, yt_dlp available")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="A/B test suite for sound quality enhancement flags.",
    )
    parser.add_argument(
        "--mode",
        choices=["compare", "sweep"],
        default="compare",
        help=(
            "compare: all-off vs all-on (2 runs/pair). "
            "sweep: per-flag isolation (5 runs/pair). Default: compare."
        ),
    )
    parser.add_argument(
        "--pairs",
        default=None,
        help="Comma-separated pair numbers for fast iteration (e.g., --pairs 1,3).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Test pairs
# ---------------------------------------------------------------------------

TEST_PAIRS = [
    {
        "name": "1-biggie-althea",
        "url_a": "https://www.youtube.com/watch?v=eaPzCHEQExs",
        "url_b": "https://www.youtube.com/watch?v=ZZNZgtj26Fk",
    },
    {
        "name": "2-ghosttown-khala",
        "url_a": "https://www.youtube.com/watch?v=qAsHVwl-MU4",
        "url_b": "https://www.youtube.com/watch?v=2QxeDecgNWg",
    },
    {
        "name": "3-encore-numb",
        "url_a": "https://www.youtube.com/watch?v=7VksyVUAwi8",
        "url_b": "https://www.youtube.com/watch?v=kXYiU_JCYtU",
    },
    {
        "name": "4-coldplay-daftpunk",
        "url_a": "https://www.youtube.com/watch?v=QtXby3twMmI",
        "url_b": "https://www.youtube.com/watch?v=IluRBvnYMoY",
    },
    {
        "name": "5-doom-scarletbegonias",
        "url_a": "https://www.youtube.com/watch?v=X-YaI5ZkRvw",
        "url_b": "https://www.youtube.com/watch?v=xt4XAz2WZ3Y",
    },
]


# ---------------------------------------------------------------------------
# Flag matrices
# ---------------------------------------------------------------------------

# Baseline flags held at production defaults for ALL runs
_BASELINE_FLAGS = {
    "AB_CONTROL_DAY3": "True",
    "AB_AUTOLVL_TUNE_V1": "True",
    "AB_VOCAL_MAKEUP_V1": "True",
    "AB_MP3_EXPORT_PATH_V1": "True",
}

# New sound-quality flags — the ones under test
_NEW_FLAGS_OFF = {
    "AB_PER_STEM_EQ_V1": "False",
    "AB_RESONANCE_DETECTION_V1": "False",
    "AB_MULTIBAND_COMP_V1": "False",
    "AB_STATIC_MASTERING_V1": "False",
}

_NEW_FLAGS_ON = {
    "AB_PER_STEM_EQ_V1": "True",
    "AB_RESONANCE_DETECTION_V1": "True",
    "AB_MULTIBAND_COMP_V1": "True",
    "AB_STATIC_MASTERING_V1": "True",
}


def _build_compare_variants() -> list[dict]:
    """Return variant configs for compare mode (control vs enhanced)."""
    return [
        {
            "name": "control",
            "flags": {**_BASELINE_FLAGS, **_NEW_FLAGS_OFF},
        },
        {
            "name": "enhanced",
            "flags": {**_BASELINE_FLAGS, **_NEW_FLAGS_ON},
        },
    ]


def _build_sweep_variants() -> list[dict]:
    """Return variant configs for sweep mode (control + 4 per-flag)."""
    control = {
        "name": "control",
        "flags": {**_BASELINE_FLAGS, **_NEW_FLAGS_OFF},
    }
    sweep_eq = {
        "name": "sweep-per-stem-eq",
        "flags": {
            **_BASELINE_FLAGS,
            **_NEW_FLAGS_OFF,
            "AB_PER_STEM_EQ_V1": "True",
        },
    }
    # Resonance detection requires per-stem EQ (dependency)
    sweep_resonance = {
        "name": "sweep-resonance-detection",
        "flags": {
            **_BASELINE_FLAGS,
            **_NEW_FLAGS_OFF,
            "AB_PER_STEM_EQ_V1": "True",
            "AB_RESONANCE_DETECTION_V1": "True",
        },
    }
    sweep_multiband = {
        "name": "sweep-multiband-comp",
        "flags": {
            **_BASELINE_FLAGS,
            **_NEW_FLAGS_OFF,
            "AB_MULTIBAND_COMP_V1": "True",
        },
    }
    sweep_mastering = {
        "name": "sweep-static-mastering",
        "flags": {
            **_BASELINE_FLAGS,
            **_NEW_FLAGS_OFF,
            "AB_STATIC_MASTERING_V1": "True",
        },
    }
    return [control, sweep_eq, sweep_resonance, sweep_multiband, sweep_mastering]


# ---------------------------------------------------------------------------
# Download phase
# ---------------------------------------------------------------------------

async def _download_pair(pair: dict) -> tuple[Path, Path, str, str]:
    """Download both songs for a pair concurrently. Returns (wav_a, wav_b, quality_a, quality_b)."""
    from musicmixer.services.youtube import download_youtube_audio

    dl_dir = REPO_ROOT / "backend" / "data" / "uploads" / f"ab-{pair['name']}"
    dl_dir.mkdir(parents=True, exist_ok=True)

    result_a, result_b = await asyncio.gather(
        download_youtube_audio(pair["url_a"], dl_dir),
        download_youtube_audio(pair["url_b"], dl_dir),
    )

    quality_a = f"youtube-{result_a.source_codec}-{result_a.source_bitrate}kbps"
    quality_b = f"youtube-{result_b.source_codec}-{result_b.source_bitrate}kbps"

    return result_a.wav_path, result_b.wav_path, quality_a, quality_b


def download_pair(pair: dict) -> tuple[Path, Path, str, str] | None:
    """Synchronous wrapper for downloading a pair. Returns None on failure."""
    try:
        return asyncio.run(_download_pair(pair))
    except Exception:
        log.exception("Download failed for pair %s — skipping", pair["name"])
        return None


# ---------------------------------------------------------------------------
# Pipeline invocation
# ---------------------------------------------------------------------------

def run_pipeline_variant(
    pair_name: str,
    variant_name: str,
    wav_a: Path,
    wav_b: Path,
    source_quality_a: str,
    source_quality_b: str,
    flag_env_vars: dict[str, str],
    output_dir: Path,
) -> dict:
    """Run one pipeline variant as a subprocess. Returns a result dict."""
    phase_name = f"{pair_name}-{variant_name}"
    session_id = f"ab-{phase_name}"
    output_mp3 = output_dir / f"{variant_name}.mp3"

    env = {
        **os.environ,
        **flag_env_vars,
        "LYRICS_LOOKUP_ENABLED": "false",
    }

    cmd = [
        "uv", "run", "python", "scripts/run_pipeline_phase.py",
        phase_name, str(wav_a), str(wav_b),
        "--source-quality-a", source_quality_a,
        "--source-quality-b", source_quality_b,
    ]

    log.info("Running %s/%s ...", pair_name, variant_name)
    t0 = time.monotonic()

    try:
        result = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT / "backend"),
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
        )
    except subprocess.TimeoutExpired:
        duration = time.monotonic() - t0
        log.error("%s/%s timed out after %.0fs", pair_name, variant_name, duration)
        return {
            "status": "timeout",
            "duration_s": round(duration, 1),
            "output_path": "",
        }

    duration = time.monotonic() - t0

    if result.returncode != 0:
        log.error(
            "%s/%s failed (exit %d). stderr:\n%s",
            pair_name,
            variant_name,
            result.returncode,
            result.stderr[-2000:] if result.stderr else "(empty)",
        )
        return {
            "status": "failed",
            "duration_s": round(duration, 1),
            "output_path": "",
        }

    # Parse output — run_pipeline_phase.py prints status then remix path
    stdout_lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
    if len(stdout_lines) < 2:
        log.error(
            "%s/%s produced unexpected output: %s",
            pair_name,
            variant_name,
            result.stdout[-500:],
        )
        return {
            "status": "failed",
            "duration_s": round(duration, 1),
            "output_path": "",
        }

    status_line = stdout_lines[-2].strip()
    remix_path_str = stdout_lines[-1].strip()

    if status_line != "complete":
        log.error("%s/%s status: %s (expected 'complete')", pair_name, variant_name, status_line)
        return {
            "status": "failed",
            "duration_s": round(duration, 1),
            "output_path": "",
        }

    # Resolve remix path (may be relative to backend dir)
    remix_path = Path(remix_path_str)
    if not remix_path.is_absolute():
        remix_path = REPO_ROOT / "backend" / remix_path

    if not remix_path.exists():
        log.error("%s/%s remix file not found: %s", pair_name, variant_name, remix_path)
        return {
            "status": "failed",
            "duration_s": round(duration, 1),
            "output_path": "",
        }

    # Copy output to mashupTests
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(remix_path, output_mp3)
    log.info("%s/%s complete in %.1fs -> %s", pair_name, variant_name, duration, output_mp3)

    # Clean pipeline data to save disk space
    stems_dir = REPO_ROOT / "backend" / "data" / "stems" / session_id
    remixes_dir = REPO_ROOT / "backend" / "data" / "remixes" / session_id
    for d in (stems_dir, remixes_dir):
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)

    return {
        "status": "ok",
        "duration_s": round(duration, 1),
        "output_path": str(output_mp3),
    }


# ---------------------------------------------------------------------------
# LUFS / true-peak measurement
# ---------------------------------------------------------------------------

def measure_audio(mp3_path: str) -> dict:
    """Measure integrated LUFS, true-peak dBTP, and crest factor from an MP3."""
    import librosa
    import numpy as np
    import pyloudnorm as pyln
    from musicmixer.services.processor import true_peak

    audio, sr = librosa.load(mp3_path, sr=None, mono=False)

    # Mono guard: if 1D, reshape to (1, samples) so .T produces (samples, 1)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]

    # librosa returns (channels, samples); pyloudnorm expects (samples, channels)
    audio = audio.T

    meter = pyln.Meter(sr)
    lufs = meter.integrated_loudness(audio)

    peak_dbtp = 20 * np.log10(true_peak(audio) + 1e-10)
    crest_factor_db = peak_dbtp - lufs

    return {
        "lufs": round(lufs, 1),
        "peak_dbtp": round(peak_dbtp, 1),
        "crest_factor_db": round(crest_factor_db, 1),
    }


# ---------------------------------------------------------------------------
# Results logging
# ---------------------------------------------------------------------------

RESULTS_HEADER = "mode | pair | variant | status | duration_s | lufs | peak_dbtp | crest_factor_db | output_path"


def write_result_row(
    results_file: Path,
    mode: str,
    pair_name: str,
    variant_name: str,
    run_result: dict,
    measurements: dict | None,
) -> None:
    """Append one pipe-delimited result row."""
    lufs = measurements["lufs"] if measurements else ""
    peak = measurements["peak_dbtp"] if measurements else ""
    crest = measurements["crest_factor_db"] if measurements else ""

    row = (
        f"{mode} | {pair_name} | {variant_name} | {run_result['status']} | "
        f"{run_result['duration_s']} | {lufs} | {peak} | {crest} | "
        f"{run_result['output_path']}"
    )

    with open(results_file, "a") as f:
        f.write(row + "\n")

    log.info("Result: %s", row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    preflight()

    # Determine which pairs to run
    if args.pairs:
        pair_numbers = {int(n.strip()) for n in args.pairs.split(",")}
        pairs = [p for p in TEST_PAIRS if int(p["name"].split("-")[0]) in pair_numbers]
        if not pairs:
            log.error("No matching pairs for --pairs %s", args.pairs)
            return 1
        log.info("Running %d pair(s): %s", len(pairs), [p["name"] for p in pairs])
    else:
        pairs = TEST_PAIRS

    # Build variant list based on mode
    if args.mode == "compare":
        variants = _build_compare_variants()
    else:
        variants = _build_sweep_variants()

    total_runs = len(pairs) * len(variants)
    log.info(
        "Mode: %s | Pairs: %d | Variants/pair: %d | Total runs: %d",
        args.mode,
        len(pairs),
        len(variants),
        total_runs,
    )

    # Clear mashupTests/ at start
    mashup_dir = REPO_ROOT / "mashupTests"
    if mashup_dir.exists():
        shutil.rmtree(mashup_dir)
    mashup_dir.mkdir(parents=True)

    # Initialize results file with header
    results_file = mashup_dir / "results.txt"
    with open(results_file, "w") as f:
        f.write(RESULTS_HEADER + "\n")

    # Empty prompt enforcement — hardcoded in run_pipeline_phase.py (prompt="")
    # and cannot be overridden from this script.

    completed = 0
    failed = 0

    for pair in pairs:
        pair_name = pair["name"]
        log.info("=== Pair: %s ===", pair_name)

        # Download both songs once per pair
        dl_result = download_pair(pair)
        if dl_result is None:
            # Download failed — skip entire pair, log all variants as failed
            for variant in variants:
                write_result_row(
                    results_file,
                    args.mode,
                    pair_name,
                    variant["name"],
                    {"status": "download-failed", "duration_s": 0, "output_path": ""},
                    None,
                )
                failed += 1
            continue

        wav_a, wav_b, quality_a, quality_b = dl_result
        pair_output_dir = mashup_dir / pair_name

        for variant in variants:
            run_result = run_pipeline_variant(
                pair_name=pair_name,
                variant_name=variant["name"],
                wav_a=wav_a,
                wav_b=wav_b,
                source_quality_a=quality_a,
                source_quality_b=quality_b,
                flag_env_vars=variant["flags"],
                output_dir=pair_output_dir,
            )

            # Measure LUFS/peak if successful
            measurements = None
            if run_result["status"] == "ok" and run_result["output_path"]:
                try:
                    measurements = measure_audio(run_result["output_path"])
                except Exception:
                    log.exception(
                        "Measurement failed for %s/%s",
                        pair_name,
                        variant["name"],
                    )

            write_result_row(
                results_file,
                args.mode,
                pair_name,
                variant["name"],
                run_result,
                measurements,
            )

            if run_result["status"] == "ok":
                completed += 1
            else:
                failed += 1

        # Clean downloaded WAVs after all variants complete for this pair
        dl_dir = REPO_ROOT / "backend" / "data" / "uploads" / f"ab-{pair_name}"
        if dl_dir.exists():
            shutil.rmtree(dl_dir, ignore_errors=True)

    log.info(
        "=== Suite complete: %d/%d succeeded, %d failed ===",
        completed,
        completed + failed,
        failed,
    )
    log.info("Results: %s", results_file)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
