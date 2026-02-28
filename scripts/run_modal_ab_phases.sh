#!/usr/bin/env bash
set -euo pipefail

# Baseline pipeline validation script.
#
# Runs the remix pipeline once with default settings (all merged flags
# hardcoded, taste model ON) to verify the pipeline produces valid output.
#
# Previously this script ran 4 phases toggling ab_autolvl_tune_v1,
# ab_vocal_makeup_v1, and ab_mp3_export_path_v1. Those flags were merged
# into the baseline pipeline during flag cleanup (2026-02-28) and no
# longer exist as toggleable flags.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND="$REPO/backend"
OUT="$REPO/notes/audio-ab-renders"
mkdir -p "$OUT" /tmp/uv-cache

AUDIO_A="${1:-$REPO/examples/The Notorious B.I.G. - Hypnotize (Official Audio).mp3}"
AUDIO_B="${2:-$REPO/examples/Althea (2013 Remaster).mp3}"

if [[ ! -f "$AUDIO_A" || ! -f "$AUDIO_B" ]]; then
  echo "Missing input files."
  echo "Usage: $0 [song_a_path] [song_b_path]"
  exit 2
fi

LOG="$OUT/run-$(date +%Y%m%d-%H%M%S).log"
echo "Writing log: $LOG"

phase="baseline"
session_id="ab-$phase"
stems_phase_dir="$BACKEND/data/stems/$session_id"
remix_phase_dir="$BACKEND/data/remixes/$session_id"

echo "=== $phase ===" | tee -a "$LOG"
rm -rf "$stems_phase_dir" "$remix_phase_dir"

tmp_output="$(mktemp)"

if ! (cd "$BACKEND" && \
  UV_CACHE_DIR=/tmp/uv-cache \
  STEM_BACKEND=modal \
  uv run python scripts/run_pipeline_phase.py "$phase" "$AUDIO_A" "$AUDIO_B" 2>&1 | tee -a "$LOG" "$tmp_output"); then
  echo "Pipeline failed. See $LOG" | tee -a "$LOG"
  rm -f "$tmp_output"
  exit 1
fi

# The phase runner prints status and remix path as the last two non-empty lines.
status="$(awk 'NF{prev=curr; curr=$0} END{print prev}' "$tmp_output")"
remix="$(awk 'NF{line=$0} END{print line}' "$tmp_output")"
rm -f "$tmp_output"
if [[ "$status" != "complete" || -z "$remix" ]]; then
  echo "Pipeline failed. See $LOG" | tee -a "$LOG"
  exit 1
fi

if find "$stems_phase_dir" -type f -name '*htdemucs_ft*.wav' | grep -q .; then
  echo "Pipeline used local fallback artifacts under $stems_phase_dir (htdemucs_ft). Aborting." | tee -a "$LOG"
  find "$stems_phase_dir" -type f -name '*htdemucs_ft*.wav' | tee -a "$LOG"
  exit 1
fi

for song in song_a song_b; do
  for stem in vocals drums bass guitar piano other; do
    if [[ ! -f "$stems_phase_dir/$song/$stem.wav" ]]; then
      echo "Missing expected Modal stem: $stems_phase_dir/$song/$stem.wav" | tee -a "$LOG"
      exit 1
    fi
  done
done

remix_src="$remix"
if [[ "$remix_src" != /* ]]; then
  remix_src="$BACKEND/$remix_src"
fi
if [[ ! -f "$remix_src" ]]; then
  echo "Produced missing remix path: $remix_src" | tee -a "$LOG"
  exit 1
fi

cp "$remix_src" "$OUT/$phase.mp3"
echo "saved=$OUT/$phase.mp3" | tee -a "$LOG"

echo "Done. File saved in: $OUT" | tee -a "$LOG"
