---
title: Vocal De-Reverb Plan
revision: 1
status: deferred
---

# Vocal De-Reverb Plan

> Deferred from v1 of the [Sound Quality Enhancement Plan](sound-quality-enhancement-plan.md). Extracted into a standalone plan so it can be picked up independently in a future cycle.

## Problem

Source recordings contain room reverb. Stem separation amplifies this because the model can't cleanly split "dry vocal" from "vocal reverb tail." When the reverb-laden vocal is layered over different instrumentals, the original room sound clashes and compounds -- making the remix sound echoey.

This was identified during the sound quality enhancement work as a real quality issue, but the ML model dependency, additional pipeline step, and post-processing complexity made it a poor fit for the v1 scope. The core enhancement chain (per-stem EQ, multiband compression, static mastering) ships without de-reverb.

## Solution: ML De-Reverb via audio-separator

`audio-separator` (already a project dependency for stem separation) supports de-reverb models. Use **`Reverb_HQ_By_FoxJoy.onnx`** (MDX-Net architecture, 67MB, ONNX runtime already installed). The model auto-downloads on first use.

The API is identical to existing stem separation:

```python
from audio_separator.separator import Separator

separator = Separator()
separator.load_model("Reverb_HQ_By_FoxJoy.onnx")
output_files = separator.separate(vocal_stem_path)
# Returns list of output file paths -- select by filename pattern, NOT by index.
# The dry vocal output contains "No Reverb" or "dry" in its filename.
# Index order may vary across model versions.
dry_vocal_path = next(f for f in output_files if "No Reverb" in f or "dry" in f.lower())
```

### Output File Selection

Always select the de-reverbed output by matching filename substrings (`"No Reverb"` or `"dry"`), not by list index. The order of `output_files` is not guaranteed across model versions. Verify this pattern against the specific `Reverb_HQ_By_FoxJoy.onnx` model's output naming convention during implementation.

> **Review feedback (from sound quality plan review):** An earlier draft used list index (`output_files[0]`) to select the dry vocal. Reviewers flagged this as fragile -- index order can change across model versions. The filename pattern approach above was adopted instead.

## Pipeline Placement

Applied at **step 7.72** -- after vocal bandpass (7.7), before per-stem EQ (7.75). Rationale:

- Bandpass first removes sub-bass rumble and high-freq artifacts, giving the de-reverb model cleaner input
- De-reverb before EQ means EQ is shaping the dry vocal, not the reverb tail
- De-reverb before tempo stretch avoids stretching reverb artifacts

```
 7.7   Vocal bandpass (150Hz-16kHz when ab_per_stem_eq_v1, else 150Hz-8kHz)
 7.72  Vocal de-reverb (ML) .............. [THIS PLAN]
 7.75  Broad preset EQ cuts (Q~1-3)
 8     Compute tempo plan
 9     Rubberband tempo match
```

## Post-De-Reverb Expander

> **Design decision (from review feedback):** An earlier draft specified a hard noise gate (1ms attack, 10:1 ratio) after ML de-reverb. Reviewers flagged this as too aggressive -- abrupt gain changes clip reverb tails mid-decay, creating audible pumping on sustained notes. The recommended alternative is an expander with a gentle ratio that attenuates low-level reverb tails gradually rather than slamming them shut.
>
> Two options were presented:
> 1. **Ship with the expander** (2:1 ratio, 5-10ms attack, 300-500ms release, -50dB threshold) -- gentle enough to preserve vocal naturalness while cleaning up residual reverb tails
> 2. **Ship without any gating** -- let the ML model do all the work, add the expander later only if listening tests reveal problematic residual reverb
>
> **Decision: expander is off by default.** Enable only if listening tests reveal problematic residual reverb tails after ML de-reverb alone. The ML model handles the heavy lifting; the expander is a surgical cleanup tool, not a primary processor.

If enabled, apply a pedalboard `NoiseGate` configured as a gentle expander (low ratio, slow release):

```python
from pedalboard import NoiseGate

# Configured as an expander, NOT a hard gate
expander = NoiseGate(threshold_db=-50.0, ratio=2.0, attack_ms=5.0, release_ms=300.0)
```

Parameters chosen to preserve vocal naturalness:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Threshold** | -50dB | Only acts on very low-level content (reverb tails in gaps) |
| **Ratio** | 2:1 | Gentle expansion -- attenuates but doesn't silence |
| **Attack** | 5ms | Slow enough to avoid clipping vocal onsets |
| **Release** | 300ms | Slow release preserves natural decay of sustained notes |

## Function Signature

```python
def dereverb_vocal(
    vocal_path: Path,
    output_dir: Path,
    sr: int = 44100,
    apply_expander: bool = False,  # off by default
    expander_threshold_db: float = -50.0,
) -> np.ndarray
```

Takes the vocal stem file path (already on disk from separation), runs de-reverb, optionally applies gentle expander, returns float32 stereo numpy array. The reverb output is discarded.

**File I/O note:** audio-separator operates on files, not numpy arrays. The vocal stem WAV is already on disk from step 1 (separation). The de-reverb model writes output files to `output_dir`. We read the dry vocal back with `soundfile.read(dtype="float32")` and clean up intermediate files.

## New Files

| File | Purpose |
|------|---------|
| `backend/src/musicmixer/services/dereverb.py` | Vocal de-reverb via audio-separator + optional expander cleanup |
| `backend/tests/test_dereverb.py` | De-reverb unit tests |

## Files to Modify

| File | Changes |
|------|---------|
| `backend/src/musicmixer/config.py` | Add `ab_vocal_dereverb_v1` feature flag |
| `backend/src/musicmixer/services/pipeline.py` | Wire in step 7.72 (gated by flag) |

## Feature Flag

```python
# In config.py Settings class, alongside existing ab_* flags
ab_vocal_dereverb_v1: bool = False
```

Gated by `ab_vocal_dereverb_v1`. When disabled, the pipeline skips straight to EQ as before.

> **AB test suite integration:** When this feature is re-integrated, `ab_vocal_dereverb_v1` must be added to the flag matrix in `scripts/run_ab_test_suite.py` and `docs/impl/ab-test-suite-plan.md`. The current AB test suite does not include this flag since it was deferred before the suite was finalized.

## Tests (`test_dereverb.py`)

Tests should use synthetic audio (sine waves), following existing patterns in `test_processor.py` and the other sound quality enhancement tests (`test_eq.py`, `test_multiband.py`, `test_mastering.py`).

Test cases:

- **ML de-reverb produces valid output** -- input vocal WAV produces a dry vocal numpy array with correct shape and dtype (float32 stereo)
- **Output file selection uses filename pattern** -- verify the function selects by `"No Reverb"` / `"dry"` substring, not by list index
- **Expander off by default** -- with `apply_expander=False`, output matches ML de-reverb output without additional processing
- **Expander applies when enabled** -- with `apply_expander=True`, output differs from ML-only output (low-level content attenuated)
- **Expander parameters respected** -- custom `expander_threshold_db` value changes behavior
- **Intermediate file cleanup** -- temporary files created by audio-separator are cleaned up after processing
- **Feature flag gating** -- pipeline skips de-reverb entirely when `ab_vocal_dereverb_v1` is False

## Dependencies

No new dependencies required. Both `audio-separator` and `pedalboard` are already in `backend/pyproject.toml` (audio-separator for stem separation, pedalboard for the sound quality enhancement chain).

## Verification

1. Run existing tests to confirm no regressions: `cd backend && uv run pytest`
2. Run de-reverb tests: `uv run pytest tests/test_dereverb.py`
3. Manual listening test: process a vocal stem with known reverb (e.g., Hypnotize vocals) through the de-reverb function and compare before/after
4. Run AB test suite with `ab_vocal_dereverb_v1=True` added to the flag matrix and compare results

## References

- **Parent plan:** [Sound Quality Enhancement Plan](sound-quality-enhancement-plan.md) -- de-reverb was originally Step 3 in this plan, deferred at revision 7
- **AB test suite:** [AB Test Suite Plan](ab-test-suite-plan.md) -- flag matrix needs updating when this feature ships
- **audio-separator docs:** Model list and API for `Reverb_HQ_By_FoxJoy.onnx`
- **pedalboard docs:** `NoiseGate` API for expander configuration
