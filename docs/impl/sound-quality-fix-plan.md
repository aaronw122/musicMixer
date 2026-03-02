---
revision: 4
---

# Sound Quality Enhancement -- Fix Plan

Consolidated from three expert reviews (audio engineer, ML/DSP scientist, senior engineer) on 2026-02-27.

Core problems: enhanced path is 3 dB below LUFS target, EQ cuts too aggressive on some stems, A/B test methodology invalid due to nondeterministic vocal assignment.

---

## P0: Must Fix Before Next A/B Test

### Fix 1: Add post-limiter LUFS correction in mastering chain

The single highest-impact fix. The constrained normalizer pushes to -12 LUFS with 3 dB headroom, but the limiter then eats 2-3 dB of integrated loudness pulling peaks back down. The chain has no feedback loop -- it normalizes, limits, and hopes.

**Scope:** This correction loop applies ONLY to the static mastering branch (`ab_static_mastering_v1=True`). The standard mastering path uses a different limiter headroom strategy and does not exhibit the same LUFS shortfall.

**Symptom:** Enhanced output at -15.0 LUFS vs -12.0 target (Pair 2). 3 dB shortfall = ~40% perceived loudness reduction.

**Root cause:** `lufs_normalize_constrained` with `headroom_db=3.0` lets peaks overshoot by 3 dB. The limiter pulls those peaks back to -1.0 dBTP, but the gain reduction also pulls integrated loudness below target. No measurement or correction happens after the limiter.

**File:** `/Users/aaron/Projects/musicMixer/backend/src/musicmixer/services/mastering.py`

- [x] Move the LUFS correction loop to the **pipeline level** (`pipeline.py`), AFTER the safety soft clip — not inside `master_static`. `master_static` does its job (normalize + limit + return). Then the pipeline measures final LUFS after the soft clip, applies correction, and re-limits. This is the standard iterate-and-converge pattern and avoids the soft clip eating the correction.

  **Architecture:** `master_static(audio)` → `safety_soft_clip(audio)` → **LUFS correction loop (new, below)** → final output.

  Add the following in `pipeline.py`, after the safety soft clip call on the mastered audio:

  ```python
  # --- Post-soft-clip LUFS correction (iterate-and-converge) ---
  # Scope: static mastering branch only (ab_static_mastering_v1=True).
  # master_static normalizes + limits, but the limiter eats 2-3 dB of
  # integrated loudness. The soft clip shaves a bit more. This loop
  # measures the actual LUFS and applies a bounded correction + a light
  # second limiter pass to catch any re-introduced peaks.
  TARGET_MASTER_LUFS = -12.0  # pipeline-level constant; define once, reference everywhere
  _meter = pyloudnorm.Meter(sr)
  _meas = np.column_stack([audio, audio]) if audio.ndim == 1 else audio
  post_clip_lufs = _meter.integrated_loudness(_meas)

  if post_clip_lufs > -70.0 and post_clip_lufs < TARGET_MASTER_LUFS - 1.0:
      correction_db = TARGET_MASTER_LUFS - post_clip_lufs
      correction_db = min(correction_db, 3.0)  # safety cap: never boost more than 3 dB

      # Apply correction unconditionally — do NOT cap by peak headroom.
      # The second limiter pass below will handle any peaks that exceed ceiling.
      audio = audio * (10 ** (correction_db / 20.0))
      logger.info(
          "Pipeline: post-soft-clip LUFS correction +%.1f dB (%.1f -> ~%.1f LUFS)",
          correction_db, post_clip_lufs, post_clip_lufs + correction_db,
      )

      # Second (lighter) limiter pass to catch re-introduced peaks.
      # Shorter lookahead = transparent on transients, only engages on
      # the peaks we just pushed up.
      # 50ms release avoids inter-transient pumping on dense material;
      # the second limiter only catches peaks from a linear gain boost
      # and does not need aggressive recovery.
      audio = true_peak_limit(
          audio, sr,
          ceiling_dbtp=-1.0,
          lookahead_ms=1.5,   # 1-2 ms range
          release_ms=50.0,    # moderate release to avoid pumping
      )

      # Log final LUFS for verification
      _meas_final = np.column_stack([audio, audio]) if audio.ndim == 1 else audio
      final_lufs = _meter.integrated_loudness(_meas_final)
      logger.info("Pipeline: final LUFS after correction + re-limit: %.1f", final_lufs)
  ```

  **Note:** `pyloudnorm` and `true_peak` must be available at module level in `pipeline.py`. Add `import pyloudnorm` to the module-level imports, and ensure `true_peak_limit` is already imported from `musicmixer.services.processor` (it should be — verify the existing import line). `numpy` is already imported in both `mastering.py` (line 16) and `pipeline.py`.

- [ ] Verify: Run Pair 2 with all enhancement flags on. Enhanced output LUFS should be within 1.0 dB of -12.0 target (i.e., -13.0 to -11.0 LUFS)

### Fix 2: Pin vocal/instrumental assignment in A/B test suite

Without this, different runs of the same pair can get different vocal sources, making the comparison meaningless (Pair 1 was already invalidated by this).

**Symptom:** Pair 1 had different vocal assignments between control and enhanced runs.

**Root cause:** `run_pipeline_phase.py` passes `prompt=""` to `run_pipeline()`, which hits the deterministic fallback (always `vocal_source="song_a"`). But the AB test suite passes `--prompt` with a real prompt to the subprocess -- however `run_pipeline_phase.py` does NOT accept a `--prompt` argument, so the prompt is silently ignored. The nondeterminism must have come from the interpreter being called with a non-empty prompt in a different code path, or from a previous version of the phase script. Regardless, the fix is to make the assignment explicitly deterministic.

**Option A (recommended): Add `--force-vocal-source` to `run_pipeline_phase.py` and `run_pipeline()`**

**File:** `/Users/aaron/Projects/musicMixer/backend/scripts/run_pipeline_phase.py`

- [x] Add `--prompt` argument to argparse (it is currently passed by the AB suite but silently dropped by argparse as an error or by `parse_known_args`)
- [x] Add `--force-vocal-source` argument (choices: `song_a`, `song_b`)
- [x] Pass `prompt=args.prompt` to `run_pipeline()` instead of hardcoded `""`
- [x] Pass `force_vocal_source=args.force_vocal_source` to `run_pipeline()`

**File:** `/Users/aaron/Projects/musicMixer/backend/src/musicmixer/services/pipeline.py`

- [x] Add `force_vocal_source: str | None = None` parameter to `run_pipeline()` (line 43)
- [x] After `interpret_prompt` returns `plan` (line 366), if `force_vocal_source` is not None, override: `plan.vocal_source = force_vocal_source`
- [x] Log the override: `logger.info("Session %s: Forced vocal_source=%s", session_id, force_vocal_source)`

**File:** `/Users/aaron/Projects/musicMixer/scripts/run_ab_test_suite.py`

- [x] In `run_pipeline_variant()` (line 282), add `"--force-vocal-source", "song_a"` to the subprocess command
- [ ] Each test pair definition should include a `vocal_source` key (all currently `"song_a"` per the prompts)
- [ ] Verify: Run Pair 1 twice. Both runs should produce identical `vocal_source` in logs

### Fix 3: Reduce EQ aggressiveness on drums and bass

Cumulative EQ cuts of up to -3 dB per band across all stems are not compensated before the mix bus. This contributes to the loudness shortfall and thins out snare body and bass definition.

**Symptom:** Enhanced mix sounds thin, especially snare and bass guitar.

**Root cause:** Plan specified aggressive cuts; code correctly implements the plan, but the plan's assumptions about appropriate cut depths were wrong for post-separation stems.

**File:** `/Users/aaron/Projects/musicMixer/backend/src/musicmixer/services/eq.py`

- [x] Line 63: Drums 400 Hz cut: change `"gain_db": -3.0` to `"gain_db": -1.5` (preserves snare body)
- [x] Line 72: Bass 800 Hz cut: change `"gain_db": -3.0` to `"gain_db": -2.0` (preserves bass definition)
- [x] Line 56: Vocals 250 Hz cut: change `"gain_db": -2.5` to `"gain_db": -1.5` (preserves male vocal warmth)
- [x] Line 70: Bass 60 Hz: **remove the `PeakFilter(60, 0.75, 0.0)` line entirely**. A `PeakFilter` with 0 dB gain is NOT a true no-op — it still applies phase response from the biquad filter, which can smear low-frequency transients. If per-stem bass fundamental shaping is desired in the future, use a small positive gain (e.g., +0.5 dB) so the filter is doing intentional work. For now, remove it and let the multiband compressor handle low-end balance
- [ ] Verify: Run EQ unit tests. Compare enhanced Pair 2 output LUFS -- should be closer to target even without Fix 1 (less cumulative level loss from cuts)

### Fix 4: Investigate and fix Pair 3 timeout

The enhanced pipeline timed out at 900s for Pair 3 (encore-numb). Control completed in 240.9s. A 3.75x slowdown suggests something is hanging, not just slow.

**Symptom:** Pair 3 enhanced variant hits 900s timeout.

**Root cause (hypothesized):** The material may cause `pyloudnorm` to return extreme LUFS values, triggering pathological behavior in `true_peak_limit` (slow convergence with many near-ceiling peaks) or the multiband LUFS restoration applying huge gain that the limiter then fights.

- [x] Add per-step timing logs to `master_static` in `mastering.py` -- log time before and after each of the 3 steps (LPF, LUFS normalize, limiter)
- [x] Add per-step timing to `multiband_compress` in `multiband.py` -- log time for: LUFS measurement, band splitting, per-band compression (each), recombination, output LUFS measurement, gain restoration
- [x] Add a per-step timeout guard in `pipeline.py`: if any single DSP step exceeds 120s, log an error and **skip the timed-out step**, continuing on the enhanced path with the pre-step signal state. Do NOT switch to the control mastering chain mid-enhanced-path — that creates an untested hybrid signal flow where control-chain assumptions (e.g., no multiband, different EQ) receive enhanced-path audio. Skipping the step and continuing is safer and keeps the signal flow coherent
- [ ] Re-run Pair 3 with verbose logging to identify which step hangs
- [ ] Verify: Pair 3 enhanced completes within 600s (2.5x control, generous tolerance)

---

## P1: Fix Soon

### Fix 5: Increase auto-leveler max_boost when multiband is active

The auto-leveler with `max_boost_db=1.0` cannot compensate for cumulative level loss from EQ cuts + multiband spectral rebalancing. The control path allows 2.0 dB boost; the enhanced path only 1.0 dB.

**File:** `/Users/aaron/Projects/musicMixer/backend/src/musicmixer/services/pipeline.py`

- [ ] **Gate on Fix 1 result (empirical):** After implementing Fix 1, run A/B tests on Pair 2. If Fix 1 closes the LUFS gap to within 1 dB of the -12.0 target, do NOT increase `max_boost_db`. Only increase to 1.5 dB if Fix 1 alone is insufficient.
- [ ] Line 831: If the gap exceeds 1 dB, change `max_boost_db=1.0` to `max_boost_db=1.5` (in the `ab_multiband_comp_v1` branch)
- [ ] Keep `max_cut_db=1.5` unchanged
- [ ] Verify: Check LUFS logs before/after auto-leveler. Confirm Fix 5 only activates when Fix 1 leaves a >1 dB gap

### Fix 6: Tune high band compressor (attack too fast, release too short)

3ms attack dulls cymbal transients. 80ms release may cause pumping on sustained HF content.

**File:** `/Users/aaron/Projects/musicMixer/backend/src/musicmixer/services/multiband.py`

- [x] Line 73: Change `attack_ms=3.0` to `attack_ms=5.0` (preserves cymbal attack character)
- [x] Line 74: Change `release_ms=80.0` to `release_ms=120.0` (prevents pumping on sustained cymbals/strings)
- [ ] Verify: Listen to Pair 2 enhanced -- cymbals should have more attack definition, less pumping on sustained passages

### Fix 7: Add NaN guard after multiband band recombination

If any band produces NaN (degenerate filter, divide-by-zero), it silently corrupts the entire output.

**File:** `/Users/aaron/Projects/musicMixer/backend/src/musicmixer/services/multiband.py`

- [x] After line 244 (`output = sum(compressed_bands.values())`), add:
  ```python
  if np.any(np.isnan(output)):
      logger.error("NaN detected in multiband output, falling back to uncompressed input")
      return audio
  ```
- [ ] Verify: Unit test with a synthetic signal that forces NaN in one band (e.g., all-zero band through log10)

### Fix 8: Add try/except around pedalboard processing for graceful degradation

A pedalboard crash on any single stem kills the entire remix. Better to skip the EQ on one stem than fail completely.

**File:** `/Users/aaron/Projects/musicMixer/backend/src/musicmixer/services/audio_utils.py`

- [x] Wrap `board(pb_input, sr)` call in try/except. On exception, log the error and return the input audio unchanged
- [ ] Verify: Unit test that passes an invalid Pedalboard config and confirms graceful fallback

### Fix 9: Add missing tests for lossy-source processing path

The entire YouTube source quality wiring is untested. A regression here silently applies HF boosts to codec artifacts.

**File:** `/Users/aaron/Projects/musicMixer/backend/tests/test_pipeline_wiring.py`

- [x] Add test: `source_quality_a="youtube-opus-128kbps"` with `ab_per_stem_eq_v1=True` -> verify `halve_hf_boosts=True` passed to vocal EQ
- [x] Add test: `source_quality_b="youtube-opus-128kbps"` with `plan.vocal_source="song_b"` -> verify lossy flag correctly maps to vocal source
- [x] Add test: either source lossy + `ab_static_mastering_v1=True` -> verify `lossy_lpf_hz=16000` passed to `master_static`
- [x] Verify: All new tests pass

### Fix 11: Raise high band compressor threshold

-24 dB threshold means the compressor is always active on the high band. Should only engage on louder material. Evaluate alongside Fix 6 since both address high-band over-processing -- if applied together, verify the combined effect does not under-compress the high band.

**File:** `/Users/aaron/Projects/musicMixer/backend/src/musicmixer/services/multiband.py`

- [x] Line 70: Change `threshold_db=-24.0` to `threshold_db=-20.0`

---

## P2: Future

### Fix 10: Add LPF to guitar and piano EQ presets

Guitar and piano separation models leave high-frequency artifacts above 12-14 kHz.

**File:** `/Users/aaron/Projects/musicMixer/backend/src/musicmixer/services/eq.py`

- [x] After line 79 (guitar preset), add: `(LowpassFilter, {"cutoff_frequency_hz": 14000.0}),`
- [x] After line 84 (piano preset), add: `(LowpassFilter, {"cutoff_frequency_hz": 16000.0}),`

### Fix 12: Improve resonance detection edge handling

`np.convolve(..., mode="same")` has edge effects near the boundary. At 200 Hz (lower bound of first detection range), the smoothed baseline is biased by zero-padding, potentially suppressing legitimate resonance detection near 200 Hz.

**File:** `/Users/aaron/Projects/musicMixer/backend/src/musicmixer/services/eq.py`

- [x] Line 193: Replace `mode="same"` convolution with edge-padded approach:
  ```python
  padded = np.pad(mag_db, baseline_window // 2, mode="edge")
  baseline_db = np.convolve(padded, kernel, mode="valid")
  ```

### Fix 13: Use explicit `math.isinf()` checks for LUFS floor comparisons

The `-inf > -40.0` comparison works in CPython but is fragile.

**File:** `/Users/aaron/Projects/musicMixer/backend/src/musicmixer/services/multiband.py`

- [x] Line 247: Change `if input_lufs > LUFS_FLOOR:` to `if not math.isinf(input_lufs) and input_lufs > LUFS_FLOOR:`
- [x] Line 254: Same pattern for `output_lufs > LUFS_FLOOR`
- [x] Add `import math` to imports

**File:** `/Users/aaron/Projects/musicMixer/backend/src/musicmixer/services/processor.py`

- [x] Line 569: Change `if current_lufs == float("-inf"):` to `if math.isinf(current_lufs):`

### Fix 14: Add silent-stem edge case tests for new modules

- [x] Test silent audio through `apply_corrective_eq` -- should return zeros without error
- [x] Test silent audio through `master_static` -- should return zeros without error
- [x] Test mono input through `master_static` -- should handle correctly

### Fix 15: Verify pedalboard.Compressor behavior with input > 0 dBFS

The bus sum before multiband compression can exceed 0 dBFS. If pedalboard clips internally, multiband receives clipped input.

- [x] Write a test: feed pedalboard.Compressor a signal peaking at +6 dBFS, verify output is not hard-clipped at 0 dBFS
- [ ] If pedalboard clips: add a pre-gain reduction before multiband compression in `pipeline.py` (normalize to -1 dBFS, then restore after)

---

## Execution Order

Fixes 1-4 (P0) must all land before the next A/B test run. Suggested implementation order:

1. **Fix 3** (EQ values) -- trivial, 5 min, reduces cumulative level loss
2. **Fix 6 + Fix 11** (high band compressor tuning + threshold) -- trivial, 2 min, both address high-band over-processing
3. **Fix 1** (post-limiter LUFS correction) -- the big one, ~30 min
4. **Fix 2** (pin vocal assignment) -- ~30 min, spans 3 files
5. **Fix 5** (auto-leveler boost) -- trivial, 2 min, gated on Fix 1 A/B results
6. **Fix 7** (NaN guard) -- 5 min
7. **Fix 4** (investigate timeout) -- variable, needs diagnostic run
8. Remaining P1/P2 fixes in priority order

After fixes 1-7 land, re-run the full 5-pair A/B test suite and compare results.
