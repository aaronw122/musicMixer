# Flag Cleanup Plan: Simplify Audio Processing Pipeline

**Date:** 2026-02-28
**Goal:** Reduce flag complexity from 9 flags to 1, merge proven behavior into baseline, cut redundant/harmful processing, fix mastering chain issues.
**Demo:** Saturday 2026-03-01
**Status:** Reviewed by 3 expert personas (senior engineer, ML scientist, mix master). All issues incorporated.

---

## Background

Two expert reviews (sound engineer, ML scientist) independently evaluated all 9 feature flags against the implementation plans, code, and A/B test methodology. A third review (mix master) evaluated the resulting pipeline for sound quality. Their recommendations converged on the same core conclusion: the pipeline has too many overlapping dynamics stages fighting each other. The fix is not more tuning — it's removal of redundant processing.

### Expert Verdicts Summary

| Flag | Sound Engineer | ML Scientist | Action |
|------|---------------|-------------|--------|
| `ab_control_day3` | Merge | Merge | **Merge** |
| `ab_autolvl_tune_v1` | Merge | Merge | **Merge** |
| `ab_vocal_makeup_v1` | Merge | Merge | **Merge** |
| `ab_mp3_export_path_v1` | Merge | Merge | **Merge** |
| `ab_per_stem_eq_v1` | Merge | Merge | **Merge** |
| `ab_resonance_detection_v1` | CUT | CUT | **Delete** |
| `ab_multiband_comp_v1` | CUT | Keep as flag (rework) | **Delete** |
| `ab_static_mastering_v1` | Keep as flag (fix chain) | Merge | **Merge + fix** |
| `ab_taste_model_v1` | Keep as flag | Keep as flag | **Keep as flag, default ON** |

### Key Redundancies Identified

1. **Resonance detection ↔ per-stem EQ**: Both target the same frequency ranges (250Hz, 400Hz, 800Hz, 2-4kHz). Corrective EQ already handles these with wider, safer filters. Resonance detection adds narrow notch cuts on top — over-processing.
2. **Multiband LUFS restoration ↔ static mastering normalization ↔ post-soft-clip correction**: Three competing LUFS normalization stages create a gain-staging cascade. Root cause of Pair 3 timeout.
3. **Multiband compression ↔ per-stem pre-limiting + vocal compression**: Frequency-dependent dynamics are already controlled per-stem before the mix bus. Multiband re-compresses already-shaped audio.

---

## Pre-Implementation Steps

### Step 0a: Capture Pre-Cleanup Baselines
Run the existing A/B test suite BEFORE any code changes. Save results to `notes/` for before/after comparison.
```bash
cd backend && python ../scripts/run_ab_test_suite.py --mode compare
```

### Step 0b: Create Pre-Cleanup Git Tag
```bash
git tag pre-flag-cleanup
```

---

## Implementation Phases

**Execution order matters.** Phases have dependencies — follow the numbered order below.

### Phase 1: Update A/B Test Suite FIRST

**CRITICAL:** The test suite's `preflight()` function checks for all 8 flag env vars via `hasattr()` and exits if missing. If we remove flags from config.py before updating the test suite, it hard-crashes. Fix the test suite first.

**Files:**
- `scripts/run_ab_test_suite.py` — Remove references to deleted flags, update `_BASELINE_FLAGS`, fix `preflight()`
- `scripts/run_modal_ab_phases.sh` — Update or delete (references 3 deleted flags as env vars)

### Phase 2: Delete Multiband Compression

**Must happen before Phase 4** (auto-leveler hardcoding depends on multiband being gone).

**Files to modify (atomically in one commit):**
- `backend/src/musicmixer/services/pipeline.py` — Remove step 13.3 block AND the `from musicmixer.services.multiband import multiband_compress` import
- `backend/src/musicmixer/services/multiband.py` — **Delete entire file**

**Rationale:**
- Sound engineer: "Doing more harm than good. Gain staging cascade with static mastering causes timeout and loudness shortfall."
- ML scientist: "Right idea but thresholds need input-adaptive calibration."
- Both agree the per-stem pre-limiting + vocal compression already handle frequency-dependent dynamics before the mix bus.
- Can be revisited post-demo with input-adaptive thresholds if needed.

### Phase 3: Delete Resonance Detection

**Files to modify:**
- `backend/src/musicmixer/services/pipeline.py` — Remove step 9.5 (the `if settings.ab_per_stem_eq_v1 and settings.ab_resonance_detection_v1:` block)
- `backend/src/musicmixer/services/eq.py` — Remove `detect_resonances()`, `_build_resonance_board()`, the `apply_resonance_cuts` parameter from `apply_corrective_eq()` signature and all call sites, and the `RESONANCE_ELIGIBLE_STEMS` constant

**Rationale:** All three experts agree — high false-positive rate on musical content in separated stems, redundant with corrective EQ, marginal audible impact with conservative parameters (-3dB max, Q=4-6).

### Phase 4: Hardcode Merged Flag Behavior in Pipeline

**File:** `backend/src/musicmixer/services/pipeline.py`

For each merged flag, replace the conditional with the "on" behavior:

#### 4a. `ab_control_day3` — No pipeline changes needed
This flag has zero references in `pipeline.py`. The config deletion in Phase 5 is sufficient.

#### 4b. `ab_autolvl_tune_v1` — Hardcode tuned auto-leveler params
- Replace the 3-way conditional (`multiband → autolvl_tune → default`) with a single set of values:
  - `window_sec = 4.0`
  - `max_boost_db = 1.5`
  - `max_cut_db = 2.5`
- Safe to hardcode now because multiband is already deleted (Phase 2).

#### 4c. `ab_vocal_makeup_v1` — Hardcode 3.0 dB makeup
- Replace `vocal_makeup_db = 3.0 if settings.ab_vocal_makeup_v1 else 4.0` with `vocal_makeup_db = 3.0`

#### 4d. `ab_mp3_export_path_v1` — Hardcode skip-dither export
- Replace `use_s16_dither = not settings.ab_mp3_export_path_v1` with `use_s16_dither = False`
- Or better: remove the dither parameter entirely from `export_mp3` in `processor.py`.

#### 4e. `ab_per_stem_eq_v1` — Make corrective EQ always run
- Remove all `if settings.ab_per_stem_eq_v1:` guards
- Vocal bandpass upper cutoff hardcoded to 16kHz
- Preset EQ always applied at step 7.75
- Resonance detection is already gone (Phase 3)

#### 4f. `ab_static_mastering_v1` — Make static mastering the only path
- Remove the `if settings.ab_static_mastering_v1:` / `else:` branching
- Delete the standard mastering path (limiter at -7dBTP → LUFS normalize → soft clip)
- Static mastering becomes the unconditional mastering chain
- **Note:** This replaces the current production default mastering chain. The standard path becomes dead code.

#### 4g. Quick win: Lower auto-leveler active floor
- Change `active_floor_db` from -40dBFS to -50dBFS
- Prevents perceived volume drops during instrumental-only sections between vocal phrases

### Phase 5: Remove Flag Definitions from Config

**File:** `backend/src/musicmixer/config.py`

Remove these 8 flag definitions from the Settings class:
```
ab_control_day3: bool = True
ab_autolvl_tune_v1: bool = False
ab_vocal_makeup_v1: bool = False
ab_mp3_export_path_v1: bool = False
ab_per_stem_eq_v1: bool = False
ab_resonance_detection_v1: bool = False
ab_multiband_comp_v1: bool = False
ab_static_mastering_v1: bool = False
```

Keep only:
```
ab_taste_model_v1: bool = True   # default ON — demo default, not validated production default
```

Safe to do now because no code references the deleted flags (Phases 2-4 already removed all usages).

### Phase 6: Fix Static Mastering Chain

**File:** `backend/src/musicmixer/services/pipeline.py` (NOT mastering.py — the soft-clip and correction loop code lives in pipeline.py)

The sound engineer identified chain ordering issues in the LUFS correction loop (pipeline steps 14.5-14.7):

#### 6a. Move safety soft clip AFTER the LUFS correction loop
**Current order:** mastering → soft clip → LUFS correction (boost + re-limit)
**Correct order:** mastering → LUFS correction (boost + re-limit) → soft clip

The correction loop boosts gain after the soft clip, potentially re-introducing inter-sample peaks. The soft clip should be the final safety stage.

**Validate independently:** Run pipeline on 3+ test pairs before and after this change. Compare LUFS and true-peak measurements.

#### 6b. Increase second limiter lookahead
The correction loop's second limiter pass uses 1.5ms lookahead (vs 5ms on the main limiter). Increase to **5ms** (not just 3ms) to match main limiter reliability.

#### 6c. DEFERRED to post-demo: Consider eliminating the correction loop
The sound engineer suggests increasing headroom from +3dB to +4-5dB in the constrained normalizer. Keep the correction loop as a safety net for now. Evaluate empirically post-demo whether it ever fires. If not, remove.

### Phase 7: Clean Up Dead Code, Imports, Docs

- Remove unused imports of deleted modules (`multiband`, resonance detection functions) from `pipeline.py`
- Remove stale log statements that reference deleted flag names (lines ~541-542, 723-724, 751-752, 935-941, 1089-1092)
- Update module docstrings in `mastering.py`, `eq.py` that reference deleted flags/modules
- Remove any env var documentation for deleted flags in `.env.example` if it exists
- Delete the standard mastering path code (dead after Phase 4f)
- Clean up `processor.py` if the `use_s16_dither` parameter is fully removed

---

## Resulting Pipeline (Post-Cleanup)

```
Step 1:   Separate stems (BS-RoFormer / htdemucs)
Step 2:   Analyze songs (BPM, key, beat grid)
Step 3:   Reconcile BPM
Step 4:   Generate mix plan (fallback: deterministic)
Step 4.5: [IF taste_model ON] Taste stage: candidates → filter → score → select
Step 5:   Determine vocal/instrumental sources
Step 6:   Load stems (44.1kHz, stereo, float32)
Step 7:   Trim to source ranges
Step 7.5: Filter silent stems (<-50 LUFS)
Step 7.7: Vocal bandpass (150Hz–16kHz)
Step 7.75: Corrective EQ per stem (always on)
Step 8:   Compute tempo plan
Step 9:   Tempo match (rubberband)
Step 10:  Re-detect beat grid post-stretch
Step 11:  Vocal compression (3:1, -20dB, 3.0dB makeup)
Step 11.5: Cross-song level matching
Step 11.8: Pre-limit drums (-6dBTP) and bass (-4dBTP)
Step 12:  Render arrangement
Step 12.5: Spectral ducking (300–3kHz pocket)
Step 13:  Sum to mix bus
Step 13.7: Auto-leveler (4s window, 1.5dB boost, 2.5dB cut)
Step 14:  Static mastering (LUFS normalize → limiter → [correction] → soft clip)
Step 15:  Fades (2s in, 3s out)
Step 16:  Export MP3 (320kbps, no pre-dither)
```

**What changed vs. current "all flags on":**
- Removed: resonance detection (step 9.5)
- Removed: multiband compression (step 13.3)
- Fixed: mastering chain soft-clip ordering
- Simplified: no flag conditionals, single code path

**What changed vs. current "all flags off" (production default):**
- Added: corrective EQ per stem (step 7.75)
- Added: 16kHz vocal bandpass (was 8kHz)
- Changed: static mastering replaces standard mastering chain
- Changed: auto-leveler uses tuned params (4s/1.5/2.5)
- Changed: vocal makeup 3.0dB (was 4.0dB)
- Changed: no MP3 pre-dithering
- Added: taste model default ON

---

## Testing Strategy

### Step 0 (pre-cleanup):
1. Run existing A/B test suite in `compare` mode on all 5 pairs — save results to `notes/`
2. These results become the "before" baseline for regression comparison

### During implementation:
1. After Phase 2 (multiband deletion): run pipeline on 2+ pairs, verify no timeouts (Pair 3 fix)
2. After Phase 4 (all hardcoding): run pipeline on all 5 pairs, compare LUFS/peak to baseline
3. After Phase 6 (mastering fix): run pipeline on 3+ pairs, compare LUFS/peak — verify soft-clip reordering doesn't change target behavior

### After merging:
1. Convert test suite to ablation testing (taste model on vs off)
2. Add 2-3 test pairs with sung vocals and electronic music (per ML scientist)
3. Consider a simple perceptual evaluation rubric (5-point scale: clarity, warmth, balance, fatigue)

---

## Files Changed (Summary)

| File | Phase | Change |
|------|-------|--------|
| `scripts/run_ab_test_suite.py` | 1 | Fix `preflight()`, remove deleted flag references |
| `scripts/run_modal_ab_phases.sh` | 1 | Update or delete (references deleted flags) |
| `backend/src/musicmixer/services/pipeline.py` | 2,3,4,6,7 | Remove multiband/resonance steps, hardcode flag behaviors, delete standard mastering path, fix mastering chain, clean imports/logs |
| `backend/src/musicmixer/services/multiband.py` | 2 | **Delete entire file** |
| `backend/src/musicmixer/services/eq.py` | 3 | Remove resonance detection functions, `apply_resonance_cuts` param, `RESONANCE_ELIGIBLE_STEMS` |
| `backend/src/musicmixer/services/processor.py` | 4 | Remove `use_s16_dither` parameter from `export_mp3` (if taking full cleanup route) |
| `backend/src/musicmixer/config.py` | 5 | Remove 8 flags, change taste_model default to True |
| `backend/src/musicmixer/services/mastering.py` | 7 | Update docstring (mastering chain is now the only path) |

---

## Open Questions (Resolved)

1. ~~Should we keep the LUFS correction loop?~~ **Keep for now** (all 3 reviewers agree: defer removal to post-demo). Fix the soft-clip ordering and lookahead, evaluate empirically.
2. ~~Should multiband be kept behind a flag?~~ **No — delete now.** The mix master confirmed per-stem pre-limiting + vocal compression covers the dynamics gap. Revisit post-demo only if mixes sound thin.
3. **Pre-limiter thresholds** (post-demo follow-up): Sound engineer notes drums at -6dBTP and bass at -4dBTP may be too aggressive now that corrective EQ exists. Not in scope.
4. **Double 16kHz LPF on vocals** (post-demo follow-up): Vocal bandpass (16kHz) + EQ preset LPF (16kHz) cascade to -12dB/oct. Consider raising EQ preset to 18kHz or removing one.
5. **Vocal compressor release** (post-demo follow-up): 80ms release is fine for rap but too fast for sustained singing. Consider adaptive release based on genre.

---

## Known Risks

1. **Production mastering chain swap:** Static mastering replaces the standard chain (current production default). Mitigated by pre-cleanup git tag for rollback.
2. **Taste model default ON:** This is a product behavior change, not just cleanup. Documented as demo default. Fallback path is robust (400ms timeout, circuit breaker, silent fallback to LLM plan).
3. **No unit test suite:** Backend has no automated tests. Mitigation: manual pipeline runs at each phase boundary.

---

## Review History

| Reviewer | Role | Key Findings |
|----------|------|-------------|
| Expert 1 | Sound Engineer | Per-stem EQ is highest value; multiband + mastering cascade is root cause of timeouts; resonance detection solves a non-problem |
| Expert 2 | ML Scientist | Signal chain interactions make isolated A/B testing invalid; sweep methodology flawed; resonance detection has high false-positive risk |
| Expert 3 | Senior Engineer | Test suite crashes after flag removal; Phase ordering dependencies; missing files in plan; no rollback strategy |
| Expert 4 | ML Scientist (plan review) | Pre-cleanup baseline needed; mastering fix should be validated independently; correction loop should be kept as safety net |
| Expert 5 | Mix Master (plan review) | Pipeline will sound good for demo; auto-leveler floor should be lowered; second limiter lookahead should be 5ms |

Full reviews: `notes/` directory (to be saved from `/tmp/` during implementation)
