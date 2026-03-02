---
revision: 4
status: implemented
integration_branch: feat/echoey-vocals-fix-integration
---

# Echoey Vocals Fix

## Problem

A remix of "10 Freaky Girls" (Metro Boomin ft. 21 Savage) vocals over "Althea" (Grateful Dead) instrumentals produces vocals that sound echoey and distant throughout vocal sections. The user prompt was "keep it smooth. let the guitar breathe. something we can dance to." The effect is most noticeable during active vocal sections (beats 32-80, 112-160, 160-208, 240-272), where 21 Savage's vocals sound like they are coming from further away than the instrumental, and at phrase boundaries where a brief reverberant wash is audible.

## Root Cause Analysis

Two reviewers (senior audio pipeline engineer, professional sound engineer/mixmaster) independently analyzed the original diagnosis. Both agreed that the revision 1 proposal overweighted gain stacking as the primary cause and underweighted two other factors. The revised diagnosis below reflects their consensus, with areas of disagreement noted.

### Primary Hypothesis A: Key clash (Sound Engineer -- Critical)

F# minor vocals over E major instrumentals is a 2-semitone root clash. The interaction between fundamentals and harmonics produces beating across the harmonic series:

| Vocal note | Instrumental center | Interval | Beating |
|---|---|---|---|
| F#3 (185 Hz) | E3 (165 Hz) | Major 2nd | 20 Hz |
| A3 (220 Hz) | G#3 (208 Hz) | Minor 2nd | 12 Hz |
| F#4 (370 Hz) | E4 (330 Hz) | Major 2nd | 40 Hz |

This beating across multiple harmonics produces a chorus/flanger-like wash that the untrained ear perceives as "echoey" or "underwater." The sound engineer rates this as the single most likely dominant cause: a semitone clash between a rap vocal fundamental and a rock instrumental key center is always an error, never an artistic choice. The `key_source` was set to `"none"` (the default -- the LLM did not choose to transpose), leaving the clash uncorrected.

### Primary Hypothesis B: Beat grid drift (Senior Engineer -- Critical)

Post-stretch beat detection found 288 beats vs the 296 beats expected by the arrangement. This 2.7% drift means vocal phrases progressively desynchronize from the instrumental beat grid. Over a 48-beat verse section (~34 seconds at 85 BPM), the cumulative drift is approximately 0.9 seconds. Vocal onsets floating off the beat produce temporal smearing that the ear perceives as echo or delay. The senior engineer considers this potentially the single most important cause and flags that both original analyses underweighted it. See "Open Investigation" section below.

### Contributing Factor: Gain stacking amplifies separation artifacts

Well-supported by logs and code. The vocal stem receives cumulative gain of ~6.4 dB:

- Compression makeup: +3.0 dB (unconditional, `pipeline.py` line 695 sets `vocal_makeup_db = 3.0`, applied at line 704)
- Level match: +3.4 dB (logs: vocal=-19.3 LUFS, inst=-17.9 LUFS, driven by `vocal_offset_db = 2.0` at `processor.py` line 318 targeting vocals 2 dB above instrumental LUFS)

6.4 dB of gain on a separated vocal stem lifts any residual reverb, backing vocal bleed, and separation artifacts into audibility. Both reviewers agree this is a real and significant factor, but secondary to the key clash and/or beat grid drift. The Metro Boomin production on "10 Freaky Girls" is relatively dry (close-mic'd, deadpan delivery, minimal reverb send), so the separation artifacts being amplified are moderate, not heavy.

### Contributing Factor: Duplicate 16 kHz lowpass kills vocal air

The vocal stem passes through two stacked 2nd-order lowpass filters at 16 kHz:
1. Bandpass at step 7.7 (`pipeline.py` line 542: `high_hz=16000.0`)
2. Vocal EQ preset (`eq.py` line 48: `LowpassFilter` at 16000 Hz)

Combined rolloff is approximately -60 dB/oct at 16 kHz (~-48 dB/oct from the bandpass alone, since `sosfiltfilt` applies the 2nd-order Butterworth twice for zero-phase filtering, yielding effective 4th-order = -24 dB/oct doubled to -48 dB/oct, plus the EQ lowpass adding another -12 dB/oct). This kills vocal "air" and brightness. The ear perceives duller sound as more distant (reverb tails are duller than direct sound, so brightness = proximity). Both reviewers agree this is a straightforward fix.

**Contingency:** If Phase 1 brightness improvement is insufficient after removing the duplicate lowpass, consider raising the bandpass upper cutoff from 16 kHz to 18-20 kHz or reducing the bandpass filter order from 2 to 1 (which would halve the remaining rolloff steepness).

## Proposed Changes

Changes are organized as an incremental sequence. Each phase should be applied and listened to before proceeding to the next. This prevents unattributable regressions from simultaneous parameter changes.

### Phase 1 -- Zero-risk fixes (apply immediately)

These changes are safe, low-interaction, and both reviewers approve them without reservation.

| # | File | Line(s) | Change | Current | Proposed | Rationale |
|---|------|---------|--------|---------|----------|-----------|
| 1a | `backend/src/musicmixer/services/eq.py` | 48 | Remove duplicate 16 kHz LowpassFilter from vocal preset | `(LowpassFilter, {"cutoff_frequency_hz": 16000.0})` | Delete this entry | Bandpass at step 7.7 (`pipeline.py` line 542) already applies 16 kHz cutoff. Double-filtering creates ~-60 dB/oct combined rolloff (~-48 dB/oct bandpass + -12 dB/oct EQ) that dulls vocal top-end, making the voice sound distant. |
| 1b | `backend/src/musicmixer/services/eq.py` | 45 | Reduce 250 Hz EQ cut | `-1.5` dB | `-0.5` dB | -1.5 dB at Q=1.5 removes body from 21 Savage's vocal fundamentals (F#2-C#4 range). -0.5 dB still addresses low-mid mud from separation while preserving warmth. |
| 1c | `backend/src/musicmixer/services/eq.py` | 46 | Reduce 800 Hz EQ cut | `-1.5` dB | `-0.75` dB | -1.5 dB at 800 Hz removes vocal body/warmth ("honk" frequency). -0.75 dB still tames the nasal quality without making the voice thin and distant. |

**After Phase 1:** Re-run the same mix ("10 Freaky Girls" + "Althea") and listen for improved vocal clarity and brightness.

### Phase 2 -- Key transposition (test dominant hypothesis)

Both reviewers agree key clash correction should be implemented before any gain chain changes. If the "echoey" perception is primarily caused by semitone beating, this single change may resolve it.

#### Existing Infrastructure

The codebase already has key transposition infrastructure that Phase 2 must integrate with rather than duplicate:

- **`MixPlan.key_source`** (`models.py:195`): Field on the plan model. When the LLM sets this to a song name, the interpreter already handles key-matching logic (`interpreter.py` lines 150, 761, 817, 860, 903).
- **`_key_semitone_distance`** (`taste_constraints.py:321`): Utility that computes chromatic semitone distance between two keys, already used for constraint scoring.
- **`_estimate_pitch_shift_semitones`** (`taste_features.py:180`): Estimates the pitch shift needed between two detected keys.
- **4-semitone cap** (`taste_constraints.py:282-326`): Existing constraint that rejects transpositions exceeding +-4 semitones to avoid unnatural pitch artifacts.
- **Taste model scoring** (`taste_model.py:340-365`): Scores key transposition decisions during plan evaluation.

Phase 2 wires the existing `plan.key_source` field into the pipeline's rubberband calls. The auto-transpose heuristic below serves only as a **safety net** for cases where the LLM sets `key_source = "none"` but a dissonant interval is detected.

#### Transposition Interval Algorithm

1. Extract the root note and mode (major/minor) from both the vocal source key and instrumental source key (from audio metadata populated at step 3).
2. Treat relative major/minor as compatible (e.g., C major and A minor share the same pitch classes and are not dissonant).
3. Compute chromatic distance (0-11 semitones) between the vocal root and the instrumental root using `_key_semitone_distance` (`taste_constraints.py:321`).
4. Flag as dissonant if the interval is in the set `{1, 2, 6, 10, 11}`. Major and minor seconds (1-2 semitones) and sevenths (10-11 semitones) produce audible beating; the tritone (6) clashes harmonically.
5. When dissonant, choose the transposition direction that minimizes absolute semitone shift. On ties, prefer downward (preserves vocal weight).
6. Cap at +-4 semitones, consistent with the existing taste constraint (`taste_constraints.py:282-326`). If the minimum transposition to a consonant interval exceeds 4 semitones, log a warning and skip transposition.
7. For this specific remix: transpose vocals -2 semitones (F# minor -> E minor). E minor is the relative minor of G major, harmonically stable over E major, and common in Grateful Dead material.

#### Phase 2 Implementation Details

Wiring `key_source` into the pipeline is new work -- the existing `rubberband_process` function accepts a `semitones` parameter (`processor.py` line 121), but the pipeline currently never passes it and never reads `plan.key_source` to compute a pitch shift.

**Pipeline integration (mirrors how `tempo_source` already works):**

1. **After step 7.75** (once key metadata is available from both sources), compute the semitone offset:
   - Use `_key_semitone_distance` (`taste_constraints.py:321`) **only for dissonance detection** (checking whether the unsigned interval falls in the dissonant set `{1, 2, 6, 10, 11}`). This function returns an unsigned distance (0-6), which has no direction information and cannot be passed to rubberband's `-p` flag.
   - For the actual signed shift, compute separately: `signed_shift = (target_root - source_root + 6) % 12 - 6`, where `target_root` and `source_root` are integer pitch classes (C=0, C#=1, ..., B=11). This yields a value in the range [-6, 5] that minimizes absolute shift and prefers downward on ties (consistent with step 5 of the Transposition Interval Algorithm).
   - If `plan.key_source != "none"`: compute the signed shift to align the vocal key to the instrumental key.
   - If `plan.key_source == "none"`: run the auto-transpose heuristic (algorithm above) as a safety net. If the interval is dissonant, compute the signed shift; otherwise, `semitones = 0`.
2. **Store the result** in a pipeline-local variable (e.g., `pitch_shift_semitones: int`). No new field on `RemixPlan` is needed -- the pipeline computes the shift at runtime from `plan.key_source` + the audio metadata already available in the pipeline context. This keeps `RemixPlan` as a declarative intent ("match keys to song B") rather than encoding a computed value.
3. **Pass `semitones=pitch_shift_semitones`** to the existing `rubberband_process` calls at **Step 9** (tempo stretch). Rubberband handles pitch and time-stretch in a single pass, so there is no additional processing step and no double-resampling quality loss.

**No new pipeline step is introduced.** The semitone offset is computed inline after step 7.75 and consumed by the existing Step 9 rubberband calls.

**After Phase 2:** Re-run with key transposition. A/B compare against the Phase 1 result. If the echoey perception is substantially reduced, the remaining phases become optimization rather than correction.

### Phase 3 -- Gain chain reduction (if echo persists after Phase 2)

Both reviewers agree gain should be reduced but disagree with the original proposal's `vocal_offset_db = 0.0` target. Both independently recommend 1.0 dB.

| # | File | Line(s) | Parameter | Current | Proposed | Rationale |
|---|------|---------|-----------|---------|----------|-----------|
| 3a | `backend/src/musicmixer/services/processor.py` | 318 | `vocal_offset_db` | `2.0` | `1.0` | The code comment on this line says "Revisit with spectral ducking (Day 4)." Ducking is now implemented (`ducking.py`), so reducing to 1.0 is timely. 0.0 would bury the vocal -- rap vocals are percussive and transient-heavy, so equal LUFS feels quieter than it measures. 1.0 dB maintains vocal presence while halving the offset contribution. |
| 3b | `backend/src/musicmixer/services/pipeline.py` | 695 | `vocal_makeup_db` | `3.0` | `1.5` | Halves amplification of reverb tails and separation artifacts. The compressor applies makeup to all frames above the gate floor (`processor.py` lines 399-409), not just compressed frames -- so the full +3.0 dB currently boosts quiet passages (reverb tails at -30 to -45 dB) that were never compressed. |

**Net effect:** Total vocal gain drops from ~6.4 dB to ~3.9 dB (1.5 makeup + ~2.4 level match). This halves artifact amplification while keeping the vocal forward in the mix.

**After Phase 3:** Re-run and verify vocal presence. If vocals feel buried, bump `vocal_offset_db` to 1.5 before reverting to 2.0.

### Phase 4 -- Ducking and mastering tweaks (only if echo persists)

These changes should only be applied if Phases 1-3 do not resolve the issue. The reviewers disagree on ducking specifics.

| # | File | Line(s) | Parameter | Current | Proposed | Rationale |
|---|------|---------|-----------|---------|----------|-----------|
| 4a | `backend/src/musicmixer/services/pipeline.py` | 925 | Second limiter pass `release_ms` | `50.0` ms | `120.0` ms | Standard mastering limiter release. 50ms is fast enough to cause audible gain pumping on vocal transients. 120ms is safe and low-impact. |
| 4b | `backend/src/musicmixer/services/ducking.py` | 132 | Release alpha time constant | `0.15` s (150 ms) | See options below | 150ms release means instrumental mid-range snaps back within one sixteenth note at 85 BPM, creating rhythmic "breathing." |
| 4c | `backend/src/musicmixer/services/pipeline.py` | 705 | Compression `gate_floor_db` | `-50.0` dB | `-35.0` dB | More targeted than reducing makeup globally. Frames between -50 dB and -35 dB (reverb tails, quiet artifacts) currently receive full makeup gain despite never being compressed. Raising the floor prevents this without affecting active vocal passages. |

**Ducking release options (choose one, do NOT apply both):**

- **Option A (Senior Engineer):** Change release from 150ms to 300ms. No hold time. Rationale: 300ms is sufficient to smooth phrase boundaries without over-ducking. 400ms combined with hold creates ~600ms of residual ducking (nearly a full beat at 85 BPM), which audibly thins the instrumental between phrases.

- **Option B (Sound Engineer):** Change release from 150ms to 400ms. Add 200ms hold time. Rationale: The hold prevents rapid on/off cycling on intermittent ad-libs (21 Savage pauses frequently between bars). 400ms release gives a full quarter-note recovery at 85 BPM.

**Constraint:** Do not apply both a release increase AND hold simultaneously with the larger values. Either 300ms release with no hold, or 400ms release with 200ms hold -- never 400ms release + 200ms hold + any other ducking band changes in the same pass.

## Deleted from Proposal

The following changes from revision 1 have been removed based on reviewer feedback:

1. **Hard "other" stem clamp to 0.0 during vocal sections** -- Both reviewers rejected this as too blunt. Zeroing the "other" stem creates an audible "room change" (the listener perceives the instrumental going from a live room to a studio when vocals enter). Replace with soft LLM guidance: add a note to the interpreter system prompt suggesting `other: 0.0-0.15` during vocal sections when the instrumental source is a live recording with probable vocal bleed. This preserves the LLM's ability to make contextual decisions per-mix.

2. **`vocal_offset_db` = 0.0** -- Both reviewers independently said this would bury the vocal. Replaced with 1.0 dB (see Phase 3).

## Risks

- **All changes affect ALL mixes globally, not just this one.** EQ presets, compression makeup, and vocal offset are applied to every remix. Incremental application with listening tests between phases prevents unattributable regressions.

- **Key transposition carries the risk of unwanted pitch shifts.** Auto-transposing vocals could produce unnatural results on tracks where the key detection is low-confidence or where the "clash" is intentional (unlikely for rap-over-rock, but possible for experimental genres). Consider only auto-transposing when both key detections are above a confidence threshold.

- **Gain reduction is additive across phases.** Reducing both `vocal_offset_db` (2.0 -> 1.0) and `makeup_db` (3.0 -> 1.5) simultaneously cuts ~4.9 dB of vocal gain. The auto-leveler (`pipeline.py` lines 847-856) uses the un-ducked instrumental bus as its detector, so it will NOT compensate for the vocal level drop. The effective vocal reduction in the final mix may be larger than the raw dB numbers suggest.

- **Gate floor increase (-50 -> -35 dB) may cut legitimate quiet vocal passages** on tracks with very dynamic vocals (whisper sections, spoken intros). Test with material that has intentionally quiet vocal moments.

## Testing

### Phase 1 Validation
Re-run the same mix ("10 Freaky Girls" + "Althea") with the same prompt after removing the duplicate lowpass and adjusting EQ cuts. Listen for:
- Improved vocal brightness and "closeness" (removing the duplicate lowpass that contributed ~-12 dB/oct to the ~-60 dB/oct combined rolloff)
- Slightly warmer vocal tone (reduced 250 Hz and 800 Hz cuts)
- No regression in instrumental clarity

### Phase 2 Validation
Re-run with key transposition (-2 semitones on vocals). A/B compare:
- With transposition: does the "echoey" or "underwater" quality resolve?
- Without transposition (Phase 1 only): is the beating audible as chorus/flanger wash?
- If echo is substantially reduced by transposition alone, remaining phases may be unnecessary.

### Phase 3 Validation
If applied, verify vocal presence:
- 21 Savage's voice should still be the dominant element in the mid-range during verse sections (beats 32-80, 112-160)
- If vocals feel buried, bump `vocal_offset_db` to 1.5 before reverting further
- Compare vocal-to-instrumental balance at both loud (chorus) and quiet (verse) sections

### Cross-Genre Regression Test
Run a mix that does NOT have echo issues (e.g., a clean studio vocal over a sparse electronic beat) to verify that the Phase 1 EQ changes and any Phase 3 gain changes do not introduce new problems. Listen for:
- Vocals not becoming boomy (the 250 Hz cut reduction on a different vocal type)
- No loss of instrumental clarity
- Overall mix level still reaching target LUFS without excessive limiting

## Deleted from Scope

The following pipeline features referenced in the original analysis have been **removed** on the `feat/flag-cleanup-integration` branch and are no longer applicable:

- **Multiband compressor** -- Module removed entirely. Cross-plan interaction C2 (double compression from vocal compressor + multiband compressor) is no longer a concern.
- **Resonance detection** -- Module removed entirely.
- **8 of 9 feature flags** removed; code paths simplified. Compression makeup is now hardcoded at 3.0 dB (line 695), auto-leveler parameters are hardcoded (line 847), and static mastering is the only mastering path.

All 759 tests pass on the current branch.

## Open Investigation: Beat Grid Drift

**Status:** Separate investigation required. Do not block the Phase 1-2 fixes on this.

The post-stretch beat detection found 288 beats vs the 296 beats expected by the arrangement. The senior engineer's analysis:

- The beat grid used for rendering is based on the 288-beat detection from the post-stretch instrumental (`pipeline.py` lines 652-684).
- The arrangement spans 296 beats (sections go up to beat 304).
- When the renderer requests `beats_to_samples(296, ...)` and the grid only has 288 entries, it falls through to extrapolation using the average interval of the last 8 beats.
- Beats 0-287 use detected positions; beats 288-304 use linear extrapolation.
- Over a 48-beat verse (~34s), cumulative drift could reach ~0.9 seconds -- massive temporal smearing perceived as echo.

**Next step:** Log sample positions at beats 0, 32, and 80 for both the vocal and instrumental buses post-stretch. Compare alignment. If they are drifting by >20ms by beat 80, this is a primary fix target that would require changes to the beat detection or tempo-stretch pipeline, not just parameter tuning.
