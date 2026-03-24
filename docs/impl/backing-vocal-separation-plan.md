---
revision: 5
---

# Backing Vocal Sub-Separation Plan

> Optional two-pass stem separation to give independent lead/backing vocal control.

## Problem

BS-RoFormer-SW (6-stem) outputs a single `vocals` stem containing lead vocals, harmonies, ad-libs, and backing vocals all mixed together. The LLM has one gain knob for all vocal content. This limits mixing decisions -- you can't strip backing vocals for a raw verse, bring them up for a full chorus, or isolate harmonies for a "choir" effect.

## Solution: Two-Pass Separation

No single model outputs 7+ stems with a lead/backing split. The standard approach is two-pass:

1. **Pass 1** (existing): BS-RoFormer-SW separates into 6 stems (vocals, drums, bass, guitar, piano, other)
2. **Pass 2** (new): Run the `vocals` stem through a karaoke model to split `lead_vocals` + `backing_vocals`
3. Replace `vocals` with the two sub-stems -> **7 stems total**

Both passes run inside a single Modal GPU call (no second cold start).

### Model Choice

| Model | Architecture | Format | VRAM | Notes |
|-------|-------------|--------|------|-------|
| **UVR_MDXNET_KARA_2.onnx** (recommended) | MDX-Net | ONNX | ~2-3 GB | Good lead/backing split, small footprint, runs efficiently alongside BS-RoFormer |
| BS-Roformer Karaoke (anvuew) | RoFormer | PyTorch | ~6-8 GB | SDR 10.22, higher quality but more VRAM pressure |

Start with UVR_MDXNET_KARA_2. Test BS-Roformer Karaoke as an upgrade if quality is insufficient.

### Feature Flag

```bash
VOCAL_SUB_SEPARATION=true  # Default: false
```

| `stem_backend` | `vocal_sub_separation` | Result |
|---|---|---|
| modal | false | 6 stems (current behavior) |
| modal | true | 7 stems (lead/backing split) |
| local | false | 4 stems (current behavior) |
| local | true | 4 stems (flag ignored, warning logged) |

---

## Cross-Plan Dependencies (with Sound Quality Enhancement Plan)

This plan interacts with the [Sound Quality Enhancement Plan](sound-quality-enhancement-plan.md) at multiple points. Implementers working on either plan should review this section.

### Vocal separation fixes that are prerequisites for sound quality features

| Vocal Sep Fix | Sound Quality Feature | Why |
|---|---|---|
| Step 7.7: iterate `vocal_audio.keys()` | Step 7.75: per-stem preset EQ | EQ iterates `vocal_audio.items()` -- sub-stem keys must exist |
| Step 9: `VOCAL_STEM_NAMES` in rubberband | Step 9.5: resonance notch cuts | Better stretch quality = fewer false-positive resonances |
| Step 11.5: sum sub-stems for LUFS | Step 14.5: reference mastering | Correct vocal levels required for matchering input |
| Step 3.5: use `lead_vocals.wav` in 7-stem mode | Per-stem EQ (step 7.75) | Structure analysis data feeds EQ preset effectiveness |
| `VOCAL_STEM_NAMES` constant (shared) | EQ preset lookup in `eq.py` | Sub-stems must map to `"vocals"` preset, not `"other"` |

### Sound quality features that need awareness of 7-stem mode

| Sound Quality Feature | Interaction | Notes |
|---|---|---|
| Per-stem EQ (`ab_per_stem_eq_v1`) | `lead_vocals` and `backing_vocals` must map to the `"vocals"` EQ preset | Without this mapping, sub-stems get the `"other"` preset (wrong HPF, no presence boost) |
| Multiband compression (`ab_multiband_comp_v1`) | Vocals receive two compression stages: single-band (step 11) then multiband (step 13.3) | **Compression stacking mitigation:** When both multiband (`ab_multiband_comp_v1`) and per-stem compression (step 11) are active, pre-reduce `backing_vocals` ratio to **2.0:1** (vs 3.0:1 for `lead_vocals`) to prevent over-compression. The effective chain for backing vocals is 2.0:1 single-band then ~2.5:1 mid-band multiband. Validate with listening tests. |
| Auto-leveler tuning (`ab_autolvl_tune_v1`) | Tuning was calibrated for single-vocal-stem energy | Two independently-gained sub-stems change vocal bus energy profile |
| Reference mastering (`ab_reference_mastering_v1`) | Depends on correct LUFS from step 11.5 | Vocal sub-stem summing fix is prerequisite |

### Recommended test configurations

Test these four configurations to cover the interaction matrix:

1. **Baseline:** `vocal_sub_separation=false`, all `ab_*` sound quality flags off
2. **Vocal sep only:** `vocal_sub_separation=true`, all `ab_*` sound quality flags off
3. **Sound quality only:** `vocal_sub_separation=false`, all `ab_*` sound quality flags on
4. **Both enabled:** `vocal_sub_separation=true`, all `ab_*` sound quality flags on

Configuration 4 is the highest-risk combination and requires dedicated listening tests to verify that compression stacking (step 11 + step 13.3), EQ preset mapping, and auto-leveler behavior are all correct.

---

## File-by-File Changes

### `backend/src/musicmixer/config.py` -- Feature flag

Add one setting:

```python
vocal_sub_separation: bool = False  # VOCAL_SUB_SEPARATION env var
```

### `backend/src/musicmixer/services/separation_modal.py` -- Core change

**What changes:**
- Add `KARAOKE_CKPT = "UVR_MDXNET_KARA_2.onnx"` constant
- Bake both model weights into the Modal image (pre-download in `run_commands`)
- Add `vocal_sub_separation: bool` parameter to `separate_stems_remote`
- After 6-stem separation, if enabled, run karaoke model on the vocals WAV
- Map karaoke outputs: "vocals" -> `lead_vocals`, "instrumental" -> `backing_vocals`
- Delete original `vocals` key, return 7-stem dict
- Increase timeout from 300s to 600s for two-pass headroom
- Graceful fallback: if karaoke pass fails, keep original `vocals` stem and log warning

**Modal image update:**

```python
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install("audio-separator[gpu]", "torch", "soundfile", "librosa")
    .run_commands(
        # Pre-download BOTH model weights into the image
        f'python -c "from audio_separator.separator import Separator; '
        f"s = Separator(); s.load_model('{MODEL_CKPT}'); s.load_model('{KARAOKE_CKPT}')\"",
        # Verify ONNX GPU execution provider is available at build time.
        # Fail the build if CUDAExecutionProvider is absent (karaoke model needs it).
        'python -c "'
        'import onnxruntime as ort; '
        'providers = ort.get_available_providers(); '
        'assert \"CUDAExecutionProvider\" in providers, '
        'f\"CUDA not available for ONNX. Got: {providers}\"; '
        'print(f\"ONNX providers OK: {providers}\")'
        '"',
    )
)
```

**Second pass logic (inside `separate_stems_remote`):**

```python
KARAOKE_STEM_MAP = {"vocals": "lead_vocals", "instrumental": "backing_vocals"}

if vocal_sub_separation and "vocals" in stems:
    karaoke_sep = Separator(output_dir=str(karaoke_dir))
    karaoke_sep.load_model(KARAOKE_CKPT)
    karaoke_sep.separate(str(vocals_path))

    # Match karaoke outputs using KARAOKE_STEM_MAP (not the standard 6-stem matcher).
    # UVR_MDXNET_KARA_2 outputs files with "(Vocals)" and "(Instrumental)" tokens.
    # The "(Vocals)" output becomes lead_vocals, "(Instrumental)" becomes backing_vocals.
    # Use _tokenize_stem_filename with KARAOKE_STEM_MAP.keys() as the expected set,
    # then remap via KARAOKE_STEM_MAP to get the final stem names.

    # Sample rate verification: BS-RoFormer outputs at the input's native sample rate,
    # but UVR-MDX-NET may output at a different rate (typically 44.1kHz). Verify and
    # resample if needed BEFORE transmitting bytes back to the caller.
    primary_sr = sf.info(io.BytesIO(stems["vocals"])).samplerate
    for karaoke_output_path in karaoke_dir.iterdir():
        karaoke_sr = sf.info(str(karaoke_output_path)).samplerate
        if karaoke_sr != primary_sr:
            audio_data, _ = sf.read(str(karaoke_output_path))
            audio_resampled = librosa.resample(
                audio_data.T, orig_sr=karaoke_sr, target_sr=primary_sr,
            ).T
            sf.write(str(karaoke_output_path), audio_resampled, primary_sr, subtype="FLOAT")

    # Safe vocal replacement: do NOT delete stems["vocals"] until both sub-stems
    # are confirmed successfully read.
    lead_bytes = <read lead WAV bytes from matched file>
    backing_bytes = <read backing WAV bytes from matched file>
    try:
        assert len(lead_bytes) > 0, "lead_vocals output is empty"
        assert len(backing_bytes) > 0, "backing_vocals output is empty"
        stems["lead_vocals"] = lead_bytes
        stems["backing_vocals"] = backing_bytes
        del stems["vocals"]
    except (AssertionError, Exception) as e:
        logger.warning("Karaoke sub-stem read failed, keeping original vocals: %s", e)
        # stems["vocals"] is preserved -- graceful fallback to 6-stem mode
```

**VRAM budget:** BS-RoFormer-SW uses ~8-10 GB. UVR_MDXNET_KARA_2 uses ~2-3 GB. A10G has 24 GB. Load karaoke model *after* primary inference tensors are freed to avoid peak overlap.

### `backend/src/musicmixer/services/separation.py` -- Dispatcher

**Flag propagation:** Both `separate_stems()` and `_separate_modal()` must accept and forward the `vocal_sub_separation` flag from settings:

```python
def separate_stems(audio_path, output_dir, progress_callback=None):
    if settings.stem_backend == "modal":
        return _separate_modal(audio_path, output_dir, progress_callback)
    ...

def _separate_modal(audio_path, output_dir, progress_callback=None):
    ...
    stem_bytes_map = separate_fn.remote(
        audio_bytes=audio_bytes,
        filename=audio_path.name,
        vocal_sub_separation=settings.vocal_sub_separation,  # NEW: thread flag to Modal
    )
```

**Conditional stem validation (lines 53-62):** The current strict set equality (`received != expected`) rejects 7-stem output. Must be conditional:

```python
if settings.vocal_sub_separation:
    expected = {"lead_vocals", "backing_vocals", "drums", "bass", "guitar", "piano", "other"}
else:
    expected = {"vocals", "drums", "bass", "guitar", "piano", "other"}
```

- Local fallback: skip vocal sub-separation, log warning

### `backend/src/musicmixer/services/interpreter.py` -- Schema + prompt

**Schema adaptation:**

Add `_adapt_schema_for_7stem()` (mirrors existing `_adapt_schema_for_4stem` pattern):
- Remove `vocals` from `stem_gains` properties/required
- Add `lead_vocals` and `backing_vocals` with descriptions
- **Preserve `additionalProperties: false`** -- the existing schema has this on the `stem_gains` object (line 138). The adaptation must: (1) remove `vocals` from both `required` and `properties`, (2) add `lead_vocals` and `backing_vocals` with descriptions, (3) leave `additionalProperties: false` intact so the LLM cannot hallucinate extra stem names.

7-stem `stem_gains` schema:

```json
{
  "required": ["lead_vocals", "backing_vocals", "drums", "bass", "guitar", "piano", "other"],
  "properties": {
    "lead_vocals": {
      "type": "number", "minimum": 0.0, "maximum": 1.0,
      "description": "Lead vocal level. The primary singing voice."
    },
    "backing_vocals": {
      "type": "number", "minimum": 0.0, "maximum": 1.0,
      "description": "Backing vocal level. Harmonies, ad-libs, response phrases."
    },
    "drums": {}, "bass": {}, "guitar": {}, "piano": {}, "other": {}
  },
  "additionalProperties": false
}
```

**Schema selection call site (~line 1393):** The existing code selects `_adapt_schema_for_4stem()` for local mode but has no branch for 7-stem. Add an `elif` so the precedence is: local -> 4-stem, vocal_sub_separation -> 7-stem, else -> 6-stem:

```python
tool_schema = REMIX_PLAN_TOOL
if settings.stem_backend == "local":
    tool_schema = _adapt_schema_for_4stem(tool_schema)
elif settings.vocal_sub_separation:
    tool_schema = _adapt_schema_for_7stem(tool_schema)
```

Without this branch, the LLM always receives the 6-stem schema regardless of the flag, making the entire 7-stem path dead code.

**`_build_stem_character()` (~line 464):** Currently iterates `ALL_STEMS`, which does not include `lead_vocals` or `backing_vocals`. When `settings.vocal_sub_separation` is enabled, iterate `SEVEN_STEMS` instead so the LLM receives energy/character descriptions for both vocal sub-stems:

```python
stem_list = SEVEN_STEMS if settings.vocal_sub_separation else ALL_STEMS
for stem_name in stem_list:
    ...
```

**Few-shot examples:** The existing `_build_few_shot_messages()` returns examples whose `stem_gains` use the `"vocals"` key. When `settings.vocal_sub_separation` is enabled, the LLM needs examples with `"lead_vocals"` and `"backing_vocals"` instead. Add a `_build_few_shot_messages_7stem()` function (mirroring the existing `_build_few_shot_messages_4stem` pattern) that remaps `"vocals"` -> `"lead_vocals"` + `"backing_vocals"` (using `DEFAULT_BACKING_VOCAL_RATIO`). Wire it in `_build_few_shot_messages()`:

```python
def _build_few_shot_messages() -> list[dict]:
    if settings.stem_backend == "local":
        return _build_few_shot_messages_4stem()
    elif settings.vocal_sub_separation:
        return _build_few_shot_messages_7stem()
    ...
```

And update the call site (~line 1387) so the messages list uses the correct examples for the stem mode.

**Constants:** Add `SEVEN_STEMS` constant alongside existing `ALL_STEMS`/`FOUR_STEMS`:

```python
SEVEN_STEMS = ["lead_vocals", "backing_vocals", "drums", "bass", "guitar", "piano", "other"]
```

**System prompt (`_build_system_prompt`):** Use `SEVEN_STEMS` conditionally:

```python
if settings.vocal_sub_separation:
    stem_list = "lead_vocals, backing_vocals, drums, bass, guitar, piano, other"
    stem_count = 7
elif is_4stem:
    stem_list = "vocals, drums, bass, other"
    stem_count = 4
else:
    stem_list = "vocals, drums, bass, guitar, piano, other"
    stem_count = 6
```

**New LLM prompt guidance:**

Add to mixing rules (only when `vocal_sub_separation` is enabled):
```
- lead_vocals is the primary singing voice. backing_vocals contains harmonies,
  ad-libs, and response phrases.
- Default: lead_vocals gain >= backing_vocals gain when lead is active.
  Backing at 0.3-0.6 adds fullness without burying the lead.
- For stripped-down verses: backing_vocals at 0.0-0.1
- For full choruses: backing_vocals at 0.5-0.8
- For "choir" effect: backing_vocals at 0.8-1.0, lead_vocals at 0.6-0.7
- Backing vocal separation is imperfect -- backing_vocals at 1.0 alongside
  lead_vocals at 1.0 may partially reconstruct the original vocal mix.
```

**`_validate_remix_plan()` (lines 1183-1189):** The `valid_stems` set is hardcoded to 6 stems. Must be conditional:

```python
if settings.vocal_sub_separation:
    valid_stems = {"lead_vocals", "backing_vocals", "drums", "bass", "guitar", "piano", "other"}
else:
    valid_stems = {"vocals", "drums", "bass", "guitar", "piano", "other"}
```

**Fallback plan:** Update `generate_fallback_plan()` and `default_arrangement()` to use `lead_vocals`/`backing_vocals` when flag is on.

Add a `_split_vocal_gains(sections)` helper that post-processes sections:

```python
DEFAULT_BACKING_VOCAL_RATIO = 0.65  # Backing at ~65% of lead level

def _split_vocal_gains(sections: list[Section]) -> list[Section]:
    """Replace 'vocals' gain with 'lead_vocals' + 'backing_vocals' in all sections."""
    for s in sections:
        vocal_gain = s.stem_gains.pop("vocals", 0.0)
        s.stem_gains["lead_vocals"] = vocal_gain
        s.stem_gains["backing_vocals"] = vocal_gain * DEFAULT_BACKING_VOCAL_RATIO
    return sections
```

Call `_split_vocal_gains()` at the end of `default_arrangement()` when `settings.vocal_sub_separation` is enabled. This avoids editing the ~30+ hardcoded `"vocals"` references throughout the function body.

### `backend/src/musicmixer/services/renderer.py` -- Bus routing

Generalize hardcoded `"vocals"` checks to a set. There are **two** locations that must change:

```python
from musicmixer.services.constants import VOCAL_STEM_NAMES
```

> **Shared constant location (M2):** `VOCAL_STEM_NAMES` should be defined in a shared location (`backend/src/musicmixer/services/constants.py` or `backend/src/musicmixer/constants.py`) rather than only in `renderer.py`. Multiple modules need this constant: `renderer.py` (bus routing), `pipeline.py` (steps 7.7, 9, 11), `interpreter.py` (stem list awareness), and the sound quality plan's `eq.py` (EQ preset mapping). Defining it in one place avoids drift between copies.
>
> ```python
> # backend/src/musicmixer/services/constants.py (new file)
> VOCAL_STEM_NAMES = {"vocals", "lead_vocals", "backing_vocals"}
> ```

**Location 1 -- Audio source lookup (lines 203-204):**

```python
# Before:
if stem_name == "vocals":
    stem_audio = vocal_stems.get("vocals")

# After:
if stem_name in VOCAL_STEM_NAMES:
    stem_audio = vocal_stems.get(stem_name)
```

**Location 2 -- Bus assignment (lines 221-222):**

```python
# Before:
if stem_name == "vocals":
    vocal_bus[:usable_len] += gained

# After:
if stem_name in VOCAL_STEM_NAMES:
    vocal_bus[:usable_len] += gained
```

Both sub-stems route to the vocal bus so that:
- Spectral ducking correctly ducks instrumental against all vocal content
- Bandpass pre-filter applies to both
- Cross-song level matching treats the full vocal signal as one unit

The LLM still controls each sub-stem's gain independently.

### Sound quality EQ preset mapping for vocal sub-stems (CRITICAL)

When both `vocal_sub_separation` and `ab_per_stem_eq_v1` are enabled, the sound quality plan's EQ module (`eq.py`) looks up a preset by stem type name. The current preset map has entries for `"vocals"`, `"drums"`, `"bass"`, `"guitar"`, `"piano"`, and `"other"` -- but not `"lead_vocals"` or `"backing_vocals"`. Without a fix, vocal sub-stems fall through to the `"other"` preset, which applies wrong EQ (e.g., an 80Hz HPF instead of the vocal 120Hz HPF, no vocal presence boost at 3kHz, no mud cut at 250Hz).

**Fix:** In the EQ preset lookup (in `eq.py`'s `apply_corrective_eq` or its internal preset selection), add a `VOCAL_STEM_NAMES` check so that any stem in `VOCAL_STEM_NAMES` maps to the `"vocals"` EQ preset:

```python
from musicmixer.services.constants import VOCAL_STEM_NAMES

def _get_eq_preset(stem_type: str) -> dict:
    if stem_type in VOCAL_STEM_NAMES:
        preset_key = "vocals"
    else:
        preset_key = stem_type
    return EQ_PRESETS.get(preset_key, EQ_PRESETS["other"])
```

This ensures `lead_vocals` and `backing_vocals` receive the vocal EQ profile (HPF 120Hz, mud cut 250Hz, presence boost 3kHz, etc.) rather than the generic "other" profile. A future refinement could add distinct presets for lead vs backing vocals (e.g., less presence boost on backing to keep them behind the lead), but the `"vocals"` preset is correct as a starting point.

### `backend/src/musicmixer/services/pipeline.py` -- Flow updates

Import `VOCAL_STEM_NAMES` from the shared constants location (see `constants.py` note in the renderer section above) for use in steps 7.7, 9, and 11.

Multiple touchpoints:

| Step | Current | Change |
|------|---------|--------|
| 3.5 (structure analysis) | `vocals.wav` hardcoded in stem filename list (line 208) | Use `lead_vocals.wav` when `settings.vocal_sub_separation` is enabled; fall back to `vocals.wav` when disabled. Filter to stems that exist (already done on line 210) |
| 5 (source assignment) | Loads `vocals` from vocal song (line 307: `for stem_name in ["vocals"]`) | Load `["lead_vocals", "backing_vocals"] if settings.vocal_sub_separation else ["vocals"]` |
| 7.7 (bandpass pre-filter) | `if "vocals" in vocal_audio` (line 380) | Iterate over `vocal_audio.keys()` instead of checking for `"vocals"` key: `for vk in list(vocal_audio.keys()): vocal_audio[vk] = bandpass_filter(...)` |
| 9 (rubberband tempo match) | `is_vocal=(stem_name == "vocals")` (line 422) | Use `is_vocal=(stem_name in VOCAL_STEM_NAMES)` so both `lead_vocals` and `backing_vocals` get the vocal-optimized R3 engine settings |
| 11 (vocal compression) | `if "vocals" in vocal_audio` (line 488) | Iterate over vocal sub-stems with per-stem ratios: `lead_vocals` ratio=3.0 (standard vocal), `backing_vocals` ratio=2.0 (lighter, preserves natural blend). Use `for vk in list(vocal_audio.keys()): ratio = 3.0 if vk != "backing_vocals" else 2.0` |
| 11.5 (level match) | `vocal_audio.get("vocals")` (line 512) | **Summed LUFS approach for sub-stem level matching:** `cross_song_level_match()` returns a gained audio array, not a `gain_db` value -- so the gain must be derived externally. Steps: (1) Sum vocal sub-stems into `vocal_sum = sum(vocal_audio.values())`, (2) Compute `pre_lufs = measure_lufs(vocal_sum)`, (3) Run `gained_sum = cross_song_level_match(vocal_sum, inst_sum, sr)` to get the leveled signal, (4) Compute `post_lufs = measure_lufs(gained_sum)`, (5) Derive `gain_db = post_lufs - pre_lufs`, (6) Apply `gain_linear = 10**(gain_db/20)` to each sub-stem independently: `vocal_audio[vk] *= gain_linear`. This preserves the relative balance between lead and backing vocals while matching the combined vocal level to the instrumental. It avoids modifying the `cross_song_level_match` function signature. |

**Cross-plan notes for pipeline steps:**

> **Step 3.5 + sound quality EQ:** This fix (using `lead_vocals.wav` in 7-stem mode) is needed for the sound quality plan's per-stem EQ to receive meaningful stem analysis data. If the structure analysis step fails to find a vocal stem file (because it looks for `vocals.wav` but only `lead_vocals.wav` exists), downstream analysis data will be missing, which degrades EQ preset effectiveness.

> **Step 7.7 + sound quality step 7.75:** When `ab_per_stem_eq_v1` is also enabled, the sound quality plan's step 7.75 vocal EQ preset includes an HPF at 120Hz, which is redundant after the 150Hz bandpass applied here. This is harmless (the 150Hz bandpass already removed everything below 150Hz, so the 120Hz HPF is a no-op), but implementers should be aware. The vocal separation fix here (iterating `vocal_audio.keys()` instead of checking for `"vocals"`) must land before or alongside the sound quality plan, since step 7.75 also iterates `vocal_audio.items()` and needs the sub-stem keys to exist.

> **Step 9 + sound quality step 9.5:** The `VOCAL_STEM_NAMES` fix here must land before the sound quality plan's step 9.5 (resonance notch cuts after tempo stretch). Better stretch quality from the R3 vocal engine produces fewer artifacts, which means the resonance detector in step 9.5 will find fewer false-positive resonances caused by stretch artifacts. Ordering: this fix first, then step 9.5 resonance detection benefits from cleaner input.

> **Step 11 + sound quality step 13.3:** When `ab_multiband_comp_v1` is also enabled, vocals receive two compression stages: per-stem single-band compression here at step 11, then multiband compression on the full mix bus at step 13.3. The `backing_vocals` ratio of 2.0 (vs 3.0 for `lead_vocals`) was intentionally chosen to be lighter specifically because the multiband pass adds another compression stage. With both features enabled, `backing_vocals` sees an effective compression chain of 2.0:1 single-band then ~2.5:1 mid-band multiband. Recommend listening tests with both `vocal_sub_separation` and `ab_multiband_comp_v1` enabled to verify backing vocal dynamics are not over-compressed.

> **Step 11.5 + sound quality step 14.5:** This vocal separation fix (summing sub-stems, deriving gain externally from `cross_song_level_match`, and applying uniform `gain_linear` to each sub-stem) is a prerequisite for correct results when the sound quality plan's reference mastering (`ab_reference_mastering_v1`, step 14.5) is also enabled. Without this fix, the LUFS-based level match would only see one sub-stem (or fail the `.get("vocals")` lookup entirely in 7-stem mode), feeding incorrect levels into matchering's reference mastering. Note: `cross_song_level_match()` returns a gained audio array, not a gain value -- the gain must be derived via pre/post LUFS comparison. This fix must land before or alongside the sound quality plan.

---

## Testing Plan

### Unit Tests (new: `backend/tests/test_vocal_sub_separation.py`)

1. **Modal 2-pass (mocked):** Verify 7-stem dict returned when flag=true
2. **Modal fallback:** If karaoke pass fails, original `vocals` preserved
3. **Dispatcher validation:** Expects 7 stems when flag=true, 6 when false
4. **Local skip:** Warning logged, 4 stems returned even when flag=true

### Unit Tests (update existing)

5. **Renderer:** 7-stem sections with `lead_vocals`/`backing_vocals` both route to vocal bus
6. **Renderer backward compat:** Existing 6-stem tests unchanged
7. **Interpreter schema:** `_adapt_schema_for_7stem` produces correct JSON schema
8. **Interpreter validation:** Accepts 7-stem stem_gains, fills missing with 0.0
9. **Fallback plan:** Uses correct stem names per flag

### Integration Test (manual)

10. Full pipeline with `VOCAL_SUB_SEPARATION=true` and test songs:
    - 7 stem files per song
    - LLM produces plan with 7-stem gains
    - Audibly different backing vocal levels across sections
    - Processing time increase within 15-30s per song

---

## Risks and Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| VRAM pressure (both models loaded) | GPU OOM on A10G | Load karaoke model after primary inference freed; UVR_MDXNET_KARA_2 is only ~2-3 GB |
| Karaoke model output file naming | Wrong stem mapping | Token-based filename matching (same pattern as primary model) with broader token set |
| Poor separation quality | Bleed between lead/backing | Feature flag allows instant rollback; test BS-Roformer Karaoke as upgrade path |
| Modal image rebuild time | Slow iteration | Pre-download both models; image rebuild is one-time |
| Processing time increase | Longer wait for users | ~15-30s per song; opt-in only; could parallelize with analysis step |
| Auto-leveler calibration with split vocals | Subtle loudness inconsistencies when `ab_multiband_comp_v1` + `ab_autolvl_tune_v1` + `vocal_sub_separation` all enabled | The sound quality plan's multiband compressor x auto-leveler tuning was calibrated for single-vocal-stem energy profiles. With `vocal_sub_separation` enabled, the vocal bus energy characteristics change (two independently-gained sub-stems with different compression ratios). Recommend listening test validation for the 3-way flag interaction (`vocal_sub_separation` + `ab_multiband_comp_v1` + `ab_autolvl_tune_v1`). |

---

## Estimated Effort

| Component | Time |
|-----------|------|
| config.py (feature flag) | 10 min |
| separation_modal.py (2-pass logic) | 2-3 hours |
| separation.py (dispatcher) | 30 min |
| interpreter.py (schema + prompt) | 1.5-2 hours |
| renderer.py (bus routing) | 30 min |
| pipeline.py (flow updates) | 1-1.5 hours |
| Unit tests | 2-3 hours |
| Integration testing + listening | 1-2 hours |
| Modal image rebuild + VRAM testing | 30 min |
| **Total** | **~8-12 hours** |

---

## Future Extensions

- **Frontend knobs:** Expose lead/backing vocal sliders in the player UI
- **Model upgrade:** Swap UVR_MDXNET_KARA_2 for BS-Roformer Karaoke if quality improves
- **Auto-detect:** Enable vocal sub-separation automatically when prompt mentions backing vocals, harmonies, or choir
- **Three-way split:** Some newer models may separate lead / backing / ad-libs as distinct stems
