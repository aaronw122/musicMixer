# Mashup Reverse-Engineering Training Pipeline

revision: 3

## Executive Summary

The original taste training plan (Phase 1+) called for 500 human pairwise A/B labels — ~8 hours of manual listening. This plan replaces that with a **zero-listening-required** approach: reverse-engineer ~50-100 popular mashups from YouTube as positive training examples, generate synthetic negatives, and train CatBoost on the structural contrast.

**What this replaces**: Phase 0 Days 5-7, Phase 1 (Weeks 2-3) human labeling loop.

**What this preserves**: All Phase 0 infrastructure (candidate planner, constraints, features, scorer, pipeline wiring) is reused as-is.

**Prerequisite**: Phase 0 Days 1-4 complete on `feat/taste-training-integration` branch.

**Prerequisite (label vocabulary)**: The `Section.label` comment in `models.py` lists `"intro" | "build" | "main" | "breakdown" | "outro"`, but the LLM interpreter's JSON schema enum in `interpreter.py` uses `["intro", "verse", "breakdown", "drop", "outro"]`. These are the labels that actually appear in inference-time plans. The `models.py` comment should be updated to match the interpreter's vocabulary before implementing this pipeline, so that reconstruction, feature extraction, and inference all share one canonical label set: `intro | verse | breakdown | drop | outro`.

---

## Why This Works

Popular mashups are crowd-validated taste data. A mashup with millions of YouTube views represents arrangement decisions (section pacing, vocal placement, energy arc, transitions) that resonated with listeners. By extracting the structural "plan" from these mashups and training CatBoost to prefer similar structures, we learn from thousands of implicit human judgments without anyone listening to a single A/B pair.

**Feature compatibility**: 22 of 28 Tier 1 features are fully computable from a reconstructed plan without knowing the source songs. The 6 harmonic/tempo features (Group 4) gracefully default to neutral values when source metadata is unavailable — this is already implemented in `taste_features.py`.

---

## Pipeline Overview

```
Step 1: Curate mashup manifest (JSON)
  |
Step 2: Download mashups (yt-dlp)
  |
Step 3: Separate stems + analyze structure (Modal GPU + existing analysis pipeline)
  |
Step 4: Reconstruct RemixPlan objects from detected structure
  |
Step 5: Generate synthetic negative plans (degraded versions)
  |
Step 6: Extract features + build training dataset (one row per plan)
  |
Step 6b: Domain alignment validation (compare feature distributions)
  |
Step 7: Train CatBoost pairwise ranker → model artifact
```

---

## Step 1: Mashup Manifest

**File**: `backend/data/mashup_manifest.json`

A curated JSON list of 50-100 popular mashups with metadata.

```json
[
  {
    "id": "mashup_001",
    "url": "https://www.youtube.com/watch?v=...",
    "title": "Girl Talk - Play Your Part (Pt. 1)",
    "artist": "Girl Talk",
    "genre_hint": "hiphop",
    "source_a_url": null,
    "source_b_url": null,
    "notes": ""
  }
]
```

**Genre hints** (for per-genre feature calibration): `hiphop`, `edm`, `pop`, `rock`, `rnb`, `experimental`.

**Curation sources**: Girl Talk, The Hood Internet, DJ Earworm, Madeon, Isosine, Collision Course (Jay-Z / Linkin Park), The Grey Album, Bootie Mashup top charts, Reddit /r/mashups top posts.

**Effort**: 1-2 hours manual curation. Over-curate to 120+ URLs to account for ~20% takedown attrition.

---

## Step 2: Download Script

**File**: `backend/scripts/mashup_pipeline.py` (subcommand: `download`)

Downloads all mashups from the manifest as WAV files.

**Reuses**: `backend/src/musicmixer/services/youtube.py` → `download_youtube_audio()`

```python
def download_mashups(manifest_path: Path, output_dir: Path, max_concurrent: int = 3):
    """Download mashups from manifest. Skips already-downloaded files."""
```

**Output**: `backend/data/mashups/raw/{id}.wav`

**Async note**: `download_youtube_audio()` is `async def`. The CLI script is synchronous, so calls must be wrapped with `asyncio.run()` or a thin synchronous wrapper (e.g., `def download_sync(url, path): return asyncio.run(download_youtube_audio(url, path))`). Do not call the async function directly from the synchronous pipeline.

**Handles**: Already-downloaded (skip), failed downloads (log + continue), rate limiting.

**Effort**: ~1 hour. Mostly wiring existing YouTube download function.

---

## Step 3: Analysis Pipeline

**File**: `backend/scripts/mashup_pipeline.py` (subcommand: `analyze`)

For each downloaded mashup:
1. `analyze_audio()` → BPM, beat_frames, duration, key
2. `separate_stems()` → 6 stems via Modal
3. `analyze_stems()` → per-bar energy, vocal activity, section boundaries

**Reuses**:
- `backend/src/musicmixer/services/analysis.py` → `analyze_audio()`, `analyze_stems()`, `detect_key()`
- `backend/src/musicmixer/services/separation.py` → `separate_stems()`

```python
def analyze_mashup(mashup_id: str, wav_path: Path, stems_dir: Path) -> dict:
    """Full analysis pipeline for one mashup. Returns serializable dict."""

def analyze_all(manifest: list[dict], raw_dir: Path, output_dir: Path, max_concurrent: int = 3):
    """Batch analysis with concurrent Modal separation."""
```

**Output**: `backend/data/mashups/analysis/{id}.json` — serialized `AudioMetadata`, `StemAnalysis`, `SongStructure`.

**Note**: Stem separation of an already-mixed mashup produces noisier results than separating clean source recordings. This is acceptable — our analysis pipeline already handles noisy stems (noise floor filtering, dual-threshold vocal detection with hysteresis). The reconstructed plans will be approximate, which is fine for training data.

**Cost**: ~$0.05-0.10 per mashup on Modal × 100 = **$5-10**.
**Time**: ~3-5 min per mashup. With 3 concurrent: ~2-3 hours for 100 mashups.
**Effort**: ~2 hours coding (orchestration around existing functions).

---

## Step 4: Plan Reconstruction

**File**: `backend/scripts/mashup_pipeline.py` (subcommand: `reconstruct`)

The core novel logic. Converts analysis results into `RemixPlan` objects.

### Algorithm

Given `AudioMetadata`, `SongStructure` (sections, vocal_gaps, total_bars), `StemAnalysis`:

**1. Map detected sections to RemixPlan labels:**

Labels must match the inference-time vocabulary used by the LLM interpreter: `intro | verse | breakdown | drop | outro`.

| SectionInfo.label | → Section.label | Condition |
|-------------------|-----------------|-----------|
| intro | intro | — |
| verse | verse | — |
| chorus | drop | — |
| instrumental | breakdown | if low energy; drop if high |
| build | verse | — |
| breakdown | breakdown | — |
| outro | outro | — |

**2. Convert bar-based to beat-based boundaries:**
```python
Section.start_beat = SectionInfo.start_bar * 4
Section.end_beat = SectionInfo.end_bar * 4
```

**3. Estimate stem_gains from StemAnalysis.bar_rms:**
- For each section, compute mean per-bar RMS per stem within section bounds
- Normalize to [0, 1] relative to the stem's global max across all sections
- These become the section's `stem_gains`

**4. Infer transition type from energy trajectory at boundaries:**

| Energy change at boundary | → transition_in | transition_beats |
|---------------------------|-----------------|-----------------|
| Sharp increase (>3dB) | cut | 2 |
| Gradual change (<3dB) | crossfade | 4 |
| Drop then rise | crossfade | 8 |
| Default | crossfade | 4 |

All reconstructed plans must emit only `"fade" | "crossfade" | "cut"` — these are the only transition types defined in the `Section` model.

**5. Set transition_beats:** per the table above (cut→2, crossfade→4 or 8 for drop-then-rise patterns).

**6. Populate RemixPlan metadata:**
```python
RemixPlan(
    vocal_source="song_a",     # convention (mashup is treated as single source)
    tempo_source="average",
    key_source="none",         # no source song metadata
    start_time_vocal=0.0,
    end_time_vocal=duration,
    start_time_instrumental=0.0,
    end_time_instrumental=duration,
    sections=reconstructed_sections,
    explanation=f"Reconstructed from mashup: {title}",
    used_fallback=False,
)
```

```python
def reconstruct_plan(analysis: dict, mashup_id: str) -> RemixPlan:
    """Convert mashup analysis to a RemixPlan object."""

def reconstruct_all(analysis_dir: Path, output_dir: Path):
    """Batch reconstruction."""
```

**Output**: `backend/data/mashups/plans/{id}.json`

**Effort**: ~3 hours. This is the most novel code.

---

## Step 5: Negative Example Generation

**File**: `backend/scripts/mashup_pipeline.py` (subcommand: `generate-negatives`)

For each positive (mashup-derived) plan, generate 3 synthetic bad plans by degrading the good plan.

### Degradation Strategies

| Strategy | What it breaks | Which features detect it |
|----------|---------------|------------------------|
| **Shuffle sections** | Energy arc | `energy_template_corr_*`, `energy_peak_timing_score` |
| **Flat energy** | Dynamic range | `energy_contrast_index`, `energy_rise_fall_sanity` |
| **Vocals in bookends** | Vocal placement | `struct_vocal_placement_fit` |
| **Off-grid boundaries** | Groove coherence | `struct_phrase_boundary_hit_rate` |
| **Wrong peak placement** | Climax timing | `energy_peak_timing_score` |
| **No contrast** | Tension/release | `energy_contrast_index`, `energy_density_contour_smoothness` |

For each positive plan, randomly select 3 of 6 strategies. Apply one strategy per negative (single degradation, not stacked — keeps the signal clean for CatBoost to learn individual feature associations).

```python
def generate_negatives(plan: RemixPlan, n: int = 3) -> list[RemixPlan]:
    """Generate n degraded versions of a positive plan."""
```

**Output**: `backend/data/mashups/negatives/{id}_neg{0,1,2}.json`

**Effort**: ~2 hours. Mechanical transformations on dataclasses.

---

## Step 6: Feature Extraction + Dataset Building

**File**: `backend/scripts/mashup_pipeline.py` (subcommand: `build-dataset`)

Runs `extract_features()` on all positive and negative plans. Builds pairwise training CSV.

### Process

1. Load all positive plans from `data/mashups/plans/`
2. Load all negative plans from `data/mashups/negatives/`
3. For each plan: `extract_features(plan, meta_a=None, meta_b=None)` → feature vector
4. For each plan: `score_candidate(plan)` → heuristic dimension scores (7 values)
5. Build the training CSV: **each row = one plan** (positive or negative) with its full feature vector, a `label` (1 = positive, 0 = negative), and a `group_id` (mashup ID). CatBoost's `PairLogit` loss automatically forms all (positive, negative) pairs within each group during training — the CSV itself contains individual plans, not explicit pairs.
6. Tag with `manifest_version` from `get_manifest()`

### Output Schema

`backend/data/mashups/training/training_data.csv`

| Column | Type | Description |
|--------|------|-------------|
| group_id | str | Mashup ID (for request-level split) |
| plan_id | str | Unique plan identifier |
| label | int | 1 = positive (mashup-derived), 0 = negative (synthetic) |
| manifest_version | str | Feature manifest hash |
| struct_section_count | float | Feature value |
| ... | float | (all 28 Tier 1 features) |
| heuristic_arrangement_quality | float | Heuristic dimension score |
| ... | float | (all 7 heuristic dimensions) |
| heuristic_total | float | Overall heuristic score |

### Expected Dataset Size

Each mashup produces 1 positive + 3 negative plans = 4 rows in the CSV. CatBoost internally forms all (positive, negative) pairs within each group (1 × 3 = 3 pairs per mashup).

| Mashups | Plans/mashup | CSV rows | Internal pairs (CatBoost) |
|---------|-------------|----------|---------------------------|
| 50 | 4 (1+3) | 200 | 150 |
| 75 | 4 (1+3) | 300 | 225 |
| 100 | 4 (1+3) | 400 | 300 |

**Effort**: ~1.5 hours.

---

## Step 6b: Domain Alignment Validation

**Purpose**: Mashup-derived plans and inference-time candidate plans are generated by different processes. Before training, verify that their feature distributions are compatible — a large distributional shift could cause the model to learn signals that don't transfer to inference.

### Process

1. Generate a sample of inference-time candidate plans from the example songs (e.g., 20-50 candidates via the candidate planner).
2. Run `extract_features()` on each candidate plan.
3. For each of the 28 features, compute mean, std, and range for both distributions (mashup-derived plans vs. candidate plans).
4. Flag any feature where the distributions diverge by >1 standard deviation in mean.
5. Document flagged features and the magnitude of divergence.

### Expected Divergences

Some divergence is expected and acceptable:
- **Harmonic/tempo features** (Group 4): mashup-derived plans use neutral defaults; candidate plans have real values. These 6 features will diverge by design.
- **Section count / duration features**: mashup-derived plans reflect real songs (variable length); candidate plans are constrained by the planner.

### Label Vocabulary Consistency Check

Before comparing feature distributions, verify that reconstructed plans and candidate plans share the same `Section.label` vocabulary. Collect the set of unique labels from each population and assert they are identical. If the reconstructed plans emit labels that never appear in candidate plans (or vice versa), any features that depend on label string values (e.g., `struct_vocal_placement_fit`, `arrangement_family_match_score`) will have systematically different distributions — a train/serve skew that numeric distribution checks alone cannot detect. Fail this step if vocabularies diverge.

### Action on Flagged Features

- Flagged features do **not** block training — this step is diagnostic, not a gate.
- If >5 non-Group-4 features diverge significantly, consider: (a) adjusting the negative generation strategies to better match candidate plan characteristics, or (b) excluding the most divergent features from training.
- Document all findings in the training manifest output alongside eval metrics.

**Effort**: ~1 hour.

---

## Step 7: CatBoost Training

**File**: `backend/scripts/mashup_pipeline.py` (subcommand: `train`)

Minimal pairwise ranker training.

### Configuration

```python
catboost_params = {
    "loss_function": "PairLogit",
    "iterations": 500,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "od_type": "Iter",        # Early stopping
    "od_wait": 50,
    "random_seed": 42,
    "verbose": 50,
}
```

### Process

1. Load `training_data.csv`
2. Split by `group_id` (request-level split, 80/20)
3. Create CatBoost `Pool` objects with group IDs
4. Train with early stopping on validation set
5. Evaluate: pairwise accuracy, top-1 preferred rate on holdout
6. Save model artifact to `backend/data/models/taste_v1.cbm`
7. Save training manifest (config hash, data hash, feature manifest version, eval metrics)

### Model Loading

The trained model plugs directly into the existing stubs in `taste_model.py`:
- `load_model("data/models/taste_v1.cbm")` → loads the CatBoost model
- `score_with_model(feature_vectors)` → returns CatBoost scores
- `select_best()` automatically uses CatBoost scores when available, falls back to heuristic

### Dependency

`catboost` is added as an optional dependency:
```toml
[project.optional-dependencies]
ml = ["catboost>=1.2.0"]
```

Install for training: `uv add --optional ml catboost`

The production server doesn't need catboost installed — it loads the model artifact file and uses the heuristic fallback if catboost isn't available.

**Effort**: ~2 hours.

---

## Cost Summary

| Item | Cost |
|------|------|
| Modal GPU (stem separation, 100 mashups) | $5-10 |
| YouTube downloads | $0 |
| CatBoost training (CPU) | $0 |
| Storage | < $1 |
| **Total** | **~$5-10** |

## Effort Summary

| Step | Effort | Can Parallelize? |
|------|--------|-----------------|
| 1. Manifest curation | 1-2h (manual) | — |
| 2. Download script | 1h | — |
| 3. Analysis pipeline | 2h | Yes (with Step 4) |
| 4. Plan reconstruction | 3h | Yes (with Step 3) |
| 5. Negative generation | 2h | — |
| 6. Dataset building | 1.5h | — |
| 6b. Domain alignment validation | 1h | — |
| 7. CatBoost training | 2h | — |
| **Total coding** | **~12-14h** | |

---

## File Layout

```
backend/
  scripts/
    mashup_pipeline.py            # All 7 steps as subcommands
  data/
    mashup_manifest.json          # Step 1: curated URLs
    mashups/
      raw/                        # Step 2: downloaded WAVs
      stems/                      # Step 3: separated stems
      analysis/                   # Step 3: analysis JSONs
      plans/                      # Step 4: reconstructed RemixPlans
      negatives/                  # Step 5: degraded plans
      training/
        training_data.csv         # Step 6: pairwise feature matrix
    models/
      taste_v1.cbm                # Step 7: trained model
```

---

## Existing Modules Reused

| Module | What we reuse |
|--------|---------------|
| `services/youtube.py` | `download_youtube_audio()` for mashup downloads |
| `services/separation.py` | `separate_stems()` for mashup stem separation |
| `services/analysis.py` | `analyze_audio()`, `analyze_stems()`, `detect_key()` |
| `services/taste_features.py` | `extract_features()`, `get_manifest()` |
| `services/taste_model.py` | `score_candidate()`, `load_model()`, `score_with_model()` |
| `services/taste_constraints.py` | `validate_candidate()` for sanity-checking reconstructed plans |
| `models.py` | `RemixPlan`, `Section`, `AudioMetadata`, `SongStructure`, `StemAnalysis` |

---

## Dependencies and Merge Order

Several modules listed above do not yet exist on `main`. The mashup pipeline branch **must** be based on or merged after `feat/taste-training-integration`, which introduces the following required modules:

| Module (from `feat/taste-training-integration`) | Required by steps |
|--------------------------------------------------|-------------------|
| `services/taste_features.py` → `extract_features()`, `get_manifest()` | Step 6 |
| `services/taste_model.py` → `score_candidate()`, `load_model()`, `score_with_model()` | Step 6, Step 7 |
| `services/taste_constraints.py` → `validate_candidate()` | Step 4 (validation), Risks mitigation |
| `models.py` → `RemixPlan`, `Section` (with taste-related fields) | Steps 4, 5, 6 |

**Merge order**: `feat/taste-training-integration` must land on `main` (or be merged into the mashup pipeline branch) before any step beyond Step 3 can run. Steps 1-3 have no dependency on that branch and can proceed independently.

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Mashup stem separation is noisy | Reconstructed plans may have inaccurate vocal activity / energy | Existing analysis pipeline handles noise (floor filtering, hysteresis). Plans are approximate — acceptable for training data. |
| Popular mashups taken down | Fewer than 50 usable mashups | Over-curate manifest (120+ URLs). Download script skips failures gracefully. |
| Synthetic negatives too easy | CatBoost learns trivial signal, doesn't generalize | Single degradation per negative (not stacked). Multiple strategies ensure feature diversity. Validate on held-out mashups. |
| 22/28 features is enough? | Missing harmonic/tempo features may limit model | Phase 0 heuristic scorer covers harmonic/tempo. CatBoost learns structural taste. Add source song metadata to manifest entries where known for full feature coverage. |
| CatBoost overfits on 150-300 pairs | Model doesn't generalize | Early stopping, L2 regularization, depth=6 cap. Request-level split prevents leakage. Monitor train/val gap. |
| Reconstructed plans don't match our candidate format | Feature values systematically differ between mashup-derived and candidate-generated plans | Run `validate_candidate()` on reconstructed plans and drop those that fail structural constraints. Ensures training data matches inference-time distribution. |
| Training/inference distribution mismatch | Model learns feature patterns from mashup-derived plans that don't transfer to candidate-generated plans at inference time | Step 6b domain alignment validation: compare feature distributions between mashup-derived and candidate plans before training. Flag features with >1 std dev shift. Inform feature selection and document in training manifest. |

---

## Success Criteria

- CatBoost model trained on 150+ pairwise rows from 50+ mashups
- Pairwise accuracy on holdout > 70%
- Model preference vs heuristic-only: >= 60% in synthetic A/B
- Model loads and scores in < 80ms (within taste stage latency budget)
- No regression when model unavailable (heuristic fallback works)

---

## Future Enhancements (Not in Scope)

- Add source song URLs to manifest for full harmonic/tempo features
- Expand to 200+ mashups with active learning (prioritize uncertain pairs)
- Layer in implicit user signals (regenerate rate, replay rate) post-launch
- Per-genre model calibration once genre distribution is balanced
- Add `filter_sweep` as a new transition type in the `Section` model, then use it in plan reconstruction for drop-then-rise energy patterns
- LLM-as-judge for additional structural preference labels (near-free, complements mashup data)
