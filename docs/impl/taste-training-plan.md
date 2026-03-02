# Lightweight Taste Training Plan: Teaching the LLM to Mashup Better

revision: 2

## Executive Summary

This plan consolidates two existing documents into one actionable training strategy:
- **mixTaste.md**: The comprehensive mixing intelligence spec (tempo, arrangement, harmonic, EQ/dynamics, genre rules)
- **Taste Model Synthesis**: The ML architecture for a CatBoost pairwise ranker

The goal is a lightweight training run that improves the LLM's ability to generate high-quality `RemixPlan` objects -- the structured arrangement decisions that determine which stems play when, at what levels, with what transitions. The training does NOT touch the DSP/rendering pipeline; it only improves plan selection.

**Core thesis**: The biggest quality lever is not better audio processing -- it's better *decisions* about what to play when. A tasteful arrangement with basic processing beats a bad arrangement with pristine mastering every time.

---

## 1. What We're Training

### 1.1 The Decision Surface

The LLM currently generates a single `RemixPlan` via deterministic fallback. The trained system will:

1. **Generate 8-12 candidate plans** by varying arrangement knobs
2. **Hard-filter** invalid candidates (constraint violations)
3. **Score** surviving candidates with a CatBoost pairwise ranker
4. **Select** the best plan and render it

### 1.2 What a RemixPlan Controls

| Decision | Current | After Training |
|----------|---------|----------------|
| Section count & boundaries | Fixed 5-section template | 4-7 sections, varied by song analysis |
| Vocal entry timing | Always bar 16 | Adaptive: bar 8, 16, or 24 based on energy/genre |
| Energy arc | Flat-ish | Template-matched (Classic/EDM/Hip-Hop/DJ-Lift arc) |
| Breakdown placement | Fixed at ~60% | 50-70%, optimized per pair |
| Transition style | Always crossfade | Cut, crossfade, filter sweep, silence gap -- genre-appropriate |
| Per-section gains | Conservative uniform | Dynamic, section-appropriate |
| Peak moment placement | Accidental | Intentional, 55-80% of timeline |

### 1.3 What We're NOT Training

- No end-to-end audio generation model
- No large audio encoder fine-tuning
- No changes to the DSP/rendering pipeline
- No per-user personalization (MVP)
- No real-time GPU inference

---

## 2. Architecture

### 2.1 Candidate Generation

Generate candidates by controlled variation, not random sampling.

**Arrangement families (4 for MVP)**:
- **Standard arc**: intro -> build -> main -> breakdown -> peak/outro
- **Hook-first**: short intro -> vocal hook -> verse -> peak -> outro
- **DJ Lift**: build -> vocal in -> peak -> vocal out -> outro
- **Quick Hit**: intro -> main vocal block -> short release

**Variation knobs** (crossed to produce candidates):
| Knob | Values | Notes |
|------|--------|-------|
| Section count | 4, 5, 6, 7 | Conditioned on duration (60-120s) |
| Vocal entry | 8, 16, 24 beats | Earlier for hook-first/DJ-lift, later for standard |
| Breakdown position | 50%, 60%, 70% of timeline | |
| Transition style | cut, crossfade, filter sweep, silence gap | Genre-biased default; filter sweep for EDM/Afrobeats, cut for hip-hop/grime |
| Transition length | 2, 4, 8 beats | Bounded by section length |
| Vocal/instrumental gain delta | -6dB, -3dB, 0dB, +3dB steps | Finer control for vocal prominence |

**Deduplication**: Hash on section labels + boundary beats + transition types + coarse gain bins. After deduplication, if fewer than 8 candidates survive, backfill with safer variants from underrepresented families. Cap at 16, target 8-12.

### 2.2 Hard Constraints (Must Pass Before Scoring)

These are non-negotiable rules derived from mixTaste.md and DJ rules. Any candidate violating these is rejected before scoring.

| Constraint | Source | Threshold |
|------------|--------|-----------|
| Contiguous non-overlapping sections | Structure validity | Monotonic boundaries, start=0 |
| Section minimum length | DJ rules | >= 8 beats for major sections, >= 4 beats minimum |
| Beat grid alignment | Phrasing rules | Boundaries on 4-beat multiples |
| MVP source split | Product constraint | One song = vocals, other = instrumentals |
| Tempo stretch safety | mixTaste 1.2 | Drums <= 12%, vocals <= 35%, other <= 40% |
| Pitch shift safety | mixTaste 3.1 | <= 4 semitones |
| Transition bounds | DJ rules | transition_beats <= half section length |
| True peak ceiling | Audio hygiene | <= -1.0 dBTP |
| LUFS window | Audio hygiene | Genre-conditional: lo-fi -16 to -12, EDM -12 to -9, default within 2 dB of -12 |
| LRA floor | Audio hygiene | Loudness Range >= 4 dB (prevents wall-of-sound) |
| Contrast requirement | DJ rules | At least one contrast event before peak with magnitude: stem count drop >= 2 OR energy drop >= 20% |
| No dual lead vocals | MVP constraint | One lead vocal at a time |
| Outro quality | DJ rules | Final section must be outro-labeled, >= 8 beats, energy below peak |
| Stem quality gate | Audio quality | No solo vocal sections when stem separation quality is below threshold (cross-bleed ratio < 0.7) |

### 2.3 Feature Engineering

Features are organized into 3 tiers by computational cost. **Phase 1 uses only Tier 1 features (20-30 total)**. Tier 2 added in Phase 2 at 1500+ labels. Tier 3 deferred to Phase 3 preview reranker.

**Tier 1: Metadata-only (computable from plan + cached song analysis, no rendering)**

**Group 1: Structure (10-12 features)**
- Section count, mean/std section duration, min section length
- Phrase boundary hit rate (% boundaries on 16-beat multiples)
- Section validity ratio (% sections >= 8 beats with monotonic bounds)
- Vocal placement fit (intro/outro vocal mute compliance)
- Arrangement family match score

**Group 2: Energy Arc (8-10 features)**
- Section energy profile correlation vs template (classic/EDM/hip-hop/DJ-lift)
- Peak timing score (max energy section at 55-80% of timeline)
- Rise/fall sanity (verse->chorus delta in expected range +2 to +6 dB)
- Contrast index (normalized variance of stem activity across sections)
- Density contour smoothness (mean absolute change in active-stem count)

**Group 4: Harmonic/Tempo Risk (6-8 features)**
- Camelot distance after pitch decision (0 = best, >3 penalized heavily)
- Absolute pitch shift semitones
- Total tempo stretch amount (per stem type)
- Stretch direction penalty (slow-down penalized more than speed-up)

**Group 7: Prompt Fit (5-8 features)**
- Energy level mismatch (prompt intent vs plan energy profile)
- Structural preference match (prompt implies "build" vs "immediate")
- Vocal prominence alignment (prompt emphasis vs vocal duty cycle)
- Genre compatibility score (from cross-genre combination matrix)
- Chaos flag (prompt asks for experimental vs plan conservatism)

**Tier 2: Proxy features (require cached analysis data, not full rendering)**

**Group 3: Vocal Clarity Risk (8-10 features)**
- Expected vocal-to-masker ratio in 2-5 kHz (from pre-computed frequency profiles)
- Masking duty cycle (% vocal-active frames with predicted VMR < -3 dB)
- Vocal exposure window ratio (total vocal-active time / remix duration)
- Sibilance harshness proxy (6-10 kHz vocal excess over mid band)

**Group 6: Groove Coherence (15-20 features)**
- Vocal onset-to-nearest-beat offset median + IQR
- Downbeat alignment score
- Swing mismatch risk (genre swing profiles: source vs target delta)
- Kick-snare timing relationship compatibility
- Groove template correlation (source groove profile vs target beat grid)
- Accent pattern alignment (rhythmic density match between vocal flow and instrumental groove)
- Pocket depth proxy (onset deviation distribution shape -- tight vs loose)

**Group 8: Spectral Balance (8-10 features)**
- Low-mid energy ratio (200-500 Hz accumulation risk from overlapping stems)
- Bass collision index (sub-200 Hz energy overlap between instrumental bus stems)
- Sub-bass mono correlation proxy
- Spectral tilt (overall balance bright vs dark)
- Stem separation quality score (cross-bleed ratio from Demucs, per stem)

**Tier 3: Render-required features (need audio rendering, deferred to Phase 3)**

**Group 5: Transition Quality (10-12 features)**
- Boundary loudness jump (LUFS delta 500ms pre/post boundary)
- Spectral discontinuity (flux z-score at boundaries vs local median)
- Click risk (max sample derivative spike near boundaries)
- Reverb tail truncation risk
- Harmonic center shift at boundary
- Crossfade low-frequency bump
- Timbral continuity score

**Group 9: Moment Quality (5-8 features, Phase 3)**
- Coincidental melodic alignment score (chroma correlation at key moments)
- Energy contour correlation between vocal and instrumental at peaks
- Lyric-music alignment proxy

**Feature versioning**: Every feature extraction run produces a versioned manifest (feature names, computation method, version hash). Training data always tagged with feature version. Never mix versions in one training set.

### 2.4 Scoring Model

**Model**: CatBoost pairwise ranker (single model for MVP)

**Objective**: Pairwise ranking loss (RankNet-style logistic on score difference)

**MVP scoring approach**: Feed all features (including heuristic subscores as input features) directly to CatBoost. Let the model learn the optimal combination rather than manually tuning weights between heuristic and learned components. The hybrid formula `score(c) = I[hard_pass(c)] * (w*x(c) + r_theta(c)) - lambda*risk(c)` is available as a fallback if CatBoost is unavailable, using hand-tuned heuristic weights.

**Calibration**: Fit Platt scaling on validation set after training. Define tie-break threshold tau in probability space (e.g., P(winner) < 0.6 triggers lower-risk preference). Report calibration curves per training run.

**Selection policy**: Pick top-1; if margin(top1, top2) < tau, choose lower-risk candidate.

**Fallback**: If model unavailable or all scored candidates below quality threshold, use deterministic baseline heuristic scorer.

**Model loading**: Load CatBoost model at server startup. Run dummy prediction to warm cache. Implement hot-reload via file mtime watching for model updates.

---

## 3. Data Collection Strategy

### 3.1 Label Sources (3 tiers)

**Tier 1: Auto-generated labels (available immediately)**
- Hard constraint failures as automatic negatives
- Heuristic rubric scores from mixTaste rules (used as weak labels, NOT mixed directly with human labels -- see 3.5)
- Deterministic "obviously bad" rejection labels

**Tier 2: Human pairwise labels (primary training signal)**
- A/B preference: "Which remix sounds better?" on 15-25s preview clips
- Per-rubric item scores (1-5) with failure tags
- Target: 500 pairwise labels in first 2-3 weeks (Phase 1 starting point)

**Phase 1 feature count discipline**: With 500 labels, use only 20-30 Tier 1 features. Scale to full feature set (80+) only at 1500+ labels in Phase 2.

**Tier 3: Implicit behavioral signals (post-launch, weak)**
- Replay rate, listen-through %, regenerate rate
- Treat as low-confidence unless corroborated by human labels

### 3.2 Labeling Protocol

**Setup requirements**:
- Closed-back headphones or calibrated monitors (no laptop speakers)
- Normalized playback monitor level per session
- **Loudness-matched pairs**: All A/B pairs normalized to -12 LUFS (+/- 0.5 dB) before playback. Verify no LUFS-preference correlation in label data.
- Blind to plan details and model version

**Label format**:
- Pairwise A/B winner + confidence (low/med/high)
- Per rubric item: integer 1-5 with anchors
- Failure tags (multi-select): `muddy`, `vocal_buried`, `clashy_key`, `awkward_transition`, `fatiguing`, `timing_off`, `low_end_muddy`, `groove_off`

**Bias controls**:
- **Position bias**: Randomize A/B presentation order for every pair. Present 5-10% of pairs in both orders to measure and correct for position bias.
- **Loudness bias**: Loudness-matched pairs as above.
- **Tie handling**: Allow "no preference / tie" as explicit option. Ties excluded from pairwise training but included in calibration analysis.

**Quality controls**:
- 10% repeated pairs to measure rater consistency
- **Minimum inter-rater agreement**: Cohen's kappa >= 0.4 before training begins. If below threshold, refine rubric anchors and re-calibrate raters.
- 3 calibration examples (bad/mid/good) at session start
- Minimum 2-3 raters on seed benchmark set
- Weekly disagreement review to improve rubric
- Confidence-weighted loss: high-confidence labels weighted 1.0, medium 0.7, low 0.4

### 3.3 The Heuristic Rubric (Weak Labels)

Derived from mixTaste.md, these generate scores before any human labels exist:

| Dimension | Weight | What It Measures |
|-----------|--------|-----------------|
| Arrangement quality | 15% | Section pacing, phrasing, stem density choices |
| Energy arc | 15% | Builds/releases feel intentional |
| Vocal intelligibility | 15% | VMR in 2-5 kHz, masking risk |
| Harmonic fit | 15% | Key distance, pitch shift amount |
| Transition quality | 15% | Boundary smoothness, no jolts, timbral continuity |
| Groove coherence | 15% | Beat alignment, swing compatibility, pocket |
| Loudness/fatigue | 10% | LUFS accuracy, crest factor, LRA |

(Weights rebalanced from R1: transition and groove elevated from 10% each to 15% each; arrangement split reduced.)

### 3.4 Genre-Specific Taste Rules (from mixTaste.md)

The training data and features must respect genre-specific conventions:

**Tempo compatibility** (per mixTaste Section 1):
- Direct match (< 8% stretch) is always preferred
- Half/double-time is natural and widely understood
- Polyrhythmic only at creativity 3+ (experimental)
- Speed-up produces fewer artifacts than slow-down

**Arrangement conventions** (per mixTaste Sections 2, 5):
- Hip-hop: 8/16-bar phrases, ad-libs fill gaps on beats 2/4
- EDM: vocals in breakdowns only, drops are instrumental
- R&B: preserve laid-back timing, sustain through barlines
- Rock: guitar takes vocal frequency range during solos
- DnB: half-time vocals over full-time drums
- Afrobeats: percussion builds over 16-32 bars, plateau energy
- Reggaeton: near-flat high-energy curve driven by dembow

**Mixing conventions** (per mixTaste Sections 3, 4):
- Key compatibility via Camelot wheel (adjacent keys safe)
- EQ carving: vocal pocket at 2-5 kHz, instrumental ducks
- Sidechain: pattern-aware (4/4 standard, syncopated simplified, breakbeat uses automation instead)

### 3.5 Label Source Mixing Strategy

**Critical**: Heuristic and human labels are NOT mixed directly. The training strategy is:

1. **Phase 0-1 (bootstrap)**: Train CatBoost on heuristic labels with source tag `heuristic`. Weight at 0.2.
2. **Phase 1 (human labels arrive)**: Add human labels with source tag `human`. Weight at 1.0.
3. **Validation**: Always evaluate on human-only holdout. Report human-only pairwise accuracy separately from mixed accuracy.
4. **Residual learning**: Optionally train on residual (human preference minus heuristic score) to learn where human taste diverges from rules.
5. **Data splits**: All splits at request level (no request appears in multiple splits). This prevents within-request pair leakage that inflates metrics.

---

## 4. Training Loop

### 4.1 Phase 0: Foundation (Days 1-7)

| Day | Deliverable | Status |
|-----|------------|--------|
| 1 | `candidate_planner.py` -- expand fallback plan into 8-12 variants across 4 families with structure hash dedupe | ✅ PR #49 |
| 2 | Hard constraint validation skeleton with failure reason codes + enum, wired into pipeline | ✅ PR #50 |
| 3 | Feature extraction module (first 20-30 Tier 1 features) with versioned manifest + logging schema | ✅ PR #51 + #53 |
| 4 | Baseline weighted heuristic scorer, integrated behind `ab_taste_model_v1` flag with timeout wrapper | ✅ PR #52 + #53 |
| 5 | Minimal local A/B labeling script (loudness-matched, position-randomized) + first batch of 25-40 remix requests | ☐ |
| 6 | First CatBoost pairwise model trained on heuristic + auto labels | ☐ |
| 7 | End-to-end smoke tests, latency validation, go/no-go report | ☐ |

**Week 1 exit criteria**:
- [x] Candidate generation (4 families) + hard constraints live behind feature flag
- [x] Baseline scoring selects a winner deterministically
- [x] Telemetry captures candidates, features, scores, selection rationale (full logging schema)
- [ ] Labeling loop functional with initial pairwise data (loudness-matched, position-randomized)
- [x] No regression when feature flag off
- Taste stage wrapped in timeout with circuit breaker

**Training run manifest** (produced by every training run):
- Config hash (hyperparameters, feature version, label sources)
- Data hash (training set fingerprint)
- Feature manifest (names, versions, computation methods)
- Model artifact (versioned, with metadata)
- Eval metrics (all offline metrics on validation set)

### 4.2 Phase 1: First Real Model (Weeks 2-3)

- Collect 500 human pairwise labels (rater kappa >= 0.4 verified)
- Train CatBoost pairwise model on 20-30 Tier 1 features + heuristic subscores as inputs
- Label source weights: human=1.0, heuristic=0.2, auto-negative=0.5
- Validate on frozen holdout (request-level split, 100+ tasks, 3-5 candidates each)
- Report human-only holdout accuracy separately
- Ship behind feature flag
- A/B: model winner vs fallback on held-out tasks, target >= 65% preference
- Log full flag configuration (all ab_* flags) with every training data point

### 4.3 Phase 2: Iteration (Weeks 4-6)

- Active-learning label queue (show uncertain pairs to raters)
- Scale to Tier 1 + Tier 2 features (40-60 total) once 1500+ labels collected
- Add Group 6 (expanded groove), Group 8 (spectral balance) features
- Per-genre calibration (monitor for genre bucket regressions)
- Retrain cadence: trigger-based (200+ new labels or drift detected), not fixed weekly. Use faster cadence (daily) in early Phase 2.
- Increase frozen holdout to 250+ tasks for per-genre statistical power

### 4.4 Phase 3: Optional Enhancements (Weeks 7-12)

- Top-2 preview reranker (render short previews of top 2, pick winner using Tier 3 render-required features)
- Moment quality features (Group 9)
- Personalization exploration
- Text/plan cross-encoder experiment (Family 2 model)
- Full feature set (80+ features) once 3000+ labels

---

## 5. Integration

### 5.1 New Files

| File | Purpose |
|------|---------|
| `backend/src/musicmixer/services/candidate_planner.py` | Generate 8-12 candidate plans across 4 arrangement families |
| `backend/src/musicmixer/services/taste_model.py` | CatBoost ranker + heuristic fallback scorer + model loading/hot-reload |
| `backend/src/musicmixer/services/taste_features.py` | Tiered feature extraction module with versioned manifests |
| `backend/src/musicmixer/services/taste_constraints.py` | Hard constraint validation with failure code enum |
| `backend/scripts/label_ab.py` | Local A/B labeling script (loudness-matched, position-randomized) |
| `backend/scripts/train_taste.py` | Training script with manifest output and reproducibility |

### 5.2 Pipeline Insertion

```
Current:  step 4 (generate_fallback_plan) -> step 5+ (render)
After:    step 4a (generate candidates) -> step 4b (hard filter) -> step 4c (score + select) -> step 5+ (render)
```

Gated behind `TASTE_MODEL_ENABLED` flag. When off, existing fallback behavior is unchanged.

**Timeout/circuit breaker**: Entire taste stage wrapped in `ThreadPoolExecutor` future with 400ms hard timeout. Circuit breaker: 5 consecutive fallbacks disables taste stage for 10 minutes (configurable). On any error/timeout, fall back to deterministic baseline in <= 50ms.

**Flag interaction**: Log full `ab_*` flag configuration with every request and training data point. When any `ab_*` flag changes, re-evaluate model on holdout and retrain if metrics shift by > 2%.

### 5.3 Logging Schema

Per request, log (as Pydantic model):
- `request_id`, prompt, song A/B metadata (including raw analysis)
- `feature_version` (manifest hash)
- `model_version` (artifact hash)
- `flag_config` (all ab_* flags + TASTE_MODEL_ENABLED state)
- All candidate plans (serialized) with structure hashes
- Hard constraint pass/fail per candidate with failure reason enum
- Feature vectors (versioned) per candidate
- Ranker scores + score margins + calibrated probabilities
- Selected candidate ID and selection rationale
- Runtime metrics (generation latency, scoring latency, total taste-stage time)
- Fallback trigger flag + reason if applicable

---

## 6. Performance Targets

| Metric | Target |
|--------|--------|
| Candidate generation | <= 120ms CPU |
| Hard filtering | <= 20ms CPU |
| Feature extraction (Tier 1, 20-30 features) | <= 50ms CPU |
| Scoring (CatBoost) | <= 80ms CPU |
| Total taste-stage P95 | <= 300ms |
| Taste-stage hard timeout | 400ms |
| Fallback recovery | <= 50ms on any error |
| Model preference vs fallback | >= 65% in blind A/B |
| Constraint violation rate | 0% (hard gate) |
| Fallback trigger rate | <= 1% |
| CatBoost cold-start (server startup) | <= 500ms (model load + warm-up prediction) |

---

## 7. Cost

| Component | Cost Estimate |
|-----------|---------------|
| Inference (CPU ranking) | Near-zero marginal |
| Training (CPU, CatBoost, trigger-based) | $0-3/run on 8 vCPU |
| GPU (stem separation, existing) | Unchanged |
| Labeling (friends, manual) | Time only |
| Storage (features, checkpoints, logs) | < $5/month S3 |
| **Total additional monthly** | **< $20** |

---

## 8. Evaluation Framework

### 8.1 Offline Metrics
- Pairwise accuracy on holdout (human-only and mixed)
- **Top-1 preferred rate** vs fallback baseline (primary metric -- only top-1 is rendered)
- **Regret@1**: How often does the model's top-1 lose to any other candidate in pairwise comparison
- Calibration curves (Platt-scaled probabilities vs observed win rates)
- Per-genre performance spread

### 8.2 Safety/Regression Checks
- Constraint violation rate (must be 0%)
- Per-genre regression monitoring
- Correlation checks: score vs loudness, stem density, transition count, tempo stretch amount, section count (detect reward hacking)
- Wall-of-sound detection (gain headroom utilization, mute ratio)
- Feature importance drift tracking across training runs

### 8.3 Online Metrics (Post-Launch)
- Regenerate rate (lower = better)
- Listen-through proxy (% of remix played)
- Replay/keep rate
- A/B preference in production (randomized)

---

## 9. What Not To Build Yet

- End-to-end generative audio model
- Large audio-text multimodal fine-tuning
- Real-time GPU reranking
- Per-user personalization
- Complex model registry platform
- Full RL/reward-model loop from implicit metrics
- Architecture rewrite of renderer/DSP pipeline

---

## 10. Risk Register

| Risk | Mitigation |
|------|-----------|
| Too few human labels early on | Heuristic weak labels bootstrap (weighted 0.2); CatBoost handles mixed-quality labels; start with 20-30 features |
| CatBoost ceiling too low | Feature engineering focus; cross-encoder (Family 2) ready as upgrade path |
| Candidate generation misses good plans | 4 arrangement families for MVP; monitor fallback-beats-model rate; expand families based on error analysis |
| Genre bias in training data | Per-genre holdout monitoring; stratified sampling; genre-conditional LUFS targets |
| Latency budget exceeded | Tier 1 features only in Phase 1; feature caching; 400ms hard timeout |
| Reward hacking (loud = "better") | Loudness-matched labeling; correlation checks; rubric diversity |
| Stem quality variation | Stem quality features + interaction terms; hard constraint against solo vocal with low-quality stems |
| Label noise from rater disagreement | Kappa >= 0.4 threshold; confidence-weighted loss; calibration examples |
| Flag interaction invalidates model | Full flag logging; re-eval on flag change; retrain trigger |

---

## Appendix A: R1 Review Changes

Changes made from Round 1 expert review (ML Scientist, ML Engineer, Sound Engineer, Mixing Master):

| Issue | Fix Applied |
|-------|------------|
| C1: Features mislabeled as metadata-only | 3-tier feature system (metadata/proxy/render-required) with phase gates |
| C2: No low-frequency features | Added Group 8 (Spectral Balance) with mud, bass collision, sub-bass |
| C3: Label mixing strategy undefined | Section 3.5 defines source weighting, residual learning, request-level splits |
| C4: Stem quality not modeled | Group 8 includes stem quality score; hard constraint added for solo vocal gate |
| C5: Only 2 arrangement families | Expanded to 4 (added DJ Lift, Quick Hit) |
| C6: Groove features too thin | Group 6 expanded to 15-20 features covering pocket, density, accent patterns |
| M1-M12 | Train/test splits, rater agreement, loudness normalization, position bias, logging schema, timeout/circuit breaker, model loading, flag interactions, calibration, feature count discipline, transition coverage, genre LUFS |
| M13-M23 | Feature versioning, holdout size, retrain cadence, correct metrics, reward hacking, scoring simplification, prompt fit, rubric weights, reproducibility, contrast strength, outro quality |
