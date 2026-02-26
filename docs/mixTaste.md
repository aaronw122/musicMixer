# Mix Taste: AI-Driven Mixing Intelligence for musicMixer

## Executive Summary

This document defines how musicMixer's AI makes mixing decisions -- the rules, heuristics, and creative intelligence that turn two uploaded songs and a text prompt into a cohesive remix. It covers five domains: tempo intelligence (how to reconcile different BPMs), arrangement intelligence (when to bring stems in and out), harmonic intelligence (key matching and spectral awareness), mixing technique (EQ, dynamics, transitions), and genre-specific conventions. Together, these form the "taste" layer -- the difference between an AI that blindly layers audio and one that produces remixes that actually sound good.

Every section is written to be implementable. Concrete numbers (dB, BPM, Hz, ms, percentages) are provided wherever research supports them. The document serves both as a reference for understanding mix decisions and as a specification for the mixing engine.


### MVP Scope (Day 3-4 Sprint)

This document describes the full mixing intelligence system. **Not all of it is in scope for the Day 3-4 sprint.** The following callout delineates what to implement now vs. what remains aspirational.

**In scope (implement Day 3-4):**
- LLM-powered stem selection and arrangement planning (Section 8.2)
- Tempo matching and time-stretching (Section 1)
- Key detection and pitch-shifting (Section 3.1)
- Basic EQ carving and vocal pocket (Section 4.2)
- Sidechain/ducking automation (Section 4.3)
- Per-stem LUFS normalization and gain staging (Section 4.3)
- Section-level volume offsets (Section 2.3)
- Energy curve templates -- simplified version in LLM prompt (Section 2.3)
- Final LUFS normalization and true peak limiting (Section 8.7)

**MVP constraint:** The MVP constrains stems to vocal-source vs. instrumental-source (vocals from one song, all instrumentals from the other). Fine-grained per-stem cross-song mixing (e.g., kick from Song A, hi-hats from Song B) described in Section 6.2 is a future enhancement.

**Aspirational (future iterations):**
- Genre detection and genre-specific rule loading (Section 5)
- Swing/groove detection and matching (Section 6.2)
- Psychoacoustic masking and Bark/ERB band analysis (Section 3.2)
- Moments engineering (Section 7.2)
- Emotional arc planning via valence-arousal model (Section 7.1)
- Per-beat phase correction (Section 8.6 -- MVP uses per-section correction)
- Polyrhythmic transition support (Section 1.1 Tier 3)
- Creativity dial integration (Section 7.3)
- Dynamic panning, M/S processing, stereo correlation monitoring

Developers should treat the entire document as a reference for understanding mixing decisions, but only implement the "In scope" items during the sprint. **Each section heading includes a scope marker** (`Sprint Scope` or `Future/Post-Sprint`) indicating what to build now vs. what is aspirational reference material.


## 1. Tempo Intelligence  <!-- Sprint Scope: Section 1.1-1.3 (tempo matching + time-stretching) -->

### 1.1 BPM Compatibility Framework

#### Genre-Specific Tempo Ranges

| Genre | Typical BPM Range | Center BPM | Notes |
|-------|-------------------|------------|-------|
| Ambient / Downtempo | 60-100 | 80 | Often beatless or very sparse |
| Lo-Fi / Chillhop | 70-90 | 80 | Slow, relaxed grooves |
| R&B (modern) | 60-80 | 70 | Classic R&B can reach 100 |
| Reggae / Dub | 60-90 | 75 | One-drop rhythm defines the genre |
| Hip-Hop (boom-bap) | 85-100 | 92 | Traditional sample-based |
| Hip-Hop (modern/trap-influenced) | 60-80 | 70 | Slower tempos with double-time hi-hats |
| Trap | 130-150 (felt as 65-75) | 140 | Almost always felt in half-time |
| Afrobeats | 100-120 | 108 | Polyrhythmic percussion, melodic bass |
| Amapiano | 110-115 | 113 | Log drum driven, piano stabs |
| Reggaeton | 95-105 | 100 | Very consistent dembow rhythm |
| Funk | 90-120 | 105 | Groove-driven, syncopated |
| Pop | 100-130 | 115 | Widest range; follows trends |
| Rock | 110-140 | 125 | Punk pushes 160+; ballads drop lower |
| House (Deep, Tech) | 115-130 | 124 | Most modern house sits 124-128 |
| UK Garage | 130-140 | 135 | 2-step rhythms |
| Trance | 125-150 | 138 | Psytrance pushes 140-150 |
| Dubstep | 138-142 (felt as ~70) | 140 | Half-time feel despite ~140 actual BPM |
| Techno | 120-145 | 132 | Industrial/hard techno trends 140+ |
| Drum & Bass / Jungle | 160-185 | 174 | Genre defined by this range |
| Hardcore / Gabber | 160-200+ | 180 | Can exceed 200 in extreme subgenres |

#### Tempo Compatibility Tiers

**Tier 1 -- Direct Match (best quality, simplest)**
Tracks whose BPMs are within ~8% of each other. Minimal stretching required.

| BPM Range | Compatible With |
|-----------|----------------|
| 118-132 | Each other (house, techno, pop overlap) |
| 85-100 | Each other (hip-hop, funk overlap) |
| 160-180 | Each other (DnB, jungle, hardcore overlap) |

**Tier 2 -- Halftime/Doubletime (natural, widely understood)**
Exact 1:2 ratio pairs, allowing ~8% stretch tolerance.

| Slow BPM | Fast BPM | Common Genres |
|----------|----------|---------------|
| 60-70 | 120-140 | R&B/downtempo to house/trance |
| 70-80 | 140-160 | trap(half)/hip-hop to dubstep/trance |
| 80-92 | 160-184 | hip-hop/funk to DnB/jungle |
| 85-95 | 170-190 | boom-bap to DnB |
| 62-70 | 124-140 | R&B to house/techno |

**Tier 3 -- Polyrhythmic (experimental only -- creativity levels 3+ only)**
2:3 and 3:4 ratio pairs. These almost always sound bad in typical mashup contexts. At creativity levels 0-2, if tempos don't match via direct or half/double-time relationships, use non-rhythmic stems from one song and commit to the other song's tempo entirely. Only attempt polyrhythmic alignment when the user explicitly requests experimental results.

| BPM A | BPM B | Ratio | Example Genres |
|-------|-------|-------|----------------|
| 80 | 120 | 2:3 | Hip-hop to house |
| 90 | 120 | 3:4 | Hip-hop to house |
| 100 | 150 | 2:3 | Reggaeton to trance |
| 88 | 132 | 2:3 | Hip-hop to techno |
| 120 | 160 | 3:4 | House to DnB |

**Tier 4 -- Incompatible (avoid or use transition techniques)**
Pairs that do not fit any clean ratio. Examples: 100+174 (ratio ~1:1.74), 75+132 (~1:1.76), 110+174 (~1:1.58).

#### The 110+160 BPM Scenario (Detailed Analysis)

This is a genuinely difficult pair. Here is how the AI should evaluate it:

| Option | Strategy | Stretch Required | Verdict |
|--------|----------|-----------------|---------|
| 1 | Meet in the middle at ~135 | A: +22.7%, B: -15.6% | BAD -- total stretch too high |
| 2 | Half-time Song B (80), match at 110 | 80 and 110 are not compatible | NOT VIABLE directly |
| 3 | Bring A down to 80 (half of 160) | A: -27.3% | TOO MUCH stretch |
| 4 | 2:3 relationship (110 x 1.5 = 165) | B: +3.1% from 160 | VIABLE but polyrhythmic |
| 5 | Stretch A up to 160 | A: +45.5% | BAD for full mix; acceptable for vocal-only stem |
| 6 | 3:4 ratio (110:80 of halved B) | Ratio is ~11:8, not clean | MARGINAL |

**Best approach for 110+160:** First, check if the user prompt implies using only non-rhythmic stems (vocals, pads) from one track. If so, stretch that stem freely -- a vocal stretched 45% has artifacts but may be acceptable in a remix context. If both rhythmic elements are needed, **prefer transitioning through a breakdown section where tempo is perceptually ambiguous** -- this is the most natural-sounding option and what a skilled DJ would do. The 2:3 polyrhythmic approach (110 and 165, stretching B just 3.1%) is technically viable but should only be attempted at creativity level 3+ as it almost always sounds chaotic in practice.

**Additional user scenarios:**
- Bring 110 to 80 doubled: Target 80 BPM, Song A stretches -27.3% (too much for drums; only viable for non-rhythmic stems)
- Bring 110 to 90 and push 160 to 180: Song A stretches -18.2% (borderline), Song B stretches +12.5% (acceptable for drums). Combined quality: degraded but usable if the mix is dense enough to mask artifacts

#### The Octave Error Problem

BPM detection frequently reports half or double the true tempo. Trap at 140 BPM may detect as 70; DnB at 170 may detect as 85.

**Mitigation strategies:**
1. Use genre hints to disambiguate -- if classified as DnB, prefer the 160-185 range
2. For any detected BPM `x`, also evaluate `x/2` and `x*2` as candidates
3. Apply a log-Gaussian prior weighting moderate tempos (90-150 BPM) as more likely
4. Analyze the snare/kick pattern -- if snare hits on what would be beat 3 of a slow interpretation, the track likely "feels" at that slower tempo

### 1.2 Time-Stretching Budget

#### Quality Zones

| Stretch Amount | Quality | Description |
|----------------|---------|-------------|
| 0-5% | Transparent | Virtually indistinguishable from original |
| 5-10% | Excellent | Minor artifacts on close listening |
| 10-15% | Good | Subtle artifacts on exposed solo material; masked in a mix |
| 15-20% | Acceptable | Noticeable on drums and transient-heavy material |
| 20-30% | Degraded | Clearly audible artifacts: transient smearing, phasiness |
| 30-50% | Poor | Significant quality loss; only for buried stems or creative effect |
| 50%+ | Creative effect only | Extreme artifacts |

#### Per-Stem Stretch Tolerances

| Stem Type | Safe Budget | Hard Limit | Why |
|-----------|-------------|------------|-----|
| Drums / Percussion | +/- 8% | +/- 12% | Transient smearing is immediately audible; Demucs-separated drums already have artifacts that compound with stretching |
| Bass | +/- 10% | +/- 20% | Phase issues appear in low frequencies |
| Vocals | +/- 15% | +/- 35% | More tolerant; pitch artifacts more noticeable than timing |
| Other (pads, strings) | +/- 20% | +/- 40% | Sustained sounds hide artifacts well |
| Guitar (clean) | +/- 10% | +/- 20% | Pick transients are sensitive |
| Guitar (distorted) | +/- 15% | +/- 30% | Distortion masks many artifacts |
| Full Mix | +/- 8% | +/- 12% | Artifacts from all elements compound |

#### Artifact Types

1. **Transient smearing**: Attack of percussive sounds becomes soft and blurred. Primary quality limiter for rhythmic content.
2. **Phasiness / loss of presence**: Hollow, underwater quality from phase misalignment.
3. **Warbling / flanging**: Periodic pitch fluctuations on sustained notes.
4. **Metallic / robotic quality**: At extreme amounts, vocals take on a synthesized character.

#### Direction Matters

Speeding up generally produces fewer artifacts than slowing down. When speeding up, the algorithm removes portions of the signal (the remainder is real audio). When slowing down, it must synthesize new content. **Prefer speeding up the slower track over slowing down the faster track when stretch amounts are similar.**

#### Recommended Engine

Rubber Band R3: open source, Python bindings via `pyrubberband`, high quality for stretches under 20%, reasonable CPU.

| Stretch Range | Stem Type | Rubber Band R3 Settings |
|---------------|-----------|------------------------|
| 0-5% | Any | engine=finer, transient_mode=crisp, window_mode=standard |
| 5-15% | Drums | engine=finer, transient_mode=crisp, detector_mode=percussive |
| 5-15% | Non-drums | engine=finer, transient_mode=smooth, window_mode=long |
| 15%+ | Vocals | engine=finer, transient_mode=smooth, phase_mode=laminar, window_mode=long |
| 15%+ | Other | engine=finer, transient_mode=mixed, window_mode=long |

### 1.3 Tempo Decision Algorithm

**Priority order:**

```
1. Exact 1:2 ratio with zero/minimal stretch (< 3%)    -> ALWAYS prefer
2. Near 1:2 ratio with small stretch (< 8% per track)   -> Strong preference
3. Direct match with moderate stretch (< 8% per track)   -> Good
4. 2:3 polyrhythmic with small stretch (< 8%)            -> Experimental only (creativity 3+)
5. 3:4 polyrhythmic with small stretch (< 8%)            -> Experimental only (creativity 3+)
6. Large stretch on non-rhythmic stems only (up to 35%)  -> Last resort
7. Transition through a breakdown                        -> Fallback when nothing works
```

**Decision flow:**

1. **Resolve octave ambiguity** for both tracks using genre hints and drum pattern analysis
2. **Identify the rhythmic anchor** -- the track providing drums/bass gets tempo priority
3. **Generate candidate target tempos** from all combinations of (felt_a, felt_a/2, felt_a*2) x (felt_b, felt_b/2, felt_b*2)
4. **Score each candidate** based on: total stretch (less is better), stretch on rhythmic stems (penalize heavily), method naturalness (direct > halftime > polyrhythmic), direction preference (speed-up over slow-down)
5. **Select the highest-scoring candidate**

**Stem-aware anchor rule:** When the user prompt implies vocals from one track and drums from another, lock the drum track's tempo as the anchor (zero stretch), and stretch the vocal stem to match (vocals are more tolerant).

#### Quick-Reference Decision Table

| Song A Genre (BPM) | Song B Genre (BPM) | Best Strategy | Target BPM | Quality |
|---------------------|---------------------|---------------|------------|---------|
| Hip-Hop (85) | DnB (170) | Halftime DnB (exact 1:2) | 85 | Perfect |
| Hip-Hop (90) | DnB (174) | Halftime DnB | 87-90 | Excellent |
| Hip-Hop (90) | House (126) | Stretch vocals to 126 | 126 | Good |
| R&B (68) | House (128) | Doubletime R&B to ~128 | 128-136 | Good |
| Trap (140/70) | Dubstep (140/70) | Direct match | 140 | Perfect |
| House (128) | Techno (135) | Direct stretch | 130-132 | Excellent |
| House (124) | DnB (170) | 3:4 polyrhythmic | 124/170 | Experimental |
| Rock (120) | Hip-Hop (92) | Stretch hip-hop vocals | 120 | Good |
| Pop (115) | House (128) | Direct stretch | 120-124 | Good |
| Funk (100) | House (128) | Stretch funk up or 2:3 | 128 or 100/150 | Acceptable |


## 2. Arrangement Intelligence  <!-- Sprint Scope: Sections 2.2-2.3 (vocal placement, section volume offsets, energy curves) -->

### 2.1 Song Structure Detection  <!-- Future/Post-Sprint: ML-based structure detection is aspirational; MVP uses heuristic energy profiling -->

#### Section Types and Characteristics

| Section | Duration | Energy | Detection Features |
|---------|----------|--------|-------------------|
| **Intro** | 4-16 bars | Low (30%) | Low RMS, limited frequency content, track start, few layers |
| **Verse** | 8-16 bars | Moderate (50-60%) | Vocal presence, lower RMS than chorus, stable harmonics |
| **Pre-Chorus** | 2-4 bars | Rising (60-70%) | Increasing spectral density, harmonic shift, between verse and chorus |
| **Chorus** | 8-16 bars | High (85-100%) | Highest RMS, max spectral density, most layers, repetitive melodic pattern |
| **Bridge** | 4-8 bars | Variable (40-70%) | Novel harmonic content, occurs once (60-75% through song) |
| **Breakdown** | 4-16 bars | Very Low (20%) | Sharp RMS drop (>6 dB), drums/bass drop out |
| **Buildup** | 4-8 bars | Rising (20-80%) | Steadily increasing RMS, accelerating rhythmic patterns, risers |
| **Drop** | 8-16 bars | Maximum (100%) | Max RMS after build, sudden full-frequency onset, highest transient density |
| **Outro** | 4-16 bars | Declining (30%) | Decreasing RMS, track end, reducing layers |

#### Detection Approaches

1. **Self-Similarity Matrix**: Compute chroma/MFCC features, create frame-to-frame similarity matrix. Repeated sections appear as bright diagonal bands; boundaries appear between blocks.
2. **Deep learning models**: SpecTNT-style Transformer classifies into a 7-class taxonomy (intro, verse, chorus, bridge, outro, instrumental, silence) directly from spectrograms.
3. **Heuristic fallback**: RMS energy contours + onset density + spectral centroid changes. Label sections by energy profile rather than exact structural names -- the mixing system needs "high-energy" more than "chorus 2."

### 2.2 When Vocals Come In

#### Entry Cues

| Signal | Action | Detection Method |
|--------|--------|-----------------|
| Phrase boundary + low instrumental energy | Good vocal entry point | Beat tracking + RMS analysis |
| Drum fill or transitional device | Prime entry moment | Transient density spike in drum stem |
| Frequency space clear in 2-5 kHz | Safe to introduce vocals | Spectral analysis of instrumental |
| Energy transition upward | Natural vocal entry zone | RMS envelope derivative |
| Brief silence or filter sweep | Rhythmic reset -- ideal entry | Amplitude gap detection |

#### Exit Cues

| Signal | Action | Detection Method |
|--------|--------|-----------------|
| Silence gap in vocal stem > 2 seconds | Natural exit point | Amplitude threshold |
| Rising melodic content in "other" stem | Instrumental showcase -- pull vocals | Pitch detection on non-vocal stems |
| RMS energy dropping > 6 dB | Approaching breakdown -- consider exit | RMS envelope tracking |
| End of repeating phrase (3rd repetition) | Vocals naturally exit here | Repetition detection |
| Vocal stem reaching natural end | Manage exit gracefully | Tail detection |

#### Phrase Boundary Alignment Rules

**Beat 1 of a new phrase** is the strongest vocal entry point. The AI should default to this.

**Pickup/anacrusis detection**: Many vocals begin 1-2 beats before the downbeat. If detected, align the downbeat arrival, not the pickup start.

**Hip-hop vocal alignment exception**: Rap vocals frequently start before the downbeat as an integral part of the flow, not as a classical pickup note. Pre-bar elements ("yeah," "uh," rhythmic run-ups) are part of the delivery, not throwaway content. For hip-hop vocals: align to the start of the vocal phrase as performed (first stressed syllable to the instrumental's rhythmic anchor point), not to the nearest downbeat. Include all pre-bar content in the aligned output. Treat ad-libs as separate rhythmic elements with their own alignment -- they often land on off-beats intentionally.

**Exit padding**: Always leave at least 1 beat of silence (or reverb tail) between vocal exit and the next vocal entry.

#### Genre-Specific Vocal Placement

| Genre | When Vocals Enter | When Vocals Exit | Special Rules |
|-------|-------------------|-----------------|---------------|
| Hip-Hop | 8 or 16-bar boundary | End of verse (16 bars) or hook (8 bars) | Ad-libs fill gaps on beats 2/4 |
| EDM | Breakdown sections | Before drops | Only vocal chops during drops |
| R&B | After intro (4-8 bars) | End of phrase, sustain through barline | Preserve laid-back timing |
| Rock | Verse start | Before guitar solo | Guitar takes vocal frequency range in solos |
| DnB | Breakdowns | Before drum re-entry | Half-time vocals over full-time drums |
| Pop | Early (by bar 8-16) | Rarely absent for more than 4 bars | Always present |
| Lo-fi | As texture, not centerpiece | Loops and fades | Can be quieter in mix than other genres |

### 2.3 Instrumental Dynamics

#### Per-Section Level Guidelines

Relative to chorus = 0 dB reference:

| Section | Vocal Level | Instrumental Level | Net Vocal Prominence |
|---------|------------|-------------------|---------------------|
| Intro (no vocals) | Muted | Full | N/A |
| Verse | 0 dB (ref) | -3 to -6 dB from chorus | Vocals clearly above |
| Pre-chorus | +1 dB | -1 to -3 dB from chorus | Gap closing |
| Chorus | +2 to +3 dB | 0 dB (full) | Vocals at parity or slightly above |
| Bridge | Variable | -4 to -8 dB | Wide variation |
| Breakdown | Solo or muted | Minimal (-12 dB) | Extreme prominence or absent |
| Outro | Fading | Fading | Either can lead |

#### Per-Instrument Dynamics by Section

| Instrument | Verse | Chorus | Bridge | Breakdown/Drop |
|-----------|-------|--------|--------|----------------|
| Kick/Drums | -3 to -6 dB | 0 dB (reference) | Variable | Full power or absent |
| Bass | -2 to -4 dB | 0 dB (reference) | Often simplified | Full or absent |
| Vocals | Center, clear | Center, louder, doubled | May be absent or solo | Usually absent |
| Pads/Keys | Present, moderate | Fuller, wider stereo | May dominate | Often filtered or absent |
| Lead instruments | Subtle or absent | Supporting | May take lead | Variable |

#### Energy Curve Templates

**Template 1 -- Classic Remix Arc:**
```
Intro    Verse          Chorus         Verse          Chorus         Bridge    Chorus          Outro
[30%] -> [45->55%]  -> [80->90%]  -> [50->60%]  -> [85->95%]  -> [40%]  -> [90->100%]  -> [30%]
```

**Template 2 -- EDM Build-Drop:**
```
Intro    Build          Drop           Break    Build           Drop            Outro
[20%] -> [50->75%]  -> [100->90%]  -> [20%] -> [55->80%]  -> [100->95%]  -> [30%]
```

**Template 3 -- Hip-Hop Mashup:**
```
Intro    Verse(A)       Hook(B)        Verse(A)       Hook(B)        Bridge    Hook(B)         Outro
[40%] -> [55->65%]  -> [75->85%]  -> [58->68%]  -> [80->90%]  -> [50%]  -> [85->95%]  -> [35%]
```

**Intra-section dynamics:** Energy is NOT flat within a section. Sections breathe -- verses build over their bars, choruses peak in the middle and slightly dip before the next section, builds ramp progressively. The arrow notation above (`45->55%`) indicates the energy curve within that section. Treat each section as a micro-arc, not a flat block. This makes the remix breathe like real music instead of feeling like Lego blocks snapped together.

#### Transition Energy Rules

| From | To | Energy Change | Style |
|------|-----|---------------|-------|
| Intro | Verse | Gradual +20% | Additive layering over 4 bars |
| Verse | Chorus | Moderate +30-40% | Energy lift at phrase boundary |
| Chorus | Verse | Moderate -30% | Subtractive at section boundary |
| Chorus | Bridge | Sharp -40-50% | Can be sudden or 2-bar transition |
| Bridge | Chorus | Sharp +50-60% | Often explosive (one beat) |
| Buildup | Drop | Explosive +80% | Instant release at downbeat |
| Any | Breakdown | Sharp -60-80% | Often sudden silence then rebuild |

### 2.4 The Power of Space  <!-- Future/Post-Sprint: Reference for understanding arrangement philosophy; not directly implemented -->

#### Creating Contrast Through Absence

The perceived impact of a musical element is proportional to the contrast from what came before. If all elements play all the time, nothing stands out.

**Key techniques:**

| Technique | Duration | When | Effect |
|-----------|----------|------|--------|
| Pre-drop silence | 1-2 beats | Before a drop/chorus | Maximum impact on downbeat |
| Breakdown | 4-16 bars | After high-energy section | Listener reset, builds anticipation |
| Vocal isolation | 2-8 bars | Bridge or special moment | Intimacy, emotional impact |
| Rhythmic gaps | Eighth or quarter note | Within a groove | Creates rhythmic interest |
| Instrumental break | 4-8 bars | Mid-remix | Let the music breathe |
| Breath space | 0.5-1 beat | Between vocal phrases | Natural phrasing |

#### The Arrangement Density Curve

```
Intro:         [drums]
Verse 1:       [drums + bass]
Pre-chorus:    [drums + bass + harmony]
Chorus:        [drums + bass + harmony + vocal + extras]
Verse 2:       [drums + bass + vocal]  (denser than Verse 1)
Chorus 2:      [ALL ELEMENTS]
Bridge:        [reduced -- maybe just vocal + pads]
Final Chorus:  [ALL ELEMENTS + additional energy]
Outro:         [elements dropping out progressively]
```

**Implementation rules:**
- Budget at least 10-15% of the remix duration as low-energy or silent moments
- Scale silence to energy contrast: bigger upcoming spike = longer preceding space
- A remix with 3 stems (vocals from A, drums+bass from B) will almost always sound cleaner than one using all 8 stems from both songs
- The "mute first" philosophy: start with everything, remove what does not contribute. If removing a stem at a given moment is not noticed, mute it.


## 3. Harmonic Intelligence  <!-- Sprint Scope: Section 3.1 (key detection + pitch-shifting); Section 3.2 is Future/Post-Sprint -->

### 3.1 Key Compatibility (Camelot Wheel)

#### The Camelot System

The Camelot Wheel maps 24 musical keys to an alphanumeric code. Inner ring (A) = minor keys; outer ring (B) = major keys.

| Camelot | Key | Camelot | Key |
|:---:|:---:|:---:|:---:|
| 1A | Ab minor | 1B | B major |
| 2A | Eb minor | 2B | F# major |
| 3A | Bb minor | 3B | Db major |
| 4A | F minor | 4B | Ab major |
| 5A | C minor | 5B | Eb major |
| 6A | G minor | 6B | Bb major |
| 7A | D minor | 7B | F major |
| 8A | A minor | 8B | C major |
| 9A | E minor | 9B | G major |
| 10A | B minor | 10B | D major |
| 11A | F# minor | 11B | A major |
| 12A | Db minor | 12B | E major |

#### Compatibility Rules

| Move | Example | Description | Safety |
|------|---------|-------------|--------|
| Same key | 8A to 8A | Identical notes and key | Perfect |
| Relative major/minor | 8A to 8B | Same number, different letter; subtle mood shift | Safe |
| Adjacent +/-1 | 8A to 7A or 9A | One note changes between keys | Safe |
| +/-2 steps | 8A to 6A or 10A | Two notes change; more dramatic | Workable |
| Percussion-only pivot | Any to any | During drum/percussion sections with no harmonic content | Always safe |
| Anything else | 8A to 3A | Multiple clashing notes | Avoid or pitch-shift |

**4 safe destinations from any key:** Same key, relative major/minor, +1 step, -1 step.

#### Energy Direction Through Key Changes

- **Ascending numbers** (clockwise on wheel): Builds tension and energy
- **Descending numbers** (counterclockwise): Releases tension
- **Major to minor** (B to A): Darkens the mood
- **Minor to major** (A to B): Brightens the mood

#### Pitch-Shifting Rules for Key Correction

1. If same key or adjacent on Camelot Wheel: proceed without pitch shift
2. If 2-3 steps apart: pitch-shift the vocal stem (fewer artifacts than shifting instrumentals)
3. If >3 steps apart: shift to nearest compatible key (max +/-3 semitones preferred)
4. Never pitch-shift more than +/-4 semitones -- quality degrades too much in MVP-quality targets
5. Clashing is most audible in vocals + melody; bass/drums-only combinations are more forgiving

### 3.2 Spectral Awareness  <!-- Future/Post-Sprint: Bark/ERB band analysis, psychoacoustic masking. MVP uses basic spectral complementarity scoring only. -->

#### Frequency Slot Assignments

| Element | Primary Slot | Secondary Presence |
|---------|-------------|-------------------|
| Kick drum | 50-100 Hz (thump), 3-5 kHz (click) | -- |
| Sub-bass | 20-60 Hz | -- |
| Bass | 80-250 Hz | 700-1000 Hz (definition) |
| Snare | 200-400 Hz (body), 2-5 kHz (crack) | -- |
| Vocals | 1-4 kHz (presence), 6-10 kHz (air) | 200-500 Hz (body) |
| Guitar | 500 Hz - 3 kHz | -- |
| Hi-hats/cymbals | 8-16 kHz | -- |

#### Frequency Collision Detection

When two stems have significant energy in the same perceptual frequency bands, they collide. Detection approach:

1. Compute spectral energy using STFT, mapped to Bark or ERB perceptual bands
2. For each band, compute overlap coefficient: `min(A, B) / max(A, B)`
3. Flag bands where overlap exceeds 0.7 as collision zones
4. Apply corrective EQ: attenuate the less important stem in collision zones

#### Spectral Complementarity Score

```
complementarity = 1 - (spectral_overlap / total_spectral_energy)
```

Score near 1.0 = stems are spectrally complementary (ideal). Score near 0.0 = heavy overlap (needs aggressive EQ).

#### Psychoacoustic Masking

- **Upward spread of masking**: Low-frequency sounds mask higher frequencies more than vice versa. A bass at 100 Hz can mask a vocal harmonic at 400 Hz. Manage bass levels carefully when combining with mid-range stems.
- **Critical bands**: The auditory system divides hearing into ~24 bands (Bark scale). Within a single critical band, masking is strongest. Two instruments in different critical bands will not mask each other even if spectrally close.
- **Temporal masking**: A loud sound masks quieter sounds for up to 100-200ms after it (forward masking) and up to 20ms before it (backward masking). Choose blend points where the outgoing track has a moment of relative quiet.

#### The Equal Loudness Problem

Human hearing is not flat -- most sensitive at 3-4 kHz, less sensitive at low and high frequencies. A mix balanced at one volume sounds bass-light at lower volumes. Use LUFS-based loudness matching (K-weighted) rather than raw RMS to ensure the mix sounds balanced at any playback level.


## 4. Mixing Technique Library  <!-- Sprint Scope: Sections 4.2 (basic EQ carving) and 4.3 (sidechain, LUFS, gain staging) -->

### 4.1 Transition Types

| Technique | Duration | Best For | How It Works |
|-----------|----------|----------|-------------|
| **Long Blend** | 8-32 bars | House, techno, trance (long-form only) | Gradual EQ swap + volume fade, phrase-aligned |
| **Quick Cut** | Instantaneous | Hip-hop, EDM drops, genre switches | Hard switch on a downbeat |
| **Drop Swap** | 1 beat | EDM, dubstep, trap | Use buildup from Track A, drop from Track B |
| **Acapella Over Instrumental** | Full section | Mashups, creative remixes | Core musicMixer feature -- layer vocal over instrumental |
| **Echo/Reverb Tail** | 1-4 bars | Mood changes, atmospheric transitions | Apply reverb to outgoing track, bring in new track clean |
| **Filter Fade** | 2-8 bars | Busy arrangements | LPF outgoing + HPF incoming, meet in the middle |
| **Crossfade** | 2-16 bars | Universal default | Volume fade with equal-power curve |
| **Stutter/Chop** | 1-4 bars | EDM, trap, hip-hop | Rhythmic gate on outgoing audio, accelerating pattern |
| **Build and Drop** | 4-8 bars | Peak-time moments, genre shifts | Build tension, cut to silence, slam in new track |

#### Genre-Specific Transition Preferences

| Genre | Typical Duration | Preferred Techniques |
|-------|-----------------|---------------------|
| Pop | 2-4 bars | Crossfade, hard cut on downbeat |
| Hip-hop | 1-4 bars | Hard cut, filter sweep, stutter |
| EDM/House | 8-32 bars | EQ transition (bass swap), filter sweep |
| Rock | 1-2 bars | Hard cut, crossfade |
| Ambient/Chill | 8-16 bars | Long crossfade, reverb tail |
| Trap | 1-4 bars | Stutter, hard cut at drop |
| DnB | 4-16 bars | Filter fade, echo tail, drop swap |
| Afrobeats | 2-4 bars | Percussion bridge (drum pattern continuous), crossfade with filter |
| Latin (Reggaeton) | 1-4 bars | Hard cut on dembow pattern, filter sweep |
| UK Grime | 1-2 bars | Hard cut (often aggressive, can be mid-bar), stutter |
| Jersey Club | 1-2 bars | Hard cut, vocal chop variation |
| Amapiano | 2-8 bars | Gradual percussion layering, filter crossfade |

#### Transition Length Scaling Rule

**Maximum transition length is genre-aware.** For most genres, the cap is **15% of total remix duration**. For genres where extended transitions are a defining characteristic (EDM, House, Trance, Ambient, DnB), the cap is relaxed to **25% of total remix duration**. This allows the genre-defining long blends that listeners expect while preventing transitions from consuming the remix.

| Genre Category | Max Transition % | Rationale |
|---------------|-----------------|-----------|
| EDM, House, Trance, Ambient, DnB | 25% | Extended transitions are a genre-defining feature |
| All other genres (hip-hop, rock, pop, etc.) | 15% | Transitions should complement, not dominate |

| Remix Duration | Max Transition Bars (at 120 BPM) |
|----------------|----------------------------------|
| 1-2 minutes | 4-8 bars |
| 2-4 minutes | 8-16 bars |
| 4-6 minutes | 16-24 bars |
| 6+ minutes | Up to 32 bars |

### 4.2 EQ Mixing Rules

#### The Golden Rule: Never Play Two Bass Lines Simultaneously

Two bass lines cause frequency masking, phase cancellation, and muddy low end. This is the single most important EQ mixing rule.

#### Three-Band EQ Reference

| Band | Frequency Range | Content |
|------|----------------|---------|
| Low (Bass) | ~20-250 Hz | Kick drums, bass lines, sub-bass |
| Mid | ~250 Hz - 4 kHz | Vocals, melodies, snares, harmonics |
| High (Treble) | ~4-20 kHz | Hi-hats, cymbals, air, sibilance |

#### Core EQ Techniques

**Bass Swap**: The bread-and-butter technique. When transitioning, cut the outgoing track's low EQ while simultaneously bringing up the incoming track's low EQ. The swap should be quick and clean, timed to a downbeat or phrase boundary.

**Frequency Carving**: When two stems compete in the same range, reduce that band on the less important stem. Example: cut mids on the instrumental when vocals are present.

**High-Pass Filtering by Stem Type**:

| Stem | HPF Frequency | Purpose |
|------|--------------|---------|
| Vocals | 80-120 Hz | Remove room rumble, proximity effect, bass bleed |
| "Other" (guitar, synth, keys) | 150-250 Hz | Clear low end for dedicated bass stem |
| Hi-hats (if isolated) | 300 Hz | Remove kick/snare bleed |
| Bass | 30 Hz | Remove sub-rumble below musical content |

#### The Vocal Pocket

When vocals are present, cut instrumental energy in the 2-5 kHz range by 2-4 dB. This is where vocal consonants live and where intelligibility is won or lost. Frequency-selective ducking (dynamic EQ) is superior to broadband ducking because it preserves the instrumental's low-end power and high-end sparkle.

### 4.3 Volume and Dynamics Processing

#### Sidechain Compression Patterns

| Trigger | Target | Gain Reduction | Purpose |
|---------|--------|---------------|---------|
| Kick drum | Bass stem | 1-3 dB | Low-end clarity, rhythmic pumping |
| Vocal | Instrumental stems | 2-4 dB | Vocal intelligibility |
| Snare | Pad/sustained sounds | 1-2 dB | Rhythmic definition |
| Any transient element | Sustained elements | 1-3 dB | Creates "breathing" |

**Implementation approach**: Analyze the envelope of the trigger stem, create an inverse gain automation curve, apply it to the target stem. This achieves the same effect as real-time sidechain compression in offline processing.

#### Kick Pattern Classification (Pattern-Aware Sidechain)

Kick-triggered sidechain only works cleanly on regular kick patterns. Before applying sidechain, classify the kick pattern:

| Pattern Type | Examples | Sidechain Approach |
|-------------|---------|-------------------|
| **Regular (4/4)** | House, techno, trance | Standard sidechain as documented above. Kick triggers bass ducking on every hit. |
| **Syncopated but periodic** | Hip-hop, funk, reggaeton | Use a simplified trigger: only the strongest hits (typically beats 1 and 3) drive the sidechain. Ignore ghost kicks and syncopated hits. Depth: 1-3 dB. |
| **Breakbeat / irregular** | DnB, breakbeat, Afrobeats | Do NOT use kick-triggered sidechain. Instead use RMS envelope following: compute a rolling RMS envelope of the kick stem and apply gentle inverse gain (-1 to -2 dB) to the bass only in the sub-bass range (below 80 Hz) where collision is physical, not rhythmic. |
| **Polyrhythmic** | Afrobeats, Latin (salsa) | Use frequency-selective ducking in sub-bass only (< 80 Hz). The rhythmic interplay between percussion layers IS the genre -- sidechain pumping destroys it. Depth: 1-2 dB maximum. |

**Fallback rule:** When kick pattern classification is uncertain, default to RMS envelope following rather than kick-triggered ducking. This produces subtle, transparent results on any pattern type.

#### Ducking Settings Reference

| Parameter | Subtle | Moderate | Aggressive |
|-----------|--------|----------|-----------|
| Ratio | 2:1 | 3:1 to 4:1 | 6:1 to 8:1 |
| Attack | 5-15 ms | 1-5 ms | 0.5-2 ms |
| Release | 100-200 ms | 50-100 ms | 30-50 ms |
| Gain reduction | 1-3 dB | 3-5 dB | 5-8 dB |
| Use case | Clean mix, sparse | Standard pop/rock | Dense, competing frequencies |

Gain reduction beyond 6-8 dB creates obvious pumping artifacts. For most remix contexts, 2-4 dB is sufficient.

#### LUFS Targets

| Context | Target Level |
|---------|-------------|
| musicMixer output | -12 LUFS integrated |
| Per-channel peak | -6 to -3 dBFS |
| Master output | Peak below 0 dBFS |
| Final limiter ceiling | -1.0 dBFS (true peak limiter) |
| True peak | -1.0 dBTP |

**Why -12 LUFS instead of the -14 streaming standard:** Source songs uploaded to musicMixer are typically mastered to -10 to -8 LUFS. At -14, the remix output is 4-6 dB quieter than the sources, making remixes sound flat and underwhelming on direct comparison. The -12 target narrows this gap while leaving headroom for the true peak limiter. The backend pipeline already uses -12.

#### Gain Staging Workflow

1. Normalize each input stem to a per-stem LUFS target before mixing (see per-stem LUFS table below -- this leaves summation headroom)

**Per-Stem LUFS Targets (calibrated to -12 LUFS master)**

The math: for N equally loud stems summing to -12 LUFS integrated, each stem should be approximately `-12 - 10*log10(N)` dB LUFS. This ensures the sum lands at the master target without requiring makeup gain that would push peaks into the limiter.

| Active Stem Count | Per-Stem LUFS Target | Math | Notes |
|-------------------|---------------------|------|-------|
| 2 stems | -15 LUFS | -12 - 10*log10(2) = -15.0 | Most common case: vocals from A + instrumentals from B. Simple EQ rules, minimal sidechain needed. |
| 3-4 stems | -18 LUFS | -12 - 10*log10(4) = -18.0 | Standard multi-stem remix |
| 5-6 stems | -20 LUFS | -12 - 10*log10(6) = -19.8 | Dense arrangements; aggressive EQ carving required |

**2-stem tier details:** The 2-stem case (vocals + instrumentals) is the MVP default. With only 3 dB of headroom needed, the limiter acts as a transparent safety net rather than actively shaping dynamics. Sidechain behavior simplifies to basic vocal ducking. EQ carving focuses on the vocal pocket (2-5 kHz) only.
2. Apply section-level volume offsets per the arrangement plan
3. Apply dynamic ducking (sidechain automation) where stems compete
4. Monitor combined output LUFS in 3-second windows to catch sections that are too quiet or too loud
5. Apply final limiter at -1 dBFS as a safety net (transparent, not crushing)
6. Use equal-power crossfader curves during transitions to prevent loudness boost at the midpoint

#### Multiband Compression  <!-- Future/Post-Sprint -->

Apply multiband compression to the combined output, targeting bands where multiple stems have significant energy. This prevents low-frequency elements from triggering compression on high-frequency elements. Split into at least 3 bands matching the standard DJ EQ: lows (<250 Hz), mids (250 Hz - 4 kHz), highs (>4 kHz).


## 5. Genre Playbook  <!-- Future/Post-Sprint: Genre detection and genre-specific rule loading. Reference material for LLM prompt context; do not implement as code during sprint. -->

### 5.1 Per-Genre Profiles

#### Hip-Hop

| Property | Value |
|----------|-------|
| **BPM** | 80-115 (boom-bap 85-100, trap 130-170 half-time) |
| **Swing** | 54-62% (boom-bap); 50% straight (trap hi-hats) |
| **Quantization** | Vocals 60-80%; drums 80-95% |
| **Structure** | Intro (4-8) / Verse (16) / Hook (8) / Verse (16) / Hook (8) / Verse (16) / Hook (8) / Outro (4-8) |
| **Vocal treatment** | On top of the beat, dry-to-light reverb, high-pass at 100-150 Hz |
| **Mixing conventions** | Vocals never compete with kick/bass; ad-libs on off-beats; sidechain optional |
| **What makes it taste good** | Groove -- vocals riding the beat with a laid-back feel (boom-bap) or locked-in precision (trap). The hook must hit with impact. Drums are the foundation; everything else serves the vocal. |

#### EDM / House

| Property | Value |
|----------|-------|
| **BPM** | House 120-130, Techno 125-150, Trance 128-145, Dubstep 140 (half-time 70) |
| **Swing** | 50% (straight, grid-locked) |
| **Quantization** | Vocals 90-100%; instruments 95-100% |
| **Structure** | Intro (16-32) / Breakdown (16-32) / Buildup (8-16) / Drop (16-32) / Breakdown / Buildup / Drop / Outro (16-32) |
| **Vocal treatment** | Full phrases in breakdowns; chops/one-shots during drops; heavy reverb, delay, pitch effects |
| **Mixing conventions** | Sidechain kick-to-bass (the signature pumping); all sections multiples of 8 bars; changes every 8 bars within sections |
| **What makes it taste good** | The tension-release cycle. Deep valleys (breakdowns) make peaks (drops) feel enormous. The buildup is entirely about tension -- risers, snare rolls, filter sweeps. The drop must feel like an explosion of energy. |

#### R&B / Soul

| Property | Value |
|----------|-------|
| **BPM** | 60-85 (classic/neo-soul), 85-110 (modern R&B) |
| **Swing** | 58-67% (heavy, laid-back) |
| **Quantization** | Vocals 40-60%; instruments 60-80% |
| **Structure** | Intro (4-8) / Verse (8-16) / Pre-Chorus (4) / Chorus (8) / Verse / Pre-Chorus / Chorus / Bridge (8) / Chorus / Outro |
| **Vocal treatment** | Centerpiece of the mix. Multiple layers: lead, harmonies, doubles, ad-libs. Medium-heavy reverb. Preserve dynamic expression. Never hard-quantize. |
| **Mixing conventions** | Cut instrumental 1-5 kHz for vocal room; subtle sidechain ducking; pan harmonies L/R (30-60%); keep lead centered |
| **What makes it taste good** | Lushness and emotional expression. The vocal IS the song. The groove should feel effortless -- heavy swing, ghost notes, behind-the-beat feel. Harmonic sophistication (extended chords, chromatic movement) provides richness. |

#### Rock

| Property | Value |
|----------|-------|
| **BPM** | 90-145+ (varies widely by subgenre: grunge/stoner 90-110, classic/indie 110-130, punk 160+, ballads 60-80) |
| **Swing** | 50-52% (mostly straight, slight human variation); groove-based rock (Led Zeppelin, Black Keys) can reach 55% |
| **Quantization** | Vocals 50-70%; instruments 60-80% |
| **Structure** | Intro (4-8) / Verse (8-16) / Chorus (8-16) / Verse / Chorus / Bridge/Solo (8-16) / Chorus / Outro. Prog rock may use irregular section lengths and non-4/4 time. Punk compresses everything: 4-bar verses, 4-bar choruses, 2-minute songs. |
| **Vocal treatment** | Moderate compression, slight edge/distortion for aggressive styles. Guitar solos replace vocals for 8-16 bars. Punk: raw, present, minimal processing. |
| **Mixing conventions** | Preserve dynamic range (do NOT over-compress -- the loudness war already crushed most rock masters). Guitars panned L/R for doubles. Dynamic contrast is essential but varies by subgenre: quiet-loud-quiet-loud (alternative/grunge), relentless intensity (punk/metal), groove-driven dynamics (blues rock, southern rock). |
| **What makes it taste good** | Dynamic contrast and the interplay between guitar riffs and vocal melody -- they should not occupy the same frequency space simultaneously. Live-sounding drums with natural variation. The feel of a room and a band playing together. Groove-based rock (Led Zeppelin, Black Keys, Queens of the Stone Age) is defined by the rhythmic lock between drums and guitar riff. Punk is defined by raw energy and speed -- do not polish it. |

#### Drum & Bass

| Property | Value |
|----------|-------|
| **BPM** | 160-180 (174 most common) |
| **Swing** | 50-55% (breakbeats may be loose, bass grid-locked) |
| **Quantization** | Vocals 70-85%; instruments 85-95% |
| **Structure** | Intro (16-32) / Drop (32-64) / Breakdown (16-32) / Drop (32-64) / Outro (16-32) |
| **Vocal treatment** | Half-time vocals over full-time drums. Soulful samples in liquid DnB; chopped stabs in neurofunk. Vocals sit 1-5 kHz (gap between sub-bass 30-80 Hz and hi-hats 8+ kHz). |
| **Mixing conventions** | Drums full-time (170 BPM), bass half-time (~85 BPM). Sharp, punchy drum sounds. Sub-bass mono, deep (30-80 Hz). |
| **What makes it taste good** | The tension between fast drums and slow bass. Rolling breakbeats create physical energy. Breakdowns strip to atmosphere, then drums crash back at full intensity. The sub-bass is felt in the body, not just heard. |

#### Pop

| Property | Value |
|----------|-------|
| **BPM** | 100-130 (modern clusters around 100-120) |
| **Swing** | 50-52% (clean, quantized) |
| **Quantization** | Vocals 80-95%; instruments 90-100% |
| **Structure** | Intro (4-8) / Verse (8-16) / Pre-Chorus (4) / Chorus (8-16) / Verse / Pre-Chorus / Chorus / Bridge (8) / Chorus / Outro |
| **Vocal treatment** | Always present (rarely absent >4 bars). Lead front and center. Verse: single vocal. Chorus: doubles, harmonies for "big" sound. |
| **Mixing conventions** | Stick to 4/8/16 bar sections. Chorus by bar 32-40. Predictable structure is a feature. |
| **What makes it taste good** | Hooks and singability. The chorus must feel like a payoff. Pre-choruses build anticipation. Vocal is always the star. Clean, polished production. Predictability gives the listener comfort; small surprises within that framework create delight. |

#### Lo-Fi / Chill

| Property | Value |
|----------|-------|
| **BPM** | 70-90 (lo-fi hip-hop), 80-110 (chillhop), 60-80 (ambient chill) |
| **Swing** | 55-65% (deliberately loose, J Dilla-influenced) |
| **Quantization** | Vocals 30-50%; instruments 50-70% |
| **Structure** | Loose loops: Loop A (8-16) / Loop B (8-16) / Loop A / Loop C / Loop A / Fade. Or ambient: gradual layer build over 32-64 bars, sustain, gradual reduction, fade. |
| **Vocal treatment** | A texture layer, not a centerpiece. Low-pass filter, pitch wobble, tape saturation, chorus effect. Can be quieter in the mix than other genres. |
| **Mixing conventions** | Deliberately imperfect: vinyl crackle, tape hiss, muffled EQ (LP at 10-12 kHz). Flat energy curve -- no drops or peaks. Texture > melody > rhythm > dynamics. |
| **What makes it taste good** | Atmosphere and warmth. The imperfections ARE the aesthetic. Heavy swing gives it a human, relaxed feel. Do NOT clean up or polish. Gentle transitions (crossfade, filter sweep, gradual layers). Consistency of mood over variety of energy. |

#### Afrobeats

| Property | Value |
|----------|-------|
| **BPM** | 100-120 (Afrobeats), 110-115 (Amapiano) |
| **Swing** | 55-60% (subtle but essential; the groove is everything) |
| **Quantization** | Vocals 60-80%; instruments 80-90% |
| **Structure** | Intro (4-8) / Verse (8-16) / Chorus (8-16) / Verse / Chorus / Bridge (8) / Chorus / Outro. Amapiano: more loop-based, extended builds over 16-32 bars. |
| **Vocal treatment** | Melodic and rhythmic vocals layered with ad-libs and call-response. Medium reverb. Vocals often double as percussion through rhythmic delivery. |
| **Mixing conventions** | Log drum (Amapiano) or percussion-heavy rhythms drive the groove. Bass patterns are melodic and syncopated. Leave space for the vocal-percussion interplay. Sidechain optional but subtle. |
| **What makes it taste good** | The infectious, interlocking rhythmic patterns. Percussion layers create a polyrhythmic bed that moves the body. The bass is melodic and bouncy, not just low-end weight. Amapiano's log drum and piano stabs are genre-defining -- preserve them. Energy builds through layering, not volume. |

#### Latin (Reggaeton / Salsa / Bachata / Cumbia)

| Property | Value |
|----------|-------|
| **BPM** | Reggaeton 95-105 (dembow), Salsa 150-250, Bachata 125-145, Cumbia 80-100 |
| **Swing** | 50-52% (reggaeton is grid-locked); higher for salsa/cumbia (55-60%) |
| **Quantization** | Vocals 70-85%; instruments 85-95% |
| **Structure** | Reggaeton: Intro / Verse / Chorus / Verse / Chorus / Bridge / Chorus / Outro. Salsa: extended instrumental sections (montuno) with call-response. |
| **Vocal treatment** | Reggaeton: rhythmic, percussive delivery, moderate reverb. Salsa: powerful, dry-ish lead with backing coro (chorus). Bachata: intimate, close-mic feel. |
| **Mixing conventions** | The dembow rhythm (reggaeton) or clave pattern (salsa/cumbia) is the rhythmic backbone -- never displace it. Bass is prominent but clean. Percussion layers are dense and interlocking. |
| **What makes it taste good** | Rhythmic drive and danceability. The dembow is one of the most recognizable rhythms in modern music. Salsa's energy comes from the interplay between rhythm section and brass. Bachata lives in the guitar arpeggios and intimate vocal. Cumbia's shuffle groove is hypnotic. |

#### UK Grime

| Property | Value |
|----------|-------|
| **BPM** | 138-142 (tight range, very consistent) |
| **Swing** | 50% (straight, aggressive grid) |
| **Quantization** | Vocals 70-85% (MCs ride the beat with precision but individuality); instruments 90-100% |
| **Structure** | Intro (4-8) / Verse (16-32) / Chorus/Hook (8) / Verse / Hook / Verse / Hook / Outro. Long verses are common -- MCs need bars. |
| **Vocal treatment** | Dry, present, aggressive. Minimal reverb. Compression for consistency. The MC's delivery IS the energy -- do not bury it. |
| **Mixing conventions** | Sparse, dark instrumentals with heavy sub-bass. Square-wave bass and icy synths. Lots of space in the mid-range for the vocal. Minimal layering -- grime is about impact through simplicity. |
| **What makes it taste good** | Raw energy and lyrical intensity. The sparse production creates space for the MC to dominate. The sub-bass hits physically. The 140 BPM tempo creates relentless forward momentum. Do NOT over-produce or polish. |

#### Jersey Club

| Property | Value |
|----------|-------|
| **BPM** | 130-145 (typically 135-140) |
| **Swing** | 50% (straight, but with characteristic syncopated bed squeaks and vocal chops) |
| **Quantization** | Vocals 80-90% (vocal chops are rhythmic elements); instruments 90-100% |
| **Structure** | Loop-based with 8-16 bar sections. Often built around a chopped vocal sample that repeats with variations. Less traditional verse-chorus structure, more DJ-tool format. |
| **Vocal treatment** | Chopped, pitched, and used as percussion. Vocal samples are sliced to sixteenth notes and rearranged rhythmically. Original vocal pitch is secondary to rhythmic placement. |
| **Mixing conventions** | Kick pattern is the genre signature (syncopated, bouncy). Bed squeaks and vocal chops layer into a percussive bed. Bass is punchy and short, not sustained. High energy throughout -- minimal breakdowns. |
| **What makes it taste good** | The bounce. Jersey Club's signature kick pattern makes you move involuntarily. The chopped vocals create infectious, repetitive hooks. Energy is relentless but the syncopation keeps it from feeling monotonous. The bed squeak sound is genre-defining. |

### 5.2 Cross-Genre Combination Matrix

#### Compatibility Ratings

| Combination | Rating | Notes |
|------------|--------|-------|
| R&B vocals + Hip-hop beat | Very High | Same groove family, natural pairing |
| R&B vocals + Lo-fi instrumental | Very High | Lo-fi already samples R&B heavily |
| Hip-hop vocals + EDM instrumental | High | Common crossover; vocals ride over the beat naturally |
| Hip-hop vocals + DnB instrumental | High | Classic combo (MC culture); half-time bass supports rap |
| Pop vocals + Lo-fi instrumental | High | Strip pop production, add lo-fi processing |
| Pop vocals + Rock instrumental | High | Pop melodies work over rock instrumentation |
| DnB instrumental + R&B vocals | Medium-High | Liquid DnB + soulful R&B is proven |
| Rock vocals + EDM instrumental | Medium | Works with ballad vocals; aggressive rock can clash with synths |
| Rock vocals + Hip-hop beat | Medium | Melodic rock vocals + 808s can work; aggressive vocals are risky |
| EDM vocals + Rock instrumental | Low-Medium | Frequency conflicts between synth-tuned vocals and dense guitar |
| Afrobeats vocals + Hip-hop beat | Very High | Shared groove DNA; melodic Afrobeats vocals ride hip-hop beats naturally |
| Hip-hop vocals + Afrobeats instrumental | High | Rap delivery works over Afrobeats polyrhythmic percussion |
| Afrobeats vocals + EDM instrumental | High | Melodic vocals over clean electronic production; tempo-compatible |
| Latin vocals + Pop instrumental | Very High | One of the biggest crossover trends; natural pairing |
| Latin vocals + Hip-hop beat | Medium-High | Reggaeton delivery is rhythmically similar to rap |
| UK Grime vocals + DnB instrumental | High | Foundational UK music combination; same tempo range (140/170 halftime) |
| Jersey Club beat + Pop/R&B vocals | High | Chopped vocal samples over bouncy kicks is the genre's basis |
| Amapiano instrumental + R&B vocals | High | Log drum grooves complement smooth R&B delivery |
| Afrobeats vocals + Latin instrumental | Medium-High | Shared rhythmic complexity; tempo-compatible |

#### Cross-Genre Adaptation Rules

1. **The instrumental defines the groove.** The vocal adapts to the instrumental's tempo, swing, and feel -- not vice versa. Exception: when the vocal groove is the defining characteristic (e.g., a famous rap flow).

2. **Apply genre-appropriate processing based on the instrumental genre:**
   - Over EDM: add reverb, delay throws, sidechain compression
   - Over Hip-hop: dry-ish, compressed, present
   - Over Lo-fi: low-pass filter, saturation, tape wobble
   - Over DnB: medium reverb, keep vocals clear of sub-bass
   - Over Rock: moderate compression, slight distortion/edge

3. **Always carve frequency space.** Cut 2-4 dB at 1-5 kHz in the instrumental to make room for vocals, regardless of genre combination.


## 6. The 4/4 Rulebook  <!-- Future/Post-Sprint: Detailed phrase alignment and stem combination rules. MVP uses simplified LLM-driven arrangement. Section 6.2 swing reconciliation is post-sprint. -->

### 6.1 Phrase-Based Decision Rules

#### The Musical Hierarchy

```
1 beat = 1 quarter note
1 bar = 4 beats
1 phrase = 4 bars (16 beats) -- THE FUNDAMENTAL UNIT
1 section = 8 bars (2 phrases) or 16 bars (4 phrases)
1 song part = 16-32 bars (verse, chorus, etc.)
```

The 4-bar phrase is the fundamental building block. An 8-bar super-phrase (two 4-bar phrases forming question-and-answer) is nearly as universal.

#### The "Power of 2" Principle

All standard phrase lengths are powers of 2 multiplied by the base unit: 4, 8, 16, 32 bars. When the AI cuts or loops a stem, ALWAYS cut at a phrase boundary that is a multiple of 4 bars. Never cut mid-phrase.

#### Timing at Standard Tempos

At 120 BPM:
- Quarter note = 500 ms
- 1 bar = 2,000 ms
- 4 bars = 8,000 ms
- 8 bars = 16,000 ms

#### Where Vocals Enter and Exit

**Entry rules (strongest to weakest):**
1. Beat 1 of a new phrase (4 or 8-bar boundary) -- the safe default
2. Beat 1 with anacrusis (pickup on beat 4 of the preceding bar) -- very common in pop/R&B/hip-hop
3. Beat 3 -- secondary strong beat, works for mid-phrase entries
4. The "and" of beat 4 -- half-beat pickup, creates urgency
5. Beat 2 or the "and" of beat 2 -- syncopated, off-kilter (hip-hop, R&B)

**Exit rules:**
1. End of beat 4 (before the next downbeat) -- cleanest
2. Beat 3 with a ring-out (reverb/delay tailing through beat 4)
3. Sustain through the barline for emotional bridging

#### Phrase Alignment Hierarchy

| Rule | Description | Priority |
|------|-------------|----------|
| Rule of 4 | Vocals start/end on 4-bar boundaries | Minimum |
| Rule of 8 | Sections start/end on 8-bar boundaries | Standard |
| Rule of 16 | Major transitions on 16-bar boundaries | Strong |
| Rule of 32 | Full tension-release cycle spans 32 bars (EDM) | EDM-specific |

#### Common Section Lengths

| Section | Typical Length | Variants |
|---------|---------------|----------|
| Intro | 4-16 bars | 4 (short pop), 8 (standard), 16-32 (EDM/DJ-friendly) |
| Verse | 16 bars | 8 (short/modern), 12, 32 (extended) |
| Pre-chorus | 4-8 bars | 2-4 (hip-hop pre-hook) |
| Chorus/Hook | 8 bars | 4 (short hook), 16 (extended) |
| Bridge | 4-8 bars | Typically once per song |
| Breakdown | 16-32 bars | EDM-specific |
| Drop | 16-32 bars | EDM-specific |
| Outro | 4-16 bars | 4 (abrupt), 8-16 (fade/DJ-friendly) |

### 6.2 Stem Combination Rules

#### The Golden Rules

1. **Key match is non-negotiable.** All melodic/harmonic stems must be in compatible keys (Camelot same number, adjacent +/-1, or same number different letter). Anything else requires pitch-shifting.

2. **Tempo match is non-negotiable.** All stems must play at the same BPM. Time-stretch as needed within quality limits.

3. **One bass at a time.** Never combine two bass lines. Choose one.

4. **One drum groove at a time.** Never layer two full kits. Choose one, or selectively combine individual elements (kick from A, hi-hats from B).

5. **Vocals get frequency priority.** When vocals are present, all other stems defer in the 2-5 kHz range.

6. **Build from the bottom up.** Start with kick + bass, add drums, add "other," add vocals last. Each layer should have its own frequency slot.

7. **Less is more.** 3 well-chosen stems almost always sound better than 8 competing stems.

#### Per-Stem Combination Constraints

| Stem Type | Can Combine From Both Songs? | Constraints |
|-----------|------------------------------|-------------|
| Vocals | Rarely | One lead vocal at a time; harmonies from one source |
| Drums | Selectively | Individual elements (kick from A, hat from B) OK; full kits NO |
| Bass | Never | One bass source per section; filter-swap at transitions |
| Other | Sometimes | Can blend if spectrally complementary; high-pass at 150-250 Hz |

#### Stem Selection Priority

| Priority | Stem | Decision Criteria |
|----------|------|-------------------|
| 1 | Vocals | From the "vocal song" -- the featured element |
| 2 | Drums | From whichever song's groove best supports the vocal phrasing |
| 3 | Bass | From the song whose bass is in the correct key and complements the drum groove |
| 4 | Other | Most flexible -- blend from both, or choose based on energy/harmonic needs |

#### Quantization and Grid Alignment

| Genre | Primary Grid | Swing % | Vocal Quantization | Instrumental Quantization |
|-------|-------------|---------|-------------------|--------------------------|
| Hip-hop (boom-bap) | 16th note | 54-62% | 60-80% | 80-95% |
| Hip-hop (trap) | 32nd note (hi-hats) | 50% | 60-80% | 80-95% |
| EDM/House | 16th note | 50% | 90-100% | 95-100% |
| R&B/Soul | 16th note | 58-67% | 40-60% | 60-80% |
| Rock | 8th note | 50-52% | 50-70% | 60-80% |
| DnB | 16th note | 50-55% | 70-85% | 85-95% |
| Pop | 8th/16th note | 50-52% | 80-95% | 90-100% |
| Lo-fi | 16th note | 55-65% | 30-50% | 50-70% |

**Key principle:** Lower quantization strength preserves human feel. Never apply 100% quantization to vocals -- it sounds robotic. Extract swing profile from the instrumental and apply it to the vocal alignment so the vocal "rides" the groove correctly.

#### Swing Reconciliation

Cross-genre mashups almost always combine stems with different swing amounts. This is not an edge case -- it is the most common scenario when mixing hip-hop/R&B with anything electronic.

**When swung vocal meets straight beat (e.g., R&B vocal at 62% swing over EDM beat at 50%):**
- **Quantize-to-target (default for creativity 0-1):** Quantize the vocal to ~70% strength with zero swing, preserving natural timing variation without the swing pattern. This sounds clean but loses some of the vocal's original groove character.
- **Partial adaptation (default for creativity 2):** Apply 60-70% of the instrumental's swing to the vocal alignment. Full adaptation sounds unnatural; zero adaptation sounds off-grid.
- **Preserve-source (creativity 3+):** Leave the vocal's swing intact as a deliberate feel contrast over the straight beat.

**When straight vocal meets swung grid (e.g., pop vocal at 50% over boom-bap beat at 58%):**
- **Partial adaptation (default):** Apply 50-60% of the instrumental's swing to the vocal alignment. The vocal gains enough swing to sit in the groove without sounding forced.
- **Full adaptation:** Apply 90-100% of the instrumental's swing. Only use when the swing amount is subtle (< 55%) or when the vocal has loose enough timing to absorb it.

**When both have swing but different amounts (e.g., 58% vs 62%):**
- Split the difference, biased toward the instrumental's swing (70/30 weighting).

**Genre-specific exceptions:**
- **Lo-fi instrumental:** Can absorb any vocal swing -- looseness IS the aesthetic. No reconciliation needed.
- **Trap:** Hi-hats are straight (50%) but the overall feel is half-time with heavy swung vocals. Treat as a swung genre for vocal alignment despite the straight grid.

**Default rules per genre combination:**

| Vocal Source | Instrumental Source | Strategy |
|-------------|-------------------|----------|
| R&B/Soul (58-67%) | EDM/House (50%) | Quantize-to-target at creativity 0-1; partial adaptation at 2+ |
| Hip-hop (54-62%) | EDM/House (50%) | Partial adaptation (hip-hop groove is more rhythmically locked than R&B) |
| Pop (50-52%) | Hip-hop (54-62%) | Apply 50-60% of hip-hop swing to vocal |
| Any | Lo-fi (55-65%) | Preserve source swing; lo-fi absorbs everything |
| R&B (58-67%) | Afrobeats (55-60%) | Split the difference; both are groove-heavy genres |
| Any (50%) | DnB (50-55%) | Minimal adjustment needed; DnB breakbeats have inherent looseness |


## 7. Creative Intelligence  <!-- Future/Post-Sprint: Emotional arc, moments engineering, creativity dial integration. Reference for understanding creative goals; not implemented during sprint. -->

### 7.1 Emotional Arc

#### The Valence-Arousal Model

Music emotion maps to two dimensions:
- **Valence**: Positive (happy, euphoric) to negative (sad, angry)
- **Arousal**: High energy (excited, aggressive) to low energy (calm, melancholic)

This creates four quadrants:
- High arousal + positive valence = **Joy/Euphoria** (dance drops, triumphant choruses)
- High arousal + negative valence = **Anger/Tension** (aggressive breakdowns, dissonant builds)
- Low arousal + positive valence = **Serenity/Nostalgia** (ambient pads, gentle melodies)
- Low arousal + negative valence = **Sadness/Melancholy** (slow minor-key passages)

#### Audio Features That Predict Emotion

| Feature | Predicts | Direction |
|---------|----------|-----------|
| RMS energy / loudness | Arousal | Higher = more energetic |
| Spectral centroid | Arousal | Brighter = more energetic |
| Onset rate | Arousal | More events/second = higher energy |
| Tempo | Arousal | Faster = higher arousal |
| Mode (major/minor) | Valence | Major = positive, minor = negative |
| Harmonic complexity | Valence | Simple = positive, complex/dissonant = darker |

#### Narrative Arc Patterns for Remixes

| Pattern | Description | Best For |
|---------|-------------|----------|
| **The Build** | Start low, progressively increase, climax, resolve | Classic DJ set arc |
| **The Contrast** | Alternate high/low energy for maximum impact | Dramatic mashups |
| **The Transformation** | Start as Song A, gradually morph to Song B through stem swaps | Seamless transitions |
| **The Collision** | Intentionally jarring combinations that create meaning through juxtaposition | Experimental mashups |

**Implementation:** The AI should plan the emotional arc before making individual stem decisions. Given a prompt, analyze emotional profiles of both sources, identify sections matching the intended arc, and plan which stems to use when.

#### Tension and Release Toolkit

| Technique | Creates Tension | Creates Release |
|-----------|----------------|-----------------|
| Spectral density | Add more frequency content over time | Reduce to fewer elements ("drop" after build) |
| Rhythmic density | Increase events per beat | Simplify to basic pulse |
| Pitch content | Rising filter sweeps, ascending melodies | Return to root, descending resolution |
| Volume | Increasing loudness | Sudden reduction or gradual fade |
| Harmonic | Introduce dissonance between stems | Resolve to consonant harmony |
| Absence | Strip elements away, create space | Return the familiar (drums crash back) |

### 7.2 Moments Engineering

The difference between a technically correct mix and one that gives listeners goosebumps is **moments** -- specific points where something unexpected and delightful happens. Great mixes have points that make people reach for their phone to Shazam their own remix.

#### Identifying Coincidental Alignments

After stems are selected and tempo/key matched, scan for happy accidents between elements from different songs:

| Alignment Type | Detection Method | Value |
|----------------|-----------------|-------|
| Melodic convergence | Pitch contour of vocal and instrumental briefly match | High -- creates harmonic resonance that sounds intentional |
| Rhythmic sync | Accent patterns from different stems coincidentally align | High -- creates a groove pocket that feels natural |
| Lyrical emphasis on chord change | Vocal stress lands on a harmonic shift in the instrumental | Very High -- the most powerful "oh shit" moment |
| Call and response | Vocal phrase and instrumental riff alternate naturally | High -- creates conversational feel between songs |
| Energy peak alignment | Both songs' loudest moments overlap | Medium -- powerful but can also be overwhelming |

#### Engineering Surprise and Delight

Beyond identifying natural alignments, the system should create moments of genuine surprise:

1. **The unexpected drop-out:** In a dense, high-energy section, suddenly mute everything except one exposed element (a vocal phrase, a guitar riff) for 1-2 beats. The listener holds their breath. When everything crashes back, the impact doubles.
2. **The recontextualization moment:** An element from Song A that felt one way in its original context takes on new meaning over Song B's instrumental. An aggressive rap verse at half speed becomes contemplative. A cheerful pop melody over dark minimal production becomes eerie.
3. **The rhythmic reveal:** Gradually introduce elements that create a new rhythmic pattern only possible through the combination of both songs -- a pattern neither song contains alone.
4. **The harmonic accident:** When a vocal melody briefly harmonizes with an instrumental chord progression in a way that creates a new, unexpected harmony. Score these moments and build the arrangement to feature them.

#### Moment Placement

Moments should not be evenly distributed. Place the strongest moment at 60-75% through the remix (the "golden moment" position). Smaller moments can occur earlier to build trust with the listener. Never place a major moment in the first 15% or last 10% of the remix.

### 7.3 The Creativity Dial

#### Five Levels

| Level | Name | Behavior |
|-------|------|----------|
| 0 | Safe | Follow all genre rules strictly |
| 1 | Subtle | Allow +/-1 bar offset from phrase boundaries; +/-2 semitone key shift for interest |
| 2 | Moderate | Allow odd bar lengths (7, 9, 13); cross-genre processing; vocals during drops |
| 3 | Bold | Allow deliberate key clashes for tension; extreme tempo manipulation (>20%); structural subversion |
| 4 | Experimental | Polymetric alignment, extreme pitch shift, genre collision, intentional dissonance |

#### Mapping User Prompt Language to Creativity Level

| User Language | Creativity Level |
|--------------|-----------------|
| "clean mix" / "smooth" / "radio-ready" | 0-1 |
| "interesting" / "unique" / "creative" | 2 |
| "wild" / "experimental" / "crazy" | 3-4 |
| No creativity indicator (default) | 1 |

#### When to Break Rules Intentionally

**Odd bar lengths:** A 9-bar loop instead of 8 creates a subtle sense of displacement. Reducing a drop to 3 bars creates an abrupt, punchy effect. Progressively shortening sections (first verse 16 bars, second verse 8) maintains momentum.

**Mismatched phrase alignment:** Placing a vocal 2 beats early creates urgency. A vocal arriving 1 bar late creates a dramatic entrance. Offsetting vocal and instrumental phrases by 1-2 bars creates polymetric tension.

**Intentional key clashes:** Two stems a tritone apart (6 semitones) create maximum dissonance -- use briefly for dramatic tension. Detuning vocals by 10-30 cents creates an eerie, unsettling effect.

### 7.4 Mashup-Specific Techniques

#### What Makes a Mashup Work

Based on analysis of successful mashup artists (Girl Talk, DJ Earworm, Madeon):

1. **Harmonic compatibility** -- Songs in compatible Camelot keys sound consonant when layered
2. **Tempo compatibility** -- BPM within ~5%, or in a 2:1 ratio
3. **Spectral complementarity** -- Elements from different songs in different frequency ranges
4. **Genre contrast with structural similarity** -- Surprise of unexpected genre combinations, coherence from shared structure (4/4 time, similar phrase lengths)
5. **Semantic/cultural contrast** -- Meaning through juxtaposition (aggressive rap over gentle acoustic guitar)

#### Key Lessons from the Masters

**Girl Talk's approach:** Rapid layering of short clips (sometimes 5 seconds), constant change, confident arrangement decisions. 373 samples across 71 minutes in "All Day." Success comes from inspired selection and rapid juxtaposition, not sophisticated processing. The AI equivalent: excellent compatibility scoring + aggressive, confident arrangement.

**DJ Earworm's approach:** Gradual layering, building texture and momentum. Focus on creating a coherent narrative structure from disparate songs. Stem isolation quality matters enormously -- the cleaner the separation, the more freedom in recombination.

#### Temporal Alignment of Phrases

Beyond beat-matching, align musical phrases. A 4-bar vocal phrase from Song A should start at a 4-bar boundary of Song B's instrumental. Align structural boundaries: place chorus vocal over chorus instrumental (both high-energy), verse vocal over verse instrumental (both lower-energy).

#### The "Happy Accidents" Pattern

At creativity level 2+, the AI should occasionally try unlikely combinations:
- Place vocals over a section they "shouldn't" fit (verse vocals over a drop)
- Shift vocal by 2 or 4 beats from the "correct" alignment
- Apply processing from a different genre (rock distortion on R&B vocals)

If the result scores well on harmonic compatibility and energy match, keep it. Otherwise revert.


## 8. Implementation Roadmap  <!-- Sprint Scope: Sections 8.1 (analysis pipeline), 8.2 (decision engine + Mix Plan schema), 8.2.1 (three-layer architecture), 8.7 (summing + final normalization). Other subsections are reference/future. -->

### 8.1 Analysis Pipeline

**What to extract from each uploaded song (processing order):**

| Step | Property | Purpose | Method |
|------|----------|---------|--------|
| 0 | Sample rate / bit depth normalization | Common processing format | Resample all inputs to 44.1 kHz, convert to 32-bit float for internal processing (see Section 8.4) |
| 1 | Stems | Independent control of elements | Source separation (6-stem model: vocals, drums, bass, guitar, piano, other). Specific model configured at implementation level (e.g., BS-RoFormer, htdemucs_6s). |
| 1.5 | Stem quality assessment | Modulate mixing aggressiveness | Cross-bleed detection, spectral hole analysis, phase artifact scoring (see Section 8.5) |
| 2 | BPM | Tempo matching | Onset detection + tempogram analysis |
| 3 | Key | Harmonic compatibility | Spectral analysis, Camelot code |
| 4 | Beat grid | Phase alignment | Onset detection + beat tracking |
| 5 | Downbeats | Bar-level alignment | Spectral difference analysis |
| 6 | Phrase boundaries | Transition timing | Energy profiling + structural segmentation |
| 7 | Sections | Song structure | ML-based classification or heuristic |
| 8 | Energy curve | Dynamic planning | RMS (LUFS-weighted) per section and per stem |
| 9 | Spectral profile | Frequency collision avoidance | STFT mapped to perceptual bands |
| 10 | LUFS loudness | Gain matching | ITU-R BS.1770 metering |

**Per-stem additional analysis:** onset density per beat, spectral centroid over time, swing percentage, vocal presence/absence timeline.

#### 8.1.1 Cross-Song Correlation (runs after per-song analysis, before LLM decision)

After analyzing each song independently, this step computes relationships between the two songs. Its output is passed to the LLM as structured context alongside the per-song analysis results.

| Computation | Method | Output | Used By |
|-------------|--------|--------|---------|
| **BPM relationship** | Compare resolved BPMs; classify as direct match, halftime/doubletime, polyrhythmic, or incompatible (Section 1.1 tiers) | Compatibility tier, candidate target BPMs with stretch amounts | Tempo matcher (Layer 1), LLM arrangement decisions (Layer 2) |
| **Key compatibility** | Compute Camelot distance between detected keys; determine if pitch-shift is needed and by how many semitones (Section 3.1) | Camelot distance, required pitch-shift semitones | Key matcher (Layer 1), LLM (Layer 2) |
| **Energy profile alignment** | Compare per-section energy curves; identify which sections from Song A pair well with which sections from Song B based on energy level similarity or complementarity | Section compatibility matrix (e.g., "A-verse pairs with B-chorus for energy contrast") | LLM arrangement planning (Layer 2) |
| **Section boundary mapping** | Align beat grids between songs after tempo matching; map bar numbers across songs so the LLM can reference "bar 16 of Song A = bar 32 of Song B" | Cross-song bar alignment table | LLM section planning (Layer 2), crossfade rendering (Layer 3) |
| **Spectral complementarity score** | Compute spectral overlap between primary stems (e.g., vocal from A vs. instrumental from B) using the formula in Section 3.2 | Score 0.0-1.0 per stem pair | LLM stem selection (Layer 2), EQ carving aggressiveness (Layer 3) |

**Pipeline ordering:** Steps 0-10 (per-song) -> Step 11 (cross-song correlation) -> LLM decision (produces Mix Plan) -> audio processing (Layer 3).

### 8.2 Decision Engine

#### How the LLM Interprets the User Prompt

1. **Parse the prompt** to identify: which elements from which songs, style modifiers ("heavy," "chill," "fast"), structural preferences ("drop the beat," "build up")
2. **Map stems to songs** -- "Hendrix guitar" = other stem from Song A; "MF Doom rapping" = vocal stem from Song B
3. **Determine supporting stems** -- need drums (which song's groove supports the vocal?), need bass (which is in the right key?)
4. **Set creativity level** from prompt language

#### How Analysis Feeds Into Mix Decisions

```
Analysis Results
    |
    +--> Tempo Matcher: Determine target BPM, stretch amounts, halftime/doubletime
    |
    +--> Key Matcher: Camelot compatibility check, pitch-shift requirements
    |
    +--> Section Mapper: Align structural boundaries between songs
    |
    +--> Energy Curve Designer: Plan the remix's emotional arc
    |
    +--> Stem Selector: Which stems from which songs at which moments
    |
    +--> Mix Parameter Generator: EQ, ducking, levels, panning per section
    |
    v
Mix Plan (the complete blueprint before any audio processing begins)
```

#### Mix Plan Schema

The Mix Plan is the concrete data structure that flows from the LLM decision layer to the audio engine. Everything upstream (analysis, prompt parsing, rule evaluation) produces this; everything downstream (time-stretching, EQ, mixing) consumes it.

```json
{
  "target_bpm": 128,                    // MVP: determined by tempo matcher
  "target_key": "8A",                   // MVP: Camelot code from key matcher
  "pitch_shift_semitones": {            // MVP: per-song pitch adjustment
    "song_a": 0,
    "song_b": -2
  },
  "vocal_source": "song_a",            // MVP: which song provides vocals
  "instrumental_source": "song_b",     // MVP: which song provides instrumentals
  "sections": [                         // MVP: ordered list of remix sections
    {
      "label": "intro",                 // Section name (for logging/debug)
      "start_bar": 0,                   // Bar offset in the remix
      "duration_bars": 8,               // Length of this section
      "stems": {                        // MVP: per-stem gain (0.0 = muted, 1.0 = full)
        "vocals_a": 0.0,
        "drums_b": 0.8,
        "bass_b": 0.7,
        "other_b": 0.5
      },
      "transition_in": "fade",         // MVP: how this section begins (fade/crossfade/cut)
      "transition_beats": 8,           // MVP: transition duration in beats
      "energy_target": 0.3             // 0.0-1.0, maps to the energy curve templates
    }
  ],
  // --- Aspirational fields (not MVP) ---
  "per_stem_effects": {                 // Future: per-stem effect chains
    "vocals_a": {
      "reverb_send": 0.3,
      "delay_send": 0.1,
      "eq_preset": "vocal_presence"
    }
  },
  "sidechain_pairs": [                  // Future: explicit sidechain routing
    {"trigger": "drums_b", "target": "bass_b", "depth_db": -3}
  ],
  "moments": [                          // Future: engineered surprise points
    {"bar": 48, "type": "dropout", "duration_beats": 2}
  ]
}
```

**MVP fields** (marked above): `target_bpm`, `target_key`, `pitch_shift_semitones`, `vocal_source`, `instrumental_source`, `sections` (with `stems`, `transition_in`, `transition_beats`). These map directly to the Day 3 `RemixPlan` tool schema.

**Aspirational fields**: `per_stem_effects`, `sidechain_pairs`, `moments`. These enable the full mixing intelligence described in this document but are not needed for the sprint.

#### The Prompt-to-Mix-Plan Flow

Given user prompt "Hendrix guitar with MF Doom rapping over it":
1. Map stems: "Hendrix guitar" = other stem from Hendrix track; "MF Doom rapping" = vocal stem from Doom track
2. Determine supporting stems: Hendrix drums + bass (match the guitar); consider Doom's drum pattern for hip-hop-over-rock fusion
3. Check tempo compatibility: hip-hop ~90 BPM vs rock ~120 BPM (significant gap -- evaluate halftime, stretch options)
4. Check key compatibility via Camelot
5. Design energy curve: hip-hop verse-hook structure, let guitar shine in transitions/hooks
6. Set mixing parameters: duck guitar 2-5 kHz when vocals present, apply genre-appropriate processing

### 8.2.1 Rule Execution Architecture (Three-Layer Pipeline)

The mixing intelligence operates as a three-layer pipeline. Each rule in this document belongs to exactly one layer. This prevents ambiguity about what runs as code vs. what runs as LLM reasoning.

**Layer 1: Deterministic Pre-Processing (code, runs before LLM)**

These rules are physics and music theory -- they have deterministic correct answers and should never be delegated to the LLM.

| Rule Category | Section | What It Computes |
|--------------|---------|-----------------|
| Sample rate normalization | 8.4 | Resample to 44.1 kHz, convert to 32-bit float |
| Stem separation | 8.1 step 1 | Source separation into individual stems |
| Stem quality assessment | 8.5 | Quality tiers per stem |
| BPM detection + octave disambiguation | 1.1 | Resolved BPM for each song |
| Key detection | 3.1 | Camelot code for each song |
| Beat grid + downbeat detection | 8.1 steps 4-5 | Beat positions and bar boundaries |
| Section/phrase boundary detection | 8.1 steps 6-7 | Structural segmentation |
| Energy curve extraction | 8.1 step 8 | Per-section energy profile |
| Spectral profile | 8.1 step 9 | Per-stem frequency distribution |
| Cross-song correlation | 8.1.1 (below) | BPM relationship, key compatibility, energy alignment |
| Tempo compatibility scoring | 1.3 | Candidate target BPMs with quality scores |
| Key compatibility scoring | 3.1 | Camelot distance, pitch-shift requirements |

**Layer 2: LLM Decision Layer (receives analysis + condensed rules, outputs Mix Plan)**

The LLM handles creative and subjective decisions where there is no single correct answer. It receives the pre-processing results as structured data and a condensed version of the relevant rules as system prompt context.

| Decision | Relevant Rules (injected as LLM context) | Input From Layer 1 |
|----------|------------------------------------------|-------------------|
| Which stems to feature | Genre profiles (5.1), stem combination rules (6.2) | Stem quality scores, spectral profiles |
| Arrangement structure | Energy curve templates (2.3), section guidelines (2.2) | Section boundaries, energy curves from both songs |
| Vocal placement | Vocal entry/exit cues (2.2), genre-specific placement (2.2) | Vocal presence timeline, phrase boundaries |
| Transition types | Genre-specific transition preferences (4.1) | Section boundaries, energy profiles |
| Energy arc | Narrative arc patterns (7.1), energy curve templates (2.3) | Energy curves from both songs |
| Creativity level | Creativity dial (7.3) | User prompt text |

**Layer 3: Deterministic Post-Processing (code, applies Mix Plan to audio)**

These operations execute the Mix Plan produced by the LLM. They are deterministic audio transformations.

| Operation | Section | Input |
|-----------|---------|-------|
| Time-stretching | 1.2 | Target BPM from Mix Plan |
| Pitch-shifting | 3.1 | Semitone shifts from Mix Plan |
| Per-stem LUFS normalization | 4.3 | Stem count from Mix Plan |
| EQ carving (vocal pocket, HPF) | 4.2 | Stem assignments from Mix Plan |
| Sidechain/ducking automation | 4.3 | Stem assignments + kick pattern classification |
| Section volume offsets | 2.3 | Section list from Mix Plan |
| Crossfade/transition rendering | 4.1 | Transition types from Mix Plan |
| Phase coherence correction | 8.6 | Stem assignments from Mix Plan |
| Summing + headroom management | 8.7 | All processed stems |
| Final LUFS normalization + limiting | 8.7 | Combined output |

**Key insight:** The LLM should never make decisions about audio physics (LUFS levels, EQ frequencies, phase correction). It should only make decisions about musical arrangement and creative direction. This keeps the LLM prompt focused and the audio quality deterministic.

### 8.3 Progressive Enhancement

#### Day 3 (Intelligence) -- Implement First

- LLM-powered stem selection from user prompt
- Spectral complementarity scoring between selected stems
- Complementary EQ carving (clear the vocal pocket)
- Sidechain automation (vocal ducks instrumental)
- Basic emotional arc: intro (sparse) -> build -> peak -> breakdown -> final peak -> outro
- Section-level volume offsets per the arrangement plan
- Arrangement density planning (which stems active at each structural point)

#### Day 4 (Polish) -- Add Next

- Genre detection and genre-specific rule loading
- Swing/groove detection and matching
- Pre-drop silence insertion
- Filter sweep transitions at section boundaries
- Dynamic ducking refinement (frequency-selective)
- Crossfade curve optimization
- Final LUFS normalization and limiting

#### Future -- Advanced Features

- Psychoacoustic frequency band analysis (Bark/ERB scale)
- Haas effect and spatial processing (stereo width management)
- Dynamic panning for movement
- Adaptive EQ (per-beat spectral balance correction)
- Highlight detection for optimal section selection
- Emotional arc planning using valence-arousal model
- Real-time visual feedback via Meyda.js
- Polyrhythmic transition support (2:3, 3:4 relationships)
- M/S processing on master output
- Stereo correlation monitoring

### 8.4 Sample Rate and Bit Depth Management  <!-- Sprint Scope: Already implemented in Day 1 pipeline -->

Audio files arrive in varying formats (MP3 at 44.1 kHz, WAV at 48 kHz, etc.). Without normalization, every downstream operation (STFT, beat detection, Rubber Band) may produce incorrect results or fail silently.

**Pipeline rules:**

1. **Resample all inputs to 44.1 kHz** before any processing (including stem separation). Use a high-quality resampler (e.g., libsamplerate SRC_SINC_BEST_QUALITY or SoXR).
2. **Convert to 32-bit float** for all internal processing. This provides ~1500 dB of headroom, eliminating clipping during intermediate stages.
3. **Apply TPDF dithering only at final output** when rendering to 16-bit WAV or lossy formats (MP3, AAC). Dithering prevents quantization distortion that is especially audible in quiet passages and fade-outs.
4. **Never dither between intermediate processing stages** -- 32-bit float has sufficient precision.

### 8.5 Stem Quality Assessment  <!-- Future/Post-Sprint: No-reference quality metrics. MVP skips quality tiers and applies uniform processing. -->

Demucs does not produce clean stems. There will always be cross-bleed between channels, spectral holes from over-aggressive separation, and phase artifacts from the separation process. The system must assess stem quality and adapt its mixing strategy accordingly.

**Quality assessment stage (runs immediately after source separation):**

| Check | Method | Impact |
|-------|--------|--------|
| Cross-bleed detection | Correlate each stem with the other stems; high correlation = bleed | Reduce EQ aggressiveness on stems with high bleed (surgical EQ amplifies artifacts) |
| Spectral completeness | Compare spectral energy of sum-of-stems to original mix; gaps = spectral holes | Avoid boosting frequencies near spectral holes |
| Phase coherence | Compute phase difference between sum-of-stems and original; large differences = phase damage | Reduce sidechain compression and dynamic processing on phase-damaged stems |
| Cross-bleed energy ratio | For each stem, compute `energy_in_expected_band / total_energy`. A vocal stem should have most energy in 200 Hz-8 kHz; if significant energy exists below 80 Hz, that is bleed. | Primary quality score -- no clean reference needed |
| Spectral flux smoothness | Compute spectral flux over time; separation artifacts produce abnormally high flux (rapid spectral changes). Compare flux of separated stem to typical values for that stem type. | Detects phase artifacts and spectral holes that other metrics miss |

**Note:** SDR (Signal-to-Distortion Ratio) requires a clean reference signal that is unavailable in production (users upload mixed songs, not multitracks). The metrics above are all no-reference metrics that can be computed from the separated stems and the original mix alone.

**Adaptive processing rules:**

- **High quality stems (cross-bleed ratio > 0.85, low spectral flux anomaly):** Apply full mixing intelligence -- spectral complementarity scoring, dynamic EQ, sidechain automation, frequency carving.
- **Medium quality stems (cross-bleed ratio 0.6-0.85, moderate spectral flux):** Use gentler processing -- basic volume balance, high-pass filtering, broad EQ moves only. Reduce sidechain ducking depth by 50%.
- **Low quality stems (cross-bleed ratio < 0.6, high spectral flux anomaly):** Favor simple processing -- volume balance and HPF only. Prefer using fewer stems at higher quality over more stems at lower quality. When stem quality is low, sophisticated processing (multiband sidechain, dynamic EQ) amplifies artifacts rather than improving the mix.

### 8.6 Phase Coherence Management  <!-- Future/Post-Sprint: MVP uses a single polarity check + offset optimization over first 8 bars. Full per-section correction is post-sprint. -->

When combining stems from two different songs recorded in different studios with different equipment, phase relationships are essentially random. This is not just a spectral collision problem -- it is a time-domain issue that EQ alone cannot solve.

**Critical concern: Bass/kick phase interaction.** When bass from Song A and kick from Song B land in the same frequency range (40-120 Hz), they will have random constructive/destructive interference that changes with every beat.

**Phase coherence pipeline:**

1. **Polarity check:** Analyze the phase relationship between kick and bass stems in the 40-120 Hz range. If predominantly destructive (negative correlation), invert the polarity of one stem.
2. **Time-offset correction:** If polarity inversion does not resolve phase issues, apply a small time offset (0-5 ms) to find the best phase alignment. Test offsets at 0.5 ms increments, select the offset that maximizes the sum's RMS energy (not peak) in the 40-120 Hz band.
3. **Per-section correction (not per-beat):** Evaluate phase alignment once per arrangement section (intro, verse, chorus, etc.). Within a section, apply a single polarity/offset correction that maximizes average energy in the 40-120 Hz band across the entire section. At section boundaries, the arrangement transition envelope (crossfade, filter sweep) naturally masks any discontinuity from offset changes. If within-section correction is needed (e.g., bass line changes register mid-section), crossfade between old and new offset values over a 50-100ms window to avoid click artifacts. **Do not apply per-beat corrections** -- changing a time offset abruptly at a bar boundary creates waveform discontinuities (clicks/pops).

**When to skip:** Phase coherence management is only needed when combining harmonic/bass stems from different songs. It does not apply to vocals-over-instrumental combinations where there is no low-frequency overlap.

### 8.7 Summing Headroom Strategy

When two or more audio signals are summed digitally, the result can exceed 0 dBFS. The system needs explicit headroom management at the summing stage.

**Summing rules:**

**Important:** The primary headroom strategy is per-stem LUFS normalization (Section 4.3, Gain Staging Workflow step 1). When per-stem LUFS normalization is active (the default), the LUFS targets already account for summation headroom, and the pre-sum gain reduction below is **skipped**. The pre-sum gain reduction values below are ONLY applied as a fallback when per-stem LUFS normalization is not used (e.g., when processing raw stems without loudness metering).

1. **Pre-sum gain reduction (fallback only -- skipped when per-stem LUFS normalization is active):** Reduce each stem by a safety margin before combining:
   - 2 stems: -3 dB each
   - 3-4 stems: -4.5 dB each
   - 5+ stems: -6 dB each
2. **Soft-clipper at the sum point:** Apply a tanh soft-clipper at the summing stage to catch peaks that exceed -1 dBFS. This prevents hard clipping while preserving transient character. Add a hard-clamp safety net after the tanh (residual at extreme input is negligible but the clamp is a one-liner).
3. **Final loudness normalization:** After summing and soft-clipping, normalize the combined output to the target LUFS (-12 integrated). When per-stem LUFS normalization was used, this step should require minimal adjustment (< 1 dB). If makeup gain exceeds 2 dB, the per-stem targets may need recalibration.
4. **True peak limiting:** Apply a true peak limiter (not sample peak) at -1.0 dBTP as the final stage. This catches inter-sample peaks that a sample-peak limiter would miss.

**Cross-reference:** See Section 4.3 Gain Staging Workflow for the primary gain management pipeline. This section (8.7) provides the summing-stage safety net.


## Appendix

### A. Quick Reference Tables

#### A.1 Tempo Compatibility Matrix

| Relationship | Ratio | Stretch Tolerance | Quality | AI Priority |
|-------------|-------|-------------------|---------|-------------|
| Direct match | 1:1 | < 8% | Excellent | 1 (highest) |
| Halftime/Doubletime | 1:2 | < 8% | Excellent | 2 |
| Near halftime | ~1:2 | < 8% each | Good | 3 |
| 2:3 Polyrhythmic | 2:3 | < 8% | Acceptable | 4 |
| 3:4 Polyrhythmic | 3:4 | < 8% | Acceptable | 5 |
| Stem-only stretch | N/A | < 35% (vocals) | Degraded | 6 |
| Breakdown transition | N/A | N/A | Variable | 7 (fallback) |

#### A.2 Key Compatibility (Camelot) Table

| Move | Camelot Distance | Mood Effect | Safety |
|------|-----------------|-------------|--------|
| Same key | 0 | No change | Perfect |
| Relative major/minor | Same number, A/B switch | Subtle mood shift | Safe |
| +1 step | 1 | Smooth energy shift | Safe |
| -1 step | 1 | Smooth energy shift | Safe |
| +2 steps | 2 | Noticeable shift | Workable |
| +7 steps (+1 semitone) | 7 | Energy boost | Bold |
| 3+ steps | 3+ | Dissonant | Avoid or pitch-shift |

#### A.3 Genre Default Parameters

| Parameter | Hip-Hop | EDM | R&B | Rock | DnB | Pop | Lo-fi | Afrobeats | Latin | Grime | Jersey Club |
|-----------|---------|-----|-----|------|-----|-----|-------|-----------|-------|-------|-------------|
| BPM range | 80-115 | 120-130 | 60-85 | 90-145+ | 160-180 | 100-120 | 70-90 | 100-120 | 95-105 | 138-142 | 130-145 |
| Swing % | 54-62 | 50 | 58-67 | 50-55 | 50-55 | 50-52 | 55-65 | 55-60 | 50-52 | 50 | 50 |
| Vocal quant. | 60-80% | 90-100% | 40-60% | 50-70% | 70-85% | 80-95% | 30-50% | 60-80% | 70-85% | 70-85% | 80-90% |
| Vocal reverb | Dry-Light | Heavy | Med-Heavy | Medium | Medium | Medium | Light+Lo-fi | Medium | Medium | Dry | Dry-Light |
| Vocal HP filter | 100 Hz | 150 Hz | 80 Hz | 120 Hz | 100 Hz | 120 Hz | 200 Hz | 100 Hz | 100 Hz | 120 Hz | 150 Hz |
| Sidechain | Optional | Yes | Yes (subtle) | No | Optional | Optional | No | Subtle | Optional | Optional | Optional |
| Section length | 8/16 bar | 16/32 bar | 8/16 bar | 8/16 bar | 16/32 bar | 8/16 bar | 8/16 bar | 8/16 bar | 8/16 bar | 16/32 bar | 8/16 bar |
| Phrase grid | 4 bar | 8 bar | 4 bar | 4 bar | 8 bar | 4 bar | 4 bar | 4 bar | 4 bar | 4 bar | 4 bar |
| Transition style | Hard cut | Long blend | Crossfade | Hard cut | Filter fade | Crossfade | Long crossfade | Crossfade | Hard cut | Hard cut | Hard cut |

#### A.4 Stem Combination Rules Summary

| Rule | Description | Priority |
|------|-------------|----------|
| One bass at a time | Never combine two bass lines | Critical |
| One drum groove at a time | Never layer two full kits | Critical |
| Key match all melodic stems | Camelot same/adjacent or pitch-shift | Critical |
| Tempo match all stems | Time-stretch within quality limits | Critical |
| Vocals get 2-5 kHz priority | Duck instrumentals when vocals present | High |
| HPF non-bass stems | Remove low-end bleed from vocals (100 Hz), other (200 Hz) | High |
| Build from the bottom up | Kick+bass first, then drums, other, vocals last | Medium |
| Less is more | 3 stems cleaner than 8 | Medium |

#### A.5 Transition Type Selection Guide

| Situation | Recommended Transition | Duration |
|-----------|----------------------|----------|
| Same genre, similar energy | Long blend with EQ swap | 8-32 bars |
| Same genre, energy drop | Filter fade (LPF outgoing) | 4-8 bars |
| Genre switch | Hard cut on downbeat | Instant |
| Big energy increase (pre-drop) | Build + silence + slam | 4-8 bars + 1-2 beats silence |
| Big energy decrease (post-drop) | Subtractive layering or hard cut | 1-4 bars |
| Tempo change | Breakdown crossover or hard cut | Variable |
| Key change (compatible) | Long blend | 8-16 bars |
| Key change (incompatible) | Percussion-only pivot or hard cut | 1-4 bars |
| Mood change (light to dark) | Echo/reverb tail | 2-4 bars |

#### A.6 Default Numeric Parameters

```
# Ducking
DUCK_AMOUNT_DB = -3             # Instrumental duck under vocals
DUCK_FREQ_LOW = 2000            # Hz - lower bound of ducking band
DUCK_FREQ_HIGH = 5000           # Hz - upper bound of ducking band
DUCK_ATTACK_MS = 5              # How fast ducking engages
DUCK_RELEASE_MS = 100           # How fast ducking releases

# Section volume offsets (relative to chorus = 0 dB)
VERSE_OFFSET_DB = -4
INTRO_OFFSET_DB = -8
BRIDGE_OFFSET_DB = -6
BREAKDOWN_OFFSET_DB = -12
OUTRO_OFFSET_DB = -8

# High-pass filters
VOCAL_HPF_HZ = 100
BASS_HPF_HZ = 30
OTHER_HPF_HZ = 200

# Crossfade defaults
DEFAULT_CROSSFADE_BARS = 4
MIN_CROSSFADE_MS = 50
MAX_CROSSFADE_BARS = 16

# Silence
PRE_DROP_SILENCE_BEATS = 1
MIN_SECTION_DURATION_BARS = 4

# Tempo/key limits
MAX_TEMPO_STRETCH_PERCENT = 8   # Per stem (rhythmic)
MAX_VOCAL_STRETCH_PERCENT = 35  # Vocals tolerate more
MAX_PITCH_SHIFT_SEMITONES = 4

# Loudness
TARGET_LUFS = -12                     # Louder than streaming standard (-14) to match source loudness; see Section 4.3
MAX_TRUE_PEAK_DB = -1
LIMITER_CEILING_DBFS = -1.0    # Must use true peak limiter (not sample peak) to catch inter-sample peaks

# Per-stem LUFS normalization (leaves headroom for summation)
# Formula: per_stem_lufs = TARGET_LUFS - 10*log10(stem_count)
PER_STEM_LUFS_2_STEMS = -15    # Target LUFS per stem when combining 2 stems (most common MVP case)
PER_STEM_LUFS_3_4_STEMS = -18  # Target LUFS per stem when combining 3-4 stems
PER_STEM_LUFS_5_PLUS = -20     # Target LUFS per stem when combining 5-6 stems
```
