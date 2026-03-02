# Record Player Playback UI

## Overview

Replace the plain HTML `<audio>` element in `RemixPlayer` with an animated top-down (bird's eye) vinyl turntable. When the remix completes, a vinyl record slides onto the platter. The user clicks play, the tonearm swings over, needle drops, and the record spins while audio plays.

## Design Spec

### Visual — Top-Down Turntable
- Bird's eye perspective looking straight down at a turntable
- **Plinth:** Warm wood grain background with rounded corners
- **Platter:** Dark circular platter inset into the plinth
- **Vinyl record:** Matte black with visible grooves catching light, sits on the platter
- **Record label:** Center circle reads "Fractal Records" with remix title below
- **Tonearm:** Brushed metal appearance, pivots from top-right, rests in a cradle when idle
- **Depth cues:** Soft drop shadows under tonearm, slight highlight on record edge

### Color Palette
- Warm wood tones for the plinth (walnut/oak feel)
- Brushed metal / silver for tonearm
- Matte black vinyl with subtle groove shimmer
- Amber/cream accents for the label
- Must integrate with existing dark theme (`bg-gray-950`)

### Animation Sequence

**Phase 1 — Record Placement** (~1.5s, on view load)
- Vinyl record slides/fades into frame onto the platter
- Gentle deceleration easing
- Tonearm parked in cradle

**Phase 2 — Idle** (waiting for play)
- Record sits still on platter
- Subtle groove shimmer animation
- Floating play triangle centered above the record
- Tonearm resting in cradle

**Phase 3 — Play Triggered** (~1.8s sequence)
1. Play button fades out
2. Tonearm lifts (subtle scale/shadow change) ~0.3s
3. Tonearm swings over record edge ~0.5s
4. Needle drops onto outer groove (micro-animation) ~0.3s
5. Record starts spinning — slow ramp to constant RPM ~0.7s
6. Audio playback starts synced to needle drop

**Phase 4 — Playing** (ongoing)
- Record spins at constant RPM
- Needle tracks along grooves from outer edge toward center (progress)
- Hovering over record reveals floating controls: pause button, rewind button
- Moving mouse away fades controls out
- Elapsed / remaining time displayed below turntable

**Phase 5 — Pause**
- Tonearm lifts and swings back to cradle
- Record decelerates to stop (~0.8s)
- Play button reappears on record
- Audio pauses

**Phase 6 — Resume**
- Same as Phase 3 (tonearm swings back, needle drops at current position, spin resumes)

### Responsive
- Turntable shrinks proportionally on mobile, maintaining aspect ratio
- Controls remain tap-friendly at small sizes

## Technical Approach

### Current State
- `RemixPlayer.tsx` renders `<audio controls autoPlay>` with browser-native controls
- No custom playback UI exists
- Existing `VinylRecord.tsx` SVG can be referenced for groove rendering patterns
- Tailwind-only styling, no animation library
- CSS keyframes already defined for `vinyl-spin`

### New Components

| Component | File | Purpose |
|-----------|------|---------|
| `TurntableScene` | `src/components/turntable/TurntableScene.tsx` | Static SVG: plinth, platter, record, tonearm, label |
| `RecordLabel` | `src/components/turntable/RecordLabel.tsx` | SVG group: "Fractal Records" text + remix title |
| `Tonearm` | `src/components/turntable/Tonearm.tsx` | SVG group: tonearm with pivot point for rotation |
| `FloatingControls` | `src/components/turntable/FloatingControls.tsx` | Play/pause/rewind overlay positioned above record |
| `RecordPlayerView` | `src/components/RecordPlayerView.tsx` | Orchestrator: assembles scene, manages animations, wires audio |
| `useAudioPlayer` | `src/hooks/useAudioPlayer.ts` | Hook: wraps HTML audio element, exposes play/pause/seek/progress |

### State Machine (useAudioPlayer + RecordPlayerView)

```
placing → idle → playing ↔ paused
                    ↓
                 seeking (rewind)
```

### Animation Approach
- CSS transitions + keyframes for record spin, tonearm swing, opacity fades
- `transform-origin` on tonearm SVG group for pivot rotation
- `transition-timing-function` for natural easing (deceleration on placement, acceleration on spin-up)
- Progress-based needle position: `needle_angle = start_angle + (progress * angle_range)`
- No external animation library needed — CSS handles everything

### Integration Points
- `RemixPlayer.tsx` replaces `<audio controls>` with `<RecordPlayerView>`
- Hidden `<audio>` element still handles actual playback
- `useAudioPlayer` hook manages audio state + exposes controls to RecordPlayerView
- Existing props (`sessionId`, `explanation`, `warnings`, etc.) remain unchanged
- Explanation text and warnings render below the turntable

## Implementation Tasks

### Bucket 1 — Foundation (parallel)

- [x] **Task A: Static Turntable SVG Components** (PR #4)
  Create `TurntableScene`, `RecordLabel`, and `Tonearm` as pure SVG components.
  - Wood grain plinth (SVG rect with warm gradient + noise pattern)
  - Dark platter circle
  - Vinyl record with grooves (adapt patterns from existing `VinylRecord.tsx`)
  - "Fractal Records" label with remix title prop
  - Tonearm with proper pivot point (`transform-origin`) in parked position
  - All elements positioned for bird's eye view
  - Responsive: viewBox-based sizing that scales with container

- [x] **Task B: useAudioPlayer Hook** (PR #3)
  Create a hook that wraps the HTML `<audio>` element.
  - Accepts `audioUrl: string`
  - Exposes: `play()`, `pause()`, `seek(time)`, `currentTime`, `duration`, `progress` (0-1), `isPlaying`, `isLoaded`
  - Manages audio element lifecycle (create/destroy)
  - Returns a ref to attach to a hidden `<audio>` element
  - Handles edge cases: loading, buffering, ended, errors

### Bucket 2 — Animation & Assembly (depends on Bucket 1)

- [x] **Task C: RecordPlayerView + FloatingControls + Integration** (PR #5)
  Build the orchestrator component that animates the turntable and wires audio.
  - Animation state machine: `placing → idle → playing ↔ paused`
  - Record placement animation (slide-in + fade on mount)
  - Tonearm swing animation (CSS transform rotation around pivot)
  - Record spin animation (CSS keyframes, controlled via animation-play-state)
  - Needle progress tracking (rotation angle = progress along grooves)
  - FloatingControls: play triangle (centered, always visible when idle/paused), hover overlay with pause + rewind (visible on hover during playback)
  - Time display: elapsed / remaining below turntable
  - Replace `<audio controls>` in RemixPlayer with RecordPlayerView
  - Keep explanation text, warnings, expiration notice, new remix button below
  - Responsive sizing via container queries or percentage-based layout
  - CSS keyframe additions to `index.css` for tonearm and placement animations
