# musicMixer — Product Requirements Document

**Date:** 2026-02-23
**Brainstorm:** docs/brainstorms/2026-02-23-prompt-based-remix-brainstorm.md

---

## What We're Building

A web app that lets anyone create a music remix by prompting AI. Upload two songs, type what you want ("Hendrix guitar with MF Doom rapping over it"), and the AI splits the songs into stems, combines them, auto-matches tempo and key, and plays it back. anyone can be a DJ!

Every remix is ephemeral — replayable for up to 3 hours or until you create a new one, then gone forever. (for legal reasons)

### Core Flow

1. Upload two songs (local MP3/WAV files)
2. Type a prompt describing the mashup
3. AI separates stems, selects the right ones, matches tempo/key
4. Transparent progress updates as it works
5. Remix plays back
6. Remix expires after 3 hours or when a new one is created

### MVP Scope

| In | Out (Deferred) |
|----|----------------|
| Two songs, local files | Multi-song mixing, streaming integration |
| Single prompt, one-shot generation | Iterative refinement ("make drums louder") |
| Ephemeral playback (3hr TTL) | Export, sharing, saving |
| Web app | Mobile app |
| AI stem selection from prompt | Voice cloning |
| Progress indicators | Alternative suggestions on failure |

### Target User

People who love music and often curate playlists. Anyone who's wondered "what would X sound like over Y" but has no DJ skills.

---

## What New Technology We're Implementing

**AI-driven audio stem separation and recombination via natural language.** This integrates multiple cutting-edge systems into a single pipeline:

- **ML-based source separation** (e.g., Demucs, BS-RoFormer) — splitting a mixed song into individual stems (vocals, drums, bass, other) is a recently-solved problem that has only become production-viable in the last ~2 years
- **LLM prompt interpretation for audio engineering** — translating "Hendrix guitar with Doom vocals" into a concrete stem selection and mixing plan. No existing product uses an LLM as the creative director of an audio pipeline
- **Automated tempo/key matching** — algorithmically aligning stems from different songs so they sound coherent together (BPM detection, key detection, time-stretching, pitch-shifting)
- **Web Audio API for real-time multi-stem playback** — rendering the combined stems in-browser with synchronized, low-latency playback via Tone.js

The platform is a **web-first application**. 

---

## How This Pushes Beyond What I Think Is Possible


**The stretches:**
i know nothing about audio files, how they work, WAV's, etc. nothing. this project will require me to learn more about how audio is mixed, how we can solve tempo issues, and how we can use **real** music to make something novel. no ai generated slop. 

a few others: 
- stem granularity is gonna be difficult. 
- could be some legal issues, but even just something that my friends and i can use is sufficient here. 
- models don't really have taste yet - will they be able to recognize when something sounds right vs wrong? how can we guide them there? will it require finetuning/training? 

what excites me: 
i've dreamed about having a tool like this for a few years, but it always felt out of reach. creativity is just putting the puzzle pieces together. the idea that someone could have an idea to mix two songs together, take vocals from one, the beat from another, and get it to work sounds incredible. anyone can be a dj. 


---

## Open Questions (To Resolve During Planning)

1. **Stem granularity** — What separation models give us, what users expect, and how to bridge the gap
2. **Processing architecture** — On-device vs. server-side stem separation (affects legal model, cost, wait times)
3. **Processing time** — Target wait time and UX strategies to manage it
4. **First-time experience** — Onboarding for users who've never used an audio tool
5. **Error handling** — How to gracefully handle incompatible songs, failed separations, tempo/key limits
6. **File constraints** — Supported formats, size limits, song length limits
