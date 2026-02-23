# Brainstorm: Prompt-Based Music Remix

**Date:** 2026-02-23
**Status:** Draft

## What We're Building

A web app where anyone can create a music mashup by describing it in plain English. No DJ skills, no timeline, no knobs. You upload two songs, type what you want ("Hendrix guitar with MF Doom rapping over it"), and the AI figures out the stems, combines them, and plays it back.

Every remix is ephemeral — replayable for up to 3 hours or until you create a new one, then gone. Like a live DJ set that only exists in the moment.

### Core Flow (MVP)

1. User uploads two songs from their device (local MP3/WAV files)
2. User types a single prompt describing the mashup they want
3. AI splits both songs into stems (vocals, drums, bass, other)
4. AI selects the right stems based on the prompt
5. AI auto-matches tempo and key
6. Transparent progress as it works ("Extracting vocals from Song A... Matching tempo... Rendering mix...")
7. Remix plays back
8. User can replay, or start prompting a new remix (previous remix is replaced)
9. Remix expires after 3 hours or when a new remix is created, whichever comes first

### What It Is NOT (MVP)

- Not a manual mixer with toggles, timelines, or faders
- Not iterative — no "make the drums louder" follow-ups (future feature)
- Not an export tool — no downloads, no sharing, no saving
- Not a voice cloning tool (future feature: licensed artist voices)
- Not a mobile app (future: mobile shell wrapping the web app)

## Why This Approach

**Prompt-based over manual mixing** because:
- The target user is a casual music fan, not a DJ or producer
- Describing a mashup in words is something everyone can do; using a mixer UI is not
- No competitor does this — DJ apps, stem splitters, and DAWs all require manual control
- It collapses hours of audio engineering into a single sentence

**One-shot over iterative** because:
- Reinforces the ephemeral identity — there's magic in the surprise
- Iteration breaks the "live concert" feel and turns it into an editing session
- Simpler to build for MVP
- Iterative refinement can be added later as a power-user feature

**Ephemeral with 3-hour TTL** because:
- Pure one-listen feels punishing (what if you're distracted?)
- A time window preserves the "not permanent" identity without frustrating users
- Legally, what matters is no permanent copy, no export, no distribution
- 3-hour window lets users relisten and enjoy; auto-expiry enforces ephemerality

**Web first** because:
- Web Audio API / Tone.js gives better multi-track audio control than React Native's expo-av
- No app store review process (important for a legally novel product)
- Instant access via URL — lowest friction for casual users
- Can wrap in a mobile shell later once the experience is proven

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Interaction model | Prompt-based (single text field) | Accessibility > control for casual users |
| Generation model | One-shot for MVP | Reinforces ephemeral identity; iteration later |
| Ephemeral model | 3-hour TTL or until next remix | Balances ephemerality with usability |
| Song count | Two songs for MVP | Core mashup experience; architecture supports more later |
| Song source | Local files only | Legally clean; user provides their own audio |
| Platform | Web first, mobile later | Better audio APIs, no app store gatekeeping |
| Export | None for MVP | All remixes ephemeral; owned-content export is future |
| Voice cloning | Not in MVP | Future feature with licensed artist voices only |
| AI transparency | Show progress steps | Builds trust, makes wait feel shorter; keep it lean |
| Output length | AI decides | Picks best sections; user can specify in prompt |
| Prompt guidance | Example placeholders + AI handles vague | Low friction onboarding without constraining input |
| Error handling | Report failure, suggest reprompting | No "suggest alternatives" — user reprompts from scratch |

## Resolved Questions

1. **Full songs or sections?** AI decides — it picks the best sections to combine based on the prompt. Output length varies. Users can optionally specify sections in their prompt ("just use the chorus from Song A").

2. **Session lifecycle** — A remix persists until the user creates a new one OR 3 hours have passed, whichever comes first. One active remix at a time for MVP.

3. **What happens with vague prompts?** Both: show inspiring example prompts as placeholder text in the input field, AND let the AI handle vague prompts gracefully.

4. **What if the blend doesn't work?** The AI reports failure and the user can reprompt. No automated alternative suggestions for MVP.

## Open Questions (Investigate During Planning)

1. **Stem granularity** — Current separation models output 4 stems (vocals, drums, bass, other). "Guitar" is buried in "other" alongside keys, synths, strings, etc. Users will prompt for specific instruments the system can't isolate. Need to investigate: what separation models are available, what granularity is realistic, and how the prompt system should handle requests that exceed stem resolution.

2. **File format/size limits** — What audio formats are supported? Is there a max file size or song length?

3. **Error states** — What if stem separation fails? What if the two songs are so incompatible (wildly different tempo/key) that no reasonable remix exists?

4. **First-time user experience** — What does a new user see before uploading anything? Is there a demo? Onboarding? The target user has never used an audio tool.

5. **Processing time expectations** — Stem separation can be slow. What's the target wait time and how does that affect the UX?

6. **Ephemerality is a social contract** — Users can screen-record the output. The app prevents export but cannot prevent capture. This is the same legal line as Spotify/Netflix (we don't provide the tool to copy). Worth acknowledging explicitly.

## Deferred Features (Post-MVP)

- **Iterative refinement** — follow-up prompts to tweak a remix
- **Concurrent remixes** — build a "remix jukebox" playlist within a session
- **Multi-song mixing** — combine stems from 3+ songs
- **Voice cloning** — licensed artist voice packs
- **Export for owned content** — download remixes made from original/CC/public domain audio
- **Social feed** — TikTok-style feed of remixes (owned content only)
- **Search-based song input** — find songs by name instead of uploading files
- **Streaming integration** — pull from Spotify/Apple Music
- **Mobile app** — native iOS/Android via Expo
- **Collaborative remixing** — real-time multi-user sessions
- **Remix chains** — remix a remix
- **Suggest alternatives on failure** — AI proposes alternative stem combinations when a blend doesn't work

## Target User

Casual music fans. Anyone who's ever thought "what would X sound like over Y" but has no idea how to make it happen. Not DJs, not producers, not audio engineers. People who consume mashups on TikTok/YouTube and wish they could create their own.

## Competitive Landscape

| Competitor | What They Do | Gap |
|-----------|-------------|-----|
| Suno, Udio | AI music generation from text | Generates new music, not remixes of real songs. No use of your actual music. |
| djay, VirtualDJ | Full DJ software | Built for DJs, steep learning curve |
| LALAL.AI, Moises | Stem separation only | Stops at splitting; no mixing, no creativity |
| Rave.dj | Auto-mashup | Minimal user control; can't describe what you want |
| Ableton, BandLab | Full DAWs | Way too complex for casual users |
| **musicMixer** | **Prompt-based mashup of real songs** | **Describe it in words, AI builds it from YOUR music. No one does this.** |

The key differentiator vs. Suno/Udio: they generate synthetic music inspired by a style. We remix the actual songs you love. Hearing Hendrix's real guitar under Doom's real voice is fundamentally different from hearing AI-generated imitations of them.
