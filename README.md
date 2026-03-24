# musicMixer

A web app that lets anyone create a music remix by prompting AI. Upload two songs, describe the mashup you want, and the AI splits songs into stems, combines them, auto-matches tempo and key, and plays it back. Anyone can be a DJ.

## Try it

[mixer.awill.co](https://mixer.awill.co)

## How it works

1. Upload two songs (MP3/WAV)
2. Type a prompt describing the mashup (e.g., "Hendrix guitar with MF Doom rapping over it")
3. AI separates stems (vocals, drums, bass, etc.), selects the right ones, matches tempo and key
4. Transparent progress updates as it works
5. Remix plays back

Every remix is ephemeral — replayable for up to 3 hours, then gone forever.

## What's under the hood

- **Stem splitting** — GPU-accelerated audio separation via Modal (A10G)
- **Key matching** — detects keys of both tracks and shifts instrumentals to align
- **BPM alignment** — matches tempo across tracks
- **LLM-driven mixing** — AI makes stylistic decisions (EQ, bass, drums, tone) based on a structured representation of each track, then translates those into deterministic audio processing
- **Frontend:** React + TypeScript
- **Backend:** Python API + audio processing pipeline
- **Compute:** Modal (cloud GPUs)
