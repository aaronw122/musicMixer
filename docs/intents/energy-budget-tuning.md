---
title: "Energy Budget Tuning"
author: "human:aaron"
version: 1
created: 2026-03-03
---

# Energy Budget Tuning

## WANT

Fix the gain mapper's too-conservative output. Currently 5 of 8 sections fire "below target" energy budget warnings because:

- Base role gains (support=0.67, background=0.45) are too low when combined with LUFS attenuation
- LUFS attenuation hits stems around -14 to -15 LUFS with ~0.80x multiplier, compounding the problem
- Example: support (0.67) x LUFS attenuation (0.80) = 0.54 effective gain, but energy budget expects 0.65-0.95

Three levers to pull:
1. **Raise base role gains** — support from 0.67 to ~0.75, background from 0.45 to ~0.55
2. **Shift LUFS neutral reference** — from -20 to -18, so stems at -15 LUFS get less attenuation
3. **Auto-correct energy budgets** — when section total is below target, scale gains up proportionally instead of just warning

## DON'T

- Don't touch the LLM prompt — the intent-based refactor is done
- Don't change the 5-role system (lead/support/background/texture/silent)
- Don't add new modules or pipeline steps — tune within gain_mapper.py and pipeline.py
- Don't over-engineer — prefer simple parameter changes over complex adaptive logic

## LIKE

- The existing gain mapper architecture (5-step pipeline) is solid — just needs better numbers
- The energy budget validation already detects the problem — make it fix the problem too

## FOR

- Backend gain_mapper.py (primary) and pipeline.py (secondary)
- Tested against Angel from Montgomery + Don't You Worry Child remix session
- Goal: remixes sound full and comparable to YouTube originals

## ENSURE

- Zero energy budget warnings on the Angel/Montgomery + Don't You Worry Child test case (currently 5 of 8 sections fire warnings)
- Final mastered LUFS within 0.5 dB of -12.0 target (currently -12.7, which is 0.7 dB short)
- Instrumentals sound subjectively full — no thin/quiet sections
- All existing tests pass (871 pass currently, 1 pre-existing flaky)
- No stem gain exceeds 1.0 after tuning
- Role ordering preserved: lead > support > background > texture > silent

## TRUST

- [autonomous] Adjust base gain numbers, LUFS thresholds, energy multipliers
- [autonomous] Convert energy budget from warn-only to auto-correct
- [autonomous] Update tests to match new expected values
- [autonomous] Run tests, iterate until green
- [autonomous] Commit, push, create PR against integration branch
