# Share Link Playback — Implementation Plan

Status: Proposed
Date: 2026-03-04
Owner: musicMixer

## Goal

Allow a user to generate a link for a completed remix so friends can open it and listen immediately, without uploading songs or creating a session themselves.

## Non-Goals (MVP)

- Authentication or user accounts
- Private/public visibility settings beyond "unlisted link"
- Social features (comments, likes, remix forks)

## Current State (Codebase Reality)

- A remix is identified by `session_id` and served from `GET /api/remix/{session_id}/audio`.
- The frontend is a single-screen app with no router; there is no deep-link handling today.
- Session metadata (`explanation`, `warnings`) is held in memory and can be lost on backend restart.
- UI says remixes expire in about 3 hours, but TTL enforcement is not currently implemented in runtime code.

## MVP Product Decisions

- Share link format: `https://<app-host>/?listen=<session_id>`
- Link model: unlisted; anyone with the link can listen
- Shared view: playback-only surface with explanation and warnings
- Expiration behavior: if missing/expired, show a friendly "remix no longer available" state
- Retention behavior (MVP): enforce a 3-hour TTL for shared remix artifacts and run cleanup continuously

## MVP Storage Contract

- Public remix assets (`manifest.json`, `remix.mp3`) must be stored on persistent shared storage available to all app instances.
- Local ephemeral container filesystems are not sufficient for MVP.
- Deployments must mount/configure a shared durable path for `data/remixes/{session_id}` before enabling share links.

## Architecture Overview

1. Persist a remix "public manifest" to disk when pipeline completes.
2. Add a public read endpoint that returns manifest + playable audio URL.
3. Add frontend URL hydration for `listen` query param.
4. Add "Copy Share Link" action in the ready player.

## Backend Plan

### Phase 1: Persist Public Remix Manifest

Files:
- `backend/src/musicmixer/services/pipeline.py`

Changes:
- On successful pipeline completion, write `data/remixes/{session_id}/manifest.json`.
- Include fields:
  - `session_id`
  - `created_at`
  - `expires_at`
  - `explanation`
  - `warnings`
  - `used_fallback`
  - `audio_filename` (`remix.mp3`)
- Write atomically (`.tmp` then rename) to avoid partial file reads.

### Phase 2: Public Remix Endpoint

Files:
- `backend/src/musicmixer/api/remix.py`

New endpoint:
- `GET /api/remix/{session_id}/public`

Response shape:
- `session_id`
- `status`: `"ready"`
- `audio_url`: `/api/remix/{session_id}/audio`
- `explanation`
- `warnings`
- `usedFallback`
- `expires_at`

Status behavior:
- `200` when manifest + audio exist and are not expired
- `404` when session/manifest/audio is missing, including after cleanup removes expired artifacts/metadata
- `410` when expired and expiry metadata still exists (pre-cleanup window)
- `400` for invalid UUID format
- `audio_url` must only point to a read path that applies the same expiry policy and the same `410` pre-cleanup / `404` post-cleanup behavior.

### Phase 3: TTL Enforcement + Cleanup (Required for MVP)

Files:
- `backend/src/musicmixer/config.py`
- `backend/src/musicmixer/main.py`
- `backend/src/musicmixer/api/remix.py`

Changes:
- Add `remix_ttl_seconds` (default `10800`).
- Check `expires_at` in all public read paths:
  - `GET /api/remix/{session_id}/public`
  - `GET /api/remix/{session_id}/audio` when serving shared links
- For expired resources, return `410` while expiry metadata still exists; after cleanup removes expired artifacts/metadata, return `404` (no tombstones).
- Add periodic cleanup job in app lifespan to delete expired remix artifacts and metadata:
  - `data/remixes/{session_id}`
  - `data/uploads/{session_id}`
  - `data/stems/{session_id}`
- Keep cleanup interval short enough that expired artifacts are removed promptly after TTL.
- Add explicit cache headers on public read responses:
  - `/public`: `Cache-Control: no-store`
  - `/audio`: `Cache-Control: private, max-age=60, must-revalidate` and never a max-age beyond remaining TTL
- Add `Referrer-Policy: no-referrer` on `/public` and `/audio` responses.
- Add query-param log redaction for `listen` values in backend request/application logs (emit masked value, never raw token).
- Document deployment guidance to apply the same `listen` query-param redaction at ingress/proxy log layers.
- Add backend tests for referrer-policy header coverage and `listen` query-param log redaction behavior.

## Frontend Plan

### Phase 4: Client + Type Contracts

Files:
- `frontend/src/types/index.ts`
- `frontend/src/api/client.ts`

Changes:
- Add `PublicRemixResponse` type.
- Add `getPublicRemix(sessionId)` API helper.
- Add small helper to build share URL from `window.location.origin`.
- Define explicit API mapping:
  - persistence/internal: `used_fallback`
  - API response field: `usedFallback`
- Add backend contract test to assert `usedFallback` is returned (and `used_fallback` is not) in `/public` responses.

### Phase 5: URL Hydration and Shared Playback Mode

Files:
- `frontend/src/App.tsx` and/or `frontend/src/components/RemixSession.tsx`
- `frontend/src/components/RemixPlayer.tsx`

Changes:
- Parse `listen` query param at app load.
- Immediately call `history.replaceState` after extracting `listen` to remove the query param from the visible URL without reload.
- If present:
  - Fetch `/api/remix/{id}/public`
  - Render ready/player state from response
  - Skip upload/progress flow
- In listen mode, suppress third-party requests/assets (analytics, beacons, embeds) until URL cleanup has completed.
- If endpoint returns `400`, render invalid-link message + CTA to create new remix.
- If endpoint returns `404` or `410`, render clear unavailable/expired message + CTA to create new remix.
- Model explicit listen-mode state in reducer/persistence flow:
  - top-level mode: `create | listen`
  - listen substates: `loading | ready | invalid | unavailable | expired`
  - transitions:
    - `loading -> ready` on `200`
    - `loading -> invalid` on `400`
    - `loading -> unavailable` on `404`
    - `loading -> expired` on `410`
  - persistence rule: never persist listen-mode hydration errors into create-mode session state.
- Add frontend tests that assert URL cleanup ordering (`history.replaceState` before third-party initialization) and request suppression before cleanup.

### Phase 6: Share Link UX

Files:
- `frontend/src/components/RemixPlayer.tsx`

Changes:
- Add `Copy Share Link` button in ready state.
- Behavior:
  - Build `/?listen=<session_id>` URL
  - Copy via `navigator.clipboard.writeText`
  - Fallback: select-and-copy text field
- Optional enhancement: `navigator.share` on mobile.

## API Contract (MVP)

`GET /api/remix/{session_id}/public`

```json
{
  "session_id": "uuid",
  "status": "ready",
  "audio_url": "/api/remix/uuid/audio",
  "explanation": "string",
  "warnings": ["string"],
  "usedFallback": false,
  "expires_at": "2026-03-04T21:10:00Z"
}
```

Mapping rule: backend storage and internal models may use `used_fallback`, but the public API contract must expose only `usedFallback`.

## Acceptance Criteria

1. After remix completion, user can click `Copy Share Link`.
2. Opening that link in a new browser (or incognito) loads a playable remix view.
3. Shared view shows explanation and warnings.
4. Invalid/missing link IDs show a graceful error state.
5. Expired links return a clear unavailable/expired state.
6. Existing remix creation flow remains unchanged when `listen` param is absent.
7. Shared-link reads never bypass TTL; expired `/public` and `/audio` reads return `410` while expiry metadata exists, then `404` after cleanup removes expired artifacts/metadata.
8. Share-view responses use cache headers consistent with TTL/expiry behavior.
9. Shared view applies leakage mitigations:
   - `Referrer-Policy: no-referrer`
   - immediate `history.replaceState` cleanup of `listen` query param after hydration capture
   - suppress third-party assets/requests in share view until URL cleanup is complete
   - redact `listen` query-param values in application and ingress logs

## Test Plan

Backend tests:
- `200` for valid public remix with manifest + audio present
- `404` for unknown session or missing files
- `410` for expired remix on both `/public` and `/audio` while expiry metadata still exists (pre-cleanup)
- `404` for previously expired remix after cleanup removes expired artifacts/metadata
- `400` for malformed session IDs
- Contract: `/public` response returns `usedFallback` and does not return `used_fallback`
- Cache headers:
  - `/public` returns `Cache-Control: no-store`
  - `/audio` response max-age does not exceed remaining TTL
- Referrer policy:
  - `/public` and `/audio` include `Referrer-Policy: no-referrer`
- Logging hygiene:
  - request/app log formatting redacts `listen` query-param values
  - deployment guidance covers ingress/proxy log redaction for `listen` values

Frontend validation:
- Happy path: generate link, open in incognito, play audio
- Error path: open malformed `listen` value and see invalid-link state
- Error path: open fake UUID link and see unavailable state
- Expired path: verify messaging for both `410` (pre-cleanup) and `404` (post-cleanup)
- Regression: normal create-remix flow still works
- URL hygiene: `history.replaceState` removes `listen` from address bar immediately after hydration capture, without reload
- Third-party suppression: no third-party network requests fire in share view before URL cleanup completes

## Rollout Plan

1. Implement backend manifest + `/public` endpoint + TTL-enforced `/audio` behavior.
2. Ship frontend hydration + share button with listen-mode state handling for `200/400/404/410`.
3. Validate end-to-end in local and deployed environments with persistent shared storage configured.
4. Enable share links only after TTL cleanup and leakage mitigations are verified.

## Future Improvements (Post-MVP)

- Tokenized links (`share_token`) instead of raw `session_id`
- One-click native share sheet on mobile
- Branded share landing metadata (Open Graph title/image)
- Optional "download remix" action with rate limits
