# Operator Console

React + Tailwind + Vite console for the Adaptive Agent control plane (SPEC "Operator console" section). Runs under Bun.

## Scope

- **Environment registry**: package validation with required-field errors that preserve user input; evaluator-unavailable warning blocks activation.
- **Runs**: run list, live event stream (SSE, cursor-based resume), budget meters, cancellation, approval dialog showing exact canonical arguments, scope, and expiry.
- **Skill library**: active / proposed / rejected / quarantined / rolled-back versions distinguished.
- **Candidates**: bounded immutable diff, *predicted effect* and *measured result* always labeled separately, gate reasons, rollback with audited reason and affected runs.

## Transport modes

- `live` (default): REST + SSE against the SPEC control API (`createRestTransport` in `src/api/rest.ts`). Every payload passes boundary-schema validation (`src/api/validate.ts`) before entering UI state; API errors surface as visible banners with correlation IDs.
- `simulation`: deterministic fixtures, available only as an **explicit dev-only opt-in** — `?sim=1` on a dev build, or the "Use simulation (dev)" toggle (hidden in production builds). Always labeled with a persistent "SIMULATED — development fixture, not live inference" banner, and its stream badge never shows a green "connected" state that could be mistaken for live inference.

Switching modes resets ALL per-mode state (runs, events, candidates, cursors); the stream badge reports stream connectivity only, never model inference.

## Operator actions

- **Create run**: goal, environment, model profile, budget ceilings (tool calls / wall time / required nonzero model token cap), execution mode; idempotency key generated per submission and reused on retry; the server pins the active bundle.
- **Run learning cycle**: submits the evidence-linked candidate proposal for independent evaluation.
- **Register environment**: full manifest — docs (with hashes + learner/operator classification), tool schemas, declared execution modes, policy/evaluator/reset references, capability metadata.
- **Cancel / approval / rollback** failures surface visibly with the API correlation ID.

## Expected backend endpoints (aligned with adaptive-agent-2 commit 7d3c2b5)

GET `/environments`, `/runs`, `/runs/{id}`, `/skills`, `/candidates`; POST `/environments` (strict full manifest: `docs[]` + `taskGoals[]` + canonical `{id,version,sha256}` refs + `executionModes[]`), `/environments/form` (string compatibility form), `/environments/validate` (string projection), `/runs` (canonical `taskRef` projection with `idempotencyKey`, `executionMode` recorded on the run and its events), `/runs/{id}/launch` (explicit launch, 202), `/runs/{id}/cancel`, `/runs/{id}/approvals/{approvalId}`, `/learning/launch` (`{runId, predictedEffect, evidenceIds}`), `/candidates/{id}/rollback`; SSE `/runs/{id}/events?cursor=N` with monotonically increasing sequence numbers. Error envelope `{code, message, correlationId, retry}` (FastAPI `detail` envelopes unwrapped).

## States covered

loading, empty (registry/skills/candidates), load failure with retry, cancellation (no undo claim), OUTCOME_UNKNOWN reconciliation guidance, stale stream (banner + cursor resume, no duplicate events), approval (focus trap, Escape, focus restoration), rollback audit history.

## Accessibility

Keyboard tab navigation (arrow/Home/End, roving tabindex), `aria-selected`, `aria-current`, focus restoration after dialogs, `aria-live` status/alert regions, every status pairs a glyph with a text label (non-color), field-associated errors with `aria-describedby`/`aria-invalid`. Layouts are usable at 375, 768, and 1440 px (Tailwind responsive grid).

## Commands

```sh
bun install
bun run dev      # vite dev server
bun run test     # vitest + testing-library (13 tests)
bun run build    # static build to console/dist (relative base)
bunx tsc --noEmit
```

## Local preview

Build, then preview `console/dist/index.html` through the AO preview panel (no public deployment):

```sh
bun run build
ao preview console/dist/index.html
```
