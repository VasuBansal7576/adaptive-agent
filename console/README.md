# Operator Console

React + Tailwind + Vite console for the Adaptive Agent control plane (SPEC "Operator console" section). Runs under Bun.

## Scope

- **Environment registry**: package validation with required-field errors that preserve user input; evaluator-unavailable warning blocks activation.
- **Runs**: run list, live event stream (SSE, cursor-based resume), budget meters, cancellation, approval dialog showing exact canonical arguments, scope, and expiry.
- **Skill library**: active / proposed / rejected / quarantined / rolled-back versions distinguished.
- **Candidates**: bounded immutable diff, *predicted effect* and *measured result* always labeled separately, gate reasons, rollback with audited reason and affected runs.

## Transport modes

- `simulation` (default): deterministic fixtures, always labeled with a persistent
  "SIMULATED — development fixture, not live inference" banner. Never presented as live inference.
- `live`: REST + SSE against the SPEC control API (`createRestTransport` in `src/api/rest.ts`). Expected endpoints are documented there; contract deltas should change only that file plus `src/api/types.ts`.

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
