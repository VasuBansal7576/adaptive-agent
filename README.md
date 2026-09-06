# Adaptive Agent

Adaptive Agent is a domain-agnostic control plane that runs tasks in unfamiliar environments and learns bounded procedural changes from verified development outcomes.
An environment provides its documentation, tools, policy, tasks, reset behavior, and trusted evaluator.
The planner stays the same across environments.

The learning loop records a run, receives a redacted trusted outcome, proposes an evidence-linked candidate, compares the candidate with the pinned baseline, and promotes only a passing candidate.
Rejected and quarantined candidates remain inactive.
This project changes procedures and execution configuration, not model weights.

Read [SPEC.md](SPEC.md) for the requirements, contracts, security rules, evaluation protocol, and acceptance meanings.
Read [MILESTONES.md](MILESTONES.md) for implementation evidence and delivery gates.

## Current status

The current local `main` is `7c65f5b`.
The root-verified real path reached the production UI, the authenticated Luna subscription, the Docker runtime, a broker write, and the trusted evaluator on the clean Store and data directory `/private/tmp/adaptive-agent-main-20260906`.
The run was `run_a1662f700a9a400b9e0f80b87a52f76e`.

The run produced learning candidate `cand_a095170f9c6d4f58943deb7539270de0` from three broker references and 7,206 learning tokens.
Candidate proposal validation passed.
No heldout panel or performance result has been run from this checkpoint.
Session 6 is fixing the full lifecycle 360/720 regression.

The accepted Luna-through-ChatGPT subscription path does not expose an API-key or provider switch.
The trusted parent accounts actual model usage and rejects after aggregate token exhaustion.
The strict provider output-token cap remains an accepted but unmet criterion.
The SDK nominal cost is a proxy, and economic billing is unknown.
No publication or submission is authorized.

## Set up

Requirements are Python 3.12 or later, [uv](https://docs.astral.sh/uv/), and [Bun](https://bun.sh/).

Install the backend and development dependencies from the project metadata.

```sh
uv sync --extra dev
```

Install the console dependencies from `package.json`.

```sh
bun install
```

## Run locally

Start the control API in one terminal.

```sh
uv run adaptive-agent --data-dir .adaptive-agent
```

The API listens on `127.0.0.1:8000` by default.
The `--data-dir` option selects the SQLite database and content-addressed artifacts directory.

Start the Vite console in a second terminal.

```sh
bun run dev
```

Open the URL printed by Vite, usually `http://localhost:5173`.
The development server proxies `/api` to the API at `http://127.0.0.1:8000`.
The console uses the live API by default.

To serve a built console from the Python process, build it first.

```sh
bun run build
uv run adaptive-agent --data-dir .adaptive-agent --console-dist console/dist
```

Open `http://127.0.0.1:8000` after the API starts.
The API accepts `--host`, `--port`, `--data-dir`, and `--console-dist`.

The authenticated planner entry point is available for an adapter that implements the source contract.

```sh
uv run adaptive-agent-plan \
  --goal "<goal>" \
  --environment @path/to/environment.json \
  --adapter module:factory
```

The adapter factory must return a model client, kernel, and evidence sink.

## Verify changes

Run the backend and console checks from the repository root.

```sh
uv run pytest
bun run test
bunx tsc --noEmit
bun run build
```

With the API already running, exercise the live session, run, event-stream, and cancellation contract.

```sh
bun run api:smoke
```

Set `API_BASE` to target another API URL.
The smoke command exits successfully with a skip message when the API is unreachable, so check its output.

## Architecture

The repository uses one Python control plane and one React console.

| Component | Responsibility |
| --- | --- |
| `src/adaptive_agent/api.py` and `app.py` | FastAPI routes, session bootstrap, run lifecycle, learning, evaluation, and static console serving |
| `Store` and content-addressed artifacts | SQLite metadata, immutable references, event history, evidence, candidates, and activation history |
| Environment registry and packages | Documentation, task inputs, tool schemas, policy, reset behavior, fixture handlers, and trusted evaluator bindings |
| Planner and Prime runtime adapter | Model calls, persistent task kernel state, optional bounded children, and runtime provenance |
| Broker and controller | Capability checks, schema validation, approvals, idempotency, budgets, dispatch, reconciliation, and event recording |
| Learning, evaluation, and promotion services | Redacted evidence retrieval, bounded candidate construction, independent baseline/candidate evaluation, mechanical gating, quarantine, and rollback |
| `console/` | React, Tailwind, and Vite operator views over REST and cursor-based SSE |

An operator registers an environment, creates a run with a model and budget, and launches it.
The executor discovers a plan and sends tool requests through the broker.
The evaluator records the trusted outcome outside the learner.
Development evidence can produce a bounded candidate.
The evaluation controller compares the candidate with a reset baseline before the promotion service changes the active pointer.
Each run stays pinned to its starting bundle.

## Trust boundary

The learner, retrieved documentation, skills, model output, and tool output are untrusted relative to policy and evaluator authority.

- The control API, broker, evaluator, and promotion service run outside the learner runtime.
- The Docker runtime receives a writable task directory and approved read-only inputs, without host credentials or evaluator answers.
- The learner cannot write active skills, policy, evaluator state, promotion records, or host files.
- The broker is the only path to external tools.
  It validates the run capability, schema, resource scope, approval token, idempotency key, and policy immediately before dispatch.
- Trusted outcomes use `evaluator_only` visibility and enter learner-visible evidence only after redaction.
- The console escapes untrusted evidence and cannot manufacture a passing evaluation.
- The default service binds to loopback and uses a same-origin operator session.
  Non-loopback hosting needs an explicit authentication and tenant-isolation design.

Documentation can describe a tool but cannot grant permission.
A skill precondition cannot replace a fresh policy check or live-state check.
An uncertain external effect becomes `OUTCOME_UNKNOWN` and blocks automatic repetition until reconciliation.

## Fixtures and real execution

Fixture and real execution are separate evidence classes.

| Mode | What it uses | What it proves |
| --- | --- | --- |
| Backend tests and probes | Deterministic providers, in-memory handlers, isolated temporary Stores, and synthetic reports | Controller, broker, storage, safety, and lifecycle behavior at the tested boundary |
| Console simulation | Deterministic UI fixtures, enabled only with `?sim=1` in a development build or the development toggle | Console rendering and interaction states; it is not model inference |
| Live console and API | REST and SSE against the Python control plane | API and console transport behavior when the API is reachable |
| Real product run | Authenticated Luna, Docker runtime, broker dispatch, isolated Store and data paths, and the trusted evaluator | A model-to-tool-to-evaluator run with retained provenance for that run |

The live console defaults to the real API.
The simulation mode always shows a simulation label and never presents stream connectivity as model inference.
Tests and probes must use newly allocated temporary Store and data paths.
Do not treat a fixture pass, a transport smoke, or a candidate proposal validation as a heldout performance result.

## Limitations

- The current candidate has passed proposal validation only.
  It has not passed an independent heldout comparison.
- No sealed four-environment B0/L/A panel or measured performance claim exists.
- Session 6 is fixing the full lifecycle 360/720 regression before the complete workload can be accepted.
- The strict per-call provider output-token cap is unmet because the accepted subscription path exposes no hard provider switch or API-key boundary.
  Aggregate trusted accounting and local post-response checks do not prove provider enforcement.
- SDK-reported nominal cost does not prove economic billing.
- The default deployment is a single loopback operator.
  Real financial, customer, or production IT writes are outside the default evaluation profile.
- Publication, deployment, and submission are not authorized.

This README summarizes the current implementation and evidence.
It does not change any requirement, acceptance criterion, blocker, or pass meaning in [SPEC.md](SPEC.md).
