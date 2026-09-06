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

The public repository is [github.com/VasuBansal7576/adaptive-agent](https://github.com/VasuBansal7576/adaptive-agent).
The verified root-committed `main` checkpoint is `ec2cad7`.
The public repository and its `main` branch were verified at this checkpoint.

The current checkpoint records 55 combined tests passed in 72.87 seconds, with source bytes verified.
This is integration evidence, not an achieved performance claim.

Historical verification at `ccf3b3a` recorded 86 focused tests and a full raw Prime scripted production lifecycle passing in 224.41 seconds across 60 training, 360 validation, and 720 final runs, including restart and tamper checks.

Earlier backend and console checks remain historical evidence, including 123 focused backend checks, 71 console tests, clean TypeScript and production-build checks, and a real local API browser pass at 375, 768, and 1440 pixel widths with no overflow or page errors.

Retained root evidence at `a882641` records the full synthetic 1,080-case run, restart coverage, and two authenticated reports passing in 48.25 seconds.
The combined production bytes at `2a54cf9` reproduced that same full synthetic pass.
Core and API verification recorded 64 passing checks, and the corrected evaluator-privacy projection test passed separately.
An earlier retained backend report recorded 240 passing checks.
These are synthetic or bounded integration evidence, not a measured generalization result.

A separate retained real product run reached the production UI, authenticated Luna, Docker, the broker, and the trusted evaluator.
It is identified by run `run_a1662f700a9a400b9e0f80b87a52f76e` and evidence-linked candidate proposal `cand_a095170f9c6d4f58943deb7539270de0`.
That candidate was not promoted and this successful product path is distinct from the later failed full benchmark.

An earlier real benchmark attempt on `2a54cf9` started at 2026-09-06 13:53 UTC and failed before the held-out phase after one successful task and five development budget exhaustions.
It produced no learning result, validation result, final metrics, or measured improvement.
M4 is therefore not complete.

The AppWorld adapter and CLI are integrated with installed AppWorld `0.1.3.post1`.
A separate actual AppWorld experiment remains RUNNING against frozen source `28baed7`.
Its 8 train cells and 16 completed dev cells have all ended with observed model-budget failures; no final report or achieved improvement claim is available, and immutable jobs continue.

Multi-run learning is integrated.
On frozen source `15e1624`, actual training completed 60 tasks, with 48 successes and 12 budget failures.
The learner selected a bounded set of 8 sources.
Learner usage of 6,678 input and 582 output tokens produced an unpromoted candidate; receipt accounting then failed, and that failure was fixed generically.

The original full experiment against frozen source `ccf3b3a` stopped at 2026-09-06 18:53 UTC.
Its 60-task training stage completed with 50 successes and 10 model-budget failures, learning generated an unpromoted candidate, and 3 transfer, 3 adaptation, and 3 safety cells completed.
Validation reached 116 completed cells and 1 infrastructure failure, then failed closed; no validation report or final 720-cell panel exists, so M4 remains unmet.

A generic-error evaluation run did not produce trusted model and outcome evidence.
The operator error was that the Prime CLI JSON stream did not contain a final assistant message.
The upstream cause was not retained and is not known.
The Prime JSON terminal-error parsing gap is being fixed separately for a future source revision; the zero-retry frozen run was not restarted or edited.

The accepted Luna-through-ChatGPT subscription path does not expose an API-key or provider switch.
The trusted parent accounts actual model usage and rejects after aggregate token exhaustion.
The strict provider output-token cap remains an accepted but unmet criterion.
The SDK nominal cost is a proxy, and economic billing is unknown.
Public repository visibility is authorized and verified.
Public demo posts, deployment, and submission remain unauthorized.

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

The API can start without Docker, Prime Agent, or the authenticated subscription.
Model-backed runs need all of them.

The Prime runtime requires a working Docker daemon, an installed `prime-agent` CLI with its bundled `rlm/repl.py` runtime, an authenticated subscription directory, and an AO session identifier.
The adapter accepts only provider `openai-codex` with model `openai-codex/gpt-5.6-luna`.
Set `PRIME_AGENT_CODING_AGENT_DIR` to the AO-authorized coding-agent directory that contains the subscription authentication state.
Set `AO_SESSION_ID` to the trusted AO session identifier.
The identifier must contain only letters, digits, `_`, `.`, `:`, or `-`.
Do not print or commit the subscription directory contents.

Check the non-secret prerequisites before a model run.

```sh
docker info
prime-agent --version
prime-agent --help
test -n "$PRIME_AGENT_CODING_AGENT_DIR"
case "${AO_SESSION_ID:-}" in
  ""|*[!A-Za-z0-9_.:-]*) exit 1 ;;
esac
```

The API starts when these checks fail because it validates the Prime runtime when a model-backed run launches.
The run fails when Docker, the Prime runtime bundle, the subscription directory, or `AO_SESSION_ID` is unavailable.

An optional authenticated reachability check invokes the subscription model and must not use a retained benchmark or demo data directory.

```sh
prime-agent --print --no-tools --provider openai-codex --model openai-codex/gpt-5.6-luna
```

## AppWorld setup and experiment

AppWorld is an external published simulated environment.
Use an existing AppWorld checkout and an existing Python executable with AppWorld exactly at version `0.1.3.post1`.
The commands below verify that boundary; they do not install or download AppWorld or Prime Agent.

Replace the portable path and image pin placeholders with the exact values for the frozen run.
The source pin must be the full revision derived from the clean frozen Adaptive Agent checkout being executed.
The image and core planner pins must be exact immutable values, not tags or `image-unpinned`.
The data directory must be new and empty for `--initialize`.

```sh
ADAPTIVE_AGENT_CHECKOUT="/path/to/adaptive-agent"
APPWORLD_ROOT="/path/to/AppWorld"
APPWORLD_PYTHON="/path/to/appworld-python"
APPWORLD_SETUP_MANIFEST="$(mktemp)"
APPWORLD_DATA_DIR="$(mktemp -d)"
cd "$ADAPTIVE_AGENT_CHECKOUT"
test -z "$(git status --porcelain)" || { printf '%s\n' "Adaptive Agent checkout must be clean" >&2; exit 1; }
SOURCE_REVISION="$(git rev-parse HEAD)"
IMAGE_DIGEST="sha256:<exact-frozen-image-digest>"
unset ADAPTIVE_AGENT_CORE_PLANNER_HASH
CORE_PLANNER_HASH="$(uv run python -c 'from adaptive_agent.app import _freeze_core_planner_hash; print(_freeze_core_planner_hash())')"

test -d "$APPWORLD_ROOT/data" || { printf '%s\n' "AppWorld root must contain data/" >&2; exit 1; }
test -x "$APPWORLD_PYTHON" || { printf '%s\n' "AppWorld Python executable is missing or not executable" >&2; exit 1; }
test "$("$APPWORLD_PYTHON" -c 'import importlib.metadata as m; print(m.version("appworld"))')" = "0.1.3.post1" || { printf '%s\n' "AppWorld version must be 0.1.3.post1" >&2; exit 1; }

uv run adaptive-agent-appworld setup \
  --root "$APPWORLD_ROOT" \
  --python "$APPWORLD_PYTHON" \
  --package-version 0.1.3.post1 \
  --output "$APPWORLD_SETUP_MANIFEST"
```

The setup manifest describes the public catalog boundary.
The catalog has no ground-truth or task-report access, and test-task access remains sealed until explicitly enabled.
The isolated evaluator uses minimal evaluator-only ground truth and returns aggregate-only responses.

Complete the Prime subscription prerequisites above, including `PRIME_AGENT_CODING_AGENT_DIR` and `AO_SESSION_ID`, then export the same frozen pins used by the command.

```sh
export ADAPTIVE_AGENT_SOURCE_REVISION="$SOURCE_REVISION"
export ADAPTIVE_AGENT_IMAGE_DIGEST="$IMAGE_DIGEST"

uv run adaptive-agent-appworld-experiment \
  --initialize \
  --data-dir "$APPWORLD_DATA_DIR" \
  --appworld-root "$APPWORLD_ROOT" \
  --appworld-python "$APPWORLD_PYTHON" \
  --source-revision "$SOURCE_REVISION" \
  --image-digest "$IMAGE_DIGEST" \
  --core-planner-hash "$CORE_PLANNER_HASH" \
  --model-profile openai-codex/gpt-5.6-luna \
  --model-tokens 20000 \
  --prime-executable prime-agent \
  --coding-agent-dir "$PRIME_AGENT_CODING_AGENT_DIR"
```

The experiment protocol fixes 8 train tasks, 20 `dev` tasks across three arms, and 20 `test_normal` tasks across three arms.
Each task has the fixed runtime budget of 20,000 model tokens, 32 tool calls, and 90 seconds.
The train, development, and final task cells total 8, 60, and 60 respectively.

To continue an initialized experiment, use the same data directory and exactly the same pins.
Resume verifies the immutable manifest, frozen runtime pins, and durable cell receipts.
Completed cells are not re-dispatched; a missing cell can continue, while a tampered or invalid receipt fails closed.

```sh
uv run adaptive-agent-appworld-experiment \
  --resume \
  --data-dir "$APPWORLD_DATA_DIR" \
  --appworld-root "$APPWORLD_ROOT" \
  --appworld-python "$APPWORLD_PYTHON" \
  --source-revision "$SOURCE_REVISION" \
  --image-digest "$IMAGE_DIGEST" \
  --core-planner-hash "$CORE_PLANNER_HASH" \
  --model-profile openai-codex/gpt-5.6-luna \
  --model-tokens 20000 \
  --prime-executable prime-agent \
  --coding-agent-dir "$PRIME_AGENT_CODING_AGENT_DIR"
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

The API smoke creates and launches a run.
It uses the live model path when the API has the required runtime and subscription configuration.
Run it only against a disposable API and data directory, and set `API_BASE` explicitly.

```sh
SMOKE_DATA_DIR="$(mktemp -d -t adaptive-agent-smoke.XXXXXX)"
uv run adaptive-agent --data-dir "$SMOKE_DATA_DIR" --port 8010
```

In a second terminal, run the smoke against that disposable API.

```sh
API_BASE=http://127.0.0.1:8010 bun run api:smoke
```

The smoke command exits successfully with a skip message when the API is unreachable, so check its output.
The smoke can consume subscription usage and can execute fixture writes.
Do not point it at the default `:8000` API when that API uses retained benchmark or demo data.

For a built-console smoke that performs only GET requests and does not launch a model run, build the console before starting a separate disposable API and set `BASE` explicitly.

```sh
APP_DATA_DIR="$(mktemp -d -t adaptive-agent-app-smoke.XXXXXX)"
bun run build
uv run adaptive-agent --data-dir "$APP_DATA_DIR" --port 8011 --console-dist console/dist
```

In a second terminal, run the GET-only smoke.

```sh
BASE=http://127.0.0.1:8011 node console/scripts/prod-app-smoke.mjs
```

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

## Acknowledgments

Adaptive Agent uses Prime Agent by Prime Intellect AI under the MIT License.
The installed Prime `0.9.2` runtime is separate from the initial static-source review pinned at [Prime Agent commit `9c54a35dac3a2ad17910074d66664859ea175666`](https://github.com/PrimeIntellect-ai/prime-agent/tree/9c54a35dac3a2ad17910074d66664859ea175666).
The exact applicable notice is retained in [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).
This notice applies to Prime Agent reuse only and does not grant a license to Adaptive Agent.

## Limitations

- The current candidate has passed proposal validation only.
  It has not passed an independent heldout comparison.
- No sealed four-environment B0/L/A panel or measured performance claim exists.
- The strict per-call provider output-token cap is unmet because the accepted subscription path exposes no hard provider switch or API-key boundary.
  Aggregate trusted accounting and local post-response checks do not prove provider enforcement.
- SDK-reported nominal cost does not prove economic billing.
- The default deployment is a single loopback operator.
  Real financial, customer, or production IT writes are outside the default evaluation profile.
- Public repository visibility is verified.
  Public demo posts, deployment, and submission are not authorized.

This README summarizes the current implementation and evidence.
It does not change any requirement, acceptance criterion, blocker, or pass meaning in [SPEC.md](SPEC.md).
