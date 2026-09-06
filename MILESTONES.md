# Adaptive Agent milestones

Status: implementation is active in parallel AO worktrees for the confirmed Track 1 product, with integration tracking recorded here.
Product acceptance and competition compliance remain unverified until implementation and submission evidence exist.
[SPEC.md](SPEC.md) defines stable requirements, contracts, acceptance scenarios, and evaluation cases.

## Authorization and sequencing

M0 and M1 completed the documentation phase.
M0 is complete through the supplied product and official-rule clarification.
M1 ends with a verified local commit containing only `SPEC.md`, `MILESTONES.md`, and `README.md`.
The local `main` fast-forward and filesystem-only origin synchronization for the documentation commits are authorized and complete.
M2 through M4 are active implementation and evaluation gates across the worker branches.
M5 and M6 require future publication or submission authority appropriate to each action.
This documentation worker does not install dependencies, create application code, or edit implementation branches.
It records the current implementation refs and evidence without treating parallel branch work as integrated.
Public remote creation, publication, pushes to public remotes, pull requests, and submission remain incomplete.

The dependency chain is M0 -> M1 -> M2 -> M3 -> M4 -> M5 -> M6.
After M2 fixes the contracts, environment fixtures and console work can proceed independently under assigned owners.
The evaluator owner must remain independent of learner-controlled code and protect validation and final test contents.
No additional workers are spawned during this documentation task.

## Implementation and integration snapshot

The following snapshot was read from the current worktrees on 2026-09-06.
The refs identify the latest observed committed state, not a final merge result.
Dirty worktrees contain additional uncommitted implementation changes, which are not counted as integrated evidence.

| AO session | Observed ref and state | Implemented component status |
| --- | --- | --- |
| adaptive-agent-2 | `0a18b31`, dirty | Python package entry points, FastAPI control API, durable runtime wiring, Luna planner/evaluator integration, and a console integration snapshot; current uncommitted API, app, controller, learning-store, model, Prime-runtime, and learning-runtime work remains pending |
| adaptive-agent-3 | `93c6e91`, dirty | Isolated Prime runtime adapter, Docker learner boundary, bounded child planner, and shared parent/child ledger; planner and child-planner tests still have uncommitted changes |
| adaptive-agent-4 | `deeee4a`, dirty | Durable SQLite Store, Broker, Candidate, Controller, evidence, and learning projection seams; controller, store, and core-test changes remain uncommitted |
| adaptive-agent-5 | `7c84998`, dirty | React/Tailwind/Vite operator console, live REST/SSE transport, dev-only simulation transport, evidence provenance, recovery, approval, cancellation, and rollback UI; `RunView.tsx` remains uncommitted |
| adaptive-agent-6 | `a1c7d18`, clean | Resumable benchmark driver, atomic task and evaluator allocation, development-smoke gate, frozen arms, and comparison-run identity fencing |
| adaptive-agent-7 | `3ba9286`, clean | Evidence-linked bounded learning, retrieval, durable learning runtime, trusted failed-run diagnostics, and narrow broker learning projections |

The implemented Python commands are `adaptive-agent` and `adaptive-agent-plan` as declared in the session-2 `pyproject.toml`.
The implemented console commands are `bun run dev`, `bun run test`, `bun run build`, `bunx tsc --noEmit`, and `bun run api:smoke` as declared in the session-5 package manifest.
The Prime bridge investigation also verified `prime-agent --version`, `prime-agent --help`, and the authenticated `prime-agent --print --no-tools --provider openai-codex --model openai-codex/gpt-5.6-luna` smoke command.

## Evidence ledger and current gate status

Real-run and fixture evidence remain separate.
The following records are the exact evidence available from the inspected worker branches and their commit reports.

| Evidence | Classification | What it establishes | What it does not establish |
| --- | --- | --- | --- |
| Prime 0.9.2 session 3, ChatGPT subscription, `openai-codex/gpt-5.6-luna`, actual Python `2+2` result `4` | Real runtime smoke | Subscription authentication and a narrow Python execution path | Product bridge, evaluator, safety, or performance acceptance |
| `prime-agent --print --no-tools --provider openai-codex --model openai-codex/gpt-5.6-luna` returned `MODEL_SPIKE_OK` in 10.24s | Real provider smoke | Model reachability through the authenticated path | A goal/tool/evaluator run or provider hard output cap |
| Prime adapter state-sharing cells returned `42` then `41`; host bridge returned structured data; forbidden learner requests were denied | Real Prime/Docker boundary smoke | Persistent kernel state, broker boundary, and denial behavior in the adapter spike | Complete control-plane learning and promotion evidence |
| Session-5 `bun run api:smoke` worker report: live FastAPI session, registration, create/launch, SSE cursor, and cancellation smoke PASS | Real control-plane transport with synthetic task | Live API/session/SSE/cancellation wiring | Real model inference or sealed evaluator evidence; the script's smoke environment is synthetic |
| Session-4 report: three SQLite integration tests passed, including reopen and positive/negative verifier cases | Real local persistence test | SQLite durability and verifier boundary behavior | External provider effects or product acceptance |
| Session-4 and session-6 backend test reports, including 31 and 30 passed suites | Synthetic fixture/test evidence | Store, broker, controller, learning, and benchmark branches exercise their contracts | Real provider behavior; session-6 explicitly reports all test drivers/providers are synthetic |
| Session-5 console reports, up to 49/49 tests with TypeScript and build clean | Synthetic UI and captured-wire evidence | Console state, accessibility assertions, transport normalization, and recovery regressions | Browser-width acceptance, live inference, or sealed evaluation |
| Devin 3000.6.14 session 4 and OpenCode session 5 each ran `pwd` successfully with no file changes | Real build-agent smoke | Those agent sessions could execute a basic command | Any product behavior |

The current gate status is deliberately conservative.
M0 and M1 are complete as documentation and local integration milestones.
M2 has substantial implementation and narrow real-run evidence, but remains unmet until the actual neutral goal-to-model-to-tool-to-trusted-evaluator path and required boundary checks are retained together.
M3 has core learning, storage, broker, candidate, and benchmark components, but the complete three-pack lifecycle and all required acceptance scenarios are not verified as one integrated product.
M4 is unmet because there is no sealed four-environment B0/L/A evaluation, measured cross-domain result, or final resource-budget evidence.
M5 and M6 have not started, and there are no public repository, social-demo, deployment, or submission claims.

One user-accepted subscription limitation is still an UNMET criterion.
The `openai-codex` subscription path has no verified provider-enforced hard output-token cap.
The implementation's local token ceilings, accounting, and post-response rejection are not equivalent to a hard provider output cap.
This remains a blocker for the resource-boundary acceptance work and must not be recorded as a pass.
The acceptance criteria, blockers, and pass meanings in [SPEC.md](SPEC.md) are intentionally untouched.

Prime reuse attribution is required in any later public artifact.
Credit Prime Agent by Prime Intellect AI, pinned for the initial static review at [`9c54a35dac3a2ad17910074d66664859ea175666`](https://github.com/PrimeIntellect-ai/prime-agent/tree/9c54a35dac3a2ad17910074d66664859ea175666), under the MIT License, and retain the applicable MIT notices.
The installed Prime 0.9.2 smoke and the pinned static source review are distinct evidence sources.

## Delivery gates

| ID | Deliverable | Dependencies | Measurable exit evidence |
| --- | --- | --- | --- |
| M0 | Confirmed scope and official constraints | SRC-001 through SRC-008 | One Track 1 product, learning mechanism, phase boundary, event window, rubric, and public artifact obligations recorded |
| M1 | Implementable documentation and local commit | M0 | DOC-001 through DOC-008 checked; only the three requested documents committed; hash and verification results sent to the orchestrator |
| M2 | Prime bridge decision and runnable end-to-end vertical slice | M1; implementation authorization; Prime adapter or isolated fallback; model/budget | A supplied neutral environment goal runs through the actual executor, external broker, trusted outcome evaluator, stored evidence, one candidate comparison, gate decision, and console; no mocked inference is reported as live |
| M3 | Full bounded learning lifecycle and three environment packs | M2; fixed contracts; trusted fixture owners | Finance, support, and IT packages use the same core; automatic proposals, protected validation, direct CRUD gating, rejection, quarantine, version conflict, and rollback pass ACC-001 through ACC-007 and ACC-009 |
| M4 | Complete independent evaluation and operational acceptance | M3; sealed fourth environment; frozen protocol; resource budget | EVAL-001 through EVAL-006 and ACC-008 through ACC-011 plus ACC-015 pass the required acceptance and final performance target; runtime and browser evidence retained |
| M5 | Reviewed release and submission package | M4; publication authority; member prerequisites; official form review | Public artifacts prepared and, when authorized, accessible; demo meets 3 to 5 minutes; rubric mapping, AO session evidence, credits, privacy review, and open limitations checked |
| M6 | Authorized submission with receipt | M5; explicit submission authority; active official form | Public GitHub and social demo post links entered in Devpost; conditional deployment URL supplied; final submission timestamp and receipt retained |

A runnable M2 is evidence that the architecture works, not permission to stop before the full learning, generalization, and evaluation scope.
A candidate that fails the performance gate remains rejected even when a deadline is close.
Completion of this plan does not imply all future acceptance scenarios have already passed.

## M2: Prove the execution and learning boundaries

Timebox the Prime bridge investigation to one working session and 90 minutes of engineering effort.
Check persistent state with two real kernel calls and verify structured output, cancellation, child isolation, external tool access, refine interception, and direct CRUD interception.
Pin the verified source version and record the MIT license notices used.
Select the small backend and runtime adapter based on those results.
Use the specified isolated IPython fallback if Prime cannot meet the adapter boundary, and disclose the choice.
An unavailable Prime API, provider, or safety boundary can delay the Prime path but cannot be hidden behind simulated evidence.

The vertical slice uses one neutral package with documentation, a stateful tool, a policy rule, a reset fixture, and an objective outcome check.
It includes a task whose first attempt can fail and whose verified outcome can support a bounded corrective proposal.
A proposal goes through a real independent comparison with fresh fixture state and an enforced gate.
The candidate may pass or fail depending on measured results.
A deterministic injected report can test controller branches in isolation, but cannot count as measured learning or a product promotion demonstration.
The console shows the real run, evidence, proposal, and decision.

Exit checks include actual kernel state continuity, a denied tool action, resource usage, idempotent run creation, and restart recovery of the active version.
Measure tokens, cost, run latency, and fixture reset time to estimate the full evaluation workload.
Those measurements set provider budgets and execution concurrency before M3 and M4, without changing pass thresholds after results are observed.

## M3: Complete the reusable learning loop

Implement the generic environment registration contract before adding domain fixtures.
Each finance, support, and IT pack supplies tools, documentation, policy, task inputs, reset behavior, and an independent outcome evaluator.
Do not ship planner patches, prewritten tool sequences, domain-specific prompt branches, or test answers through those packs.
Core planner hashes must stay identical across packs.

Keep evidence retrieval, live tool state, kernel working state, and procedural skills distinct.
Require evidence-linked immutable candidates, bounded edits, and validation of all allowed artifact types.
Route Prime refine and direct harness writes into the same candidate path.
Demonstrate rejection of a bad refinement and rollback of a promoted version with unchanged permission enforcement.
Include poisoned feedback, attempted evaluator edits, missing evidence, and stale base hashes in the safety checks.

Build the console's empty, loading, error, approval, cancellation, and recovery states alongside the functional views.
An evidence view must distinguish predicted effects from measured results and must not expose sealed evaluator content.
Define the sealed fourth environment through the independent evaluator owner before freezing final evaluation.

## M4: Measure the full product honestly

Freeze model/provider versions, prompts, core hash, task families, partitions, seeds, budgets, metrics, thresholds, and analysis code before scoring candidates.
Training tasks teach the agent, validation tasks select candidates, and sealed final tasks measure the frozen result.
Do not reuse validation tasks after candidate feedback or expose sealed traces through retrieval, child sessions, exported memories, or operator transcripts available to the learner.

The proposed validation panel has 20 tasks per known environment, three seeds per task, and two arms, totaling 360 scored runs per candidate.
The final panel has 20 tasks per environment, three seeds per task, and fixed baseline, learned agent, and memory-disabled ablation arms across four environments, totaling 720 scored runs.
Training attempts, safety tests, rejected candidates, transfer experiments, and retries add to those totals.
For `N` candidates, the predeclared total workload is `N * 360 + 720 + training + transfer + safety + retries` scored or attempted runs under one overall model, tool, wall-time, and cost budget.
Freeze that overall budget before candidate generation and record every term, including cancelled, rejected, or provider-retry work.
Use the M2 measurements to calculate actual elapsed time and cost before reserving the run budget.
If the panel cannot be completed, report incomplete evidence and withhold unsupported improvement claims.
If the final performance target fails, M4 remains unmet even when all runs finished and the software checks pass.

Run separate within-environment adaptation and leave-one-environment-out experiments.
The sealed fourth environment measures genuine novelty beyond the three known packs, with no planner changes or manual prompt tuning.
Show the primary final transfer result before any optional fourth-environment adaptation experiment.
Report accuracy, reliability, cost, and speed with uncertainty, failures, learning overhead, and the limits of synthetic fixtures.
Compare fixed baseline B0, learned agent L, and memory-disabled ablation A with identical model/task/budget conditions.

Verify the real broker, persistence, cancellation, replay, isolation, and rollback boundaries.
Inspect the actual console at 375, 768, and 1440 pixel widths with keyboard navigation, visible focus, field-associated errors, and non-color status indicators.
A screenshot alone cannot establish keyboard behavior, error recovery, or safe tool execution.
Retain command results and runtime evidence with exact version references.

## M5 and M6: Prepare and submit under separate authority

The official build window is 2026-09-05 21:30 IST to 2026-09-07 03:30 IST.
The corresponding UTC window is 2026-09-05 16:00 to 2026-09-06 22:00.
Use the official closing time as the deadline and recheck the form for any updated cutoff before submission.
A proposed operational target is to finish artifact review at least 60 minutes before closing and submit at least 30 minutes before closing.
These buffers are planning targets, not additional official rules or guaranteed available time.
The orchestrator must compare actual remaining time with measured execution and review needs.
If the full objective cannot fit, surface the conflict without relabeling incomplete work as complete.

Every team member must register and join the official Discord.
Confirm that the submission selects only Track 1.
Verify whether the intended synthetic scenarios are acceptable because organizer feedback was tentative.
Do not make sponsor integration a dependency unless it materially supports the product.

Prepare a public GitHub repository, a public X or LinkedIn demo post, and the corresponding links for Devpost.
Prepare a deployment URL only if deployment is chosen and authorized.
Check public access without relying on a logged-in session.
Redact credentials, private data, sealed test answers, and unnecessary transcripts from all public artifacts.
Keep authoritative evaluator fixtures or expected answers private when exposing them would undermine continued evaluation.
Provide public reproduction instructions and non-sealed sample fixtures instead.

The demo target is 3 minutes within the official 3 to 5 minute range.
The proposed sequence is 0:00 to 0:20 for the problem and generic environment contract, 0:20 to 0:50 for a real run, and 0:50 to 1:30 for evidence-driven refinement and its measured decision.
Use 1:30 to 2:10 for the three-domain and new-environment comparison, 2:10 to 2:30 for bad-refinement rejection and safe rollback, and 2:30 to 3:00 for AO sessions, contribution, and limitations.
Keep the recorded content truthful if a metric fails or a scenario remains unverified.

| Rubric | Weight | Evidence to retain |
| --- | --- | --- |
| AO usage | 25% | Meaningful roles, dashboard session count, task ownership, review and implementation history from the beginning through submission |
| Execution | 25% | Real runnable workflow, bounded tools, failure recovery, and reproducible validation |
| Track fit | 25% | Generated execution configuration, attempts, outcome evaluation, and iterative improvement across unfamiliar tasks |
| Demo | 15% | Clear 3 to 5 minute demonstration with public links and honest results |
| Innovation | 10% | Independent evidence-backed promotion boundary and reusable procedural learning, with open-source contributions credited |

Retain the actual submission confirmation, timestamp, artifact versions, and public links.
A prepared form, draft post, or uploaded video without a submission receipt does not satisfy M6.

## AO ownership, review, and CI

Build-time agents implement and review the product under AO.
The product runtime executes user tasks and optional bounded RLM child work.
These are separate systems, and AO's development session count is not a runtime swarm requirement.

The orchestrator assigns owners for the runtime adapter and broker, the learning controller, the independent evaluator and fixtures, the console, and release evidence.
Each workstream gets a branch/worktree boundary, explicit contracts, dependencies, and a reviewer.
Cross-session contract changes and true blockers go through AO.
The evaluation owner controls hidden partitions and reports rather than delegating that authority to the learner.

Codex AO worker and orchestrator defaults, including existing AO sessions, use GPT-5.6 Luna Medium.
Devin build-time work uses SWE-1.7 Medium.
OpenCode build-time work must use GLM-5.3-Flash, selected, or Muse Spark1.3.
Big Pickle is not permitted.
Do not assign Astra, Sol, or Terra, and do not spawn additional workers for this plan.
Prime is the proposed product runtime and is also available for bounded bridge work.
Prime runtime smoke uses GPT-5.6 Luna through a ChatGPT subscription.
Prime 0.9.2 session 3 authenticated that subscription, configured `openai-codex/gpt-5.6-luna`, and returned `4` for an actual Python `2+2` call.
Default Prime Inference returned HTTP 402 for lack of balance, while the subscription provider worked.
The official kernel bootstrap was repaired with `UV_NO_CACHE=1` after a broken setuptools cache.
Devin 3000.6.14 session 4 and OpenCode session 5 each ran `pwd` successfully with no file changes.
The product bridge, safety isolation, and behavioral evaluation remain unverified.
These smoke results do not by themselves prove task completion.
The GLM contract review is complete, and its findings are addressed in this correction.
No new or repeated review is part of this plan.
Qodo is an optional code reviewer, not an outcome evaluator or a replacement for product tests.
Backpass and Vision are influences and do not require installation.

M1 uses local document checks because the empty repository has no application CI.
After stack selection, required checks include schema validation, backend and frontend checks, targeted runtime integration tests, policy/isolation tests, promotion/rollback tests, and console interaction checks.
The full sealed evaluation is a controlled evidence run, not a routine test that repeatedly leaks answers through CI logs.
Use non-sealed fixtures in regular CI and restrict sealed reports and credentials.
Assign an owner to every failing check and distinguish code regressions from provider outages or missing credentials.
Any future authorized pull request needs review findings addressed and relevant checks passing before integration.
The current documentation task creates no pull request or public remote operation.
Its local `main` fast-forward and filesystem-only `origin/main` synchronization are complete.

## Requirement traceability

| Requirement | Acceptance or verification | Delivery milestone |
| --- | --- | --- |
| DOC-001 | Explicit staged file list and final commit diff | M1 |
| DOC-002 | Source attribution and scope review | M0, M1 |
| DOC-003 | Requirement/acceptance/milestone reference validation | M1 |
| DOC-004 | Workflow and contract walkthrough | M1 |
| DOC-005 | Promotion and evaluation consistency review | M1 |
| DOC-006 | Security, dependencies, ownership, and submission review | M1 |
| DOC-007 | Local links, IDs, formatting, preview, and contradiction checks | M1 |
| DOC-008 | Confirmed commit authority and commit metadata review | M1 |
| TRK-001 | ACC-001 | M2, M4 |
| TRK-002 | ACC-004, EVAL-001 | M3, M4 |
| TRK-003 | ACC-008, ACC-015, EVAL-001, EVAL-002, EVAL-006 | M4, M5 |
| TRK-004 | ACC-012 | M0, M1, M2, M3, M4, M5, M6 |
| TRK-005 | ACC-014 | M5, M6 |
| TRK-006 | ACC-014 | M5, M6 |
| TRK-007 | ACC-014 | M5, M6 |
| TRK-008 | ACC-014 | M0, M5, M6 |
| PRD-001 | ACC-001 | M2, M3, M4 |
| PRD-002 | ACC-002 | M2, M3 |
| PRD-003 | ACC-003 | M2, M3 |
| PRD-004 | ACC-004 | M2, M3 |
| PRD-005 | ACC-005, EVAL-001, EVAL-003 | M2, M3, M4 |
| PRD-006 | ACC-006, EVAL-003 | M2, M3, M4 |
| PRD-007 | ACC-007, EVAL-004 | M3, M4 |
| PRD-008 | ACC-008, EVAL-001, EVAL-002 | M4 |
| PRD-009 | ACC-009, EVAL-003 | M2, M3, M4 |
| PRD-010 | ACC-010 | M2, M3, M4 |
| PRD-011 | ACC-011, EVAL-004, EVAL-005 | M2, M3, M4 |
| PRD-012 | ACC-005, ACC-013 | M2, M3, M5 |
| PRD-013 | ACC-015, EVAL-006 | M4, M5 |

## Documentation verification and handoff

Check every requirement ID for a stable definition, acceptance reference, and milestone mapping.
Review the source boundaries, trust separation, baseline definitions, split isolation, numerical defaults, event timing, and authorization wording across all three files.
Check local links and verify external URLs against the supplied sources, reporting inaccessible sources rather than claiming they loaded.
For SRC-008, preserve that the rules were established by prior browser research from the controlling assistant and relayed to this worker, not independently verified by this worker or this AO orchestrator.
Inspect SPEC.md in the AO Browser without creating a server or installing dependencies.
Check for whitespace errors, em dashes, and prose paragraphs with multiple sentences on one physical line.
For the original M1 commit, stage only the three named specification documents.
For this integration-tracking update, stage only `README.md` and `MILESTONES.md`, then inspect the staged diff before making a conventional local commit without an agent co-author.
Report the commit, changed files, checks, and remaining execution assumptions through AO.
The handoff includes completed local specification commits and local `main` integration, not completed implementation, public publication, or submission.
