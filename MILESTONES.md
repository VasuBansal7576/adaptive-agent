# Adaptive Agent milestones

Status: implementation is active in parallel AO worktrees for the confirmed Track 1 product, with integration tracking recorded here.
Product acceptance and competition compliance remain unverified until implementation and submission evidence exist.
[SPEC.md](SPEC.md) defines stable requirements, contracts, acceptance scenarios, and evaluation cases.

## Authorization and sequencing

M0 and M1 completed the documentation phase.
M0 is complete through the supplied product and official-rule clarification.
M1 ends with a verified local commit containing only `SPEC.md`, `MILESTONES.md`, and `README.md`.
The local `main` assembly of the committed backend, console, and documentation refs is authorized and complete at the current root checkpoint below.
M2 through M4 are active implementation and evaluation gates across the worker branches.
M5 and M6 require future publication or submission authority appropriate to each action.
This documentation worker did not install dependencies, create application code, or edit implementation branches.
Historical local-main assembly records and dirty worker overlays remain excluded from current evidence.
The public GitHub repository is now visible and verified; public demo publication, deployment, pushes to other public remotes, pull requests, and submission remain incomplete or unauthorized.

The dependency chain is M0 -> M1 -> M2 -> M3 -> M4 -> M5 -> M6.
After M2 fixes the contracts, environment fixtures and console work can proceed independently under assigned owners.
The evaluator owner must remain independent of learner-controlled code and protect validation and final test contents.
No additional workers are spawned during this documentation task.

## Current root checkpoint

As of 2026-09-06, the verified public repository is [github.com/VasuBansal7576/adaptive-agent](https://github.com/VasuBansal7576/adaptive-agent), and the clean root-committed `main` checkpoint is `61e700b`.
This checkpoint includes the integrated backend and console evidence; the prior backup remains retained.

Current verification records 123 focused backend checks passed, 71 console tests passed, clean TypeScript and production-build checks, and a real local API browser pass at 375, 768, and 1440 pixel widths with no overflow or page errors.
These are scripted integration and browser checks, not a performance measurement.

Retained root evidence at `a882641` records the full synthetic 1,080-case run, restart coverage, and two authenticated reports passing in 48.25 seconds.
The combined production bytes at `2a54cf9` reproduced that same full synthetic pass.
Core and API verification recorded 64 passing checks, and the corrected privacy projection test passed separately.
An earlier retained backend report recorded 240 passing checks.
These are synthetic or bounded integration evidence, not a measured generalization result.

The real benchmark attempt on `2a54cf9` started at 2026-09-06 13:53 UTC and failed before the held-out phase after one successful task and five development budget exhaustions.
It produced no learning result, validation result, final metrics, or measured improvement.
M4 is not complete and must not be represented as passed.

The AppWorld work remains in worker 2, where the full runtime integration still has a serialization failure and is not accepted or merged.
Multi-run learning in worker 4 is still awaiting its fix and commit.

The strict provider output-token cap remains an accepted but unmet criterion.
The SDK nominal cost is a proxy, and economic billing is unknown.
Public repository visibility is authorized and verified.
Public demo posts, deployment, and submission remain unauthorized.

This checkpoint updates the current status only.
The historical evidence ledger, QA-data hold, delivery gates, and requirement traceability below remain preserved.

## Implementation and integration snapshot

The following snapshot was read from the worktrees on 2026-09-06 and is retained as dated provenance.
The refs identify observed states at that time, not the current `main` checkpoint.
Dirty worktrees contain additional uncommitted implementation changes, which are not counted as integrated evidence.

| AO session | Observed ref and state | Implemented component status |
| --- | --- | --- |
| adaptive-agent-2 | `f3785f0`, dirty | Current probe and full backend integration lineage for the control API, durable runtime, evaluation launch, evaluator observation, and API regressions; overlay remains uncommitted |
| adaptive-agent-3 | `6aabe2e`, clean apart from generated cache files | Prime runtime boundary and the two-case EVAL-005 runtime probe |
| adaptive-agent-4 | `83d2e01`, clean | Durable Controller, Store, Broker, evidence, and isolated EVAL-004/005 probes |
| adaptive-agent-5 | `ae482da` after `29afe35`, clean | React/Tailwind/Vite operator console, live REST/SSE transport, evaluation status, evidence provenance, recovery, and honest uncertainty states |
| adaptive-agent-6 | `c69a366`, dirty | Resumable benchmark and evaluator-boundary work; current dirty changes are not part of the local-main assembly plan |
| adaptive-agent-7 | `09aa9b9`, clean | Evidence-linked bounded learning, retrieval, durable learning runtime, and integrated runtime budget-boundary regression |

The implemented Python commands are `adaptive-agent` and `adaptive-agent-plan` as declared in the session-2 `pyproject.toml`.
The implemented console commands are `bun run dev`, `bun run test`, `bun run build`, `bunx tsc --noEmit`, and `bun run api:smoke` as declared in the session-5 package manifest.
The Prime bridge investigation also verified `prime-agent --version`, `prime-agent --help`, and the authenticated `prime-agent --print --no-tools --provider openai-codex --model openai-codex/gpt-5.6-luna` smoke command.

## Evidence ledger and current gate status

Real-run and fixture evidence remain separate.
The current checkpoint summary is above.
The following table preserves dated evidence records, including the prior root path and historical smoke reports.

| Evidence | Classification | What it establishes | What it does not establish |
| --- | --- | --- | --- |
| Prior root-verified production UI to authenticated Luna, Docker, broker write, and trusted evaluator on `/private/tmp/adaptive-agent-main-20260906`; run `run_a1662f700a9a400b9e0f80b87a52f76e`; candidate `cand_a095170f9c6d4f58943deb7539270de0` | Dated prior real product evidence | A clean Store and data path completed the real execution chain, and candidate proposal validation used three broker references and 7,206 learning tokens | Heldout panels, a performance result, a completed candidate comparison, or a promotion decision |
| Session-2 probe and integration lineage at `f3785f0` with a reviewed dirty overlay; 171 backend tests passed | Dated prior integration evidence | The control-plane and evaluator-launch integration was test-covered at the reported checkpoint | A committed merge, real provider product run, heldout evaluation, or final performance result |
| Actual Docker-backed EVAL-003 from the session-2 lineage: 7 obligations passed on isolated temporary Store and data paths | Dated prior boundary evidence | The seven obligations crossed the actual Docker runtime boundary while using an isolated fixture and no paid model call | General model safety, product acceptance, or sealed evaluator performance |
| EVAL-003 safety hardening at `3261afc` atop `f423fac`; focused safety, core, and lifecycle tests 39 passed | Dated prior safety evidence | Attacker-controlled public-document output, forged approval attempts, no-provider-write behavior, Docker-provenance requirements, fail-closed controller execution, and self-contained payload/content-hash receipts | A full pass without an injected Docker-backed Prime adapter, heldout evaluation, or paid-provider result |
| Actual Docker-backed EVAL-005 at session-3 ref `6aabe2e`: `child_failure_recovery_and_cap` and `shared_model_cost_token_cap` passed | Dated prior runtime-boundary evidence | Two parent-child, failure, and shared-cost runtime cases crossed the Docker boundary with bounded deterministic probe inputs | Paid-provider behavior, complete EVAL-005 acceptance, or final resource-budget evidence |
| Session-4 Controller probe at `83d2e01`: EVAL-004 8/8 and EVAL-005 8/8 on an isolated temporary Store and data directory | Dated prior control-boundary evidence | `promotion_crash_reopen_atomic`, `child_failure_propagated`, and `event_reconnect_resume`, including active-pointer preservation, persisted parent/child failure, and exact terminal SSE tail recovery | A sealed evaluation, cross-environment performance, or proof that historical root QAdata was isolated |
| Session-5 console lineage `29afe35` through clean tip `ae482da`: 61/61 tests, TypeScript clean, and build clean; latest report had no live smoke | Dated prior console evidence | Console parsing, evaluation status, learning eligibility, responsive master-detail behavior, and honest uncertainty states | Browser-width acceptance, live inference, or sealed evaluation |
| Isolated development smoke at `/tmp/adaptive-run-2fc3c680.fiZAAA`: durable run `run_7f5b3196fff541129d4e9c8345d0ba75`, `finance-development-00`, seed 17, sequence 16; scoped verifier `0561041`; full pytest 147 passed with two dependency warnings | Dated prior isolated development-smoke evidence | Trusted `evaluator_only` outcome with operator-visible `model_response`, resume completion, one candidate, one transfer run, pinned Docker execution, aggregate usage, and confirmed broker invoice/payment reads plus `apply_payment` | Heldout or final evaluation, performance, public reproducibility, or closure of M2-M4 |
| Prime 0.9.2 session 3, ChatGPT subscription, `openai-codex/gpt-5.6-luna`, actual Python `2+2` result `4` | Real runtime smoke | Subscription authentication and a narrow Python execution path | Product bridge, evaluator, safety, or performance acceptance |
| `prime-agent --print --no-tools --provider openai-codex --model openai-codex/gpt-5.6-luna` returned `MODEL_SPIKE_OK` in 10.24s | Real provider smoke | Model reachability through the authenticated path | A goal/tool/evaluator run or provider hard output cap |
| Prime adapter state-sharing cells returned `42` then `41`; host bridge returned structured data; forbidden learner requests were denied | Real Prime/Docker boundary smoke | Persistent kernel state, broker boundary, and denial behavior in the adapter spike | Complete control-plane learning and promotion evidence |
| Session-5 `bun run api:smoke` worker report: live FastAPI session, registration, create/launch, SSE cursor, and cancellation smoke PASS | Real control-plane transport with synthetic task | Live API/session/SSE/cancellation wiring | Real model inference or sealed evaluator evidence; the script's smoke environment is synthetic |
| Earlier session-4 SQLite and session-6 backend reports | Historical synthetic/local evidence | Earlier Store, Broker, Controller, learning, and benchmark contract checks | The current final2 integration, real provider behavior, or product acceptance |
| Devin 3000.6.14 session 4 and OpenCode session 5 each ran `pwd` successfully with no file changes | Real build-agent smoke | Those agent sessions could execute a basic command | Any product behavior |

The isolated development-smoke receipt used `/tmp/adaptive-run-2fc3c680.fiZAAA` and durable run `run_7f5b3196fff541129d4e9c8345d0ba75`.
It recorded `finance-development-00` baseline seed 17, `lastEventSequence=16`, bundle `507a47cc70856f5e680d689ad3400b92f67daa4dbc9bc1e1209b5ae78a1e7b6`, protocol `1cced7b2...`, core `54b3b7...`, Docker image `adaptive-prime-runtime@sha256:e1242afd...`, analysis `3e5371a...`, budget `5f3b9d...`, and model `efaeffc...`.
Aggregate usage was 8407 input, 1088 output, and 9495 total tokens.
The resume smoke completed with `completed=true`, `status=complete`, `candidateCount=1`, and `transferRuns=1` against expected 1.
The scoped `0561041` verifier accepted trusted outcome visibility as `evaluator_only` while requiring `model_response` visibility as operator.
Broker invoice and payment reads plus `apply_payment` were confirmed.
The full post-fix pytest report was 147 passed with two dependency warnings.
This receipt is isolated development-smoke evidence only, with no heldout, final, public, or performance claim.

The current EVAL-003 probe accepts optional runtime injection but fails closed without it.
The `require_runtime=False` option is control-plane unit mode only and cannot establish a full EVAL-003 result.
Its full runtime path must report Docker in adapter provenance and covers broker escalation, evaluator and harness write denials, and filesystem access denial.
The focused safety, core, and lifecycle verification was 39 passed, with no service startup or paid model call.

The session-4 probe closure did not modify the cost ledger, EVAL-003, or evaluator lifecycle.
It used no services or paid calls.

## Prior local-main assembly record

The earlier authorized local-main assembly record remains historical provenance for backend tip `feef752`, console tip `eea3bdc`, and documentation tip `eeb5f05`.
The backend tip contains the requested session-2 ancestry through `83d7e87`, `2168124`, `8fcb18a`, and `96a84d3`, plus `dbf412d` and `889a10f`.
The console tip contains the requested session-5 lineage through `29afe35` and `ae482da`.
Dirty worker overlays and uncommitted lock files were excluded from that assembly.

That historical assembly's public `main` checkpoint was `2a54cf9` and included the controller privacy fix.

Post-assembly verification must use newly allocated temporary Store and data paths and must preserve the reported 171 backend tests and 61/61 console result as prior evidence rather than replaying paid or shared-data smoke.
Cost accounting, the B0/L/A ablation, and the final report remain incomplete.
The assembled tree is not release-ready.

Session 3, session 4, session 6, and session 7 are not assembly inputs in this plan.
Their runtime and probe results remain separately attributed evidence.

## QA data-integrity hold

The original real evidence for `run_2fc` is on hold because synthetic `model_response` rows were appended at sequences 20 and 21.
The affected rows are `learning-model-fake-resp-1` and `learning-model-r20`.
Preserve the complete history, but do not erase, relabel, or count those rows as real evidence.

The repository search found no literal occurrence of `run_2fc`, `learning-model-fake-resp-1`, or `learning-model-r20`.
The source and ownership of those specific append operations therefore remain unverified and must not be inferred from the related probe code.
The known related evaluator-owned probe path is session 2's `Controller.execute_probe`, authored in commits `59adfb5` and `f423fac`.
Its static isolation proof is `tempfile.TemporaryDirectory(prefix="aa-probe-")` followed by `Store(Path(tmp) / "store")` for each probe execution.
That proof establishes the intended probe boundary only, not the provenance or isolation of `run_2fc`.

Do not run transport smoke against root QAdata or port 8000 during this hold.
The real authenticated path invokes the paid model.
All tests and probes must use a newly allocated temporary Store and data directory, with the resolved paths recorded in the verification output.
Existing test code that uses temporary directories is not, by itself, proof that a shared QA run was untouched.

Session 6 owns the next admissible evaluation step.
It must create clean independent evaluation data, regenerate and validate real DEVELOPMENT evidence, and regenerate and validate the candidate from that clean source.
Only after those checks and exact path-level isolation proof pass may any sealed panel be started.
Until then, no contaminated run, appended synthetic row, historical root-data smoke, or unproven candidate may be used for a sealed result.
The historical evidence entries above remain preserved as reports, but any entry without exact Store and data-target proof is excluded from clean QA evidence.

The current gate status is deliberately conservative.
M0 and M1 are complete as documentation and local integration milestones.
M2 now has a root-verified production UI to authenticated model, Docker, broker, and trusted-evaluator path, but remains unmet until the required candidate comparison, gate decision, and boundary evidence are retained together.
M3 has core learning, storage, broker, candidate, and benchmark components, but the complete three-pack lifecycle and all required acceptance scenarios are not verified as one integrated product.
M4 is unmet because there is no clean sealed four-environment B0/L/A evaluation, measured cross-domain result, or final resource-budget evidence.
M5 and M6 remain incomplete.
The public repository is visible and verified, but there are no public social-demo, deployment, or submission claims.
Cost, ablation, and final-report gates remain incomplete, so this assembled tree is not release-ready.

One user-accepted subscription limitation is still an UNMET criterion.
The accepted Luna-through-ChatGPT subscription path has no API-key or provider switch.
The trusted parent records actual model usage and rejects after aggregate token exhaustion.
Local token limits and post-response rejection are not hard provider enforcement, so the strict per-call output-token-cap criterion remains UNMET.
This remains a blocker for the resource-boundary acceptance work and must not be recorded as a pass.
This limitation is recorded in the session-2 docs commit `356085e`.
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

The public GitHub repository [github.com/VasuBansal7576/adaptive-agent](https://github.com/VasuBansal7576/adaptive-agent) is visible and verified.
Prepare a public X or LinkedIn demo post and the corresponding links for Devpost only after explicit publication authority.
Prepare a deployment URL only if deployment is chosen and authorized.
Public GitHub access was checked without relying on a logged-in session.
Redact credentials, private data, sealed test answers, and unnecessary transcripts from all public artifacts.
Keep authoritative evaluator fixtures or expected answers private when exposing them would undermine continued evaluation.
Provide public reproduction instructions and non-sealed sample fixtures instead.

The demo target is 3 minutes within the official 3 to 5 minute range.
The prepared sequence is 0:00 to 0:20 for the problem and generic environment contract, 0:20 to 0:50 for the verified real run, and 0:50 to 1:30 for evidence-driven refinement.
Show candidate validation honestly as `[PENDING: learning/validation result and promotion gate]` because the current benchmark ended before held-out evaluation.
Use 1:30 to 2:10 for the three-domain and new-environment comparison, labeled `[PENDING: EVAL-001/EVAL-002/EVAL-006 accuracy, reliability, cost, speed, uncertainty, and exposure metrics]`.
Use 2:10 to 2:30 for bad-refinement rejection and safe rollback, labeled `[PENDING: final integrated rejection/rollback evidence if not retained]`.
Use 2:30 to 3:00 for AO session count, task ownership, contribution history, Prime Agent credit, and limitations, labeled `[PENDING: final AO history and submission record]`.
Credit Prime Agent by Prime Intellect AI under the MIT License and point to [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).
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
