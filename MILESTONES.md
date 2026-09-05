# Adaptive Agent milestones

Status: implementable delivery plan for the confirmed Track 1 product, with only the documentation phase currently authorized.
Product acceptance and competition compliance remain unverified until implementation and submission evidence exist.
[SPEC.md](SPEC.md) defines stable requirements, contracts, acceptance scenarios, and evaluation cases.

## Authorization and sequencing

M0 and M1 are the documentation phase.
M0 is complete through the supplied product and official-rule clarification.
M1 ends with a verified local commit containing only `SPEC.md`, `MILESTONES.md`, and `README.md`.
The local `main` fast-forward and filesystem-only origin synchronization for the documentation commits are authorized and complete.
M2 through M6 require future implementation, publication, or submission authority appropriate to each action.
The current worker does not install dependencies, create application code, publish, push to public remotes, open pull requests, or submit.
It has completed local integration into `main` and synchronization to the filesystem-only origin.

The dependency chain is M0 -> M1 -> M2 -> M3 -> M4 -> M5 -> M6.
After M2 fixes the contracts, environment fixtures and console work can proceed independently under assigned owners.
The evaluator owner must remain independent of learner-controlled code and protect validation and final test contents.
No additional workers are spawned during this documentation task.

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
Stage only the three named documents and inspect the staged diff before making a conventional local commit without an agent co-author.
Report the commit, changed files, checks, and remaining execution assumptions through AO.
The handoff includes completed local specification commits and local `main` integration, not completed implementation, public publication, or submission.
