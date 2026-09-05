# Adaptive Agent specification

Status: implementable specification with product scope and local commit authorization confirmed on 2026-09-06.
Numerical defaults are proposed engineering targets, not measured results.
This is an implementation design for Syndicate Track 1 only.
No application has been implemented, evaluated, published, or submitted.

## Purpose and authorization

Adaptive Agent learns how to solve unfamiliar tasks from its own attempts and independently verified outcomes.
A new environment supplies goals, documentation, tool contracts, access policy, and an outcome evaluator.
The same core discovers a solution, records evidence, proposes reusable procedural skills, and tests whether those skills improve unseen tasks.
Finance, customer support, and IT are evaluation environments, not separate products or handwritten workflows.
Results on these environments cannot establish success in arbitrary domains.

The authorized work is limited to `SPEC.md`, `MILESTONES.md`, and `README.md`, their verification, and a local specification commit under the confirmed documentation authorization.
The initial repository is empty at `a09b012` and has a local filesystem origin.
Implementation, dependency installation, remote creation, publication, pushes, pull requests, merging to `main`, and submission are outside this phase.
Later implementation and submission gates are specified so the documentation can support the full objective.

## Sources and evidence limits

| ID | Source | Status and authority |
| --- | --- | --- |
| SRC-001 | Authorized documentation task | Confirmed phase boundary and document acceptance |
| SRC-002 | User working preferences | Confirmed engineering and stack preferences |
| SRC-003 | [Syndicate event page](https://luma.com/d0kq45ek) | Read on 2026-09-06; primary source for the Track 1 and AO obligations below |
| SRC-004 | Product clarification relayed by the orchestrator on 2026-09-06 | Confirmed product intent and learning constraints |
| SRC-005 | Prime Agent source review relayed by the orchestrator | Supplied findings; exact repository revision and independent source verification pending |
| SRC-006 | [Backpass](https://github.com/kunchenguid/backpass) review relayed by the orchestrator | Supplied influence assessment; not a required dependency |
| SRC-007 | [Vision](https://github.com/kunchenguid/vision) review relayed by the orchestrator | Supplied influence assessment; not a required dependency |
| SRC-008 | [Official Devpost rules](https://syndicate-by-maximor.devpost.com/rules) and [official Syndicate brief](https://maaztwts.notion.site/Syndicate-3cc32902e4a38075bfa9f03149ef150d) | Constraints verified by the user and orchestrator and supplied on 2026-09-06; direct fetch in this worker was unavailable |

SRC-003 calls for architecture creation, execution, diagnosis, and iterative improvement on unfamiliar tasks.
It asks for evidence across domains covering accuracy, reliability, cost, and speed.
AO must be used during development through submission, with its use and sessions visible in the final demonstration.
SRC-008 establishes one-track participation, registration and Discord membership for every team member, a public GitHub repository, and a public X or LinkedIn demo post linked in Devpost.
The official demo length is 3 to 5 minutes, with a project target of 3 minutes.
A deployed URL is required only if the project is deployed.
The build window is 2026-09-05 21:30 IST through 2026-09-07 03:30 IST, equivalent to 2026-09-05 16:00 UTC through 2026-09-06 22:00 UTC.
The rubric weights AO usage 25%, execution 25%, track fit 25%, demo 15%, and innovation 10%.
The demo must show meaningful AO usage and the dashboard session count.
Neatlogs, TensorMux, AIGI, and Dodo are optional sponsors or integrations.
Organizer feedback tentatively accepts synthetic scenarios, but that is not a blanket rule or confirmed eligibility for every fixture design.
These are documented obligations, not a claim that this project has met them.
Publication and submission remain future actions requiring authorization.

The Prime review identifies a persistent IPython workspace, optional independent RLM child sessions, editable memories, prompts, Python skills, and refine proposals with rollback.
Its `expectedOutcome` is a prediction, not measured performance.
Structural validation, conflict handling, audit, and undo exist, but the inspected apply path did not establish a held-out performance gate.
Direct harness create, update, and delete operations must pass through this project's promotion boundary too.
The small JSON harness store is not a semantic knowledge base.

Backpass contributes transcript evidence, bounded memory edits, and staged proposals, but its default human approval is not an autonomous outcome evaluator.
Vision mines real project history, which this initially empty repository does not have.
Surya's painting work trained model weights on the Prime Intellect platform and does not demonstrate that Agent refine performs reinforcement learning.
This project's adaptation changes versioned procedures and bounded configuration, not model weights.
Any Prime reuse must retain its MIT license and notices after the pinned source license is verified.
The future README and demo must credit reused code and distinguish these influences from the independent learning and promotion loop built here.

## Decisions and defaults

The following defaults make the design implementable without turning provisional choices into track requirements.
The implementation owner can change a default through a reviewed specification update before collecting evaluation evidence.
Requirement IDs retain their meaning when details change.

| Decision | Resolution or remaining question | Effect |
| --- | --- | --- |
| DEC-001 | Track 1 and official submission constraints supplied and confirmed in SRC-008 | Recheck official changes before submission |
| DEC-002 | One domain-agnostic agent that learns procedures from verified attempts | No domain-specific planner branches or manual prompt tuning |
| DEC-003 | Versioned procedural skills and bounded execution configuration, evaluated outside the learner | No model-weight training or unmeasured automatic activation |
| DEC-004 | Start with isolated test environments; concrete adapters, providers, and data rights remain to be selected | Live external effects require separate access and policy decisions |
| DEC-005 | Paired held-out evaluation with the proposed thresholds below | Protocol must be frozen before candidate selection |
| DEC-006 | Build window, rubric, public artifacts, membership, demo length, and conditional deployment URL are confirmed | Exact submission form fields, team completion, and synthetic-fixture acceptability need release-time verification |
| DEC-007 | Bounded Prime API bridge spike selects the small backend; Python with SQLite is the default | Model, provider, resource cap, and Prime source revision must be pinned before execution |
| DEC-008 | Tool broker enforces permissions; explicit approval for policy-designated actions | Learned content cannot authorize actions |
| DEC-009 | Documentation completion and local commit are authorized; orchestrator coordinates later integration and execution | No further product-scope confirmation is needed for this commit |

## Stable requirements

Use `DOC-nnn` for documentation acceptance, `TRK-nnn` for sourced event obligations, and `PRD-nnn` for product behavior.
Acceptance scenarios use `ACC-nnn`; evaluation cases use `EVAL-nnn`.
Retired identifiers remain recorded and are never reassigned.
The requirement-to-milestone mapping is in [MILESTONES.md](MILESTONES.md).

| Requirement | Source | Acceptance criterion |
| --- | --- | --- |
| DOC-001 | SRC-001 | The local specification commit includes only the three requested Markdown files |
| DOC-002 | SRC-001 | Scope and track obligations have references, with unknowns and evidence limits explicit |
| DOC-003 | SRC-001 | Every active product and track requirement maps to observable acceptance and a milestone |
| DOC-004 | SRC-001 | The primary workflow and failures resolve across architecture, data, and contract boundaries |
| DOC-005 | SRC-001 | The adaptation rule, baseline, independent evaluation, promotion, and rollback are implementable |
| DOC-006 | SRC-001 | Security, reliability, dependencies, ownership, and submission gates are specified |
| DOC-007 | SRC-001 | Links and requirement references resolve and documents agree on scope and authorization |
| DOC-008 | SRC-001 | Final scope coordination precedes the local commit, whose metadata contains no agent co-author |
| TRK-001 | SRC-003 | A supplied unfamiliar goal and tools produce an executable agent configuration and run, demonstrated by ACC-001 |
| TRK-002 | SRC-003 | Failed attempts lead to a documented change and a measured independent comparison, demonstrated by ACC-004 and EVAL-001 |
| TRK-003 | SRC-003 | The demonstration reports cross-domain accuracy, reliability, cost, and speed with evidence, covered by ACC-008 |
| TRK-004 | SRC-003, SRC-008 | Meaningful AO use, dashboard session count, and development evidence are retained through demo and submission, checked by ACC-012 |
| TRK-005 | SRC-008 | Only Track 1 is entered and each member completes registration and Discord membership, checked by ACC-014 |
| TRK-006 | SRC-008 | Authorized submission links a public GitHub repository and a public X or LinkedIn demo post in Devpost, checked by ACC-014 |
| TRK-007 | SRC-008 | Demo is 3 to 5 minutes, supplies a deployment URL only if deployed, and includes evidence for the five rubric categories, checked by ACC-014 |
| TRK-008 | SRC-008 | Build and submission planning respects the stated IST window and retains actual submission timing, checked by ACC-014 |
| PRD-001 | SRC-004 | New environments require only tools, docs, policy, tasks, and evaluator, with an unchanged core planner, tested by ACC-001 |
| PRD-002 | SRC-004 | The executor discovers its action sequence within a persistent task workspace and bounded optional children, tested by ACC-002 |
| PRD-003 | SRC-004 | Knowledge, live tool evidence, task state, and reusable skills have distinct storage and authority, tested by ACC-003 |
| PRD-004 | SRC-004 | Every candidate edit cites verified attempt evidence and has a bounded immutable diff, tested by ACC-004 |
| PRD-005 | SRC-004 | Independent paired held-out results and required safety checks gate every activation path, tested by ACC-005 |
| PRD-006 | SRC-004 | Evaluators, policies, credentials, and hidden answers cannot be modified or retrieved by the learner, tested by ACC-006 |
| PRD-007 | SRC-004 | Candidates can be rejected, quarantined, superseded, and rolled back with audit history, tested by ACC-007 |
| PRD-008 | SRC-004 | Evaluation separates development, promotion, and final audit tasks and limits generalization claims, tested by ACC-008 |
| PRD-009 | SRC-004 | Tools validate permissions, arguments, side effects, and approvals outside the kernel, tested by ACC-009 |
| PRD-010 | SRC-002, SRC-004 | Operators can inspect run evidence and learning decisions, cancel runs, and recover failures in an accessible console, tested by ACC-010 |
| PRD-011 | SRC-004 | Resource limits, duplicate protection, crash recovery, and version pinning survive process failure, tested by ACC-011 |
| PRD-012 | SRC-004, SRC-005 | Prime integration and attribution preserve source limits and route direct harness writes through promotion, tested by ACC-005 and ACC-013 |
| PRD-013 | SRC-004 | Fixed baseline, learned agent, and memory-disabled ablation are compared on a sealed final test including a genuinely new environment, tested by ACC-015 |

## Primary operator workflow

An operator registers an environment package and chooses a goal, a model profile, and a resource budget.
Validation rejects missing tool schemas, policy, or trusted evaluator registration before a run starts.
The agent reads relevant documentation, inspects available tools, and proposes an execution configuration.
The configuration selects tool subsets, existing compatible skills, workspace limits, and whether bounded child tasks are useful.
It contains no environment-authored action sequence.
The executor works toward the goal and emits tool calls through the broker.
The operator sees live status, evidence references, tool outcomes, and budget use without private reasoning being required.

After a development attempt, the trusted evaluator records the observed outcome and a permitted diagnostic summary.
The learner uses that evidence to propose a bounded skill or configuration edit.
The controller validates the proposal and runs a baseline and candidate in independent, reset environments.
A failed gate preserves the existing active version and records the reason.
A passed gate atomically activates the candidate for future runs.
An already running task remains pinned to its starting version.
The operator can inspect the comparison and request rollback, which is also audited.

## Architecture

The implementation starts with a bounded Prime API bridge spike, limited to one working session with a 90-minute engineering timebox.
The spike must create a persistent kernel, execute two state-sharing calls, obtain structured outputs, cancel execution, and intercept refine and direct CRUD before activation.
It also checks child isolation, external tool routing, artifact export, version pinning, and the available authentication path.
Prime 0.9.2 is installed and AO reaches its login screen, but user login and real inference remain unverified as of the supplied setup status.
The installed version does not prove any required API is available.
The spike records the pinned source revision, MIT notices, verified callable interfaces, and unsupported operations without inventing Prime endpoints.

The default is a small Python control plane with an ASGI API and SQLite because Prime execution and learned Python skills fit that runtime.
If Prime offers a stable API or CLI with structured I/O, a thin adapter uses that interface.
If Prime cannot expose the required interception and isolation, the fallback is an isolated IPython executor behind the same adapter contract, with a bounded independent-child implementation only where needed.
The fallback must still meet every product requirement and disclose that the Prime integration was not used.
An authentication delay does not justify reporting simulated output as working inference.
Avoid separate microservices when modules in one trusted backend suffice.
A React, Tailwind, and Vite console reads the control API and event stream.
The initial deployment is a single authenticated operator on a loopback-only service.
A server-issued operator token, strict Origin/Host checks, and same-origin requests protect control mutations from unrelated local web pages.
The learner never receives the operator token, and the console renders untrusted evidence as escaped text rather than executable HTML.
Non-loopback hosting requires an explicit authentication, tenant isolation, and deployment design before release.
SQLite transactions store run metadata, immutable version references, and activation history.
Content-addressed files store redacted evidence, documentation snapshots, and candidate bundles.
Convex is deferred because the first design needs local Python execution and transactional version control rather than shared cloud application state.
A later shared operator product can revisit that choice without changing the execution or promotion contracts.
Bun is the proposed frontend package manager in this unconfigured repository, with PNPM as fallback.
No dependencies are installed in this phase.

| Component | Owns | Excluded authority |
| --- | --- | --- |
| Control API and coordinator | Validation, run lifecycle, task version pins, cancellation, event cursors | Does not accept learner-supplied claims as evaluator results |
| Prime execution adapter | Persistent per-task IPython workspace and optional bounded RLM children | Cannot write active skills, policy, evaluator state, or host files |
| Tool broker | Schema checks, credentials, capability checks, approvals, idempotency, authoritative tool envelopes | Cannot widen permissions based on docs, skills, or tool output |
| Evidence store and retrieval | Immutable provenance, document indexing, redaction, access-filtered retrieval | Never indexes hidden tests, answers, credentials, or promotion traces for the learner |
| Candidate builder | Evidence-linked bounded patches against a pinned base version | Cannot directly activate harness CRUD or edit trusted components |
| Evaluation controller | Partition allocation, reset fixtures, paired runs, scoring, sealed reports | Does not expose hidden outcomes or traces to the learner |
| Promotion service | Mechanical gate, compare-and-swap activation, quarantine, rollback | Cannot accept `expectedOutcome` or a human bypass as evidence of improvement |
| Operator console | Run progress, document evidence, candidate diffs, comparisons, approval and rollback controls | Cannot manufacture a passing evaluation |

The control plane, broker, evaluator, and promotion service run outside the learner sandbox.
A process boundary alone is insufficient if the learner can access the same filesystem, network, or credentials.
Use an execution container or equivalent OS sandbox with a writable task directory, read-only approved skill snapshot, and no ambient host mounts or credentials.
Restrict egress to the broker and approved model proxy.
The broker validates an expiring run capability and serves only that run's allowlisted operations.
Kernel-created subprocesses inherit the same restrictions.
If the selected Prime path cannot satisfy these controls, M2 fails until its adapter is changed or a compatible execution boundary is selected.

### Data flow and authority

1. The coordinator validates an environment package and stores a versioned manifest.
2. The coordinator creates a run pinned to environment, policy, model, budget, and active skill versions.
3. Retrieval exposes only permitted docs, development evidence, and active compatible procedures to that run.
4. The learner requests tool operations and receives broker-signed or broker-recorded result envelopes.
5. The evidence store appends structured actions and results, with hashes and provenance.
6. The external evaluator inspects the isolated outcome and produces a trusted result.
7. For development runs only, the learner receives a sanitized result and may create a candidate proposal.
8. The evaluation controller allocates unseen tasks and compares the pinned base and immutable candidate without online updates.
9. The promotion service checks the report, safety results, and current base before atomically changing the active pointer.
10. Future runs load the promoted version, while rejected or quarantined proposals remain inactive.

Retrieved documentation is evidence about an environment, not policy authority.
Live broker results describe observed system state and take precedence over stale documentation for state claims.
A tool result still cannot grant permissions or issue instructions to the control plane.
Persistent IPython variables are working state for one task and are not reusable knowledge by default.
Procedural skills are versioned strategies with applicability, preconditions, evidence, and failure handling.
A skill's stated preconditions never replace a fresh policy check or live-state verification before an external action.

## Boundary contracts

All external inputs use schema validation and reject unknown privileged fields.
Identifiers are opaque strings; timestamps are UTC; immutable content references use SHA-256.
Untrusted strings never become filesystem paths, executable configuration, or capabilities without validation.
The following are proposed contracts, not files or APIs already present in the repository.

| Contract | Required fields and constraints |
| --- | --- |
| `EnvironmentManifest` | `schemaVersion`, `environmentId`, `version`, `docs[]` with hashes and classifications, `toolSchemas[]`, `policyRef`, trusted `evaluatorRef`, `resetRef`, and capability metadata; no planner patch or action sequence |
| `TaskInput` | `taskId`, `environmentRef`, natural-language `goal`, allowed input references, and evaluation partition assigned by the controller; hidden answers are absent |
| `RunRequest` | Task reference, `modelProfileRef`, `budgetRef`, and idempotency key; active version is pinned by the server, not chosen by untrusted text |
| `ExecutionConfig` | Allowed tool subset, compatible skill references, instruction variant, step limit, and child count/depth limits; all values bounded by server policy |
| `ToolCall` | `runId`, `callId`, tool name, schema-valid arguments, and optional approval token; capability supplied out of band by the adapter |
| `ToolResult` | Call reference, tool/schema version, observed time, success/error union, redacted output reference, side-effect status, and broker provenance |
| `EvidenceRecord` | Run reference, sequence, event type, content hash, source reference, trust class, visibility class, and redaction status; append-only |
| `SkillVersion` | Skill ID, immutable version and parent, applicability predicates, preconditions, procedure text or Python source, expected tool contracts, failure handling, evidence references, and content hash |
| `CandidateProposal` | Candidate ID, base bundle hash, allowed edit operations, changed artifact hashes, supporting development evidence IDs, predicted effect, and proposer version |
| `EvaluationReport` | Candidate/base hashes, protocol hash, partition reference, paired run IDs, metric aggregates, uncertainty, safety results, validity status, and evaluator provenance |
| `PromotionDecision` | Candidate/base hashes, trusted report reference, gate version, decision/reason, prior/new active pointer, and transaction timestamp |

Visibility classes are `learner`, `operator`, and `evaluator_only`.
The retrieval service filters visibility and environment access before ranking, including cached queries and child-session requests.
Development evidence can become learner-visible only after redaction.
The initial retrieval implementation uses SQLite full-text search with metadata filters rather than assuming Prime's JSON store provides semantic search.
Document chunks carry source URI, document version, content hash, chunk offset, environment scope, and ingestion time.
Skill retrieval additionally checks tool-schema compatibility and applicability metadata.
Return at most eight chunks within a proposed 6,000-token retrieval budget, with each claim traceable to its chunk reference.
A missing match yields an explicit empty result, not invented evidence.
Cache keys include access scope, source versions, and active bundle hash; deletion or version change invalidates affected entries.
A semantic index can be added only if measured retrieval failures justify it, without changing provenance or access filters.
Promotion and audit traces, expected answers, and scorer internals remain `evaluator_only` and are absent from skill evidence exports.
Operator reports show aggregate metrics by default and keep hidden task details out of learner-accessible transcripts.

The control API accepts environment registration, run creation, cancellation, candidate proposal, evaluation requests, and rollback requests.
Reads return run status, evidence references, candidate diffs, evaluation summaries, and active version history.
Run events use monotonically increasing sequence numbers and resume from the last acknowledged cursor.
Validation failures return `INVALID_INPUT`; forbidden requests return `FORBIDDEN`; stale base versions return `VERSION_CONFLICT`.
Exhausted budgets return `BUDGET_EXHAUSTED`; unavailable tools return `TOOL_UNAVAILABLE`; uncertain side effects return `OUTCOME_UNKNOWN`.
These errors include a correlation ID and safe recovery guidance, not credentials or hidden evaluator data.
Retries reuse idempotency keys and cannot create duplicate runs, evaluations, promotions, or external actions.

### Example learning contract

A development task fails because a record changed between inspection and update.
A trusted tool result records a version conflict, and the evaluator confirms that the intended outcome was not achieved.
The learner proposes a skill that rechecks the resource version before requesting an update and handles a conflict by reading the current state again.
The proposal cites those development evidence IDs and predicts fewer stale updates.
That prediction is not a passing score.
Independent tasks with different records and interference timings test the candidate against the existing version.
The broker still controls which records can be updated and whether approval is required.
This is an illustrative learned procedure, not an action sequence shipped in any environment package.

### Proposed manifest and runtime types

The JSON-serializable public contracts use the following structural types.
The trusted registry resolves policy, evaluator, and reset references before passing a sanitized manifest into the kernel.
`JsonValue` means null, boolean, finite number, string, an array of JSON values, or an object of JSON values.
Tool input and output schemas use a pinned JSON Schema dialect selected during the bridge spike.

```typescript
type JsonValue = null | boolean | number | string | JsonValue[] | { [key: string]: JsonValue };
type ArtifactRef = { id: string; version: string; sha256: string };
type EnvironmentManifest = {
	schemaVersion: 1;
	environmentId: string;
	version: string;
	docs: Array<ArtifactRef & { classification: "learner" | "operator" }>;
	toolSchemas: Array<{
		name: string;
		version: string;
		inputSchema: JsonValue;
		outputSchema: JsonValue;
		effect: "read" | "write";
	}>;
	policyRef: ArtifactRef;
	evaluatorRef: ArtifactRef;
	resetRef: ArtifactRef;
	capabilities: string[];
};
type ToolRequest = {
	runId: string;
	stepId: string;
	callId: string;
	tool: string;
	arguments: Record<string, JsonValue>;
	idempotencyKey: string;
	approvalToken?: string;
};
type ToolError = {
	code: "INVALID_INPUT" | "FORBIDDEN" | "VERSION_CONFLICT" |
		"BUDGET_EXHAUSTED" | "TOOL_UNAVAILABLE" | "OUTCOME_UNKNOWN";
	message: string;
	correlationId: string;
	retry: "never" | "safe_read" | "after_reconciliation";
};
type ToolResult = {
	callId: string;
	toolVersion: string;
	observedAt: string;
	brokerEvidenceRef: ArtifactRef;
} & (
	| { status: "ok"; output: JsonValue; effect: "none" | "confirmed" }
	| { status: "error"; error: ToolError; effect: "none" | "unknown" }
);
type RunRecord = {
	runId: string;
	taskRef: ArtifactRef;
	environmentRef: ArtifactRef;
	policyRef: ArtifactRef;
	modelProfileRef: ArtifactRef;
	skillBundleRef: ArtifactRef;
	budgetRef: ArtifactRef;
	status: "queued" | "running" | "awaiting_approval" |
		"succeeded" | "failed" | "cancelled" | "timed_out";
	lastEventSequence: number;
	outcomeRef?: ArtifactRef;
};
type StepRecord = {
	stepId: string;
	runId: string;
	sequence: number;
	kind: "retrieve" | "execute" | "tool" | "child" | "evaluate";
	status: "planned" | "running" | "awaiting_approval" |
		"succeeded" | "failed" | "cancelled" | "outcome_unknown";
	inputRefs: ArtifactRef[];
	outputRefs: ArtifactRef[];
	callId?: string;
	error?: ToolError;
};
```

The step state advances from `planned` to `running` or `awaiting_approval`, then to a terminal step status.
A tool timeout can enter `outcome_unknown`; only broker reconciliation can resolve it to success or failure.
The next state-changing step is blocked while a prior effect remains unknown.
Run success requires all required outcome checks and no unresolved step effects.
Candidate and evaluation records add the immutable hashes and state fields listed above, with evaluation states `queued`, `running`, `valid`, `invalid`, or `cancelled`.
A metric failure can be a valid evaluation report but leads to a rejected candidate.
Only the trusted controller can mark an evaluation valid.

### Durable storage and recovery

SQLite stores environments, runs, steps, tool operations, evidence references, candidates, evaluations, bundle versions, active pointers, and promotion decisions.
Foreign keys bind each run, report, and decision to existing immutable versions.
Unique constraints cover run idempotency keys, `(runId, sequence)` events, `(runId, callId)` operations, candidate content hashes, and promotion decision IDs.
The same idempotency key with different canonical input is a conflict, never a second operation.
Artifact files are written to a temporary name, hashed, and atomically renamed before the transaction references them.
Unreferenced files can be garbage-collected after recovery; referenced missing files invalidate evaluation and block promotion.

The coordinator records a tool operation as prepared before dispatch and appends its result durably after return.
On restart, prepared operations without a result are reconciled against the provider's idempotency status or marked `OUTCOME_UNKNOWN`.
A lease with an expiry and monotonically increasing fencing token prevents two coordinators from dispatching the same claimed step.
The broker rejects stale fencing tokens.
Evaluation reports become immutable only after every expected pair and artifact is present.
A report's validity, candidate transition, promotion decision, and active-pointer update use one transaction where they must succeed together.
An interrupted evaluation cannot activate a candidate and can resume only from validated complete pairs or restart with clean fixtures.

A generic environment package might declare `inventory.read` and `inventory.update`, documentation references, and trusted fixture reset and outcome evaluator references.
It supplies no order in which to call those tools, no example answer, and no domain-named branch in the planner.
Operator registration binds those tools to broker implementations and validates every declared output against its schema.
Schema-incompatible tool upgrades invalidate affected skill compatibility and require fresh evaluation.

## Learning and promotion algorithm

The active bundle contains approved procedural skills and bounded prompt or orchestration configuration.
Tool schemas, evaluators, policy, the promotion algorithm, resource ceilings, and broker code are outside the editable bundle.
Derived tool wrappers may be proposed as Python skills, but can call only broker-authorized tools.
No core planner edits or manual prompt tuning are allowed between evaluation environments.

The initial active bundle contains only generic execution and safety instructions, with no finance, support, or IT solution procedure.
The candidate builder consumes completed development attempts with trusted evaluator results.
A success can teach a reusable procedure; a failure can motivate a corrective proposal.
Unsupported hypotheses remain draft proposals and cannot reach evaluation without linked evidence.

Proposed initial bounds are one candidate in evaluation at a time, at most three changed artifacts per proposal, and at most 200 changed logical lines.
Each proposal must declare its intended applicability and affected tool contracts.
These bounds are project defaults, subject to review before the protocol is frozen.
They limit the size of a change under test without restricting the eventual skill library to one domain.

The candidate state machine is `draft -> validated -> evaluating -> promoted` or `rejected`.
Invalid or suspicious proposals enter `quarantined`; stale-base proposals enter `superseded`.
A promoted version can later enter `rolled_back` while remaining in history.
Rejected proposals need a new immutable candidate ID for revision and new evaluation allocation.

Validation checks evidence provenance, allowed paths and edit types, size bounds, schema compatibility, import restrictions, secrets, and attempts to alter trusted controls.
A valid proposal is materialized as a read-only candidate bundle in a disposable evaluation sandbox.
Baseline and candidate runs use separate kernel state, storage, fixtures, and task outputs.
Neither arm updates its skills during the comparison.
The controller verifies matching model, task seeds, environment state, and resource budgets before accepting a report.

The promotion service accepts only reports created by the trusted evaluator for the exact candidate and base hashes.
It evaluates the frozen metric gate and required safety suite mechanically.
Activation uses a transaction that checks the current active hash still equals the evaluated base hash.
A changed base causes `VERSION_CONFLICT` and supersedes the proposal instead of silently rebasing it.
Every Prime refine/apply path and direct harness create, update, or delete path becomes a proposal submission or is denied.
The learner never holds a writable active harness store.

Rollback atomically selects a previously approved immutable version and records the reason and affected runs.
New runs use that version immediately.
In-flight runs stay pinned unless the rollback reflects a safety problem, in which case the coordinator cancels affected runs and revokes broker capabilities.
Unknown external effects require reconciliation before resuming work.
Rollback restores agent artifacts, not external systems already changed by tools.

## Independent evaluation

The evaluation unit is a task instance with a reproducible initial environment state and a trusted outcome check.
Environment authors provide tool behavior, policy, tasks, and outcome predicates, but no demonstration solutions or planner changes.
Evaluators prefer objective fixture assertions and authoritative tool state over model opinions.
If a subjective judge is required, its version, rubric, variance, and limits are recorded separately and cannot override hard safety checks.

Training tasks supply feedback for learning and are also called development tasks in the runtime contracts.
Validation tasks test candidate changes for promotion without exposing their expected answers or traces to the learner.
Final test tasks remain sealed until the candidate and protocol are frozen for the submission evidence run.
Promotion tasks and final audit tasks mean validation tasks and sealed final test tasks respectively.
Splits separate task templates and entities, not merely random identifiers in the same solution.
The controller records partition hashes before candidate construction.
A promotion task allocation is consumed after use, even when a candidate is rejected.
The learner receives a coarse promotion status, not hidden failures or per-task scores.
Repeated accept/reject signals can still reveal information, so candidate attempts and holdout allocations are capped and logged.
The proposed limit is three candidate evaluations per training cycle, each with a disjoint validation allocation assigned before the cycle starts.
An exhausted pool stops promotion until the evaluator owner supplies a fresh isolated pool.

Use finance, customer support, and IT fixtures with distinct tool contracts, policies, and failure modes.
Within each environment, hold out task families from learning rather than replaying development examples.
For domain transfer, reserve one environment from candidate learning, rotate the reserved environment across independent experiments, and reset the active bundle each time.
Also measure adaptation after exposing a new environment's development tasks, followed by its untouched query tasks.
Report these as separate transfer and within-environment adaptation results.
Neither proves arbitrary-domain competence.
In addition to the three known evaluation packs, the evaluator owner supplies a genuinely new fourth environment for the sealed final test.
A proposed fourth pack is laboratory sample scheduling with unfamiliar resource and chain-of-custody tools, but its exact schemas, policies, and task families stay hidden from learner development.
It must differ in tool contracts and task structure, not merely rename finance or support entities.
The evaluator owner verifies this distinction and records prior exposure.
No planner edits, manual prompt tuning, or fourth-environment training are allowed before the primary final transfer measurement.
An optional later adaptation measurement uses a separate fourth-environment support set and untouched query set, clearly separated from the primary result.

Final evidence compares three frozen arms: the original fixed baseline B0, the learned agent L, and a memory-disabled ablation A.
B0 retains the initial generic instructions and has no acquired procedural skills or adaptive updates.
L uses the final promoted bundle with learning disabled during scoring.
A uses L's execution configuration but removes acquired procedural memory, retrieved learned skills, and copies of learned procedures embedded in prompts.
A retains environment docs, live tools, and per-task working state so the ablation measures reusable learning rather than basic task access.
The report records any learned configuration that remains in A and limits causal claims accordingly.
Promotion comparisons use the current active parent Bp against the candidate, while final reporting compares B0, L, and A.
All arms use matching models, seeds, budgets, fixtures, and isolated state.
The final test is not reused to select a new candidate after scores are observed.

### Metrics and proposed gate

Accuracy is the fraction of tasks whose trusted outcome checks all pass.
Reliability is the fraction that meet the task outcome and resource bounds under declared disturbances such as retries, stale state, or transient tool errors.
Safety violations are separately counted policy or isolation failures, even if the task succeeds.
Cost includes model and tool charges across the parent and children, with token counts when pricing is unavailable.
Speed is end-to-end task time, reported as median and p95 with timeouts included as failures.
Report development, candidate evaluation, and online execution costs separately so adaptation overhead is visible.

The proposed initial validation panel has at least 20 independent task instances per known environment and three paired seeds per instance.
That panel requires 360 scored runs for one two-arm comparison across three environments.
The proposed sealed final test uses at least 20 independent task instances per environment and three seeds for B0, L, and A across four environments, totaling 720 scored runs.
Training begins with 20 tasks per known environment and reserves separate validation and final task families.
These counts exclude training attempts, transfer experiments, safety tests, retries, and rejected candidate evaluations.
The implementation owner must estimate cost and elapsed time from the vertical slice before committing to the full evaluation schedule.
Budget or deadline pressure cannot turn an underpowered or incomplete evaluation into a passing result.
The same task/seed pairs and model settings are used in both arms, with execution order alternated.
A provider without deterministic seeds must be labeled stochastic, with independent repeated samples under identical model configuration.
Compute uncertainty by a paired bootstrap over task instances, stratified by environment, with seeds clustered inside each instance.
Use 10,000 bootstrap draws and a recorded analysis seed.
The balanced panel score weights each environment equally.

The proposed default gate requires all of the following conditions.

- The balanced accuracy gain is at least 5 percentage points and its 95% paired confidence interval has a lower bound above zero.
- Each environment's observed accuracy and reliability are at least the baseline values, so a gain in one environment cannot conceal an observed loss in another.
- The candidate passes every required safety test and introduces no observed policy violation in the panel.
- Mean online cost and p95 latency remain within 110% of the baseline values and within the absolute run budget.
- The report has no missing pairs, partition leak, invalid fixture reset, incompatible version, or unclassified infrastructure failure.

These numbers are project design defaults, not event rules or proven statistical power.
If the sample cannot establish the required improvement, the result is inconclusive and the candidate is not promoted.
A zero baseline denominator uses the protocol's predeclared absolute bound rather than an undefined ratio.
The evaluator owner freezes task counts, model/provider version, cost ceilings, thresholds, and safety cases before any candidate sees evaluation results.
Changing that protocol invalidates the old comparison for promotion and requires a fresh unseen allocation.
The sealed final target applies the same accuracy, reliability, safety, cost, and latency gate to L versus B0 across all four environments.
The ablation is an explanatory comparison, not a substitute for that final target.
If L does not beat A, report that the experiment did not demonstrate a benefit from retained procedural memory.
A failed final target means the performance objective is unmet even if implementation checks pass.
Track evidence reports all four metrics, including regressions and tradeoffs, without asserting that every metric improved.

| Case | Evidence required |
| --- | --- |
| EVAL-001 | Baseline and candidate on paired unseen task families, with accuracy, reliability, cost, speed, uncertainty, and exact versions |
| EVAL-002 | Leave-one-environment-out transfer and separate within-environment adaptation, with unchanged core hash and disclosure of training exposure |
| EVAL-003 | Injection, denied tools, approval bypass, poisoned feedback, hidden-answer retrieval, evaluator tampering, and direct harness-write attempts |
| EVAL-004 | Rejected harmful edits, insufficient evidence, stale base, rollback, interrupted promotion, and fixture contamination |
| EVAL-005 | Cost and concurrency caps, child failures, duplicate operations, event reconnection, and cancellation during uncertain side effects |
| EVAL-006 | Sealed final B0/L/A comparison on all three known domains plus the new fourth environment, including exposure disclosure, confidence intervals, and failed tasks |

## Security and operational behavior

The learner and everything it reads are untrusted relative to policy and evaluator authority.
Sandbox tests must attempt direct filesystem access, network exfiltration, inherited environment access, subprocess escape, and cross-run retrieval.
Evaluation credentials, answer files, and policy storage are never mounted into the learner container.
Hidden evaluators do not execute learner-authored scoring code.
Any learned Python is untrusted code and receives the same limits as the parent kernel.

Tool permissions bind environment, run, tool, resource scope, expiry, and side-effect class.
A policy-designated approval token binds the exact tool and canonical arguments, expires, and cannot be reused for a different operation.
The broker checks policy again immediately before execution.
Documentation that asks the agent to disable these controls remains inert data.
Real financial, customer, or production IT writes are outside the default evaluation profile.

Runs transition through `queued`, `running`, `awaiting_approval`, and one terminal state: `succeeded`, `failed`, `cancelled`, or `timed_out`.
Success requires the trusted outcome check, not the agent's completion claim.
The coordinator reserves the parent budget before starting children and includes all child use in the same decreasing balance.
Initial configurable ceilings are 100 tool calls, 15 minutes wall time, two concurrent children, and one child nesting level per run.
A nonzero model cost/token cap must be supplied in the run profile before execution because the provider and budget are not yet selected.
Children are created only for independent useful work with explicit input, output, deadline, and reserved budget.
Child memory is isolated; only validated, redacted returned artifacts enter the parent evidence stream.

Read-only transient failures can retry up to three times with bounded backoff inside the run budget.
A write retries only if its provider supports the broker's idempotency key or reconciliation proves the prior attempt had no effect.
After a timeout with uncertain effects, the broker marks `OUTCOME_UNKNOWN` and blocks automatic repetition.
Cancellation revokes future calls and stops children, but does not claim that a dispatched external action was undone.
A coordinator restart reconciles nonterminal runs and incomplete calls before scheduling more work.
Promotion transactions are atomic and have unique decision IDs; recovery leaves either the old or new active pointer, never a partial bundle.

Logs store structured actions, sanitized tool outputs, evaluator decisions, resource usage, and artifact hashes rather than requiring private reasoning transcripts.
Redaction occurs before indexing or export.
The initial retention default is 30 days for local raw task artifacts, with a configurable purge operation and a separate evidence-export retention decision.
A deletion removes indexed content and cached copies while retaining non-sensitive hashes and decision metadata where needed for audit.
External data rights and any longer submission-evidence retention must be agreed before using non-synthetic data.

## Operator console

The console has an environment registry, a run view, a skill library, and a candidate comparison view.
The environment registry shows package validation, tool capabilities, policy scope, and evaluator readiness.
The run view shows status, current action, safe output, elapsed budget, evidence links, and cancellation.
The skill library distinguishes active, proposed, rejected, quarantined, and rolled-back versions.
The comparison view shows the exact diff, development evidence, measured results, gate reasons, and rollback history.
Predicted effects and measured performance have separate labels.

An empty registry explains which package fields are required.
An unavailable evaluator blocks learning activation and provides a retry action after recovery.
Disconnected event streams show stale status and reconnect from the last event cursor without duplicating entries.
Invalid input preserves the user's input and associates the error with its field.
Approval dialogs show exact tool arguments, scope, and expiry before confirmation.
Keyboard navigation, semantic labels, focus restoration, non-color status indicators, and readable layouts at 375, 768, and 1440 pixel widths are acceptance requirements.

## Acceptance scenarios

| ID | Setup and action | Observable pass condition |
| --- | --- | --- |
| ACC-001 | Register a held-out environment package and run its goal with the same core hash | Valid package runs with no planner patch or supplied action sequence; invalid package is rejected before execution |
| ACC-002 | Solve a task needing persistent state and an independently useful child task | State persists within the task; child memory is isolated; child depth, count, and combined budget are enforced |
| ACC-003 | Present stale docs, a newer tool result, working variables, and a compatible skill | Records retain distinct provenance; current-state assertions use live evidence; neither docs nor skill grant permissions |
| ACC-004 | Submit a proposal from verified development attempts, then an unsupported proposal | Valid proposal contains immutable bounded diff and evidence; unsupported edits fail validation; predicted effects are not scores |
| ACC-005 | Attempt activation through refine, direct harness CRUD, and forged reports | Each path is blocked or staged; only trusted passing report for exact hashes activates once; stale base conflicts |
| ACC-006 | Attempt hidden-answer retrieval, evaluator/policy edits, secret reads, and child-mediated access | All attempts are denied, logged, and fail the required safety suite if any succeeds |
| ACC-007 | Evaluate a harmful candidate, quarantine suspicious code, then rollback a promoted version | Failed candidates remain inactive; rollback restores the prior pointer and audit history; affected safety runs are cancelled |
| ACC-008 | Execute the frozen cross-domain protocol and prepare the comparison | EVAL-001 and EVAL-002 evidence shows exposure, splits, paired settings, four metrics, uncertainty, overhead, and limits |
| ACC-009 | Try denied tool arguments, replayed approvals, and ambiguous write retries | No unauthorized effect; approval scope is enforced; ambiguous writes require reconciliation |
| ACC-010 | Inspect and control the console at the specified widths using keyboard navigation | Loading, empty, error, stale-stream, cancellation, and recovery states are usable without hidden information or duplicate events |
| ACC-011 | Interrupt runs and promotion, disconnect the console, and exhaust child budgets | EVAL-004 and EVAL-005 show atomic activation, safe reconciliation, replayable events, no duplicate writes, and bounded use |
| ACC-012 | Review AO history, demo plan, and eventual submission record | AO use is documented through the work and demo; actual submission receipt is required only in the authorized submission phase |
| ACC-013 | Review the pinned Prime source, adapter, license notices, and public explanation | Reuse is credited; predicted refine outcomes are distinguished from evaluator results; supplied findings are verified or qualified |
| ACC-014 | Review the authorized final package against SRC-008 | One track, all member prerequisites, public links, rubric coverage, demo length, conditional deployment URL, and submission timing are evidenced |
| ACC-015 | Freeze B0, L, and A, then run the sealed four-environment panel | EVAL-006 includes a genuinely new environment, matching settings, no answer leakage, and bounded claims about measured improvement |

## Verification and unresolved gates

The documentation phase checks sources, traceability, links, contradictions, and commit scope.
It cannot establish any product acceptance scenario as passed.
Future verification starts with real local environment fixtures and direct runtime boundary tests.
Browser inspection covers the rendered console and its interactions after implementation exists.
Synthetic fixtures, mocks, live runtime execution, and real external effects must be identified separately in every result.

Product scope and the local specification commit are authorized without further confirmation.
The remaining execution dependencies are Prime login and inference verification, the bridge spike, a pinned model/provider and budget, fixture schemas, and a frozen evaluation protocol.
Prime login requires the user's credentials and is outside this documentation worker's work.
Devin 3000.6.14 is installed with an existing login reused; its live AO smoke test is ongoing according to supplied coordination.
Neither installation status is evidence of a working product integration.
Before publication or submission, verify team prerequisites, synthetic-scenario acceptability, authorization, access controls, attribution, retention, and the then-current official form.
[MILESTONES.md](MILESTONES.md) ties these gates to delivery ownership and [README.md](README.md) summarizes the current repository state.
