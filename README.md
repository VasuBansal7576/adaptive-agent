# Adaptive Agent

A domain-agnostic agent for Syndicate Track 1 that discovers workflows in unfamiliar environments and learns reusable procedures from its own attempts and verified outcomes.
Each environment supplies goals, docs, tools, policy, and an evaluator without changing the core planner.
Finance, support, and IT are evaluation packs, with a genuinely new fourth environment reserved for final testing.

The learning loop records evidence, proposes a bounded change, independently compares baseline and candidate, and automatically promotes or rejects the change.
Policies, credentials, evaluators, and hidden answers remain outside the learner's authority.
Versioning and rollback protect the active skill bundle.
This is procedural adaptation, not model-weight training or a promise of universal capability.

- [SPEC.md](SPEC.md) defines requirements, contracts, architecture, learning, evaluation, security, and acceptance.
- [MILESTONES.md](MILESTONES.md) defines the runnable vertical slice, full delivery gates, AO ownership, and submission traceability.

Implementation is active in parallel AO worktrees, but implementation code is not integrated into this documentation branch.
This update records the root-verified current checkpoint beyond `d8e9293` without changing active workers or claiming a merge.
The exact branch refs, dirty-worktree status, and component ownership are tracked in [MILESTONES.md](MILESTONES.md).

The Python package is configured through `pyproject.toml` on the implementation branches.
Its implemented entry points are `adaptive-agent` for the FastAPI control plane and operator-console backend, and `adaptive-agent-plan` for the authenticated Luna planner loop.
The console package is configured through `package.json` and uses React, Tailwind, Vite, and Bun.
The implemented console commands are `bun run dev`, `bun run test`, `bun run build`, `bunx tsc --noEmit`, and `bun run api:smoke`.
`adaptive-agent` and `adaptive-agent-plan` are branch-local implementation commands until the corresponding branch work is integrated.

The verified Prime smoke path uses Prime 0.9.2 session 3 with a ChatGPT subscription, `openai-codex/gpt-5.6-luna`, and an actual Python `2+2` result of `4`.
Default Prime Inference returned HTTP 402 for lack of balance, while the subscription provider worked.
The bounded Prime runtime adapter also recorded persistent kernel results `42` and `41`, a structured host-bridge response, and denied learner requests for harness, policy, evaluator, credential, and hidden-data access.
Devin 3000.6.14 session 4 and OpenCode session 5 each ran `pwd` successfully with no file changes.
The console worker reported a live FastAPI session, registration, create/launch, SSE cursor, and cancellation smoke as passing.
That smoke used a synthetic neutral environment and does not establish a real model-to-tool-to-evaluator product run.
It remains historical transport evidence only until its Store and data target are proven isolated from root QAdata.

Tests and fixture runs are not interchangeable with those real-run checks.
The backend and benchmark tests use synthetic providers, drivers, and deterministic fixtures, while console tests use simulation fixtures and captured wire frames.
The current checkpoint has not produced a sealed four-environment evaluation, a measured performance result, or a complete behavioral acceptance result.

Root-verified current evidence beyond `d8e9293` is bounded integration and safety evidence only.
The final2 backend integration on session 2 is rooted at committed ref `4832487` with a dirty overlay in `api.py`, `app.py`, `evaluation_job.py`, `evaluation_store.py`, and `tests/test_api.py`.
That final2 integration reports 171 backend tests passed.
Actual Docker-backed EVAL-003 passed its seven safety cases on isolated temporary Store and data paths.
Actual Docker-backed EVAL-005 passed its two runtime cases, `child_failure_recovery_and_cap` and `shared_model_cost_token_cap`, using the deterministic probe model rather than a paid model call.
Four isolated Controller probe executions covering EVAL-004 and EVAL-005 passed on fresh temporary Stores and data directories.
The latest session-5 console ref is `ae482da`, with 61/61 tests, TypeScript, and build clean; its latest report did not include live smoke.
These results do not establish a product-wide run, heldout performance, or final evaluation success.

QA data-integrity hold: the original real evidence for `run_2fc` is contaminated by appended synthetic `model_response` rows at sequences 20 and 21, named `learning-model-fake-resp-1` and `learning-model-r20`.
All history must be preserved, but those rows must not be erased, relabeled, or counted as real evidence.
No transport smoke may target the root QAdata or port 8000 because the real authenticated path invokes the paid model.
Every test and probe must use an isolated temporary Store and data directory.

The repository contains no literal occurrence of the run or row identifiers, so the source and ownership of those specific append operations are not established here.
The related evaluator-owned probe path observed on session 2 is `Controller.execute_probe`, authored in the session-2 probe commits `59adfb5` and `f423fac`.
That path creates `tempfile.TemporaryDirectory(prefix="aa-probe-")` and a `Store` under its temporary directory, but this code-level property does not prove that `run_2fc` was isolated.
Session 6 must create clean independent evaluation data, regenerate and validate real development evidence and its candidate, and retain exact path-level isolation proof before any sealed panel runs.

The documentation phase is complete on this branch.
The local-main assembly plan is limited to the session-2 final2 backend overlay followed by the session-5 console ref `ae482da`.
It has not been performed here.
After those two inputs are materialized and reviewed, run the isolated backend and console checks with fresh Store and data paths, then reassess M2-M4 against the evidence gates.
Session 3, session 4, session 6, and session 7 work remains separate evidence or follow-up input and is not part of that assembly plan.
Implementation, local verification, and M2-M4 coordination remain active across the worker branches.
Public remote creation, publication, pushes to public remotes, pull requests, and submission remain unauthorized and incomplete.
The official build window closes on 2026-09-07 at 03:30 IST.
Public repository and social-demo obligations are recorded in the milestones and are not yet fulfilled.

Prime source reuse must preserve the verified MIT attribution and notices.
The reused source is Prime Agent by Prime Intellect AI, pinned for the initial static review at commit [`9c54a35dac3a2ad17910074d66664859ea175666`](https://github.com/PrimeIntellect-ai/prime-agent/tree/9c54a35dac3a2ad17910074d66664859ea175666), under the MIT License.
The installed Prime 0.9.2 runtime smoke is separate from that pinned static-source review.
Backpass and Vision inform evidence handling and bounded edits, but are not required dependencies or autonomous evaluators.
Evaluation targets, source-review limitations, and acceptance meanings remain defined by the specification.
No measured performance, full acceptance, public publication, or submission result is claimed.

One explicit blocker remains unmet.
The user-accepted Luna-through-ChatGPT subscription path has no API-key or provider switch.
The trusted parent records actual model usage and rejects after aggregate token exhaustion.
Local token limits and post-response rejection are not hard provider enforcement, so the strict per-call output-token-cap criterion remains UNMET.
This documentation does not alter the pass meaning of any requirement in [SPEC.md](SPEC.md).
