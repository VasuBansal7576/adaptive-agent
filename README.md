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

Implementation is active from this clean integration branch.
The Python control-plane package, Prime adapter, Luna planner, evaluator, and integration tests are now present and configured through `pyproject.toml`.
The durable production Store/Broker, reviewed console transport, and application CI remain separate delivery work.
The proposed stack is React, Tailwind, Vite, and Bun with a small backend selected by a bounded Prime API bridge spike.
Prime is the proposed runtime, with isolated IPython as a documented fallback if its interfaces cannot support the required boundaries.

The verified smoke path uses Prime 0.9.2 session 3 with a ChatGPT subscription, `openai-codex/gpt-5.6-luna`, and an actual Python `2+2` result of `4`.
Default Prime Inference returned HTTP 402 for lack of balance, while the subscription provider worked.
Devin 3000.6.14 session 4 and OpenCode session 5 each ran `pwd` successfully with no file changes.
A bounded live neutral path now exercises authenticated Luna, Docker Prime, a read-only broker fixture, a trusted objective evaluator, and console-facing run events.
Production Devin broker authority and the complete behavioral evaluation remain pending.

The active phase covers implementation, local verification, and M2-M4 coordination.
Local specification commits, fast-forward integration into local `main`, and synchronization to the filesystem-only origin are complete.
Public remote creation, publication, pushes to public remotes, pull requests, and submission require later authorization.
The official build window closes on 2026-09-07 at 03:30 IST.
Public repository and social-demo obligations are recorded in the milestones and are not yet fulfilled.

Prime source reuse must preserve verified MIT notices.
Backpass and Vision inform evidence handling and bounded edits, but are not required dependencies or autonomous evaluators.
Evaluation targets and source-review limitations are documented in the specification; no measured performance or track-compliance result is claimed.
