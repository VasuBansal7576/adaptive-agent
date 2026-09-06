# Prime HTTP child integration smoke

Session2 app wiring recipe (after importing `LunaChildPlanner`):

```python
prime = PrimeRuntimeAdapter(
    PrimeRuntimeConfig(
        task_id=run_id,
        max_model_tokens=max(1, max_tokens),
        max_total_wall_seconds=max(0.1, wall_seconds),
        child_runs=max(0, child_runs),
        max_child_depth=1,
    ),
    broker=CapabilityBroker(run_id, authorizer=authorize),
)
client = PrimeCliModelClient(coding_agent_dir=coding_dir)

def persist(evidence):
    runtime._record_model_response(run_id, package, evidence)
    return prime.record_model_observation(evidence, trusted_parent=True)

child_planner = LunaChildPlanner(
    client,
    budget=prime.planner_budget,
    observation_sink=persist,
)
prime.child_planner = child_planner
parent_client = child_planner.parent_model_client(observation_sink=persist)
planner = LunaPlanner(parent_client, prime, Sink(), limits=limits, emit=emit)
```

`SharedLedgerModelClient` persists each parent receipt before accounting and
charges the shared ledger exactly once. `LunaChildPlanner` persists each child
receipt before plan parsing or exhaustion, then charges that same ledger once.
The planner conversation remains unchanged, so generated parent code can call
`await host_request("rlm.run", {"prompt": ..., "kwargs": ...})`.

## Runnable HTTP smoke

Run from the session2 worktree after applying the recipe, with existing AO
authentication only (no credential values are printed):

```bash
PRIME_AGENT_CODING_AGENT_DIR=/Users/vasu/.ao/data/agent-runtime/prime-agent PYTHONPATH=src .venv/bin/python http_smoke.py
```

The smoke uses `create_runtime_app`, registers a temporary environment through
HTTP, creates a run with `childRuns: 1`, launches it through `POST /runs/{id}/launch`,
and inspects the durable SSE/evidence artifacts. The generated parent executed
one Docker child with structured `kwargs` records.

Observed output from the authenticated run:

```text
runId=run_66d59c55602b47119a5a7bec698098ea
parent responseId=resp_0c56f31844da6924016a9cc64c08dc87d0a4ec68c4ccc9ac0f totalTokens=1824
child responseId=resp_0dc115a43d399dd6016a9cc65290a087d0983d730295d58f79 totalTokens=1654
parent continuation responseId=resp_0fadc4f7a52d0d5a016a9cc66017f087d0b51862ee5cc55ea9 totalTokens=2089
child Docker kernel result={'records': [{'id': 'alpha', 'sum': 12}, {'id': 'beta', 'sum': 11}], 'total': 23, 'verified': True}
child model_response evidence persisted: true
shared ledger=5567 (1824 + 1654 + 2089)
```

The final HTTP run status in this temporary environment was `failed` only
because its `_RegisteredPackage` has the deliberate unconfigured evaluator;
the durable event stream still proves parent model -> `rlm.run` -> child model
-> isolated Docker kernel -> parent result. Production environments must wire
a trusted evaluator separately. No host fallback or provider switch is used.
