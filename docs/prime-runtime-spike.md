# Prime runtime integration spike

This note records the bounded local spike against the installed interfaces. It is
not a claim that the full product boundary is complete.

## Verified

- `prime-agent --version` returned `0.9.2`.
- `prime-agent --help` exposed the real CLI flags `--provider`, `--model`,
  `--print`, and `--mode`; no undocumented inference endpoint is used here.
- The installed bundle contains `dist/core/kernel/repl-manager.d.ts` and
  `ReplKernelManager.execute(code, opts)` / `start()` / `shutdown()` / `kill()`.
- The installed Python source is
  `dist/prime-agent-runtime/src/rlm/repl.py`. Its adjacent `repl.md` documents
  protocol version 3, newline-delimited JSON, `execute`, `interrupt`,
  `host_reply`, `snapshot`, `restore`, `list_names`, and `shutdown`.
- The adapter starts `python -m rlm.repl`, performs the `ready` handshake, and
  executed state-sharing cells (`value = 40; value + 2`, then `value + 1`) with
  results `42` and `41`.
- A real host bridge request returned run-scoped capability metadata and a
  broker call returned a structured result. Learner requests for
  `harness.write`, policy/evaluator/promotion writes, credentials, and hidden
  data were denied.
- The configured model provenance is the authenticated subscription selector
  `openai-codex/gpt-5.6-luna` through provider `openai-codex`. The adapter does
  not call the balance-gated default Prime Inference path (which previously
  returned HTTP 402).
- A bounded CLI model smoke using the real authenticated path,
  `prime-agent --print --no-tools --provider openai-codex
  --model openai-codex/gpt-5.6-luna`, returned `MODEL_SPIKE_OK` in 10.24s.
  This verifies model reachability only; it is not a complete goal/tool/
  evaluator run.

## Boundary and limitations

The adapter now fails closed unless Docker is available. It builds a minimal
image containing the verified Prime runtime source, then starts one learner
container per task with no host mounts or Docker socket, `--network=none`, a
nonroot uid, read-only root, writable tmpfs `/tmp`, dropped capabilities,
`no-new-privileges`, pids/memory/cpu limits, and bounded file descriptors.
The parent keeps the JSON-lines protocol on stdio, so broker replies work
without learner network access. Provider credentials and model authentication
remain in the trusted parent and are never copied into the image or container.

The image build itself is a trusted deployment operation and may pull the
pinned Python base image. If Docker or that image is unavailable, adapter
startup fails; there is no host-process fallback. The learner source policy is
defense in depth only. Container policy is the isolation boundary. The
configured model provenance is not a model-call result: a complete goal-to-
model-to-tool/evaluator evidence chain still belongs to the control-plane
integration and is not claimed by this bridge spike.
