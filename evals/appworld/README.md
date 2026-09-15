# AppWorld evaluation adapter

## 2026-09-09 新入口

正常 ConversationRuntime + GENERAL / 真实 CODE / Code Reviewer 的薄适配见
[`APPWORLD_HANDOFF.md`](../../APPWORLD_HANDOFF.md)。入口是
`python -m evals.appworld.run_conversation --help`；免费接线脚本为
`scripts/probe_appworld_conversation.py`。旧 General-only `run.py` 已删除，
避免它误读当前 Qwen/GLM 的分角色配置。

Status (2026-08-31): real container smoke passed; two bounded calibration runs completed, including one officially successful Train task. A frozen batch baseline and optimization comparison are still pending. See ../../docs/appworld-calibration.md.
The adapter evaluates the existing PersonalOps supervisor/executor/reporter/replanner graph. It does not replace it with a separate toy agent.

## Environment boundary

- Agent: existing Python 3.10 environment and current LangChain/LangGraph dependencies.
- World: AppWorld 0.1.3.post1 in Python 3.12, inside a disposable Docker container.
- Transport: JSON over the container's stdin/stdout, avoiding Python dependency conflicts and an exposed HTTP port.
- Runtime container: no network, no host bind mounts, non-root user, read-only base filesystem, dropped capabilities, memory/process limits; only temporary output directories are writable.
- Controller can initialize/finish/evaluate/export. The model gets appworld_discover for documentation and appworld_execute for business operations.
- Tool calls are serialized. The model must put dependent operations in one code call.
- No personal memory, checkpoint, Feishu connector, local shell, browser, or real account tool is attached.
- This is defense in depth using Docker and AppWorld's syntax/execution guards, not a proof of safety against hostile code. The official AppWorld engine and grader dependencies/data reside in the worker image; the protocol prevents ordinary agent access to grading, but does not establish an independent OS boundary between executor and grader. A stronger adversarial isolation audit remains pending.

AppWorld's encrypted source/data is unpacked only locally. Do not publish the built image, raw benchmark contents, traces or output archives. Respect the [upstream restrictions](https://github.com/StonyBrookNLP/appworld).

## Build and verify

Build through the helper so the official encrypted data bundle is downloaded with a timeout, checked against its pinned SHA-256, and staged with only the required build files. The entire repository is never sent to Docker.

```text
python -m evals.appworld.build
python -m evals.appworld.smoke
python -m evals.appworld.prepare_phoenix
```

On this Windows machine these commands can use C:/Python313/python.exe. The helper discovers Docker inside WSL, including Snap installations whose executable is not on WSL's non-login PATH.

The smoke test uses only fixed arithmetic, not an LLM. It checks variable persistence, actual container settings, rejection of premature grading, rejection of execution after finish, and a response from the official evaluator. It deliberately does not solve the Train task; its success/failure must never be reported as an Agent score.

requirements.snapshot.txt pins all 80 AppWorld-side dependencies. The Docker build enforces those versions and runs pip check after the bootstrap installation. The Python base image and official data SHA-256 are pinned; each trial also captures the final image ID. The first completed bootstrap layer is retained to avoid repeating the long dependency download.

## One real, bounded development trial

First inspect configuration without sending an API request:

```text
C:/ProgramData/anaconda3/envs/agent/python.exe scripts/show_model_roles.py
```

Run only after smoke verification:

```text
C:/ProgramData/anaconda3/envs/agent/python.exe -m evals.appworld.run_conversation --task TASK_ID --split train --max-calls 60 --max-interactions 90 --allow-paid
```

This command can incur provider costs. It uses model names and keys from the configured environment/.env, never sends keys into Docker, and does not load Feishu settings. Provider retries are disabled so they cannot bypass the model-call ceiling. Other .env settings do not silently alter planning defaults. Usage absent after errors is unknown, not zero.

--skill PATH adds a reusable skill to both the planning context and executor system prompt. No task-specific answer belongs in that file. All comparison groups must use identical model, environment, limits and data manifests.

--wall-timeout applies to the asynchronous Agent run. Container transport and cleanup have separate bounded timeouts; it is not a hard deadline for the whole CLI process.

--phoenix starts/reuses only the local Phoenix service on port 6007 and associates the trial with task/split/skill metadata. Without it, private local trajectory and result files are still written; do not claim full Phoenix coverage for such a run.

The default development image exposes only Train and Dev. Test-N uses the separate
`personalops-appworld-test-normal` image, which exposes only `test_normal`.
Create one immutable seeded A/B manifest with `run_test_normal_batch --prepare`;
run either half from that manifest without scanning Phoenix or historical trials.
Test-C remains unavailable.

Test-N half execution now uses bounded multi-process parallelism.  The default is
four independent task processes; use `--parallelism 1` for serial execution or a
value up to eight after checking host and provider capacity. `--num-processes`
is accepted as an alias matching AppWorld's upstream CLI terminology. For example:

```text
C:/ProgramData/anaconda3/envs/agent/python.exe -m evals.appworld.run_test_normal_batch --batch-id BATCH_ID --half A --parallelism 4 --allow-paid
```

Every process still enters through `evals.appworld.run_conversation`, receives a
unique Docker world and private trial directory, and runs the normal online
single-task evaluator.  The batch parent starts/reuses Phoenix before fan-out,
writes one log per task, and serializes monitor aggregation after each completion.
It never retries a failed or interrupted task.  `progress.json` contains
`active_task_indices` and `active_tasks` for concurrent status inspection.

## Evidence files

Each trial writes to .agent/evaluations/aw_<uuid>/, which is Git-ignored:
- result.json: outcome, model usage completeness, budgets, source hashes, package versions, image ID and skill/schema hashes.
- trajectory.private.json: actual code and environment observations.
- runtime.private.log: dependency/validation warnings, which may embed benchmark data and must not be published.
- grade.private.json: official Train/Dev grading output, invisible to the model.
- appworld-output.private.zip: original task output files for subsequent official re-evaluation; no automatic host extraction.

A model's self_reported_final_status is separate from official_task_success. Infrastructure failures retain a null official score; batch aggregation must keep them in the declared trial accounting rather than silently dropping them.
The Test-N batch runner provides frozen manifests and Pass@1 execution. SGC grouping, uncertainty analysis, SkillOpt optimization, Test-C and a claim-ready final protocol remain separate work.

## Offline regression tests

```text
C:/ProgramData/anaconda3/envs/agent/python.exe -m unittest discover -s tests -v
```

Twenty-two tests passed through 2026-09-01, covering the Phoenix audit, real graph adapter, failed-response usage, and development-only diagnostics. The graph integration test uses scripted model responses and an execute-only stub; it verifies actual PersonalOps graph wiring, not real model quality.
The installed dependency stack emits a nul-file ResourceWarning on interpreter exit. It does not fail these checks; it has not been diagnosed as an AppWorld defect.


## Calibration and observability limits

The installed data manifests contain 90 Train and 57 Dev tasks. Do not substitute paper counts without verifying the release. A single Train success is not a general success rate.

prepare_phoenix prefetches the official WASM binary with the SHA-256 pinned by Phoenix 19.4.0. The local trace service can otherwise block on this optional component's network download. The helper does not disable integrity or TLS checks.

The runner records purpose (calibration/baseline/experiment), thinking mode, model limits, provider retry policy, SDK versions and trace ID. Explicit model temperature is currently unset; provider defaults apply and must be reported. World seed does not make model sampling deterministic.

Before stopping an owned Phoenix service, the runner verifies that its root span reached the SQLite database. This checks root persistence, not complete delivery of all child spans. Trace delivery problems remain separate from official task success. One calibration trial lacks its root span; its original record is not rewritten.

Run local zero-model delivery verification with the Agent environment:

```text
python -m evals.appworld.trace_delivery
```

Run deterministic triage with Python 3.13 (for Phoenix SQLite JSONB support):

```text
python -m evals.appworld.diagnose .agent/evaluations/<trial_id> --phoenix-db .agent/phoenix/phoenix.db
```

Parsing failure can still be billable: the usage callback retains structured provider usage from truncation errors. Transport failures without provider usage remain unknown. A successful callback may precede an upper-layer schema failure; official task success, Agent self-report, and runtime status stay separate.

## Day-one closeout

The user's latest scope is to stop after integration verification and learn from the traces before further optimization. Start with ../../docs/SESSION_HANDOFF.md and ../../docs/worklogs/2026-08-31.md. Do not start a batch or SkillOpt from an older plan automatically.

The final zero-model acceptance passed on 2026-08-31: actual Docker isolation, persistent execution, official grader response, and root-span persistence after stopping Phoenix. This is infrastructure evidence, not another solved benchmark task.

```text
C:/ProgramData/anaconda3/envs/agent/python.exe -m evals.appworld.acceptance
```

Private raw evidence and source/database/session snapshots are indexed in .agent/handoffs/2026-08-31/INDEX.md. The transcript snapshot has a stated cutoff; it does not reconstruct unrecorded or truncated terminal output.


## Readable Phoenix evaluation view (2026-09-01)

Evaluation runs set PHOENIX_TRACE_PROFILE=curated. Curated mode disables broad
OpenInference auto-instrumentation and records application-owned spans instead.
This prevents one provider call from appearing as both ChatDeepSeek and
ChatCompletion and removes RunnableSequence/LangGraph implementation noise from
the interview view. Normal PersonalOps runs keep the default full profile for
deep framework debugging.

Use two terminals for a live, zero-model demonstration.

Terminal A keeps the local UI available:

    C:/ProgramData/anaconda3/envs/agent/python.exe scripts/eval_demo.py serve

Open http://127.0.0.1:6007. Terminal B generates a real compatibility trace
without loading model credentials:

    C:/ProgramData/anaconda3/envs/agent/python.exe -m evals.appworld.acceptance

Choose project personalops-eval-infrastructure. The trace is named
EVAL / Infrastructure acceptance and contains ordered PREPARE, RUN and GRADE
stages plus one child span for each AppWorld tool execution. This acceptance
deliberately leaves the sampled task unsolved, so official_task_success=false is
expected while infrastructure passed=true.

Print a safe result summary without starting Phoenix or printing task/tool
content:

    C:/ProgramData/anaconda3/envs/agent/python.exe scripts/eval_demo.py list
    C:/ProgramData/anaconda3/envs/agent/python.exe scripts/eval_demo.py inspect --trial-id <id>

A paid single-task run with --phoenix uses project personalops-eval-train or
personalops-eval-dev and root EVAL / AppWorld trial. Its top-level stages are:

1. PREPARE: load the pinned container task and verify isolation.
2. RUN: run the existing PersonalOps graph. Manual LLM spans show role, call
   index, status and token usage. Manual tool spans show API names/counts and
   lengths, while raw code/output remain private.
3. GRADE: lock execution and call the official AppWorld evaluator through the
   controller-only protocol.
4. SAVE: persist private grade and archive evidence locally.

Historical full-profile traces are immutable and remain in the database. The new
project separation gives future evaluations a clean view; it does not rewrite or
delete old evidence.
