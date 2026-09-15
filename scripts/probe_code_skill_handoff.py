"""Opt-in, model-free live CODE containers with a deliberately faulty candidate.

Uses the existing sandbox manager/backends, isolated synthetic inputs, independent
contract tests, immutable host exports and ownership-checked cleanup. Does not
build images, start the daemon, invoke agents or publish into any real project.
"""

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from planning_models import CodeTaskContract
from workers.docker_sandbox import CodeSandboxManager, CodeSandboxPolicy, DockerCLI
from workers.code_publisher import compute_code_tree_revision


class ExistingDaemonCLI(DockerCLI):
    def ensure_daemon(self):
        self.require_success(["version", "--format", "{{.Server.Version}}"], timeout=30,
                             operation="Read-only Docker readiness check")


def main():
    root = Path(__file__).resolve().parents[1]
    fixtures = root / "tests" / "fixtures" / "code_skill_probe"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8]
    output = root / ".agent" / "code-skill-probes" / run_id
    output.mkdir(parents=True, exist_ok=False)
    handoff = output / "handoff"
    handoff.mkdir()
    contract = CodeTaskContract.model_validate_json((fixtures / "contract.json").read_text(encoding="utf-8"))
    (handoff / "contract.json").write_text(contract.model_dump_json(indent=2), encoding="utf-8")
    (handoff / "input.csv").write_text("id,category,amount\n甲,餐饮,12.50\n乙,交通,2\n", encoding="utf-8")
    (handoff / "config.json").write_text('{"minimum_amount": 10}', encoding="utf-8")
    clean_source = (fixtures / "filter_report.py").read_bytes()
    assert clean_source.count(b"amount >= minimum") == 1
    faulty_source = clean_source.replace(b"amount >= minimum", b"amount > minimum")
    review_tests = (fixtures / "review_checks.py").read_bytes()
    manager = CodeSandboxManager(CodeSandboxPolicy(auto_build=False, execute_timeout_seconds=30),
                                 cli=ExistingDaemonCLI())
    pair = None
    log = {"run_id": run_id, "mode": "manual role substitution; no model calls",
           "deliberate_fault": "strict > instead of required >=", "events": []}

    def save():
        (output / "result.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

    def event(name, result):
        log["events"].append({"name": name, "result": result})
        save()
        print(json.dumps({"name": name, "result": result}, ensure_ascii=False), flush=True)

    def execute(backend, name, command, expected=0):
        result = backend.execute(command, timeout=30)
        event(name, {"command": command, "exit_code": result.exit_code,
                     "output": result.output, "truncated": result.truncated})
        if expected is not None:
            assert result.exit_code == expected, (name, result.exit_code, result.output)
        return result

    def upload(backend, name, path, content, should_fail=False):
        responses = backend.upload_files([(path, content)])
        errors = [r.error for r in responses]
        event(name, {"path": path, "errors": errors})
        assert bool(any(errors)) == should_fail, (name, errors)

    try:
        pair = manager.create_pair("skill-field-test-" + run_id, handoff_root=handoff)
        log["pair"] = asdict(pair)
        log["isolation"] = manager.isolation_report(pair)
        save()
        worker = manager.worker_backend(pair)
        reviewer = manager.reviewer_backend(pair)
        upload(worker, "worker_candidate", "/workspace/filter_report.py", faulty_source)
        execute(worker, "worker_reads_shared_input", "python -c 'from pathlib import Path; print(Path(\"/handoff/input.csv\").read_text())'")
        upload(worker, "worker_cannot_modify_handoff", "/handoff/probe-write.txt", b"denied", should_fail=True)
        upload(worker, "worker_private_state", "/tmp/worker-private.txt", b"PRIVATE_WORKER_CONTEXT")
        execute(worker, "worker_smoke", "python /workspace/filter_report.py --input /handoff/input.csv --config /handoff/config.json --output /workspace/worker-output.csv")
        pair = manager.handoff(pair, "REVIEWER")
        files = reviewer.download_files(["/workspace/filter_report.py", "/workspace/worker-output.csv", "/handoff/contract.json"])
        assert all(r.error is None for r in files)
        assert files[0].content == faulty_source
        event("reviewer_reads_candidate_and_output", {"paths": [r.path for r in files],
            "candidate_sha256": hashlib.sha256(files[0].content).hexdigest(),
            "csv": files[1].content.decode("utf-8")})
        execute(reviewer, "reviewer_no_worker_private_state", "test ! -e /tmp/worker-private.txt")
        upload(reviewer, "reviewer_cannot_modify_candidate", "/workspace/filter_report.py", b"denied", should_fail=True)
        assert reviewer.download_files(["/workspace/filter_report.py"])[0].content == faulty_source
        upload(reviewer, "reviewer_tests", "/review/test_contract.py", review_tests)
        execute(reviewer, "reviewer_environment", "python -c 'import sys,importlib.util,shutil; print(sys.version); print({k:importlib.util.find_spec(k) is not None for k in (\"pytest\",\"playwright\",\"pydantic\")}); print({k:shutil.which(k) for k in (\"node\",\"chromium\",\"chromium-browser\")})'")
        # No assertion weakening: this is a test invocation error, not a defect.
        execute(reviewer, "collection_without_candidate_path", "python -m pytest /review/test_contract.py --collect-only -q -p no:cacheprovider", expected=2)
        execute(reviewer, "correct_collection", "PYTHONPATH=/workspace python -m pytest /review/test_contract.py --collect-only -q -p no:cacheprovider")
        execute(reviewer, "review_revision_1", "PYTHONPATH=/workspace python -m pytest /review/test_contract.py -q -p no:cacheprovider --basetemp=/review/tmp-r1 --junitxml=/review/revision-1.xml", expected=1)
        manager.export_candidate(pair, output / "candidate-r1")
        event("candidate_r1_revision", compute_code_tree_revision(output / "candidate-r1"))
        pair = manager.handoff(pair, "WORKER")
        execute(worker, "worker_cannot_read_reviewer_tests", "test ! -e /review/test_contract.py")
        upload(worker, "worker_repair", "/workspace/filter_report.py", clean_source)
        pair = manager.handoff(pair, "REVIEWER")
        assert reviewer.download_files(["/workspace/filter_report.py"])[0].content == clean_source
        execute(reviewer, "review_revision_2", "PYTHONPATH=/workspace python -m pytest /review/test_contract.py -q -p no:cacheprovider --basetemp=/review/tmp-r2 --junitxml=/review/revision-2.xml")
        pair = manager.freeze(pair)
        manager.export_candidate(pair, output / "candidate-r2")
        manager.export_review(pair, output / "review")
        event("candidate_r2_revision", compute_code_tree_revision(output / "candidate-r2"))
        log["outcome"] = "PASS: handoff, read-only scopes, diagnostic/repair/retest and exports"
    except Exception as error:
        log["outcome"] = "FAILED"
        log["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if pair is not None:
            # Exact resources created above only; no wildcard/global prune.
            owned = (("container", pair.worker_container), ("container", pair.reviewer_container),
                     ("volume", pair.candidate_volume), ("volume", pair.review_volume))
            for kind, name in owned:
                assert name.startswith(pair.pair_id + "-")
                manager._inspect_owned_resource(kind, name)
            manager.cleanup(pair)
            log["cleanup"] = {name: manager._inspect_owned_resource(kind, name) is None for kind, name in owned}
        save()
        print(json.dumps({"evidence": str(output), "outcome": log.get("outcome"), "cleanup": log.get("cleanup")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
