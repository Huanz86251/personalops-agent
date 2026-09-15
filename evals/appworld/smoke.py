"""No-model smoke test of a real, isolated AppWorld and its official evaluator."""
import argparse
import json
from pathlib import Path
import time
import uuid

from .protocol import DockerWorld, WorldError


def check_world(image="personalops-appworld:0.1.3.post1"):
    from observability import trace_span, set_span_output
    from .adapter import make_execute_tool

    start = time.monotonic()
    with DockerWorld(image) as world:
        with trace_span(
            "1 / PREPARE / verify isolated AppWorld",
            kind="chain",
            input_value={"image": image, "split": "train"},
        ) as setup_span:
            ids = world.request("list_tasks", split="train")
            isolation = world.isolation_report()
            assert isolation["network_mode"] == "none"
            assert isolation["host_bind_mounts"] == []
            assert isolation["readonly_rootfs"] and not isolation["privileged"]
            assert isolation["user"] == "10001:10001"
            assert not isolation["model_credentials_present"]
            trial = "smoke_" + uuid.uuid4().hex
            task = world.request(
                "initialize", task_id=ids[0], split="train", trial_id=trial
            )
            set_span_output(setup_span, {
                "task_loaded": True,
                "network_disabled": True,
                "host_bind_mounts": 0,
                "model_credentials_present": False,
            })

        with trace_span(
            "2 / RUN / deterministic tool probe",
            kind="tool",
            input_value={"external_model_calls": 0, "probe": "persistent Python state"},
        ) as run_span:
            # Infrastructure probe, not an attempt to solve the sampled task.
            execute_tool = make_execute_tool(world)
            first = execute_tool.invoke({
                "source_refs": ["MODEL"],
                "reason": "Deterministic infrastructure arithmetic probe.",
                "code": "probe_value = 6 * 7\nprint(probe_value)"
            })
            second = execute_tool.invoke({"source_refs": ["MODEL"],
                "reason": "Reuse the prior probe variable.", "code": "print(probe_value + 1)"})
            assert first.strip() == "42", first
            assert second.strip() == "43", second
            assert world.request("task_completed") is False
            set_span_output(run_span, {
                "first_probe": 42,
                "second_probe": 43,
                "state_persisted": True,
                "task_deliberately_unsolved": True,
            })

        with trace_span(
            "3 / GRADE / verify protected evaluator",
            kind="chain",
            input_value={"agent_cannot_call_grader": True},
        ) as grade_span:
            try:
                world.request("evaluate")
            except WorldError:
                premature_evaluation_rejected = True
            else:
                raise AssertionError("Grader was available before finish")
            world.request("finish")
            try:
                world.execute("print(1)")
            except WorldError:
                post_finish_execution_rejected = True
            else:
                raise AssertionError("Execution was allowed after finish")
            grade = world.request("evaluate")
            assert isinstance(grade.get("success"), bool)
            assert grade.get("num_tests", 0) > 0
            set_span_output(grade_span, {
                "premature_evaluation_rejected": premature_evaluation_rejected,
                "post_finish_execution_rejected": post_finish_execution_rejected,
                "official_evaluator_returned": True,
                "official_task_success": grade["success"],
                "task_success_expected": False,
            })

        report = {
            "kind": "infrastructure_smoke_not_agent_benchmark",
            "appworld_version": task["appworld_version"], "image_id": world.image_id,
            "train_task_count": len(ids), "task_id": task["task_id"],
            "isolation": isolation,
            "persistent_execution_verified": True,
            "premature_evaluation_rejected": premature_evaluation_rejected,
            "post_finish_execution_rejected": post_finish_execution_rejected,
            "official_evaluator_returned": True,
            "official_task_success": grade["success"],
            "official_num_tests": grade["num_tests"],
            "model_calls": 0, "elapsed_seconds": round(time.monotonic() - start, 3),
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="personalops-appworld:0.1.3.post1")
    parser.add_argument("--output", type=Path, default=Path(".agent/evaluations/appworld-smoke.json"))
    args = parser.parse_args()
    report = check_world(args.image)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
