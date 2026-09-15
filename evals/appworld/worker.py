"""Container-side AppWorld protocol; stdout is reserved for JSON responses."""
import contextlib
import json
import os
import sys

VERSION = "0.1.3.post1"
world = None
phase = "new"


def allowed_splits():
    values = tuple(
        value.strip()
        for value in os.getenv(
            "PERSONALOPS_APPWORLD_ALLOWED_SPLITS", "train,dev"
        ).split(",")
        if value.strip()
    )
    if not values or any(value not in {"train", "dev", "test_normal"} for value in values):
        raise ValueError("Invalid frozen AppWorld split policy")
    return values


def handle(request):
    global world, phase
    from appworld import AppWorld, load_task_ids
    op = request["op"]
    if op == "list_tasks":
        if request["split"] not in allowed_splits():
            raise ValueError("Requested split is not exposed by this frozen worker")
        return load_task_ids(request["split"])
    if op == "initialize":
        if phase != "new":
            raise ValueError("One task per container; initialize cannot be repeated")
        split = request["split"]
        if split not in allowed_splits():
            raise ValueError("Requested split is not exposed by this frozen worker")
        if request["task_id"] not in load_task_ids(split):
            raise ValueError("Task does not belong to declared split")
        world = AppWorld(
            task_id=request["task_id"], experiment_name=request["trial_id"],
            max_interactions=request.get("max_interactions", 45),
            timeout_seconds=request.get("execution_timeout_seconds", 30),
            random_seed=request.get("seed", 100),
            raise_on_unsafe_syntax=True, null_patch_unsafe_execution=True,
        )
        phase = "running"
        task = world.task
        return {
            "task_id": task.id, "instruction": task.instruction,
            "datetime": str(task.datetime),
            "split": split,
            "frozen_split_policy": list(allowed_splits()),
            "supervisor": {key: task.supervisor[key]
                           for key in ("first_name", "last_name", "email", "phone_number")},
            "appworld_version": VERSION,
        }
    if op == "execute":
        if phase != "running":
            raise ValueError("Execution is allowed only before finish")
        code = request["code"]
        if not isinstance(code, str) or len(code) > 100_000:
            raise ValueError("Invalid or oversized code")
        return world.execute(code)
    if op == "task_completed":
        if phase != "running":
            raise ValueError("No active execution")
        return world.task_completed()
    if op == "finish":
        if phase != "running":
            raise ValueError("No active execution")
        phase = "finished"
        world.save_logs()
        return {"execution_locked": True}
    if op == "evaluate":
        if phase != "finished":
            raise ValueError("Evaluation must follow finish; it is never an agent tool")
        result = world.evaluate(suppress_errors=True).to_dict(stats_only=False)
        phase = "graded"
        return result
    if op == "export":
        if phase != "graded":
            raise ValueError("Artifacts can only be exported after grading")
        import base64
        import io
        from pathlib import Path
        import zipfile
        buffer = io.BytesIO()
        directory = Path(world.output_directory)
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(directory.rglob("*")):
                if path.is_file() and not path.is_symlink():
                    archive.write(path, str(Path("tasks") / world.task_id / path.relative_to(directory)))
        return {"zip_base64": base64.b64encode(buffer.getvalue()).decode("ascii")}

    if op == "close":
        if world is not None:
            world.close()
        phase = "closed"
        return None
    raise ValueError("Unsupported operation")


def main():
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                output = handle(request)
            response = {"ok": True, "output": output}
        except Exception as exc:
            response = {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
        print(json.dumps(response, ensure_ascii=False, default=str), flush=True)
        if phase == "closed":
            break


if __name__ == "__main__":
    main()
