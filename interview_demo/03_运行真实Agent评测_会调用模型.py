"""点右上角运行：真实模型 + PersonalOps + AppWorld + 官方 grader。

当前固定为一条 Train calibration：
- 最多 24 次模型调用；
- 单次最多 8192 输出 Token；
- provider retry 为 0；
- 会读取项目 .env 中的模型配置并产生真实费用；
- 不会自动开始第二题或批量运行。
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from _demo_helpers import (
    ROOT,
    ensure_project_python,
    open_chrome,
    prepare_imports,
    print_ready,
    wait_and_stop,
)


def main():
    ensure_project_python()
    prepare_imports()

    from phoenix_runtime import PhoenixServerRuntime
    from scripts.eval_demo import result_files, safe_summary

    runtime = PhoenixServerRuntime()
    previous = {path.resolve() for path in result_files()}
    try:
        print("正在启动 Phoenix……")
        runtime.start()
        print("正在运行一条真实 Train calibration。")
        print("这会调用外部模型；最多 24 次调用，不会自动运行第二题。")
        completed = subprocess.run(
            [
                sys.executable,
                "-m", "evals.appworld.run",
                "--split", "train",
                "--task-index", "0",
                "--purpose", "calibration",
                "--max-calls", "24",
                "--max-output-tokens", "8192",
                "--wall-timeout", "600",
                "--phoenix",
            ],
            cwd=ROOT,
        )
        new_results = [
            path for path in result_files()
            if path.resolve() not in previous
            and path.parent.name.startswith("aw_")
        ]
        if not new_results:
            raise RuntimeError(
                "真实评测没有生成新的 result.json，退出码为 "
                + str(completed.returncode)
            )
        result = safe_summary(new_results[0])
        browser = open_chrome()
        print_ready(result=result, browser=browser, paid=True)
        wait_and_stop(runtime)
        if completed.returncode:
            raise SystemExit(completed.returncode)
    except Exception:
        runtime.stop()
        raise


if __name__ == "__main__":
    main()
