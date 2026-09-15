"""点右上角运行：重新验收 Docker/AppWorld/grader，然后打开 Phoenix。

安全边界：外部模型调用固定为 0；不会读取模型密钥；不会运行 SkillOpt。
"""
from __future__ import annotations

import sys

from _demo_helpers import (
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
    from evals.appworld.acceptance import main as run_acceptance
    from scripts.eval_demo import safe_summary, select_result

    runtime = PhoenixServerRuntime()
    try:
        print("正在启动 Phoenix……")
        runtime.start()
        print("正在运行零模型 Docker/AppWorld/grader 验收，通常需要几十秒……")
        run_acceptance()
        result = safe_summary(select_result())
        browser = open_chrome()
        print_ready(result=result, browser=browser)
        wait_and_stop(runtime)
    except Exception:
        runtime.stop()
        raise


if __name__ == "__main__":
    main()
