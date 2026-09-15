"""点右上角运行：生成一条完整、零费用的 Trace 教学样例并打开 Phoenix。"""
from __future__ import annotations

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
    from evals.appworld.learning_preview import generate_learning_trace

    runtime = PhoenixServerRuntime()
    try:
        print("正在生成完整 Trace 教学样例；不会调用任何外部模型……")
        runtime.start()
        result = generate_learning_trace()
        browser = open_chrome()
        print_ready(result=result, browser=browser, paid=False)
        wait_and_stop(runtime)
    except Exception:
        runtime.stop()
        raise


if __name__ == "__main__":
    main()
