"""点右上角运行：不跑 Docker，只打开已经保存的 Phoenix Trace。"""
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
    from scripts.eval_demo import safe_summary, select_result

    runtime = PhoenixServerRuntime()
    try:
        print("正在启动 Phoenix；不会运行 Docker，也不会调用模型……")
        runtime.start()
        result = safe_summary(select_result())
        browser = open_chrome()
        print_ready(result=result, browser=browser)
        wait_and_stop(runtime)
    except Exception:
        runtime.stop()
        raise


if __name__ == "__main__":
    main()
