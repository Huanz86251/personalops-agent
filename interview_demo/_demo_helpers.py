from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import webbrowser


ROOT = Path(__file__).resolve().parents[1]
AGENT_PYTHON = Path(r"C:\ProgramData\anaconda3\envs\agent\python.exe")
PHOENIX_URL = "http://127.0.0.1:6007"

for stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="replace")


def ensure_project_python():
    """Relaunch this clicked script with the pinned project interpreter."""
    try:
        current = Path(sys.executable).resolve()
        expected = AGENT_PYTHON.resolve()
    except OSError:
        current = Path(sys.executable)
        expected = AGENT_PYTHON
    if current == expected:
        return
    if not AGENT_PYTHON.is_file():
        raise FileNotFoundError(
            "找不到项目解释器：" + str(AGENT_PYTHON)
        )
    completed = subprocess.run(
        [str(AGENT_PYTHON), str(Path(sys.argv[0]).resolve()), "--project-python"],
        cwd=ROOT,
    )
    raise SystemExit(completed.returncode)


def prepare_imports():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def open_chrome(url=PHOENIX_URL):
    """Prefer Chrome, then fall back to the Windows default browser."""
    if os.getenv("PERSONALOPS_DEMO_NO_BROWSER") == "1":
        return {"opened": False, "browser": "disabled for automated check", "url": url}
    candidates = [
        Path(os.environ.get("PROGRAMFILES", "")) / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Google/Chrome/Application/chrome.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            subprocess.Popen(
                [str(candidate), "--new-window", url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return {"opened": True, "browser": "Chrome", "url": url}
    return {
        "opened": bool(webbrowser.open_new(url)),
        "browser": "Windows default browser",
        "url": url,
    }


def print_ready(*, result=None, browser=None, paid=False):
    print()
    print("=" * 68)
    if paid:
        print("真实 Agent 评测已经结束：本次调用了外部模型，费用以 provider 账单为准。")
    else:
        print("演示已经准备好：没有调用任何外部模型，也没有产生模型费用。")
    phoenix = (result or {}).get("phoenix") or {}
    project = phoenix.get("project") or "personalops-eval-infrastructure"
    result_kind = (result or {}).get("kind")
    is_agent_trial = result_kind == "appworld_agent_evaluation"
    is_guided_preview = result_kind == "guided_trace_preview"
    if is_agent_trial:
        trace_name = "EVAL / AppWorld trial"
    elif is_guided_preview:
        trace_name = "EVAL / Guided learning trace"
    else:
        trace_name = "EVAL / Infrastructure acceptance"
    print("浏览器地址：", PHOENIX_URL)
    print("Phoenix 项目：", project)
    print("Trace 名称：", trace_name)
    print()
    print("在 Phoenix 中：")
    print("1. 进入 " + project + " 项目。")
    print("2. 打开最新的 " + trace_name + "。")
    if is_agent_trial or is_guided_preview:
        print("3. 按 INPUT、RUN 内的 PLAN/EXECUTE/REPORT/REVIEW、GRADE、SAVE 查看。")
    else:
        print("3. 展开 PREPARE、RUN、两个 TOOL、GRADE。")
    if result:
        print()
        print("本次安全摘要：")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    if browser and not browser.get("opened"):
        print()
        print("浏览器未自动打开，请手动复制上面的地址。")
    print("=" * 68)
    print()


def wait_and_stop(runtime):
    if os.getenv("PERSONALOPS_DEMO_NO_WAIT") == "1":
        runtime.stop()
        return
    try:
        input("演示结束后，在这里按 Enter 关闭本脚本启动的 Phoenix：")
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        runtime.stop()
