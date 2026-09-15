"""Run real Worker and Reviewer graphs against an existing offline Code image.

No auto-build, daemon start, provider calls, private account access or sends.
Own synthetic fixtures and containers/volumes only; always clean up the pair.
"""

import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from test_local_ocr import pdf_bytes, image_bytes
from scripts.probe_file_routes import SequenceModel
from langchain_core.messages import ToolMessage
from langchain.agents.middleware import AgentMiddleware, hook_config
from workers.code_worker import create_code_worker
from workers.code_reviewer import create_code_reviewer
from workers.docker_sandbox import (
    DockerCLI,
    CodeSandboxManager,
    CodeSandboxPolicy,
    CODE_SANDBOX_IMAGE,
)


TEST_PROGRAM = r"""
import json, pathlib, socket, subprocess
import pytest
BASE = pathlib.Path(__file__).parent

def test_python_and_schema():
    import pandas, numpy, scipy, matplotlib, pydantic, jsonschema, hypothesis, fitz, pypdf, pdfplumber, reportlab, docx, pptx, openpyxl, xlsxwriter, fastapi, httpx, lxml, bs4, yaml, jinja2, sympy
    jsonschema.validate({'count': 2}, {'type': 'object', 'properties': {'count': {'type': 'integer'}}, 'required': ['count']})
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({'count': 'bad'}, {'type': 'object', 'properties': {'count': {'type': 'integer'}}})
    assert sympy.expand((sympy.Symbol('x') + 1)**2) == sympy.Symbol('x')**2 + 2*sympy.Symbol('x') + 1

def test_network_is_offline():
    with pytest.raises(OSError):
        socket.create_connection(('1.1.1.1', 443), timeout=2)

def test_office_export():
    from docx import Document
    import fitz
    document = Document()
    document.add_paragraph('Offline Office roundtrip')
    source = BASE / 'office.docx'
    document.save(source)
    p = subprocess.run(['libreoffice', '-env:UserInstallation=file:///tmp/lo-probe', '--headless', '--convert-to', 'pdf', '--outdir', str(BASE), str(source)], capture_output=True, timeout=50)
    assert p.returncode == 0, p.stderr
    with fitz.open(BASE / 'office.pdf') as doc:
        assert 'Offline Office roundtrip' in doc[0].get_text()

def test_local_browser_and_node():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=['--disable-dev-shm-usage'])
        try:
            page = browser.new_page()
            page.set_content('<button onclick="this.textContent=\'clicked\'">Press</button>')
            page.get_by_role('button').click()
            assert page.get_by_role('button').inner_text() == 'clicked'
            page.screenshot(path=str(BASE / 'browser.png'))
        finally:
            browser.close()
    p = subprocess.run(['node', '-e', "const {JSDOM}=require('jsdom'); const React=require('react'); console.log(new JSDOM('<p>ready</p>').window.document.querySelector('p').textContent);"], capture_output=True, text=True, timeout=20)
    assert p.returncode == 0 and 'ready' in p.stdout, p.stderr
    assert subprocess.run(['tsc', '--version'], capture_output=True).returncode == 0

def test_system_tools_and_chinese_ocr_data():
    import shutil
    assert all(shutil.which(name) for name in ['pdftotext', 'pdftoppm', 'pandoc', 'tesseract', 'ruff', 'jq', 'rg', 'sqlite3', 'gcc', 'g++'])
    result = subprocess.run(['tesseract', '--list-langs'], capture_output=True, text=True)
    assert 'chi_sim' in result.stdout and 'eng' in result.stdout
"""


class StopAfterReads(AgentMiddleware):
    """Stop this bounded plumbing probe; never fabricate a business submission."""
    @hook_config(can_jump_to=["end"])
    def before_model(self, state, runtime):
        calls = {m.tool_call_id for m in state.get("messages", []) if isinstance(m, ToolMessage)}
        if {"route-0", "route-1"}.issubset(calls):
            return {"jump_to": "end"}


async def main():
    cli = DockerCLI()
    # A WSL distribution may just have resumed; wait for its already configured
    # Docker service without treating a missing startup socket as a file failure.
    for _ in range(30):
        if cli.run(["version", "--format", "{{.Server.Version}}"], timeout=10).returncode == 0:
            break
        time.sleep(1)
    cli.require_success(
        ["version", "--format", "{{.Server.Version}}"],
        operation="existing Docker check",
    )
    cli.require_success(
        ["image", "inspect", CODE_SANDBOX_IMAGE], operation="existing image check"
    )
    key = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8]
    root = Path(".agent/offline-code-document-probes").resolve() / key
    handoff = root / "handoff"
    handoff.mkdir(parents=True)
    data = pdf_bytes([image_bytes(), None])
    (handoff / "fixture.pdf").write_bytes(data)
    findings = {
        "evidence": str(root),
        "provider_calls": 0,
        "fixture": "synthetic bilingual PDF; test harness handoff, not a claimed model review",
    }
    manager = CodeSandboxManager(CodeSandboxPolicy(auto_build=False), cli=cli)
    pair = None
    try:
        pair = manager.create_pair("offline-doc-" + key, handoff_root=handoff)
        for role, factory in [
            ("WORKER", create_code_worker),
            ("REVIEWER", create_code_reviewer),
        ]:
            if role == "REVIEWER":
                pair = manager.handoff(pair, role)
            backend = (
                manager.worker_backend(pair)
                if role == "WORKER"
                else manager.reviewer_backend(pair)
            )
            workdir = "/workspace" if role == "WORKER" else "/review"
            installed = backend.download_files(["/opt/personalops/read_document.py"])[0]
            assert (
                installed.content
                == Path("docker/code-agent/read_document.py").read_bytes()
            ), "Rebuild the image after reader changes"
            if role == "WORKER":
                assert (
                    backend.upload_files([("/workspace/private.pdf", data)])[0].error
                    is None
                )
            uploaded = backend.upload_files(
                [(workdir + "/test_basics.py", TEST_PROGRAM.encode())]
            )
            assert all(item.error is None for item in uploaded), uploaded
            check = backend.execute(
                f"python -m pytest -q {workdir}/test_basics.py", timeout=120
            )
            (root / (role.lower() + "-tests.txt")).write_text(check.output, "utf-8")
            assert check.exit_code == 0, check.output
            model = SequenceModel(
                actions=[
                    {
                        "tool": "read_file",
                        "args": {"file_path": "/handoff/fixture.pdf", "limit": 1},
                    },
                    {
                        "tool": "read_file",
                        "args": {
                            "file_path": "/workspace/private.pdf",
                            "offset": 1,
                            "limit": 1,
                        },
                    },
                ]
            )
            graph = factory(model, tools=[], backend=backend, middleware=[StopAfterReads()])
            state = await graph.ainvoke(
                {
                    "event_id": key,
                    "worker_id": role.lower() + key,
                    "step_id": "1",
                    "messages": [
                        {
                            "role": "user",
                            "content": "Read the synthetic shared PDF page 1 locally",
                        }
                    ],
                }
            )
            messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]
            (root / (role.lower() + "-tool-results.json")).write_text(
                json.dumps([m.model_dump(mode="json") for m in messages], ensure_ascii=False, indent=2), "utf-8"
            )
            assert messages[0].content.lstrip().startswith("{"), str(messages[0].content)[:2000]
            result = json.loads(messages[0].content)
            private_read = json.loads(messages[-1].content)
            assert (
                private_read["start_page"] == 2
                and "Native heading" in private_read["text"]
            )
            (root / (role.lower() + "-read.json")).write_text(
                json.dumps(result, ensure_ascii=False, indent=2), "utf-8"
            )
            assert (
                "Native heading" in result["text"]
                and "1200" in result["text"]
                and "中文" in result["text"]
            ), result
            assert result["next_page"] == 2 and not result["warnings"], result
            assert all(isinstance(m.content, str) for m in messages)
            files = backend.download_files(["/handoff/fixture.pdf"])
            assert (
                hashlib.sha256(files[0].content).hexdigest()
                == hashlib.sha256(data).hexdigest()
            )
            assert backend.upload_files([("/handoff/denied.txt", b"denied")])[0].error
            inspected = json.loads(
                cli.require_success(
                    [
                        "inspect",
                        pair.worker_container
                        if role == "WORKER"
                        else pair.reviewer_container,
                    ],
                    operation="inspect boundary",
                ).stdout
            )[0]
            assert inspected["HostConfig"]["NetworkMode"] == "none"
            findings[role.lower()] = {
                "pytest": check.output.strip(),
                "real_read_file_graph": True,
                "bilingual_pdf_ocr": True,
                "raw_pdf_model_blocks": False,
                "network": "none",
                "private_candidate_page_2_read": True,
                "uid": inspected["Config"]["User"],
                "handoff_hash_matches": True,
                "handoff_write_denied": True,
            }
            (root / "summary.json").write_text(
                json.dumps(findings, ensure_ascii=False, indent=2), "utf-8"
            )
            print(json.dumps({role: findings[role.lower()]}, ensure_ascii=False), flush=True)
    finally:
        if pair:
            manager.cleanup(pair)
            cli.require_success(["version", "--format", "{{.Server.Version}}"], operation="cleanup verification requires a reachable daemon")
            findings["owned_resources_removed"] = all(
                manager._inspect_owned_resource(kind, name) is None
                for kind, name in [
                    ("container", pair.worker_container),
                    ("container", pair.reviewer_container),
                    ("volume", pair.candidate_volume),
                    ("volume", pair.review_volume),
                ]
            )
        (root / "summary.json").write_text(
            json.dumps(findings, ensure_ascii=False, indent=2), "utf-8"
        )
        print(
            json.dumps(
                {
                    "evidence": str(root),
                    "cleanup": findings.get("owned_resources_removed"),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    asyncio.run(main())
