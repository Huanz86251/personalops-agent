"""Native, on-demand tools. No MCP transport, daemon, or model API.

File tools use checkpoint files and the run's read-only /handoff namespace.
They never accept a host filesystem path. Outputs are checkpoint artifacts,
so the existing submission/review/publication path still owns delivery.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import importlib
import io
import ipaddress
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import zipfile
from functools import wraps
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal
from urllib.parse import urljoin, urlsplit

from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import ToolException, tool
from langgraph.types import Command
from pydantic import Field

DEPENDENCY_ROOT = Path(__file__).resolve().parents[1] / ".agent" / "local-tool-deps"
OCR_DEPENDENCY_ROOT = Path(__file__).resolve().parents[1] / ".agent" / "ocr-deps"
OCR_WORKER_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "local_ocr_worker.py"
)
OCR_TIMEOUT_SECONDS = 180
from file_limits import TASK_FILE_MAX_BYTES as MAX_BYTES
MAX_TEXT = 12000
_capacity = threading.BoundedSemaphore(1)


def _dependency(name):
    # Append, never override the running Agent's already installed libraries.
    directory = str(DEPENDENCY_ROOT)
    if directory not in sys.path and DEPENDENCY_ROOT.is_dir():
        sys.path.append(directory)
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise ToolException(
            "Local dependency unavailable; run scripts/setup_local_tools.py."
        ) from error


def _bounded(function):
    @wraps(function)
    def invoke(*args, **kwargs):
        if not _capacity.acquire(blocking=False):
            raise ToolException(
                "Another local conversion/check is running; try after it finishes."
            )
        try:
            return function(*args, **kwargs)
        except ToolException:
            raise
        except Exception as error:
            raise ToolException(
                f"{type(error).__name__}: {str(error)[:1200]}"
            ) from error
        finally:
            _capacity.release()

    return invoke


def _virtual(path: str) -> str:
    if not path or "\\" in path or ":" in path or "\x00" in path:
        raise ValueError("Use a virtual file path, not a host path or URL.")
    value = PurePosixPath("/" + path.lstrip("/"))
    if ".." in value.parts:
        raise ValueError("Parent path traversal is not allowed.")
    return str(value)


def _read(path: str, runtime: ToolRuntime) -> bytes:
    key = _virtual(path)
    record = None if key.startswith(("/downloads/", "/handoff/")) else (runtime.state.get("files") or {}).get(key)
    if record is not None:
        from deepagents.backends.utils import file_data_to_string

        text = file_data_to_string(record)
        if len(text) > MAX_BYTES * 2:
            raise ValueError("File exceeds the local tool size limit.")
        data = (
            base64.b64decode(text, validate=True)
            if record.get("encoding") == "base64"
            else text.encode("utf-8")
        )
    elif key.startswith("/downloads/"):
        from task_files import read_download
        return _checked_bytes(read_download(key, runtime.state))
    elif key.startswith("/handoff/"):
        from run_workspace import RUN_WORKSPACE_ROOT, stable_storage_key

        state = runtime.state
        run_id = str(state.get("event_id") or state.get("planning_run_id") or "")
        if not run_id:
            raise ValueError("Missing run identity for handoff access.")
        storage = Path(state.get("run_storage_root") or RUN_WORKSPACE_ROOT)
        root = (
            storage / stable_storage_key(run_id, prefix="run") / "handoff"
        ).resolve()
        actual = (root / key.removeprefix("/handoff/")).resolve()
        actual.relative_to(root)
        if actual.stat().st_size > MAX_BYTES:
            raise ValueError("File exceeds 20 MiB.")
        data = actual.read_bytes()
    else:
        raise ValueError(
            "File not in this Worker checkpoint or /handoff. Import it through the task file workflow first."
        )
    return _checked_bytes(data)


def _checked_bytes(data: bytes) -> bytes:
    if len(data) > MAX_BYTES:
        raise ValueError("File exceeds 20 MiB.")
    # Office documents are ZIP files; bound expanded input before conversion.
    if data.startswith(b"PK"):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
            if (
                len(entries) > 10000
                or sum(e.file_size for e in entries) > 100 * 1024 * 1024
            ):
                raise ValueError("Expanded archive exceeds local limits.")
    return data


def _save(
    path: str, data: bytes, runtime: ToolRuntime, *, overwrite=False, text=False
) -> Command:
    from deepagents.backends.utils import create_file_data

    key = _virtual(path)
    if not key.startswith("/artifacts/"):
        raise ValueError(
            "Write outputs under /artifacts/; shared /handoff is read-only."
        )
    if not overwrite and key in (runtime.state.get("files") or {}):
        raise ValueError(
            "Output exists; choose a new name or explicitly set overwrite=true."
        )
    if len(data) > MAX_BYTES:
        raise ValueError("Output exceeds 20 MiB.")
    content = data.decode("utf-8") if text else base64.b64encode(data).decode("ascii")
    result = {
        "path": key,
        "bytes": len(data),
        "status": "created",
        "next_step": "Submit this WORKSPACE_FILE artifact for review; it is not published yet.",
    }
    if text:
        result.update(preview=content[:MAX_TEXT], truncated=len(content) > MAX_TEXT)
    return Command(
        update={
            "files": {
                key: create_file_data(content, encoding="utf-8" if text else "base64")
            },
            "messages": [
                ToolMessage(
                    content=json.dumps(result, ensure_ascii=False),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


def _reading_process(data: bytes, name: str, **options) -> dict:
    """One child owns all document pages; run() kills and reaps on deadline."""
    if not (OCR_DEPENDENCY_ROOT / "rapidocr").is_dir():
        raise ValueError("OCR runtime unavailable; run scripts/setup_ocr.py first.")
    env = {
        key: value
        for key, value in os.environ.items()
        if key.upper() in {"SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP"}
    }
    env.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
    with tempfile.TemporaryDirectory(prefix="personalops-reading-") as directory:
        root = Path(directory)
        source, request, output = (
            root / "input.bin",
            root / "request.json",
            root / "result.json",
        )
        source.write_bytes(data)
        request.write_text(
            json.dumps({"input_path": str(source), "name": name, **options}),
            encoding="utf-8",
        )
        with (root / "worker.log").open("wb") as log:
            try:
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-I",
                        "-S",
                        str(OCR_WORKER_SCRIPT),
                        str(request),
                        str(output),
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=log,
                    env=env,
                    timeout=OCR_TIMEOUT_SECONDS,
                    check=False,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except subprocess.TimeoutExpired as error:
                raise ValueError(
                    "Attachment reading timed out; OCR process released. Read a smaller page range."
                ) from error
        if completed.returncode or not output.is_file():
            raise ValueError(
                "Attachment reader failed; its process has exited. Inspect the OCR installation."
            )
        if output.stat().st_size > MAX_BYTES:
            raise ValueError("Reading result too large; request fewer pages.")
        result = json.loads(output.read_text(encoding="utf-8"))
        if "error" in result:
            raise ValueError(result["error"])
        result["summary"]["process_released"] = True
        return result


def _read_attachment_result(path, output_path, runtime, **options):
    output_path = _virtual(output_path)
    if not output_path.startswith("/artifacts/") or not output_path.endswith(".md"):
        raise ValueError("Use a new /artifacts/name.md output path.")
    detail_path = output_path.removesuffix(".md") + ".json"
    for key in (output_path, detail_path):
        if key in (runtime.state.get("files") or {}):
            raise ValueError("Output exists; choose a new name.")
    name = path
    if path.startswith("/downloads/"):
        for record in runtime.state.get("worker_downloaded_artifacts") or []:
            if path == "/downloads/" + record["candidate_id"]:
                name = record["filename"]
                break
    result = _reading_process(_read(path, runtime), name, **options)
    command = _save(output_path, result["markdown"].encode("utf-8"), runtime, text=True)
    details = _save(
        detail_path,
        json.dumps(result, ensure_ascii=False).encode("utf-8"),
        runtime,
        text=True,
    )
    command.update["files"].update(details.update["files"])
    message = command.update["messages"][0]
    metadata = json.loads(message.content)
    summary = dict(result["summary"])
    # Keep uncertainty visible without flooding the model with per-region errors.
    summary["warning_count"] = len(summary["warnings"])
    summary["warnings"] = summary["warnings"][:12]
    metadata.update(
        reading=summary,
        details_path=detail_path,
        next_step="Read the saved Markdown/JSON as needed. Report reading limitations; do not infer unseen image content.",
    )
    message.content = json.dumps(metadata, ensure_ascii=False)
    return command


TaskInput = Annotated[
    str,
    Field(
        description="Task file path: checkpoint, /handoff/... or this Worker's /downloads/<candidate_id>. No host paths or URLs.",
        min_length=1,
        max_length=512,
    ),
]
ReadingOutput = Annotated[
    str,
    Field(
        description="New /artifacts/name.md; full Markdown and a matching .json with OCR boxes/scores are saved.",
        min_length=1,
        max_length=512,
    ),
]


@tool
@_bounded
def attachment_to_text(
    path: TaskInput,
    output_path: ReadingOutput,
    runtime: ToolRuntime,
    ocr: Annotated[
        Literal["auto", "force", "off"],
        Field(
            description="auto reads embedded image text; force OCRs entire PDF pages; off skips OCR and reports unread images."
        ),
    ] = "auto",
    start_page: Annotated[
        int,
        Field(
            description="First PDF page/PPTX slide, 1-based. Other formats require 1.",
            ge=1,
        ),
    ] = 1,
    max_pages: Annotated[
        int,
        Field(
            description="Maximum PDF pages/PPTX slides this call. Follow next_page to continue.",
            ge=1,
            le=20,
        ),
    ] = 20,
) -> Command:
    """READ PDF/PPTX/DOCX/XLSX/images into Markdown, automatically including local Chinese/English OCR.

    Identifies binary content even with a .bin suffix. Native text and OCR are labeled
    with source locations. PDF/PPTX are paged; DOCX/XLSX images are an appendix.
    Read returned warnings: OCR does not describe scenes or guarantee table structure.
    One CPU OCR engine is reused across this document call and released on completion.
    """
    return _read_attachment_result(
        path,
        output_path,
        runtime,
        operation="document",
        ocr=ocr,
        start_page=start_page,
        max_pages=max_pages,
    )


@tool
@_bounded
def ocr_image(
    path: TaskInput,
    output_path: ReadingOutput,
    runtime: ToolRuntime,
    crop: Annotated[
        list[int] | None,
        Field(
            description="Optional [left, top, right, bottom] crop in pixels after image orientation; within image bounds.",
            min_length=4,
            max_length=4,
        ),
    ] = None,
) -> Command:
    """Read Chinese/English TEXT from a task image, screenshot or image table using local CPU OCR.

    Choose an existing image path, including a returned /downloads/<candidate_id>.
    Returns text, pixel boxes and recognition scores in saved Markdown/JSON.
    No scene descriptions; table structure is not guaranteed. For PDF/PPTX use
    attachment_to_text, which automatically applies the same OCR internally.
    This standalone call releases its OCR process immediately after completion.
    """
    return _read_attachment_result(
        path, output_path, runtime, operation="image", crop=crop
    )


@tool
@_bounded
def convert_document(path: str, output_path: str, runtime: ToolRuntime) -> Command:
    """Convert task-local Markdown/TXT/HTML/DOCX to DOCX/HTML/Markdown using local Pandoc.

    No server. A short-lived Pandoc command runs with a 30-second timeout and
    --sandbox. Input is a checkpoint or /handoff file; output is /artifacts/.
    PDF output is not enabled (requires an additional rendering engine).
    """
    formats = {".md": "markdown", ".txt": "markdown", ".html": "html", ".docx": "docx"}
    source = formats.get(PurePosixPath(path).suffix.lower())
    target = formats.get(PurePosixPath(output_path).suffix.lower())
    if source is None or target is None:
        raise ValueError("Supported formats: .md, .txt, .html, .docx")
    binary = _dependency("pypandoc").get_pandoc_path()
    with tempfile.TemporaryDirectory(prefix="personalops-convert-") as directory:
        src = Path(directory) / ("input" + PurePosixPath(path).suffix)
        dst = Path(directory) / ("output" + PurePosixPath(output_path).suffix)
        src.write_bytes(_read(path, runtime))
        completed = subprocess.run(
            [
                binary,
                "--sandbox",
                "--from",
                source,
                "--to",
                target,
                str(src),
                "-o",
                str(dst),
            ],
            capture_output=True,
            check=False,
            timeout=30,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if completed.returncode:
            raise ValueError(completed.stderr.decode("utf-8", errors="replace")[:1200])
        if dst.stat().st_size > MAX_BYTES:
            raise ValueError("Converted document exceeds 20 MiB.")
        return _save(output_path, dst.read_bytes(), runtime, text=target != "docx")


def _workbook(path, runtime, *, data_only=False):
    if PurePosixPath(path).suffix.lower() != ".xlsx":
        raise ValueError("Only .xlsx workbooks are supported.")
    return _dependency("openpyxl").load_workbook(
        io.BytesIO(_read(path, runtime)), data_only=data_only
    )


def _save_workbook(workbook, output_path, runtime, overwrite=False):
    if not output_path.endswith(".xlsx"):
        raise ValueError("Workbook output must end in .xlsx.")
    buffer = io.BytesIO()
    workbook.save(buffer)
    workbook.close()
    return _save(output_path, buffer.getvalue(), runtime, overwrite=overwrite)


@tool
@_bounded
def spreadsheet_read(
    path: str,
    runtime: ToolRuntime,
    sheet: str = "",
    start_row: int = 1,
    max_rows: int = 40,
    max_columns: int = 20,
) -> dict:
    """Read an XLSX sheet in bounded pages. Empty sheet lists sheet names only.

    Formulas are returned as formulas, not recalculated values. Input must be
    a checkpoint or /handoff file. At most 100 rows and 50 columns per call.
    """
    if start_row < 1 or not 1 <= max_rows <= 100 or not 1 <= max_columns <= 50:
        raise ValueError("Invalid row/column bounds.")
    workbook = _workbook(path, runtime)
    try:
        if not sheet:
            return {"sheets": workbook.sheetnames}
        ws = workbook[sheet]
        rows = []
        size = 0
        for row in ws.iter_rows(
            min_row=start_row,
            max_row=min(ws.max_row, start_row + max_rows - 1),
            max_col=min(max_columns, ws.max_column),
            values_only=True,
        ):
            values = [
                v if isinstance(v, (str, int, float, bool, type(None))) else str(v)
                for v in row
            ]
            rendered = json.dumps(values, ensure_ascii=False)
            if size + len(rendered) > MAX_TEXT:
                if not rows:
                    raise ValueError(
                        "One row exceeds the response limit; request fewer columns."
                    )
                break
            rows.append(values)
            size += len(rendered)
        next_row = start_row + len(rows)
        return {
            "rows": rows,
            "next_row": next_row if next_row <= ws.max_row else None,
            "total_rows": ws.max_row,
            "total_columns": ws.max_column,
            "columns_truncated": ws.max_column > max_columns,
            "formula_values_recalculated": False,
        }
    finally:
        workbook.close()


@tool
@_bounded
def spreadsheet_write(
    output_path: str,
    sheet: str,
    rows: list[list[str | int | float | bool | None]],
    runtime: ToolRuntime,
    source_path: str = "",
    start_cell: str = "A1",
    overwrite: bool = False,
) -> Command:
    """Create or update an XLSX workbook with up to 5000 cells.

    Optional source_path reads an existing task file. Output must be /artifacts/*.xlsx.
    Strings beginning '=' are Excel formulas; this tool writes but does not calculate them.
    overwrite=true is needed to replace an existing checkpoint output.
    """
    if not rows or sum(len(r) for r in rows) > 5000 or len(json.dumps(rows)) > 200000:
        raise ValueError("Supply 1..5000 cells and at most 200000 input characters.")
    openpyxl = _dependency("openpyxl")
    wb = _workbook(source_path, runtime) if source_path else openpyxl.Workbook()
    if not source_path:
        wb.active.title = sheet
    ws = wb[sheet] if sheet in wb.sheetnames else wb.create_sheet(sheet)
    row_num, col_num = openpyxl.utils.cell.coordinate_to_tuple(start_cell)
    if row_num + len(rows) - 1 > 1048576 or col_num + max(map(len, rows)) - 1 > 16384:
        raise ValueError("Data exceeds XLSX row or column limits.")
    for r, values in enumerate(rows, start=row_num):
        for c, value in enumerate(values, start=col_num):
            ws.cell(r, c, value)
    return _save_workbook(wb, output_path, runtime, overwrite)


@tool
@_bounded
def spreadsheet_format(
    path: str,
    output_path: str,
    sheet: str,
    cell_range: str,
    runtime: ToolRuntime,
    bold: bool = False,
    number_format: str = "General",
    column_width: float = 18,
    overwrite: bool = False,
) -> Command:
    """Format at most 5000 XLSX cells; save a reviewed-task artifact, with no Excel server."""
    op = _dependency("openpyxl")
    a, b, c, d = op.utils.cell.range_boundaries(cell_range)
    if (
        not all((a, b, c, d))
        or min(a, b) < 1
        or (c - a + 1) * (d - b + 1) > 5000
        or not 1 <= column_width <= 100
    ):
        raise ValueError("Use a bounded cell range (max 5000 cells) and width 1..100.")
    wb = _workbook(path, runtime)
    ws = wb[sheet]
    from copy import copy

    for row in ws.iter_rows(min_row=b, max_row=d, min_col=a, max_col=c):
        for cell in row:
            font = copy(cell.font)
            font.bold = bold
            cell.font = font
            cell.number_format = number_format
    for col in range(a, c + 1):
        ws.column_dimensions[op.utils.get_column_letter(col)].width = column_width
    return _save_workbook(wb, output_path, runtime, overwrite)


@tool
@_bounded
def spreadsheet_chart(
    path: str,
    output_path: str,
    sheet: str,
    data_range: str,
    runtime: ToolRuntime,
    kind: Literal["bar", "line", "pie"] = "bar",
    title: str = "Chart",
    anchor: str = "E2",
    overwrite: bool = False,
) -> Command:
    """Add an Excel chart: first row contains headers, first column categories, others numeric series."""
    op = _dependency("openpyxl")
    a, b, c, d = op.utils.cell.range_boundaries(data_range)
    if (
        not all((a, b, c, d))
        or min(a, b) < 1
        or c <= a
        or d <= b
        or (c - a + 1) * (d - b + 1) > 5000
    ):
        raise ValueError(
            "Use a rectangular range with headers, categories and values, max 5000 cells."
        )
    if kind == "pie" and c - a != 1:
        raise ValueError("Pie charts require exactly one numeric series.")
    op.utils.cell.coordinate_to_tuple(anchor)
    wb = _workbook(path, runtime)
    ws = wb[sheet]
    chart_module = _dependency("openpyxl.chart")
    chart = {
        "bar": chart_module.BarChart,
        "line": chart_module.LineChart,
        "pie": chart_module.PieChart,
    }[kind]()
    chart.title = title[:200]
    chart.add_data(
        chart_module.Reference(ws, min_col=a + 1, max_col=c, min_row=b, max_row=d),
        titles_from_data=True,
    )
    chart.set_categories(
        chart_module.Reference(ws, min_col=a, min_row=b + 1, max_row=d)
    )
    ws.add_chart(chart, anchor)
    return _save_workbook(wb, output_path, runtime, overwrite)


def _expression(expression: str):
    """Construct SymPy objects from a small AST; never eval/sympify model text."""
    if len(expression) > 500:
        raise ValueError("Expression limit: 500 characters.")
    tree = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 80:
        raise ValueError("Expression too complex.")

    # Reject nested powers BEFORE constructing SymPy objects: a short input
    # can otherwise ask for an integer with billions of digits in-process.
    def construction_cost(node):
        if isinstance(node, ast.Constant):
            cost = len(str(node.value))
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            cost = 12 * construction_cost(node.left) + construction_cost(node.right)
        else:
            cost = max(
                1, sum(construction_cost(child) for child in ast.iter_child_nodes(node))
            )
        if cost > 128:
            raise ValueError(
                "Expression exceeds the local computation complexity limit."
            )
        return cost

    construction_cost(tree)
    sp = _dependency("sympy")
    functions = {
        name: getattr(sp, name)
        for name in ("sin", "cos", "tan", "sqrt", "log", "exp", "Abs")
    }

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            if abs(node.value) > 10000:
                raise ValueError("Numeric literal exceeds 10000.")
            return sp.Rational(str(node.value))
        if isinstance(node, ast.Name) and node.id in {"x", "y", "z", "pi", "E"}:
            return {"pi": sp.pi, "E": sp.E}.get(node.id, sp.Symbol(node.id))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow) and right.is_number and abs(right) <= 12:
                return left**right
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in functions
            and len(node.args) == 1
            and not node.keywords
        ):
            value = visit(node.args[0])
            if node.func.id == "exp" and value.is_number and abs(value) > 100:
                raise ValueError("Exponential argument exceeds local bound.")
            return functions[node.func.id](value)
        raise ValueError(
            "Use arithmetic, x/y/z, pi/E, and sin/cos/tan/sqrt/log/exp/Abs only; powers <=12."
        )

    return sp, visit(tree.body)


@tool
@_bounded
def symbolic_math(
    expression: str,
    operation: Literal[
        "simplify", "expand", "differentiate", "solve", "numeric"
    ] = "simplify",
    variable: Literal["x", "y", "z"] = "x",
) -> dict:
    """Local SymPy algebra, differentiation, numeric evaluation, or polynomial roots (degree <=4).

    solve interprets expression=0. Use Python arithmetic (** for powers), no code,
    imports or attribute access. Expressions are limited to 500 characters.
    """
    sp, value = _expression(expression)
    symbol = sp.Symbol(variable)
    if operation == "solve":
        polynomial = sp.Poly(value, symbol)
        if polynomial.degree() > 4 or value.free_symbols - {symbol}:
            raise ValueError("Solve supports univariate polynomials of degree <=4.")
        result = sp.solve(value, symbol)
    elif operation == "differentiate":
        result = sp.diff(value, symbol)
    elif operation == "expand":
        result = sp.expand(value)
    elif operation == "numeric":
        result = sp.N(value, 15)
    else:
        result = sp.simplify(value)
    rendered = str(result)
    return {"result": rendered[:MAX_TEXT], "truncated": len(rendered) > MAX_TEXT}


@tool
@_bounded
def python_syntax_check(code: str) -> dict:
    """Parse Python without executing it. Does not validate API behavior or sandbox safety."""
    if len(code) > 100000:
        raise ValueError("Code limit: 100000 characters.")
    try:
        compile(ast.parse(code), "<syntax-check>", "exec")
        return {"valid_syntax": True, "executed": False}
    except SyntaxError as error:
        return {
            "valid_syntax": False,
            "line": error.lineno,
            "column": error.offset,
            "message": error.msg,
            "executed": False,
        }


@tool
@_bounded
def python_static_check(code: str) -> dict:
    """Run local Ruff E/F checks on a complete Python file; never execute or auto-fix code.

    Do not apply undefined-variable findings blindly to persistent REPL snippets.
    Uses a short-lived command (10 seconds), no service or project configuration.
    """
    if len(code) > 100000:
        raise ValueError("Code limit: 100000 characters.")
    _dependency("ruff")
    env = {
        key: value
        for key, value in os.environ.items()
        if key.upper() in {"SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP", "USERPROFILE"}
    }
    env["PYTHONPATH"] = str(DEPENDENCY_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--isolated",
            "--no-cache",
            "--select",
            "E,F",
            "--output-format",
            "json",
            "--stdin-filename",
            "input.py",
            "-",
        ],
        input=code,
        capture_output=True,
        check=False,
        text=True,
        encoding="utf-8",
        timeout=10,
        env=env,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if completed.returncode not in (0, 1):
        raise ValueError(completed.stderr[:1200])
    diagnostics = json.loads(completed.stdout)
    return {
        "diagnostics": diagnostics[:30],
        "total": len(diagnostics),
        "truncated": len(diagnostics) > 30,
        "executed": False,
    }


def _public_url(url):
    parts = urlsplit(url)
    if (
        parts.scheme not in {"http", "https"}
        or not parts.hostname
        or parts.username
        or parts.password
    ):
        raise ValueError("Use a public http(s) URL without embedded credentials.")
    port = parts.port or (443 if parts.scheme == "https" else 80)
    addresses = socket.getaddrinfo(parts.hostname, port, type=socket.SOCK_STREAM)
    if not addresses or any(
        not ipaddress.ip_address(entry[4][0]).is_global for entry in addresses
    ):
        raise ValueError(
            "Private, loopback and local network destinations are not allowed."
        )


@tool
@_bounded
def fetch_webpage(url: str, start_index: int = 0, max_length: int = 8000) -> dict:
    """Read public text, returning fetch_status, http_status, content and next_action.

    EMPTY_CONTENT suggests browser evidence; ACCESS_DENIED (403) is not success;
    RATE_LIMITED (429) must not trigger immediate retries. NETWORK_ERROR has no
    HTTP response. SUCCESS means readable text, not verified task correctness.
    PAGINATION_EXHAUSTED means the requested offset is past existing text.
    No automatic retry or browser call is performed. Maximum chunk: 12000 chars.
    """
    import httpx
    try:
        result = _fetch_webpage(url, start_index, max_length)
    except httpx.HTTPStatusError as error:
        code = error.response.status_code
        status, action = {
            401: ("AUTH_REQUIRED", "Use an authorized source; do not assume access."),
            403: ("ACCESS_DENIED", "Use another permitted public source; do not bypass access restrictions."),
            429: ("RATE_LIMITED", "Respect Retry-After and the task budget; do not immediately retry."),
        }.get(code, ("HTTP_ERROR", "Report the HTTP failure or use another source within the task budget."))
        return {"source_url": url, "final_url": str(error.response.url), "http_status": code,
                "fetch_status": status, "content": "", "total_chars": 0, "next_index": None,
                "evidence_available": False, "next_action": action,
                "retry_after": error.response.headers.get("retry-after")}
    except (httpx.RequestError, socket.gaierror) as error:
        return {"source_url": url, "final_url": None, "http_status": None,
                "fetch_status": "NETWORK_ERROR", "error_type": type(error).__name__,
                "content": "", "total_chars": 0, "next_index": None, "evidence_available": False,
                "next_action": "Network failure: report the gap or try another source within the task budget; no automatic retry."}
    if not result["content"].strip():
        empty = not result.pop("_has_text")
        result.update(fetch_status="EMPTY_CONTENT" if empty else "PAGINATION_EXHAUSTED",
                      evidence_available=False,
                      next_action="No readable text. Try the browser or another public source within the task budget; if still empty, report insufficient evidence."
                      if empty else "Requested chunk is empty. Read an earlier offset; do not treat this as a blocked or empty page.")
    else:
        result.pop("_has_text")
        result.update(fetch_status="SUCCESS", evidence_available=True,
                      next_action="Verify relevance and cite the returned evidence; HTTP success alone does not prove the task claim.")
    return result


def _fetch_webpage(url: str, start_index: int = 0, max_length: int = 8000) -> dict:
    """Fetch a public HTML/text page directly, without a browser or MCP server.

    Returns Markdown in bounded chunks (max 12000 characters). No login or JS;
    use browser tools for dynamic pages. Each call fetches again; no cache yet.
    """
    if start_index < 0 or not 1 <= max_length <= MAX_TEXT:
        raise ValueError("Invalid text range.")
    import httpx

    original = url
    with httpx.Client(
        timeout=15,
        follow_redirects=False,
        trust_env=False,
        headers={"User-Agent": "PersonalOps/1.0 (public-page reader)"},
    ) as client:
        for _ in range(6):
            _public_url(url)
            with client.stream("GET", url) as response:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise ValueError("Redirect lacks Location header.")
                    url = urljoin(url, location)
                    continue
                response.raise_for_status()
                mime = response.headers.get("content-type", "").split(";")[0].strip()
                if mime not in {
                    "text/html",
                    "application/xhtml+xml",
                    "text/plain",
                    "text/markdown",
                    "application/json",
                }:
                    raise ValueError(
                        "Not a text page; use the attachment download workflow."
                    )
                data = bytearray()
                for chunk in response.iter_bytes():
                    data.extend(chunk)
                    if len(data) > 2 * 1024 * 1024:
                        raise ValueError("Page exceeds 2 MiB.")
                text = bytes(data).decode(
                    response.encoding or "utf-8", errors="replace"
                )
                break
        else:
            raise ValueError("Too many redirects.")
    if mime in {"text/html", "application/xhtml+xml"}:
        # Bytes retain meta-charset/GBK detection. HTTPX defaults to UTF-8 even
        # when a legacy Chinese page declares its encoding only in HTML.
        declared = response.headers.get("content-type", "")
        charset = declared.split("charset=", 1)[1].split(";", 1)[0].strip(' "\'') if "charset=" in declared else None
        soup = _dependency("bs4").BeautifulSoup(bytes(data), "html.parser", from_encoding=charset)
        for element in soup(["script", "style", "noscript"]):
            element.decompose()
        text = _dependency("markdownify").markdownify(str(soup), heading_style="ATX")
    return {
        "source_url": original,
        "final_url": url,
        "http_status": response.status_code,
        "_has_text": bool(text.strip()),
        "content": text[start_index : start_index + max_length],
        "total_chars": len(text),
        "next_index": start_index + max_length
        if start_index + max_length < len(text)
        else None,
    }


LOCAL_FILE_TOOLS = [
    attachment_to_text,
    ocr_image,
    convert_document,
    spreadsheet_read,
    spreadsheet_write,
    spreadsheet_format,
    spreadsheet_chart,
]
LOCAL_COMPUTE_TOOLS = [symbolic_math, python_syntax_check, python_static_check]
LOCAL_NATIVE_TOOLS = [*LOCAL_FILE_TOOLS, *LOCAL_COMPUTE_TOOLS, fetch_webpage]
for _tool in LOCAL_NATIVE_TOOLS:
    _tool.handle_tool_error = True
