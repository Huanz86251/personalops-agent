"""Task-scoped facts derived from real tool execution, not model prose."""
from __future__ import annotations

import ast
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import json
import re
from threading import RLock

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage


ACTIVE_EXECUTION_STATE = ContextVar("active_execution_state", default=None)
_API_PATTERN = re.compile(
    r"\bapis\.([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)"
)


def _api_names(code: str) -> list[str]:
    return list(dict.fromkeys(".".join(match) for match in _API_PATTERN.findall(code)))


def _assignment_facts(code: str) -> tuple[dict[str, list[str]], list[str]]:
    """Return API-derived assignments and external names used by the snippet."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}, []
    derived: dict[str, list[str]] = {}
    assigned: set[str] = set()
    loaded: set[str] = set()

    def targets(node):
        if isinstance(node, ast.Name):
            return {node.id}
        if isinstance(node, (ast.Tuple, ast.List)):
            return set().union(*(targets(item) for item in node.elts))
        return set()

    for statement in tree.body:
        target_nodes = []
        value = None
        if isinstance(statement, ast.Assign):
            target_nodes = statement.targets
            value = statement.value
        elif isinstance(statement, ast.AnnAssign):
            target_nodes = [statement.target]
            value = statement.value
        elif isinstance(statement, ast.AugAssign):
            target_nodes = [statement.target]
            value = statement.value
        current = set().union(*(targets(node) for node in target_nodes))
        assigned.update(current)
        if value is not None:
            names = {node.id for node in ast.walk(value) if isinstance(node, ast.Name)}
            sources = []
            for node in ast.walk(value):
                if not isinstance(node, ast.Call):
                    continue
                method = node.func
                if (isinstance(method, ast.Attribute)
                        and isinstance(method.value, ast.Attribute)
                        and isinstance(method.value.value, ast.Name)
                        and method.value.value.id == "apis"):
                    sources.append(f"{method.value.attr}.{method.attr}")
            for name in names:
                sources.extend(derived.get(name, ()))
            sources = list(dict.fromkeys(sources))
            if sources:
                for name in current:
                    derived[name] = sources
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            loaded.add(node.id)
    ignored = {"apis", "print", "len", "str", "int", "float", "bool", "list",
               "dict", "set", "tuple", "range", "enumerate", "zip", "min", "max",
               "sum", "any", "all", "sorted", "True", "False", "None"}
    return {name: derived[name] for name in sorted(derived)}, sorted(loaded - assigned - ignored)


@dataclass
class ExecutionStateLedger:
    """Compact event ledger shared by Workers in one task."""

    revision: int = 0
    calls: list[dict] = field(default_factory=list)
    changes: list[dict] = field(default_factory=list)
    available: dict[str, dict] = field(default_factory=dict)
    candidates: dict[str, dict] = field(default_factory=dict)
    lock: RLock = field(default_factory=RLock, repr=False)

    def record(self, *, index: int, role: str, code: str, succeeded: bool,
               error_type: str = ""):
        apis = _api_names(code)
        derived, used = _assignment_facts(code)
        with self.lock:
            before_available = dict(self.available)
            before_candidates = dict(self.candidates)
            # A later successful call using a candidate proves that the symbol exists
            # in the persistent world, without claiming anything about its secret value.
            if succeeded:
                for name in used:
                    prior = self.candidates.pop(name, None)
                    if prior:
                        self.available[name] = {
                            **prior,
                            "status": "available",
                            "verified_by_call": index,
                        }
            destination = self.available if succeeded else self.candidates
            for name, sources in derived.items():
                if succeeded:
                    self.candidates.pop(name, None)
                destination[name] = {
                    "variable": name,
                    "source_call": index,
                    "source_apis": sources,
                    "status": "available" if succeeded else "unverified_after_error",
                }
            self.calls.append({
                "call": index,
                "role": role,
                "status": "SUCCESS" if succeeded else "ERROR",
                "apis": apis,
                **({"error_type": error_type} if error_type else {}),
            })
            self.calls = self.calls[-12:]
            self.revision += 1
            self.changes.append({
                "revision": self.revision,
                "call": dict(self.calls[-1]),
                "available_added_or_updated": [
                    value for name, value in self.available.items()
                    if before_available.get(name) != value
                ],
                "unverified_added_or_updated": [
                    value for name, value in self.candidates.items()
                    if before_candidates.get(name) != value
                ],
                "unverified_removed": sorted(
                    set(before_candidates) - set(self.candidates)
                ),
            })
            self.changes = self.changes[-32:]

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "revision": self.revision,
                "available_variables": list(self.available.values())[-16:],
                "unverified_variables": list(self.candidates.values())[-8:],
                "recent_calls": list(self.calls[-8:]),
            }

    def changes_since(self, revision: int) -> dict:
        with self.lock:
            return {
                "from_revision": revision,
                "to_revision": self.revision,
                "changes": [
                    dict(item) for item in self.changes
                    if item["revision"] > revision
                ],
            }


@contextmanager
def execution_state_scope(ledger: ExecutionStateLedger):
    token = ACTIVE_EXECUTION_STATE.set(ledger)
    try:
        yield ledger
    finally:
        ACTIVE_EXECUTION_STATE.reset(token)


class ExecutionStateMiddleware(AgentMiddleware):
    """Inject a new compact snapshot only when real execution changed it."""

    def __init__(self):
        self.last_revision = -1

    def before_model(self, state, runtime):
        ledger = ACTIVE_EXECUTION_STATE.get()
        if ledger is None:
            return None
        snapshot = ledger.snapshot()
        revision = snapshot["revision"]
        if revision <= 0 or revision == self.last_revision:
            return None
        content = snapshot if self.last_revision < 0 else ledger.changes_since(
            self.last_revision
        )
        self.last_revision = revision
        return {"messages": [HumanMessage(
            content=(
                "[执行状态账本，Harness根据真实工具调用自动生成]\n"
                + json.dumps(content, ensure_ascii=False, separators=(",", ":"))
                + "\navailable_variables可直接复用；unverified_variables来自失败调用，"
                  "必须先用成功的只读调用确认。账本不包含变量真实值，详情按需读取执行档案。"
            ),
            additional_kwargs={"execution_state_catalog": True},
        )]}

    async def abefore_model(self, state, runtime):
        return self.before_model(state, runtime)


__all__ = [
    "ExecutionStateLedger",
    "ExecutionStateMiddleware",
    "execution_state_scope",
]
