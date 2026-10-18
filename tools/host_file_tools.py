"""Bounded host-file discovery and approved import into a Worker workspace."""

from __future__ import annotations

import asyncio
import json
import os
from collections import deque
from pathlib import Path
from typing import Literal

from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command

from host_file_access import (
    HOST_FILE_ACCESS,
    HostFileAccessDenied,
)
from task_files import register_file
from task_file_imports import materialize_import


def _configured_root_lines() -> str:
    roots = tuple(HOST_FILE_ACCESS.policy.roots)
    if not roots:
        return "当前没有可读取的本机磁盘；不要猜测路径，直接报告运行环境缺口。"
    rendered = "、".join(f"`{root}`" for root in roots)
    return (
        f"当前可只读访问这些本机磁盘及其子目录：{rendered}。"
        "使用绝对路径；网络共享、符号链接和NTFS附加数据流仍不属于普通本机文件。"
    )


def refresh_host_file_tool_descriptions() -> None:
    """Rebuild model-visible descriptions from the current owner allowlist."""

    roots = _configured_root_lines()
    host_list_files.description = (
        "列出已配置主机目录中的一层文件名和元数据；不读取文件内容。"
        "支持cursor分页；目录过大时保留已找到的子集并标记truncated，不会整次报错。"
        "查找Downloads等根目录中的文件时优先调用本工具，"
        "不要先递归扫描整棵目录树。"
        + roots
    )
    host_search_files.description = (
        "在本机目录中按文件名做有界广度优先搜索；根目录文件优先，随后逐层进入子目录。"
        "即使达到扫描上限也返回已找到结果并标记truncated，不会丢弃结果后整次报错。"
        "contains为大小写不敏感包含，glob使用文件名通配符；可用max_depth控制深度。"
        + roots
    )
    host_import_file.description = (
        "把一个本机普通文件只读复制到当前Worker隔离工作区，返回稳定的/downloads引用。"
        "Code成功导入后文件已在沙盒，execution_path可直接用于execute或程序输入；无需先read_file。"
        "其他Worker保留私有副本，execution_path为空；导入失败返回明确错误。"
        "读取和复制到隔离工作区默认允许，不触发外发审批；这不等于允许发送、修改或删除原文件。"
        + roots
    )


def _require_scope(runtime: ToolRuntime) -> dict:
    state = runtime.state
    return state


def _page(items: list[dict], cursor: str | None, page_size: int) -> dict:
    if not 1 <= page_size <= 100:
        raise HostFileAccessDenied("page_size 必须在 1 到 100 之间。")
    try:
        start = int(cursor or "0")
    except ValueError as error:
        raise HostFileAccessDenied("cursor 无效。") from error
    if start < 0:
        raise HostFileAccessDenied("cursor 无效。")
    end = min(len(items), start + page_size)
    return {
        "items": items[start:end],
        "next_cursor": str(end) if end < len(items) else None,
        "returned": end - start,
        "total": len(items),
    }


@tool
async def host_list_files(
    path: str,
    runtime: ToolRuntime,
    cursor: str | None = None,
    page_size: int = 50,
    scan_limit: int = 5_000,
) -> dict:
    """列出可读本机目录中的一层文件名和元数据；不读取文件内容。支持cursor分页。"""
    _require_scope(runtime)
    resolved, _ = await asyncio.to_thread(HOST_FILE_ACCESS.policy.validate, path)
    if not resolved.is_dir():
        raise HostFileAccessDenied("host_list_files 只接受目录。")
    if not 100 <= scan_limit <= 50_000:
        raise HostFileAccessDenied("scan_limit 必须在 100 到 50000 之间。")
    entries = []
    scanned = 0
    truncated = False
    try:
        with os.scandir(resolved) as scanner:
            for entry in scanner:
                if scanned >= scan_limit:
                    truncated = True
                    break
                scanned += 1
                child = Path(entry.path)
                # Validate each child so a junction cannot be surfaced as a usable path.
                try:
                    checked, _ = HOST_FILE_ACCESS.policy.validate(str(child))
                    is_directory = entry.is_dir(follow_symlinks=False)
                    is_file = entry.is_file(follow_symlinks=False)
                    if not (is_directory or is_file):
                        continue
                    info = entry.stat(follow_symlinks=False)
                except (HostFileAccessDenied, OSError):
                    continue
                entries.append(
                    {
                        "name": checked.name,
                        "path": str(checked),
                        "kind": "directory" if is_directory else "file",
                        "size_bytes": info.st_size if is_file else None,
                        "modified_ns": info.st_mtime_ns,
                    }
                )
    except OSError as error:
        raise HostFileAccessDenied("目录无法读取。") from error
    entries.sort(key=lambda item: (item["kind"] != "directory", item["name"].casefold()))
    result = _page(entries, cursor, page_size)
    result.update(
        {
            "scanned_entries": scanned,
            "truncated": truncated,
            "next_action": (
                "目录项很多；如目标未出现，请改用host_search_files并提供更具体的文件名关键词。"
                if truncated
                else None
            ),
        }
    )
    return result


@tool
async def host_search_files(
    root: str,
    query: str,
    runtime: ToolRuntime,
    match_mode: Literal["contains", "glob"] = "contains",
    cursor: str | None = None,
    page_size: int = 50,
    max_depth: int = 4,
    scan_limit: int = 20_000,
) -> dict:
    """按文件名有界搜索本机目录；根目录优先并保留截断前结果。"""
    _require_scope(runtime)
    resolved, _ = await asyncio.to_thread(HOST_FILE_ACCESS.policy.validate, root)
    if not resolved.is_dir():
        raise HostFileAccessDenied("host_search_files 的 root 必须是目录。")
    needle = query.strip()
    if not needle:
        raise HostFileAccessDenied("query 不能为空。")
    if not 0 <= max_depth <= 32:
        raise HostFileAccessDenied("max_depth 必须在 0 到 32 之间。")
    if not 100 <= scan_limit <= 50_000:
        raise HostFileAccessDenied("scan_limit 必须在 100 到 50000 之间。")
    matches: list[dict] = []
    visited = 0
    truncated = False
    deepest_visited = 0
    queue: deque[tuple[Path, int]] = deque([(resolved, 0)])
    while queue and not truncated:
        current, depth = queue.popleft()
        deepest_visited = max(deepest_visited, depth)
        child_directories: list[Path] = []
        try:
            with os.scandir(current) as scanner:
                for entry in scanner:
                    visited += 1
                    if visited > scan_limit:
                        truncated = True
                        break
                    candidate = Path(entry.path)
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            if depth < max_depth:
                                checked, _ = HOST_FILE_ACCESS.policy.validate(str(candidate))
                                child_directories.append(checked)
                            continue
                        if not entry.is_file(follow_symlinks=False):
                            continue
                        selected = (
                            needle.casefold() in entry.name.casefold()
                            if match_mode == "contains"
                            else candidate.match(needle)
                        )
                        if not selected:
                            continue
                        checked, _ = HOST_FILE_ACCESS.policy.validate(
                            str(candidate), require_file=True
                        )
                        info = entry.stat(follow_symlinks=False)
                    except (HostFileAccessDenied, OSError):
                        continue
                    matches.append(
                        {
                            "name": checked.name,
                            "path": str(checked),
                            "size_bytes": info.st_size,
                            "modified_ns": info.st_mtime_ns,
                            "depth": depth,
                        }
                    )
        except OSError:
            continue
        queue.extend((directory, depth + 1) for directory in child_directories)
    matches.sort(
        key=lambda item: (
            item["depth"],
            -int(item["modified_ns"]),
            item["path"].casefold(),
        )
    )
    result = _page(matches, cursor, page_size)
    result.update(
        {
            "root": str(resolved),
            "scanned_entries": min(visited, scan_limit),
            "max_depth": max_depth,
            "deepest_visited": deepest_visited,
            "truncated": truncated,
            "next_action": (
                "已返回截断前找到的结果；如目标未出现，请缩小root、减少max_depth或使用更具体关键词。"
                if truncated
                else None
            ),
        }
    )
    return result


@tool
async def host_import_file(path: str, reason: str, runtime: ToolRuntime) -> Command:
    """只读复制一个本机文件到当前Worker隔离工作区。"""
    state = _require_scope(runtime)
    tool_call_id = str(runtime.tool_call_id or "host_import_file")
    try:
        resolved, root = await asyncio.to_thread(
            HOST_FILE_ACCESS.policy.validate, path, require_file=True
        )
        record = await asyncio.to_thread(
            register_file,
            resolved,
            source_root=root,
            state=state,
            tool_call_id=tool_call_id,
            origin="HOST",
            # This is Harness-authored provenance, not a model-visible path
            # guess. It lets a later disclosure Judge verify that the stable
            # /downloads reference is the isolated copy of this exact source.
            source_ref="host:" + str(resolved),
        )
        record = await asyncio.to_thread(materialize_import, record)
        result = {
            "status": "IMPORTED",
            "candidate_id": record.candidate_id,
            "filename": record.filename,
            "size_bytes": record.size_bytes,
            "sha256": record.sha256,
            "reading_path": "/downloads/" + record.candidate_id,
            "execution_path": record.sandbox_path,
            "next_step": (
                "文件已复制到沙盒；可直接用execution_path运行或作为程序输入，read_file仅在需要查看内容时使用。"
                if record.sandbox_path else "文件已复制到当前Worker私有存储；可用read_file(reading_path)查看，不要再使用主机绝对路径。"
            ),
        }
        return Command(
            update={
                "worker_downloaded_artifacts": [record.model_dump(mode="json")],
                "messages": [
                    ToolMessage(
                        content=json.dumps(result, ensure_ascii=False),
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )
    except (HostFileAccessDenied, OSError, ValueError) as error:
        result = {
            "status": "POLICY_DENIED" if isinstance(error, HostFileAccessDenied) else "IMPORT_FAILED",
            "code": getattr(error, "code", "IMPORT_FAILED"),
            "message": str(error),
            "path": path,
            "retryable": False,
        }
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=json.dumps(result, ensure_ascii=False),
                    tool_call_id=tool_call_id,
                    status="error",
                )
            ]
        }
    )


HOST_FILE_TOOLS = [host_list_files, host_search_files, host_import_file]

refresh_host_file_tool_descriptions()

for _tool in HOST_FILE_TOOLS:
    _tool.metadata = {
        "capability": "host_file_interaction",
        "risk": "host_file_boundary",
        "executor": "host_after_policy_checks",
    }


__all__ = [
    "HOST_FILE_TOOLS",
    "host_import_file",
    "host_list_files",
    "host_search_files",
    "refresh_host_file_tool_descriptions",
]
