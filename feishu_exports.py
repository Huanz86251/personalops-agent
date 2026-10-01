"""Owner-bound file export requests. Models cannot choose recipients.

No network or filesystem work at import time. An authenticated ingress creates
the event scope; General can directly send a file from the configured outbox.
SENDING is deliberately not retryable: after an ambiguous response we prefer
a missed delivery to a duplicate export.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
import stat
import time
import zipfile
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from feishu_attachments import safe_name

PLATFORM_UPLOAD_LIMIT = 30_000_000  # Conservative decimal MB, IM file upload endpoint.
SOURCE_LIMIT = 100_000_000
MAX_FILES = 100
TTL_SECONDS = 15 * 60
DENIED_PARTS = {".env", ".git", ".agent", ".ssh", ".aws", ".azure", ".codex"}


class ExportDenied(ValueError):
    """Safe, actionable refusal suitable for the owner/tool result."""


class ExportUncertain(RuntimeError):
    """A delivery was attempted but no durable success receipt is available."""


@dataclass(frozen=True)
class ExportIdentity:
    event_id: str
    conversation_id: str
    sender_id: str
    chat_id: str


@dataclass(frozen=True)
class ExportPolicy:
    owners: frozenset[str] = frozenset()
    chats: frozenset[str] = frozenset()
    roots: tuple[Path, ...] = ()
    max_upload_bytes: int = PLATFORM_UPLOAD_LIMIT

    def __post_init__(self):
        object.__setattr__(
            self, "roots", tuple(Path(root).resolve() for root in self.roots)
        )
        if (
            not isinstance(self.max_upload_bytes, int)
            or isinstance(self.max_upload_bytes, bool)
            or self.max_upload_bytes <= 0
        ):
            raise ValueError("文件发送上限必须大于 0 字节。")

    @property
    def effective_upload_limit(self):
        # A local setting may be stricter, but cannot raise Feishu's API cap.
        return min(self.max_upload_bytes, PLATFORM_UPLOAD_LIMIT)

    @classmethod
    def from_env(cls):
        def strings(name):
            values = json.loads(os.environ.get(name, "[]"))
            if not isinstance(values, list) or any(
                not isinstance(v, str) or not v.strip() for v in values
            ):
                raise ValueError(f"{name} 必须是非空字符串组成的 JSON 数组。")
            return values

        roots = []
        for value in strings("FEISHU_EXPORT_ROOTS"):
            path = Path(value)
            if not path.is_absolute() or str(path).startswith(("\\\\", "//")):
                raise ValueError("导出目录必须是本地绝对路径。")
            roots.append(path.resolve())
        raw_max_mb = os.environ.get("FEISHU_EXPORT_MAX_FILE_MB", "30").strip()
        try:
            max_mb = int(raw_max_mb)
        except ValueError as error:
            raise ValueError("FEISHU_EXPORT_MAX_FILE_MB 必须是正整数（MB）。") from error
        if max_mb <= 0:
            raise ValueError("FEISHU_EXPORT_MAX_FILE_MB 必须是正整数（MB）。")
        return cls(
            frozenset(strings("FEISHU_EXPORT_OWNER_IDS")),
            frozenset(strings("FEISHU_EXPORT_CHAT_IDS")),
            tuple(roots),
            max_mb * 1_000_000,
        )

    @property
    def enabled(self):
        # Source paths are authorized per operation by the independent Judge.
        # ``roots`` is retained as a deployment compatibility field but no
        # longer acts as the user-delivery boundary.
        return bool(self.owners and self.chats)

    def authorize(self, sender, chat):
        if not self.enabled or sender not in self.owners or chat not in self.chats:
            raise ExportDenied("文件导出未配置，或当前用户／会话没有文件导出权限。")


_identity: ContextVar[ExportIdentity | None] = ContextVar(
    "feishu_export_identity", default=None
)
_role: ContextVar[str] = ContextVar("feishu_export_role", default="")
_service: ExportService | None = None


def configure_exports(service):
    global _service
    _service = service


@contextmanager
def export_event_scope(identity):
    token = _identity.set(identity)
    try:
        yield
    finally:
        _identity.reset(token)


@contextmanager
def export_worker_scope(worker_kind: str):
    normalized = str(worker_kind).strip().lower()
    if normalized not in {"general", "web", "code"}:
        raise ValueError("文件发送执行角色必须是 general、web 或 code。")
    token = _role.set(normalized)
    try:
        yield
    finally:
        _role.reset(token)


@contextmanager
def export_general_scope():
    """Backward-compatible spelling for the General runtime."""

    with export_worker_scope("general"):
        yield


def export_tool_available():
    identity = _identity.get()
    if not _service or not identity:
        return False
    try:
        _service.policy.authorize(identity.sender_id, identity.chat_id)
        return True
    except ExportDenied:
        return False


def current_export_identity() -> ExportIdentity | None:
    """Return the authenticated run-scoped identity; never infer a recipient."""
    return _identity.get()


async def request_export(path, event_id):
    identity = _identity.get()
    if (
        not export_tool_available()
        or _role.get() not in {"general", "web", "code"}
        or identity.event_id != event_id
    ):
        raise ExportDenied("只能在已验证飞书任务的 Worker 执行环境中申请文件发送。")
    return await _service.request(identity, path)


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ExportService:
    def __init__(self, root, policy, channel):
        self.root = Path(root).resolve()
        self.policy = policy
        self.channel = channel
        self._lock = asyncio.Lock()

    def _query(self, sql, args=()):
        self.root.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.root / "exports.sqlite3", timeout=10)
        try:
            db.row_factory = sqlite3.Row
            db.execute("""CREATE TABLE IF NOT EXISTS exports (
                id TEXT PRIMARY KEY, dedup TEXT UNIQUE, event_id TEXT, conversation_id TEXT,
                sender_id TEXT, chat_id TEXT, source TEXT, filename TEXT, snapshot TEXT,
                sha256 TEXT, size INTEGER, manifest TEXT, expires REAL, status TEXT,
                message_id TEXT DEFAULT '', file_key TEXT DEFAULT '')""")
            with db:
                return [dict(row) for row in db.execute(sql, args).fetchall()]
        finally:
            db.close()

    def _check_path(self, path):
        path = Path(path)
        if not path.is_absolute() or str(path).startswith(("\\\\", "//")):
            raise ExportDenied(
                "只接受允许目录内的本地绝对路径，不接受 URL、网络共享或任务虚拟路径。"
            )
        # Inspect the original chain as well as the resolved target, rejecting
        # Windows junctions/reparse points, Unix symlinks and NTFS ADS.  The
        # directory allowlist was intentionally removed: external disclosure
        # is controlled by the independent per-operation Judge instead.
        for part in (path, *path.parents):
            if ":" in part.name:
                raise ExportDenied("不支持特殊文件流。")
            info = part.lstat()
            if (
                stat.S_ISLNK(info.st_mode)
                or getattr(info, "st_file_attributes", 0) & 0x400
            ):
                raise ExportDenied("不允许导出符号链接或目录联接。")
        resolved = path.resolve(strict=True)
        info = resolved.stat()
        if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
            raise ExportDenied("只支持普通文件和目录。")
        if stat.S_ISREG(info.st_mode) and info.st_nlink > 1:
            raise ExportDenied("不允许导出硬链接。")
        return resolved

    def _snapshot(self, raw_path):
        source = self._check_path(raw_path)
        upload_limit = self.policy.effective_upload_limit
        limit_reason = (
            "飞书平台上限"
            if upload_limit == PLATFORM_UPLOAD_LIMIT
            else "本机配置上限"
        )
        limit_label = f"{upload_limit / 1_000_000:g} MB（{upload_limit} 字节，{limit_reason}）"
        files = []
        if source.is_dir():
            # Do not silently omit secrets, oversized files or deep subtrees.
            visited = 0

            def walk_error(error):
                raise ExportDenied(
                    "目录中有无法读取的内容，未打包发送；请缩小范围。"
                ) from error

            for current, directories, names in os.walk(
                source, followlinks=False, onerror=walk_error
            ):
                visited += len(directories) + len(names)
                if visited > 1000:
                    raise ExportDenied("目录项超过 1000，请缩小范围。")
                for name in directories:
                    self._check_path(Path(current) / name)
                for name in sorted(names):
                    files.append(self._check_path(Path(current) / name))
                    if len(files) > MAX_FILES:
                        raise ExportDenied(
                            f"目录最多允许 {MAX_FILES} 个文件，请缩小范围。"
                        )
        else:
            files = [source]
        if not files:
            raise ExportDenied("目录为空。")
        manifest = []
        total = 0
        for file in sorted(files):
            size = file.stat().st_size
            if not source.is_dir() and size > upload_limit:
                raise ExportDenied(
                    f"文件大小 {size} 字节超过单次发送上限 {limit_label}；"
                    "系统已拦截，未上传。请先询问用户是否允许拆分，不要自行拆分或重发。"
                )
            total += size
            if total > SOURCE_LIMIT:
                raise ExportDenied("原文件合计不得超过 100 MB。")
            manifest.append(
                {
                    "name": str(file.relative_to(source))
                    if source.is_dir()
                    else file.name,
                    "bytes": size,
                }
            )
        if not source.is_dir() and total == 0:
            raise ExportDenied("不能发送空文件；文件没有上传。")
        objects = self.root / "objects"
        objects.mkdir(parents=True, exist_ok=True)
        snapshot = objects / (uuid4().hex + ".blob")
        try:
            copied = 0
            with snapshot.open("xb") as target:
                archive = (
                    zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED)
                    if source.is_dir()
                    else None
                )
                try:
                    for file, entry in zip(sorted(files), manifest):
                        self._check_path(file)
                        before = file.stat()
                        with file.open("rb") as input_file:
                            opened = os.fstat(input_file.fileno())
                            if (before.st_dev, before.st_ino) != (
                                opened.st_dev,
                                opened.st_ino,
                            ):
                                raise ExportDenied("文件在读取前发生变化，请重试。")
                            output = (
                                archive.open(entry["name"].replace("\\", "/"), "w")
                                if archive
                                else target
                            )
                            count = 0
                            try:
                                for chunk in iter(
                                    lambda: input_file.read(1024 * 1024), b""
                                ):
                                    copied += len(chunk)
                                    count += len(chunk)
                                    if copied > SOURCE_LIMIT:
                                        raise ExportDenied("读取期间文件超过 100 MB。")
                                    output.write(chunk)
                                    if target.tell() > upload_limit:
                                        raise ExportDenied(
                                            "发送文件（目录为 ZIP）超过单次发送上限 "
                                            f"{limit_label}；"
                                            "系统已拦截，未上传。请先询问用户是否允许拆分。"
                                        )
                            finally:
                                if archive:
                                    output.close()
                            after = os.fstat(input_file.fileno())
                        if count != entry["bytes"] or (
                            opened.st_size,
                            opened.st_mtime_ns,
                        ) != (after.st_size, after.st_mtime_ns):
                            raise ExportDenied("文件正在变化，请保存后重试。")
                finally:
                    if archive:
                        archive.close()
            if not 0 < snapshot.stat().st_size <= upload_limit:
                raise ExportDenied(
                    "发送文件必须非空且不超过单次发送上限 "
                    f"{limit_label}；未上传。"
                    "如果超限，请先询问用户是否允许拆分。"
                )
            filename = safe_name(
                source.name + ".zip" if source.is_dir() else source.name
            )
            return source, snapshot, filename, manifest
        except BaseException:
            snapshot.unlink(missing_ok=True)
            raise

    def _prepare(self, identity, path):
        self.policy.authorize(identity.sender_id, identity.chat_id)
        source, snapshot, filename, manifest = self._snapshot(path)
        digest = _digest(snapshot)
        dedup = hashlib.sha256(
            json.dumps(
                [
                    identity.event_id,
                    identity.sender_id,
                    identity.chat_id,
                    str(source),
                    digest,
                ]
            ).encode()
        ).hexdigest()
        existing = self._query("SELECT * FROM exports WHERE dedup=?", (dedup,))
        if existing:
            snapshot.unlink()
            return existing[0]
        if len(self._query("SELECT id FROM exports WHERE status='PENDING'")) >= 8:
            snapshot.unlink()
            raise ExportDenied(
                "最多保留 8 个未完成的文件发送请求，请稍后再试。"
            )
        request_id = "fx_" + uuid4().hex[:24]
        self._query(
            "INSERT INTO exports (id,dedup,event_id,conversation_id,sender_id,chat_id,source,filename,snapshot,sha256,size,manifest,expires,status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                request_id,
                dedup,
                identity.event_id,
                identity.conversation_id,
                identity.sender_id,
                identity.chat_id,
                str(source),
                filename,
                snapshot.name,
                digest,
                snapshot.stat().st_size,
                json.dumps(manifest, ensure_ascii=False),
                time.time() + TTL_SECONDS,
                "PENDING",
            ),
        )
        return self._query("SELECT * FROM exports WHERE id=?", (request_id,))[0]

    def _view(self, row):
        status = row["status"]
        if status == "PENDING" and row["expires"] < time.time():
            status = "EXPIRED"
        return {
            "request_id": row["id"],
            "status": status,
            "filename": row["filename"],
            "bytes": row["size"],
            "sha256": row["sha256"],
            "message_id": row["message_id"],
            "instruction": "文件正在准备发送；只有 SENT 且存在飞书 message_id 才算交付成功。"
            if status == "PENDING"
            else "以状态和飞书 message_id 判断结果，不要把未确认的发送结果说成成功。",
        }

    async def request(self, identity, path):
        self.policy.authorize(identity.sender_id, identity.chat_id)
        async with self._lock:
            await asyncio.to_thread(self.cleanup)
            row = await asyncio.to_thread(self._prepare, identity, path)
        if row["status"] == "PENDING":
            # The owner/chat/outbox configuration is standing authorization.
            # The host still verifies the immutable snapshot and atomically
            # claims it before uploading; no per-file approval is needed.
            return await self.decide(
                row["id"], identity.sender_id, identity.chat_id, approve=True
            )
        return self._view(row)

    def cleanup(self):
        rows = self._query(
            "SELECT * FROM exports WHERE status IN ('PENDING','SENT','REJECTED','EXPIRED')"
        )
        for row in rows:
            if row["status"] != "PENDING" or row["expires"] < time.time():
                if row["status"] == "PENDING":
                    self._query(
                        "UPDATE exports SET status='EXPIRED' WHERE id=? AND status='PENDING'",
                        (row["id"],),
                    )
                (self.root / "objects" / row["snapshot"]).unlink(missing_ok=True)

    async def decide(self, request_id, sender, chat, *, approve):
        self.policy.authorize(sender, chat)
        async with self._lock:
            rows = self._query(
                "SELECT * FROM exports WHERE id=? AND sender_id=? AND chat_id=?",
                (request_id, sender, chat),
            )
            if not rows:
                raise ExportDenied("申请不存在或不属于当前用户／会话。")
            row = rows[0]
            if row["status"] != "PENDING":
                return self._view(row)
            if row["expires"] < time.time():
                self._query(
                    "UPDATE exports SET status='EXPIRED' WHERE id=?", (request_id,)
                )
                self.cleanup()
                return {**self._view(row), "status": "EXPIRED"}
            if not approve:
                self._query(
                    "UPDATE exports SET status='REJECTED' WHERE id=?", (request_id,)
                )
                self.cleanup()
                return {**self._view(row), "status": "REJECTED"}
            # Re-check current operator policy. Upload the prepared immutable
            # snapshot, not the possibly changed source file.
            self._check_path(row["source"])
            snapshot = self.root / "objects" / row["snapshot"]
            if (
                snapshot.is_symlink()
                or snapshot.stat().st_size != row["size"]
                or await asyncio.to_thread(_digest, snapshot) != row["sha256"]
            ):
                raise ExportDenied("待发送快照校验失败，禁止上传。")
            # Atomically claim even when two application processes receive an
            # send. A crash from here remains SENDING and is never retried.
            claimed = self._query(
                "UPDATE exports SET status='SENDING' WHERE id=? AND status='PENDING' RETURNING id",
                (request_id,),
            )
            if not claimed:
                return self._view(
                    self._query("SELECT * FROM exports WHERE id=?", (request_id,))[0]
                )
            from lark_channel.channel.types import MediaSource, OutboundFile

            try:
                # ``lark_channel`` intentionally derives the upload filename
                # from ``MediaSource.path`` for file sources.  Our immutable
                # snapshot has a random ``.blob`` basename, so passing it as a
                # file source would silently replace the user-facing filename
                # supplied below.  A buffer source preserves the verified
                # snapshot bytes while allowing ``file_name`` to remain the
                # original sanitized name (for example, ``resume.pdf``).
                snapshot_bytes = await asyncio.to_thread(snapshot.read_bytes)
                key = await asyncio.wait_for(
                    self.channel.upload_media(
                        MediaSource(kind="buffer", buffer=snapshot_bytes),
                        kind="file",
                        file_name=row["filename"],
                        file_type="stream",
                    ),
                    timeout=120,
                )
                if not isinstance(key, str) or not key:
                    raise RuntimeError("Missing file key")
                self._query(
                    "UPDATE exports SET file_key=? WHERE id=?", (key, request_id)
                )
                result = await asyncio.wait_for(
                    self.channel.send(
                        chat,
                        OutboundFile(
                            source=MediaSource(kind="key", key=key),
                            file_name=row["filename"],
                        ),
                        {"uuid": request_id, "receive_id_type": "chat_id"},
                    ),
                    timeout=60,
                )
                if not result.success or not result.message_id:
                    raise RuntimeError("No confirmed message receipt")
                self._query(
                    "UPDATE exports SET status='SENT',message_id=? WHERE id=?",
                    (result.message_id, request_id),
                )
            except BaseException as error:
                self._query(
                    "UPDATE exports SET status='UNKNOWN' WHERE id=?", (request_id,)
                )
                if isinstance(
                    error, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)
                ):
                    raise
                raise ExportUncertain("发送结果不确定，禁止自动重试。") from error
            finally:
                snapshot.unlink(missing_ok=True)
            return self._view(
                self._query("SELECT * FROM exports WHERE id=?", (request_id,))[0]
            )

    async def handle_control(self, message, text):
        """Only called by authenticated channel ingress, never by a tool."""
        parts = text.split()
        if not parts or parts[0] not in {
            "/file_whoami",
            "/file_approve",
            "/file_reject",
            "/file_status",
        }:
            return False
        sender = str(getattr(getattr(message, "sender", None), "open_id", "") or "")
        chat = str(getattr(message, "chat_id", "") or "")
        if parts[0] == "/file_whoami":
            reply = f"你的 open_id：{sender}\n当前 chat_id：{chat}\n这仅查询身份，不会授予权限；由电脑上的配置指定授权用户和导出目录。"
        else:
            try:
                self.policy.authorize(sender, chat)
                # Batched/quoted/attachment text is not a standalone approval.
                if (
                    getattr(message, "resources", ())
                    or getattr(message, "batched_sources", None)
                    or getattr(getattr(message, "content", None), "kind", "") != "text"
                    or len(parts) != 2
                ):
                    raise ExportDenied("请单独发送纯文字命令和一个申请编号。")
                if parts[0] == "/file_status":
                    rows = self._query(
                        "SELECT * FROM exports WHERE id=? AND sender_id=? AND chat_id=?",
                        (parts[1], sender, chat),
                    )
                    if not rows:
                        raise ExportDenied("申请不存在或不属于当前用户／会话。")
                    result = self._view(rows[0])
                else:
                    result = await self.decide(
                        parts[1], sender, chat, approve=parts[0] == "/file_approve"
                    )
                reply = json.dumps(result, ensure_ascii=False)
            except (ExportDenied, OSError) as error:
                reply = f"未发送文件：{error}"
            except Exception:  # noqa: BLE001 - channel failures must not claim delivery or expose credentials
                reply = "文件发送结果不确定，请先在飞书检查是否收到。系统不会自动重发；可用 /file_status 申请编号 查询。"
        await self.channel.send(chat, {"text": reply})
        return True

    def admit_message(self, message):
        # This app shares one owner conversation. Once owners are configured,
        # refuse strangers before they can read shared history or enqueue work.
        if not self.policy.owners:
            return True
        sender = str(getattr(getattr(message, "sender", None), "open_id", "") or "")
        return (
            sender in self.policy.owners
            and str(getattr(message, "chat_id", "")) in self.policy.chats
        )
