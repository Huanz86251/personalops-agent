"""One-message, durable RAG upload mode. No conversational model is invoked."""
from __future__ import annotations
import asyncio
import sqlite3
from contextlib import closing
from pathlib import Path

from observability import trace_span, set_span_output
from feishu_attachments import safe_name, key_for


class RagUploadInbox:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _db(self):
        db = sqlite3.connect(self.root / "upload-mode.sqlite3", timeout=20)
        db.execute("CREATE TABLE IF NOT EXISTS pending (chat TEXT, sender TEXT, PRIMARY KEY(chat,sender))")
        db.execute("CREATE TABLE IF NOT EXISTS receipts (message TEXT PRIMARY KEY, status TEXT, detail TEXT)")
        return db

    def arm(self, chat, sender):
        with closing(self._db()) as db, db:
            db.execute("INSERT OR IGNORE INTO pending VALUES (?,?)", (chat, sender))

    def claim(self, chat, sender, message):
        key = key_for(chat, sender, message)
        with closing(self._db()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            previous = db.execute("SELECT status,detail FROM receipts WHERE message=?", (key,)).fetchone()
            if previous:
                return key, previous
            if not db.execute("DELETE FROM pending WHERE chat=? AND sender=?", (chat,sender)).rowcount:
                return None, None
            db.execute("INSERT INTO receipts VALUES (?,'processing','正在入库；不会进入任务队列。')", (key,))
            return key, None

    def finish(self, key, status, detail):
        with closing(self._db()) as db, db:
            db.execute("UPDATE receipts SET status=?,detail=? WHERE message=?", (status, detail, key))

    async def consume(self, message, attachment_store, channel, hub, conversation_id, scope="owner"):
        chat = str(getattr(message, "chat_id", ""))
        sender = str(getattr(getattr(message, "sender", None), "open_id", ""))
        message_id = str(getattr(message, "message_id", "") or getattr(message, "id", ""))
        if not message_id:
            return None
        key, previous = self.claim(chat, sender, message_id)
        if key is None:
            return None
        if previous:
            return previous[1]
        with trace_span("RAG / Upload", input_value={"message_id": message_id, "scope": scope}) as span:
            try:
                from knowledge_rag.service import digest, SUPPORTED
                if hub is None:
                    raise ValueError("RAG 服务尚未启动")
                kind = str(getattr(getattr(message, "content", None), "kind", ""))
                # Voice is accepted only with a real channel-provided transcript.
                transcript = str(getattr(message, "transcript", "") or "").strip()
                if kind in {"audio", "voice"}:
                    if not transcript:
                        raise ValueError("语音没有可用转写文本，请发送转写文字")
                    inbound = None
                    text = transcript
                else:
                    if kind in {"video", "media", "image", "sticker"}:
                        raise ValueError("此媒体无法解析为 RAG 文本")
                    inbound = await attachment_store.prepare(message, conversation_id, channel)
                    # The normal task instruction contains private handoff paths.
                    # RAG admission stores only the user's visible note and files.
                    clean_text = getattr(attachment_store, "_clean_text", None)
                    if callable(clean_text):
                        text = str(clean_text(message) or "").strip()
                    else:
                        text = str(
                            getattr(message, "content_text", "")
                            or getattr(message, "body_text", "")
                            or getattr(inbound, "instruction", "")
                            or ""
                        ).strip()
                directory = hub.root / "uploads" / digest(scope)[:24] / key
                directory.mkdir(parents=True, exist_ok=True)
                paths = []
                for item in inbound.attachments if inbound else ():
                    path = directory / (item["id"] + "-" + safe_name(item["name"]))
                    if path.suffix.lower() not in SUPPORTED:
                        raise ValueError(f"暂不支持 {path.suffix}；可上传 TXT、Markdown、JSON、JSONL、CSV、HTML、PDF、DOCX、XLSX、PPTX")
                    import hashlib
                    data = (attachment_store.root / "objects" / item["object_name"]).read_bytes()
                    if hashlib.sha256(data).hexdigest() != item["sha256"]:
                        raise ValueError("保存的附件校验失败")
                    path.write_bytes(data)
                    paths.append(path)
                if text.strip():
                    path = directory / "message.txt"
                    path.write_text(text, encoding="utf-8")
                    paths.append(path)
                if not paths:
                    raise ValueError("没有可解析的文本或文档")
                results = await hub.ingest(scope, paths)
                (directory / ".ready").write_text("indexed", encoding="utf-8")
                detail = f"已存入 RAG：{len(results)} 份资料。本条消息没有触发 Agent。"
                self.finish(key, "indexed", detail)
                set_span_output(span, {"status": "indexed", "files": results})
                return detail
            except Exception as error:
                detail = f"无法完成 RAG 入库：{error}。请重新点击“上传 RAG”重试；本条消息没有进入任务队列。"
                self.finish(key, "failed", detail)
                set_span_output(span, {"status": "failed", "error": str(error)})
                return detail
