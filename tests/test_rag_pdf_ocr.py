from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.documents import Document

from knowledge_rag.runtime import RetrievalHub


class _KnowledgeBase:
    def __init__(self):
        self.calls = []

    def ingest(self, path, *, documents):
        self.calls.append((Path(path), documents))
        return {
            "status": "indexed",
            "source": Path(path).name,
            "parser": documents[0].metadata.get("parser"),
        }


class RagPdfOcrTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.hub = RetrievalHub(None, None, {"root": str(self.root / "rag")})
        self.kb = _KnowledgeBase()
        self.hub._run = lambda scope, operation: operation(self.kb)

    async def test_scanned_pdf_uses_isolated_ocr_fallback(self):
        source = self.root / "scan.pdf"
        source.write_bytes(b"%PDF synthetic")
        ocr = {
            "markdown": "# OCR result\n\nReadable text",
            "summary": {"ocr_pages": [1], "process_released": True},
        }
        with (
            patch("knowledge_rag.service.read_document", return_value=[]),
            patch("tools.local_native._reading_process", return_value=ocr) as reader,
        ):
            result = await self.hub.ingest("owner", [source])
        reader.assert_called_once_with(source.read_bytes(), source.name, ocr="auto")
        self.assertEqual(result[0]["parser"], "ocr_markdown")
        self.assertIn("Readable text", self.kb.calls[0][1][0].page_content)

    async def test_native_pdf_does_not_start_ocr(self):
        source = self.root / "native.pdf"
        source.write_bytes(b"%PDF synthetic")
        parsed = [Document(page_content="native text", metadata={"parser": "docling"})]
        with (
            patch("knowledge_rag.service.read_document", return_value=parsed),
            patch("tools.local_native._reading_process") as reader,
        ):
            result = await self.hub.ingest("owner", [source])
        reader.assert_not_called()
        self.assertEqual(result[0]["parser"], "docling")

    async def test_unreadable_non_pdf_is_rejected_without_ocr(self):
        source = self.root / "empty.txt"
        source.write_text("", encoding="utf-8")
        with (
            patch("knowledge_rag.service.read_document", return_value=[]),
            patch("tools.local_native._reading_process") as reader,
        ):
            with self.assertRaisesRegex(ValueError, "没有可提取文字"):
                await self.hub.ingest("owner", [source])
        reader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
