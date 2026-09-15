"""Offline tests for resource-scoped Web Worker execution."""

import asyncio
import unittest
from contextlib import asynccontextmanager
from pathlib import Path

from langchain_core.tools import tool

from workers.web_runtime import WebStepRuntime
from workers.web_worker import select_web_worker_tools


@tool
def web_search(query: str) -> str:
    """Return an offline Web-search fixture."""

    return query


@tool
def business_probe(value: str) -> str:
    """Represent a non-Web business tool."""

    return value


class _BrowserRuntime:
    def __init__(self, owner_id: str, browser_tool) -> None:
        self.owner_id = owner_id
        self.tools = [browser_tool]


class _Pool:
    def __init__(self) -> None:
        self.entered: list[str] = []
        self.released: list[str] = []

    @asynccontextmanager
    async def lease(self, owner_id: str):
        @tool("browser_snapshot")
        def browser_snapshot() -> str:
            """Return the current page for this isolated owner."""

            return owner_id

        self.entered.append(owner_id)
        try:
            yield _BrowserRuntime(owner_id, browser_snapshot)
        finally:
            self.released.append(owner_id)


class _EventStore:
    pass


class _Graph:
    def __init__(self, owner_id: str, fail: bool = False) -> None:
        self.owner_id = owner_id
        self.fail = fail

    async def astream(self, input_state, **kwargs):
        if self.fail:
            raise RuntimeError("worker failed")
        yield "values", {"owner_id": self.owner_id, **input_state}


class WebWorkerToolTests(unittest.TestCase):
    def test_only_web_and_browser_tools_are_selected(self):
        @tool("browser_navigate")
        def browser_navigate(url: str) -> str:
            """Navigate an offline browser fixture."""

            return url

        selected = select_web_worker_tools(
            [web_search, browser_navigate, business_probe]
        )
        self.assertEqual(
            [current.name for current in selected],
            ["web_search", "browser_navigate"],
        )


class WebStepRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def _runtime(self, pool: _Pool, *, fail_owner: str = ""):
        def worker_factory(model, *, tools, **kwargs):
            browser_tool = next(
                current for current in tools if current.name == "browser_snapshot"
            )
            owner_id = browser_tool.invoke({})
            return _Graph(owner_id, fail=owner_id == fail_owner)

        return WebStepRuntime(
            "offline-model",
            playwright_pool=pool,
            event_store=_EventStore(),
            tools=[web_search],
            decision_handler=None,
            progress_every_tool_calls=4,
            run_storage_root=Path(self._temp_directory.name),
            worker_factory=worker_factory,
        )

    def setUp(self):
        import tempfile

        self._temp_directory = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._temp_directory.cleanup()

    async def test_parallel_workers_receive_distinct_sessions_and_release(self):
        pool = _Pool()
        runtime = self._runtime(pool)
        results = await asyncio.gather(
            *(
                runtime.ainvoke(
                    {"worker_id": f"web-{index}", "event_id": "run-web"}
                )
                for index in range(3)
            )
        )
        self.assertEqual(
            {result["owner_id"] for result in results},
            {"web-0", "web-1", "web-2"},
        )
        self.assertCountEqual(pool.entered, ["web-0", "web-1", "web-2"])
        self.assertCountEqual(pool.released, pool.entered)

    async def test_session_releases_when_worker_fails(self):
        pool = _Pool()
        runtime = self._runtime(pool, fail_owner="web-fail")
        with self.assertRaisesRegex(RuntimeError, "worker failed"):
            await runtime.ainvoke(
                {"worker_id": "web-fail", "event_id": "run-web"}
            )
        self.assertEqual(pool.entered, ["web-fail"])
        self.assertEqual(pool.released, ["web-fail"])

    async def test_download_limit_is_forwarded_to_worker_factory(self):
        pool = _Pool()
        captured: dict[str, int] = {}

        def worker_factory(model, *, tools, **kwargs):
            browser_tool = next(
                current for current in tools if current.name == "browser_snapshot"
            )
            captured["download_max_file_mib"] = kwargs[
                "download_max_file_mib"
            ]
            return _Graph(browser_tool.invoke({}))

        runtime = WebStepRuntime(
            "offline-model",
            playwright_pool=pool,
            event_store=_EventStore(),
            tools=[web_search],
            progress_every_tool_calls=4,
            download_max_file_mib=13,
            run_storage_root=Path(self._temp_directory.name),
            worker_factory=worker_factory,
        )
        await runtime.ainvoke(
            {"worker_id": "web-download-config", "event_id": "run-web"}
        )

        self.assertEqual(captured["download_max_file_mib"], 13)


if __name__ == "__main__":
    unittest.main()
