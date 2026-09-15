from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import (
    patch,
)

from config import (
    _load_delivery_settings,
    _load_runtime_concurrency_settings,
    _load_code_sandbox_settings,
    _load_worker_runtime_settings,
)
from delivery.models import DeliveryApprovalMode
from mcp_runtime import (
    PLAYWRIGHT_OUTPUT_PATH,
    PLAYWRIGHT_PROFILE_PATH,
    PlaywrightMCPPool,
)
from tools.web_tools import (
    configure_web_search_parallelism,
    get_web_search_max_parallelism,
)


class _FakePlaywrightRuntime:
    """不启动Node或浏览器的池生命周期替身。"""

    def __init__(
        self,
        *,
        profile_path: Path,
        output_path: Path,
        owner_id: str,
    ) -> None:
        self.profile_path = profile_path
        self.output_path = output_path
        self.owner_id = owner_id
        self.started = False
        self.stopped = False
        self.reset_count = 0

    @property
    def tools(
        self,
    ) -> list:
        return [
            f"tool-for-{self.owner_id}"
        ]

    async def start(
        self,
    ) -> None:
        self.started = True

    async def reset_page(
        self,
    ) -> None:
        self.reset_count += 1

    async def stop(
        self,
    ) -> None:
        self.stopped = True


class RuntimeConcurrencyConfigTests(
    unittest.TestCase
):
    def test_defaults_match_current_policy(
        self,
    ) -> None:
        environment_names = {
            "RUNTIME_WEB_SEARCH_MAX_PARALLELISM",
            "RUNTIME_PLAYWRIGHT_MAX_SESSIONS",
            "RUNTIME_CODE_MAX_WRITERS",
            "RUNTIME_WORKER_PROGRESS_EVERY_TOOL_CALLS",
            "RUNTIME_WORKER_FINALIZATION_MODEL_ROUNDS",
            "RUNTIME_WORKER_SCHEMA_REPAIR_MAX_ROUNDS",
            "RUNTIME_WEB_DOWNLOAD_MAX_FILE_MIB",
            "RUNTIME_CODE_REVIEW_MAX_REPAIR_ROUNDS",
            "RUNTIME_WORKER_WORKSPACE_RETENTION_MINUTES",
            "RUNTIME_CODE_SANDBOX_AUTO_BUILD",
            "RUNTIME_CODE_SANDBOX_MEMORY_MB",
        }
        clean_environment = {
            key: value
            for key, value in os.environ.items()
            if key not in environment_names
        }

        with patch.dict(
            os.environ,
            clean_environment,
            clear=True,
        ):
            settings = (
                _load_runtime_concurrency_settings()
            )
            worker_settings = (
                _load_worker_runtime_settings()
            )
            sandbox_settings = _load_code_sandbox_settings()

        self.assertEqual(
            settings.web_search_max_parallelism,
            3,
        )
        self.assertEqual(
            settings.playwright_max_sessions,
            3,
        )
        self.assertEqual(
            settings.code_max_writers,
            1,
        )

        self.assertEqual(
            worker_settings.code_review_max_repair_rounds,
            2,
        )
        self.assertEqual(
            worker_settings.progress_every_tool_calls,
            4,
        )
        self.assertEqual(worker_settings.finalization_model_rounds, 4)
        self.assertEqual(worker_settings.schema_repair_max_rounds, 3)
        self.assertEqual(worker_settings.web_download_max_file_mib, 20)
        self.assertEqual(worker_settings.leadership_single_worker_reports, 2)
        self.assertEqual(worker_settings.leadership_multi_worker_reports, 1)
        self.assertEqual(worker_settings.leadership_silence_timeout_seconds, 60)
        self.assertEqual(worker_settings.workspace_retention_minutes, 14400)
        self.assertTrue(sandbox_settings.auto_build)
        self.assertEqual(sandbox_settings.memory_mb, 1536)
        self.assertEqual(sandbox_settings.cpu_count, 2)
        self.assertEqual(sandbox_settings.pids_limit, 128)
        self.assertEqual(sandbox_settings.execute_timeout_seconds, 120)

    def test_environment_cannot_exceed_hard_max(
        self,
    ) -> None:
        with patch.dict(
            os.environ,
            {
                "RUNTIME_PLAYWRIGHT_MAX_SESSIONS": "4",
            },
        ):
            with self.assertRaises(
                RuntimeError
            ):
                _load_runtime_concurrency_settings()

        with patch.dict(
            os.environ,
            {
                "RUNTIME_WEB_DOWNLOAD_MAX_FILE_MIB": "21",
            },
        ):
            with self.assertRaises(RuntimeError):
                _load_worker_runtime_settings()

        with patch.dict(
            os.environ,
            {
                "RUNTIME_WORKER_PROGRESS_EVERY_TOOL_CALLS": "9",
            },
        ):
            with self.assertRaises(
                RuntimeError
            ):
                _load_worker_runtime_settings()

        with patch.dict(
            os.environ,
            {
                "RUNTIME_WORKER_WORKSPACE_RETENTION_MINUTES": "43201",
            },
        ):
            with self.assertRaises(RuntimeError):
                _load_worker_runtime_settings()

        with patch.dict(
            os.environ,
            {
                "RUNTIME_WORKER_SCHEMA_REPAIR_MAX_ROUNDS": "4",
            },
        ):
            with self.assertRaises(RuntimeError):
                _load_worker_runtime_settings()

        with patch.dict(
            os.environ,
            {
                "RUNTIME_CODE_REVIEW_MAX_REPAIR_ROUNDS": "4",
            },
        ):
            with self.assertRaises(RuntimeError):
                _load_worker_runtime_settings()

        with patch.dict(
            os.environ,
            {
                "RUNTIME_CODE_SANDBOX_MEMORY_MB": "4097",
            },
        ):
            with self.assertRaises(RuntimeError):
                _load_code_sandbox_settings()

        with patch.dict(
            os.environ,
            {
                "RUNTIME_CODE_SANDBOX_AUTO_BUILD": "sometimes",
            },
        ):
            with self.assertRaises(RuntimeError):
                _load_code_sandbox_settings()

        with patch.dict(
            os.environ,
            {
                "RUNTIME_CODE_MAX_WRITERS": "2",
            },
        ):
            with self.assertRaises(
                RuntimeError
            ):
                _load_runtime_concurrency_settings()

    def test_delivery_approval_mode_is_explicit(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                _load_delivery_settings().approval_mode,
                DeliveryApprovalMode.AUTO,
            )
        with patch.dict(
            os.environ,
            {"RUNTIME_DELIVERY_APPROVAL_MODE": "auto"},
            clear=True,
        ):
            self.assertEqual(
                _load_delivery_settings().approval_mode,
                DeliveryApprovalMode.AUTO,
            )
        with patch.dict(
            os.environ,
            {"RUNTIME_DELIVERY_APPROVAL_MODE": "small-model"},
            clear=True,
        ):
            with self.assertRaises(RuntimeError):
                _load_delivery_settings()

    def test_web_search_gate_uses_configured_limit(
        self,
    ) -> None:
        configure_web_search_parallelism(
            2
        )

        self.assertEqual(
            get_web_search_max_parallelism(),
            2,
        )

        with self.assertRaises(
            ValueError
        ):
            configure_web_search_parallelism(
                4
            )

        # 不让本测试改变同一进程中后续测试的默认策略。
        configure_web_search_parallelism(
            3
        )


class PlaywrightMCPPoolTests(
    unittest.IsolatedAsyncioTestCase
):
    async def test_fourth_owner_waits_until_capacity_returns(
        self,
    ) -> None:
        created: list[
            _FakePlaywrightRuntime
        ] = []

        def factory(
            **kwargs,
        ) -> _FakePlaywrightRuntime:
            runtime = _FakePlaywrightRuntime(
                **kwargs
            )
            created.append(
                runtime
            )
            return runtime

        pool = PlaywrightMCPPool(
            max_sessions=3,
            runtime_factory=factory,
        )
        release_first_three = (
            asyncio.Event()
        )
        entered_first_three = [
            asyncio.Event()
            for _ in range(3)
        ]
        fourth_entered = asyncio.Event()

        async def hold(
            owner_id: str,
            entered: asyncio.Event,
        ) -> None:
            async with pool.lease(
                owner_id
            ):
                entered.set()
                await (
                    release_first_three
                    .wait()
                )

        first_tasks = [
            asyncio.create_task(
                hold(
                    f"worker-{index}",
                    entered_first_three[index],
                )
            )
            for index in range(3)
        ]

        await asyncio.gather(
            *(
                event.wait()
                for event
                in entered_first_three
            )
        )

        async def use_fourth(
        ) -> None:
            async with pool.lease(
                "worker-3"
            ):
                fourth_entered.set()

        fourth_task = (
            asyncio.create_task(
                use_fourth()
            )
        )

        await asyncio.sleep(
            0.05
        )

        self.assertEqual(
            pool.active_count,
            3,
        )
        self.assertFalse(
            fourth_entered.is_set()
        )

        release_first_three.set()

        await asyncio.gather(
            *first_tasks,
            fourth_task,
        )

        self.assertTrue(
            fourth_entered.is_set()
        )
        self.assertEqual(
            pool.active_count,
            0,
        )
        self.assertEqual(
            len(created),
            4,
        )
        self.assertEqual(
            len(
                {
                    runtime.profile_path
                    for runtime
                    in created
                }
            ),
            4,
        )
        self.assertTrue(
            all(
                runtime.stopped
                for runtime
                in created
            )
        )

    async def test_same_owner_cannot_hold_two_sessions(
        self,
    ) -> None:
        pool = PlaywrightMCPPool(
            max_sessions=2,
            runtime_factory=(
                _FakePlaywrightRuntime
            ),
        )

        async with pool.lease(
            "same-worker"
        ):
            with self.assertRaises(
                RuntimeError
            ):
                async with pool.lease(
                    "same-worker"
                ):
                    self.fail(
                        "重复owner不应拿到Session。"
                    )

            self.assertEqual(
                pool.active_count,
                1,
            )

    async def test_primary_facade_preserves_existing_runtime_api(
        self,
    ) -> None:
        created: list[
            _FakePlaywrightRuntime
        ] = []

        def factory(
            **kwargs,
        ) -> _FakePlaywrightRuntime:
            runtime = _FakePlaywrightRuntime(
                **kwargs
            )
            created.append(
                runtime
            )
            return runtime

        pool = PlaywrightMCPPool(
            max_sessions=3,
            runtime_factory=factory,
        )

        await pool.start()

        self.assertTrue(
            pool.is_started
        )
        self.assertEqual(
            pool.tools,
            [
                "tool-for-conversation-runtime"
            ],
        )
        self.assertEqual(
            created[0].profile_path,
            PLAYWRIGHT_PROFILE_PATH,
        )
        self.assertEqual(
            created[0].output_path,
            PLAYWRIGHT_OUTPUT_PATH,
        )

        await pool.reset_page()

        self.assertEqual(
            created[0].reset_count,
            1,
        )

        await pool.stop()

        self.assertFalse(
            pool.is_started
        )
        self.assertTrue(
            created[0].stopped
        )
        self.assertEqual(
            pool.active_count,
            0,
        )

    def test_pool_rejects_more_than_hard_max(
        self,
    ) -> None:
        with self.assertRaises(
            ValueError
        ):
            PlaywrightMCPPool(
                max_sessions=4,
                runtime_factory=(
                    _FakePlaywrightRuntime
                ),
            )


if __name__ == "__main__":
    unittest.main()
