import base64
import json
import unittest
from types import SimpleNamespace
from langchain_core.tools import ToolException

from langchain_core.utils.function_calling import convert_to_openai_tool

from tools import ALL_TOOLS
from tools.desktop_tools import capture_desktop_screenshot
from toolsets import DEFAULT_TOOLSET_REGISTRY


class DesktopToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_schema_hides_runtime_and_group_is_registered(self):
        schema = convert_to_openai_tool(capture_desktop_screenshot)["function"]
        self.assertNotIn("runtime", schema["parameters"]["properties"])
        self.assertIn("output_path", schema["parameters"]["properties"])
        resolution = DEFAULT_TOOLSET_REGISTRY.resolve("DESKTOP_OBSERVATION", ALL_TOOLS)
        self.assertTrue(resolution.is_available)
        self.assertEqual(
            set(resolution.tool_names),
            {"capture_desktop_screenshot", "ocr_image"},
        )

    async def test_real_windows_capture_returns_task_jpeg(self):
        runtime = SimpleNamespace(
            state={"files": {}},
            tool_call_id="desktop-smoke",
        )
        try:
            command = await capture_desktop_screenshot.coroutine(
                output_path="/artifacts/desktop-smoke.jpg",
                runtime=runtime,
                all_screens=False,
                quality=55,
            )
        except ToolException as error:
            if "screen grab failed" in str(error):
                self.skipTest("No interactive Windows desktop is available in this test session.")
            raise
        record = command.update["files"]["/artifacts/desktop-smoke.jpg"]
        data = base64.b64decode(record["content"], validate=True)
        metadata = json.loads(command.update["messages"][0].content)
        self.assertTrue(data.startswith(b"\xff\xd8\xff"))
        self.assertGreater(len(data), 1000)
        self.assertGreater(metadata["width"], 0)
        self.assertGreater(metadata["height"], 0)
        self.assertEqual(metadata["status"], "created")


if __name__ == "__main__":
    unittest.main()
