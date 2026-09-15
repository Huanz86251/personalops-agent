"""Local desktop observation tools owned by PersonalOps itself."""

from __future__ import annotations

import asyncio
import io
import json
import sys
from pathlib import PurePosixPath

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import ToolException
from pydantic import BaseModel, ConfigDict, Field

from tools.local_native import MAX_BYTES, OCR_DEPENDENCY_ROOT, _save, _virtual


class CaptureDesktopScreenshotInput(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    output_path: str = Field(
        default="/artifacts/desktop-screenshot.jpg",
        min_length=16,
        max_length=240,
        description=(
            "当前任务中的新JPG输出路径，必须位于/artifacts/，例如"
            "/artifacts/desktop-before-click.jpg。"
        ),
    )
    all_screens: bool = Field(
        default=True,
        description="true截取整个虚拟桌面（含多显示器）；false只截主显示器。",
    )
    quality: int = Field(
        default=75,
        ge=40,
        le=90,
        description="JPEG质量40到90；默认75用于控制任务文件大小。",
    )
    runtime: ToolRuntime


def _image_grab():
    try:
        from PIL import ImageGrab
    except ImportError:
        dependency_root = str(OCR_DEPENDENCY_ROOT)
        if dependency_root not in sys.path and OCR_DEPENDENCY_ROOT.is_dir():
            sys.path.append(dependency_root)
        try:
            from PIL import ImageGrab
        except ImportError as error:
            raise ToolException(
                "桌面截图依赖尚未安装；请先运行 scripts/setup_ocr.py。"
            ) from error
    return ImageGrab


def _capture_jpeg(*, all_screens: bool, quality: int) -> tuple[bytes, tuple[int, int]]:
    if sys.platform != "win32":
        raise ToolException("桌面截图工具当前只支持Windows。")
    image = _image_grab().grab(all_screens=all_screens)
    try:
        size = tuple(image.size)
        if size[0] <= 0 or size[1] <= 0:
            raise ToolException("Windows返回了空截图。")
        if image.mode != "RGB":
            image = image.convert("RGB")
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=quality, optimize=True)
        data = output.getvalue()
        if not data or len(data) > MAX_BYTES:
            raise ToolException(
                "桌面截图超过20 MiB任务文件上限；请降低quality或只截主显示器。"
            )
        return data, size
    finally:
        image.close()


@tool(args_schema=CaptureDesktopScreenshotInput)
async def capture_desktop_screenshot(
    output_path: str,
    runtime: ToolRuntime,
    all_screens: bool = True,
    quality: int = 75,
):
    """截取当前Windows桌面并保存为当前任务的JPG artifact。只在用户明确要求查看或截屏时调用；不得持续监控、后台连拍或把截图视为已发送。截图可再交给ocr_image读取文字，发送到飞书仍需独立的文件导出授权和确认。"""

    try:
        normalized = _virtual(output_path)
        suffix = PurePosixPath(normalized).suffix.lower()
        if not normalized.startswith("/artifacts/") or suffix not in {".jpg", ".jpeg"}:
            raise ValueError("截图输出必须是/artifacts/下的新.jpg或.jpeg路径。")
        data, size = await asyncio.to_thread(
            _capture_jpeg,
            all_screens=all_screens,
            quality=quality,
        )
        command = _save(normalized, data, runtime)
        # The binary stays in task state.  Dimensions are intentionally encoded
        # in the tool artifact metadata instead of exposing a host temp path.
        message = command.update["messages"][0]
        metadata = json.loads(message.content)
        metadata.update(
            width=size[0],
            height=size[1],
            all_screens=all_screens,
        )
        message.content = json.dumps(metadata, ensure_ascii=False)
        return command
    except (OSError, ValueError, RuntimeError) as error:
        raise ToolException(str(error)) from error


capture_desktop_screenshot.handle_tool_error = True
capture_desktop_screenshot.metadata = {
    "capability": "desktop_observation",
    "execution": "local_host",
    "network": "none",
    "capture": "windows_desktop_screenshot",
    "requires_explicit_user_intent": True,
    "contains_potentially_sensitive_pixels": True,
    "external_side_effect": False,
}


DESKTOP_TOOLS = [capture_desktop_screenshot]
