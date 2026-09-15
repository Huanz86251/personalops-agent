"""Small dependency-free adapter for native Windows toast notifications."""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
import os


@dataclass(frozen=True, slots=True)
class NotificationReceipt:
    accepted_by_windows: bool
    provider: str
    detail: str


class WindowsToastNotifier:
    """Submit ToastGeneric XML through the built-in Windows Runtime API.

    A successful process exit means Windows accepted the toast request. Focus
    Assist, user notification settings, or a disconnected desktop session can
    still prevent the user from seeing it, so the receipt deliberately does
    not claim visual delivery.
    """

    def __init__(
        self,
        *,
        app_id: str = "PersonalOps.Agent",
        powershell_path: str = "powershell.exe",
        timeout_seconds: float = 15.0,
    ) -> None:
        self.app_id = str(app_id).strip()
        self.powershell_path = str(powershell_path).strip()
        self.timeout_seconds = timeout_seconds
        if not self.app_id or not self.powershell_path:
            raise ValueError("Windows通知的app_id和PowerShell路径不能为空。")

    @staticmethod
    def _encoded_script(*, app_id: str, title: str, message: str, tag: str) -> str:
        def encoded(value: str) -> str:
            return base64.b64encode(value.encode("utf-8")).decode("ascii")

        script = f"""
$ErrorActionPreference = 'Stop'
function Decode([string]$value) {{
  [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($value))
}}
$appId = Decode '{encoded(app_id)}'
$title = [Security.SecurityElement]::Escape((Decode '{encoded(title)}'))
$message = [Security.SecurityElement]::Escape((Decode '{encoded(message)}'))
$tag = Decode '{encoded(tag)}'
[void][Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType=WindowsRuntime]
[void][Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType=WindowsRuntime]
$xml = New-Object Windows.Data.Xml.Dom.XmlDocument
$xml.LoadXml("<toast><visual><binding template='ToastGeneric'><text>$title</text><text>$message</text></binding></visual></toast>")
$toast = New-Object Windows.UI.Notifications.ToastNotification $xml
if ($tag) {{ $toast.Tag = $tag.Substring(0, [Math]::Min(16, $tag.Length)) }}
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($appId).Show($toast)
""".strip()
        return base64.b64encode(script.encode("utf-16le")).decode("ascii")

    async def send(self, *, title: str, message: str, tag: str = "") -> NotificationReceipt:
        if os.name != "nt":
            raise RuntimeError("Windows原生通知只能在Windows主机上发送。")
        normalized_title = str(title).strip()
        normalized_message = str(message).strip()
        if not normalized_title or not normalized_message:
            raise ValueError("通知标题和正文不能为空。")
        process = await asyncio.create_subprocess_exec(
            self.powershell_path,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-EncodedCommand",
            self._encoded_script(
                app_id=self.app_id,
                title=normalized_title,
                message=normalized_message,
                tag=str(tag).strip(),
            ),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self.timeout_seconds
            )
        except TimeoutError:
            process.kill()
            await process.communicate()
            raise RuntimeError("Windows通知提交超时。")
        if process.returncode != 0:
            detail = (stderr or stdout).decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Windows拒绝通知请求：{detail or 'unknown error'}")
        return NotificationReceipt(
            accepted_by_windows=True,
            provider="Windows.UI.Notifications",
            detail="通知请求已提交给Windows；专注助手和系统设置仍可能隐藏显示。",
        )
