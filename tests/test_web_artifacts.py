"""Offline tests for bounded Web Worker artifact downloads."""

from pathlib import Path

import httpx
import pytest

from tools.web_artifacts import WebDownloadError, download_to_record


@pytest.fixture(autouse=True)
def offline_dns(monkeypatch):
    # No external DNS in these transport tests. Loopback keeps its true address.
    import socket
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, port, **kw: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1" if host == "127.0.0.1" else "93.184.216.34", port))
    ])


def test_redirect_cannot_enter_loopback(tmp_path):
    calls = []
    def respond(request):
        calls.append(str(request.url))
        return httpx.Response(302, headers={"location": "http://127.0.0.1/private"}, request=request)
    with pytest.raises(WebDownloadError) as caught:
        download_to_record(url="https://example.com/start", worker_id="w", tool_call_id="t",
                           max_bytes=1000, root=tmp_path, transport=httpx.MockTransport(respond))
    assert caught.value.failure.code == "UNSAFE_DESTINATION"
    assert calls == ["https://example.com/start"]


def test_login_html_is_not_a_successful_pdf(tmp_path):
    with pytest.raises(WebDownloadError) as caught:
        download_to_record(url="https://example.com/report.pdf", worker_id="w", tool_call_id="t",
            max_bytes=1000, root=tmp_path, transport=httpx.MockTransport(
                lambda request: httpx.Response(200, headers={"content-type": "text/html"}, content=b"<html>Login</html>", request=request)))
    assert caught.value.failure.code == "UNEXPECTED_CONTENT"
    assert not any(path.is_file() for path in tmp_path.rglob("*"))


def test_download_streams_file_and_returns_runtime_metadata(tmp_path: Path) -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            headers={"content-type": "text/plain; charset=utf-8"},
            content=b"verified artifact",
            request=request,
        )
    )
    record = download_to_record(
        url="https://example.com/notes.txt",
        worker_id="web-worker-1",
        tool_call_id="download-call-1",
        max_bytes=1024,
        root=tmp_path,
        transport=transport,
    )

    stored = Path(record.storage_path)
    assert stored.read_bytes() == b"verified artifact"
    assert stored.is_relative_to(tmp_path)
    assert record.candidate_id.startswith("download-")
    assert record.tool_call_id == "download-call-1"
    assert record.size_bytes == 17


def test_download_rejects_stream_that_crosses_limit_and_removes_partial(
    tmp_path: Path,
) -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            content=b"0123456789",
            request=request,
        )
    )
    with pytest.raises(WebDownloadError, match="file limit"):
        download_to_record(
            url="https://example.com/large.bin",
            worker_id="web-worker-2",
            tool_call_id="download-call-2",
            max_bytes=5,
            root=tmp_path,
            transport=transport,
        )

    assert not list(tmp_path.rglob("*.part"))
    assert not [path for path in tmp_path.rglob("*") if path.is_file()]


def test_limit_error_exposes_machine_readable_reason(tmp_path: Path) -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            headers={"content-length": "101"},
            content=b"ignored",
            request=request,
        )
    )
    with pytest.raises(WebDownloadError) as captured:
        download_to_record(
            url="https://example.com/large.bin",
            worker_id="web-worker-structured-error",
            tool_call_id="download-call-structured-error",
            max_bytes=100,
            root=tmp_path,
            transport=transport,
        )

    failure = captured.value.failure
    assert failure.code == "FILE_TOO_LARGE"
    assert failure.stage == "RESPONSE_HEADERS"
    assert failure.limit_bytes == 100
    assert failure.announced_bytes == 101
    assert failure.retryable is False


def test_download_rejects_non_http_url(tmp_path: Path) -> None:
    with pytest.raises(WebDownloadError, match="HTTP or HTTPS"):
        download_to_record(
            url="file:///etc/passwd",
            worker_id="web-worker-3",
            tool_call_id="download-call-3",
            max_bytes=1024,
            root=tmp_path,
        )
