"""Cache Phoenix 19.4's official WASM asset before offline/local server startup.

Uses the same URL and SHA-256 as Phoenix's download helper. No library patch,
TLS override, dummy binary, or disabled integrity check is involved.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import urllib.request

URL = ("https://github.com/vmware-labs/webassembly-language-runtimes/releases/download/"
       "python%2F3.12.0%2B20231211-040d5a6/python-3.12.0.wasm")
SHA256 = "e5dc5a398b07b54ea8fdb503bf68fb583d533f10ec3f930963e02b9505f7a763"


def main():
    destination = Path(__file__).resolve().parents[2] / ".agent/phoenix/wasm/python-3.12.0.wasm"
    started = time.monotonic()
    cached = destination.is_file() and hashlib.sha256(destination.read_bytes()).hexdigest() == SHA256
    if not cached:
        chunks = []
        with urllib.request.urlopen(URL, timeout=30) as response:
            while True:
                if time.monotonic() - started > 90:
                    raise TimeoutError("Phoenix asset transfer exceeded the 90-second read-loop budget")
                block = response.read(1024 * 1024)
                if not block:
                    break
                chunks.append(block)
        data = b"".join(chunks)
        if hashlib.sha256(data).hexdigest() != SHA256:
            raise ValueError("Official Phoenix WASM asset failed SHA-256 verification")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".wasm.verified-tmp")
        temporary.write_bytes(data)
        temporary.replace(destination)
    print(json.dumps({"path": str(destination), "sha256": SHA256,
                      "bytes": destination.stat().st_size, "already_cached": cached,
                      "elapsed_seconds": round(time.monotonic() - started, 3)}))


if __name__ == "__main__":
    main()
