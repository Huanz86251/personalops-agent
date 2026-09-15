"""An owned, disposable Docker world. No host mounts, network, or model secrets."""
from __future__ import annotations

from collections import deque
import json
import os
import queue
import subprocess
import threading
import time
import uuid


class WorldError(RuntimeError):
    pass


def docker_prefix(wsl="Ubuntu"):
    if os.name != "nt":
        return ["docker"]
    prefix = ["wsl", "-d", wsl, "--exec"]
    binary = subprocess.check_output(
        prefix + ["sh", "-lc", "command -v docker"], text=True, encoding="utf-8", errors="replace", timeout=30,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW,
    ).strip()
    if not binary.startswith("/") or "\n" in binary:
        raise WorldError("Cannot locate Docker in WSL")
    return prefix + [binary]


class DockerWorld:
    def __init__(self, image="personalops-appworld:0.1.3.post1", *, wsl="Ubuntu",
                 timeout_seconds=120):
        self.prefix = docker_prefix(wsl)
        self.name = "personalops-eval-" + uuid.uuid4().hex
        self.timeout_seconds = timeout_seconds
        self.responses = queue.Queue()
        self.request_lock = threading.RLock()
        self.stderr_tail = deque(maxlen=100)
        self.closed = False
        creation = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        for attempt in range(6):
            info = subprocess.run(self.prefix + ["image", "inspect", image, "--format", "{{.Id}}"],
                                  capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30, **creation)
            if info.returncode == 0:
                break
            detail = (info.stderr or "").replace("\x00", "").strip()
            # WSL may return before its Docker daemon has finished starting.
            # Retry only this read operation, never world execution or model calls.
            if "docker.sock" not in detail or attempt == 5:
                raise WorldError("Docker image inspection failed: " + detail[-1000:])
            time.sleep(2)
        self.image_id = info.stdout.strip()
        self.process = subprocess.Popen(
            self.prefix + [
                "run", "--rm", "-i", "--name", self.name,
                "--network", "none", "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges", "--read-only",
                "--memory", "2g", "--cpus", "2", "--pids-limit", "128",
                "--tmpfs", "/tmp:rw,nosuid,nodev,size=256m,uid=10001,gid=10001",
                "--tmpfs", "/world/experiments:rw,nosuid,nodev,size=512m,uid=10001,gid=10001",
                self.image_id,
            ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", bufsize=1, **creation,
        )
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()

    def _read_stdout(self):
        for line in self.process.stdout:
            try:
                self.responses.put(json.loads(line))
            except ValueError:
                self.responses.put({"ok": False, "error": "Non-JSON worker output"})
        self.responses.put({"ok": False, "error": "Worker exited"})

    def _read_stderr(self):
        for line in self.process.stderr:
            self.stderr_tail.append(line)

    def request(self, op, **parameters):
        with self.request_lock:
            return self._request(op, **parameters)

    def _request(self, op, **parameters):
        if self.closed:
            raise WorldError("World is closed")
        try:
            self.process.stdin.write(json.dumps({"op": op, **parameters}) + "\n")
            self.process.stdin.flush()
            response = self.responses.get(timeout=self.timeout_seconds)
        except (BrokenPipeError, OSError, queue.Empty) as exc:
            # A timed-out mutation has unknown state. Never retry it silently.
            self.close()
            raise WorldError("Worker transport failed; trial aborted without retry") from exc
        if not response.get("ok"):
            raise WorldError(response.get("error", "Unknown worker failure"))
        return response["output"]

    def execute(self, code):
        return self.request("execute", code=code)


    def isolation_report(self):
        creation = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        result = subprocess.run(self.prefix + ["inspect", self.name],
                                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30, **creation)
        if result.returncode:
            raise WorldError("Cannot verify container isolation")
        info = json.loads(result.stdout)[0]
        host = info["HostConfig"]
        env_names = [value.split("=", 1)[0] for value in info["Config"].get("Env", [])]
        return {
            "network_mode": host["NetworkMode"],
            "host_bind_mounts": [m["Destination"] for m in info.get("Mounts", []) if m["Type"] == "bind"],
            "privileged": host["Privileged"],
            "readonly_rootfs": host["ReadonlyRootfs"],
            "user": info["Config"]["User"],
            "cap_drop": host.get("CapDrop", []),
            "memory_bytes": host["Memory"],
            "pids_limit": host["PidsLimit"],
            "model_credentials_present": any(name.endswith("API_KEY") for name in env_names),
        }

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.process.stdin:
            self.process.stdin.close()
        # Target only this object's randomly named container, never other work.
        creation = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        subprocess.run(self.prefix + ["rm", "-f", self.name], capture_output=True,
                       timeout=30, **creation)
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
