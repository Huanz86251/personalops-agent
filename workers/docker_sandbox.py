"""Deep Agents Docker sandbox backend and per-attempt CODE lifecycle."""

from __future__ import annotations
from runtime_tracing import operation

import json
import os
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox

from evals.appworld.protocol import docker_prefix
from path import PROJECT_ROOT

if TYPE_CHECKING:
    from config import CodeSandboxSettings


CodeSandboxRole = Literal["WORKER", "REVIEWER"]

CODE_SANDBOX_IMAGE = "personalops-code-sandbox:py312-v2"
CODE_SANDBOX_LABEL = "personalops.code_sandbox=true"
CODE_SANDBOX_OUTPUT_MAX_CHARS = 200_000


class CodeSandboxError(RuntimeError):
    """Raised when an owned CODE sandbox cannot be prepared or controlled."""


def docker_install_guidance() -> str:
    """Return copy-pasteable setup instructions without installing software."""

    if os.name == "nt":
        return (
            "Docker is not available. Install Docker Desktop with:\n"
            "  winget install --exact --id Docker.DockerDesktop\n"
            "Then start Docker Desktop, enable WSL integration for Ubuntu, "
            "and rerun:\n"
            "  python scripts/setup_code_sandbox.py"
        )
    return (
        "Docker is not available. On Ubuntu/Debian install Docker Engine with:\n"
        "  curl -fsSL https://get.docker.com -o get-docker.sh\n"
        "  sudo sh get-docker.sh\n"
        "Then rerun:\n"
        "  python scripts/setup_code_sandbox.py"
    )


def _decode_docker_stream(
    payload: bytes,
    *,
    strip_wsl_launcher_warning: bool = False,
) -> str:
    """Decode Docker output and remove WSL's mixed UTF-16 launcher warning."""

    content = payload
    if strip_wsl_launcher_warning and os.name == "nt" and b"\x00" in content[:512]:
        boundary = content.find(b"\n\x00")
        if boundary >= 0:
            content = content[boundary + 2 :]
    return content.decode("utf-8", errors="replace").replace("\x00", "")


@dataclass(frozen=True)
class CodeSandboxPolicy:
    image: str = CODE_SANDBOX_IMAGE
    wsl_distribution: str = "Ubuntu"
    auto_build: bool = True
    memory_mb: int = 1536
    cpu_count: int = 2
    pids_limit: int = 128
    execute_timeout_seconds: int = 120

    @classmethod
    def from_settings(
        cls,
        settings: CodeSandboxSettings,
    ) -> CodeSandboxPolicy:
        """Convert the application config into the sandbox boundary object."""

        return cls(
            image=settings.image,
            wsl_distribution=settings.wsl_distribution,
            auto_build=settings.auto_build,
            memory_mb=settings.memory_mb,
            cpu_count=settings.cpu_count,
            pids_limit=settings.pids_limit,
            execute_timeout_seconds=settings.execute_timeout_seconds,
        )

    def __post_init__(self) -> None:
        if self.memory_mb < 256 or self.memory_mb > 4096:
            raise ValueError("CODE sandbox memory_mb must be between 256 and 4096")
        if self.cpu_count < 1 or self.cpu_count > 4:
            raise ValueError("CODE sandbox cpu_count must be between 1 and 4")
        if self.pids_limit < 32 or self.pids_limit > 256:
            raise ValueError("CODE sandbox pids_limit must be between 32 and 256")
        if self.execute_timeout_seconds < 10 or self.execute_timeout_seconds > 600:
            raise ValueError(
                "CODE sandbox execute timeout must be between 10 and 600 seconds"
            )


class DockerCLI:
    """Small subprocess adapter reusing the verified AppWorld Docker lookup."""

    def __init__(self, wsl_distribution: str = "Ubuntu") -> None:
        self.wsl_distribution = wsl_distribution
        try:
            self.prefix = docker_prefix(wsl_distribution)
        except (OSError, subprocess.SubprocessError) as exc:
            raise CodeSandboxError(docker_install_guidance()) from exc
        self._creation = (
            {"creationflags": subprocess.CREATE_NO_WINDOW}
            if os.name == "nt"
            else {}
        )

    def run(
        self,
        arguments: list[str],
        *,
        input_bytes: bytes | None = None,
        timeout: int = 60,
    ) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            [*self.prefix, *arguments],
            input=input_bytes,
            capture_output=True,
            timeout=timeout,
            **self._creation,
        )

    def require_success(
        self,
        arguments: list[str],
        *,
        input_bytes: bytes | None = None,
        timeout: int = 60,
        operation: str,
    ) -> subprocess.CompletedProcess[bytes]:
        result = self.run(
            arguments,
            input_bytes=input_bytes,
            timeout=timeout,
        )
        if result.returncode:
            detail = _decode_docker_stream(
                result.stderr or result.stdout,
                strip_wsl_launcher_warning=True,
            ).strip()
            raise CodeSandboxError(
                f"{operation} failed: {detail[-2000:] or 'unknown Docker error'}"
            )
        return result

    def host_path(self, path: Path) -> str:
        resolved = path.resolve()
        if os.name != "nt":
            return str(resolved)
        result = subprocess.run(
            [
                "wsl",
                "-d",
                self.wsl_distribution,
                "--exec",
                "wslpath",
                "-a",
                str(resolved),
            ],
            capture_output=True,
            timeout=30,
            **self._creation,
        )
        if result.returncode:
            raise CodeSandboxError(
                "Cannot translate the Windows path for WSL Docker."
            )
        translated = result.stdout.decode("utf-8", errors="replace").strip()
        if not translated.startswith("/") or "\n" in translated:
            raise CodeSandboxError("WSL returned an invalid host path.")
        return translated

    def ensure_daemon(self) -> None:
        if self.run(["version", "--format", "{{.Server.Version}}"], timeout=30).returncode == 0:
            return

        if os.name == "nt":
            prefix = [
                "wsl",
                "-d",
                self.wsl_distribution,
                "-u",
                "root",
                "--exec",
            ]
            candidates = (
                ["systemctl", "start", "docker"],
                ["service", "docker", "start"],
                ["snap", "start", "docker"],
            )
            for command in candidates:
                try:
                    subprocess.run(
                        [*prefix, *command],
                        capture_output=True,
                        timeout=30,
                        **self._creation,
                    )
                except (OSError, subprocess.SubprocessError):
                    continue
                for _ in range(6):
                    if self.run(
                        ["version", "--format", "{{.Server.Version}}"],
                        timeout=30,
                    ).returncode == 0:
                        return
                    time.sleep(1)

        raise CodeSandboxError(
            "Docker was found, but Docker Engine is not running and could not "
            "be started automatically. Start Docker Desktop or the Docker "
            "service, then rerun `python scripts/setup_code_sandbox.py`."
        )


class DockerSandboxBackend(BaseSandbox):
    """Deep Agents backend whose primitive is `docker exec`."""

    def __init__(
        self,
        cli: DockerCLI,
        *,
        container_name: str,
        backend_id: str,
        default_timeout_seconds: int = 120,
    ) -> None:
        self.cli = cli
        self.container_name = container_name
        self._id = backend_id
        self.default_timeout_seconds = default_timeout_seconds

    @property
    def id(self) -> str:
        return self._id

    @operation('Sandbox / Execute', fields=('command', 'timeout'))
    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        effective_timeout = (
            self.default_timeout_seconds
            if timeout is None
            else max(int(timeout), 1)
        )
        result = self.cli.run(
            [
                "exec",
                self.container_name,
                "timeout",
                "--signal=KILL",
                f"{effective_timeout}s",
                "bash",
                "--noprofile",
                "--norc",
                "-lc",
                command,
            ],
            timeout=effective_timeout + 10,
        )
        output = (
            _decode_docker_stream(result.stdout)
            + _decode_docker_stream(
                result.stderr,
                strip_wsl_launcher_warning=True,
            )
        )
        truncated = len(output) > CODE_SANDBOX_OUTPUT_MAX_CHARS
        if truncated:
            half = CODE_SANDBOX_OUTPUT_MAX_CHARS // 2
            output = (
                output[:half]
                + "\n...[Docker output truncated]...\n"
                + output[-half:]
            )
        return ExecuteResponse(
            output=output,
            exit_code=result.returncode,
            truncated=truncated,
        )

    @operation('Sandbox / Upload Files', fields=())
    def upload_files(
        self,
        files: list[tuple[str, bytes]],
    ) -> list[FileUploadResponse]:
        script = (
            "import pathlib,sys; "
            "p=pathlib.Path(sys.argv[1]); "
            "p.parent.mkdir(parents=True,exist_ok=True); "
            "p.write_bytes(sys.stdin.buffer.read())"
        )
        responses: list[FileUploadResponse] = []
        for path, content in files:
            result = self.cli.run(
                [
                    "exec",
                    "-i",
                    self.container_name,
                    "python",
                    "-c",
                    script,
                    path,
                ],
                input_bytes=content,
                timeout=self.default_timeout_seconds,
            )
            error = None
            if result.returncode:
                error = _decode_docker_stream(
                    result.stderr,
                    strip_wsl_launcher_warning=True,
                )[-1000:]
            responses.append(FileUploadResponse(path=path, error=error))
        return responses

    @operation('Sandbox / Read Files', fields=('paths',))
    def download_files(
        self,
        paths: list[str],
    ) -> list[FileDownloadResponse]:
        script = (
            "import pathlib,sys; "
            "p=pathlib.Path(sys.argv[1]); "
            "sys.stdout.buffer.write(p.read_bytes())"
        )
        responses: list[FileDownloadResponse] = []
        for path in paths:
            result = self.cli.run(
                [
                    "exec",
                    self.container_name,
                    "python",
                    "-c",
                    script,
                    path,
                ],
                timeout=self.default_timeout_seconds,
            )
            if result.returncode:
                error = _decode_docker_stream(
                    result.stderr,
                    strip_wsl_launcher_warning=True,
                )[-1000:]
                responses.append(FileDownloadResponse(path=path, error=error))
            else:
                responses.append(
                    FileDownloadResponse(path=path, content=result.stdout)
                )
        return responses


@dataclass(frozen=True)
class CodeSandboxPair:
    pair_id: str
    workspace_id: str
    candidate_volume: str
    review_volume: str
    worker_container: str
    reviewer_container: str
    image: str
    active_role: CodeSandboxRole | None
    handoff_root: str | None = None


class CodeSandboxManager:
    """Own two serially-active containers and their persistent named volumes."""

    def __init__(
        self,
        policy: CodeSandboxPolicy | None = None,
        *,
        cli: DockerCLI | None = None,
    ) -> None:
        self.policy = policy or CodeSandboxPolicy()
        self.cli = cli or DockerCLI(self.policy.wsl_distribution)

    @operation('Sandbox / Check Availability', fields=())
    def ensure_ready(self) -> None:
        self.cli.ensure_daemon()
        inspect = self.cli.run(
            ["image", "inspect", self.policy.image],
            timeout=30,
        )
        if inspect.returncode == 0:
            return
        if not self.policy.auto_build:
            raise CodeSandboxError(
                f"CODE sandbox image is missing: {self.policy.image}"
            )
        self.build_image()

    def build_image(self, *, domestic_mirrors: bool = False) -> None:
        """Explicit setup-time rebuild; mirror selection never changes runtime networking."""
        context = PROJECT_ROOT / "docker" / "code-agent"
        mirror_args = []
        if domestic_mirrors:
            for value in (
                "DEBIAN_MIRROR=https://mirrors.huaweicloud.com/debian",
                "DEBIAN_SECURITY_MIRROR=https://mirrors.huaweicloud.com/debian-security",
                "PIP_INDEX_URL=https://mirrors.huaweicloud.com/repository/pypi/simple",
                "PLAYWRIGHT_DOWNLOAD_HOST=https://cdn.npmmirror.com/binaries/playwright",
            ):
                mirror_args.extend(["--build-arg", value])
        self.cli.require_success(
            [
                "build",
                "--progress",
                "plain",
                *mirror_args,
                "-t",
                self.policy.image,
                self.cli.host_path(context),
            ],
            # Browser binaries + Office dependencies are installed only here,
            # never during a Worker command. First builds can exceed 15 minutes.
            timeout=3600,
            operation="CODE sandbox image build",
        )

    @staticmethod
    def _safe_token(workspace_id: str) -> str:
        import hashlib

        return hashlib.sha256(workspace_id.encode("utf-8")).hexdigest()[:12]

    def _common_container_arguments(self, name: str) -> list[str]:
        return [
            "create",
            "--name",
            name,
            "--label",
            CODE_SANDBOX_LABEL,
            "--network",
            "none",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--read-only",
            "--memory",
            f"{self.policy.memory_mb}m",
            "--cpus",
            str(self.policy.cpu_count),
            "--pids-limit",
            str(self.policy.pids_limit),
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=256m,uid=10001,gid=10001",
            "--user",
            "10001:10001",
        ]

    def _initialize_volume(self, volume: str, target: str) -> None:
        self.cli.require_success(
            [
                "run",
                "--rm",
                "--network",
                "none",
                "--user",
                "0:0",
                "--mount",
                f"type=volume,source={volume},target={target}",
                self.policy.image,
                "sh",
                "-lc",
                f"chown -R 10001:10001 {target}",
            ],
            operation="CODE sandbox volume initialization",
        )

    @operation('Sandbox / Create Worker and Reviewer', fields=('workspace_id',))
    def create_pair(
        self,
        workspace_id: str,
        *,
        handoff_root: Path | None = None,
    ) -> CodeSandboxPair:
        normalized = str(workspace_id).strip()
        if not normalized:
            raise ValueError("workspace_id cannot be empty")
        self.ensure_ready()

        token = self._safe_token(normalized)
        nonce = uuid4().hex[:10]
        prefix = f"personalops-code-{token}-{nonce}"
        resolved_handoff: Path | None = None
        if handoff_root is not None:
            resolved_handoff = Path(handoff_root).resolve()
            if not resolved_handoff.is_dir():
                raise ValueError("CODE handoff root must be an existing directory")
        handoff_mount = (
            "type=bind,"
            f"source={self.cli.host_path(resolved_handoff)},"
            "target=/handoff,readonly"
            if resolved_handoff is not None
            else None
        )
        candidate_volume = f"{prefix}-candidate"
        review_volume = f"{prefix}-review"
        worker_container = f"{prefix}-worker"
        reviewer_container = f"{prefix}-reviewer"

        created: list[tuple[str, str]] = []
        try:
            for volume in (candidate_volume, review_volume):
                self.cli.require_success(
                    [
                        "volume",
                        "create",
                        "--label",
                        CODE_SANDBOX_LABEL,
                        volume,
                    ],
                    operation="CODE sandbox volume creation",
                )
                created.append(("volume", volume))
            self._initialize_volume(candidate_volume, "/workspace")
            self._initialize_volume(review_volume, "/review")

            worker_args = self._common_container_arguments(worker_container)
            worker_args.extend(
                [
                    "--workdir",
                    "/workspace",
                    "--mount",
                    (
                        "type=volume,"
                        f"source={candidate_volume},target=/workspace"
                    ),
                    self.policy.image,
                ]
            )
            if handoff_mount is not None:
                worker_args[1:1] = ["--mount", handoff_mount]
            self.cli.require_success(
                worker_args,
                operation="Code Worker container creation",
            )
            created.append(("container", worker_container))

            reviewer_args = self._common_container_arguments(
                reviewer_container
            )
            reviewer_args.extend(
                [
                    "--workdir",
                    "/review",
                    "--mount",
                    (
                        "type=volume,"
                        f"source={candidate_volume},target=/workspace,readonly"
                    ),
                    "--mount",
                    (
                        "type=volume,"
                        f"source={review_volume},target=/review"
                    ),
                    self.policy.image,
                ]
            )
            if handoff_mount is not None:
                reviewer_args[1:1] = ["--mount", handoff_mount]
            self.cli.require_success(
                reviewer_args,
                operation="Code Reviewer container creation",
            )
            created.append(("container", reviewer_container))
            self.cli.require_success(
                ["start", worker_container],
                operation="Code Worker container start",
            )
        except Exception:
            for kind, name in reversed(created):
                if kind == "container":
                    self.cli.run(["rm", "-f", name], timeout=30)
                else:
                    self.cli.run(["volume", "rm", "-f", name], timeout=30)
            raise

        return CodeSandboxPair(
            pair_id=prefix,
            workspace_id=normalized,
            candidate_volume=candidate_volume,
            review_volume=review_volume,
            worker_container=worker_container,
            reviewer_container=reviewer_container,
            image=self.policy.image,
            active_role="WORKER",
            handoff_root=(
                str(resolved_handoff) if resolved_handoff is not None else None
            ),
        )

    def worker_backend(self, pair: CodeSandboxPair) -> DockerSandboxBackend:
        return DockerSandboxBackend(
            self.cli,
            container_name=pair.worker_container,
            backend_id=f"{pair.pair_id}:worker",
            default_timeout_seconds=self.policy.execute_timeout_seconds,
        )

    def reviewer_backend(self, pair: CodeSandboxPair) -> DockerSandboxBackend:
        return DockerSandboxBackend(
            self.cli,
            container_name=pair.reviewer_container,
            backend_id=f"{pair.pair_id}:reviewer",
            default_timeout_seconds=self.policy.execute_timeout_seconds,
        )

    @operation('Sandbox / Transfer Ownership', fields=('pair', 'role'))
    def handoff(self, pair: CodeSandboxPair, role: CodeSandboxRole) -> CodeSandboxPair:
        if pair.active_role == role:
            return pair
        if pair.active_role == "WORKER":
            self.cli.require_success(
                ["stop", "--time", "5", pair.worker_container],
                operation="Code Worker container stop",
            )
        elif pair.active_role == "REVIEWER":
            self.cli.require_success(
                ["stop", "--time", "5", pair.reviewer_container],
                operation="Code Reviewer container stop",
            )

        target = (
            pair.worker_container
            if role == "WORKER"
            else pair.reviewer_container
        )
        self.cli.require_success(
            ["start", target],
            operation=f"Code {role.title()} container start",
        )
        return replace(pair, active_role=role)

    @operation('Sandbox / Freeze', fields=('pair',))
    def freeze(self, pair: CodeSandboxPair) -> CodeSandboxPair:
        if pair.active_role is not None:
            target = (
                pair.worker_container
                if pair.active_role == "WORKER"
                else pair.reviewer_container
            )
            self.cli.require_success(
                ["stop", "--time", "5", target],
                operation="CODE sandbox freeze",
            )
        return replace(pair, active_role=None)

    def _inspect_owned_resource(
        self,
        kind: Literal["container", "volume"],
        name: str,
    ) -> dict | None:
        arguments = (
            ["inspect", name]
            if kind == "container"
            else ["volume", "inspect", name]
        )
        result = self.cli.run(arguments, timeout=30)
        if result.returncode:
            detail = _decode_docker_stream(result.stderr, strip_wsl_launcher_warning=True).strip()
            if detail and not any(marker in detail.lower() for marker in ("no such object", "no such container", "no such volume")):
                raise CodeSandboxError(f"Cannot inspect Docker {kind} {name}: {detail[-1000:]}")
            return None
        records = json.loads(result.stdout.decode("utf-8"))
        if len(records) != 1:
            raise CodeSandboxError(
                f"Expected one Docker {kind} record for {name}"
            )
        record = records[0]
        labels = (
            record.get("Config", {}).get("Labels", {})
            if kind == "container"
            else record.get("Labels", {})
        ) or {}
        if labels.get("personalops.code_sandbox") != "true":
            raise CodeSandboxError(
                f"Refusing to recover unowned Docker {kind}: {name}"
            )
        return record

    def freeze_all(self, pair: CodeSandboxPair) -> CodeSandboxPair:
        """Stop both owned roles without trusting a stale active-role flag."""

        for container in (pair.worker_container, pair.reviewer_container):
            record = self._inspect_owned_resource("container", container)
            if record is None:
                raise CodeSandboxError(
                    f"CODE recovery container is missing: {container}"
                )
            if bool(record.get("State", {}).get("Running")):
                self.cli.require_success(
                    ["stop", "--time", "5", container],
                    operation="CODE recovery freeze",
                )
        return replace(pair, active_role=None)

    def import_review_snapshot(
        self,
        pair: CodeSandboxPair,
        source: Path,
    ) -> CodeSandboxPair:
        """Restore Reviewer-authored tests and notes into its private volume."""

        source_path = Path(source).resolve()
        if not source_path.is_dir():
            raise ValueError("CODE Reviewer snapshot must be an existing directory")
        pair = self.handoff(pair, "REVIEWER")
        self.cli.require_success(
            [
                "cp",
                f"{self.cli.host_path(source_path)}/.",
                f"{pair.reviewer_container}:/review",
            ],
            timeout=300,
            operation="CODE Reviewer snapshot import",
        )
        self.cli.require_success(
            [
                "exec",
                "--user",
                "0:0",
                pair.reviewer_container,
                "chown",
                "-R",
                "10001:10001",
                "/review",
            ],
            timeout=300,
            operation="CODE Reviewer snapshot ownership update",
        )
        return pair

    @operation('Sandbox / Recover Pair', fields=())
    def recover_pair(
        self,
        pair: CodeSandboxPair,
        *,
        candidate_snapshot: Path | None,
        reviewer_snapshot: Path | None,
    ) -> tuple[CodeSandboxPair, bool]:
        """Reuse an intact pair, or rebuild it solely from host snapshots.

        A partially present pair is never guessed at or silently deleted. That
        state requires an operator decision because its surviving resource may
        contain the newest copy of the work.
        """

        self.ensure_ready()
        resources = {
            ("container", pair.worker_container): self._inspect_owned_resource(
                "container", pair.worker_container
            ),
            ("container", pair.reviewer_container): self._inspect_owned_resource(
                "container", pair.reviewer_container
            ),
            ("volume", pair.candidate_volume): self._inspect_owned_resource(
                "volume", pair.candidate_volume
            ),
            ("volume", pair.review_volume): self._inspect_owned_resource(
                "volume", pair.review_volume
            ),
        }
        present = [record is not None for record in resources.values()]
        if all(present):
            worker = resources[("container", pair.worker_container)] or {}
            reviewer = resources[("container", pair.reviewer_container)] or {}
            if worker.get("Config", {}).get("Image") != pair.image:
                raise CodeSandboxError("Recovered Code Worker image does not match")
            if reviewer.get("Config", {}).get("Image") != pair.image:
                raise CodeSandboxError("Recovered Code Reviewer image does not match")
            mounts = {
                item.get("Destination"): item.get("Name")
                for item in worker.get("Mounts", [])
            }
            reviewer_mounts = {
                item.get("Destination"): item.get("Name")
                for item in reviewer.get("Mounts", [])
            }
            if mounts.get("/workspace") != pair.candidate_volume:
                raise CodeSandboxError("Recovered Worker volume does not match")
            if (
                reviewer_mounts.get("/workspace") != pair.candidate_volume
                or reviewer_mounts.get("/review") != pair.review_volume
            ):
                raise CodeSandboxError("Recovered Reviewer volumes do not match")
            return self.freeze_all(pair), False

        if any(present):
            names = ", ".join(
                name
                for (_, name), record in resources.items()
                if record is not None
            )
            raise CodeSandboxError(
                "CODE sandbox pair is only partially present; preserving "
                f"surviving resources for inspection: {names}"
            )

        if candidate_snapshot is None or not candidate_snapshot.is_dir():
            raise CodeSandboxError(
                "CODE containers are gone and no candidate snapshot can rebuild them"
            )
        rebuilt = self.create_pair(
            pair.workspace_id,
            handoff_root=(Path(pair.handoff_root) if pair.handoff_root else None),
        )
        try:
            rebuilt = self.copy_source(rebuilt, candidate_snapshot)
            if reviewer_snapshot is not None and reviewer_snapshot.is_dir():
                rebuilt = self.import_review_snapshot(rebuilt, reviewer_snapshot)
            return self.freeze_all(rebuilt), True
        except BaseException:
            self.cleanup(rebuilt)
            raise

    @operation('Sandbox / Import Candidate', fields=('pair', 'source'))
    def copy_source(
        self,
        pair: CodeSandboxPair,
        source: Path,
    ) -> CodeSandboxPair:
        source_path = source.resolve()
        if not source_path.is_dir():
            raise ValueError("CODE sandbox source must be an existing directory")
        pair = self.handoff(pair, "WORKER")
        self.cli.require_success(
            [
                "cp",
                f"{self.cli.host_path(source_path)}/.",
                f"{pair.worker_container}:/workspace",
            ],
            timeout=300,
            operation="CODE sandbox source import",
        )
        # docker cp imports root-owned files. The Worker deliberately has
        # cap-drop ALL, so even exec --user 0 cannot chown them. Reuse the
        # short-lived host-controlled initializer; never elevate the Worker.
        self._initialize_volume(pair.candidate_volume, "/workspace")
        return pair

    @operation('Sandbox / Export Candidate', fields=('pair', 'destination'))
    def export_candidate(self, pair: CodeSandboxPair, destination: Path) -> Path:
        return self._export_container_directory(
            container=pair.worker_container,
            source="/workspace",
            destination=destination,
            operation="CODE sandbox candidate export",
        )

    @operation('Sandbox / Export Review Evidence', fields=('pair', 'destination'))
    def export_review(self, pair: CodeSandboxPair, destination: Path) -> Path:
        return self._export_container_directory(
            container=pair.reviewer_container,
            source="/review",
            destination=destination,
            operation="CODE sandbox review export",
        )

    def _export_container_directory(
        self,
        *,
        container: str,
        source: str,
        destination: Path,
        operation: str,
    ) -> Path:
        target = destination.resolve()
        if target.exists():
            if not target.is_dir() or any(target.iterdir()):
                raise ValueError(
                    "sandbox export destination must be an empty directory"
                )
        target.mkdir(parents=True, exist_ok=True)
        self.cli.require_success(
            [
                "cp",
                f"{container}:{source}/.",
                self.cli.host_path(target),
            ],
            timeout=300,
            operation=operation,
        )
        return target

    def isolation_report(self, pair: CodeSandboxPair) -> dict:
        result = self.cli.require_success(
            ["inspect", pair.worker_container, pair.reviewer_container],
            operation="CODE sandbox isolation inspection",
        )
        records = json.loads(result.stdout.decode("utf-8"))
        report = {}
        for record in records:
            name = record["Name"].lstrip("/")
            host = record["HostConfig"]
            mounts = {
                item["Destination"]: bool(item.get("RW"))
                for item in record.get("Mounts", [])
            }
            report[name] = {
                "network_mode": host.get("NetworkMode"),
                "privileged": host.get("Privileged"),
                "readonly_rootfs": host.get("ReadonlyRootfs"),
                "cap_drop": host.get("CapDrop", []),
                "memory_bytes": host.get("Memory"),
                "pids_limit": host.get("PidsLimit"),
                "mounts": mounts,
                "user": record.get("Config", {}).get("User"),
            }
        return report

    @operation('Sandbox / Cleanup', fields=('pair',))
    def cleanup(self, pair: CodeSandboxPair) -> None:
        errors = []
        for args in (["rm", "-f", pair.worker_container], ["rm", "-f", pair.reviewer_container],
                     ["volume", "rm", "-f", pair.candidate_volume], ["volume", "rm", "-f", pair.review_volume]):
            result = self.cli.run(args, timeout=30)
            if result.returncode:
                detail = _decode_docker_stream(result.stderr, strip_wsl_launcher_warning=True).strip()
                if not any(marker in detail.lower() for marker in ("no such object", "no such container", "no such volume")):
                    errors.append(detail or f"Docker removal failed: {args[-1]}")
        if errors:
            raise CodeSandboxError("CODE cleanup not confirmed: " + "; ".join(errors))


__all__ = [
    "CODE_SANDBOX_IMAGE",
    "CodeSandboxError",
    "CodeSandboxManager",
    "CodeSandboxPair",
    "CodeSandboxPolicy",
    "DockerCLI",
    "DockerSandboxBackend",
    "docker_install_guidance",
]
