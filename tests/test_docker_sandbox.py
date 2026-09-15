"""Offline lifecycle tests for the Deep Agents Docker sandbox adapter."""

from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from workers.docker_sandbox import (
    CodeSandboxManager,
    CodeSandboxPair,
    CodeSandboxError,
    CodeSandboxPolicy,
    DockerSandboxBackend,
    _decode_docker_stream,
)


def completed(
    arguments: list[str],
    *,
    stdout: bytes = b"",
    stderr: bytes = b"",
    returncode: int = 0,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.CompletedProcess(
        arguments,
        returncode,
        stdout=stdout,
        stderr=stderr,
    )


class FakeDockerCLI:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []
        self.daemon_checks = 0
        self.inspect_output: bytes | None = None

    def ensure_daemon(self) -> None:
        self.daemon_checks += 1

    def host_path(self, path: Path) -> str:
        return "/host/" + path.name

    def run(
        self,
        arguments: list[str],
        *,
        input_bytes: bytes | None = None,
        timeout: int = 60,
    ) -> subprocess.CompletedProcess[bytes]:
        self.commands.append(list(arguments))
        if arguments[:2] == ["image", "inspect"]:
            return completed(arguments, stdout=b"[]")
        return completed(arguments)

    def require_success(
        self,
        arguments: list[str],
        *,
        input_bytes: bytes | None = None,
        timeout: int = 60,
        operation: str,
    ) -> subprocess.CompletedProcess[bytes]:
        self.commands.append(list(arguments))
        if arguments and arguments[0] == "inspect" and self.inspect_output:
            return completed(arguments, stdout=self.inspect_output)
        return completed(arguments)


class DockerSandboxLifecycleTests(unittest.TestCase):
    def test_source_import_repairs_volume_ownership_without_elevating_worker(self):
        cli = FakeDockerCLI()
        manager = CodeSandboxManager(CodeSandboxPolicy(auto_build=False), cli=cli)
        pair = manager.create_pair("source-import")
        cli.commands.clear()
        with TemporaryDirectory() as directory:
            manager.copy_source(pair, Path(directory))
        self.assertTrue(any(c[0] == "cp" for c in cli.commands))
        self.assertFalse(any(c[0] == "exec" and "chown" in c for c in cli.commands))
        helper = next(c for c in cli.commands if c[0] == "run")
        self.assertIn("--rm", helper)
        self.assertIn("none", helper)
        self.assertIn(f"type=volume,source={pair.candidate_volume},target=/workspace", helper)

    @staticmethod
    def recovery_pair() -> CodeSandboxPair:
        return CodeSandboxPair(
            pair_id="pair-recovery",
            workspace_id="workspace-recovery",
            candidate_volume="candidate-volume",
            review_volume="review-volume",
            worker_container="worker-container",
            reviewer_container="reviewer-container",
            image="recovery-image",
            active_role="REVIEWER",
        )

    def test_policy_is_created_from_application_settings(self) -> None:
        settings = SimpleNamespace(
            image="custom-code-image:v2",
            wsl_distribution="Ubuntu-24.04",
            auto_build=False,
            memory_mb=2048,
            cpu_count=3,
            pids_limit=96,
            execute_timeout_seconds=180,
        )

        policy = CodeSandboxPolicy.from_settings(settings)

        self.assertEqual(policy.image, "custom-code-image:v2")
        self.assertEqual(policy.wsl_distribution, "Ubuntu-24.04")
        self.assertFalse(policy.auto_build)
        self.assertEqual(policy.memory_mb, 2048)
        self.assertEqual(policy.cpu_count, 3)
        self.assertEqual(policy.pids_limit, 96)
        self.assertEqual(policy.execute_timeout_seconds, 180)

    def test_pair_uses_named_volumes_and_only_worker_starts(self) -> None:
        cli = FakeDockerCLI()
        manager = CodeSandboxManager(
            CodeSandboxPolicy(auto_build=False),
            cli=cli,
        )

        pair = manager.create_pair("workspace:group-1:attempt:1")

        self.assertEqual(cli.daemon_checks, 1)
        self.assertEqual(pair.active_role, "WORKER")
        creates = [command for command in cli.commands if command[0] == "create"]
        self.assertEqual(len(creates), 2)
        worker_create, reviewer_create = creates
        self.assertIn("none", worker_create)
        self.assertIn("no-new-privileges", worker_create)
        self.assertIn("ALL", worker_create)
        self.assertIn("--read-only", worker_create)
        self.assertTrue(
            any(
                "target=/workspace,readonly" in part
                for part in reviewer_create
            )
        )
        self.assertTrue(
            any("target=/review" in part for part in reviewer_create)
        )
        starts = [command for command in cli.commands if command[0] == "start"]
        self.assertEqual(starts, [["start", pair.worker_container]])

    def test_pair_mounts_run_handoff_read_only_for_both_roles(self) -> None:
        cli = FakeDockerCLI()
        manager = CodeSandboxManager(
            CodeSandboxPolicy(auto_build=False),
            cli=cli,
        )
        with TemporaryDirectory() as directory:
            handoff = Path(directory) / "handoff"
            handoff.mkdir()
            pair = manager.create_pair(
                "workspace-shared-handoff",
                handoff_root=handoff,
            )

        creates = [command for command in cli.commands if command[0] == "create"]
        self.assertEqual(len(creates), 2)
        for command in creates:
            self.assertTrue(
                any(
                    "source=/host/handoff,target=/handoff,readonly" in part
                    for part in command
                )
            )
        self.assertEqual(pair.handoff_root, str(handoff.resolve()))

    def test_handoff_serializes_worker_and_reviewer_execution(self) -> None:
        cli = FakeDockerCLI()
        manager = CodeSandboxManager(
            CodeSandboxPolicy(auto_build=False),
            cli=cli,
        )
        pair = manager.create_pair("workspace-2")
        before = len(cli.commands)

        pair = manager.handoff(pair, "REVIEWER")
        handoff_commands = cli.commands[before:]

        self.assertEqual(pair.active_role, "REVIEWER")
        self.assertEqual(
            handoff_commands,
            [
                ["stop", "--time", "5", pair.worker_container],
                ["start", pair.reviewer_container],
            ],
        )
        before = len(cli.commands)
        same = manager.handoff(pair, "REVIEWER")
        self.assertEqual(same, pair)
        self.assertEqual(len(cli.commands), before)

    def test_cleanup_targets_only_owned_pair_resources(self) -> None:
        cli = FakeDockerCLI()
        manager = CodeSandboxManager(
            CodeSandboxPolicy(auto_build=False),
            cli=cli,
        )
        pair = manager.create_pair("workspace-3")
        before = len(cli.commands)

        manager.cleanup(pair)

        self.assertEqual(
            cli.commands[before:],
            [
                ["rm", "-f", pair.worker_container],
                ["rm", "-f", pair.reviewer_container],
                ["volume", "rm", "-f", pair.candidate_volume],
                ["volume", "rm", "-f", pair.review_volume],
            ],
        )

    def test_candidate_and_review_exports_use_their_owned_containers(
        self,
    ) -> None:
        cli = FakeDockerCLI()
        manager = CodeSandboxManager(
            CodeSandboxPolicy(auto_build=False),
            cli=cli,
        )
        pair = manager.create_pair("workspace-export")
        before = len(cli.commands)

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.subTest("candidate"):
                manager.export_candidate(pair, root / "candidate-export")
            with self.subTest("review"):
                manager.export_review(pair, root / "review-export")

            self.assertEqual(
                cli.commands[before:],
                [
                    [
                        "cp",
                        f"{pair.worker_container}:/workspace/.",
                        "/host/candidate-export",
                    ],
                    [
                        "cp",
                        f"{pair.reviewer_container}:/review/.",
                        "/host/review-export",
                    ],
                ],
            )

    def test_isolation_report_exposes_reviewer_read_only_candidate(self) -> None:
        cli = FakeDockerCLI()
        manager = CodeSandboxManager(
            CodeSandboxPolicy(auto_build=False),
            cli=cli,
        )
        pair = manager.create_pair("workspace-4")
        cli.inspect_output = json.dumps(
            [
                {
                    "Name": "/" + pair.worker_container,
                    "HostConfig": {
                        "NetworkMode": "none",
                        "Privileged": False,
                        "ReadonlyRootfs": True,
                        "CapDrop": ["ALL"],
                        "Memory": 1610612736,
                        "PidsLimit": 128,
                    },
                    "Config": {"User": "10001:10001"},
                    "Mounts": [{"Destination": "/workspace", "RW": True}],
                },
                {
                    "Name": "/" + pair.reviewer_container,
                    "HostConfig": {
                        "NetworkMode": "none",
                        "Privileged": False,
                        "ReadonlyRootfs": True,
                        "CapDrop": ["ALL"],
                        "Memory": 1610612736,
                        "PidsLimit": 128,
                    },
                    "Config": {"User": "10001:10001"},
                    "Mounts": [
                        {"Destination": "/workspace", "RW": False},
                        {"Destination": "/review", "RW": True},
                    ],
                },
            ]
        ).encode("utf-8")

        report = manager.isolation_report(pair)

        self.assertTrue(
            report[pair.worker_container]["mounts"]["/workspace"]
        )
        self.assertFalse(
            report[pair.reviewer_container]["mounts"]["/workspace"]
        )
        self.assertTrue(
            report[pair.reviewer_container]["mounts"]["/review"]
        )

    def test_recovery_reuses_exact_owned_pair_and_freezes_both_roles(self) -> None:
        cli = FakeDockerCLI()
        manager = CodeSandboxManager(
            CodeSandboxPolicy(image="recovery-image", auto_build=False),
            cli=cli,
        )
        pair = self.recovery_pair()
        records = {
            ("container", pair.worker_container): {
                "Config": {"Image": pair.image},
                "State": {"Running": True},
                "Mounts": [
                    {"Destination": "/workspace", "Name": pair.candidate_volume}
                ],
            },
            ("container", pair.reviewer_container): {
                "Config": {"Image": pair.image},
                "State": {"Running": True},
                "Mounts": [
                    {"Destination": "/workspace", "Name": pair.candidate_volume},
                    {"Destination": "/review", "Name": pair.review_volume},
                ],
            },
            ("volume", pair.candidate_volume): {},
            ("volume", pair.review_volume): {},
        }
        manager._inspect_owned_resource = lambda kind, name: records[(kind, name)]

        recovered, rebuilt = manager.recover_pair(
            pair,
            candidate_snapshot=None,
            reviewer_snapshot=None,
        )

        self.assertFalse(rebuilt)
        self.assertIsNone(recovered.active_role)
        stops = [command for command in cli.commands if command[0] == "stop"]
        self.assertEqual(
            stops,
            [
                ["stop", "--time", "5", pair.worker_container],
                ["stop", "--time", "5", pair.reviewer_container],
            ],
        )

    def test_recovery_refuses_partially_present_pair(self) -> None:
        cli = FakeDockerCLI()
        manager = CodeSandboxManager(
            CodeSandboxPolicy(image="recovery-image", auto_build=False),
            cli=cli,
        )
        pair = self.recovery_pair()
        manager._inspect_owned_resource = lambda kind, name: (
            {} if name == pair.candidate_volume else None
        )

        with self.assertRaisesRegex(CodeSandboxError, "partially present"):
            manager.recover_pair(
                pair,
                candidate_snapshot=None,
                reviewer_snapshot=None,
            )


class DockerSandboxBackendTests(unittest.TestCase):
    def test_wsl_utf16_launcher_warning_is_removed_from_stderr(self) -> None:
        warning = "wsl: localhost proxy warning\r\n".encode("utf-16-le")
        payload = warning + b"real container error\n"

        decoded = _decode_docker_stream(
            payload,
            strip_wsl_launcher_warning=True,
        )

        self.assertEqual(decoded, "real container error\n")

    def test_execute_runs_inside_owned_container_with_timeout(self) -> None:
        cli = FakeDockerCLI()
        backend = DockerSandboxBackend(
            cli,
            container_name="owned-worker",
            backend_id="pair:worker",
            default_timeout_seconds=20,
        )

        result = backend.execute("python -V", timeout=12)

        self.assertEqual(result.exit_code, 0)
        self.assertEqual(backend.id, "pair:worker")
        self.assertEqual(
            cli.commands[-1],
            [
                "exec",
                "owned-worker",
                "timeout",
                "--signal=KILL",
                "12s",
                "bash",
                "--noprofile",
                "--norc",
                "-lc",
                "python -V",
            ],
        )


class CodeImageBuildTests(unittest.TestCase):
    def test_daemon_error_is_not_a_missing_resource(self):
        cli = FakeDockerCLI()
        cli.run = lambda args, **kwargs: completed(args, stderr=b"failed to connect to the docker API", returncode=1)
        manager = CodeSandboxManager(cli=cli)
        with self.assertRaisesRegex(CodeSandboxError, "Cannot inspect"):
            manager._inspect_owned_resource("container", "owned-probe")
        with self.assertRaisesRegex(CodeSandboxError, "cleanup not confirmed"):
            manager.cleanup(DockerSandboxLifecycleTests.recovery_pair())

    def test_domestic_sources_are_build_arguments_only(self):
        cli = FakeDockerCLI()
        manager = CodeSandboxManager(cli=cli)
        manager.build_image(domestic_mirrors=True)
        args = next(command for command in cli.commands if command[0] == "build")
        self.assertEqual(args.count("--build-arg"), 4)
        self.assertIn("PLAYWRIGHT_DOWNLOAD_HOST=https://cdn.npmmirror.com/binaries/playwright", args)
        self.assertEqual(args[-1], "/host/code-agent")
        runtime_args = manager._common_container_arguments("test-owned")
        self.assertEqual(runtime_args[runtime_args.index("--network") + 1], "none")


if __name__ == "__main__":
    unittest.main()
