# 本机环境恢复（2026-10-18）

本次空间整理保留了 `D:\PythonProject` 的代码、`.agent` 私有记录、
`.models` 本地模型和 `agent` Conda 环境。Ubuntu 的完整磁盘备份存放在
`D:\C-drive-cleanup-backup-20261018\wsl\Ubuntu.vhdx`，校验记录为同目录的
`manifest.json`。备份不上传 GitHub；GitHub 代码不能替代这份本地环境备份。

## 哪些功能依赖 Ubuntu

- `evals/appworld/protocol.py::docker_prefix` 在 Windows 上通过
  `wsl -d Ubuntu --exec` 调用 Linux Docker。
- `workers/docker_sandbox.py::DockerCLI` 使用同一条 WSL/Docker 路径。
- `config.py` 的 `RUNTIME_CODE_SANDBOX_WSL_DISTRIBUTION` 默认是 `Ubuntu`。

Ubuntu 注销后，AppWorld 和 Docker 代码沙盒需要先恢复环境再使用。
Windows 项目源码和私有任务记录仍在；本次没有恢复旧任务、执行评测或调用付费模型。

## 从完整备份恢复到 D 盘

在 PowerShell 中执行。先确认没有同名 Ubuntu；已有同名环境时不要覆盖：

```powershell
wsl --list --verbose
$backupRoot = 'D:\C-drive-cleanup-backup-20261018\wsl'
$backup = Join-Path $backupRoot 'Ubuntu.vhdx'
$manifest = Get-Content -LiteralPath (Join-Path $backupRoot 'manifest.json') -Raw | ConvertFrom-Json
if ((Get-FileHash -LiteralPath $backup -Algorithm SHA256).Hash -ne $manifest.backup_sha256) {
    throw 'WSL backup SHA256 mismatch'
}
if (Test-Path -LiteralPath 'D:\WSL\Ubuntu') {
    throw 'D:\WSL\Ubuntu already exists; inspect it before importing'
}
New-Item -ItemType Directory -Path 'D:\WSL' -Force | Out-Null
wsl --import Ubuntu D:\WSL\Ubuntu $backup --vhd --version 2
if ($LASTEXITCODE -ne 0) { throw 'WSL import failed' }
wsl --set-default Ubuntu
```

`--import --vhd` 在安装位置创建独立副本，不会把备份直接变成运行磁盘。
因此恢复时 D 盘需要额外容纳约147GiB，备份仍应保留。

导入后默认用户可能是 root。原环境默认 UID 为1000，可以先查询用户名，
再用本机 WSL 支持的命令设置：

```powershell
wsl -d Ubuntu -u root -- sh -lc 'getent passwd 1000 | cut -d: -f1'
# 把下面的 linux_user 替换成上一条实际输出：
wsl --manage Ubuntu --set-default-user linux_user
```

原 `%USERPROFILE%\.wslconfig` 配置过32GB swap，并将 swapFile 放在 C 盘。
恢复前建议将其中 swapFile 改为 `D:\\WSL\\wsl-swap.vhdx`，保留其他参数，
避免以后再次产生 C 盘交换磁盘。本次未擅自修改这份全局配置。

## 恢复后只做基础检查

```powershell
wsl -d Ubuntu -- sh -lc 'command -v docker'
wsl -d Ubuntu -- sh -lc 'docker version'
wsl -d Ubuntu -- sh -lc 'docker image inspect personalops-appworld:0.1.3.post1 --format "{{.Id}}"'
```

若非登录 PATH 找不到 Docker，旧环境使用过 `/snap/bin/docker`，可用该绝对路径检查。
若 Docker daemon 尚未运行，按该备份中的原安装方式启动服务后重试基础检查。
完整 VHDX 备份包含原 Linux 文件和 Docker 存储；是否能正常启动仍以恢复后检查为准。
本次做了导出和备份完整性核对，没有运行一轮恢复启动验收。

需要重建镜像时先阅读 `evals/appworld/README.md` 和
`docs/SESSION_HANDOFF.md`；不要为了验证空间整理而启动付费模型、批量评测或旧任务。

Windows Agent 的原入口继续使用：

```powershell
Set-Location D:\PythonProject
C:\ProgramData\anaconda3\envs\agent\python.exe main.py
```

这条命令是恢复后的人工启动入口，本次空间整理没有执行它。

WSL 命令参考：[Microsoft WSL 命令文档](https://learn.microsoft.com/windows/wsl/basic-commands)。
