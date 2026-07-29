# SLURM AutoSync

面向任意 SLURM GPU 集群的训练工作流 Skill。它把本地项目、登录节点和计算节点的职责明确分开，提供配置校验、脚本渲染、代码同步、作业提交、监控、结果回收和 CSV 指标分析。

该项目不绑定特定学校、网络、GPU、分区、训练框架或模型。YOLO 配置仅作为完整示例；任意框架都可通过命令参数数组接入。

## 前提

- Python 3.11 或更高版本。
- 本机已安装并能调用 OpenSSH `ssh` 与 `scp`。
- 集群登录节点提供 `sbatch`、`squeue`、`scontrol`、`sacct` 和 `scancel`。
- 已从管理员获得 SSH 主机或别名、工作目录、分区、账户/QoS、模块和 Conda 激活方式。
- 若集群计算节点无外网，必须在登录节点完成依赖、数据和模型权重准备。

## 目录

```text
SKILL.md                         Claude Code Skill 入口
config/                          可复制 JSON 配置样例
scripts/slurm_autosync.py        本地编排 CLI，仅使用 Python 标准库
templates/                       通用 SLURM 模板
tests/                           离线 unittest 测试
```

## 快速开始

1. 建立私有集群配置。`config/cluster.json` 已被 `.gitignore` 排除。

```powershell
Copy-Item config/cluster.example.json config/cluster.json
Copy-Item config/experiment.single.example.json config/experiment.local.json
```

2. 编辑 `config/cluster.json`。填写登录节点、用户名、远端工作目录、分区和 Conda 设置。主机可写 SSH Config alias；不要填密码或私钥内容。

3. 编辑实验配置。`inputs.data`、`inputs.model`、`output_dir` 都相对于 `remote.workspace`，`train_command` 必须是字符串数组，不要传入整段 shell。

4. 先打印诊断命令并确认无误，再执行真实连接诊断。

```powershell
python scripts/slurm_autosync.py --cluster config/cluster.json --dry-run doctor
python scripts/slurm_autosync.py --cluster config/cluster.json doctor
```

5. 渲染并审核单卡脚本。

```powershell
python scripts/slurm_autosync.py --cluster config/cluster.json render --experiment config/experiment.local.json --output rendered/baseline.slurm
```

6. 同步源代码，上传并提交已审核的脚本。

```powershell
python scripts/slurm_autosync.py --cluster config/cluster.json sync --source .
python scripts/slurm_autosync.py --cluster config/cluster.json submit --script rendered/baseline.slurm
```

7. 使用返回的 job ID 查看状态或取消作业。

```powershell
python scripts/slurm_autosync.py --cluster config/cluster.json status 12345
python scripts/slurm_autosync.py --cluster config/cluster.json cancel 12345
```

8. 拉取结果并分析 CSV。

```powershell
python scripts/slurm_autosync.py --cluster config/cluster.json fetch --remote-path runs/baseline-single --destination .\runs
python scripts/slurm_autosync.py analyze .\runs\baseline-single\results.csv --output .\runs\baseline-summary.json
```

## 配置

### 集群配置

`config/cluster.example.json` 的字段：

| 字段 | 作用 |
|---|---|
| `connection` | 登录节点、用户、端口和可选私钥路径。认证由 OpenSSH/agent 管理。 |
| `remote` | 远端工作目录及代码、数据、权重、结果、日志和 staging 的相对目录。 |
| `slurm` | 默认分区、账户、QoS、资源、模块和 Conda 设置。 |
| `policy.compute_nodes_have_internet` | 默认 `false`；阻止训练命令包含安装、下载和 clone 行为。 |
| `policy.allow_remote_delete` | 默认 `false`；只有管理员同意后才允许远端删除。 |

所有 `remote` 子路径必须是相对 POSIX 路径，`remote.workspace` 必须是绝对 POSIX 路径。实验名和 trial 名仅允许字母、数字、点、下划线和连字符。

### 单卡实验

`mode: single` 为一个 Slurm 任务配置一个或多个 GPU。对于单进程训练，通常设为一张 GPU。命令参数支持 `{data}`、`{model}`、`{name}`、`{output_dir}`、`{output_parent}` 和 `{best_model}` 占位符。

### 并行实验矩阵

`mode: parallel` 的 `trials` 是独立实验数组。工具从 trial 数量生成 `--ntasks`，并为每个任务分配 `--gpus-per-task` 张 GPU。每个 trial 名必须唯一。模板交给 Slurm 的 `CUDA_VISIBLE_DEVICES` 处理隔离，绝不把 `SLURM_PROCID` 当物理显卡编号。

trial 参数可在命令中以 `{batch}`、`{epochs}` 等占位符使用。

## 数据集检查

YOLO 数据集可使用示例 profile 离线检查目录和标签配对：

```powershell
python scripts/slurm_autosync.py check-dataset --dataset . --profile config/dataset.yolo.example.json
```

它检查 train/val 的图片与标签目录，以及每张图片是否存在同名 `.txt` 标签。远端数据集应在登录节点以相同规则检查，并确保数据配置引用集群上的绝对路径。

## 作业脚本约束

渲染出的脚本会：

- 使用 `set -euo pipefail`。
- 创建日志目录、加载模块、初始化/激活 Conda、切换工作目录。
- 在分配到计算节点后检查数据和权重文件是否存在。
- 不在计算节点执行 `pip`、`conda install`、`wget`、`curl` 或 `git clone`。
- 使用 `%x-%j` 或 `%x-%j-%t` 命名日志。
- 并行任务使用 `srun --exclusive --gpus-per-task`。

依赖安装、模型下载和大规模数据准备应在登录节点完成，且最好由项目自己的环境管理脚本负责。

## 结果与分析

`fetch` 可以拉取任何工作区相对路径。它不默认删除远端结果。只有同时满足以下条件才会删除：

1. `cluster.json` 中 `allow_remote_delete` 为 `true`。
2. 显式提供 `--delete-remote`。
3. `--confirm` 与将删除的绝对远端路径完全一致。

`analyze` 读取一个或多个 CSV 并输出 JSON。对 YOLO/Ultralytics 常见列名自动识别 `mAP50`、`mAP50-95`、Precision 和 Recall，给出最佳值、对应 epoch 和最终值。

## 命令参考

```text
python scripts/slurm_autosync.py --help
python scripts/slurm_autosync.py --cluster config/cluster.json doctor
python scripts/slurm_autosync.py --cluster config/cluster.json sync --source PATH
python scripts/slurm_autosync.py --cluster config/cluster.json render --experiment PATH [--output PATH]
python scripts/slurm_autosync.py --cluster config/cluster.json submit --script PATH
python scripts/slurm_autosync.py --cluster config/cluster.json status JOB_ID
python scripts/slurm_autosync.py --cluster config/cluster.json cancel JOB_ID
python scripts/slurm_autosync.py --cluster config/cluster.json fetch --remote-path PATH --destination PATH
python scripts/slurm_autosync.py check-dataset --dataset PATH --profile PATH
python scripts/slurm_autosync.py analyze RESULT.csv [MORE.csv ...] [--output REPORT.json]
```

## 安全

- 使用 SSH key、agent 或企业 SSO，不把密码放入配置或命令行。
- 保持 `config/cluster.json`、私钥、实验产物和日志不入库。
- 使用 `--dry-run` 审阅所有外部命令。
- 先 `render` 后 `submit`，避免把未经审核的命令送入队列。
- 没有集群管理员批准时，不猜测分区、账户、QoS、GPU 型号、模块名或网络访问路径。

## 验证

```powershell
python -m unittest discover -s tests -v
```

## 诊断顺序

发生问题时按以下顺序检查：

1. 本地 SSH 连接、主机别名、端口和认证。
2. 登录节点上的工作目录、模块、Conda 环境、数据和模型权重。
3. `status JOB_ID` 输出中的 Slurm Pending 原因、状态与退出码。
4. 作业 stdout/stderr 和框架自己的结果文件。
5. CUDA 兼容性、显存、数据加载吞吐和恢复 checkpoint。

不要通过直接 SSH 到假定的 GPU 节点或手动指定物理 GPU 来绕过 Slurm。

