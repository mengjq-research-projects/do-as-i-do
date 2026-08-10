# 环境说明

这个仓库采用三种不同的环境策略，因为每个阶段的运行时约束都不一样。简要结论如下：

- `reconstruction/` 继续使用 Conda，因为它需要在四个对 CUDA 较敏感的环境之间切换，这些环境来自仓库内 vendored 的研究代码分支。
- `retargeting/` 适合用 `uv` 管理，因为它本质上是一个单独的 Python 3.12 项目。
- `deployment/` 继续使用 Conda，因为它同时依赖仿真包、仅硬件侧使用的包，以及专有 SDK。

## 环境矩阵

| 区域 | 管理方式 | Python | 默认环境名 | 事实来源 |
|---|---|---|---|---|
| `reconstruction/` | Conda | 3.10 和 3.12 | `sam3`, `sam3d`, `hawor`, `tapnet` | [`reconstruction/env/README.md`](reconstruction/env/README.md) 与 [`reconstruction/setup/01_create_envs.sh`](reconstruction/setup/01_create_envs.sh) |
| `retargeting/` | `uv` 或 Conda | 3.12 | `uv` 下为 `.venv`，Conda 下为 `retargeting` | [`retargeting/pyproject.toml`](retargeting/pyproject.toml) 与 [`retargeting/env/README.md`](retargeting/env/README.md) |
| `deployment/` | Conda | 3.12 | `deployment` | [`deployment/env/README.md`](deployment/env/README.md) |

## 共享系统前置条件

- Linux 或 WSL，并且 `reconstruction/` 和 `retargeting/` 所在机器需要有可用的 NVIDIA 驱动。
- `git` 和 `git submodule` 支持。
- 如果你不想用 `GIT_LFS_SKIP_SMUDGE=1` 跳过大文件拉取，则需要 `git-lfs`。
- 用于 viser 可视化界面的浏览器。
- 用于访问 `facebook/sam3` 和 `facebook/sam-3d-objects` 的 Hugging Face 认证。
- HaWoR 需要的 MANO 资源。
- 只有在使用 `deployment/robot_replay/` 时才需要专有的 Sharpa Wave SDK。

## 根目录 `uv` 工作流

根目录的 [`pyproject.toml`](pyproject.toml) 提供了一个工作区级别的 `uv` manifest，当前用于管理：

- 共享开发工具，例如 `ruff`

`retargeting` 真正的运行时环境定义在 [`retargeting/pyproject.toml`](retargeting/pyproject.toml) 中，而不是根目录 manifest。

如果系统里还没有 `uv`，可以先安装：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# or: pipx install uv
```

在仓库根目录执行：

```bash
uv sync
uv sync --group dev
```

这里 `uv` **不会** 替代以下内容：

- [`reconstruction/run_pipeline.sh`](reconstruction/run_pipeline.sh) 会显式切换 [`reconstruction/config/paths.sh`](reconstruction/config/paths.sh) 里定义的四个 Conda 环境。
- `deployment/` 依赖 [`deployment/env/deployment.yml`](deployment/env/deployment.yml) 导出的 Conda 环境。

如果要用 `uv` 创建实际可运行的 retargeting 环境：

```bash
cd retargeting
uv sync
uv run python launch.py --task whisking --raw-dir ../reconstruction/whisking
```

把三个阶段强行合并成一个 `uv` 环境并不会提升可复现性，反而会降低可维护性：`reconstruction` 有意保留了上游 fork 各自的安装流程，`deployment` 又依赖无法通过 PyPI 分发的硬件侧组件。

## 重建阶段环境

`reconstruction/run_pipeline.sh` 会在以下四个环境之间切换：

- `sam3`：抽帧与 SAM3 分割
- `sam3d`：网格生成、pointmap、重力估计和可选 3D 可视化
- `hawor`：手部重建
- `tapnet`：速度跟踪

这些环境名定义在 [`reconstruction/config/paths.sh`](reconstruction/config/paths.sh) 中。备用创建脚本位于 [`reconstruction/setup/01_create_envs.sh`](reconstruction/setup/01_create_envs.sh)：

```bash
cd reconstruction
./setup/01_create_envs.sh
```

注意事项：

- 推荐优先遵循 vendored fork 自身的安装说明。
- `SAM3_DISPLAY=:1` 用于基于点击交互的 SAM3 界面；如果是无头机器，需要自行准备可用的 X display 方案。
- `reconstruction/env/*.yml` 主要是精确版本参考，并不是唯一受支持的安装方式。

完整说明和已知限制请参考 [`reconstruction/env/README.md`](reconstruction/env/README.md)。

## 重定向阶段环境

`retargeting/` 是整个仓库里最适合用 `uv` 管理的部分，因为它已经是一个独立的 Python 项目，并且自带 [`pyproject.toml`](retargeting/pyproject.toml)。

它的 `uv` 配置还顺手规避了一个来自 `mujoco-warp` 的传递依赖 source 问题：当前 pin 住的上游 commit 会声明一个专门的 MuJoCo index，但该地址在这里返回 `404`。本地的 [`retargeting/pyproject.toml`](retargeting/pyproject.toml) 通过静态元数据解析这个 git 依赖，在锁定时避免继承那个坏掉的 MuJoCo source，同时把 `warp-lang` 固定到 NVIDIA 的 index，并将运行时 Python 版本约束在 3.12。

这个工作流已经用 `retargeting/` 目录下的 `uv lock` 和 `uv sync --locked --dry-run` 验证通过。

当前支持两种方式：

### 方案 A：使用 `uv`

```bash
cd retargeting
uv sync
uv run python launch.py --task whisking --raw-dir ../reconstruction/whisking
```

### 方案 B：在 `retargeting/` 内使用 Conda

```bash
cd retargeting
conda create -y -n retargeting python=3.12
conda activate retargeting
pip install -e .
```

精确依赖版本仍然记录在 [`retargeting/env/README.md`](retargeting/env/README.md) 中。

## 部署阶段环境

部署阶段仍然使用一个单独导出的 Conda 环境：

```bash
cd deployment
conda env create -f env/deployment.yml
conda activate deployment
```

额外要求：

- 将 [`deployment/robot_replay/config.example.yaml`](deployment/robot_replay/config.example.yaml) 复制为 `deployment/robot_replay/config.yaml`
- 填入机器人 IP 和手部序列号
- 将专有 Sharpa Wave SDK 安装到 `deployment/robot_replay/Sharpa/`

由于这个阶段可能直接控制真实硬件，建议先用 MuJoCo 回放和 IK 检查轨迹，确认结果正确后再进入 `robot_replay/`。