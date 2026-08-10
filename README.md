<img width="2563" height="742" alt="Frame 261" src="https://github.com/user-attachments/assets/fc36b68d-80af-4c8d-ab09-1e0e44a2193e" />

# Do as I Do

[**Project Page**](https://do-as-i-do.com/) | [**arXiv**](https://arxiv.org/abs/2606.19333) 

这是 Do as I Do 的代码发布版本：整个流程分为三个阶段，从单段手物交互演示视频出发，先重建物体与手部运动，再将该运动重定向到机器人手上，最后可选地在双 UR3e 机械臂和 Sharpa Wave 灵巧手上回放结果。

## 流程概览

| 阶段 | 主要入口 | 输入 | 输出 |
|---|---|---|---|
| `reconstruction/` | `reconstruction/run_pipeline.sh` | 演示视频 + 参考帧 / 目标物体 / 锚定手 | mask、物体网格、pointmap、手部网格、物体 6-DoF 轨迹 |
| `retargeting/` | `retargeting/launch.py` | reconstruction 输出目录 | MuJoCo 场景、IK 轨迹、物理优化后的 `trajectory_mjwp.npz` |
| `deployment/` | `deployment/mujoco_replay/replay_retarget.py`, `deployment/robot_replay/run_npz.py` | retargeting 生成的 `trajectory_mjwp.npz` | 双 UR3e 关节轨迹，以及可选的真实硬件回放 |

仓库尽量保持三个阶段彼此独立。外部依赖代码以 git submodule 的形式随仓库一并提供，并已经带有项目所需的定制修改。

## 仓库结构

- **`reconstruction/`** — 从手物交互演示视频中完成物体与手部重建，以及 6-DoF 位姿跟踪（SAM3 -> SAM3D mesh -> MoGe pointmaps -> HaWoR -> TAPIR -> guided diffusion tracking -> 可选投影）。详见 [`reconstruction/README.md`](reconstruction/README.md)。
- **`retargeting/`** — 将重建得到的手物演示转换为机器人手轨迹（数据处理 -> 凸分解 -> MJCF 场景生成 -> IK -> MuJoCo Warp 中的采样式 MPC）。详见 [`retargeting/README.md`](retargeting/README.md)。
- **`deployment/`** — 将 retargeting 结果适配到双 UR3e 场景，并可进一步发送到真实机器人系统。详见 [`deployment/README.md`](deployment/README.md)。
- **`reconstruction/whisking/`** — 仓库内附带的 whisk 示例处理结果。
- **`retargeting/outputs/sharpa/right/whisking/0/`** — 仓库内附带的 whisk 示例 retargeting 输出，其中包括可直接查看的 `trajectory_mjwp.npz`。

## 快速开始

使用 submodule 一起克隆：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules https://github.com/malik-group/do-as-i-do.git
cd do-as-i-do
```

然后按你的需求选择阶段：

1. 使用 [`reconstruction/run_pipeline.sh`](reconstruction/run_pipeline.sh) 从视频执行重建。
2. 使用 [`retargeting/launch.py`](retargeting/launch.py) 将结果重定向到机器人手。
3. 使用 [`deployment/`](deployment/README.md) 下的工具进行预览或部署。

## 环境概览

仓库已经附带了分阶段的环境定义，只是此前这些信息分散在各个子目录里。现在统一的根目录环境说明位于 [`ENVIRONMENT.md`](ENVIRONMENT.md)。

| 区域 | 管理方式 | 默认环境名 | 说明 |
|---|---|---|---|
| `reconstruction/` | Conda | `sam3`, `sam3d`, `hawor`, `tapnet` | 四个阶段横跨不同的 Python/CUDA 依赖，并依赖仓库内 vendored fork |
| `retargeting/` | `uv` 或 Conda | `uv` 下为 `.venv`，Conda 下为 `retargeting` | 单个 Python 3.12 项目，自带独立 `pyproject.toml` |
| `deployment/` | Conda | `deployment` | 还需要专有 Sharpa Wave SDK 和机器人配置 |

### 根目录的 `uv` 工作流

根目录的 [`pyproject.toml`](pyproject.toml) 用来给 `uv` 管理工作区级开发工具。它只在仓库根目录维护一个小型共享开发环境，而 `retargeting` 真正的运行时环境仍由 [`retargeting/pyproject.toml`](retargeting/pyproject.toml) 管理。

```bash
# 如果还没安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 可选：在仓库根目录安装共享开发工具
uv sync
uv sync --group dev

# retargeting 运行时环境
cd retargeting
uv sync
uv run python launch.py --task whisking --raw-dir ../reconstruction/whisking
```

`retargeting/` 的 `uv` 项目基于 Python 3.12 解析依赖。如果你需要与导出的 Conda 环境严格对齐的参考包版本，请以 [`retargeting/env/retargeting.yml`](retargeting/env/retargeting.yml) 作为版本参考。

这里让 `uv` 覆盖的范围仅限于共享开发工具和 `retargeting/` 这个 Python 项目。`reconstruction/` 仍然依赖四个对 CUDA 比较敏感的 Conda 环境，`deployment/` 也仍然依赖它自己的导出 Conda 环境以及专有硬件 SDK。

## 分阶段入口

### 重建

主入口：

```bash
cd reconstruction
./run_pipeline.sh whisking/whisking.mp4 125 whisk right
```

该流程会在四个专用 Conda 环境之间切换：`sam3`、`sam3d`、`hawor`、`tapnet`。输出的物体轨迹和手部重建结果会写回到视频对应目录下。完整安装说明见 [`reconstruction/README.md`](reconstruction/README.md)。

### 重定向

主入口：

```bash
cd retargeting
python launch.py --task whisking --raw-dir ../reconstruction/whisking
```

该阶段读取 reconstruction 的输出，构建 MuJoCo 场景、求解 IK，并执行 MuJoCo Warp 物理优化。完整安装说明见 [`retargeting/README.md`](retargeting/README.md)。

### 部署

在 MuJoCo / viser 中预览仓库内附带的 retargeting 结果：

```bash
cd deployment/mujoco_replay
python replay_retarget.py \
    --side right \
    --traj ../../retargeting/outputs/sharpa/right/whisking/0/trajectory_mjwp.npz \
    --speed 0.25
```

真实硬件回放位于 `deployment/robot_replay/`，需要 UR3e 场地配置和 Sharpa Wave SDK。完整说明见 [`deployment/README.md`](deployment/README.md)。

## 外部资源与权限

- **子模块**：`reconstruction/modules/` 是 reconstruction 流程所必需的。
- **Hugging Face 权限**：访问 `facebook/sam3` 和 `facebook/sam-3d-objects` 时需要。
- **MANO 资源**：HaWoR 依赖该资源。
- **Sharpa Wave SDK**：仅 `deployment/robot_replay/` 需要，仓库中不附带该 SDK。

关于各环境对应的命令、精确依赖版本和安装取舍，请结合 [`ENVIRONMENT.md`](ENVIRONMENT.md) 以及上面链接到的各阶段文档一起查看。

