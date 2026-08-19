<img width="2563" height="742" alt="Frame 261" src="https://github.com/user-attachments/assets/fc36b68d-80af-4c8d-ab09-1e0e44a2193e" />

# Do as I Do

[**Project Page**](https://do-as-i-do.com/) | [**arXiv**](https://arxiv.org/abs/2606.19333) 

这是 Do as I Do 的代码发布版本：从单段手物交互演示视频出发，先重建物体与手部运动，再将该运动重定向到机器人手，并生成可由 MuJoCo 与 Isaac Sim 共同消费的 UR3e＋Sharpa＋物体运动学场景。完整目录关系、数据协议和端到端命令见 [`PROJECT_STRUCTURE_AND_WORKFLOW.md`](PROJECT_STRUCTURE_AND_WORKFLOW.md)。

## 流程概览

| 阶段 | 主要入口 | 输入 | 输出 |
|---|---|---|---|
| `reconstruction/` | `reconstruction/run_pipeline.sh` | 演示视频 + 参考帧 / 目标物体 / 锚定手 | mask、物体网格、pointmap、手部网格、物体 6-DoF 轨迹 |
| `retargeting/` | `retargeting/run_pipeline.sh` | reconstruction 输出目录 | MuJoCo 场景、IK 轨迹、物理优化后的 `trajectory_mjwp.npz` |
| `isaac_export/` | `isaac_export/run_pipeline.sh` | Level A 或最终机器人包 | 标准轨迹、USD、Isaac 回放和回读报告 |
| `deployment/` | `deployment/run_pipeline.sh` | retargeting 生成的 `trajectory_mjwp.npz` | 双 UR3e＋Sharpa＋物体最终包，以及可选真实硬件回放 |

面向使用者的阶段入口统一为 `run_pipeline.sh`。各目录中的 `.py` 文件是内部实现或测试模块，不需要手动选择 Python 解释器；shell 入口会自动使用对应的托管环境或仓库本地环境。

各模块保持独立环境和清晰的数据边界。`isaac_export/` 可以直接消费 `retargeting/` 的 Level A 数据，也可以消费 `deployment/` 产生的最终机器人场景包。外部依赖代码以 git submodule 的形式随仓库一并提供，并已经带有项目所需的定制修改。

## 仓库结构

- **`reconstruction/`** — 从手物交互演示视频中完成物体与手部重建，以及 6-DoF 位姿跟踪（SAM3 -> SAM3D mesh -> MoGe pointmaps -> HaWoR -> TAPIR -> guided diffusion tracking -> 可选投影）。详见 [`reconstruction/README.md`](reconstruction/README.md)。
- **`retargeting/`** — 将重建得到的手物演示转换为机器人手轨迹（数据处理 -> 凸分解 -> MJCF 场景生成 -> IK -> MuJoCo Warp 中的采样式 MPC）。详见 [`retargeting/README.md`](retargeting/README.md)。
- **`isaac_export/`** — 支持自由手根 Level A 与最终 UR3e 场景两种包；负责 USD 转换、运动学回放和状态回读。详见 [`isaac_export/README.md`](isaac_export/README.md)。
- **`deployment/`** — 将 retargeting 结果适配到双 UR3e 场景，统一变换并显示任务物体，保存最终场景包；也可进一步发送到真实机器人系统。详见 [`deployment/README.md`](deployment/README.md)。
- **`reconstruction/whisking/`** — 仓库内附带的 whisk 示例处理结果。
- **`retargeting/outputs/sharpa/right/whisking/0/`** — 仓库内附带的 whisk 示例 retargeting 输出，其中包括可直接查看的 `trajectory_mjwp.npz`。
- **`isaac_export/outputs/whisking/`** — 运行导出器后生成的 whisking 标准轨迹和报告；该目录是本地生成物，不进入版本控制。

## 快速开始

使用 submodule 一起克隆：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules \
  https://github.com/mengjq-research-projects/do-as-i-do.git
cd do-as-i-do

# 从共享依赖包离线恢复并校验全部运行环境
./setup_all.sh --managed-offline

# 为当前终端加载托管环境、模型和解释器路径
source dependency_management/activate.sh
```

这一条命令已经包含资源校验、运行时链接生成、四个 Reconstruction Conda 环境恢复、Retargeting venv、完整 Isaac Sim 6.0.1.0 venv，以及 CUDA/A100 烟雾测试。它不会访问 GitHub、PyPI、Hugging Face 或 NVIDIA，也不需要再单独执行 `dependency_management/manage.sh`。命令可以安全重复运行。

`activate.sh` 不会安装或修改环境，只会为当前终端设置 `$ENV_SAM3`、`$ENV_SAM3D`、`$ENV_HAWOR`、`$ENV_TAPNET`、`$RETARGETING_PYTHON`、`$ISAAC_PYTHON` 以及模型路径。每次打开新终端后，如需直接使用这些变量，应重新执行该 `source` 命令；项目提供的 Shell 流水线入口会自行加载路径配置。

共享依赖包位于 `/data/jiaqimeng/retargeting_dev`，模型、环境和缓存均与 Git 仓库分离。MANO 受许可证限制，仍需由获授权的使用者手动提供。环境结构和维护说明见 [`ENVIRONMENT.md`](ENVIRONMENT.md) 与 [`dependency_management/README.md`](dependency_management/README.md)。

然后按你的需求选择阶段：

1. 使用 [`reconstruction/run_pipeline.sh`](reconstruction/run_pipeline.sh) 从视频执行重建。
2. 使用 [`retargeting/run_pipeline.sh`](retargeting/run_pipeline.sh) 将结果重定向到机器人手。
3. 使用 [`isaac_export/run_pipeline.sh`](isaac_export/run_pipeline.sh) 生成 Isaac Level A 标准轨迹和自动检查报告。

MuJoCo 最终场景预览会复用 Retargeting 环境，不需要额外安装。只有真实硬件回放才需要 Deployment Conda 环境、Sharpa Wave SDK、UR3e 配置和机器人网络。

## 环境概览

仓库已经附带了分阶段的环境定义，只是此前这些信息分散在各个子目录里。现在统一的根目录环境说明位于 [`ENVIRONMENT.md`](ENVIRONMENT.md)。

| 区域 | 管理方式 | 默认环境名 | 说明 |
|---|---|---|---|
| `reconstruction/` | Conda | `sam3`, `sam3d`, `hawor`, `tapnet` | 四个阶段横跨不同的 Python/CUDA 依赖，并依赖仓库内 vendored fork |
| `retargeting/` | 托管 venv | `current/installed-envs/venv/retargeting` | 由离线 uv 缓存按锁文件重建 |
| `isaac_export/` | 托管 venv | `current/installed-envs/venv/isaac` | 包含固定的 Isaac Sim 6.0.1.0 |
| `deployment/` | Conda | `deployment` | 还需要专有 Sharpa Wave SDK 和机器人配置 |

### 新人一键环境准备

推荐从仓库根目录执行：

```bash
./setup_all.sh --managed-offline
source dependency_management/activate.sh
```

如需先预览脚本将执行的操作：

```bash
./setup_all.sh --managed-offline --dry-run
```

## 分阶段入口

### 重建

主入口：

```bash
cd reconstruction
./run_pipeline.sh whisking/whisking.mp4 125 whisk right "584,529" "1"
```

其中 `584,529` 是参考帧 125 中 whisk 内部的正样本像素坐标，`1` 表示正样本。该命令不需要 GUI 或 X Server，适合远端无显示器服务器。流程会在四个专用 Conda 环境之间切换：`sam3`、`sam3d`、`hawor`、`tapnet`。输出的物体轨迹和手部重建结果会写回到视频对应目录下。完整安装说明见 [`reconstruction/README.md`](reconstruction/README.md)。

#### 处理全新视频

运行数据放在 Git 仓库之外；下面以杯子任务为例：

```bash
cd ~/projects/do-as-i-do
source dependency_management/activate.sh

mkdir -p /data/jiaqimeng/do-as-i-do-runs/cup_demo
cp /path/to/my_demo.mp4 /data/jiaqimeng/do-as-i-do-runs/cup_demo/input.mp4

VIDEO=/data/jiaqimeng/do-as-i-do-runs/cup_demo/input.mp4
FRAME_N=100  # 从 0 开始计数，选择手和物体都清楚可见的一帧

# 先导出参考帧，用 VS Code Remote 或复制到本机查看像素坐标。
"$ENV_SAM3/bin/ffmpeg" -y -i "$VIDEO" \
  -vf "select=eq(n\,${FRAME_N})" \
  -frames:v 1 -update 1 \
  /data/jiaqimeng/do-as-i-do-runs/cup_demo/reference.png
```

在 `reference.png` 中选取物体内部一点，例如 `(620,410)`，然后执行：

```bash
cd ~/projects/do-as-i-do/reconstruction
CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 ./run_pipeline.sh \
  /data/jiaqimeng/do-as-i-do-runs/cup_demo/input.mp4 \
  100 cup right "620,410" "1"
```

先用 `nvidia-smi` 选择空闲卡，再通过 `CUDA_VISIBLE_DEVICES=<物理 GPU 编号>` 为本次流水线指定 GPU；上例使用 GPU 3，脚本默认使用 GPU 0。最后两个参数分别是分号分隔的像素坐标和标签；`1` 是物体内部正样本，`0` 是排除区域。例如 `"620,410;80,80" "1;0"`。输出会写到输入视频所在目录。当前高质量 Fast-SAM3D 配置每帧进行 25 个姿态采样和 render-compare；在 A100 上，约 138 帧的视频通常需要 1.5–2 小时完成该阶段，时长随帧数近似线性增加。当前 Stage 3 不支持从中间帧自动续跑，因此运行期间不要中断。

### 重定向

主入口：

```bash
cd retargeting
./run_pipeline.sh --task whisking --raw-dir ../reconstruction/whisking
```

该阶段读取 reconstruction 的输出，构建 MuJoCo 场景、求解 IK，并执行 MuJoCo Warp 物理优化。完整安装说明见 [`retargeting/README.md`](retargeting/README.md)。
看到 `Saved info to .../trajectory_mjwp.npz`、最终 tracking error 和
`Optimization complete` 后，结果已经完整保存；如果 Viser 继续保持服务，
此时可以安全按 `Ctrl+C` 返回终端。

### Isaac Level A 导出

仓库附带的 whisking 结果可以直接转换，不需要重新运行 Reconstruction 或 Retargeting：

```bash
cd "$(git rev-parse --show-toplevel)"
./isaac_export/run_pipeline.sh export
```

该入口使用相对于仓库根目录的默认输入和输出路径，因此不要在
`reconstruction/` 或 `retargeting/` 目录中以 `./isaac_export/...` 调用。

默认读取 `retargeting/outputs/sharpa/right/whisking/0/`，跳过 MJWP warmup，并输出到 `isaac_export/outputs/whisking/`：

- `trajectory.npz`：显式时间戳、手根位姿、22 个手指关节和物体位姿；
- `manifest.json`：关节顺序、坐标、单位、四元数、时间和源文件约定；
- `quality_report.json`：有限值、四元数、跳变、关节限位和时间连续性检查；
- `environment_report.json`：当前机器能否执行 USD 转换、Isaac 回放和无头渲染。

导出其他右手 Sharpa 任务时只需要更换运行目录：

```bash
cd "$(git rev-parse --show-toplevel)"
./isaac_export/run_pipeline.sh export \
    --run-dir retargeting/outputs/sharpa/right/<task>/<id> \
    --output-dir isaac_export/outputs/<task>
```

当前层级 A 已完成与 Isaac 无关的标准数据边界。`environment_report.json`
是在启动 Isaac App 之前生成的；某些 wheel 安装中，`omni.usd` 只有在
`SimulationApp` 启动后才进入模块路径，因此报告可能显示
`Isaac environment: unavailable`，即使 `isaacsim` 和 `pxr` 已安装。
这不影响标准导出，实际 USD 能力以 `build-usd` 命令为准。生成
`scene.usd`、回放视频和关键帧截图仍需要完整 Isaac Sim 环境。详细命令和
当前边界见 [`isaac_export/README.md`](isaac_export/README.md)。

### 部署

在 MuJoCo / viser 中预览仓库内附带的 retargeting 结果：

```bash
cd "$(git rev-parse --show-toplevel)"
./deployment/run_pipeline.sh mujoco-replay \
    --side right \
    --traj retargeting/outputs/sharpa/right/whisking/0/trajectory_mjwp.npz \
    --speed 0.25
```

`mujoco-replay` 会自动复用一键离线安装创建的托管 Retargeting 环境，不需要
额外创建 Deployment 环境。点击 **Save retarget** 后会同时保存机械臂、手指、
物体轨迹、最终场景 XML 和 manifest；校验与 Isaac 完整场景命令见
[`PROJECT_STRUCTURE_AND_WORKFLOW.md`](PROJECT_STRUCTURE_AND_WORKFLOW.md)。

真实硬件回放位于 `deployment/robot_replay/`，需要 UR3e 场地配置和 Sharpa Wave SDK。完整说明见 [`deployment/README.md`](deployment/README.md)。

## 外部资源与权限

- **子模块**：`reconstruction/modules/` 是 reconstruction 流程所必需的。
- **Hugging Face 权限**：只有发布维护者刷新受控权重时需要；新人离线启动不需要。
- **MANO 资源**：HaWoR 处理没有缓存的新视频时依赖 `MANO_LEFT.pkl` 和 `MANO_RIGHT.pkl`；许可证要求获授权用户手动安装到 `current/assets/licensed/mano/`，不得提交到 Git。
- **Isaac Sim**：只在生成 USD、执行 Isaac 运动学回放和渲染时需要；标准轨迹与质量报告导出不依赖 Isaac Sim。
- **Sharpa Wave SDK**：当前项目不做真机部署，因此完全不需要安装；它只与保留的可选 `deployment/robot_replay/` 模块有关。

关于各环境对应的命令、精确依赖版本和安装取舍，请结合 [`ENVIRONMENT.md`](ENVIRONMENT.md) 以及上面链接到的各阶段文档一起查看。

## TODO

- [ ] 为 Isaac 场景添加默认摄像机，并支持在无显示器环境中渲染完整回放 MP4。
