# Do as I Do · 重定向

这个模块负责把已经重建好的手物交互演示重定向到 **机器人手** 上。输入是 [`reconstruction/`](../reconstruction/README.md) 阶段的输出目录，包含 MANO 手部轨迹、物体网格和逐帧物体位姿。随后流水线会清理并对齐这些轨迹、构建 MuJoCo 场景、求解逆运动学，并执行基于采样 MPC 的物理优化（MuJoCo Warp），最终产出物理上更一致的机器人手与物体轨迹。

## 目录结构

```
retargeting/
├── launch.py                # 唯一主入口（5 个阶段的流水线）
├── pyproject.toml           # Python 项目与依赖定义
├── config/
│   ├── default.yaml         # 优化器 / 仿真器默认参数
│   └── override/do_as_i_do.yaml  # 数据集相关覆盖配置
├── retargeting/             # Python 包本体
│   ├── config.py            # 运行配置
│   ├── pipeline/            # 按执行顺序组织的 5 个阶段
│   ├── utils/               # 仿真、优化、viewer 与通用辅助函数
│   └── assets/robots/       # 机器人模型（sharpa、mano）
└── env/                     # Conda 环境文档（见 env/README.md）
```

## 运行要求

- 需要一张支持 CUDA 的 NVIDIA GPU（MuJoCo Warp 会在 GPU 上进行物理优化）。
- 需要一个浏览器来打开 viser viewer 的 Web 界面。
- 需要一个 reconstruction 阶段的输出目录作为输入，例如 whisking 示例。

## 环境准备（一次性）

在仓库根目录运行统一的离线安装入口：

```bash
cd ~/projects/do-as-i-do
./setup_all.sh --managed-offline
```

该命令会从托管依赖包中离线恢复并校验 Retargeting 环境。新人不需要再运行
`conda create`、`pip install`、`uv sync` 或 `dependency_management/manage.sh`。
依赖发布和故障排查细节见
[`dependency_management/README.md`](../dependency_management/README.md)。

## 运行

先对一段视频运行 reconstruction 流程，例如 whisking 示例；之后在 `retargeting/` 目录中执行：

```bash
./run_pipeline.sh --task whisking --raw-dir ../reconstruction/whisking
```

其中 `--raw-dir` 指向 reconstruction 的输出目录，也就是视频对应目录；`--task` 是该视频的任务名。常用参数：

- `--robot-type sharpa`：目标机器人手，默认是 `sharpa`
- `--no-show-viewer`：关闭 viser viewer，适用于无头环境
- `--max-sim-steps N`：限制优化步数，`0` 表示完整轨迹
- `--no-wait-on-finish`：结束后直接退出，而不是保持 viewer 常驻

## 流水线阶段

| # | 阶段 | 模块 |
|---|---|---|
| 1 | 数据处理（清理 + 重力对齐 -> 关键点轨迹） | `retargeting/pipeline/process_dataset.py` |
| 2 | 物体凸分解（CoACD） | `retargeting/pipeline/decompose_mesh.py` |
| 3 | MJCF 场景生成（机器人 + 物体 + UR3 手腕圆柱） | `retargeting/pipeline/generate_scene.py` |
| 4 | 逆运动学求解（mink） | `retargeting/pipeline/solve_ik.py` |
| 4.5 | pedestal 解析（`scene_ik.xml` -> `scene.xml`） | `retargeting/pipeline/resolve_pedestal.py` |
| 5 | 物理优化（基于采样的 MPC，MuJoCo Warp） | `retargeting/pipeline/optimize_physics.py` |

## 输出结果

输出写在 `outputs/` 目录下，相对于 `retargeting/`：

- `outputs/mano/{hand}/{task}/0/trajectory_keypoints.npz`：清理后的参考关键点轨迹（阶段 1）
- `outputs/assets/objects/{object}/`：物体网格和凸分解结果（阶段 1-2）
- `outputs/{robot}/{hand}/{task}/0/scene.xml`：生成的 MuJoCo 场景（阶段 3-4.5）
- `outputs/{robot}/{hand}/{task}/0/capture_metadata.json`：从 Reconstruction
  透传的 `ego/exo` 与 `moving/static` 相机元数据，供 Deployment 使用
- `outputs/{robot}/{hand}/{task}/0/trajectory_kinematic.npz`：IK 轨迹（阶段 4）
- `outputs/{robot}/{hand}/{task}/0/trajectory_mjwp.npz` 与 `config.yaml`：优化后的轨迹，以及本次运行解析后的配置（阶段 5）；逐步 tracking error 指标也会存储在 `.npz` 中

当终端显示 `Saved info to .../trajectory_mjwp.npz`、最终 tracking error 和
`Optimization complete` 时，优化结果已经完整保存。如果 Viser 随后继续保持
服务，可以安全按 `Ctrl+C` 结束服务器并返回终端。

## 可视化一个已完成的重定向轨迹

`replay_viser.py` 可以在交互式 [viser](https://github.com/nerfstudio-project/viser) viewer 中回放一个已经完成的结果，**不会重新执行优化**。它直接复用了流水线中的 `retargeting.utils.viser_viewer`，并最多叠加三层已经对齐的可视化内容，每一层都可以在 GUI 中单独开关：

1. **MANO 参考层**（橙色）: 变形后的 MANO 手部网格，以及阶段 1 输出 `outputs/mano/{hand}/{task}/{id}/trajectory_keypoints.npz` 中的物体跟踪位姿
2. **IK 参考层**（半透明蓝色）: 阶段 4 生成的运动学解 `trajectory_kinematic.npz`
3. **重定向结果层**（实体）: 阶段 5 输出的物理优化结果 `trajectory_mjwp.npz`

在 `retargeting` 环境中，从 `retargeting/` 目录执行：

```bash
./run_pipeline.sh replay                     # whisking demo → http://localhost:8081
```

打开终端打印出的 URL，然后使用 **Frame** 滑条和 **Play** 按钮。常用参数：

- `--run-dir DIR`：要回放的运行目录，默认是 `outputs/sharpa/right/whisking/0`
- `--no-skip-warmup`：包含前面的 warmup / settling 帧，默认会跳过
- `--port N`：viser 端口，默认 `8081`
- `--scene scene.xml --traj trajectory_mjwp.npz`：显式指定场景和轨迹文件
- `--camera-mode ego|scene|auto`：选择第一视角或普通场景相机；`auto` 根据
  `viewpoint` 元数据选择
- `--viewpoint ego|exo|auto`：覆盖输入视角元数据

统一 Deployment 入口还支持 `--render-mode auto|hand-only|full-arm`。其中
`hand-only` 就是调用本 viewer，显示自由根 Sharpa 手与物体；`full-arm` 才会
继续求解 UR3e 机械臂 IK：

```bash
cd "$(git rev-parse --show-toplevel)"
./deployment/run_pipeline.sh mujoco-replay \
  --viewpoint ego --render-mode hand-only \
  --traj retargeting/outputs/sharpa/right/<task>/<id>/trajectory_mjwp.npz
```

仓库内附带的 `whisking` 示例可以直接运行，不需要先重新跑完整流水线；并且 **三层内容都可以正常显示**，因为它已经打包了 viewer 所需的全部资源：

- `scene.xml`、`trajectory_mjwp.npz`、`config.yaml`：重定向结果层（实体）
- `trajectory_kinematic.npz`：IK 参考层（蓝色半透明）
- `outputs/mano/right/whisking/0/trajectory_keypoints.npz`：MANO 参考层（橙色）
- `scene.xml` 所引用的 `outputs/assets/` 下网格资源，包括 `objects/whisking/visual.obj` 与 `convex/`；其中 `visual.obj` 也会作为 MANO 层的物体网格使用，以及引用到的 `robots/sharpa/meshes/*.STL`

当你在自己的视频上运行完整流水线时，只要对应文件存在，每一层都会自动启用；不存在的层则会被静默跳过。

## 致谢与许可证

本代码库基于 [SPIDER](https://github.com/facebookresearch/spider) 构建。物理优化使用 [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp)，IK 使用 [mink](https://github.com/kevinzakka/mink)，可视化使用 [viser](https://github.com/nerfstudio-project/viser)。感谢这些项目原作者的工作。
