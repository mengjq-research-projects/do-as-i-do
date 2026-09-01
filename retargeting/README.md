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

## Ego Mug 回归案例：从原始 MP4 到 data-id 5

本案例的规范化端到端记录位于根目录
[`EGO_DEMO_WALKTHROUGH.md`](../EGO_DEMO_WALKTHROUGH.md)，其中还包含最终回放、
动态背景、时间锚点和目标 RL 展示层的独立 overfitting 归因。本节保留
Retargeting 模块内的参数细节。

本节记录 `mug_pickplace_10s` 的**实际调试历史**，目的是同时回答两个问题：

1. 最终可用的 `data-id 5` 是怎样从原始 Ego MP4 产生的；
2. 为了跑通这个 case，哪些地方加入了视频、物体或任务级先验。

这不是一条干净、可泛化到任意视频的推荐配置。核心代码中没有
`if task == "mug_pickplace_10s"` 一类运行时分支；实现形式是通用 opt-in
机制，但本 case 为这些机制提供了 Mug 专用数值，因此仍然属于 case overfitting。

### 0. 原始视频与 Reconstruction

输入视频：

```text
/data/jiaqimeng/do-as-i-do-runs/mug_pickplace_10s/mug_pickplace_10s.mp4
```

实际执行的 Reconstruction 命令：

```bash
cd ~/projects/do-as-i-do
source dependency_management/activate.sh
cd reconstruction

CUDA_VISIBLE_DEVICES=4 PYTHONUNBUFFERED=1 ./run_pipeline.sh \
  /data/jiaqimeng/do-as-i-do-runs/mug_pickplace_10s/mug_pickplace_10s.mp4 \
  35 \
  mug \
  right \
  "855,690;890,716;760,660;960,660" \
  "1;1;0;0" \
  ego \
  moving
```

这里已经包含视频级人工输入：参考帧 `35`、物体类别 `mug`、右手、四个
提示点及其正负标签，以及 `ego/moving` 相机元数据。它们不是 Retargeting
算法自动估计的通用参数。

### 1. Retargeting trial 演进

下表按输出文件时间校正实际先后。`0_before_upright_20260824` 是第一次 id 0
结果的归档；当前 `0/` 已被后来的 upright trial 覆盖。

| Trial | 相对上一版新增或改变的先验 | 结果 |
|---|---|---|
| `0_before_upright_20260824` | 无 Mug 姿态/撤手/杯把先验 | 原始基线 |
| `0` | `--object-upright`，当时仍使用默认 mesh local `+Z` | `pos=.0575, quat=.1846` |
| `1` | 斜向 up vector、末尾撤手、强制首尾 pedestal、移除 UR3 wrist collision proxy | `pos=.0217, quat=.0432` |
| `2` | 将真实杯口轴改为 mesh local `+Y` | `pos=.0225, quat=.0463` |
| `3` | 增加杯把锚点 `[0,0,-.055]` | `pos=.0227, quat=.0489`，末段仍拖杯 |
| `4` | in-hand 阈值 `0.1 -> 0.015 m` | upright 开关漏传，杯子姿态错误，弃用 |
| `5` | 正确 `+Y` reference + `0.015 m` gate 重新跑 physics | `pos=.0170, quat=.0378`，最终结果 |

#### 1.1 原始 id 0 与 upright id 0

原始 id 0 在 GPU 4 运行后又在 GPU 3 重试；换 GPU 不构成算法参数变化：

```bash
cd ~/projects/do-as-i-do
source dependency_management/activate.sh
cd retargeting

CUDA_VISIBLE_DEVICES=4 PYTHONUNBUFFERED=1 ./run_pipeline.sh \
  --task mug_pickplace_10s \
  --raw-dir /data/jiaqimeng/do-as-i-do-runs/mug_pickplace_10s \
  --data-id 0 \
  --no-show-viewer \
  --no-wait-on-finish
```

GPU 3 重试使用完全相同的命令，只将 `CUDA_VISIBLE_DEVICES=4` 改为
`CUDA_VISIBLE_DEVICES=3`。

upright 版本没有显式传 `--data-id`，因此默认仍写入并覆盖 id 0：

```bash
cd ~/projects/do-as-i-do/retargeting

CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 ./run_pipeline.sh \
  --raw-dir /data/jiaqimeng/do-as-i-do-runs/mug_pickplace_10s \
  --task mug_pickplace_10s \
  --hand-type right \
  --object-upright \
  --no-show-viewer \
  --no-wait-on-finish
```

#### 1.2 data-id 1：斜向杯口轴、末尾撤手、移除 UR3 wrist collision proxy

第一次没有强制端点 pedestal，运行在 `700/1629` 提前终止：

```bash
CUDA_VISIBLE_DEVICES=3 ./run_pipeline.sh \
  --task mug_pickplace_10s \
  --raw-dir /data/jiaqimeng/do-as-i-do-runs/mug_pickplace_10s \
  --hand-type right \
  --robot-type sharpa \
  --data-id 1 \
  --object-upright \
  --object-up-vector 0.03157368 0.31375258 -0.94897968 \
  --terminal-hand-retreat \
  --no-add-ur3-arm \
  --no-show-viewer \
  --no-wait-on-finish
```

随后用同一 id 强制起点和终点支撑，覆盖得到当前 id 1：

```bash
CUDA_VISIBLE_DEVICES=3 ./run_pipeline.sh \
  --task mug_pickplace_10s \
  --raw-dir /data/jiaqimeng/do-as-i-do-runs/mug_pickplace_10s \
  --hand-type right \
  --robot-type sharpa \
  --data-id 1 \
  --object-upright \
  --object-up-vector 0.03157368 0.31375258 -0.94897968 \
  --terminal-hand-retreat \
  --force-pedestal-start \
  --force-pedestal-end \
  --no-add-ur3-arm \
  --no-show-viewer \
  --no-wait-on-finish
```

#### 1.3 data-id 2：将杯口语义轴写为 mesh local +Y

```bash
CUDA_VISIBLE_DEVICES=4 ./run_pipeline.sh \
  --task mug_pickplace_10s \
  --raw-dir /data/jiaqimeng/do-as-i-do-runs/mug_pickplace_10s \
  --hand-type right \
  --robot-type sharpa \
  --data-id 2 \
  --object-upright \
  --object-up-vector 0 1 0 \
  --terminal-hand-retreat \
  --force-pedestal-start \
  --force-pedestal-end \
  --no-add-ur3-arm \
  --no-show-viewer \
  --no-wait-on-finish
```

`[0,1,0]` 是这个 Mug mesh 的局部杯口方向，不是其他物体可以直接复用的
世界坐标约束。处理阶段将该局部方向对齐世界 `+Z`，删除物体 roll/pitch、保留
yaw，并对手施加相同的逐帧刚体旋转以维持手物相对几何。

#### 1.4 data-id 3：增加杯把抓取锚点

```bash
CUDA_VISIBLE_DEVICES=4 ./run_pipeline.sh \
  --task mug_pickplace_10s \
  --raw-dir /data/jiaqimeng/do-as-i-do-runs/mug_pickplace_10s \
  --hand-type right \
  --robot-type sharpa \
  --data-id 3 \
  --object-upright \
  --object-up-vector 0 1 0 \
  --hand-grasp-anchor-vector 0 0 -0.055 \
  --terminal-hand-retreat \
  --force-pedestal-start \
  --force-pedestal-end \
  --no-add-ur3-arm \
  --no-show-viewer \
  --no-wait-on-finish
```

`[0,0,-0.055] m` 是这个 Mug mesh 的杯把附近锚点。代码自动检测闭合的
拇指–食指夹点何时接近该锚点，并在抓持区间刚体平移手；抓持帧不是命令中写死
的帧号。

#### 1.5 data-id 4：收紧 in-hand gate，但漏开 upright

```bash
cd ~/projects/do-as-i-do/retargeting

CUDA_VISIBLE_DEVICES=4 \
/data/jiaqimeng/retargeting_dev/current/installed-envs/venv/retargeting/bin/python \
launch.py \
  --raw-dir /data/jiaqimeng/do-as-i-do-runs/mug_pickplace_10s \
  --task mug_pickplace_10s \
  --hand-type right \
  --data-id 4 \
  --no-add-ur3-arm \
  --object-up-vector 0 1 0 \
  --terminal-hand-retreat \
  --hand-grasp-anchor-vector 0 0 -0.055 \
  --hand-object-distance-thresh 0.015 \
  --force-pedestal-start \
  --force-pedestal-end \
  --max-sim-steps 1629 \
  --no-show-viewer \
  --no-wait-on-finish
```

当时接口仍要求单独传 `--object-upright`，因此 id 4 静默忽略了 up vector。
其 `trajectory_keypoints.npz` 明确记录
`object_upright=False, object_up_vector=[0,0,1]`。当前 `launch.py` 已改为只要
传入 `--object-up-vector` 就自动启用 upright，避免重复这个错误。

### 2. 最终 data-id 5 的真实执行边界

远端交互 history 中**没有**一条完整的 id 5 `run_pipeline.sh` 命令。能够精确
确认的最终 GPU 命令只运行物理阶段：

```bash
cd /home/jiaqimeng/projects/do-as-i-do/retargeting

CUDA_VISIBLE_DEVICES=4 \
/data/jiaqimeng/retargeting_dev/current/installed-envs/venv/retargeting/bin/python - <<'PY'
from launch import load_mjwp_config
from retargeting.pipeline.optimize_physics import main as optimize_physics

config = load_mjwp_config(
    dataset_name="do_as_i_do",
    task="mug_pickplace_10s",
    data_id=5,
    robot_type="sharpa",
    embodiment_type="right",
    seed=0,
    wait_on_finish=False,
    max_sim_steps=1629,
    force=True,
    show_viewer=False,
    hand_object_distance_thresh=0.015,
)
optimize_physics(config)
PY
```

运行这条命令前，id 5 的 CPU reference、场景和 IK 已经生成。产物证据为：

- id 3 与 id 5 的 `trajectory_keypoints.npz` SHA-256 完全相同；
- id 3 与 id 5 的 `trajectory_kinematic.npz` SHA-256 完全相同；
- id 3 与 id 5 的 `scene_ik.xml` 完全相同；
- 两份 physics config 除 trial 路径外，唯一参数变化是
  `hand_object_distance_thresh: 0.1 -> 0.015`；
- `scene.xml` 只存在 pedestal/support 尺寸的亚毫米差异。

因此，准确的继承关系是：

```text
id 3 的 upright(+Y) + retreat + grasp-anchor reference/IK
    + 1.5 cm in-hand/rest gate
    + 重新解析 pedestal/support
    + 重新运行 physics
    = 最终 id 5
```

id 5 最终输出：

```text
outputs/sharpa/right/mug_pickplace_10s/5/trajectory_mjwp.npz
Final object tracking error: pos=0.0170, quat=0.0378
```

`config.yaml` 不序列化 `object_upright`、up vector、撤手、抓取锚点、强制
pedestal 或 `no-add-ur3-arm` 等上游参数，因此不能只读最终 YAML 回溯完整
trial。需要同时检查 MANO NPZ metadata、`task_info.json`、`scene.xml` 和命令历史。

### 3. 当前代码下的一条完整等价命令

下面命令表达 id 5 的**有效配置**，便于今后做回归；它不是远端 history 中实际
执行过的单条命令，也不能保证在代码或 CUDA 版本变化后逐字节复现旧轨迹：

```bash
cd ~/projects/do-as-i-do/retargeting

CUDA_VISIBLE_DEVICES=<IDLE_GPU> ./run_pipeline.sh \
  --task mug_pickplace_10s \
  --raw-dir /data/jiaqimeng/do-as-i-do-runs/mug_pickplace_10s \
  --hand-type right \
  --robot-type sharpa \
  --data-id 5 \
  --object-upright \
  --object-up-vector 0 1 0 \
  --hand-grasp-anchor-vector 0 0 -0.055 \
  --terminal-hand-retreat \
  --hand-object-distance-thresh 0.015 \
  --force-pedestal-start \
  --force-pedestal-end \
  --max-sim-steps 1629 \
  --no-add-ur3-arm \
  --no-show-viewer \
  --no-wait-on-finish
```

该命令会使用 GPU；先用 `nvidia-smi` 选择空闲卡，再由操作者显式运行。

### 4. Overfitting 清单

| 参数或机制 | id 5 值 | 分类与影响 |
|---|---|---|
| Reconstruction 参考帧/点提示 | frame 35 + 四个坐标/标签 | 视频级强 overfit |
| `object_up_vector` | `[0,1,0]` | Mug mesh 局部杯口轴，物体级强 overfit |
| `hand_grasp_anchor_vector` | `[0,0,-.055] m` | 该 mesh 杯把抓点，物体/抓取级强 overfit |
| `terminal_hand_retreat` | 开启 | 把末尾 tracker 丢失解释为手撤离，pick-place 行为先验 |
| 强制首尾 pedestal | 两端开启 | 假设起点和终点均放置在桌面，任务状态先验 |
| `hand_object_distance_thresh` | `0.015 m` | 针对此手物尺度调节的 gate |
| `no-add-ur3-arm` | 开启 | 移除 palm 上模拟 UR3e wrist/forearm 体积的无质量碰撞圆柱，属于场景/碰撞模型简化 |
| `max_sim_steps` | `1629` | 与该视频长度绑定；本 case 等于自动完整长度，影响较小 |

opt-in 机制内部还有固定默认值：terminal retreat 用最近 12 个有效帧估速且速度
上限为 `0.06 m/frame`；抓取锚点要求拇指–食指开度 `< 0.06 m`、距锚点
`< 0.04 m`，前后 fade 5 帧。这些机制不是按任务名分支，但仍需在更多数据上验证。

这里的 `--no-add-ur3-arm` 与 Deployment 的 `--render-mode hand-only/full-arm`
不是一回事：前者改变 Retargeting 场景中的碰撞代理，后者只选择回放自由根手，
还是继续求解并显示完整 UR3e。

### 5. 最终成功来自什么，不来自什么

归一化 trial 路径后，id 0–5 的 physics `config.yaml` 只有两个实质差异：

```text
hand_object_distance_thresh: id0-3 = 0.1, id4-5 = 0.015
npair:                       id0 = 602, id1-5 = 700
```

Reward scale 没有逐 trial 调节，最终也不是 contact guidance：

```text
contact_guidance = false
contact_rew_scale = 0
drop_penalty_scale = 0
```

真正 active 的放置/释放约束是 `pedestal_penalty_scale=0.5` 的二值接触状态
不匹配惩罚。`0.015 m` gate 将参考分成手持和放置区间：举起阶段要求实际
hand-object contact，放置阶段要求实际 object-pedestal contact。对 id 5 复算，
约 176 帧要求手物接触，1694 帧要求物体支撑接触；后者约 1094 帧位于 warmup
之后。因此 id 5 相对 id 3 的主要物理改进来自 gate 及其驱动的 pedestal contact
约束，而不是新 IK 或新增 contact reward。

随机扰动参数在候选规划机制中仍配置为非零，但 id 5 的 held-and-lifted 连续段
经过 100 步双边时间侵蚀后，最终 perturb gate 为 `0/2329`，这个 case 没有帧
实际施加随机力/矩。3 秒/600 步 warmup、warmup weld + 零重力、初始 clearance、
penetration penalty 等属于 `do_as_i_do` 通用稳定化配置，不是 Mug CLI hardcode。

### 6. 最终回放与展示层 overfitting

最终稳定俯视回放命令：

```bash
cd ~/projects/do-as-i-do

nohup ./deployment/run_pipeline.sh mujoco-replay \
  --render-mode hand-only \
  --camera-mode top-down \
  --traj retargeting/outputs/sharpa/right/mug_pickplace_10s/5/trajectory_mjwp.npz \
  --speed 0.25 \
  --ego-fov 50 \
  --ego-distance 0.75 \
  --port 8081 \
  >/tmp/mug_pickplace_id5_replay.log 2>&1 &
```

`hand-only`、`top-down`、颜色和背景只改变展示，不修改
`trajectory_mjwp.npz`。在 `top-down` 分支中，`--ego-distance 0.75` 实际不参与
相机计算。

最终左右对比视频还有一层独立的 presentation overfitting：15 个手工时间锚点、
固定俯视相机、Viser 白色手体/绿色指尖、隐藏 collision/support，以及从原视频
提取的静态 clean plate。原视频背景版还使用：

```text
foreground_affine = [[0.8432, -0.1315, 227.6],
                     [0.1315,  0.8432, 237.25]]
shadow_opacity = 0.12
```

这些设置只让最终视频更接近原视频，不参与 Reconstruction、IK 或 physics。

#### 6.1 动态相机与目标 RL 演进 2×2 Demo

这个视频是给组内说明**未来要实现的 RL 效果**的 concept/target demo，不是在声称
当前项目已经训练出 RL policy。使用 `render-evolution --narrative target-rl`
生成四格同步叙事：

```text
Human Ego Demonstration       Before RL — Affordance Failure
Learning — Strategy Evolution Converged Policy — Stable Grasp & Place
```

因此视频主体明确写成 `Before RL -> Learning -> Converged Policy`。现阶段仍使用
kinematic、错误 affordance physics 和最终 MJWP 轨迹作为这三个目标状态的视觉
proxy；它们不是真实 RL checkpoint，也不是已经训练出策略的实验证据。等 RL
系统完成后，应以真实 policy checkpoint、episode/return 和多 seed 评估替换这些
proxy。kinematic panel 中物体使用 reference pose，真正的错误抓取证据仍来自
左下角 id 2 physics。

完整复现命令：

```bash
cd ~/projects/do-as-i-do

RUN_ROOT=/home/jiaqimeng/projects/do-as-i-do/retargeting/outputs/sharpa/right/mug_pickplace_10s
DATA_ROOT=/data/jiaqimeng/do-as-i-do-runs/mug_pickplace_10s

./retargeting/run_pipeline.sh render-evolution \
  --narrative target-rl \
  --source-video "$DATA_ROOT/mug_pickplace_10s.mp4" \
  --background-image "$DATA_ROOT/comparison_id5_background_v1/clean_table_background.png" \
  --mask-root "$DATA_ROOT/video_segmentation/masks" \
  --stage "Before RL - Affordance Failure|$RUN_ROOT/2|trajectory_kinematic.npz|kinematic" \
  --stage "Learning - Grasp Strategy Evolution|$RUN_ROOT/2|trajectory_mjwp.npz|physics" \
  --stage "Converged Policy - Stable Handle Grasp & Place|$RUN_ROOT/5|trajectory_mjwp.npz|physics" \
  --final-stage-yaw-degrees 24 \
  --output "$DATA_ROOT/comparison_target_rl_dynamic_v5/mug_pickplace_target_rl_progression_dynamic.mp4" \
  --panel-width 960 \
  --panel-height 540 \
  --mosaic-mask-dilation 18 \
  --mosaic-cleanplate-mode temporal \
  --preset medium \
  --crf 18
```

该入口只离线渲染已有轨迹，不运行 IK、MJWP/MPC 或 GPU 优化；无图形界面的远端
终端会自动通过 Xvfb 创建软件 OpenGL context。原视频仍是 150 帧、15 FPS 的
master clock。脚本用 Reconstruction 的手/杯 mask 排除动态区域，从桌面纹理
逐帧估计 frame 75 到当前帧的 homography，并对三个仿真 panel 应用完全相同的
pan/roll/zoom/perspective。最终版本 150/150 帧配准成功，桌面重投影 RMSE 中位数
约 `1.13 px`；杯子轨迹自动 similarity alignment 的 inlier error 中位数约
`2.39 px`。`--final-stage-yaw-degrees 24` 根据未遮挡杯把帧校准右下角最终阶段：
它绕逐帧杯子中心对杯子和机械手施加同一个世界 Z 轴刚性旋转，因此保留手—杯
相对接触关系且不修改保存的 `trajectory_mjwp.npz`。校准后，前 10 个未遮挡采样
帧的杯把方向中位绝对偏差从 `24.11°` 降至 `2.95°`。

动态相机不再使用镜像补边。脚本先把 150 帧原视频反投影到 frame 75 的统一坐标，
排除杯子、手和前臂后生成 `1086×640` 的 expanded clean mosaic；高覆盖区使用
稳健时间中值，低覆盖边缘优先选择距离动态区域最远的真实观测。`temporal` 模式
直接使用这个多帧 clean mosaic，不把旧的 `clean_table_background.png` 覆盖回参考
视野；这是因为旧图本身含有低对比度前臂拖影和橙色残留。只有在外部 clean plate
已经单独验证干净时才使用兼容选项 `--mosaic-cleanplate-mode feathered-input`。
mosaic 和最终逐帧采样均使用 premultiplied-alpha
normalized warp 与 constant border，不使用 reflection、replication 或 inpaint。
当前 v3 的 required background hole 为 `0`，150 帧逐帧 uncovered fraction 最大值
为 `0`，同帧 guard pixel 也为 `0`。因此桌沿、木纹、机械手和杯子都不会在画面
边缘产生镜像副本。

输出及完整参数 manifest：

```text
/data/jiaqimeng/do-as-i-do-runs/mug_pickplace_10s/comparison_target_rl_dynamic_v5/
├── mug_pickplace_target_rl_progression_dynamic.mp4
├── mug_pickplace_target_rl_progression_dynamic.json
├── mug_pickplace_target_rl_progression_dynamic_background_mosaic.png
├── mug_pickplace_target_rl_progression_dynamic_mosaic_alpha.png
└── mug_pickplace_target_rl_progression_dynamic_mosaic_coverage.png
```

## 通用运行

先对一段视频运行 reconstruction 流程，例如 whisking 示例；之后在 `retargeting/` 目录中执行：

```bash
./run_pipeline.sh --task whisking --raw-dir ../reconstruction/whisking
```

其中 `--raw-dir` 指向 reconstruction 的输出目录，也就是视频对应目录；`--task` 是该视频的任务名。常用参数：

- `--robot-type sharpa`：目标机器人手，默认是 `sharpa`
- `--no-show-viewer`：关闭 viser viewer，适用于无头环境
- `--max-sim-steps N`：限制优化步数，`0` 表示完整轨迹
- `--no-wait-on-finish`：结束后直接退出，而不是保持 viewer 常驻
- `--object-upright`：将物体网格的语义上轴约束到世界 `+Z`，移除
  roll/pitch、保留逐帧 yaw。`--object-up-axis` 可指定有符号坐标轴；当重建网格
  本身相对坐标轴存在倾斜时，用 `--object-up-vector X Y Z` 指定精确局部向量。
  只用于杯子直立搬运等已知全程不应倾斜的任务；倒水、倾倒等动作不要启用
- `--terminal-hand-retreat`：手在完成放置、退出画面后跟踪丢失时，根据最后几帧
  可见手腕速度外推撤离轨迹。该参数是显式任务先验，不应在遮挡期间仍握持物体的
  case 中启用
- `--hand-grasp-anchor-vector X Y Z`：在物体局部坐标中指定一个语义抓取点；闭合的
  拇指–食指夹点接近该点后，在抓持阶段将手整体平移并稳定到该点。用于杯把等
  已知接触区域，可修正手和物体两套单目三维结果之间的相对漂移；不改变手指姿态
  或物体轨迹
- `--force-pedestal-start/--force-pedestal-end`：明确知道物体在视频起点或终点
  放在桌面上时，强制添加端点 pedestal，避免手部重建误差把相近但未抓持的状态
  误判为 in-hand

下面是最小 upright 示例，不是前文最终 id 5 配置的完整复现命令。例如，重建
产生了错误杯身倾角、但原视频中的杯子始终直立时：

```bash
./run_pipeline.sh \
  --task mug_pickplace_10s \
  --raw-dir /data/jiaqimeng/do-as-i-do-runs/mug_pickplace_10s \
  --hand-type right \
  --object-upright \
  --object-up-vector 0 1 0 \
  --hand-grasp-anchor-vector 0 0 -0.055 \
  --terminal-hand-retreat \
  --force-pedestal-start \
  --force-pedestal-end
```

该约束在阶段 1 写入参考物体姿态，之后会重新执行 IK 和物理优化；不要只修改
最终的 `trajectory_mjwp.npz`，否则手与物体的接触关系不会重新求解。

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
