# 环境说明

## 新人一键准备

代码到达 `PhysicsAssets` 后，在仓库根目录只需要执行：

```bash
./setup_all.sh --managed-offline
```

这条命令完全离线，并且可以安全重复运行。它会自动完成：

1. 核对五个 submodule 的固定提交，但不连接 GitHub；
2. 校验 `/data/jiaqimeng/retargeting_dev` 中的模型和外部资源；
3. 生成 SAM3D 所需的轻量运行时链接并立即校验；
4. 恢复和检查 `sam3`、`sam3d`、`hawor`、`tapnet` 四个 Conda 环境；
5. 恢复 Retargeting venv；
6. 恢复完整的 Isaac Sim 6.0.1.0 venv；
7. 检查 Python、关键 import、CUDA 和 A100 计算能力。

因此新人不需要单独运行 `prepare-runtime`、`verify-runtime`、`uv sync`、
`pip install` 或权重下载命令。需要预览操作时使用：

```bash
./setup_all.sh --managed-offline --dry-run
```

## 环境与入口

| 阶段 | 托管环境 | Shell 入口 |
|---|---|---|
| Reconstruction | `sam3`、`sam3d`、`hawor`、`tapnet` | `./reconstruction/run_pipeline.sh` |
| Retargeting | `installed-envs/venv/retargeting` | `./retargeting/run_pipeline.sh` |
| Isaac Export / USD / Replay | `installed-envs/venv/isaac` | `./isaac_export/run_pipeline.sh` |
| 可选真实部署 | 独立 `deployment` 环境及专有 SDK | `./deployment/run_pipeline.sh` |

各 Shell 入口会自动选择正确解释器，用户不需要直接执行内部 `.py` 文件。
如果需要在当前终端查看解析后的环境路径，可以选择执行：

```bash
source dependency_management/activate.sh
printf '%s\n' "$ENV_SAM3" "$ENV_SAM3D" "$RETARGETING_PYTHON" "$ISAAC_PYTHON"
```

这一步只设置便捷变量，不是运行流水线的前置条件。

## 外部边界

- 版本化依赖根目录：`/data/jiaqimeng/retargeting_dev`
- 当前发布：`releases/2026.08-a100`，由 `current` 符号链接选中
- MANO：受许可证限制，必须由获授权的使用者手动提供
- Sharpa Wave SDK 和真实机器人配置：只在 Deployment 阶段需要
- GitHub 操作和新资源下载：由本地跳板机负责，不在远端一键安装中执行

## 维护与排错

`dependency_management/manage.sh` 提供细粒度的资源、环境和校验命令，
仅供发布维护或定位某一个失败步骤使用。正常新人安装始终只运行
`./setup_all.sh --managed-offline`。详细维护说明见
[`dependency_management/README.md`](dependency_management/README.md)。
