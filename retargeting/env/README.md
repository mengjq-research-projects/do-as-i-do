# 重定向环境

与 `reconstruction/` 需要四个 Conda 环境不同，retargeting 只需要一个 **单独的 Conda 环境**，并以 pip 包的形式安装：

```bash
conda create -y -n retargeting python=3.12
conda activate retargeting
pip install -e .          # from the retargeting/ directory
```

等价的 `uv` 工作流如下：

```bash
cd retargeting
uv sync
uv run python launch.py --task whisking --raw-dir ../reconstruction/whisking
```

说明：
- `pyproject.toml` 中的 `mujoco-warp` 被固定到一个已验证可用的 commit，它会引入兼容版本的 `mujoco` 和 `warp-lang`。`uv` 在锁定时通过静态元数据解析这个 git 依赖，因此不会继承上游坏掉的 MuJoCo index 覆盖。物理优化阶段仍然需要带 CUDA 的 NVIDIA GPU。
- `pyproject.toml` 会把 `warp-lang` 固定到 NVIDIA 的 package index，并将 `uv` 环境的 Python 版本约束为 3.12。
- 如果你需要的是已验证环境的精确包集合，而不是 `pyproject.toml` 中相对宽松的版本约束，那么 `env/retargeting.yml` 仍然是最准确的导出参考环境。
- 默认的 viser viewer 会启动一个 Web 界面，请在浏览器中打开终端输出的 URL。

## 精确版本参考

如果后续依赖解析结果发生漂移，`env/retargeting.yml` 是一个从已知可用环境导出的完整 `conda env export`，主要用作手动安装时的版本参考。可以在一台已验证可用的机器上通过下面的命令重新生成：

```bash
conda env export -n retargeting > env/retargeting.yml
```
