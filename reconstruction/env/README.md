# Conda environments

The pipeline switches between **4 conda envs** (names set in `config/paths.sh`).

**Recommended:** build each env by following its fork's own setup instructions —
[`malik-group/sam3`](https://github.com/malik-group/sam3) (`sam3`),
[`mengjq-research-projects/sam-3d-objects`](https://github.com/mengjq-research-projects/sam-3d-objects) (`sam3d`),
[`malik-group/HaWoR`](https://github.com/malik-group/HaWoR) (`hawor`),
[`malik-group/tapnet`](https://github.com/malik-group/tapnet) (`tapnet`).

The two options below are **fallbacks** (e.g. on Blackwell / RTX 50xx, where older upstream CUDA pins won't run):

- **Build fresh** — `./setup/01_create_envs.sh` (or one at a time: `./setup/01_create_envs.sh sam3|sam3d|hawor|tapnet`). Builds each env from the upstream repos' own dependency files + the recipes in the script. 
- **From the exact pins here** — the `env/*.yml` files are full `conda env export`s of known-working envs, to be treated mainly as a version reference for manual installation.

After the `sam3d` env is set up — whether via your own steps or the commands above —
run this once to un-shadow the repo's `notebook/` package:
```bash
pip uninstall -y notebook   # ensure `notebook.inference` imports cleanly
```

The `sam3d` env also needs GeoCalib for the gravity-estimation step
(`scripts/predict_video_gravity.py`; installed automatically by `01_create_envs.sh`):
```bash
pip install "geocalib @ git+https://github.com/cvg/GeoCalib.git"
```
Note: `env/sam3d.yml` predates this addition — if installing from the exact pins, add geocalib on top.

If SAM 3D inference crashes during GLB/texture baking with `NameError: name 'GaussianRasterizationSettings' is not defined` try below:

```bash
conda activate sam3d
git clone --recursive https://github.com/autonomousvision/mip-splatting.git
cd mip-splatting/submodules/diff-gaussian-rasterization
CUDA_HOME=$CONDA_PREFIX TORCH_CUDA_ARCH_LIST=12.0 FORCE_CUDA=1 python setup.py install
```

HaWoR's DROID-SLAM fork targets Blackwell `sm_120`. The setup script therefore
uses the known-working CUDA 12.8, GCC/G++ 14.3, Torch 2.9 + cu128 combination
from `env/hawor.yml`. It also applies the required lietorch `dispatch.h`
compatibility change automatically before building:
```diff
-    at::ScalarType _st = ::detail::scalar_type(the_type);
+    at::ScalarType _st = the_type.scalarType();
```

| env (default name) | exact-pin YAML | stages | deps from |
|---|---|---|---|
| `sam3`   | `env/sam3.yml`   | 1 — SAM3 segmentation        | `env/sam3.yml` (or refer to original repository) |
| `sam3d`  | `env/sam3d.yml`  | 2, 3, 4 — meshes, pose, opt  | `modules/sam-3d-objects/environments/default.yml` + `requirements*.txt` |
| `hawor`  | `env/hawor.yml`  | 2 — hand reconstruction      | `modules/HaWoR/requirements.txt` + CUDA 12.8 / torch 2.9 cu128 |
| `tapnet` | `env/tapnet.yml` | 2.5 — velocity tracking      | `modules/tapnet[torch]` (Python 3.10, torch 2.7 cu128) |

