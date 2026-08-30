# ObjectNav remote deployment status

Last updated: 2026-08-30 (Asia/Shanghai)

This document records the environment and asset preparation completed on the
remote workstation `admin1-WS-E900-G4-WS980T`. It is a deployment record, not
an experiment specification or a formal result. No ObjectNav evaluation or
TTA experiment has been launched. The existing research-plan scope in
`README.md` and `STATUS.md` is unchanged until a separate protocol decision is
recorded.

## Roots and pinned sources

| Item | Path or revision |
| --- | --- |
| NavTTA checkout | `/data1/wxy/code/NavTTA` |
| NavTTA server baseline before this document | `8ad73d9cefcb7153832cdbae69c849f253829c11` |
| External ObjectNav storage | `/data1/wxy/exp_data/NavTTA/objectnav` |
| PONI checkout | `/data1/wxy/exp_data/NavTTA/objectnav/repos/PONI` at `30682c2bdcd820eec8f72043b2579eb045d547bf` |
| GOAL checkout | `/data1/wxy/exp_data/NavTTA/objectnav/repos/GOAL` at `3687643343d7a4a9a9bfda89d8558c65cef07719` |
| Habitat-Lab source | `bc85d0961cef3b4a08bc9263869606109fb6ff0a` |
| Habitat-Sim source | `fc7fb11ccec407753a73ab810d1dbb5f57d0f9b9` |
| `astar_pycpp` source | `7b1e9ea2413b30feca8d457fefbff2ef94da37cb` |

The PONI and GOAL working trees are runtime checkouts outside the top-level
Git repository. Research changes must not be made in those checkouts.

## Environments

Two isolated Conda-prefix environments are configured:

| Profile | Prefix | Confirmed core components |
| --- | --- | --- |
| GOAL evaluation | `/data1/wxy/exp_data/NavTTA/objectnav/envs/goal-eval` | Python 3.8, PyTorch `1.12.0+cu116`, Habitat-Lab `0.2.1`, Habitat-Sim `0.2.1`, `spconv-cu116` `2.3.6`, editable `navtta-core` |
| PONI evaluation | `/data1/wxy/exp_data/NavTTA/objectnav/envs/poni-eval` | Clone of the GOAL environment plus Detectron2 `0.6` CPU-only, Pillow `9.5.0`, and PONI inference dependencies; editable `navtta-core` |

`PONI/dependencies/astar_pycpp/astar.so` is compiled. GOAL uses it through:

```text
/data1/wxy/exp_data/NavTTA/objectnav/repos/GOAL/nav/astar_pycpp
  -> /data1/wxy/exp_data/NavTTA/objectnav/repos/PONI/dependencies/astar_pycpp
```

Headless Habitat rendering requires the system NVIDIA GL libraries to take
precedence over the Conda GLVND libraries:

```bash
LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:"$ENV/lib"
LD_PRELOAD=/lib/x86_64-linux-gnu/libGLX_nvidia.so.0:/lib/x86_64-linux-gnu/libGLdispatch.so.0
__GLX_VENDOR_LIBRARY_NAME=nvidia
__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
```

The Gym maintenance notice and the `timm.models.layers` deprecation warning
are expected for these pinned legacy environments and are not installation
failures.

## MP3D scenes

The canonical scene tree is:

```text
/data1/wxy/data/scene_datasets/mp3d
```

It contains 90 MP3D scenes; the GLB, navmesh, house, and semantic PLY coverage
was checked during setup. These links ultimately resolve to that tree:

```text
/data1/wxy/exp_data/NavTTA/shared/scene_datasets/mp3d
/data1/wxy/code/NavTTA/vln/data/scene_datasets/mp3d
/data1/wxy/code/NavTTA/objectnav/data/scene_datasets/mp3d
/data1/wxy/exp_data/NavTTA/objectnav/repos/PONI/data/scene_datasets/mp3d
/data1/wxy/exp_data/NavTTA/objectnav/repos/GOAL/data/scene_datasets/mp3d
```

## MP3D ObjectNav episodes

The official Habitat archive is retained at:

```text
/data1/wxy/exp_data/NavTTA/objectnav/assets/downloads/objectnav_mp3d_v1.zip
```

| Property | Value |
| --- | --- |
| Source | `https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/m3d/v1/objectnav_mp3d_v1.zip` |
| Size | `180952652` bytes |
| SHA256 | `b1c9cd40bd94043705e20539034254a0bf8c49dd405d93dab0cef75ae2fd5bdf` |
| Canonical extracted root | `/data1/wxy/exp_data/NavTTA/objectnav/assets/datasets/objectnav/mp3d/v1` |
| Derived validation partitions | `val_part_0` through `val_part_10` |
| Validation coverage | 11 scenes, 2195 episodes |
| `val_parts` tree SHA256 | `9578c01b6355ff06e073cf9dae08184b560de1fb211b339c28fa22d57d82a17e` |

The shared `mp3d` episode parent is exposed at the following consumer paths,
so each evaluator reads `mp3d/v1/val_parts/...` without duplicating data:

```text
/data1/wxy/code/NavTTA/objectnav/data/datasets/objectnav/mp3d
/data1/wxy/exp_data/NavTTA/objectnav/repos/PONI/data/datasets/objectnav/mp3d
/data1/wxy/exp_data/NavTTA/objectnav/repos/GOAL/data/datasets/objectnav/mp3d
```

Each path links to:

```text
/data1/wxy/exp_data/NavTTA/objectnav/assets/datasets/objectnav/mp3d
```

## Checkpoints

Checkpoint binaries remain outside Git under
`/data1/wxy/exp_data/NavTTA/objectnav/checkpoints/`.

### GOAL

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `goal/area_potential.pth` | 52570961 | `0b2460ef3dfb8f4b8cab44815340eb638753fa4d8fa4d4ee1ec2f88bf8f3a6fa` |
| `goal/mp3d_chatgpt.pth` | 555076411 | `9a2867e5673a83c3e6c2d66a6a2c834050e0b72082f68c47b8d1980fddbebcba` |
| `goal/spconv_state.pth` | 156818103 | `bc5d1244a78646fb333ab7d84fafb90a07dc2283c4924014e95c6b23cb200bda` |

The runtime link is:

```text
/data1/wxy/exp_data/NavTTA/objectnav/repos/GOAL/pretrained_models
  -> /data1/wxy/exp_data/NavTTA/objectnav/checkpoints/goal
```

### PONI

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `poni/mp3d_models/poni_seed_123.ckpt` | 52570961 | `0b2460ef3dfb8f4b8cab44815340eb638753fa4d8fa4d4ee1ec2f88bf8f3a6fa` |
| `poni/rednet_mp3d.pth` | 656482964 | `a50eca38d1babf992c9bfd24de24f9287190942cba7c2b606d7270c3781820cd` |

The MP3D PONI checkpoint is byte-identical to GOAL's
`area_potential.pth`. Runtime links are:

```text
/data1/wxy/exp_data/NavTTA/objectnav/repos/PONI/pretrained_models/mp3d_models/poni_seed_123.ckpt
  -> /data1/wxy/exp_data/NavTTA/objectnav/checkpoints/poni/mp3d_models/poni_seed_123.ckpt

/data1/wxy/exp_data/NavTTA/objectnav/repos/PONI/pretrained_models/rednet_mp3d.pth
  -> /data1/wxy/exp_data/NavTTA/objectnav/checkpoints/poni/rednet_mp3d.pth
```

## Completed validation

- PyTorch CUDA and `spconv` CUDA smoke tests passed on an NVIDIA RTX 3090.
- Habitat-Sim `0.2.1` loaded an MP3D scene, navmesh, and RGB sensor through
  headless NVIDIA EGL.
- GOAL imports and all three checkpoint strict-load tests passed:
  sparse UNet `39158038` parameters, flow model `138756608` parameters, and
  area-potential model `4368984` parameters.
- PONI imports, CUDA, EGL, checkpoint structure checks, and strict loads
  passed: PONI `4368984` parameters and RedNet `81964049` parameters.
- `pip check` reported no broken requirements in the prepared environments.

## Known runtime state and remaining gate

- Importing GOAL rewrites bytecode files that its upstream repository tracks,
  so that external checkout currently reports modified `__pycache__/*.pyc`.
  Its `nav/astar_pycpp` and `pretrained_models` runtime links are untracked.
  These are deployment artifacts rather than NavTTA source changes.
- PONI's configured success distance is 0.1 m, while GOAL's is 1.0 m. A
  unified comparison protocol must be explicitly frozen before producing any
  cross-model result.
- Dataset and checkpoint provenance must be promoted into task manifests, and
  the project-level ObjectNav baseline decision must be updated, before a run
  can qualify as a formal NavTTA result.
