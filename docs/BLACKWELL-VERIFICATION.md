# S3063 Blackwell installation verification

Base source commit: 05d3bb50ce2dca90b50b49977a7dc48c3d3036db; working branch s3063-blackwell.

Environment: `/mnt/NAS/pierce/.workstation/S3063/conda/envs/layerpano3d-blackwell`.

```json
{
  "python": "3.10.22 | packaged by conda-forge | (main, Oct  6 2026, 23:02:24) [GCC 14.4.0]",
  "prefix": "/mnt/NAS/pierce/.workstation/S3063/conda/envs/layerpano3d-blackwell",
  "real_prefix": "/mnt/NAS/pierce/.workstation/S3063/conda/envs/layerpano3d-blackwell",
  "torch": "2.7.1+cu128",
  "torch_cuda": "12.8",
  "cuda_arch_list": [
    "sm_75",
    "sm_80",
    "sm_86",
    "sm_90",
    "sm_100",
    "sm_120",
    "compute_120"
  ],
  "capability": [
    12,
    0
  ],
  "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
  "bf16": true
}
```

## Recorded checks
| Check | Exit | Command |
|---|---:|---|
| pip-check | 0 | `/mnt/NAS/pierce/.workstation/S3063/conda/envs/layerpano3d-blackwell/bin/python -m pip check` |
| torch | 0 | `/mnt/NAS/pierce/.workstation/S3063/conda/envs/layerpano3d-blackwell/bin/python scripts/smoke_blackwell.py torch` |
| knn | 0 | `/mnt/NAS/pierce/.workstation/S3063/conda/envs/layerpano3d-blackwell/bin/python scripts/smoke_blackwell.py knn` |
| gaussian | 0 | `/mnt/NAS/pierce/.workstation/S3063/conda/envs/layerpano3d-blackwell/bin/python scripts/smoke_blackwell.py gaussian` |
| attention | 0 | `/mnt/NAS/pierce/.workstation/S3063/conda/envs/layerpano3d-blackwell/bin/python scripts/smoke_blackwell.py attention` |
| depth | 0 | `/mnt/NAS/pierce/.workstation/S3063/conda/envs/layerpano3d-blackwell/bin/python scripts/smoke_blackwell.py depth` |
| imports | 0 | `/mnt/NAS/pierce/.workstation/S3063/conda/envs/layerpano3d-blackwell/bin/python scripts/smoke_blackwell.py imports` |
| gen_refpano.py-help | 0 | `/mnt/NAS/pierce/.workstation/S3063/conda/envs/layerpano3d-blackwell/bin/python gen_refpano.py --help` |
| gen_panodepth.py-help | 0 | `/mnt/NAS/pierce/.workstation/S3063/conda/envs/layerpano3d-blackwell/bin/python gen_panodepth.py --help` |
| gen_traindata.py-help | 0 | `/mnt/NAS/pierce/.workstation/S3063/conda/envs/layerpano3d-blackwell/bin/python gen_traindata.py --help` |
| pasd-run_layers_pasd.py-help | 0 | `/mnt/NAS/pierce/.workstation/S3063/conda/envs/layerpano3d-blackwell/bin/python pasd/run_layers_pasd.py --help` |
| run_layerpano.py-help | 0 | `/mnt/NAS/pierce/.workstation/S3063/conda/envs/layerpano3d-blackwell/bin/python run_layerpano.py --help` |

The final source build and combined smoke also run in logs/final-build.log; no full run.sh, model downloads or pretrained inference were executed.

## Real fixture output

### torch
```text
PASS torch matmul FP32/BF16 2.7.1+cu128 12.8 NVIDIA RTX PRO 6000 Blackwell Server Edition
ALL REQUESTED SMOKE TESTS PASSED
```

### knn
```text
knn native /mnt/NAS/pierce/.workstation/S3063/conda/envs/layerpano3d-blackwell/lib/python3.10/site-packages/simple_knn/_C.cpython-310-x86_64-linux-gnu.so
PASS simple-knn GPU vs brute force [4.666666507720947, 5.333333492279053, 7.333333492279053, 10.666666984558105, 20.33333396911621]
ALL REQUESTED SMOKE TESTS PASSED
```

### gaussian
```text
rasterizer source /mnt/NAS/pierce/.workstation/S3063/conda/envs/layerpano3d-blackwell/lib/python3.10/site-packages/diff_gaussian_rasterization/__init__.py native /mnt/NAS/pierce/.workstation/S3063/conda/envs/layerpano3d-blackwell/lib/python3.10/site-packages/diff_gaussian_rasterization/_C.cpython-310-x86_64-linux-gnu.so
PASS custom gaussian four-output forward/backward [4, 4] color sum 10.564342498779297
ALL REQUESTED SMOKE TESTS PASSED
```

### attention
```text
PASS project DINO normal-Tensor fallback forward/backward vs SDPA
PASS project PASD SDPA forward/backward torch.float32
PASS project PASD SDPA forward/backward torch.float16
PASS project PASD SDPA forward/backward torch.bfloat16
ALL REQUESTED SMOKE TESTS PASSED
```

### depth
```text
depth native /mnt/NAS/pierce/.workstation/S3063/conda/envs/layerpano3d-blackwell/lib/python3.10/site-packages/instaOmniDepth/depthmapAlign.cpython-310-x86_64-linux-gnu.so
depth native /mnt/NAS/pierce/.workstation/S3063/conda/envs/layerpano3d-blackwell/lib/python3.10/site-packages/EigenSolvers.cpython-310-x86_64-linux-gnu.so
PASS original instaOmniDepth Ceres optimization MAE before 0.5 after 0.10364995608932522
ALL REQUESTED SMOKE TESTS PASSED
```

### imports
```text
PASS import src.pipeline_flux
PASS import src.pipeline_flux_fill
PASS import src.Infusion.depth_inpainting.inference.depth_inpainting_pipeline_half
PASS import src.Depth_Anything_V2.depth_anything_v2.dpt
PASS import utils.lama
PASS import utils.depth_alignment
PASS import gaussian_renderer
PASS import scene
PASS import LayerPano
PASS import transformers
PASS import segment_anything
PASS import optional SR pasd.pipelines.pipeline_pasd
PASS import optional SR pasd.models.pasd.unet_2d_condition
PASS import optional SR pasd.models.pasd.controlnet
PASS import optional SR pasd.myutils.vaehook
PASS import optional SR open_clip
PASS scalar depth pyramid and PASD no-xformers fallback routing
ALL REQUESTED SMOKE TESTS PASSED
```

## Changes / resolved failures
- Explicit cstdint/cfloat includes fix CUDA12.8/GCC13 transitive-header build failures; original kernels unchanged.
- Ceres2.1 user-prefix CMake compiles the original sources; Eigen3.4.0 pinned after actual Eigen5 discovery failure. No sudo/system prefixes.
- xformers0.0.31 FP16 default Hopper FA3 runtime failed; package removed, existing DINO/Infusion/PASD fallbacks selected and exercised. PASD flag defaults False and VAE availability guard calls its function.
- scikit-image scalar pyramid keyword migrated to channel_axis=None and exercised.
- natten/mmcv removed: no active native pipeline import/callsite; mmcv attribution comments are not dependencies.
- OpenCV Python/headless wheels aligned at4.10.0.84 for existing albumentations/ultralytics metadata. C++ OpenCV4.12 is separately provided by Conda.

## Evidence / boundary
All logs, explicit/export/freeze, compiler/version information: `/mnt/NAS/pierce/.workstation/S3063/layerpano3d`.
First failed attempts are retained under logs/bootstrap.log, stage2.log, stage3.log, stage4.log; successful checks under verification/.
Original repo checkpoint inventory had no .pth/.ckpt/.safetensors/.bin weights. HF cache had only an unrelated MoGe model.pt plus empty Wan/FantasyWorld model directories; required LayerPano weights were not present. See INSTALL-BLACKWELL.md for the precise required assets.
No pretrained end-to-end generation, full-resolution peak-memory/performance, alternate ZoeDepth/BoostingMonocularDepth branches, nested DINO tensors or desktop/GUI rendering is verified. Conda emitted an unused librsvg/gdk-pixbuf post-link warning; headless required operations passed.
No shared base, driver, system installation, reboot, other jobs or existing weights were modified.
Native wheels/build scratch retained under personal NAS tmp/layerpano3d-stage3 and tmp/layerpano3d-final.
