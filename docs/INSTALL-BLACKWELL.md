# Blackwell installation (S3063)

This is the maintained Linux installation entry point for the RTX PRO 6000
Blackwell Server Edition (compute capability 12.0). The historical CUDA 11.8
recipe is not compatible with this GPU. Do not mix its wheels with this environment.
The application models and the custom four-output rasterizer are not replaced.

## Location and isolation

Repository: `/mnt/NAS/E3DQA_project/E3DQA_generation_methods/LayerPano3D`

Environment: `layerpano3d-blackwell`, prefix
`/mnt/NAS/pierce/.workstation/S3063/conda/envs/layerpano3d-blackwell`.
The shared Miniforge base, driver, other environments and running GPU jobs must
not be modified. No sudo or system-wide installation is required.
S3063's personal shell configuration already redirects Conda, pip, HF and Torch
caches to `/mnt/NAS/pierce/.workstation/S3063/`; keep these defaults. A mount guard
must succeed before any installation. A new SSH session inherits the configuration;
for an old shell source `~/.config/s3063/environment.sh` and check its exit status.

## Create from source

Run from the repository root in a dedicated tmux session (not another user's pane).
The YAML is the human-maintained recipe (Eigen is pinned to Ceres 2.1's exact 3.4.0 ABI requirement, not Conda's default Eigen 5); the installation evidence includes an
exact Conda explicit export and pip freeze for the successfully tested environment.

```bash
source /opt/miniforge/etc/profile.d/conda.sh
conda env create --file environment-blackwell.yml
conda activate layerpano3d-blackwell
export TMPDIR=/mnt/NAS/pierce/.workstation/S3063/tmp/layerpano3d-rebuild-01
mkdir -p "$TMPDIR"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 GLOG_log_dir="$TMPDIR"
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip check
bash scripts/build_native.sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/smoke_blackwell.py
```

Use a fresh TMPDIR for each native rebuild. Build directories and wheels are
retained for debugging, not deleted. `MAX_JOBS=4` limits compile parallelism.
Conda activation selects the GCC/G++ 13 toolchain. The build script sets CUDA_HOME
and CUDACXX to the active environment, compiles `TORCH_CUDA_ARCH_LIST=12.0`, and
uses no global CUDA installation. Confirm `which python`, `which nvcc`,
`nvcc --version`, `$CC --version`, and `$CXX --version` when diagnosing a build.

The original repository contains prebuilt Python 3.9/Windows extensions. These
are not valid here. The build script copies the source into fresh NAS scratch,
excluding old build outputs, builds wheels without build isolation/dependency
resolution, and installs them in the environment (not editable source imports).
Do not install the Infusion confidence rasterizer: it has the same Python package
name but a different interface. `scripts/smoke_blackwell.py` prints the loaded
Python/native paths and checks the root rasterizer's color/radii/depth/alpha outputs.

The native CUDA source now includes `<cstdint>` (rasterizer) and `<cfloat>`
(simple-knn) explicitly. The first CUDA 12.8/GCC 13 build exposed missing
`std::uintptr_t`/`uint32_t` and `FLT_MAX` declarations that older transitive headers
had supplied. No kernel math, nearest-neighbor logic, or research model changed.

360monodepth uses the original `depth_stitch` and `EigenSolvers` sources via
`scripts/instaOmniDepth/CMakeLists.txt`. CMake discovers Python/NumPy, pybind11,
OpenCV C++ headers/libs, Ceres 2.1, Boost, Eigen 3.4.0 and glog inside the environment.
This avoids upstream hard-coded Python/OpenCV paths and system-prefix installation.
The original CMake/setup files are retained as historical source, not the build
entry point. `EigenSolvers` is installed top-level to match `utility/blending.py`;
`instaOmniDepth.depthmapAlign` and its shared library use an environment-local RPATH.
No process-wide LD_LIBRARY_PATH overwrite is needed.

## Dependency decisions

- PyTorch 2.7.1 / torchvision 0.22.1 CUDA 12.8 are a matched binary set. `constraints-blackwell.txt` forbids pip from silently replacing Torch
  with a CPU or CUDA 11.8 build. Install Torch before application requirements.
  xformers is deliberately absent: 0.0.31 passed FP32 but dispatched FP16 to a
  Hopper FlashAttention3 kernel that terminated with CUDA invalid argument on SM120.
  Do not reinstall it into this recipe. DINO uses its original normal-Tensor
  attention fallback, Infusion keeps its existing try/except fallback, and PASD
  explicitly defaults its xformers flag to False and uses existing Torch SDPA.
  This changes an optional backend, not attention math or research weights;
  performance and peak memory are not guaranteed identical. Nested DINO tensors
  requiring xformers are outside the native default pipeline and remain unsupported.
- No active tracked Python import or callsite uses natten or mmcv in this native
  pipeline. The mmcv references in BasicSR utilities are attribution comments.
  These packages and the old cu118 wheel index are excluded, not replaced by a
  different attention/research algorithm.
- Hydra 1.3.2 / OmegaConf 2.3 replace Hydra 1.1's incompatible metadata; NumPy stays
  1.26.4 for the original NumPy C API. h5py uses a Python 3.10-compatible wheel.
- timm 0.9.16 satisfies open_clip 2.30's declared requirements; the old README's
  `timm==0.4.12 --no-deps` instruction must not be run over this environment.
- PASD's VAE availability guard now calls `is_xformers_available()` rather than
  testing the function object. Its existing SDPA fallback remains intact.
- The old scalar `pyramid_gaussian(multichannel=False)` keyword is updated to
  `channel_axis=None` for scikit-image 0.24; the scalar-image behavior is unchanged.
- `requirements-legacy-cu118.txt` is an archival copy, explicitly not supported on
  Blackwell. Never combine it with the current requirements.

## Verification boundary

The smoke suite is deliberately bounded and offline: real FP32/BF16 CUDA matmul;
simple-knn compared with a brute-force nearest-neighbor result; custom Gaussian
forward/backward; actual DINO fallback and PASD SDPA forward/backward (FP32/FP16/BF16);
NumPy/OpenCV native roundtrip; Eigen sparse solve; actual Ceres depth-map optimization;
and imports of the native pipeline modules and optional PASD modules.
It must return exit 0; import success alone is not sufficient.

No full `run.sh` or research generation batch is part of installation verification.
Pretrained inference requires separately provided checkpoints and access to gated
models. The repository had no weight files at initial inspection. Required assets:

- `checkpoints/pano_lora_720*1440_v1.safetensors` (literal asterisk in the original name);
- `checkpoints/ControlNetLama.pth`, `checkpoints/sam_vit_h_4b8939.pth`,
  `checkpoints/depth_anything_v2_vitl.pth`;
- `checkpoints/Infusion/{vae,scheduler,text_encoder,tokenizer,unet}`;
- HF `black-forest-labs/FLUX.1-dev`, `black-forest-labs/FLUX.1-Fill-dev`,
  `shi-labs/oneformer_coco_swin_large`, `llava-hf/llava-1.5-7b-hf`;
- optional SR: `checkpoints/pasd/stable-diffusion-v1-5`,
  `checkpoints/pasd/checkpoint-100000` and the additional model selection in
  `pasd/run_layers_pasd.py` (including its open_clip/resnet choices).

Do not launch stage scripts merely to test imports: some load weights and run work
at module scope. The smoke script imports safe implementation modules instead and
sets HF offline mode. Missing weights mean pretrained/end-to-end inference remains
unverified, not that an installation smoke is equivalent to successful generation.

## Evidence and exact reconstruction

Installation logs, manifests and native wheel/build evidence are retained on S3063
under `/mnt/NAS/pierce/.workstation/S3063/layerpano3d/`.
After any accepted package change regenerate the evidence:

```bash
EVIDENCE=/mnt/NAS/pierce/.workstation/S3063/layerpano3d/manifests
mkdir -p "$EVIDENCE"
conda list --explicit > "$EVIDENCE/conda-explicit.txt"
conda env export > "$EVIDENCE/environment-export.yml"
python -m pip freeze --all > "$EVIDENCE/pip-freeze.txt"
python -m pip check > "$EVIDENCE/pip-check.txt"
```

The explicit file reconstructs the Conda layer with
`conda create -n layerpano3d-blackwell-rebuild --file conda-explicit.txt`.
The repository also includes `environment-blackwell-linux-64.explicit.txt` and
`requirements-blackwell-lock.txt` for the exact verified binary/package inventory.
Reinstall the pinned Torch/application requirements and rebuild native extensions
from this commit; a raw pip freeze contains machine-local wheel paths and is an
inventory, not a portable build command. Do not copy binaries from another GPU ABI.
Keep bulk logs/caches on NAS, not in Vault. See the verified report alongside this
manual for actual test commands, exit codes, package versions and remaining risks.
