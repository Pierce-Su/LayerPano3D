# LayerPano3D: Layered 3D Panorama for Hyper-Immersive Scene Generation(SIGGRAPH 2025)
This repository is the official implementation of LayerPano3D



[**Project page**](https://ys-imtech.github.io/projects/LayerPano3D/) | [**Paper**](https://arxiv.org/abs/2408.13252) | [**Dataset**](https://huggingface.co/datasets/ysmikey/Layerpano3D_PanoData) | [**Huggingface**](https://huggingface.co/ysmikey/Layerpano3D-FLUX-Panorama-LoRA)

![](assets/teaser.jpeg)

</br>

[Shuai Yang*](https://ys-imtech.github.io/), 
[Jing Tan*](https://sparkstj.github.io/), 
[Mengchen Zhang](https://github.com/kszpxxzmc/), 
[Tong Wu✉️](https://wutong16.github.io/), 
[Yixuan Li](https://yixuanli98.github.io/), 
[Gordon Wetzstein](https://stanford.edu/~gordonwz/), 
[Ziwei Liu](https://liuziwei7.github.io/), 
[Dahua Lin✉️](http://dahua.me/)

<p style="font-size: 0.6em; margin-top: -1em">*Equal Contribution,   ✉️Corresponding author</p>

Accepted by SIGGRAPH 2025


<p align="center">
<a href="https://arxiv.org/abs/2408.13252"><img src="https://img.shields.io/badge/arXiv-Paper-<color>"></a>
<a href="https://ys-imtech.github.io/projects/LayerPano3D/"><img src="https://img.shields.io/badge/Project-Website-red"></a>
<a href="https://huggingface.co/datasets/ysmikey/Layerpano3D_PanoData"><img src="https://img.shields.io/badge/Dataset-Huggingface-yellow"></a>
<a href="https://www.youtube.com/watch?v=dXvoFRbHOiw"><img src="https://img.shields.io/static/v1?label=Demo&message=Video&color=orange"></a>
<a href="" target='_blank'>
<img src="https://visitor-badge.laobi.icu/badge?page_id=YS-IMTech.LayerPano3D" />
</a>
</p>



![](assets/layerpano3d.png)

### ✨ Updates:
[2025-07-23] 🔥 Release our 360 Panorama dataset Upright360, please check and download it on [**Huggingface Dataset**](https://huggingface.co/datasets/ysmikey/Layerpano3D_PanoData).

### 🔧 Installation (Blackwell / S3063)

For RTX PRO 6000 Blackwell (SM 12.0), use the maintained
[Blackwell installation guide](docs/INSTALL-BLACKWELL.md).
The previous Python 3.9 / Torch 2.4 CUDA 11.8 recipe is **not** compatible with this GPU.

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

No sudo, driver changes, shared-base installs, or prebuilt CUDA 11.8 extensions.
The native build retains the project's custom four-output Gaussian rasterizer and
360monodepth Ceres implementation; all compiler/dependency paths come from the
activated environment. See the guide for exact environment prefix, source build,
lock inventories, tests, evidence and missing model assets. A passing environment
smoke is not an end-to-end pretrained-generation result. The historical dependency
list is archived in `requirements-legacy-cu118.txt`, not used by current installs.

- **Checkpoints Download**: 
  - Download the Panorama LoRA [checkpoint](https://huggingface.co/ysmikey/Layerpano3D-FLUX-Panorama-LoRA/resolve/main/lora_hubs/pano_lora_720*1440_v1.safetensors?download=true) and put it in the ``checkpoints`` folder. We will release more powerful and better panorama_lora_v2.safetensors() in the future, now version-1 is avaliable in [huggingface](https://huggingface.co/ysmikey/Layerpano3D-FLUX-Panorama-LoRA).
  - Download the Lama [checkpoint](https://huggingface.co/lllyasviel/Annotators/resolve/main/ControlNetLama.pth?download=true) and put it in the ``checkpoints`` folder.
  - Download the SAM model ViT-H [checkpoints](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) and put it in the ``checkpoints`` folder.
  - Download the Depth-Anything-V2-Large [checkpoints](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true) and put it in the ``checkpoints`` folder.
  - Download the Infusion [checkpoint](https://huggingface.co/Johanan0528/Infusion/tree/main) and put it in the ``checkpoints/Infusion`` folder.
  - [**Optional**] Download the PASD related checkpoints. **If you do not need SR panorama for better visual quality, this step can be skip**.
    - Download SD1.5 models from [huggingface](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) and put them into ``checkpoints/pasd/stable-diffusion-v1-5``. 
    - Download PASD pre-trained models [pasd](https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/PASD/pasd.zip) and put them into ``checkpoints/pasd/checkpoint-100000``.

#### S3063 模型與快取準備（核心與原生 PASD SR）

在 S3063，FLUX.1-dev 使用 DiT360 已有的 Hugging Face 共用快取；不要複製到 `checkpoints`，也不要另開 dev 下載。FLUX.1-Fill-dev、OneFormer 與 LLaVA 亦由共用快取讀取。`HF_HOME=/mnt/NAS/pierce/.workstation/S3063/cache/huggingface`，`HF_HUB_CACHE=$HF_HOME/hub`。已確認離線快取可解析至固定版本：dev `45d95c497444bbaebe766e1528792fc9158c0c3a`（DiT360 管理；23 個必要檔，33,746,383,552 B；另有一個快取 LICENSE.md）、Fill `7326a96224b2513681b5d97281c83af967926bb6`（23 檔）、OneFormer `3a263017ca5c75adbea145f25f81b118243d4394`（10 檔）、LLaVA `b234b804b114d9e37bb655e11cbbb5f5e971b7a9`（17 檔）。dev 使用 DiT360 既有快取，沒有重複下載；LLaVA 在原快取上續傳中斷的 shard，其餘 Fill、OneFormer 與核心本地權重為本次取得。離線解析不代表已載入模型或完成生成。

全景 LoRA 檔名包含字面上的 `*`：`pano_lora_720*1440_v1.safetensors`。`checkpoints/` 根目錄路徑是指向 `checkpoints/lora_hubs/` 的符號連結，以免重複儲存。前次核心下載僅涵蓋非 SR；現已補齊原始 run.sh 所需 SD1.5、full PASD UNet/ControlNet 與 CoCa（21 檔實讀 hash 通過），原生 PASD/CoCa CPU 載入與短 caption decode 通過。MajicMix／LCM／ResNet 是未啟用條件分支，不是此預設路徑缺項。GPU SR fixture 因其他工作占用而安全拒絕，尚未驗證；完整 `run.sh` 仍未執行。精確權重版本、授權、快取 symlink、重建／測試命令及驗證邊界見 [S3063 原生 PASD SR 手冊](docs/PASD-SR-S3063.md)，不要只因下載完成就啟動完整生成。FLUX gated 模型條款仍適用；可下載不代表後續用途已獲授權。
### 🌌 Inference:

Quick start: one-step running bash. **From a single text prompt to generates full-view, explorable panoramic 3D scene**.
```sh
huggingface-cli login
bash run.sh
```
If you want to try or debug each stage in our pipeline, we also provide a complete [running results](https://drive.google.com/drive/folders/1op5qimyAcr1-k7-8lgOwg-sWa4_qiG1T?usp=drive_link) (including each step).

### 🌆 Rendering:
In the generated 3DGS panoramic scene, the resulting folder contains files formatted as ``gsplat_layer{id}.ply``, where ``id`` represents the index of the layer. The number of layers varies across different cases. Notably, the 3dgs file corresponding to the maximum id value represents the final 3D scene result.

```sh
bash render.sh
```

### 🌉 Visualize `.ply` files


We recommend use [@playcanvas](https://github.com/playcanvas)'s [Super-Splat](https://github.com/playcanvas/super-splat) project ([Live demo](https://playcanvas.com/super-splat)). 
![](assets/supersplat.png)



## 📧 Contact Us
Shuai Yang: [yang_shuai@sjtu.edu.cn](mailto:yang_shuai@sjtu.edu.cn)  
Jing Tan: [tj023@ie.cuhk.edu.hk](mailto:tj023@ie.cuhk.edu.hk)  
Tong Wu: [wutong16@stanford.edu](mailto:wutong16@stanford.edu)  

## ✒️ Citation
If you find our work helpful for your research, please consider giving a star ⭐ and citation 📝

```bibtex
@article{yang2024layerpano3d,
  title={LayerPano3D: Layered 3D Panorama for Hyper-Immersive Scene Generation},
  author={Yang, Shuai and Tan, Jing and Zhang, Mengchen and Wu, Tong and Li, Yixuan and Wetzstein, Gordon and Liu, Ziwei and Lin, Dahua},
  journal={arXiv preprint arXiv:2408.13252},
  year={2024}
}
```

