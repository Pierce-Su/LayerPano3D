# S3063 原生 PASD SR 準備與驗證

本頁延續 Blackwell 安裝／核心模型交付，準備原始 `run.sh` 的 PASD SR 路徑，不替換模型，不使用 PASD-light、SDXL 或其他 SR 方法。原始安裝驗證當時沒有權重；那是歷史邊界，不是目前的缺項清單。完整場景 `run.sh` 未執行。

## 實際載入路徑

`run.sh` 呼叫 `python pasd/run_layers_pasd.py --inputs_dir outputs/layering --upscale 2`，未傳任何 `--use_*` 選項。

| 資產 | 原生預設是否需要 | 原因 |
|---|---|---|
| SD1.5 scheduler / tokenizer / text_encoder / VAE / feature_extractor | 是 | `load_pasd_pipeline()` 逐一載入 |
| PASD `checkpoint-100000/{unet,controlnet}` | 是 | 原生 full PASD 類別及權重，非 base SD UNet |
| CoCa `coca_ViT-L-14` / `mscoco_finetuned_laion2B-s13B-b90k` | 是 | `high_level_info=caption` 且 `use_blip=False` |
| MajicMix realistic v7 | 否 | 僅 `--use_personalized_model` 啟用；字串 default 不代表啟用 |
| LCM-LoRA SD1.5 | 否 | 僅 `--use_lcm_lora` 啟用 |
| ResNet50 | 否 | 僅 `high_level_info=classification` |
| BLIP / YOLO / grayscale / PASD-light | 否 | 非原始 run.sh 的分支 |
| SD1.5 base UNet / safety_checker | 否 | UNet 使用 PASD，原生 pipeline 明確 `safety_checker=None` |

SD1.5 snapshot 因此是原生 PASD 所需的選取子集，不是可供任意 `StableDiffusionPipeline.from_pretrained()` 的完整 SD1.5。額外小檔 model_index / UNet config 保留作版本索引，未下載重複且不使用的 base UNet。

## 來源、固定版本與授權

- SD1.5：`stable-diffusion-v1-5/stable-diffusion-v1-5`，revision `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`，模型卡 CreativeML Open RAIL-M；用途限制仍適用。
- PASD：作者 `yangtao9009/PASD`，revision `9c7c980143c6702c887b4a15d1675b0dd2a8acfd`，模型卡 Apache-2.0。只取 `pasd/checkpoint-100000` 的 UNet、ControlNet config + safetensors，以及 README；不取訓練 scaler 或 light/color 分支。
- CoCa：`laion/mscoco_finetuned_CoCa-ViT-L-14-laion2B-s13B-b90k`，revision `f895105fb18cf5ea3ba7a1172966e4cc0ad8d743`，模型卡 MIT。以安裝的 open_clip `get_pretrained_cfg()` 查出此作者發布的 HF repo，而非猜測名稱；來源模型卡、選取檔案及內容雜湊都在 NAS `manifest.json`。原生 loader 未傳 revision，因此另用官方 Hub 讀取小型 config 建立 main ref，並斷言其 snapshot 與 pin 一致；不手工偽造 refs。
- 原 PASD [官方 upstream README](https://github.com/yangxy/PASD) 同時指向上述作者 HF 及 [OSS pasd.zip](https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/PASD/pasd.zip)。已先讀 HEAD 與 ZIP64 central directory：8 members，必要 UNet/ControlNet 四檔名稱／解壓長度與作者 HF 相符；路徑無 absolute / `..` / backslash。採作者 HF 的同一 full PASD 發布，不再下載 5.45 GB ZIP 與重複解壓副本。這是來源及格式對照，不宣稱已比對 ZIP payload 與 HF 的逐位元 hash。
- 不需要 MajicMix，故未下載未知鏡像或代替模型，亦未接受任何新 gated 條款。若未來啟用個人化／LCM／分類分支，需另外準備並查核其確切來源和授權；本頁不宣稱那些分支就緒。

## 快取與可重建準備

沿用 `layerpano3d-blackwell`、torch 2.7.1+cu128 / CUDA12.8 及原生 SDPA；禁止裝回已在 SM120 FP16 crash 的 xformers，不降版 Torch，不變更 driver/shared base。

HF 共用 cache：`/mnt/NAS/pierce/.workstation/S3063/cache/huggingface/hub`。新增的 project links 只在目標不存在時建立：

- `checkpoints/pasd/stable-diffusion-v1-5` → SD1.5 exact snapshot。
- `checkpoints/pasd/checkpoint-100000` → PASD exact snapshot 內的 `pasd/checkpoint-100000`。

既有非相同目標會明確失敗，不覆寫；不複製 FLUX 或其他核心權重。HF 使用 library 原有認證及 locking/resume，未讀出 token。下載每檔最多三次重試，內容實讀 SHA256（LFS）或 Git-blob SHA1（小檔）對來源 metadata，另保存所有檔案 SHA256、size、expected/actual。下載完成 marker 不代表推論通過。

本次 evidence root：`/mnt/NAS/pierce/.workstation/S3063/layerpano3d-sr-t_23325b27`。專用 tmux：`layerpano3d-sr-t_23325b27`、`layerpano3d-sr-validate-t_23325b27`。可追 exact 執行版本 `prepare_sr.py`、`download_sr.sh`、`test_sr.py`、`validate_sr.sh`，以及 `manifest.json`、`download.log`、`download.EXIT`、`DOWNLOAD_DONE`、`source-audit.json`、`pasd-zip-{listing,head}.json`。

```bash
source /opt/miniforge/etc/profile.d/conda.sh || exit
conda activate layerpano3d-blackwell || exit
cd /mnt/NAS/pierce/.workstation/S3063/layerpano3d-sr-t_23325b27 || exit
# 已完成資產可安全以同 pin 重跑；不加 force，也不碰其他專案的 jobs。
bash download_sr.sh
# CPU：原生 loader、text encoder/tokenizer、CoCa 權重和短 caption decode
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 timeout 1200 python -u test_sr.py cpu
# GPU：先確認無任何其他 compute process，script 也會檢查；忙碌直接拒絕，不 kill jobs。
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 timeout 600 python -u test_sr.py gpu
```

## 驗證邊界

下載及實讀內容 hash：21/21 檔、3 repos、9,258,971,608 bytes 全通過；`download.EXIT=0`、`DOWNLOAD_DONE` 存在。`manifest.json` 每筆保存 size、SHA256 與來源 expected/actual digest；小檔另以 Git-blob SHA1 比對官方 metadata。各模型沒有 sharded index，必要單檔權重均已完整取得，不需轉檔或改副檔名。

CPU：`cpu-test.EXIT=0`，`cpu-result.json` 最後為 PASS。原生 `load_pasd_pipeline()` 實載 PASD UNet（1,107,755,524 parameters）、ControlNet（361,289,497 parameters）、SD1.5 VAE／scheduler／feature extractor／tokenizer／text encoder；token shape `[1,8]`、finite embedding shape `[1,8,768]`。原生 CoCa loader 實載權重、處理 64×64 色塊並完成 8-token 短 caption decode。此 fixture 是機制測試，不是 caption 品質評估。只有既有 diffusers LoRA mixin deprecation warning，未改套件。

GPU：未驗證。先前 `gpu-test.EXIT=1` 因 busy guard 拒絕 PID 309310；續接時只讀查詢曾見空閒，但實際 fixture 前重新檢查發現其他使用者 CCNet PID 423290，`gpu-retry-test.EXIT=1` 再次安全拒絕，未載入 GPU 模型、未執行 denoise、未干擾或終止該工作。沒有 `gpu-result.json`，沒有成功 SR 輸出。待獨佔時段確認後可用上列命令重試；保留兩次拒絕 log，不把它誤稱為模型錯誤或測試成功。

`test_sr.py gpu` 已準備單張極小圖、兩步 PASD denoise、固定 seed、原生 pretrained UNet/ControlNet/VAE/文字編碼與 SDPA 的有界 fixture，但目前僅是待執行測試。此卡完成資產與 CPU 整合準備，不宣稱 GPU SR、20-step 品質、原生 768 process_size、大型 ERP tiled latent、整個 layered SR CLI 或 text-to-3D 場景驗收；沒有啟動完整 `run.sh`。未來圖片只保留 NAS，不放同步 Vault。
