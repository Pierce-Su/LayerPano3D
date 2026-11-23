# Curated two-scene resident runner (S3063)

This is a native-quality execution adapter, not a replacement research pipeline.
Only the two approved literal text prompts in `scripts/curated_two.py` are used.
The original curated images are never opened. The source metadata SHA256 and both
selected records are checked again at prepare and resume.

## Lifetime and quality

The CLI entrypoints now expose import-safe `load_models` / `run_sample` functions.
Their native algorithm statements, dimensions, guidance and steps are preserved.
Each stage runs in one subprocess and handles both captions sequentially:

1. FLUX.1-dev reference panoramas (50 steps, 720x1440, guidance 7).
2. Panorama depth for both samples, then native autolayering for both samples.
   One DepthAnything vitl network is injected into the original depth utility;
   sample-specific estimators, projection buffers, paths and masks are NOT shared.
   The same network also serves autolayering, with OneFormer resident once.
3. Native SAM + LLaVA + FLUX.1-Fill-dev + LaMa layer generation.
   Fill uses native 1024x2048 / 50 steps / guidance 30 / strength 0.6.
4. Full PASD + CoCa once, both samples; native 20 steps / guidance 9 /
   process_size 768 / upscale 2 / native VAE and latent tiles. No light or LCM.
5. Depth network reloads once because stages 2 and 5 have separate lifetimes.
   Infusion components load once and are injected into fresh sample-local
   `Gen_traindata` objects. Native `--sr` and 24 perspective views per layer.
6. Original 3DGS construction (first layer 3001, remaining layers 2001 iterations).
7. Native 360 render at elevation 0 (2048², 160 frames, 20 fps), then native zigzag
   render (1024², 30 fps; actual trajectory length recorded at runtime).
8. Full artifact/hash, PLY finite-field and video full-decode verification.

Reference/Fill use the original bfloat16 and CPU offload; PASD/Infusion use native
float16. No dependency downgrade, driver change, xformers reinstall, quantization,
reduced resolution or shortened denoising. No alternative higher-quality preset
was found in the native run.sh/entrypoints/README. Larger guidance is not a quality
preset. Infusion's original time.time seed is explicitly replaced by 16806 for
both samples and recorded as a reproducibility change, not a native default.

Model-load events wrap actual `from_pretrained`, weight/LoRA constructors and local
checkpoint calls. The requested and resolved checkpoint/cache paths are logged.
Counts distinguish processor/tokenizer, model constructors and file loads; they
are not all synonyms for a complete neural network. Scene-specific PLY loads
cannot be shared between distinct scenes and are logged separately. Failed
attempts/reloads are preserved, not hidden behind an assertion of globally-once.

## Runtime compatibility evidence (2026-11-23)

Both shell layers disable nounset only during Conda activation, then restore it.
Entry PID/log/EXIT trap are installed before activation. Native FLUX LoRA loading
passes the exact local basename as `weight_name`: installed Diffusers offline
mode otherwise rejects even an existing local file. Both adapters retain the
same checkpoint bytes and quality settings. The CPU test asserts this argument.

OneFormer also requires the small dataset metadata file, beyond its model cache:
`shi-labs/oneformer_demo`, dataset revision
`4d683bd5bf84e9c8b5537dce306230bde409fe89`, `coco_panoptic.json` (5849 bytes),
SHA256 `2aab920e13f27c2846ad0a66568e8533fe049f4f0523d174ea8e944fa9ad84ae`.
A CPU-only online `AutoProcessor.from_pretrained` populated this native dependency;
then the same call passed with both offline flags set and CUDA uninitialized.
The run records the exact NAS cache path/hash in
`evidence/oneformer-dataset-metadata.json`. No model substitution was made.

Failed attempts remain in append-only logs/events and archived FAILED/EXIT
markers. Runtime code migrations are explicit audit records; the initial
BATCH_STARTED clock is retained. Verified reference panoramas are resumed, not
regenerated after a missing metadata dependency. Do not infer visual quality or
complete 3D success from a successful reference panorama.

## CPU preparation and tests (no GPU permission needed)

From the activated `layerpano3d-blackwell` environment in the repository:

    PYTHONDONTWRITEBYTECODE=1 python tests/test_curated_two.py
    CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python tests/import_curated_native_cpu.py

The orchestration tests execute actual native function bodies against explicit
CPU doubles: two distinct captions, load-once, sample-local paths, shared pure
depth network, injected PASD/Infusion, fail-stop, input/output/upstream hash
rejection, resume, CUDA timing boundaries and shared-load accounting. These are
not generated scenes and do not prove GPU forward or image quality.
The separate native import test imports every full real adapter with visible GPUs
removed and model loading forbidden; no CUDA context may be initialized.

Create a NEW, nonexisting run root below the approved NAS assets directory:

    python scripts/curated_two.py --root /mnt/NAS/E3DQA_project/assets/LayerPano3D/RUN_ID --prepare
    python scripts/prepare_curated_ready.py --root /mnt/NAS/E3DQA_project/assets/LayerPano3D/RUN_ID

The second command publishes READY.json only after both real CPU test commands
exit 0 and the identity is rechecked. READY contains exact launch/resume argv,
identity digest, test log hashes, output root, lock, tmux, PID and log paths.
The preparation command never launches inference or creates a monitoring cron.

## Coordinated GPU launch ownership

PM transferred the sole runtime repair/launch ownership to Developer at
2026-11-23 22:48 +08:00 after an early shell activation failure. Assistant must
not compete for launch; its existing monitor must check completion/failure and
trigger Developer acceptance. The legacy CLI `--owner assistant` and original
READY `launch_owner` are compatibility tokens, not a second execution owner;
`actual_execution_owner` and the explicit migration evidence identify Developer.
The current run is `/mnt/NAS/E3DQA_project/assets/LayerPano3D/curated_two_t_b7a371cb_final`.
Launch/resume remains:

    bash scripts/launch_curated_two.sh /mnt/NAS/E3DQA_project/assets/LayerPano3D/RUN_ID

This activates the existing environment/cache configuration, checks code/metadata/
settings/test evidence, obtains a launch flock and refuses an existing dedicated
tmux session. It refuses ANY compute PID rather than guessing from free VRAM or
instantaneous utilization. The driver also rechecks idleness immediately before
stages. No other user's data is read and no other job is stopped.

The dedicated session is `layerpano3d-curated-two`. Read back
`evidence/driver.PID`, `evidence/driver.log`, actual `tmux list-panes` and the first
stage log/events before saying a launch happened. The durable driver runs every
stage automatically, halts on nonzero exit/invalid upstream artifacts and writes
FAILED plus history. `tmux-entry.EXIT` is the actual shell/driver exit status.
No short polling loop is needed; Assistant owns the existing monitor and the
completion trigger back to developer task t_b7a371cb.

## Resume, accounting and acceptance

Same command, same root and identity resumes verified sample/stage checkpoints;
no model loads occur for an already-complete stage. Code, exact prompts, quality
settings, upstream stamp identity and artifact SHA256 must match. Changed code
requires an explicit audited manifest migration, NOT silent adoption of old
outputs. The code deliberately has no automatic force/ignore-identity option.

The 360 video filenames are native `renders_0deg.mp4` / `depths_0deg.mp4`, not
invented `renders_360.mp4` names. Final PLY is the numerically last actual
`scene/gsplat_layerN.ply`. Videos are fully decoded, checking trajectory frame
count, dimensions, fps and duration. All PLY scalar fields must be finite and
have more than 1000 vertices. Success requires every stage stamp for both samples;
exit 0 alone cannot produce DONE. File integrity is not aesthetic approval.
Developer still inspects real final results and requests PM review only afterwards.

`events.jsonl`, `timings.json/csv` preserve failures/retries and interruptions.
Batch wall time begins at the first actual model-loader invocation, ends after
both final scenes/renders are verified, and includes interruptions via the UTC
span. Within one boot the monotonic span is also retained. Shared stage load time
(including imports) and sample processing (including artifact validation) are
separate; nested loader time is not added twice, and sample sums are not falsely
called independent end-to-end wall times. CPU readiness preparation is recorded
separately. The READY-to-launch waiting interval must not be called inference.
`gpu-sampled.csv` is five-second device sampling; allocator max allocated/reserved
is a separate per-worker metric, not an exact total device peak.

All bulk outputs/evidence remain on NAS. Do not copy them to the synced Vault.
Run-root README.zh-TW.md is generated from actual identity and, only upon verified
completion, actual timing summaries. Preserve first-seed visual failures; do not
quietly resample until attractive.
