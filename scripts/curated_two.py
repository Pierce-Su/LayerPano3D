#!/usr/bin/env python3
"""Two text-only curated scenes, native quality, sequential stage-major residency."""
import argparse
import fcntl
import gc
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.curated_runtime import (artifact_record, canonical, digest, event, save_json,
                                     summarize, timed, utc, verify_records)

STAGES = ['refpano', 'depth_autolayer', 'layerdata', 'sr', 'traindata', 'gs', 'render360', 'renderzigzag', 'validate']
OPS = ['refpano', 'panodepth', 'autolayering', 'layerdata', 'sr', 'traindata', 'gs', 'render360', 'renderzigzag', 'validate']
PREV = dict(zip(OPS[1:], OPS))
SOURCE_SHA = '9fb991ad7208445d59ceb7e540189345f401a06704737e633727bf1169a410b0'
PROMPTS = [
    ('photorealistic_indoor_0000', 0, 'indoor', '0000.jpg', 'In the Cozy Café, a warm ambiance envelops the bar counter and dining tables topped with inviting chairs.'),
    ('photorealistic_outdoor_0501', 501, 'outdoor', '0501.jpg', 'In the Mountain Lake Serenity scene, a wooden cabin rests peacefully by the lake, with majestic mountains towering in the background.'),
]
SETTINGS = dict(seed=16806, conditioning='text-only', refpano=dict(height=720,width=1440,steps=50,guidance=7,blend_extend=2,saved_size=[2048,1024],precision='bfloat16',cpu_offload=True,vae_tiling=True),
                layerdata=dict(height=1024,width=2048,steps=50,guidance=30,strength=0.6,max_sequence_length=512,precision='bfloat16',llava_precision='float16',lama_preliminary_size=[512,256],cpu_offload=True),
                sr=dict(full_pasd=True,upscale=2,steps=20,guidance=9,process_size=768,precision='fp16',encoder_tile=1024,decoder_tile=224,latent_tile=320,latent_overlap=8),
                traindata=dict(sr=True,views_per_layer=24,infusion_precision='float16',infusion_seed_fix='explicit 16806 replaces native time.time seed'),
                gs=dict(first_layer_iterations=3001,subsequent_layer_iterations=2001,outlier_thresh=4,precision='native float32'),
                render360=dict(width=2048,height=2048,frames=160,fps=20,elevation=0),
                renderzigzag=dict(width=1024,height=1024,fps=30,trajectory='native points/angles/n_poses; actual view count logged'),
                depth_residency='Shared across panorama+autolayering and both samples; released before FLUX Fill; reloaded once for traindata due stage lifetime.',
                quality_reason='Native full run.sh path including optional full PASD; no documented higher-quality preset found in entrypoints; no artificial guidance/step maximization.')


def code_records():
    names = subprocess.check_output(['git','ls-files'], cwd=REPO, text=True).splitlines()
    names += ['scripts/curated_two.py','scripts/curated_runtime.py']
    return {name: digest(REPO/name) for name in sorted(set(names))
            if Path(name).suffix in ('.py','.sh','.yaml','.yml','.json','.toml') and not name.startswith('checkpoints/') and (REPO/name).is_file()}


def idle_gate():
    query = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'], text=True).strip()
    if query:
        raise RuntimeError('GPU_BUSY: no coexistence authorized; compute apps: ' + query)
    snapshot = subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader'], text=True).strip()
    event('gpu_idle_gate', snapshot=snapshot)
    return snapshot


def prepare(root, metadata):
    mount_types=subprocess.check_output(['findmnt','-n','-o','FSTYPE','-T',str(root.parent)], text=True).splitlines()
    if not mount_types or mount_types[-1].strip() not in ('nfs','nfs4'):
        raise RuntimeError('Output parent is not the expected NAS NFS mount')
    if digest(metadata) != SOURCE_SHA:
        raise ValueError('Source metadata SHA differs from approved source')
    records = json.loads(Path(metadata).read_text())['samples']
    selected = []
    for sid, idx, scene_type, filename, prompt in PROMPTS:
        record = next(r for r in records if r['scene_type']==scene_type and 'photorealistic' in r)
        if (record['index'],record['photorealistic']['filename'],record['photorealistic']['prompt']) != (idx,filename,prompt):
            raise ValueError('Selection/identity/prompt mismatch')
        selected.append(dict(sample_id=sid,source_record=record,prompt=prompt,scene_type=scene_type,seed=16806))
    if __import__('shutil').disk_usage(root.parent).free < 100 * 1024**3:
        raise RuntimeError('Need at least 100 GiB free before preparing native run')
    root.mkdir(exist_ok=False)
    (root/'evidence/stages').mkdir(parents=True)
    for sample in selected:
        (root/sample['sample_id']).mkdir()
    identity = dict(task_id='t_b7a371cb',source_path=str(metadata),source_sha256=SOURCE_SHA,
                    samples=selected,settings=SETTINGS,code_sha256=code_records())
    manifest = dict(identity=identity,identity_sha256=canonical(identity),created_utc=utc(),repo=str(REPO),
                    repo_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                    output_root=str(root),launch_owner='assistant',status='CPU_PREPARED_NOT_GPU_VALIDATED')
    save_json(root/'manifest.json',manifest)
    save_json(root/'evidence/metadata-selected.json',[s['source_record'] for s in selected])
    subprocess.run(['git','diff','56efe11c919095d4006b7234b40c56ce61162c22','HEAD','--'],cwd=REPO,stdout=(root/'evidence/runtime.patch').open('w'),check=True)
    write_readme(root,manifest)
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


def check_identity(root):
    m=json.loads((root/'manifest.json').read_text())
    if canonical(m['identity']) != m['identity_sha256'] or m['identity']['settings'] != SETTINGS:
        raise ValueError('Run identity/settings changed')
    if m['identity']['code_sha256'] != code_records():
        raise ValueError('Code changed: refuse implicit resume/adoption. Explicit audited migration required.')
    if digest(m['identity']['source_path']) != m['identity']['source_sha256']:
        raise ValueError('Original metadata changed')
    return m


def check_ready(root):
    m=check_identity(root)
    ready=json.loads((root/'READY.json').read_text())
    if ready['status']!='CPU_READY_NOT_GPU_VALIDATED' or ready['launch_owner']!='assistant' or ready['identity_sha256']!=m['identity_sha256']:
        raise ValueError('READY identity/owner/status mismatch')
    for test in ready['cpu_tests']:
        if test['exit_code']!=0 or digest(root/test['log'])!=test['sha256']:
            raise ValueError('CPU readiness test evidence mismatch')
    return ready


def stamp_path(root,op,sample):
    return root/'evidence/stages'/f"{op}.{sample['sample_id']}.json"


def pending(root, op, sample, identity):
    path=stamp_path(root,op,sample)
    if path.exists():
        stamp=json.loads(path.read_text())
        if stamp['identity_sha256']!=identity or stamp['sample_id']!=sample['sample_id'] or stamp['op']!=op:
            raise ValueError('Checkpoint identity mismatch')
        verify_records(root,stamp['artifacts'])
        if op in PREV:
            upstream=stamp_path(root,PREV[op],sample)
            if not upstream.exists() or stamp['upstream_sha256']!=digest(upstream):
                raise ValueError('Checkpoint upstream identity changed')
        return False
    if op in PREV:
        upstream=stamp_path(root,PREV[op],sample)
        if not upstream.exists():
            raise ValueError(f'Missing prerequisite: {upstream}')
        pending(root,PREV[op],sample,identity)
    return True


def outputs(path,op):
    from PIL import Image
    import numpy as np
    layer=path/'layering'
    layers=sorted(layer.glob('layer[0-9]*'))
    def image(p,size=None):
        with Image.open(p) as im:
            im.load()
            if size is not None and im.size!=size:
                raise ValueError(f'Wrong dimensions {p}: {im.size}, need {size}')
    if op=='refpano':
        files=[path/'rgb.png']; image(files[0],(2048,1024))
    elif op=='panodepth':
        files=[layer/'rgb.png',layer/'depth.npy',layer/'depth.png']
        image(files[0],(2048,1024))
        d=np.load(files[1]); assert d.shape==(1024,2048) and np.isfinite(d).all() and d.max()>d.min()
    elif op=='autolayering':
        if not layers: raise ValueError('Native autolayering detected zero instances; do not invent layers')
        files=[layer/'depth_pred.png']+[p/f'{p.name}_mask.png' for p in layers]
        for p in files: image(p,(2048,1024))
    elif op=='layerdata':
        files=[p/f'{p.name}_inpaint.png' for p in layers]+[p/f'{p.name}_effective_prompt.txt' for p in layers]
        for p in files:
            if p.suffix=='.png': image(p,(2048,1024))
    elif op=='sr':
        files=[layer/'rgb_sr.png']+[p/f'{p.name}_inpaint_sr.png' for p in layers]
        for p in files: image(p,(4096,2048))
        files+=list(layer.rglob('*.pasd-prompt.json'))
    elif op=='traindata':
        folders=sorted((path/'traindata').glob('layer[0-9]*'))
        assert len(folders)==len(layers)+1
        for p in folders:
            assert len(list((p/'frames').glob('rgb_*.png')))==24
            assert len(list((p/'frames').glob('transform_matrix_*.npy')))==24
            for frame in (p/'frames').glob('rgb_*.png'): image(frame,(1024,1024))
        files=[p for p in (path/'traindata').rglob('*') if p.is_file()]
    elif op=='gs':
        from plyfile import PlyData
        files=sorted((path/'scene').glob('gsplat_layer*.ply'))
        assert len(files)==len(layers)+1
        for p in files:
            v=PlyData.read(str(p))['vertex']; assert len(v)>1000
            for name in v.data.dtype.names:
                assert np.isfinite(v[name]).all(), f'Nonfinite PLY field: {p} {name}'
    elif op.startswith('render'):
        import cv2
        rotation=op=='render360'
        names=['renders_0deg.mp4','depths_0deg.mp4'] if rotation else ['renders_zigzag.mp4','depths_zigzag.mp4']
        files=[path/'scene'/n for n in names]
        # Actual native trajectory evidence emitted from the corresponding renderer.
        root=path.parent
        events=[json.loads(l) for l in (root/'evidence/events.jsonl').read_text().splitlines()]
        trajectories=[e for e in events if e['kind']=='render_trajectory' and e['stage']==op and e['save_dir']==str(path/'scene')]
        expected=trajectories[-1]['frames']
        assert not rotation or expected==160
        for p in files:
            cap=cv2.VideoCapture(str(p)); assert cap.isOpened(), str(p)
            fps=cap.get(cv2.CAP_PROP_FPS); count=0; size=None
            while True:
                ok,frame=cap.read()
                if not ok: break
                assert frame is not None and frame.size>0
                size=(frame.shape[1],frame.shape[0]); count+=1
            cap.release()
            assert count==expected and size==((2048,2048) if rotation else (1024,1024))
            assert abs(fps-(20 if rotation else 30))<0.01
            event('video_verified',path=str(p),frames=count,fps=fps,duration_seconds=count/fps,size=size)
    elif op=='validate':
        # Previous ops already fully validated; final report records actual final artifact identity.
        for prev in ['refpano','sr','gs','render360','renderzigzag']:
            outputs(path,prev)
        files=[path/'scene'/'VALIDATION.json']
        save_json(files[0],dict(sample_id=path.name,utc=utc(),status='PASS',final_ply=str(max((path/'scene').glob('gsplat_layer*.ply'),key=lambda p:int(p.stem.split('layer')[-1])))))
    else: raise ValueError(op)
    return files


def process_samples(root,manifest,op,fn):
    for sample in manifest['identity']['samples']:
        identity=manifest['identity_sha256']
        if not pending(root,op,sample,identity):
            event('resume_skip',op=op,sample_id=sample['sample_id']); continue
        # Check input hashes immediately before processing, not only driver startup.
        upstream=stamp_path(root,PREV[op],sample) if op in PREV else None
        if upstream:
            verify_records(root,json.loads(upstream.read_text())['artifacts'])
        path=root/sample['sample_id']
        with timed('sample',op=op,sample_id=sample['sample_id']):
            fn(sample,path)
            from scripts.curated_runtime import sync_cuda
            sync_cuda()
            with timed('artifact_validation',op=op,sample_id=sample['sample_id']):
                files=outputs(path,op)
                artifacts=artifact_record(root,files)
            save_json(stamp_path(root,op,sample),dict(identity_sha256=identity,sample_id=sample['sample_id'],op=op,
                      upstream_sha256=digest(upstream) if upstream else None,artifacts=artifacts,completed_utc=utc(),exit_code=0))
        gc.collect()
        import torch
        if torch.cuda.is_initialized(): torch.cuda.empty_cache()


def run_worker(root,stage):
    os.environ['CURATED_RUN_ROOT']=str(root); os.environ['CURATED_STAGE']=stage
    os.chdir(REPO)
    m=check_identity(root)
    ops=['panodepth','autolayering'] if stage=='depth_autolayer' else [stage]
    if all(not pending(root,op,s,m['identity_sha256']) for op in ops for s in m['identity']['samples']):
        event('stage_already_complete'); return
    if stage=='validate':
        process_samples(root,m,'validate',lambda s,p:None); return
    ns=types.SimpleNamespace
    lora='checkpoints/pano_lora_720*1440_v1.safetensors'
    with timed('shared_load'):
        if stage=='refpano':
            mod=importlib.import_module('gen_refpano')
            mod.load_models(ns(lora_path=lora))
            fn=lambda s,p:mod.run_sample(ns(lora_path=lora,save_dir=str(p),seed=s['seed'],prompt=s['prompt']))
        elif stage=='depth_autolayer':
            mod=importlib.import_module('gen_autolayering')
            depthmod=importlib.import_module('gen_panodepth')
            mod.load_models(ns())
            depthmod.depthmap_utils.resident_depth_model=mod.depth_model
            depthfn=lambda s,p:depthmod.run_sample(ns(depth_model='DepthAnythingv2',save_dir=str(p),input_path=str(p/'rgb.png')))
            fn=lambda s,p:mod.run_sample(ns(input_dir=str(p),scene_type=s['scene_type'],debug=False))
        elif stage=='layerdata':
            mod=importlib.import_module('gen_layerdata'); mod.load_models(ns(lora_path=lora))
            fn=lambda s,p:mod.run_sample(ns(base_dir=str(p/'layering'),seed=s['seed'],maxsize=2048,strength=0.6))
        elif stage=='sr':
            sys.path.insert(0,str(REPO/'pasd'))
            spec=importlib.util.spec_from_file_location('curated_native_pasd',REPO/'pasd/run_layers_pasd.py')
            mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            args=mod.build_parser().parse_args(['--upscale','2','--seed','16806'])
            resident=mod.load_models(args)
            def fn(s,p):
                sample_args=mod.build_parser().parse_args(['--upscale','2','--seed',str(s['seed']),'--inputs_dir',str(p/'layering')])
                mod.main(sample_args,resident=resident)
        elif stage=='traindata':
            mod=importlib.import_module('gen_traindata')
            dmod=importlib.import_module('gen_autolayering')
            mod.depthmap_utils.resident_depth_model=dmod.load_depth_model('checkpoints/depth_anything_v2_vitl.pth')
            pipe=mod.Gen_traindata.load_depth_inpainting_model(ns(device='cuda',seed=16806))
            def fn(s,p):
                mod.seed_all(s['seed'])
                obj=mod.Gen_traindata(save_dir=str(p),sr=True,layerpano_dir=str(p/'layering'),pipe_dp=pipe,seed=s['seed'])
                obj.run()
        elif stage=='gs':
            mod=importlib.import_module('LayerPano')
            def fn(s,p):
                import random, numpy as np, torch
                random.seed(s['seed']); np.random.seed(s['seed']); torch.manual_seed(s['seed']); torch.cuda.manual_seed_all(s['seed'])
                obj=mod.LayerPano(save_dir=str(p/'scene'))
                obj.create(str(p),outlier_thresh=4)
        elif stage in ('render360','renderzigzag'):
            mod=importlib.import_module('rendering.render_video_360' if stage=='render360' else 'rendering.render_video_zigzag')
            def fn(s,p):
                args=mod.build_parser().parse_args(['--save_dir',str(p/'scene')]+(['--elevation','0'] if stage=='render360' else []))
                event('scene_ply_load',sample_id=s['sample_id'],path=str(max((p/'scene').glob('gsplat_layer*.ply'),key=lambda q:int(q.stem.split('layer')[-1]))))
                mod.run_sample(args)
        else: raise ValueError(stage)
    if stage=='depth_autolayer':
        process_samples(root,m,'panodepth',depthfn)
        process_samples(root,m,'autolayering',fn)
    else:
        process_samples(root,m,stage,fn)
    import torch
    event('allocator_peak',max_memory_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
          max_memory_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0,
          note='PyTorch allocator peak in this stage worker, not nvidia-smi sampled device usage')


def write_readme(root,m,completed=False):
    text='''# LayerPano3D：兩個 curated photorealistic 文字場景\n\n'''
    text+='狀態：'+('2/2 原生流程與檔案完整性驗證通過；視覺品質需另外檢查。' if completed else 'CPU 準備完成；尚未宣稱 GPU 或場景成功。')+'\n\n'
    text+='輸入僅為 metadata 的原始文字，未讀取 curated 原圖作 conditioning。\n'
    for s in m['identity']['samples']:
        text+=f"\n{s['sample_id']} ({s['scene_type']}，seed={s['seed']})\n原文：{s['prompt']}\n路徑：{root/s['sample_id']}\n"
    text+='\n觀看順序：各 sample/rgb.png → layering/rgb_sr.png → scene/renders_0deg.mp4 → scene/renders_zigzag.mp4。depths_* 為深度影片；最終 PLY 為 scene/gsplat_layerN.ply 中最大數字 N。\n'
    text+='\n實際入口：python -u scripts/curated_two.py --root "'+str(root)+'" --run --owner assistant\n'
    text+='\n設定、原始記錄與來源 SHA256：manifest.json、evidence/metadata-selected.json。逐 stage/sample 的上游 identity、檔案 SHA256、完成 exit_code：evidence/stages/。原生模型 load call、requested/resolved 路徑、caption/negative prompt 與耗時：evidence/events.jsonl、各 *_effective_prompt.txt、*.pasd-prompt.json。\n'
    text+='\n品質：遵循原作 run.sh 的完整預設，含 PASD SR 2×；非 light/LCM/量化/少步。refpano 720×1440 50步 guidance7，輸出2048×1024；Fill 1024×2048 50步 guidance30 strength0.6；PASD 20步 guidance9 process768，輸出4096×2048；3DGS 首層3001步，其餘2001步。有效設定詳 manifest.settings。Infusion 原生以 time.time 設 seed，這次顯式固定16806以可重現。\n'
    text+='\n模型生命週期：每 stage 子程序只載一次，依序處理兩個 sample，結束釋放。depth 在 panorama+autolayering 共用；Fill/SR 之後 traindata 因生命週期重載一次。SAM/LLaVA/LaMa/Fill 僅 layerdata 常駐；PASD/CoCa 僅 SR 常駐；不把所有大模型同時放進96G。不宣稱所有模型跨所有階段只讀一次。每場景 Gaussian/PLY 本來不同，不能共用 scene 狀態。\n'
    text+='\n耗時：timings.json/csv 分列共用載入與 sample 處理。sample 跨階段累計不是獨立端到端 wall time，shared load 不重算。首實際 load 起算至兩場景最終輸出；UTC span 包括失敗/重試/中斷，單次實際 elapsed 使用 monotonic 並在邊界 CUDA sync。準備/排隊以 preparation/launch 事件區分，不能稱 inference。失敗保留 FAILED 與事件，resume 不重置歷史。\n'
    text+='\nVRAM：evidence/gpu-sampled.csv 為 nvidia-smi 每5秒 sampled device usage（不等於精確峰值）；events 的 allocator_peak 為 PyTorch stage allocator max allocated/reserved，不包括其他CUDA配置。\n'
    text+='\n限制：CPU 測試只驗 orchestration/loader 路徑與 resume，不能代替 GPU forward；模型品質、視覺瑕疵與接縫不因檔案可讀而自動合格。保留首次 seed 結果，不暗中重抽。DONE 僅在所有 stage 有驗證 stamp 後產生。\n'
    if completed:
        report=json.loads((root/'timings.json').read_text())
        text+=f"\n本批含中斷的總 wall 秒：{report['batch_wall_seconds_including_interruptions']}；共用載入秒：{report['shared_stage_load_seconds']}；成功 sample-stage 處理秒：{report['successful_sample_stage_seconds']}。\n"
    (root/'README.zh-TW.md').write_text(text)


def run_driver(root,owner):
    if owner!='assistant': raise ValueError('Launch ownership belongs exclusively to Assistant')
    os.environ['CURATED_RUN_ROOT']=str(root)
    lock=(root/'evidence/driver.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    m=check_identity(root)
    check_ready(root)
    if (root/'DONE').exists():
        for op in OPS:
            for sample in m['identity']['samples']: assert not pending(root,op,sample,m['identity_sha256'])
        print('Already complete; verified stamps, no relaunch.'); return
    event('driver_attempt_start',owner=owner)
    monitor=None
    try:
        idle_gate()
        monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,uuid,memory.used,memory.total,utilization.gpu','--format=csv','-l','5'],stdout=(root/'evidence/gpu-sampled.csv').open('a'),stderr=subprocess.STDOUT)
        for stage in STAGES:
            # No stage overlaps another. New external compute job => fail safely, not kill.
            if stage!='validate': idle_gate()
            cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--root',str(root),'--worker',stage]
            event('stage_launch',command=cmd)
            with (root/'evidence'/f'{stage}.log').open('a') as log:
                result=subprocess.run(cmd,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
            event('stage_exit',worker_stage=stage,exit_code=result.returncode)
            if result.returncode: raise RuntimeError(f'{stage} failed: exit {result.returncode}; see evidence/{stage}.log')
        for op in OPS:
            for sample in m['identity']['samples']:
                if pending(root, op, sample, m['identity_sha256']):
                    raise RuntimeError(f'Exit 0 without verified stage stamp: {op}/{sample["sample_id"]}')
        start=json.loads((root/'evidence/BATCH_STARTED.json').read_text())
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        event('batch_done',elapsed_seconds=time.perf_counter()-start['monotonic'] if start['boot_id']==boot else None,
              note='Monotonic elapsed across processes valid only within same boot; UTC span in timings includes all interruptions.')
        summarize(root)
        m['status']='COMPLETE_FILES_VALIDATED_PENDING_VISUAL_REVIEW'
        m['completed_utc']=utc()
        save_json(root/'manifest.json',m)
        write_readme(root,m,True)
        save_json(root/'DONE',dict(utc=utc(),identity_sha256=m['identity_sha256'],samples=2))
    except BaseException as exc:
        event('driver_failed',error=str(exc))
        save_json(root/'FAILED',dict(utc=utc(),error=str(exc)))
        raise
    finally:
        if monitor is not None:
            monitor.terminate(); monitor.wait(timeout=20)
        if (root/'evidence/events.jsonl').exists(): summarize(root)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',action='store_true'); group.add_argument('--check',action='store_true')
    group.add_argument('--check-ready',action='store_true'); group.add_argument('--gate',action='store_true')
    group.add_argument('--run',action='store_true'); group.add_argument('--worker',choices=STAGES)
    parser.add_argument('--metadata',type=Path,default=Path('/mnt/NAS/E3DQA_project/curated_set/metadata.json'))
    parser.add_argument('--owner',choices=['assistant'])
    args=parser.parse_args(); root=args.root.resolve()
    if args.prepare: prepare(root,args.metadata)
    elif args.check:
        print(json.dumps(check_identity(root),ensure_ascii=False,indent=2))
    elif args.check_ready:
        print(json.dumps(check_ready(root),ensure_ascii=False,indent=2))
    elif args.gate:
        check_ready(root)
        idle_gate()
    elif args.worker: run_worker(root,args.worker)
    else: run_driver(root,args.owner)

if __name__=='__main__': main()
