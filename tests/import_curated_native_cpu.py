#!/usr/bin/env python3
"""Import every real native adapter on CPU; any model load during import is an error."""
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from unittest import mock

os.environ['CUDA_VISIBLE_DEVICES']=''
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts import curated_runtime as audit
import torch
assert not torch.cuda.is_initialized()
result={}
with mock.patch.object(audit,'trace_load',side_effect=AssertionError('unexpected import-time model load')):
    for name in ['gen_refpano','gen_panodepth','gen_autolayering','gen_layerdata','gen_traindata','LayerPano','rendering.render_video_360','rendering.render_video_zigzag']:
        module=importlib.import_module(name)
        result[name]='IMPORTED_WITH_ZERO_MODEL_LOADS'
    sys.path.insert(0,str(REPO/'pasd'))
    spec=importlib.util.spec_from_file_location('curated_import_test_pasd',REPO/'pasd/run_layers_pasd.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    args=module.build_parser().parse_args(['--upscale','2','--seed','16806'])
    assert (args.process_size,args.num_inference_steps,args.guidance_scale,args.mixed_precision)==(768,20,9.0,'fp16')
    assert not (args.use_pasd_light or args.use_lcm_lora or args.use_personalized_model)
    result['pasd/run_layers_pasd.py']='IMPORTED_WITH_ZERO_MODEL_LOADS'
assert not torch.cuda.is_initialized()
from utils.loss import nearMean_map
for height,width in [(3,5),(7,4)]:
    source=torch.arange(height*width,dtype=torch.float32).reshape(height,width)
    mask=(source.remainder(3)!=0).float()
    kernel=torch.tensor([[0.,1.,0.],[1.,0.,1.],[0.,1.,0.]]).reshape(1,1,3,3)
    expected=torch.nn.functional.conv2d((source*mask)[None,None],kernel,padding=1)
    counts=torch.nn.functional.conv2d(mask[None,None],kernel,padding=1)
    expected=(expected/(counts+1e-8)).squeeze()
    assert torch.equal(nearMean_map(source,mask),expected)
result['lazy_fixed_kernel']='CPU_NUMERIC_PARITY_PASS'
result.update(torch=torch.__version__,cuda_context_initialized=False,status='PASS')
print(json.dumps(result,indent=2))
