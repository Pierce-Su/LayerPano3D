"""CPU orchestration contract tests; fakes are test doubles, never scene results."""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts import curated_two as runner
from scripts import curated_runtime as audit


def functions_only(name,names,env):
    """Execute exact production function AST with explicit CPU test dependencies."""
    tree=ast.parse((REPO/name).read_text())
    body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    if len(body)!=len(names): raise ValueError('Test function selection incomplete')
    exec(compile(ast.Module(body=body,type_ignores=[]),str(REPO/name),'exec'),env)
    return types.SimpleNamespace(**{n:env[n] for n in names})


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        (self.root/'evidence/stages').mkdir(parents=True)
        self.env=mock.patch.dict(os.environ,{'CURATED_RUN_ROOT':str(self.root),'CURATED_STAGE':'test'})
        self.env.start()
        self.quiet=contextlib.redirect_stdout(io.StringIO()); self.quiet.__enter__()
        self.samples=[dict(sample_id=sid,prompt=prompt,scene_type=scene,seed=16806) for sid,idx,scene,filename,prompt in runner.PROMPTS]
        for s in self.samples: (self.root/s['sample_id']).mkdir()
        self.m=dict(identity=dict(samples=self.samples),identity_sha256='test-identity')
        self.torch=types.SimpleNamespace(cuda=types.SimpleNamespace(is_initialized=lambda:False,empty_cache=lambda:None))

    def tearDown(self):
        self.quiet.__exit__(None,None,None); self.env.stop(); self.temp.cleanup()

    def fixture(self,s,p):
        (p/'fixture.txt').write_text(s['prompt'])

    def fixture_outputs(self,p,op): return [p/'fixture.txt']

    def process(self,op,fn=None):
        with mock.patch.dict(sys.modules,{'torch':self.torch}), mock.patch.object(runner,'outputs',side_effect=self.fixture_outputs):
            runner.process_samples(self.root,self.m,op,fn or self.fixture)

    def test_actual_refpano_functions_load_once_two_distinct_prompts(self):
        calls=[]; loads=[]
        class Image:
            def __init__(self,prompt): self.prompt=prompt
            def resize(self,size): self.size=size; return self
            def save(self,path): Path(path).write_text(json.dumps(dict(prompt=self.prompt,size=self.size)))
        class Pipe:
            @classmethod
            def from_pretrained(cls,path,**kw): loads.append(path); return cls()
            def to(self,device): return self
            def load_lora_weights(self,path,**kw):
                assert kw['weight_name']==os.path.basename(path)
                loads.append(path)
            def enable_model_cpu_offload(self): pass
            def enable_vae_tiling(self): pass
            def __call__(self,prompt,**kw): calls.append((prompt,kw)); return types.SimpleNamespace(images=[Image(prompt)])
        class Generator:
            def __init__(self,device): pass
            def manual_seed(self,seed): self.seed=seed; return self
        fake_torch=types.SimpleNamespace(bfloat16='bf16',Generator=Generator)
        env=dict(torch=fake_torch,os=os,FluxPipeline=Pipe,trace_load=audit.trace_load)
        mod=functions_only('gen_refpano.py',['load_models','run_sample'],env)
        mod.load_models(types.SimpleNamespace(lora_path='native-lora'))
        for s in self.samples:
            mod.run_sample(types.SimpleNamespace(save_dir=str(self.root/s['sample_id']),prompt=s['prompt'],seed=s['seed'],lora_path='native-lora'))
        self.assertEqual(loads,['black-forest-labs/FLUX.1-dev','native-lora'])
        self.assertEqual([p for p,kw in calls],[s['prompt'] for s in self.samples])
        for _,kw in calls:
            self.assertEqual((kw['height'],kw['width'],kw['num_inference_steps'],kw['guidance_scale']),(720,1440,50,7))
            self.assertEqual(kw['generator'].seed,16806)
        for s in self.samples:
            self.assertEqual(json.loads((self.root/s['sample_id']/'rgb.png').read_text())['prompt'],s['prompt'])
        self.assertEqual(sum(e['kind']=='model_load_done' for e in self.events()),2)

    def test_real_stage_worker_load_once_outside_two_sample_loop(self):
        loads=[]; calls=[]
        fake=types.SimpleNamespace(load_models=lambda a:loads.append(a.lora_path),run_sample=lambda a:calls.append((a.prompt,a.save_dir)))
        self.torch.cuda.max_memory_allocated=lambda:0
        with mock.patch.object(runner,'check_identity',return_value=self.m), mock.patch.object(runner.importlib,'import_module',return_value=fake),mock.patch.object(runner,'process_samples',side_effect=lambda root,m,op,fn:[fn(s,root/s['sample_id']) for s in self.samples]),mock.patch.dict(sys.modules,{'torch':self.torch}):
            runner.run_worker(self.root,'refpano')
        self.assertEqual(len(loads),1)
        self.assertEqual([c[0] for c in calls],[s['prompt'] for s in self.samples])
        self.assertNotEqual(calls[0][1],calls[1][1])

    def test_depth_stage_network_shared_but_operations_stage_major(self):
        self.process('refpano')
        loads=[]; calls=[]; shared=object()
        fake=types.SimpleNamespace(depth_model=shared,load_models=lambda a:loads.append('depth+oneformer'),run_sample=lambda a:calls.append(('auto',a.scene_type)))
        depth=types.SimpleNamespace(depthmap_utils=types.SimpleNamespace(),run_sample=lambda a:calls.append(('depth',a.save_dir)))
        def importer(name): return fake if name=='gen_autolayering' else depth
        with mock.patch.object(runner,'check_identity',return_value=self.m),mock.patch.object(runner.importlib,'import_module',side_effect=importer),mock.patch.object(runner,'process_samples',side_effect=lambda root,m,op,fn:[fn(s,root/s['sample_id']) for s in self.samples]),mock.patch.dict(sys.modules,{'torch':self.torch}):
            runner.run_worker(self.root,'depth_autolayer')
        self.assertEqual(loads,['depth+oneformer'])
        self.assertIs(depth.depthmap_utils.resident_depth_model,shared)
        self.assertEqual([c[0] for c in calls],['depth','depth','auto','auto'])
        self.assertEqual([c[1] for c in calls[2:]],['indoor','outdoor'])

    def test_infusion_real_constructor_never_reloads_injected_pipe(self):
        native=functions_only('gen_traindata.py',['Gen_traindata'],dict(os=os,time=time))
        cls=native.Gen_traindata; shared=object()
        with mock.patch.object(cls,'load_depth_inpainting_model',side_effect=AssertionError('must not reload')),mock.patch.object(cls,'count_layers',return_value=1):
            objs=[cls(save_dir=str(self.root/s['sample_id']),layerpano_dir='layering',pipe_dp=shared,seed=16806) for s in self.samples]
        self.assertIs(objs[0].pipe_dp,objs[1].pipe_dp)
        self.assertIsNot(objs[0].layerdir_list,objs[1].layerdir_list)
        self.assertNotEqual(objs[0].save_dir,objs[1].save_dir)
        self.assertEqual([o.seed for o in objs],[16806,16806])

    def test_hidden_depth_function_reuses_network_for_both_samples_and_erp(self):
        calls=[]
        class Array:
            def astype(self,dtype): return self
        class Network:
            def infer_image(self,image): calls.append(image); return 'CPU-depth-double'
        env=dict(resident_depth_model=Network(),np=types.SimpleNamespace(uint8='uint8'),
                 torch=self.torch,gc=__import__('gc'),trace_load=mock.Mock(side_effect=AssertionError('unexpected weight reload')),
                 DepthAnythingV2=mock.Mock(side_effect=AssertionError('unexpected constructor')))
        native=functions_only('submodules/360monodepth/code/python/src/utility/depthmap_utils.py',['Depth_Anything_prediction'],env)
        for sample in self.samples:
            for projection in ['perspectives','erp']:
                self.assertEqual(native.Depth_Anything_prediction([Array()]),['CPU-depth-double'])
        self.assertEqual(len(calls),4)
        self.assertEqual(env['trace_load'].call_count,0)

    def test_model_loads_are_not_in_sample_entrypoints(self):
        for filename in ['gen_refpano.py','gen_autolayering.py','gen_layerdata.py']:
            tree=ast.parse((REPO/filename).read_text())
            run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_sample')
            calls=[n for n in ast.walk(run) if isinstance(n,ast.Call)]
            self.assertFalse(any(isinstance(n.func,ast.Attribute) and n.func.attr in ('from_pretrained','load_lora_weights','load_models') for n in calls),filename)
            self.assertFalse(any(isinstance(n.func,ast.Name) and n.func.id in ('trace_load','load_depth_model','LamaInpainting') for n in calls),filename)

    def test_pasd_real_main_with_resident_skips_both_loaders(self):
        # No input layers, but native rgb processing and sample-local generator still execute.
        calls=[]; load=mock.Mock(side_effect=AssertionError('must not load per sample'))
        class Gen:
            def __init__(self,device): self.seed=None
            def manual_seed(self,seed): self.seed=seed
        class Image:
            size=(32,16)
            def convert(self,mode): return self
            def resize(self,size): self.size=size; return self
            def save(self,path): Path(path).write_text('CPU-test-double')
        pipeline=lambda args,prompt,image,**kw:(calls.append((args.inputs_dir,prompt,kw['generator'].seed)) or types.SimpleNamespace(images=[image]))
        transform=types.SimpleNamespace(Compose=lambda x:lambda img:img,Resize=lambda *a,**k:None,InterpolationMode=types.SimpleNamespace(BILINEAR=1))
        env=dict(count_layer=lambda p:0,set_seed=lambda s:None,load_models=load,transforms=transform,torch=types.SimpleNamespace(Generator=Gen),Image=types.SimpleNamespace(open=lambda p:Image()),
                 get_validation_prompt=lambda a,*rest:a.inputs_dir,wavelet_color_fix=lambda im,ref:im,os=os)
        native=functions_only('pasd/run_layers_pasd.py',['build_parser','main'],dict(env,argparse=__import__('argparse')))
        resident=(types.SimpleNamespace(is_main_process=True,device='cpu'),pipeline,object(),object(),None)
        for s in self.samples:
            p=self.root/s['sample_id']
            args=native.build_parser().parse_args(['--inputs_dir',str(p),'--seed','16806','--upscale','2'])
            native.main(args,resident=resident)
        self.assertEqual(load.call_count,0)
        self.assertEqual(len(calls),2)
        self.assertNotEqual(calls[0][1],calls[1][1])
        self.assertEqual([c[2] for c in calls],[16806,16806])

    def test_resume_skips_completed_samples_and_preserves_events(self):
        calls=[]
        fn=lambda s,p:(calls.append(s['sample_id']),self.fixture(s,p))
        self.process('refpano',fn)
        before=len(self.events()); self.process('refpano',fn)
        self.assertEqual(len(calls),2)
        self.assertGreater(len(self.events()),before)
        self.assertEqual(sum(e['kind']=='resume_skip' for e in self.events()),2)

    def test_failure_stops_second_sample_and_resume_retries_only_missing(self):
        count=[]
        def fail(s,p):
            count.append(s['sample_id'])
            raise RuntimeError('honest fixture failure')
        with self.assertRaises(RuntimeError): self.process('refpano',fail)
        self.assertEqual(len(count),1)
        self.assertFalse(any((self.root/'evidence/stages').iterdir()))
        self.process('refpano')
        self.assertEqual(sum(e['kind']=='sample_failed' for e in self.events()),1)
        self.assertEqual(sum(e['kind']=='sample_done' for e in self.events()),2)

    def test_changed_output_hash_rejected(self):
        self.process('refpano')
        (self.root/self.samples[0]['sample_id']/'fixture.txt').write_text('tamper')
        with self.assertRaisesRegex(ValueError,'changed/missing'): self.process('refpano')

    def test_changed_identity_rejected(self):
        self.process('refpano'); self.m['identity_sha256']='different'
        with self.assertRaisesRegex(ValueError,'identity mismatch'): self.process('refpano')

    def test_missing_upstream_rejected_without_advancing(self):
        with self.assertRaisesRegex(ValueError,'Missing prerequisite'): self.process('panodepth')
        self.assertFalse(any((self.root/'evidence/stages').iterdir()))

    def test_changed_upstream_stamp_rejected(self):
        self.process('refpano'); self.process('panodepth')
        p=runner.stamp_path(self.root,'refpano',self.samples[0]); record=json.loads(p.read_text()); record['unexpected']='change'; audit.save_json(p,record)
        with self.assertRaisesRegex(ValueError,'upstream identity changed'): self.process('panodepth')

    def test_timing_sums_shared_once_and_retains_failure(self):
        audit.event('batch_start'); audit.event('shared_load_done',elapsed_seconds=3)
        audit.event('model_load_done',elapsed_seconds=2,loader='fixture',requested_path='native')
        for s in self.samples: audit.event('sample_done',elapsed_seconds=5,sample_id=s['sample_id'])
        audit.event('sample_failed',elapsed_seconds=1,sample_id=self.samples[0]['sample_id'])
        audit.event('batch_done',elapsed_seconds=14)
        r=audit.summarize(self.root)
        self.assertEqual(r['shared_stage_load_seconds'],3)
        self.assertEqual(r['successful_sample_stage_seconds'],10)
        self.assertEqual(list(r['sample_processing_seconds'].values()),[5,5])
        self.assertEqual(r['completed_attempt_monotonic_seconds'],14)
        self.assertTrue(any(e['kind']=='sample_failed' for e in r['events']))
        self.assertTrue((self.root/'timings.csv').exists())

    def test_cuda_sync_boundaries_exercised(self):
        marks=[]
        fake=types.SimpleNamespace(cuda=types.SimpleNamespace(is_initialized=lambda:True,synchronize=lambda:marks.append('sync')))
        with mock.patch.dict(sys.modules,{'torch':fake}):
            with audit.timed('fixture'): marks.append('work')
        self.assertEqual(marks,['sync','work','sync'])

    def test_busy_guard_refuses_without_reading_other_users_data(self):
        with mock.patch.object(runner.subprocess,'check_output',return_value='123, 17274') as query:
            with self.assertRaisesRegex(RuntimeError,'GPU_BUSY'): runner.idle_gate()
        self.assertEqual(query.call_count,1)

    def test_only_assistant_can_launch(self):
        with self.assertRaisesRegex(ValueError,'exclusively'): runner.run_driver(self.root,'developer')

    def events(self):
        p=self.root/'evidence/events.jsonl'
        return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []

if __name__=='__main__': unittest.main(verbosity=2)
