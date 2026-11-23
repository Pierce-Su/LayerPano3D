#!/usr/bin/env python3
"""CPU-only readiness publisher. This does not launch GPU work."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.curated_runtime import digest, save_json, utc
from scripts.curated_two import check_identity


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args(); root=args.root.resolve()
    m=check_identity(root)
    if (root/'READY.json').exists():
        raise RuntimeError('READY already published; do not silently overwrite readiness history')
    started=utc(); start=time.perf_counter()
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TMPDIR=str(root/'evidence/tmp'))
    Path(env['TMPDIR']).mkdir(exist_ok=True)
    tests=[]
    for name,script in [('cpu-orchestration','tests/test_curated_two.py'),('cpu-native-imports','tests/import_curated_native_cpu.py')]:
        command=[sys.executable,'-u',script]
        stamp=utc(); t=time.perf_counter()
        log=root/'evidence'/f'{name}.log'
        with log.open('w') as out:
            result=subprocess.run(command,cwd=REPO,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=900)
        text=log.read_text()
        tests.append(dict(name=name,command=command,start_utc=stamp,end_utc=utc(),elapsed_seconds=time.perf_counter()-t,exit_code=result.returncode,log=str(log.relative_to(root)),sha256=digest(log)))
        save_json(root/'evidence/cpu-test-results.json',tests)
        if result.returncode:
            raise RuntimeError(f'{name} failed: {log}')
        if name=='cpu-orchestration':
            count=re.search(r'Ran (\d+) tests?',text)
            if not count or int(count[1])<16 or not re.search(r'^OK\s*$',text,re.MULTILINE):
                raise RuntimeError('CPU test success/count evidence not found')
            tests[-1]['tests_passed']=int(count[1])
        elif '"cuda_context_initialized": false' not in text or '"status": "PASS"' not in text:
            raise RuntimeError('Real native CPU import success evidence not found')
    # Recheck after tests; code/settings cannot change between tested and published identity.
    check_identity(root)
    preparation=dict(start_utc=started,end_utc=utc(),elapsed_seconds=time.perf_counter()-start,kind='CPU readiness tests only, not inference or full developer effort',tests=tests)
    save_json(root/'evidence/preparation.json',preparation)
    subprocess.run(['conda','list','--explicit'],stdout=(root/'evidence/conda-explicit.txt').open('w'),check=True)
    ready=dict(status='CPU_READY_NOT_GPU_VALIDATED',created_utc=utc(),task_id='t_b7a371cb',launch_owner='assistant',monitor_card='t_eae9b14d',
               identity_sha256=m['identity_sha256'],manifest=str(root/'manifest.json'),output_root=str(root),repo=str(REPO),repo_commit=m['repo_commit'],
               cpu_tests=tests,launch_argv=['bash',str(REPO/'scripts/launch_curated_two.sh'),str(root)],
               resume_argv=['bash',str(REPO/'scripts/launch_curated_two.sh'),str(root)],
               gate_argv=[sys.executable,str(REPO/'scripts/curated_two.py'),'--root',str(root),'--gate'],
               tmux='layerpano3d-curated-two',launch_lock=str(root/'evidence/launch.lock'),driver_lock=str(root/'evidence/driver.lock'),
               expected_driver_pid_file=str(root/'evidence/driver.PID'),driver_log=str(root/'evidence/driver.log'),events=str(root/'evidence/events.jsonl'),
               done=str(root/'DONE'),failed=str(root/'FAILED'),entry_exit=str(root/'evidence/tmux-entry.EXIT'),
               completion_owner='developer on t_b7a371cb; Assistant triggers continuation, does not claim final scene acceptance',
               notes=['No GPU job launched by Developer. CPU tests use explicit test doubles, not generated scenes.',
                      'Assistant alone performs nvidia-smi idle gate then launches. Never coexist with another compute process or kill other jobs.',
                      'Same identity resume skips verified samples; history/timers are not reset. Exit 0 alone never creates DONE.',
                      'Do not create a second polling cron; existing Assistant cron 137d0f4b1fa2 owns the 10-minute checks.'])
    save_json(root/'READY.json',ready)
    print(json.dumps(ready,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
