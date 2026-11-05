"""Attempt a genuine CUDA extension build and call, retaining any failure."""
from __future__ import annotations
import argparse
from contextlib import redirect_stdout,redirect_stderr
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True)
    args=parser.parse_args()
    run=(ROOT/args.run_dir).resolve()
    if not run.is_relative_to(ROOT/'results/gpu_codec_v1'): raise ValueError('Invalid run directory')
    env=json.loads((run/'environment.json').read_text(encoding='utf-8'))
    resultpath=run/'logs/minimal_cuda_probe.json'
    if resultpath.exists(): raise FileExistsError('Use a fresh attempt directory; do not overwrite the probe')
    compiler=env['compiler']
    added=[str(Path(compiler[k]).parent) for k in ('host_compiler_path','ninja_path') if compiler.get(k)]
    if compiler.get('nvcc_path'):
        os.environ['CUDA_HOME']=str(Path(compiler['nvcc_path']).parents[1])
        added.append(str(Path(compiler['nvcc_path']).parent))
    os.environ['PATH']=os.pathsep.join(added+[os.environ.get('PATH','')])
    os.environ['TORCH_CUDA_ARCH_LIST']='8.9'
    os.environ['MAX_JOBS']='1'
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    build=run/'build/minimal_cuda'; build.mkdir(parents=True,exist_ok=True)
    sources=[ROOT/'src/g0_probe.cpp',ROOT/'src/g0_probe.cu']
    flags=['-O3','--fmad=false','--ftz=false','--prec-div=true','--prec-sqrt=true']
    record={'schema_version':'gpu_codec_v1.0','run_id':run.name,'started_at':datetime.now().astimezone().isoformat(),
        'command':[sys.executable,str(Path(__file__).resolve()),'--run-dir',str(run)],'status':'running',
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'extra_cflags':['/O2'],'extra_cuda_cflags':flags,'with_cuda':True,'target_arch':'8.9','max_jobs':1,
        'subprocess_local_path_additions':added,'global_environment_modified':False,'package_installation_performed':False,
        'build_directory':str(build),'module_called':False,'output_equal':None,'exception':None}
    logpath=run/'logs/minimal_cuda_build.log'
    started=time.perf_counter()
    with logpath.open('x',encoding='utf-8') as log, redirect_stdout(log),redirect_stderr(log):
        print(json.dumps(record,indent=2),flush=True)
        try:
            import torch
            from torch.utils.cpp_extension import load
            module=load(name='gradient_comm_lab_g0_probe',sources=[str(p) for p in sources],
                build_directory=str(build),extra_cflags=['/O2'],extra_cuda_cflags=flags,with_cuda=True,verbose=True)
            x=torch.arange(256,device='cuda',dtype=torch.float32)/8
            y=module.increment(x); torch.cuda.synchronize()
            record['module_called']=True
            record['output_equal']=bool(torch.equal(y,x+1))
            if not record['output_equal']: raise AssertionError('CUDA probe output mismatch')
            record['status']='pass'
        except Exception as exc:
            record['status']='blocked' if not compiler.get('nvcc_path') else 'failed'
            record['exception']={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
            traceback.print_exc()
    record.update(finished_at=datetime.now().astimezone().isoformat(),wall_seconds=time.perf_counter()-started,
        build_log='logs/minimal_cuda_build.log',build_log_sha256=hashlib.sha256(logpath.read_bytes()).hexdigest())
    resultpath.write_text(json.dumps(record,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    env['compiler'].update(flags={'extra_cflags':['/O2'],'extra_cuda_cflags':flags,'arch':'8.9'},
        minimal_cuda_extension_test=record['status'],build_log='logs/minimal_cuda_build.log',probe_record='logs/minimal_cuda_probe.json')
    env['notes'].append('Minimal CUDA build used discovered existing MSVC/Ninja directories in the current Python process PATH only.')
    (run/'environment.json').write_text(json.dumps(env,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':record['status'],'module_called':record['module_called'],'exception':record['exception']['message'] if record['exception'] else None,'record':str(resultpath)}))
    return 0 if record['status']=='pass' else 2


if __name__=='__main__': sys.exit(main())
