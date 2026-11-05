"""Versioned local CUDA codec build; never edits previous attempts or environments."""
from __future__ import annotations
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
from probe_toolchain_v2 import configure,snapshot
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',required=True);parser.add_argument('--attempt',required=True);args=parser.parse_args()
    run=(ROOT/args.run_dir).resolve();attempt=args.attempt
    if not attempt.isalnum() or not run.is_relative_to(ROOT/'results/gpu_codec_v1') or (run/'artifact_integrity.json').exists():raise ValueError('Requires new unsealed run and alphanumeric attempt')
    build=run/f'build/cuda_{attempt}';build.mkdir(parents=True,exist_ok=False)
    record=dict(run_id=run.name,attempt=attempt,started_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),command=[sys.executable]+sys.argv,status='running')
    start=time.perf_counter()
    try:
        configuration=configure()
        import torch
        from torch.utils.cpp_extension import load
        environment=snapshot(run,configuration)
        environment['notes']=['Fresh G1b environment snapshot; old G1a source and environment reused read-only.','Project-local CUDA 11.8/MSVC 19.38, existing SDK/Ninja, no package or global environment changes.','Instantaneous device readings are not formal benchmark conditions.']
        environment['compiler']['provisioning_record']='../../gpu_codec_v1_20261101_002/logs/toolchain_summary.json'
        dump(run/'environment.json',environment)
        source_dir=build/'sources';source_dir.mkdir()
        sources=[]
        for name in ('codec.cpp','codec.cu'):
            source=ROOT/'src/gradient_comm_lab/csrc'/name;target=source_dir/name;shutil.copyfile(source,target);sources.append(target)
        for source in (ROOT/'src/gradient_comm_lab/cuda_codec.py',Path(__file__)):
            shutil.copyfile(source,source_dir/source.name)
        flags=['-O3','--fmad=false','--ftz=false','--prec-div=true','--prec-sqrt=true','--use-local-env','-lineinfo','-Xptxas=-v','-ccbin',str(Path(configuration['msvc_root'])/'bin/Hostx64/x64')]
        name='gradient_comm_lab_cuda_'+attempt
        record.update(configuration=configuration,module_name=name,build_directory=str(build),source_sha256={str(p):sha(p) for p in source_dir.iterdir()},extra_cuda_cflags=flags,extra_cflags=['/O2'])
        module=load(name=name,sources=[str(p) for p in sources],build_directory=str(build),extra_cflags=['/O2'],extra_cuda_cflags=flags,with_cuda=True,verbose=True)
        path=Path(module.__file__);record.update(status='compiled_loaded',module_path=str(path),module_sha256=sha(path),build_ninja_sha256=sha(build/'build.ninja'))
        dump(run/'logs/cuda_build_manifest.json',{k:record[k] for k in ('run_id','attempt','module_name','module_path','module_sha256','source_sha256','extra_cuda_cflags','extra_cflags','build_ninja_sha256')})
        environment['compiler']['codec_extension_build_record']=f'logs/cuda_build_{attempt}.json';environment['compiler']['codec_extension_status']='compiled_loaded';dump(run/'environment.json',environment)
    except Exception as exc:
        record.update(status='failed',exception=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()));traceback.print_exc()
    record.update(finished_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),wall_seconds=time.perf_counter()-start)
    dump(run/f'logs/cuda_build_{attempt}.json',record);print(json.dumps(dict(status=record['status'],attempt=attempt)))
    return 0 if record['status']=='compiled_loaded' else 1
if __name__=='__main__':sys.exit(main())
