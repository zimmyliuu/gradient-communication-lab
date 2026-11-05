"""Run the predeclared independent small/large GPU activity matrix sequentially."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone,timedelta
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def now():return datetime.now(timezone(timedelta(hours=8))).isoformat()
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',required=True)
    args=parser.parse_args();run=(ROOT/args.run_dir).resolve()
    assert run.is_relative_to(ROOT/'results/gpu_codec_v1') and not (run/'artifact_integrity.json').exists()
    assert read(run/'logs/benchmark_execution.json')['status']=='complete'
    protocol=[]
    for method,tag in [('torch_cuda_repaired','t'),('cuda_optimized_repaired','c')]:
        for bits in (4,8):
            for n,size in [(65536,'s'),(4194304,'l')]:
                protocol.append({'method_id':method,'nominal_bits':bits,'original_numel':n,'attempt':f'{tag}{bits}{size}a01'})
    write(run/'logs/trace_matrix_registration.json',{'created_at':now(),'traces':protocol,'independent_from_formal_timing':True,
        'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__).resolve(),ROOT/'scripts/profile_cuda_pair.py',ROOT/'scripts/audit_cuda_pair_trace.py',ROOT/'scripts/cupti_trace_collector_v2.cpp']}})
    completed=[]
    for case in protocol:
        attempt=case['attempt']
        commands=[
            [sys.executable,str(ROOT/'scripts/profile_cuda_pair.py'),'--run-dir',str(run),'--attempt',attempt,
             '--method',case['method_id'],'--bits',str(case['nominal_bits']),'--numel',str(case['original_numel'])],
            [sys.executable,str(ROOT/'scripts/audit_cuda_pair_trace.py'),'--run-dir',str(run),'--attempt',attempt]]
        results=[]
        for index,command in enumerate(commands):
            log=run/f'logs/trace_{attempt}_{index}_console.log'
            started=now()
            with log.open('xb') as output:
                result=subprocess.run(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
            results.append({'command':command,'started_at':started,'finished_at':now(),'returncode':result.returncode,'console_log':log.relative_to(run).as_posix(),'console_sha256':sha(log)})
            if result.returncode:
                write(run/f'logs/trace_{attempt}_execution.json',{'status':'failed','case':case,'commands':results})
                raise RuntimeError(f'Trace {attempt} failed; see {log}')
        manifest_path=f'traces/gpu_activity_manifest_{attempt}.json';manifest=read(run/manifest_path)
        audit=read(run/manifest['audit_path'])
        assert audit['status']=='pass' and audit['dropped_records']==0 and audit['kernel_count']>0
        write(run/f'logs/trace_{attempt}_execution.json',{'status':'pass','case':case,'commands':results})
        completed.append({**case,'manifest_path':manifest_path,'manifest_sha256':sha(run/manifest_path),
            'audit_path':manifest['audit_path'],'audit_sha256':sha(run/manifest['audit_path']),
            'kernel_count':audit['kernel_count'],'kernels_per_operation':audit['kernels_per_operation'],
            'device_memcpy_direction_counts':audit['device_memcpy_direction_counts'],'device_memcpy_bytes':audit['device_memcpy_bytes']})
        print(json.dumps({'traces_completed':len(completed),'total':8,**completed[-1]}),flush=True)
    write(run/'traces/g1b_trace_index.json',{'schema_version':'gpu_activity_g1b.0','run_id':run.name,'status':'pass','finished_at':now(),
        'source_sha256':sha(__file__),'traces':completed,'scope':'Three independent preallocated trusted roundtrips for each method/bits/size; trace durations never used as formal latency samples.'})

if __name__=='__main__':main()
