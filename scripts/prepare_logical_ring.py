"""Prepare a new G2a run, retaining explicit user override and frozen inputs."""
import argparse,json,sys,shutil,platform
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from gradient_comm_lab.cpu_reference import tensor_identity,lookup_table
from gradient_comm_lab.ring_fixtures import sha,make_thresholds,case_events
OLD=ROOT/'results/gpu_codec_v1/gpu_codec_v1_20261104_001'
def write(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',required=True);a=ap.parse_args()
    run=(ROOT/a.run_dir).resolve()
    if not run.is_relative_to(ROOT/'results/logical_ring_v1'):raise ValueError('Invalid run directory')
    run.mkdir(parents=True,exist_ok=False)
    for name in ('fixtures','logs','code','failures'): (run/name).mkdir()
    checked=[]
    for row in read(OLD/'artifact_integrity.json')['files']:
        p=OLD/row['path'];assert sha(p)==row['sha256'] and p.stat().st_size==row['bytes'],str(p)
        checked.append(row)
    write(run/'logs/prior_seal_check.json',dict(status='pass',checked_files=len(checked),
        prior_run=str(OLD),prior_integrity_sha256=sha(OLD/'artifact_integrity.json')))
    write(run/'execution_authority.json',dict(date='2026-11-04',user_instruction='不，按照原计划完成，不要那个G2a暂缓的版本',
        effective_scope='Original review sections 3-6 G2a implementation and delivery; header deferral superseded by direct user instruction',
        review_path=str(ROOT/'reviews/gpu_codec_v1_20261104_001_review_and_g2a_plan.md'),
        review_sha256=sha(ROOT/'reviews/gpu_codec_v1_20261104_001_review_and_g2a_plan.md')))
    build=read(OLD/'logs/cuda_build_manifest.json')
    assert sha(build['module_path'])==build['module_sha256']
    write(run/'logs/cuda_build_manifest.json',build)
    cases=[]
    def add(cid,inputs,n,b,mode,kind,pattern,source=None):
        d=inputs[0].numel();p=0 if d==0 else (d if n==1 else ((d+n*1024-1)//(n*1024))*n*1024)
        path=run/'fixtures'/f'{cid}.pt'
        torch.save(dict(case_id=cid,inputs=inputs,entry_mode=mode,bits=b,world_size=n,logical_numel=d),path)
        row=dict(case_id=cid,input_id=cid,worker_count=n,world_size=n,nominal_bits=b,bits=b,
            logical_numel=d,original_numel=d,padded_numel=p,chunk_numel=(p//n if n>1 else 0),
            entry_mode=mode,source_kind=kind,kind=kind,pattern=pattern,fixture_path=path.relative_to(run).as_posix(),
            fixture_sha256=sha(path),fixture_bytes=path.stat().st_size,
            workers=[dict(rank=r,tensor=tensor_identity(x),source=(source[r] if source else None)) for r,x in enumerate(inputs)])
        cases.append(row)
    for n in (1,2,4):
        lengths=sorted(set([0,1,255,256,257,1023,1024,1025,n*1024-1,n*1024,n*1024+1,65536]))
        configs=[(d,'signed') for d in lengths]+[(1025,p) for p in ('zeros','cancellation','finite_extremes')]
        for d,pattern in configs:
            for mode in ('raw_fp32','predivided_bf16'):
                xs=[]
                for rank in range(n):
                    if pattern=='zeros':x=torch.zeros(d)
                    elif pattern=='cancellation':x=torch.linspace(-2,2,d)*((-1.)**rank)
                    elif pattern=='finite_extremes':
                        tiny=torch.finfo(torch.bfloat16).tiny
                        values=torch.tensor([tiny/128,-tiny/128,tiny,-tiny,2.**120,-2.**120,0.,-0.],dtype=torch.float32)
                        x=values.repeat((d+7)//8)[:d].clone()*((-1.)**rank)
                    else:x=torch.linspace(-1,1,d)+(rank-n/2)*0.125
                    # For N1 raw input retains FP32 distinctions at the identity boundary.
                    if mode=='raw_fp32':x=x+torch.tensor(0.00001) if pattern=='signed' else x
                    else:x=x.to(torch.bfloat16).div(n)
                    xs.append(x.contiguous())
                for b in (4,8):add(f'{pattern}_n{n}_d{d}_{mode}_b{b}',xs,n,b,mode,'synthetic',pattern)
    oldinputs=[r for r in read(OLD/'input_manifest.json')['inputs'] if r['source_kind']=='real_gradient']
    for step in (100,500,2500):
        rows=sorted([r for r in oldinputs if r['update_step']==step],key=lambda r:r['rank']);assert len(rows)==4
        xs=[];sources=[]
        for r in rows:
            assert sha(r['fixture_path'])==r['fixture_sha256']
            f=torch.load(r['fixture_path'],map_location='cpu',weights_only=False)
            x=f['input_logical'].clone();assert x.numel()==818048 and tensor_identity(x)['sha256']==r['input_sha256']
            xs.append(x);sources.append(dict(input_id=r['input_id'],fixture_path=r['fixture_path'],fixture_sha256=r['fixture_sha256'],
                update_step=step,training_seed=r['training_seed'],rank=r['rank'],already_divided_by=4,original_numel=r['original_numel']))
        for b in (4,8):add(f'real_update{step:06d}_n4_b{b}',xs,4,b,'predivided_bf16','real_gradient','real_gradient',sources)
    write(run/'input_manifest.json',dict(schema_version='logical_ring_v1.0',run_id=run.name,cases=cases,inputs=cases,
        real_collectives=6,real_snapshots=3,holdout_opened=False,input_generation='Deterministic explicit formulas in frozen prepare_logical_ring.py'))
    threshold_rows=[]
    for c in cases:
        for e in case_events(c):
            _,_,m=make_thresholds(e,c['chunk_numel']);threshold_rows.append(m)
    write(run/'threshold_manifest.json',dict(schema_version='logical_ring_v1.0',events=threshold_rows,
        seed_rule='SHA256(UTF8(logical_ring_v1: + canonical sorted compact JSON event)) first8 bytes little-endian mod2^63',
        generation='CPU torch.rand FP32 main then scale -> BF16 -> multiply BF16(65535)=65536; no LCG',
        torch_version=str(torch.__version__)))
    write(run/'ring_contract.json',dict(schema_version='logical_ring_v1.0',run_id=run.name,
        stage='G2a',scope='fixed_b4_b8_explicit_thresholds_single_gpu_logical_ring',world_sizes=[1,2,4],bits=[4,8],
        raw_fp32='FP32 -> BF16 -> BF16 divide N -> pad -> ring -> trim -> FP32; N1 identity',
        predivided_bf16='no division; BF16 ring sum -> trim; N1 identity',padding='P=N*1024*ceil(D/(N*1024)); L=P/N for N>1,D>0',
        bypass='N1 or D0: original entry dtype, zero messages and codec/threshold events',
        route='right neighbor; RS h0..N-2 chunk(r-h)%N; sink chunk(r+1)%N encode+selfdecode; AG h0..N-2 sends(r+1-h)%N receives(r-h)%N',
        ownership='Owned three-buffer clone after encode, owned decoded clone before reuse, round barriers; AG forwards owned immutable sink packet; one current CUDA stream',
        ledger='Actual tensor numel*element_size for payload,scale_codes,top_scales; audit metadata is not transmitted header',
        expected='B=L*(b+1)/8+L/16+2*L/256; sends2*N*(N-1); encodesN*N; AG no encode',
        cpu_reference_path=str(ROOT.parent/'spl_model_communication/src/splcomm/dynamiq.py'),
        cpu_reference_sha256=sha(ROOT.parent/'spl_model_communication/src/splcomm/dynamiq.py'),
        inherited_codec_contract_sha256=sha(OLD/'codec_contract.json'),build_manifest='logs/cuda_build_manifest.json',
        build_manifest_sha256=sha(run/'logs/cuda_build_manifest.json'),module_sha256=build['module_sha256'],
        input_manifest_sha256=sha(run/'input_manifest.json'),threshold_manifest_sha256=sha(run/'threshold_manifest.json'),
        luts={str(b):tensor_identity(lookup_table(b)) for b in (4,8)},formal_execution_not_started=True))
    write(run/'environment.json',dict(captured_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        platform=platform.platform(),python=sys.version,torch=str(torch.__version__),cuda_runtime=torch.version.cuda,
        cuda_available=torch.cuda.is_available(),device=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        prior_toolchain=str(OLD/'environment.json'),prior_environment_sha256=sha(OLD/'environment.json'),
        binary_reused=True,no_install_or_build=True,execution='single GPU sequential logical workers; not a network benchmark'))
    print(json.dumps(dict(run_dir=str(run),cases=len(cases),threshold_events=len(threshold_rows),real_collectives=6)))
if __name__=='__main__':main()
