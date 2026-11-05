"""Deterministic G1a fixtures, retaining the accepted G0 real-gradient bytes."""
from __future__ import annotations
from datetime import datetime
import hashlib
import json
from pathlib import Path
import torch
from .cpu_reference import lookup_table,pad_input,threshold_fixture,tensor_identity,wire_size

ROOT=Path(__file__).resolve().parents[2]
OLD_RUN=ROOT/'results/gpu_codec_v1/gpu_codec_v1_20261101_001'
LENGTHS=(256,4096,65536,262144,1048576,4194304)
SEEDS=(7301,7302,7303)
DISTRIBUTIONS=('standard_normal','group_scaled_normal')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()

def save_json(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)

def synthetic_id(length,distribution,seed=7301):
    return f'synthetic_{distribution}_seed{seed}_n{length}'

def synthetic(length,distribution,seed):
    generator=torch.Generator(device='cpu').manual_seed(seed)
    x=torch.randn(length,generator=generator,dtype=torch.float32)
    if distribution=='group_scaled_normal':
        group=torch.arange(length,dtype=torch.int64)//256
        scales=torch.pow(torch.tensor(10.,dtype=torch.float32),-6.+6.*(group%7).to(torch.float32)/6.)
        x=x*scales
    elif distribution!='standard_normal':raise ValueError('Unknown distribution')
    return x.to(torch.bfloat16)

def edge_inputs():
    for length in (0,1,15,16,255,256,257,1023,1024,1025):
        yield f'edge_length_{length}',torch.linspace(-1.,1.,length,dtype=torch.float32).to(torch.bfloat16),{'case':'outer_length','logical_length':length},('generated',)
    tiny=torch.finfo(torch.bfloat16).tiny
    cases={'zero':torch.zeros(256,dtype=torch.bfloat16),'positive_constant':torch.ones(256,dtype=torch.bfloat16),
        'negative_constant':-torch.ones(256,dtype=torch.bfloat16),
        'signed_zero':torch.tensor([0.,-0.]*128,dtype=torch.bfloat16),
        'subnormal':torch.tensor([tiny/128,-tiny/128,tiny/2,-tiny/2,tiny,-tiny,0.,1e-23]*32,dtype=torch.bfloat16),
        'group_scale_disparity':synthetic(512,'group_scaled_normal',7301)}
    for bits in (4,8):
        table=lookup_table(bits);bound=1<<(bits-1)
        central=table[(1<<bits)-bound:(1<<bits)+bound+1]
        midpoint=((central[:-1].float()+central[1:].float())/2).to(torch.bfloat16)
        before=torch.nextafter(midpoint,torch.full_like(midpoint,float('-inf')))
        after=torch.nextafter(midpoint,torch.full_like(midpoint,float('inf')))
        pattern=torch.cat((central,midpoint,before,after))
        cases[f'lut{bits}_endpoints_midpoints_neighbors']=pattern.repeat((256+pattern.numel()-1)//pattern.numel())
    for name,value in cases.items():
        yield 'edge_'+name,value.contiguous(),{'case':name,'generation':'explicit BF16 constants or CPU LUT central legal range, arithmetic midpoints and adjacent BF16'},('generated','zero','maximum')

def prepare_fixtures(run_dir):
    run=Path(run_dir).resolve()
    if not run.is_relative_to(ROOT/'results/gpu_codec_v1'):raise ValueError('Invalid independent run')
    if (run/'input_manifest.json').exists():
        manifest=json.loads((run/'input_manifest.json').read_text(encoding='utf-8'))
        for row in manifest['inputs']:
            if sha(row['fixture_path'])!=row['fixture_sha256']:raise ValueError('Existing fixture changed')
        return manifest
    directory=ROOT/'fixtures'/run.name
    directory.mkdir(parents=True,exist_ok=False)
    tables={b:lookup_table(b) for b in (4,8)}
    lut={str(b):tensor_identity(t) for b,t in tables.items()}
    rows=[]
    def add(input_id,x,kind,configuration,threshold_mode='generated'):
        logical=x.contiguous();padded=pad_input(logical)
        main,scale,generator=threshold_fixture(input_id,padded.numel())
        if threshold_mode!='generated':
            value=0. if threshold_mode=='zero' else 65536.
            main.fill_(value);scale.fill_(value)
            generator.update(mode='explicit_constant',constant=value,main=tensor_identity(main),scale=tensor_identity(scale))
        else:generator['mode']='generated'
        path=directory/f'{input_id}.pt'
        torch.save({'schema_version':'gpu_codec_v1.0','input_id':input_id,'input_logical':logical,
            'input_padded':padded,'main_thresholds':main,'scale_thresholds':scale,'lookup_tables':tables,
            'threshold_generator':generator},path)
        rows.append({'input_id':input_id,'source_kind':kind,'original_numel':logical.numel(),'padded_numel':padded.numel(),
            'dtype':'bfloat16','shape':[logical.numel()],'byte_order':'little','input_sha256':tensor_identity(logical)['sha256'],
            'padded_input_sha256':tensor_identity(padded)['sha256'],'fixture_path':str(path),'fixture_sha256':sha(path),'fixture_bytes':path.stat().st_size,
            'generator_or_loader':'gradient_comm_lab.fixtures_v2','generator_config':{**configuration,'thresholds':generator},
            'source_path':None,'source_sha256':None,'training_seed':None,'update_step':None,'rank':None,
            'transform_order':['CPU FP32 generation or exact BF16 edge construction','to(BF16)','zero pad to multiple of256'],
            'main_thresholds_sha256':generator['main']['sha256'],'scale_thresholds_sha256':generator['scale']['sha256'],
            'lut_sha256_by_bits':{b:e['sha256'] for b,e in lut.items()},'expected_packet_bytes_by_bits':{str(b):wire_size(padded.numel(),b) for b in (4,8)},
            'threshold_mode':threshold_mode})
    for length in LENGTHS:
        for distribution in DISTRIBUTIONS:
            for seed in SEEDS:
                add(synthetic_id(length,distribution,seed),synthetic(length,distribution,seed),'synthetic',
                    {'generator':'torch.Generator(device=cpu)','seed':seed,'distribution':distribution,
                     'pytorch_version':torch.__version__,'reset_generator_per_input':True,
                     'group_scale_formula':'CPU FP32 10**(-6+6*(floor(index/256)%7)/6)' if distribution=='group_scaled_normal' else None})
    for identity,x,configuration,modes in edge_inputs():
        for mode in modes:add(identity+'_'+mode,x,'edge',configuration,mode)
    old=json.loads((OLD_RUN/'input_manifest.json').read_text(encoding='utf-8'))
    for source in old['inputs']:
        if sha(source['fixture_path'])!=source['fixture_sha256']:raise ValueError('Accepted real fixture changed')
        rows.append({**source,'accepted_G0_input_manifest':str(OLD_RUN/'input_manifest.json'),
            'accepted_G0_input_manifest_sha256':sha(OLD_RUN/'input_manifest.json'),'reused_exact_fixture':True,'threshold_mode':'generated'})
    manifest={'schema_version':'gpu_codec_v1.0','template_only':False,'run_id':run.name,'created_at':datetime.now().astimezone().isoformat(),
        'inputs':rows,'holdout_opened':False,'scope':'G1a CPU_GPU parity and torch-only performance input catalog',
        'source_sha256':{str(Path(__file__).resolve()):sha(__file__),str(ROOT/'src/gradient_comm_lab/cpu_reference.py'):sha(ROOT/'src/gradient_comm_lab/cpu_reference.py')},
        'counts_by_source_kind':{k:sum(r['source_kind']==k for r in rows) for k in ('edge','synthetic','real_gradient')}}
    save_json(run/'input_manifest.json',manifest)
    save_json(run/'fixtures_manifest.json',{'schema_version':'gpu_codec_v1.0','run_id':run.name,'files':[
        {'path':r['fixture_path'],'sha256':r['fixture_sha256'],'bytes':r['fixture_bytes'],'input_id':r['input_id']} for r in rows],
        'source_G0_run':str(OLD_RUN),'rebuild':'scripts/python.ps1 scripts/check_torch_codec.py --run-dir <fresh-run> --prepare-only',
        'reconstruction':'Identical input, LUT, threshold tensor hashes; archive file hashes depend on filename and serialization version.'})
    return manifest

def load_fixture(row,device='cpu',verify=True):
    if verify and sha(row['fixture_path'])!=row['fixture_sha256']:raise ValueError('Fixture archive SHA mismatch')
    value=torch.load(row['fixture_path'],map_location='cpu',weights_only=False)
    if verify:
        for key,field in (('input_logical','input_sha256'),('input_padded','padded_input_sha256'),
                          ('main_thresholds','main_thresholds_sha256'),('scale_thresholds','scale_thresholds_sha256')):
            if tensor_identity(value[key])['sha256']!=row[field]:raise ValueError('Fixture tensor mismatch: '+key)
        for b,table in value['lookup_tables'].items():
            if tensor_identity(table)['sha256']!=row['lut_sha256_by_bits'][str(b)]:raise ValueError('Fixture LUT mismatch')
    if str(device)!='cpu':
        for key in ('input_logical','input_padded','main_thresholds','scale_thresholds'):value[key]=value[key].to(device)
        value['lookup_tables']={b:t.to(device) for b,t in value['lookup_tables'].items()}
    return value
