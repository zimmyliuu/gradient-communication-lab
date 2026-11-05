"""Isolated encoders and original-only decoder. Never switch eden_utils in-process."""
import argparse,hashlib,json,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
import torch
import numpy as np
from gradient_comm_lab.epsilon_diagnostic import codec
BASE=ROOT/'results/bounded_discovery_v1/bounded_discovery_v1_20261104_001'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def thash(x):return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def dump(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def append(p,v):
    with Path(p).open('a',encoding='utf-8') as f:f.write(json.dumps(v,ensure_ascii=False,allow_nan=False)+'\n')
def domain(wire,bits,padded):
    gp=16*bits//8;block=16*(gp+1)+2
    assert len(wire)==padded//256*block
    data=np.frombuffer(wire,dtype=np.uint8).reshape(-1,block);groups=data[:,:-2].reshape(-1,gp+1)
    main=groups[:,:gp].reshape(-1)
    if bits==4:main=np.stack([main&15,main>>4],axis=1).reshape(-1)
    units=(data[:,-2].astype(np.uint32)|(data[:,-1].astype(np.uint32)<<8))<<16
    values=units.view(np.float32)
    return dict(reserved_code_count=int(np.count_nonzero(main==(1<<bits)-1)),invalid_scale_count=int(np.count_nonzero(groups[:,-1]==0))+int(np.count_nonzero(~np.isfinite(values)|(values<=0))),
        boundary_code_fraction=float(np.mean((main==0)|(main==(1<<bits)-2))),main_code_count=main.size,counts_include_padding=True,clamp_count=None)
def items(run,phase):
    if phase=='formal':return read(run/'input_randomness_manifest.json')['pairs']
    d0=read(BASE/'logs/d0_execution.json');result=[]
    for r in d0['cases']:
        name=r['case_id'];b=r['bits']
        if not ((b==8 and name in ['zero_native_min','constant','mixed_signed','sparse_large','tail257','real_seed1101_update100_rank0']) or (b==4 and name=='mixed_signed')):continue
        result.append(dict(pair_id=f'qualification_{name}_b{b}',bits=b,input_path=str(BASE/'d0_cases'/f'{name}.pt'),input_key='input',
            input_file_sha256=sha(BASE/'d0_cases'/f'{name}.pt'),input_tensor_sha256=r['input_sha256'],logical_numel=r['logical_numel'],padded_numel=r['padded_numel'],
            original_seed=r['seed'],original_pool_sha256=r['rand_pool_sha256'],original_packet_path=str(BASE/'d0_cases'/f'{name}_b{b}.bin'),
            original_packet_sha256=r['packet_sha256'],expected_decoded_tensor_sha256=r['output_sha256']))
    assert len(result)==7
    return result
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',required=True);ap.add_argument('--operation',choices=['encode','decode'],required=True)
    ap.add_argument('--condition',choices=['control_add1e7','diagnostic_add0'],required=True);ap.add_argument('--phase',choices=['qualification','formal'],required=True)
    ap.add_argument('--build-manifest',required=True);a=ap.parse_args();run=(ROOT/a.run_dir).resolve()
    assert run.is_relative_to(ROOT/'results/epsilon_diagnostic_v1')
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    assert (run/'experiment_contract.json').exists(),'Freeze the experiment before any codec calls'
    contract=read(run/'experiment_contract.json')
    ch=sha(run/'experiment_contract.json') if contract else None
    if contract:
        assert sha(run/'input_randomness_manifest.json')==contract['input_randomness_manifest_sha256']
        assert sha(run/'source_manifest.json')==contract['source_manifest_sha256']
        for r in read(run/'source_manifest.json')['files']:
            assert sha(r['path'])==r['sha256'],r['path']
        if a.condition=='diagnostic_add0' and a.phase=='formal':
            parity=[json.loads(x) for x in (run/'control_parity.jsonl').read_text().splitlines()]
            assert len(parity)==96 and all(r['status']=='pass' for r in parity)
    manifest=Path(a.build_manifest).resolve()
    if a.operation=='decode':assert manifest==BASE/'logs/official_build_a03.json'
    else:assert manifest.is_relative_to(run/'build')
    codec.set_build_manifest(manifest);loaded=codec.build_identity()
    if contract:
        expected=contract['decoder_binary_sha256'] if a.operation=='decode' else contract['encoder_binary_sha256'][a.condition]
        assert loaded['module_sha256']==expected
    key=f'{a.phase}_{a.operation}_{a.condition}';log=run/'logs'/f'{key}.jsonl';assert not log.exists()
    gpu_start=time.perf_counter();gpu_window=0.;passed=0;failed=0;began=time.time()
    dump(run/'logs'/f'{key}_process.json',dict(pid=os.getpid(),operation=a.operation,condition_id=a.condition,phase=a.phase,
        loaded_module=loaded,contract_sha256=ch,entrypoints=['scaling_compress' if a.operation=='encode' else 'scaling_decompress'],
        dtype='BF16',chunk_size=16,strategy=3,original_decoder_only=a.operation=='decode',started_unix=began))
    entries=items(run,a.phase)
    previous_wall=sum(read(p)['wall_seconds'] for p in (run/'logs').glob('*_completion.json'))
    build_upper=contract.get('build_wall_seconds_upper_bound',0) if contract else 0
    encoded={}
    if a.operation=='decode':encoded={r['pair_id']:r for r in [json.loads(x) for x in (run/'logs'/f'{a.phase}_encode_{a.condition}.jsonl').read_text().splitlines()]}
    for item in entries:
        if previous_wall+build_upper+time.time()-began>=1800:raise RuntimeError('Conservative cumulative GPU budget reached')
        pid=item['pair_id'];bits=item.get('bits',8);padded=item['padded_numel'];d=item['logical_numel']
        base=dict(pair_id=pid,condition_id=a.condition,phase=a.phase,contract_sha256=ch,process_id=os.getpid(),loaded_module=loaded,
            logical_numel=d,padded_numel=padded,bits=bits)
        evidence=run/'packets'/a.phase/a.condition/f'{pid}.bin';evidence.parent.mkdir(parents=True,exist_ok=True)
        x=None;pool=None;native=None;y=None
        try:
            if a.operation=='encode':
                assert sha(item['input_path'])==item['input_file_sha256'];x=torch.load(item['input_path'],map_location='cpu',weights_only=False)[item.get('input_key','input_logical')].contiguous()
                assert thash(x)==item['input_tensor_sha256'] and x.numel()==d and x.dtype==torch.bfloat16
                pool=codec.make_random_pool(padded,item['original_seed'],device='cpu');pool_hash=thash(pool)
                assert pool_hash==item['original_pool_sha256']
                torch.cuda.synchronize();t=time.perf_counter()
                xg=x.cuda();pg=pool.cuda();packet=codec.encode(xg,bits,rand_pool=pg);wire=packet.to_bytes()
                # Save produced artifacts before any postcondition can fail.
                with evidence.open('xb') as f:f.write(wire)
                validity=domain(wire,bits,padded);assert validity['reserved_code_count']==validity['invalid_scale_count']==0
                immutable_input=thash(xg)==item['input_tensor_sha256'];immutable_pool=thash(pg)==pool_hash
                torch.cuda.synchronize();gpu_window+=time.perf_counter()-t
                assert immutable_input and immutable_pool
                assert len(wire)==codec.required_bytes(padded,bits)
                mismatches=None
                if a.condition=='control_add1e7' or bits==4:
                    original=Path(item['original_packet_path']).read_bytes();assert hashlib.sha256(original).hexdigest()==item['original_packet_sha256']
                    assert len(original)==len(wire);mismatches=sum(aa!=bb for aa,bb in zip(original,wire));assert mismatches==0
                append(log,dict(**base,status='pass',packet_path=str(evidence),packet_sha256=sha(evidence),packet_bytes=len(wire),
                    packet_byte_mismatch_count=mismatches,input_hash_match=immutable_input,pool_hash_match=immutable_pool,
                    pool_sha256=pool_hash,seed=item['original_seed'],input_path=item['input_path'],packet_domain=validity))
            else:
                entry=encoded[pid];assert entry['status']=='pass';wire=Path(entry['packet_path']).read_bytes();assert hashlib.sha256(wire).hexdigest()==entry['packet_sha256']
                torch.cuda.synchronize();t=time.perf_counter()
                p=codec.OfficialPacket.from_bytes(wire,logical_numel=d,bits=bits);native=codec.decode(p,padded=True).cpu();y=native[:d].clone()
                out=run/'decoded'/a.phase/a.condition/f'{pid}.pt';out.parent.mkdir(parents=True,exist_ok=True);torch.save(y,out)
                torch.cuda.synchronize();gpu_window+=time.perf_counter()-t
                assert y.numel()==d and y.dtype==torch.bfloat16
                finite=int((~torch.isfinite(native)).sum());validity=domain(wire,bits,padded)
                assert finite==0 and validity['reserved_code_count']==validity['invalid_scale_count']==0
                mismatch=None
                if a.condition=='control_add1e7' or bits==4:
                    if a.phase=='formal':
                        assert sha(item['original_decoded_path'])==item['original_decoded_sha256']
                        old=torch.load(item['original_decoded_path'],map_location='cpu',weights_only=True)
                        mismatch=int(np.unpackbits(np.bitwise_xor(old.view(torch.uint8).numpy(),y.view(torch.uint8).numpy())).sum());assert mismatch==0
                    else:assert thash(y)==item['expected_decoded_tensor_sha256'];mismatch=0
                row=dict(**base,status='pass',packet_path=entry['packet_path'],packet_sha256=entry['packet_sha256'],
                    decoded_path=str(out),decoded_sha256=sha(out),decoded_tensor_sha256=thash(y),nonfinite_output_count=finite,
                    decoded_bit_mismatch_count=mismatch,packet_domain=validity,tail_numel=padded-d,tail_nonfinite_count=int((~torch.isfinite(native[d:])).sum()),
                    tail_nonzero_count=int(torch.count_nonzero(native[d:])),pool_hash_match=entry['pool_hash_match'],input_hash_match=entry['input_hash_match'])
                append(log,row)
                if a.phase=='formal' and a.condition=='control_add1e7':
                    append(run/'control_parity.jsonl',dict(**base,status='pass',packet_byte_mismatch_count=entry['packet_byte_mismatch_count'],
                        decoded_bit_mismatch_count=mismatch,pool_hash_match=entry['pool_hash_match'],input_hash_match=entry['input_hash_match'],
                        encoder_identity=entry['loaded_module'],decoder_identity=loaded))
            append(run/'correctness.jsonl',dict(contract_sha256=ch,condition_id=a.condition,pair_id=pid,check=a.phase+'_'+a.operation,
                status='pass',details=base,counterexample_path=None));passed+=1
        except Exception as exc:
            faildir=run/'counterexamples'/a.phase/a.condition/pid;faildir.mkdir(parents=True,exist_ok=True)
            if x is None:x=torch.load(item['input_path'],map_location='cpu',weights_only=False)[item.get('input_key','input_logical')]
            if pool is None:pool=codec.make_random_pool(padded,item['original_seed'],device='cpu')
            if x is not None:torch.save(x,faildir/'input.pt')
            if pool is not None:torch.save(pool,faildir/'random_pool.pt')
            if native is not None:torch.save(native,faildir/'decoded_native.pt')
            if y is not None:torch.save(y,faildir/'decoded_logical.pt')
            failed+=1;failure=dict(**base,status='fail',reason=repr(exc),traceback=traceback.format_exc(),counterexample_path=str(evidence) if evidence.exists() else None,
                input_path=item.get('input_path'),seed=item.get('original_seed'),pool_sha256=item.get('original_pool_sha256'))
            append(log,failure);append(run/'issues.jsonl',failure)
            append(run/'correctness.jsonl',dict(contract_sha256=ch,condition_id=a.condition,pair_id=pid,check=a.phase+'_'+a.operation,status='fail',details=failure,counterexample_path=failure['counterexample_path']))
            break
    dump(run/'logs'/f'{key}_completion.json',dict(status='pass' if passed==len(entries) and failed==0 else 'fail',passed=passed,failed=failed,expected=len(entries),
        gpu_window_seconds=gpu_window,wall_seconds=time.time()-began,loaded_module=loaded,pid=os.getpid(),phase=a.phase,condition_id=a.condition,contract_sha256=ch))
    print(json.dumps(dict(stage=key,passed=passed,failed=failed,gpu_window_seconds=gpu_window)),flush=True)
    if failed:raise SystemExit(1)
if __name__=='__main__':main()
