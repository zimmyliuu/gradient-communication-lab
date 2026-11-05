"""CPU-only E0 identity, matched-pool regeneration and scale/region precheck.

No codec call, CUDA context, new gradients, or output-error driven selection.
"""
from __future__ import annotations
import argparse,hashlib,json,math,sys,time
from datetime import datetime,timezone,timedelta
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/bounded_discovery_v1/bounded_discovery_v1_20261104_001'
COMMIT='96da2ee480bbfd8e2049de3fbd8595cc9d7a30b8'
OFFICIAL='official_dynamiq_fixed_codec_b8';UNIFORM='uniform_g128_q8'
D=818048;P=818176

def now():return datetime.now(timezone(timedelta(hours=8))).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def raw_sha(t):return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def rows(path):return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]
def dump(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def ref(path):
    path=Path(path).resolve();return dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)

def c8_estimate(source):
    s=source.read_text(encoding='utf-8-sig')
    assert 'void build_lookup_table_host()' in s
    assert 'float epses[9] = {0, 0.23, 0.23, 0.18, 0.18, 0.18, 0.1, 0.1, 0.1}' in s
    assert 'pow_eps2 *= one_plus_2eps2;' in s
    f=np.float32;eps=f(.1);denom=f(f(f(2)*eps)*eps);factor=f(f(1)+denom)
    last_factor=f(f(1)+f(eps*eps));power=f(1)
    for j in range(1,128):
        power=f(power*factor)
        value=f(f(f(power-f(1))/denom)*last_factor)
        representative=np.float16(value)
    word=int(np.array([representative],dtype='<f2').view('<u2')[0])
    return dict(status='estimated',value=float(representative),fp16_bits_hex=f'0x{word:04x}',fp16_bits_uint16=word,
        mee_eps8=.1,mee_eps8_float32=float(eps),table_index=254,source=ref(source),source_function='build_lookup_table_host',
        source_lines=[873,897],launcher_conversion='lut[8].chunk_max_mee is table[254] as FP16; launcher converts this stored half to float',
        method='Explicit NumPy float32 intermediates, host expression left-to-right multiplication, 127 sequential float32 power updates; each representative converted to float16 round-to-nearest-even. No Python-double recurrence oracle.',
        final_float32_representative=float(value),float32_factor=float(factor),error_bound=None,
        uncertainty='Not exported from the loaded host LUT. Compiler evaluation/reassociation equivalence is unverified and error bound unknown. Estimate only; cannot prove CUDA raw-scale invariance or justify scientific early stop.')

def describe(v,weights=None):
    v=np.asarray(v,dtype=np.float64);probs=[0,.01,.1,.25,.5,.75,.9,.99,1]
    if not len(v):return dict(count=0,mean=None,quantiles=None,energy_weighted_mean=None,energy_weighted_quantiles=None)
    result=dict(count=len(v),mean=float(v.mean()),quantiles={str(p):float(x) for p,x in zip(probs,np.quantile(v,probs))})
    if weights is not None:
        w=np.asarray(weights,dtype=np.float64);assert w.shape==v.shape and np.all(w>=0)
        total=float(w.sum());result['weight_sum']=total
        if total:
            ids=np.argsort(v,kind='stable');vs=v[ids];ws=w[ids];positive=ws>0;vs=vs[positive];ws=ws[positive]
            cumulative=np.cumsum(ws);targets=np.array(probs)*cumulative[-1]
            positions=np.minimum(np.searchsorted(cumulative,targets,side='left'),len(vs)-1)
            result.update(energy_weighted_mean=float(np.sum(v*w)/total),energy_weighted_quantiles={str(p):float(x) for p,x in zip(probs,vs[positions])})
        else:result.update(energy_weighted_mean=None,energy_weighted_quantiles=None)
    return result

def packet_scales(wire):
    assert len(wire)==875704
    blocks=np.frombuffer(wire,dtype=np.uint8).reshape(-1,274)
    groups=blocks[:,:272].reshape(-1,16,17)
    codes=groups[:,:,:16].reshape(-1).copy();scales=groups[:,:,16].reshape(-1).copy()
    words=blocks[:,272:].copy().reshape(-1).view('<u2')
    units=(words.astype('<u4')<<16).view('<f4')
    assert len(codes)==P and len(scales)==P//16 and len(units)==P//256
    assert np.all(codes!=255) and np.all(scales>=1) and np.all(np.isfinite(units)) and np.all(units>0)
    return scales,words.copy(),units.copy()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',required=True);args=ap.parse_args()
    run=(ROOT/args.run_dir).resolve();allowed=(ROOT/'results/epsilon_diagnostic_v1').resolve()
    assert run.is_relative_to(allowed) and not run.is_relative_to(BASE)
    torch.set_num_threads(4);start=time.perf_counter();began=now()
    seal=read(BASE/'integrity_manifest.json');sealed={r['path']:r for r in seal['files']}
    dependencies={}
    def verify(path):
        path=Path(path).resolve()
        if str(path) in dependencies:return dependencies[str(path)]
        r=ref(path)
        if path.is_relative_to(BASE):assert r['sha256']==sealed[path.relative_to(BASE).as_posix()]['sha256']
        dependencies[str(path)]=r;return r
    manifest_path=BASE/'input_manifest.json';contract_path=BASE/'experiment_contract.json';ledger_path=BASE/'packet_accounting.jsonl'
    for p in [manifest_path,contract_path,ledger_path,BASE/'source_manifest.json',BASE/'logs/official_build_a03.json',BASE/'metrics_raw.jsonl']:verify(p)
    contract=read(contract_path);assert sha(contract_path)=='97e8595532125d1fed44956c6a70da959031581884e753b449b64c1b3b195f47'
    assert contract['upstream_commit']==COMMIT and sha(manifest_path)==contract['input_manifest_sha256']
    source_manifest=read(BASE/'source_manifest.json');assert sha(BASE/'source_manifest.json')==contract['source_manifest_sha256']
    build=read(BASE/'logs/official_build_a03.json');assert build['official_commit']==COMMIT and build['status']=='compiled_and_loaded'
    verify(build['module_path']);assert sha(build['module_path'])==build['module_sha256']
    actual_src=Path(build['source_directory'])
    for name,digest in build['source_hashes'].items():verify(actual_src/name);assert sha(actual_src/name)==digest
    c8=c8_estimate(actual_src/'csrc/cuda_hadamard.cu')
    audit_path=ROOT/'reviews/bounded_discovery_v1_20261104_001_audit/saved_block_analysis.json'
    audit_code=ROOT/'reviews/bounded_discovery_v1_20261104_001_audit/analyze_saved_blocks.py'
    verify(audit_path);verify(audit_code);old_audit=read(audit_path);assert old_audit['raw_sha256']==sha(BASE/'metrics_raw.jsonl')
    audit_snapshots={r['snapshot_id']:r for r in old_audit['snapshots']}
    input_rows=read(manifest_path)['inputs'];assert len(input_rows)==12
    inputs={r['input_id']:r for r in input_rows};data={};per_input=[];array_rows=[];scales_by_input={};snapshot_energy={}
    for r in input_rows:
        assert r['training_seed']==1101 and r['update_step'] in (100,500,2500) and r['partition']=='development'
        assert r['already_divided_by_world_size'] and r['world_size']==4 and r['logical_numel']==D
        verify(r['fixture_path']);assert sha(r['fixture_path'])==r['fixture_sha256']
        verify(r['source_path']);assert sha(r['source_path'])==r['source_sha256']
        x=torch.load(r['fixture_path'],map_location='cpu',weights_only=False)['input_logical'].contiguous()
        assert x.dtype==torch.bfloat16 and x.shape==(D,) and raw_sha(x)==r['input_identity']['sha256']
        data[r['input_id']]=x
    accounts=rows(ledger_path)
    officials=sorted([r for r in accounts if r['method_id']==OFFICIAL],key=lambda r:(r['snapshot_id'],r['worker_id'],r['repeat_id']))
    uniform={(r['snapshot_id'],r['worker_id'],r['repeat_id']):r for r in accounts if r['method_id']==UNIFORM}
    assert len(officials)==len(uniform)==96
    pairs=[];packet_records=[];pool_records=[]
    for old in officials:
        key=(old['snapshot_id'],old['worker_id'],old['repeat_id']);u=uniform[key];inp=inputs[old['input_id']]
        assert old['logical_numel']==D and old['padded_numel']==P and old['packet_total_bytes']==875704
        assert old['input_manifest_sha256']==sha(manifest_path) and old['contract_sha256']==sha(contract_path)
        seed=old['seed'];assert seed==old['metadata']['seed']
        g=torch.Generator(device='cpu').manual_seed(seed);pool=torch.rand(P,dtype=torch.float32,generator=g).to(torch.bfloat16)
        pool_hash=raw_sha(pool);assert pool_hash==old['metadata']['pool_sha256']
        paths=[BASE/old['packet_path'],BASE/old['decoded_output_path'],BASE/u['packet_path'],BASE/u['decoded_output_path']]
        for p in paths:verify(p)
        assert sha(paths[0])==old['packet_sha256'] and sha(paths[1])==old['decoded_output_sha256']
        assert sha(paths[2])==u['packet_sha256'] and sha(paths[3])==u['decoded_output_sha256']
        for p,row in [(paths[1],old),(paths[3],u)]:
            y=torch.load(p,map_location='cpu',weights_only=True)
            assert y.dtype==torch.bfloat16 and y.shape==(D,) and raw_sha(y)==row['decoded_tensor_identity']['sha256']
        scale_codes,unit_bits,unit_values=packet_scales(paths[0].read_bytes())
        comp_hash=hashlib.sha256(scale_codes.tobytes()+unit_bits.tobytes()).hexdigest()
        previous=scales_by_input.get(old['input_id'])
        if previous is not None:assert comp_hash==previous[-1],'Input-only scales unexpectedly changed across RNG repeats'
        else:scales_by_input[old['input_id']]=(scale_codes,unit_bits,unit_values,comp_hash)
        pair_id=f"{old['input_id']}__r{old['repeat_id']:02d}"
        pair=dict(pair_id=pair_id,snapshot_id=old['snapshot_id'],worker_id=old['worker_id'],repeat_id=old['repeat_id'],input_id=old['input_id'],
            input_path=inp['fixture_path'],input_file_sha256=inp['fixture_sha256'],input_tensor_sha256=inp['input_identity']['sha256'],
            logical_numel=D,padded_numel=P,dtype='torch.bfloat16',already_divided_by_four=True,training_seed=1101,update_step=inp['update_step'],partition='development',
            input_loader="torch.load(input_path,map_location='cpu',weights_only=False)['input_logical']; no additional division or previous padding",
            original_method_id=OFFICIAL,original_seed=seed,original_pool_sha256=old['metadata']['pool_sha256'],regenerated_pool_sha256=pool_hash,
            original_packet_path=str(paths[0]),original_packet_sha256=old['packet_sha256'],original_packet_id=old['packet_id'],
            original_decoded_path=str(paths[1]),original_decoded_sha256=old['decoded_output_sha256'],original_decoded_tensor_sha256=old['decoded_tensor_identity']['sha256'],
            original_record_key=dict(record_type='local',snapshot_id=key[0],worker_id=key[1],method_id=OFFICIAL,repeat_id=key[2]),
            uniform_packet_or_output_reference=dict(method_id=UNIFORM,original_record_key=dict(record_type='local',snapshot_id=key[0],worker_id=key[1],method_id=UNIFORM,repeat_id=key[2]),
                packet_path=str(paths[2]),packet_sha256=u['packet_sha256'],decoded_path=str(paths[3]),decoded_sha256=u['decoded_output_sha256'],decoded_tensor_sha256=u['decoded_tensor_identity']['sha256'],
                packet_total_bytes=u['packet_total_bytes'],packet_bytes=u['packet_total_bytes'],reused=True),
            pool_regeneration_recipe='fresh torch.Generator(cpu).manual_seed(original_seed); torch.rand(P,float32)->BF16, retaining possible 1.0; CUDA upload is downstream',
            regenerated_pool_exact_one_count=int((pool==1).sum()),status='pass')
        pairs.append(pair)
        packet_records.append(dict(pair_id=pair_id,packet_path=str(paths[0]),packet_sha256=old['packet_sha256'],packet_bytes=len(paths[0].read_bytes()),
            scale_codes_sha256=hashlib.sha256(scale_codes.tobytes()).hexdigest(),super_units_raw_bytes_sha256=hashlib.sha256(unit_bits.tobytes()).hexdigest(),
            scale_code_min=int(scale_codes.min()),scale_code_max=int(scale_codes.max()),unit_min=float(unit_values.min()),unit_max=float(unit_values.max()),
            scale_code_count=P//16,super_scale_count=P//256,includes_padding=True,status='pass'))
    assert len({r['pair_id'] for r in pairs})==96 and len({r['original_seed'] for r in pairs})==96
    for inp in input_rows:
        iid=inp['input_id'];x=data[iid].double();v=x.numpy().reshape(-1,16)
        maxima=np.abs(v).max(1);energy=(v*v).sum(1);nonzero=maxima>0
        add=float(np.float32(1e-7));base=maxima/c8['value'];fraction=add/(base+add)
        scale_codes,unit_bits,unit_values,comp_hash=scales_by_input[iid]
        group_units=np.repeat(unit_values,16);stored=scale_codes.astype(np.float64)*group_units.astype(np.float64)
        padded=np.zeros(P,dtype=np.float64);padded[:D]=x.numpy();block_energy=padded.reshape(-1,256)**2;block_energy=block_energy.sum(1)
        if inp['snapshot_id'] not in snapshot_energy:snapshot_energy[inp['snapshot_id']]=block_energy.copy()
        else:snapshot_energy[inp['snapshot_id']]+=block_energy
        path=run/'logs'/f'e0_groups_{iid}.npz'
        np.savez_compressed(path,group_id=np.arange(len(maxima)),max_abs=maxima,input_energy=energy,
            estimated_base_scale=base,estimated_add_fraction=fraction,stored_scale_codes_padded=scale_codes,
            stored_bf16_unit_bits=unit_bits,stored_bf16_unit_values=unit_values,stored_scale_product_padded=stored,
            coordinate_block_input_energy=block_energy)
        array_rows.append(dict(input_id=iid,path=path.relative_to(run).as_posix(),sha256=sha(path),bytes=path.stat().st_size))
        item=dict(input_id=iid,snapshot_id=inp['snapshot_id'],worker_id=inp['worker_id'],logical_group_count=len(maxima),
            zero_group_count=int((~nonzero).sum()),nonzero_group_count=int(nonzero.sum()),zero_group_energy=float(energy[~nonzero].sum()),
            zero_group_add_fraction=1.0 if np.any(~nonzero) else None,padding_only_groups=(P-D)//16,input_energy=float(energy.sum()),
            max_abs_nonzero_groups=describe(maxima[nonzero],energy[nonzero]),input_energy_nonzero_groups=describe(energy[nonzero]),
            add_fraction_nonzero_groups=describe(fraction[nonzero],energy[nonzero]),
            stored_group_scale_code_distribution_padded=describe(scale_codes),stored_super_scale_unit_distribution_padded=describe(unit_values),
            stored_scale_product_distribution_logical_groups=describe(stored[:len(maxima)],energy),
            stored_scales_repeat_invariant=True,stored_scale_product_note='FP64 product of actual packet UINT8 group scale and BF16 super-unit; not the unquantized raw scale or a CPU reproduction of fast-math decoding.',
            raw_arrays=array_rows[-1])
        per_input.append(item)
    regions=[]
    for snapshot,en in sorted(snapshot_energy.items()):
        ids=np.lexsort((np.arange(len(en)),-en))[:32];total=float(en.sum());share=float(en[ids].sum()/total)
        prior=audit_snapshots[snapshot]
        assert prior['blocks']==len(en) and prior['top_1pct_energy_blocks']['count']==32
        assert math.isclose(total,prior['source_energy'],rel_tol=1e-12,abs_tol=1e-24)
        assert math.isclose(share,prior['top_1pct_energy_blocks']['input_energy_share'],rel_tol=1e-12,abs_tol=1e-15)
        regions.append(dict(snapshot_id=snapshot,block_ids=[int(i) for i in ids],blocks=[dict(block_id=int(i),start=int(i)*256,end=min((int(i)+1)*256,D),energy=float(en[i]),input_energy=float(en[i])) for i in ids],
            coordinate_interval_convention='half-open [start,end); crop to D',logical_coordinate_count=sum(min((int(i)+1)*256,D)-int(i)*256 for i in ids),
            logical_numel=D,worker_count=4,block_size=256,block_count=len(en),region_energy=float(en[ids].sum()),source_energy=total,input_energy_share=share,
            old_audit_energy_share=prior['top_1pct_energy_blocks']['input_energy_share'],old_audit_agreement=True,
            old_audit_ids_available=False,old_audit_note='Prior audit stored top32 count/energy share, not all IDs. IDs are independently recomputed with identical energy-only stable tie rule and frozen before intervention.'))
    record=dict(schema_version='epsilon_input_randomness_v1',run_id=run.name,created_at=now(),status='pass',base_run=str(BASE),
        base_run_id=BASE.name,base_contract_sha256=sha(contract_path),base_input_manifest_sha256=sha(manifest_path),base_packet_accounting_sha256=sha(ledger_path),
        selected_original_method_id=OFFICIAL,expected_pairs=96,pair_count=len(pairs),pool_hash_matches=96,input_hash_matches=96,
        conditions_share_identical_original_pool=True,uniform_uses_different_original_random_stream=True,pairs=pairs)
    dump(run/'input_randomness_manifest.json',record)
    dump(run/'region_manifest.json',dict(schema_version='epsilon_regions_v1',run_id=run.name,frozen_at=now(),status='pass',
        region_selection='Sum four workers FP64 input energy at shared 256-coordinate blocks; take top32 by descending energy and ascending block_id for ties; selection independent of errors/intervention',
        audit_source=ref(audit_path),audit_program=ref(audit_code),input_manifest_sha256=sha(manifest_path),regions=regions))
    all_max=np.concatenate([data[r['input_id']].double().numpy().reshape(-1,16).__abs__().max(1) for r in input_rows])
    all_energy=np.concatenate([(data[r['input_id']].double().numpy().reshape(-1,16)**2).sum(1) for r in input_rows]);nz=all_max>0
    add=float(np.float32(1e-7));all_fraction=add/(all_max/c8['value']+add)
    dump(run/'scale_precheck.json',dict(schema_version='epsilon_scale_precheck_v1',run_id=run.name,status='pass',created_at=now(),scope='CPU-only descriptive E0; no causal effect or invariant-arithmetic claim',
        C8=c8,scale_add_nominal=1e-7,scale_add_binary32=float(np.float32(1e-7)),mee_eps8_unchanged=.1,
        fraction_formula='float64 descriptive max_abs / estimated FP16 C8; binary32(1e-7) / (base_scale + binary32(1e-7)); not actual CUDA intermediate arithmetic',
        quantile_rule='Ordinary quantiles: NumPy linear interpolation. Energy-weighted quantiles: smallest observed value whose inclusive cumulative energy reaches p; p=0 smallest positive-weight value.',
        logical_group_count=len(all_max),zero_group_count=int((~nz).sum()),zero_groups_excluded_from_main_fraction_distribution=True,
        total_padding_only_groups=12*(P-D)//16,zero_group_energy=float(all_energy[~nz].sum()),
        all_nonzero_group_add_fraction=describe(all_fraction[nz],all_energy[nz]),all_nonzero_group_max_abs=describe(all_max[nz],all_energy[nz]),
        inputs=per_input,packet_scale_records=packet_records,
        no_early_stop_from_magnitude=True,interpretation='Small add_fraction does not prove no effect because downstream thresholds are discrete. Source/path qualification, not a magnitude cutoff, controls continuation.'))
    dump(run/'logs/e0_source_identity.json',dict(status='pass',base_commit=COMMIT,base_contract_sha256=sha(contract_path),
        actual_binary=verify(build['module_path']),base_seal_manifest_sha256=sha(BASE/'integrity_manifest.json'),external_references=list(dependencies.values()),
        actual_compiled_source_root=str(actual_src),all_referenced_base_files_match_seal=True,input_tensor_hashes_verified=12,
        official_packet_hashes_verified=96,official_decoded_hashes_verified=96,uniform_packet_hashes_verified=96,uniform_decoded_hashes_verified=96,
        note='Only required dependencies verified; old complete 3.57GB bundle not copied or rewritten.'))
    dump(run/'logs/e0_execution.json',dict(status='pass',started_at=began,finished_at=now(),wall_seconds=time.perf_counter()-start,
        active_elapsed_upper_bound_seconds=time.perf_counter()-start,precheck_cap_seconds=1800,cuda_executed=False,codec_executed=False,
        source_sha256=sha(Path(__file__)),command=[sys.executable,*sys.argv],pair_count=96,C8=c8['value'],C8_status='estimated',
        input_manifest_sha256=sha(run/'input_randomness_manifest.json'),region_manifest_sha256=sha(run/'region_manifest.json'),scale_precheck_sha256=sha(run/'scale_precheck.json')))
    print(json.dumps(dict(status='pass',pairs=96,C8=c8['value'],C8_bits=c8['fp16_bits_hex'],elapsed_seconds=time.perf_counter()-start,zero_groups=int((~nz).sum()),fraction_statistics=describe(all_fraction[nz],all_energy[nz])),allow_nan=False))
if __name__=='__main__':main()
