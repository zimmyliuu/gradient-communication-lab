"""Independently recompute G1b evidence and seal only a complete delivery."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
METHODS=('torch_cuda_repaired','cuda_optimized_repaired')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def reject(v): raise ValueError('Nonfinite JSON token: '+v)
def finite(v):
    if isinstance(v,float): assert math.isfinite(v)
    elif isinstance(v,dict):
        for x in v.values(): finite(x)
    elif isinstance(v,list):
        for x in v: finite(x)
    return v
def read(p): return finite(json.loads(Path(p).read_text(encoding='utf-8'),parse_constant=reject))
def rows(p): return [finite(json.loads(s,parse_constant=reject)) for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()]
def write(p,v): Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def now(): return datetime.now(timezone(timedelta(hours=8))).isoformat()
def quantile(values,q):
    values=sorted(values); at=(len(values)-1)*q; lo=math.floor(at); hi=math.ceil(at)
    return values[lo]+(values[hi]-values[lo])*(at-lo)
def equal(a,b): assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),(a,b)

def expected_benchmark(inputs):
    expected={}
    def add(inp,bits,operation,allocation,validation):
        case=f"{inp['input_id']}__b{bits}__{operation}__{allocation}__{validation}"
        for method in METHODS:
            expected[(case,method)]={'input_id':inp['input_id'],'nominal_bits':bits,'operation':operation,
                'allocation_mode':allocation,'validation_mode':validation,'original_numel':inp['original_numel'],
                'padded_numel':inp['padded_numel']}
    for inp in inputs:
        if inp['source_kind']=='synthetic' and inp['generator_config']['seed']==7301:
            for bits in (4,8):
                for operation in ('encode','decode','roundtrip'):add(inp,bits,operation,'preallocated','trusted_inputs')
                if inp['generator_config']['distribution']=='standard_normal' and inp['original_numel'] in (65536,1048576):
                    for validation in ('trusted_inputs','checked_api'):add(inp,bits,'roundtrip','api_allocating',validation)
        elif inp['source_kind']=='real_gradient':
            for bits in (4,8):add(inp,bits,'roundtrip','preallocated','trusted_inputs')
    assert len(expected)==208
    return expected

def audit_design(run,config,expected,K,registration):
    protocol=read(run/'pilot_protocol.json');design=read(run/'logs/benchmark_design_registration.json')
    assert sha(run/'pilot_protocol.json')==design['protocol_sha256']==config['pilot_protocol_sha256']
    assert sha(run/'logs/benchmark_design_registration.json')==config['design_registration_sha256']
    assert protocol['benchmark_source_sha256']==registration['source_hashes']['scripts/benchmark_cuda_pair.py']
    assert protocol['contract_sha256']==registration['contract_sha256'] and protocol['input_manifest_sha256']==registration['input_manifest_sha256']
    assert protocol['methods']==list(METHODS) and protocol['case_count']==104 and protocol['session_count']==5 and protocol['blocks_per_session']==20
    assert protocol['warmup_calls']==50 and config['warmup_calls_min']==50
    assert config['K_selection']==protocol['pilot_rule'] and protocol['pilot_rule']['max_actual_block_observations']==4
    assert protocol['pilot_rule']['accept_interval_ms']==[15,30] and protocol['pilot_rule']['target_ms']==20
    assert sha(run/'logs/K_pilot.jsonl')==config['pilot_sha256']
    assert sha(run/'logs/pilot_order.json')==design['pilot_order_sha256']
    pilot=rows(run/'logs/K_pilot.jsonl');order=read(run/'logs/pilot_order.json')
    assert len(pilot)==208 and len(order)==208
    assert [(r['case_id'],r['method_id']) for r in pilot]==[(r['case_id'],r['method_id']) for r in order]
    assert {(r['case_id'],r['method_id']) for r in pilot}==set(expected)
    for r in pilot:
        singles=r['single_call_pilots'];assert len(singles)==3
        assert all(x['gpu_elapsed_ms']>0 and x['wall_elapsed_ms']>0 for x in singles)
        median=statistics.median(x['gpu_elapsed_ms'] for x in singles);equal(median,r['initial_selection_median_ms'])
        k=max(1,min(10000,math.ceil(20/max(median,1e-9))))
        observed=r['actual_block_observations'];assert 1<=len(observed)<=4
        for index,item in enumerate(observed):
            assert item['inner_repeats']==k and item['gpu_elapsed_ms']>0 and item['wall_elapsed_ms']>0
            elapsed=item['gpu_elapsed_ms'];next_k=max(1,min(10000,math.ceil(k*20/max(elapsed,1e-9))))
            stop=15<=elapsed<=30 or next_k==k or index==3
            if index<len(observed)-1:assert not stop;k=next_k
            else:assert stop
        assert r['inner_repeats']==k==K[(r['case_id'],r['method_id'])]
    all_cases={case for case,method in expected};pair_orders={}
    for session in range(5):
        path=f'logs/benchmark_session_{session}_order.json';data=read(run/path)
        assert sha(run/path)==design['order_sha256'][path]
        assert data['session_id']==session and len(data['case_order'])==104 and set(data['case_order'])==all_cases
        assert len(data['paired_orders'])==104 and [r['case_id'] for r in data['paired_orders']]==data['case_order']
        for item in data['paired_orders']:
            assert Counter(tuple(x) for x in item['orders'])==Counter({METHODS:10,tuple(reversed(METHODS)):10})
            for block,order in enumerate(item['orders']):pair_orders[(item['case_id'],session,block)]=order
    return pair_orders

def audit(run):
    required=['REPORT.md','summary.json','environment.json','codec_contract.json','benchmark_config.json',
        'input_manifest.json','source_manifest.json','correctness.jsonl','benchmark_raw.jsonl',
        'benchmark_summary.jsonl','memory.jsonl','issues.jsonl','REPRODUCE.md','requirements.txt',
        'fixtures_manifest.json','execution_registration.json','correctness_acceptance.json','analysis.json',
        'pilot_protocol.json','logs/cuda_build_manifest.json','traces/g1b_trace_index.json']
    for p in required: assert (run/p).is_file(),p
    summary=read(run/'summary.json');registration=read(run/'execution_registration.json')
    assert summary['execution_status']=='complete'
    assert summary['correctness_status']=='pass' and summary['benchmark_validity']=='valid' and summary['g1_acceptance']=='pass'
    for key in ('full_dynamiq_gpu_validated','training_quality_newly_validated','multigpu_speedup_measured'):
        assert summary['claims'][key] is False,key
    contract=sha(run/'codec_contract.json');confighash=sha(run/'benchmark_config.json')
    assert summary['contract_sha256']==registration['contract_sha256']==contract
    assert summary['benchmark_config_sha256']==confighash
    assert sha(run/'input_manifest.json')==registration['input_manifest_sha256']
    assert sha(run/'correctness_case_manifest.json')==registration['correctness_case_manifest_sha256']
    assert sha(run/'logs/cuda_build_manifest.json')==registration['optimized_build_manifest_sha256']
    build=read(run/'logs/cuda_build_manifest.json')
    assert sha(build['module_path'])==build['module_sha256']
    for name,expected in registration['source_hashes'].items():
        assert sha(ROOT/name)==expected and sha(run/'code'/name)==expected,name
    for item in read(run/'source_manifest.json')['files']:
        p=run/item['path'];assert sha(p)==item['sha256'] and p.stat().st_size==item['bytes'],str(p)
        if item.get('original_path'): assert sha(item['original_path'])==item['sha256'],item['original_path']
    for item in read(run/'fixtures_manifest.json')['files']:
        p=Path(item['path']);assert sha(p)==item['sha256'] and p.stat().st_size==item['bytes'],str(p)
    analysis=read(run/'analysis.json');assert analysis['validation_error_count']==0
    assert analysis['timing_validity']=='valid' and analysis['benchmark_validity']=='valid' and analysis['trace_completeness']=='complete'
    assert analysis['generator_sha256']==sha(ROOT/'scripts/summarize_g1b.py')
    for p,h in analysis['source_sha256'].items():assert sha(run/p)==h,p
    assert len(analysis['figures'])>=4
    for p in analysis['figures']:assert (run/p).stat().st_size>0,p
    generator=summary['report_generator']; assert sha(ROOT/generator['path'])==generator['sha256']
    for p,h in generator['source_sha256'].items():assert sha(run/p)==h,p
    correct=rows(run/'correctness.jsonl');accept=read(run/'correctness_acceptance.json')
    pytest_record=read(run/'logs/correctness_development_attempts.json')['pytest']
    junit=ET.parse(run/pytest_record['path']).getroot()
    cases=list(junit.iter('testcase'))
    assert len(cases)==pytest_record['tests'] and pytest_record['failures']==0
    assert not any(list(case.iter('failure')) or list(case.iter('error')) or list(case.iter('skipped')) for case in cases)
    assert accept['method_status']=='pass' and accept['contract_sha256']==contract
    assert accept['execution_registration_verified'] and accept['deterministic_algorithms'] and not accept['fill_uninitialized_memory']
    assert accept['correctness_sha256']==sha(run/'correctness.jsonl')
    loaded=accept['start_identity']['loaded_build']
    assert loaded['module_path']==build['module_path'] and loaded['module_sha256']==build['module_sha256']
    assert loaded['manifest']==build
    toolchain=read(run/'logs/toolchain_summary.json');assert toolchain['status']=='pass'
    for item in toolchain['evidence']:
        assert sha(run/item['path'])==item['sha256'],item['path']
    assert Counter(r['status'] for r in correct)=={'pass':len(correct)}
    assert all(r['run_id']==run.name and r['contract_sha256']==contract for r in correct)
    seen=set()
    for r in correct:
        k=(r['case_id'],r['method_id']); assert k not in seen,k;seen.add(k)
        for key in ('code_mismatch_count','scale_code_mismatch_count','top_scale_bit_mismatch_count','packet_byte_mismatch_count','decoded_bit_mismatch_count'):
            assert r.get(key) in (None,0),(k,key)
    case_manifest=read(run/'correctness_case_manifest.json');expected_checks=case_manifest['expected_cases']
    assert seen=={(r['case_id'],r['method_id']) for r in expected_checks}
    expected_counts=Counter(r['method_id'] for r in expected_checks)
    assert Counter(r['method_id'] for r in correct)==expected_counts
    assert accept['counts'].get('pass')==len(correct) and not accept['counts'].get('fail',0) and not accept['counts'].get('skipped',0)
    for method,count in expected_counts.items():
        status=accept['method_statuses'][method]
        assert status['correctness_status']=='pass' and status['counts'].get('pass')==count and not status['counts'].get('fail',0) and not status['counts'].get('skipped',0)
    rule_path=ROOT/case_manifest['rules_path'];assert sha(rule_path)==case_manifest['rules_sha256']
    inherited={(r['input_id'],r['nominal_bits']):{'exception_type':r['exception_type'],'message':r['message']} for r in read(rule_path)['rejection_rules']}
    abi={'abi_input_cpu':'input: expected contiguous tensor on cuda:0','abi_lut_short':'LUT: unexpected tensor dtype or shape',
        'abi_threshold_dtype':'main thresholds: unexpected tensor dtype or shape','abi_packet_wrong_identity':'Packet identity differs from plan',
        'abi_packet_wrong_type':'Expected a DevicePacket','abi_packet_output_alias':'packet payload: aliases another plan-owned buffer',
        'abi_scale_threshold_nan':'Invalid explicit thresholds'}
    for r in correct:
        if r['check_type']!='invalid_input':continue
        rule=r['expected_rejection_rule'];observed=r['observed_rejection']
        assert observed['exception_type']==rule['exception_type'] and observed['primary_message']==rule['message']
        key=(r['input_id'],r['nominal_bits'])
        if key in inherited:assert rule==inherited[key]
        elif r['input_id'] in abi:assert rule=={'exception_type':'ValueError','message':abi[r['input_id']]}
        else:
            assert r['method_id']==METHODS[1] and rule==case_manifest['native_rejection_rules'][r['input_id']]
            assert rule['exception_type']=='RuntimeError' and rule['message'].startswith(('ABI_METADATA: ','ABI_BITS: ','ABI_ALIAS: '))
    assert not accept['missing_coverage'] and not accept['unexpected_coverage'] and not accept['duplicate_coverage']
    inputs=read(run/'input_manifest.json')['inputs']; assert len(inputs)==82
    assert sum(r['source_kind']=='real_gradient' for r in inputs)==12
    for inp in inputs:
        for b in (4,8):
            for method in METHODS:
                for check in ('encode','decode','roundtrip','repeatability','input_immutability'):
                    found=[r for r in correct if r['input_id']==inp['input_id'] and r['nominal_bits']==b and r['check_type']==check and r['method_id']==method]
                    assert len(found)==1,(inp['input_id'],b,method,check)
            quant=[r for r in correct if r['input_id']==inp['input_id'] and r['nominal_bits']==b and r['check_type']=='roundtrip']
            assert {r['method_id'] for r in quant}=={*METHODS,'cpu_ref_repaired'}
            for key in ('quant_squared_error','quant_mse','quant_input_energy','quant_normalized_squared_error','quant_max_abs_error'):
                assert all(r[key]==quant[0][key] for r in quant),(inp['input_id'],b,key)
    raw=rows(run/'benchmark_raw.jsonl');mem=rows(run/'memory.jsonl');stats=rows(run/'benchmark_summary.jsonl')
    assert len(raw)==20800 and len(mem)==1040 and len(stats)==208
    config=read(run/'benchmark_config.json');K={(r['case_id'],r['method_id']):r['inner_repeats'] for r in config['repeat_counts']}
    expected_grid=expected_benchmark(inputs)
    assert len(config['repeat_counts'])==len(K)==208 and set(K)==set(expected_grid)
    pair_orders=audit_design(run,config,expected_grid,K,registration)
    by=defaultdict(list);pairs=defaultdict(list);seen=set()
    for r in raw:
        key=tuple(r[k] for k in ('case_id','method_id','session_id','block_id'))
        assert key not in seen,key;seen.add(key)
        ck=key[:2];by[ck].append(r);pairs[(r['case_id'],r['session_id'],r['block_id'])].append(r)
        assert all(r[name]==value for name,value in expected_grid[ck].items())
        assert r['run_id']==run.name and r['contract_sha256']==contract and r['benchmark_config_sha256']==confighash
        assert r['method_id'] in METHODS and r['record_status']=='ok' and r['exclusion_reason'] is None
        assert r['inner_repeats']==K[ck] and 1<=r['inner_repeats']<=10000
        assert r['gpu_elapsed_ms']>0 and r['wall_elapsed_ms']>0
        assert r['rng_work_included'] is False and r['graph_mode']=='off' and r['compile_mode']=='eager'
        n=r['original_numel'];L=(n+255)//256*256;b=r['nominal_bits']
        assert r['padded_numel']==L and r['stored_bits']==b+1
        assert r['wire_bytes']==(L*(b+1)+7)//8+L//16+2*(L//256)
        assert r['wire_bytes']==sum(r[k] for k in ('payload_bytes','scale_bytes','header_bytes','alignment_bytes'))
    assert set(by)==set(K)=={(r['case_id'],r['method_id']) for r in stats}
    assert len(pairs)==10400 and all(len(p)==2 and {r['method_id'] for r in p}==set(METHODS) for p in pairs.values())
    assert set(pairs)==set(pair_orders)
    for (case,session,block),pair in pairs.items():
        order=pair_orders[(case,session,block)];assert [r['method_id'] for r in pair]==order
        for position,r in enumerate(pair):
            assert r['pair_order']==order and r['pair_position']==position and r['paired_block_id']==f'{case}__s{session}__p{block:02d}'
    recomputed={}
    for item in stats:
        ck=(item['case_id'],item['method_id']);group=by[ck];assert len(group)==100
        medians=[];walls=[];noise=[]
        for sid in range(5):
            session=[r for r in group if r['session_id']==sid]
            assert len(session)==20 and {r['block_id'] for r in session}==set(range(20))
            event=[r['gpu_elapsed_ms']*1000/r['inner_repeats'] for r in session]
            wall=[r['wall_elapsed_ms']*1000/r['inner_repeats'] for r in session]
            medians.append(statistics.median(event));walls.append(statistics.median(wall))
            saved=next(s for s in item['session_statistics'] if s['session_id']==sid)
            for name,q in (('p25',.25),('median',.5),('p75',.75),('p95',.95)):equal(saved['gpu_'+name+'_us'],quantile(event,q))
            equal(saved['wall_median_us'],statistics.median(wall))
            noise.append((quantile(event,.75)-quantile(event,.25))/statistics.median(event)>.1)
            assert saved['noise_flag']==noise[-1]
            recomputed[(ck[0],ck[1],sid)]=medians[-1]
        equal(item['gpu_session_median_us_median'],statistics.median(medians))
        equal(item['gpu_session_median_us_min'],min(medians));equal(item['gpu_session_median_us_max'],max(medians))
        equal(item['session_median_max_over_min'],max(medians)/min(medians))
        equal(item['wall_session_median_us_median'],statistics.median(walls))
        assert item['noise_flag']==any(noise)
        equal(item['input_equivalent_gbps'],group[0]['original_numel']*2/(statistics.median(medians)*1e-6)/1e9)
        assert item['valid_session_count']==5 and item['valid_block_count']==100
    for item in stats:
        if item['method_id']==METHODS[1]:
            assert item['paired_reference_method']==METHODS[0]
            paired=item['paired_speedup_by_session'];assert len(paired)==5
            for entry in paired:
                sid=entry['session_id'];expected=recomputed[(item['case_id'],METHODS[0],sid)]/recomputed[(item['case_id'],METHODS[1],sid)]
                equal(entry['speedup'],expected)
    mk=set()
    for item in mem:
        key=(item['case_id'],item['method_id'],item['session_id']);assert key not in mk;mk.add(key)
        assert item['run_id']==run.name and item['contract_sha256']==contract and item['benchmark_config_sha256']==confighash
        base=expected_grid[key[:2]];assert all(item[name]==value for name,value in base.items())
        names=('input_bytes','threshold_bytes','lut_bytes','packet_buffer_bytes','decoded_output_bytes','workspace_bytes',
            'total_plan_owned_storage_bytes','total_named_device_storage_bytes','allocated_before_bytes','allocated_after_bytes',
            'peak_allocated_bytes','reserved_before_bytes','peak_reserved_bytes','peak_extra_allocated_bytes')
        assert all(type(item[name]) is int and item[name]>=0 for name in names)
        assert item['peak_extra_allocated_bytes']==item['peak_allocated_bytes']-item['allocated_before_bytes']>=0
        assert item['peak_allocated_bytes']>=item['allocated_after_bytes'] and item['peak_reserved_bytes']>=item['reserved_before_bytes']
        n,L,b=base['original_numel'],base['padded_numel'],base['nominal_bits']
        assert item['input_bytes']==n*2 and item['threshold_bytes']==2*L+L//8 and item['lut_bytes']==(2*(1<<b)+1)*2
        assert item['packet_buffer_bytes']==(L*(b+1)+7)//8+L//16+2*(L//256) and item['decoded_output_bytes']==2*L
        assert item['total_plan_owned_storage_bytes']==item['packet_buffer_bytes']+item['decoded_output_bytes']+item['workspace_bytes']
        assert item['total_named_device_storage_bytes']==item['total_plan_owned_storage_bytes']+item['input_bytes']+item['threshold_bytes']+item['lut_bytes']
    assert mk=={(c,m,s) for c,m in K for s in range(5)}
    traces=read(run/'traces/g1b_trace_index.json');assert traces['status']=='pass' and len(traces['traces'])==8
    expected={(m,b,n) for m in METHODS for b in (4,8) for n in (65536,4194304)}
    observed=set()
    for entry in traces['traces']:
        manifest=read(run/entry['manifest_path']);assert sha(run/entry['manifest_path'])==entry['manifest_sha256']
        observed.add((manifest['method_id'],manifest['nominal_bits'],manifest['original_numel']))
        assert manifest['contract_sha256']==contract and manifest['benchmark_config_sha256']==confighash
        assert sha(run/'traces'/manifest['trace'])==manifest['sha256']
        for item in manifest['artifacts']:
            p=run/item['path'];assert sha(p)==item['sha256'] and p.stat().st_size==item['bytes'],str(p)
        audit=read(run/manifest['audit_path']);assert audit['status']=='pass' and audit['chrome_records_exactly_match_measured_raw']
        assert audit['kernel_count']>0 and audit['dropped_records']==0 and not audit['collection_errors']
        assert set(audit['kernels_per_operation'])=={'0','1','2'} and all(v>0 for v in audit['kernels_per_operation'].values())
        assert audit['kernel_count']==sum(audit['kernels_per_operation'].values()) and audit['conflicting_external_correlations']==0
        assert audit['measured_invalid_or_outside_window_records']==0
    assert observed==expected
    old=ROOT/'results/gpu_codec_v1/gpu_codec_v1_20261101_002';oldseal=read(old/'artifact_integrity.json')
    for item in oldseal['files']:
        p=old/item['path'];assert sha(p)==item['sha256'] and p.stat().st_size==item['bytes'],str(p)
    for item in read(old/'source_manifest.json')['files']:
        if item.get('original_path'):assert sha(item['original_path'])==item['sha256'],item['original_path']
    for p in run.rglob('*.json'):read(p)
    for p in run.rglob('*.jsonl'):rows(p)
    return {'status':'pass','verified_at':now(),'run_id':run.name,'command':[sys.executable,*sys.argv],
        'correctness_rows':len(correct),'inputs':len(inputs),'benchmark_cases':104,'method_case_rows':208,
        'raw_blocks':len(raw),'pair_blocks':len(pairs),'memory_rows':len(mem),'GPU_traces':8,
        'contract_sha256':contract,'benchmark_config_sha256':confighash,
        'independent_quantiles_and_paired_speedups':'all_match','old_run_sealed_files_unchanged':len(oldseal['files']),
        'old_source_originals_unchanged':True,'all_JSON_strict':True,'full_G1':'pass'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--seal',action='store_true');a=p.parse_args()
    run=(ROOT/a.run_dir).resolve();assert run.is_relative_to(ROOT/'results/gpu_codec_v1')
    if (run/'artifact_integrity.json').exists():raise FileExistsError('Already sealed')
    result=audit(run);write(run/'logs/delivery_validation.json',result)
    if a.seal:
        manifest={'schema_version':'gpu_codec_v1.0','run_id':run.name,'created_at':now(),'status':'sealed_G1_complete','files':[]}
        for path in sorted(run.rglob('*')):
            if path.is_file():manifest['files'].append({'path':path.relative_to(run).as_posix(),'sha256':sha(path),'bytes':path.stat().st_size})
        write(run/'artifact_integrity.json',manifest)
        result.update(sealed_files=len(manifest['files']),artifact_integrity_sha256=sha(run/'artifact_integrity.json'))
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
