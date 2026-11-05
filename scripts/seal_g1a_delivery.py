"""Independent delivery audit and immutable file seal; no GPU execution."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def reject(value):raise ValueError('Nonfinite JSON token: '+value)
def finite(v):
    if isinstance(v,float):assert math.isfinite(v),'Nonfinite parsed number'
    elif isinstance(v,dict):
        for x in v.values():finite(x)
    elif isinstance(v,list):
        for x in v:finite(x)
    return v
def read(p):return finite(json.loads(Path(p).read_text(encoding='utf-8'),parse_constant=reject))
def rows(p):return [finite(json.loads(s,parse_constant=reject)) for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()]
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def quantile(values,q):
    values=sorted(values);at=(len(values)-1)*q;lo=math.floor(at);hi=math.ceil(at)
    return values[lo]+(values[hi]-values[lo])*(at-lo)

def audit(run):
    required=['REPORT.md','summary.json','environment.json','codec_contract.json','benchmark_config.json',
        'input_manifest.json','source_manifest.json','correctness.jsonl','benchmark_raw.jsonl',
        'benchmark_summary.jsonl','memory.jsonl','issues.jsonl','REPRODUCE.md','requirements.txt',
        'fixtures_manifest.json','execution_registration.json','correctness_acceptance.json','analysis.json']
    for name in required:assert (run/name).is_file(),name
    summary=read(run/'summary.json');registration=read(run/'execution_registration.json')
    report_source=summary['report_generator']
    assert sha(ROOT/report_source['path'])==report_source['sha256']
    for path,expected in report_source['source_sha256'].items():assert sha(run/path)==expected,path
    analysis=read(run/'analysis.json')
    assert analysis['torch_benchmark_validity']=='valid' and analysis['validation_error_count']==0
    assert analysis['trace']['status']=='analyzed' and analysis['trace']['kernel_count']>0
    assert sha(run/analysis['trace']['trace_path'])==analysis['trace']['sha256']
    for path,expected in analysis['source_sha256'].items():assert sha(run/path)==expected,path
    assert analysis['generator_sha256']==sha(ROOT/'scripts/summarize_g1a.py')
    assert len(analysis['figures'])>=4
    for path in analysis['figures']:assert (run/path).is_file() and (run/path).stat().st_size>0,path
    contract=sha(run/'codec_contract.json');config_hash=sha(run/'benchmark_config.json')
    assert summary['contract_sha256']==registration['contract_sha256']==contract
    assert summary['benchmark_config_sha256']==config_hash
    for key in ('execution_status','correctness_status','benchmark_validity','g1_acceptance'):assert summary[key]=='incomplete',key
    assert summary['stage']=='G1' and summary['substage']=='torch_cuda_baseline'
    assert summary['optimization_gain']=='not_evaluated'
    for key in ('full_dynamiq_gpu_validated','training_quality_newly_validated','multigpu_speedup_measured'):assert summary['claims'][key] is False
    toolchain=read(run/'logs/toolchain_summary.json')
    assert toolchain['status']=='pass' and toolchain['gpu_call']['module_called'] is True and toolchain['gpu_call']['output_equal'] is True
    for path,expected in toolchain['evidence_sha256'].items():assert sha(run/path)==expected,path
    for name,expected in registration['source_hashes'].items():
        assert sha(ROOT/name)==expected and sha(run/'code'/name)==expected,name
    assert sha(run/'input_manifest.json')==registration['input_manifest_sha256']
    for item in read(run/'source_manifest.json')['files']:
        source=run/item['path'];assert sha(source)==item['sha256'] and source.stat().st_size==item['bytes'],str(source)
        if item.get('original_path'):assert sha(item['original_path'])==item['sha256'],item['original_path']
    for item in read(run/'fixtures_manifest.json')['files']:
        f=Path(item['path']);assert sha(f)==item['sha256'] and f.stat().st_size==item['bytes'],str(f)
    correct=rows(run/'correctness.jsonl');accept=read(run/'correctness_acceptance.json')
    assert correct and accept['method_status']=='pass' and accept['contract_sha256']==contract
    assert accept['deterministic_algorithms'] is True and accept['fill_uninitialized_memory'] is False and accept['execution_registration_verified'] is True
    assert accept['correctness_sha256']==sha(run/'correctness.jsonl')
    assert dict(Counter(r['status'] for r in correct))=={'pass':len(correct)}
    assert all(r['contract_sha256']==contract and r['run_id']==run.name for r in correct)
    for row in correct:
        for key in ('code_mismatch_count','scale_code_mismatch_count','top_scale_bit_mismatch_count','packet_byte_mismatch_count','decoded_bit_mismatch_count'):
            assert row[key] in (None,0),(row['case_id'],key)
    inputs=read(run/'input_manifest.json')['inputs'];by_input={r['input_id']:r for r in inputs}
    assert len(inputs)==len(by_input) and len([r for r in inputs if r['source_kind']=='real_gradient'])==12
    protocol=[r for r in correct if r['input_id'] not in by_input]
    assert len(protocol)==117
    assert dict(Counter(r['check_type'] for r in protocol))=={'invalid_input':60,'decode':39,'pack':10,'encode':4,'roundtrip':4}
    assert all(r.get('expected_rejection') for r in protocol if r['check_type']=='invalid_input')
    for row in inputs:
        for bits in (4,8):
            for check in ('encode','decode','roundtrip','repeatability','input_immutability'):
                found=[r for r in correct if r['input_id']==row['input_id'] and r['nominal_bits']==bits and r['check_type']==check and r['method_id']=='torch_cuda_repaired']
                assert len(found)==1,(row['input_id'],bits,check)
    raw=rows(run/'benchmark_raw.jsonl');memory=rows(run/'memory.jsonl');stats=rows(run/'benchmark_summary.jsonl')
    assert len(raw)==10400 and len(memory)==520 and len(stats)==104
    config=read(run/'benchmark_config.json');K={r['case_id']:r['inner_repeats'] for r in config['repeat_counts']}
    assert len(K)==104
    bycase=defaultdict(list);seen=set()
    for r in raw:
        key=tuple(r[k] for k in ('case_id','method_id','session_id','block_id'))
        assert key not in seen,key
        seen.add(key);bycase[r['case_id']].append(r)
        assert r['run_id']==run.name and r['contract_sha256']==contract and r['benchmark_config_sha256']==config_hash
        assert r['method_id']=='torch_cuda_repaired' and r['record_status']=='ok' and r['exclusion_reason'] is None
        assert r['inner_repeats']==K[r['case_id']] and 1<=r['inner_repeats']<=10000
        assert r['gpu_elapsed_ms']>0 and r['wall_elapsed_ms']>0
        assert r['rng_work_included'] is False and r['graph_mode']=='off' and r['compile_mode']=='eager'
        n=r['original_numel'];L=(n+255)//256*256;b=r['nominal_bits']
        assert r['padded_numel']==L and r['stored_bits']==b+1
        assert r['wire_bytes']==(L*(b+1)+7)//8+L//16+2*(L//256)
        assert r['wire_bytes']==sum(r[k] for k in ('payload_bytes','scale_bytes','header_bytes','alignment_bytes'))
    assert set(bycase)==set(K)=={r['case_id'] for r in stats} and len({r['case_id'] for r in stats})==len(stats)
    assert set(bycase)=={r['case_id'] for r in memory}
    for item in stats:
        group=bycase[item['case_id']];assert len(group)==100
        medians=[];wall_medians=[];noise=[]
        for sid in range(5):
            session=[r for r in group if r['session_id']==sid]
            assert len(session)==20 and {r['block_id'] for r in session}==set(range(20))
            event=[r['gpu_elapsed_ms']*1000/r['inner_repeats'] for r in session]
            wall=[r['wall_elapsed_ms']*1000/r['inner_repeats'] for r in session]
            medians.append(statistics.median(event));wall_medians.append(statistics.median(wall))
            saved=next(s for s in item['session_statistics'] if s['session_id']==sid)
            for name,q in (('p25',.25),('median',.5),('p75',.75),('p95',.95)):
                assert math.isclose(saved['gpu_'+name+'_us'],quantile(event,q),rel_tol=1e-12)
            assert math.isclose(saved['wall_median_us'],statistics.median(wall),rel_tol=1e-12)
            noise.append((quantile(event,.75)-quantile(event,.25))/statistics.median(event)>.1)
            assert saved['noise_flag']==noise[-1]
        assert math.isclose(item['gpu_session_median_us_median'],statistics.median(medians),rel_tol=1e-12)
        assert item['gpu_session_median_us_min']==min(medians) and item['gpu_session_median_us_max']==max(medians)
        assert math.isclose(item['wall_session_median_us_median'],statistics.median(wall_medians),rel_tol=1e-12)
        assert item['noise_flag']==any(noise)
        assert math.isclose(item['input_equivalent_gbps'],group[0]['original_numel']*2/(statistics.median(medians)*1e-6)/1e9,rel_tol=1e-12)
        assert item['valid_session_count']==5 and item['valid_block_count']==100
        assert item['paired_reference_method'] is None and not item['paired_speedup_by_session']
    memory_keys=set()
    for item in memory:
        key=(item['case_id'],item['method_id'],item['session_id']);assert key not in memory_keys;memory_keys.add(key)
        assert item['run_id']==run.name and item['contract_sha256']==contract and item['benchmark_config_sha256']==config_hash
        assert item['peak_extra_allocated_bytes']==item['peak_allocated_bytes']-item['allocated_before_bytes']>=0
        assert item['workspace_bytes']>=0
    assert memory_keys=={(c,'torch_cuda_repaired',s) for c in K for s in range(5)}
    trace=read(run/'traces/manifest.json');assert sha(run/'traces'/trace['trace'])==trace['sha256']
    old=ROOT/'results/gpu_codec_v1/gpu_codec_v1_20261101_001'
    oldseal=read(old/'artifact_integrity.json')
    for item in oldseal['files']:assert sha(old/item['path'])==item['sha256'] and (old/item['path']).stat().st_size==item['bytes']
    for item in read(old/'source_manifest.json')['files']:
        if item.get('original_path'):assert sha(item['original_path'])==item['sha256'],item['original_path']
    for p in run.rglob('*.json'):read(p)
    for p in run.rglob('*.jsonl'):rows(p)
    return {'status':'pass','verified_at':datetime.now().astimezone().isoformat(),'run_id':run.name,
        'command':[sys.executable,*sys.argv],
        'required_files':len(required),'correctness_rows':len(correct),'inputs':len(inputs),'benchmark_cases':104,
        'raw_blocks':len(raw),'sessions_per_case':5,'blocks_per_session':20,'memory_rows':len(memory),
        'contract_sha256':contract,'benchmark_config_sha256':config_hash,
        'independent_summary_median_recalculation':'all_match','old_run_sealed_files_unchanged':len(oldseal['files']),
        'old_source_originals_unchanged':True,'all_JSON_strict':True,'full_G1':'incomplete_by_design_optimized_method_missing'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--seal',action='store_true');args=p.parse_args()
    run=(ROOT/args.run_dir).resolve()
    if not run.is_relative_to(ROOT/'results/gpu_codec_v1'):raise ValueError('Invalid run')
    if (run/'artifact_integrity.json').exists():raise FileExistsError('Already sealed; do not alter closed attempts')
    result=audit(run);write(run/'logs/delivery_validation.json',result)
    if args.seal:
        manifest={'schema_version':'gpu_codec_v1.0','run_id':run.name,'created_at':datetime.now().astimezone().isoformat(),
            'status':'sealed_G1a_intermediate_full_G1_incomplete','files':[]}
        for path in sorted(run.rglob('*')):
            if path.is_file():manifest['files'].append({'path':path.relative_to(run).as_posix(),'sha256':sha(path),'bytes':path.stat().st_size})
        write(run/'artifact_integrity.json',manifest)
        result.update(sealed_files=len(manifest['files']),artifact_integrity_sha256=sha(run/'artifact_integrity.json'))
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
