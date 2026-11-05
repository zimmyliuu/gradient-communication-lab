"""Select D1 configurations from actual packet sizes only, then freeze identity."""
import argparse,sys,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
import torch
from gradient_comm_lab.bounded_discovery import official_codec as official,uniform_codec as uniform
from gradient_comm_lab.bounded_discovery.common import *

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',required=True);a=ap.parse_args();run=(ROOT/a.run_dir).resolve()
    qualification=read(run/'baseline_qualification.json')
    assert qualification['gate_decision']=='pass', 'D0 must pass before any comparison'
    assert not (run/'experiment_contract.json').exists()
    official.set_build_manifest(run/'logs/official_build_a03.json')
    torch.set_num_threads(4)
    d=818048;zeros=torch.zeros(d,dtype=torch.bfloat16);budgets={};budget_rows=[]
    torch.cuda.synchronize();start=time.perf_counter()
    for b in (4,8):
        p=official.encode(zeros.cuda(),b,seed=seed_for('budget_only',f'official_b{b}',0))
        account=p.accounting();wire=p.to_bytes();assert len(wire)==account['packet_total_bytes']
        budgets[b]=len(wire);budget_rows.append(dict(method=f'official_b{b}',**account,packet_sha256=hashlib.sha256(wire).hexdigest()))
    torch.cuda.synchronize();gpu_seconds=time.perf_counter()-start
    measured={}
    for g in (16,128):
        for q in range(2,11):
            p=uniform.encode(zeros,q,g,seed_for('budget_only',f'uniform_g{g}_q{q}',0))
            measured[g,q]=p.accounting()['packet_total_bytes'];budget_rows.append(dict(method=f'uniform_g{g}_q{q}',**p.accounting()))
    selected={};selection=[]
    for label,b in [('low',4),('high',8)]:
        for g in (16,128):
            eligible=[q for q in range(2,11) if measured[g,q]<=budgets[b]]
            minus=max(eligible) if eligible else None
            selection.append(dict(budget=label,group_size=g,q_minus=minus,q_plus=(minus+1 if label=='low' and minus is not None and minus<10 else None)))
            if minus is not None:selected.setdefault((g,minus),[]).append(label+'_lower_budget')
            if label=='low' and minus is not None and minus<10:selected.setdefault((g,minus+1),[]).append('low_higher_budget')
    methods=[]
    for b in (4,8):
        methods.append(dict(method_id=f'official_dynamiq_fixed_codec_b{b}',family='official_dynamiq_fixed_codec',
            implementation_origin='pinned official CUDA through glue-only adapter',nominal_parameter=b,stored_code_bits=b,group_size=16,
            scale_dtype='UINT8 group scale + BF16 super-scale-unit',arithmetic_profile='official strategy3 BF16 path, FP32 arithmetic, FP16 table, original --use_fast_math',
            output_conversion='native BF16 output, crop logical length',packet_format='official interleaved tight uint8 packet',
            shared_state='static chunk16/strategy3/bits/dtype/D/P and input-independent official tables; ephemeral local random pool not transmitted',
            rng_mode='official functional-test CPU FP32 rand -> BF16 -> CUDA, fresh independent generator per worker/config/repeat',timing_eligible=False,
            deterministic=False,repeats=8,budget_role='low_official' if b==4 else 'high_official',packet_total_bytes=budgets[b],padded_numel=official.padded_numel(d)))
    for (g,q),roles in sorted(selected.items()):
        methods.append(dict(method_id=f'uniform_g{g}_q{q}',family='block_uniform',implementation_origin='simple CPU/PyTorch reference',
            nominal_parameter=q,stored_code_bits=q,group_size=g,scale_dtype='FP32',arithmetic_profile='FP32 normalization/floor/probability; scale nextafter until FP64 coverage; FP32 multiply to BF16 RNE',
            output_conversion='FP32 multiply -> BF16 RNE -> crop',packet_format='16-byte header + q-bit tight payload + FP32 block scales',
            shared_state='versioned wire specification only',rng_mode='fresh CPU FP32 torch.rand per worker/config/repeat',timing_eligible=False,
            deterministic=False,repeats=8,budget_role=','.join(roles),packet_total_bytes=measured[g,q],padded_numel=((d+g-1)//g)*g))
    methods.append(dict(method_id='bf16_identity',family='bf16_identity',implementation_origin='raw BF16 serialized bytes',nominal_parameter=16,stored_code_bits=16,
        group_size=None,scale_dtype=None,arithmetic_profile='identity; one-shot sum uses FP64',output_conversion='none',packet_format='BF16 little-endian vector',
        shared_state='agreed logical length and dtype',rng_mode='deterministic',timing_eligible=False,deterministic=True,repeats=1,budget_role='uncompressed_reference',packet_total_bytes=2*d,padded_numel=d))
    dump(run/'logs/budget_selection.json',dict(actual_packet_size_only=True,error_metrics_not_computed=True,official_budgets={str(k):v for k,v in budgets.items()},
        measured=budget_rows,selection=selection,freeze_gpu_wall_seconds=gpu_seconds))
    inputs=read(run/'input_manifest.json');shutil.copy2(run/'input_manifest.json',run/'logs/input_manifest_D0.json')
    for r in inputs['inputs']:r['padding_by_method']={m['method_id']:m['padded_numel'] for m in methods}
    inputs['preliminary_until_method_padding_frozen']=False;dump(run/'input_manifest.json',inputs,exclusive=False)
    source_rows=[]
    paths=list((ROOT/'src/gradient_comm_lab/bounded_discovery').glob('*.py'))+list((ROOT/'scripts/bounded_discovery').glob('*.py'))
    for p in paths:
        rel=p.relative_to(ROOT);dest=run/'code'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        source_rows.append(dict(path=dest.relative_to(run).as_posix(),original_path=str(p),bytes=p.stat().st_size,sha256=sha(p),role='execution_or_analysis_source'))
    upstream=read(run/'logs/official_source_manifest.json')
    build=read(run/'logs/official_build_a03.json')
    original_sources=[]
    for r in upstream['files']:
        p=ROOT/r['path'];assert sha(p)==r['sha256']
        dest=run/'code/upstream'/r['upstream_path'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        original_sources.append(dict(path=dest.relative_to(run).as_posix(),original_path=str(p),bytes=r['bytes'],sha256=r['sha256'],url=r['url'],github_blob_sha1=r['github_blob_sha1']))
    source_manifest=dict(schema_version='bounded_discovery_v1.0',run_id=run.name,files=source_rows,original_official_sources=original_sources,upstream_manifest='logs/official_source_manifest.json',
        upstream_manifest_sha256=sha(run/'logs/official_source_manifest.json'),binary=dict(path=build['module_path'],sha256=build['module_sha256'],bytes=Path(build['module_path']).stat().st_size),
        source_glue_diff=build.get('source_glue_diff_sha256'),build_manifest_sha256=sha(run/'logs/official_build_a03.json'),frozen_at=now())
    dump(run/'source_manifest.json',source_manifest)
    contract=dict(schema_version='bounded_discovery_v1.0',run_id=run.name,revision=1,frozen_at=now(),scope='official_fixed_codec_development_diagnosis',
        upstream_repository='https://github.com/CharlesHan24/DynamiQ',upstream_commit=official.COMMIT,source_manifest_sha256=sha(run/'source_manifest.json'),
        official_entrypoints=['eden_utils.scaling_compress','eden_utils.scaling_decompress'],official_configurations=[dict(nbits=b,chunk_size=16,strategy=3,dtype='BF16') for b in (4,8)],
        wrapper_changes='allocating validation/padding/stream selection/cropping/byte serialization only; core arithmetic unchanged; build glue separately recorded',
        methods=methods,comparison_budget_rule='For each g in {16,128}, choose max actual packed q in [2,10] <= official low budget, plus q+1 when allowed; high budget lower neighbor only; deduplicate',
        selected_uniform_configurations=selection,input_manifest_sha256=sha(run/'input_manifest.json'),logical_input_dtype='BF16',decoded_output_dtype='BF16',
        padding_by_method={m['method_id']:m['padded_numel'] for m in methods},rng_by_method={m['method_id']:m['rng_mode'] for m in methods},repeats=8,
        seed_derivation='first8 little endian SHA256(UTF8(bounded_discovery_v1:{input_id}:{method_id}:{repeat_id})) modulo2^63; distinct streams, not matched randomness',
        aggregation_definition='one_shot_codec_sum: each worker encodes once; FP64 sums of the same four decoded outputs; no ring multiplier',
        metric_definitions='local SSE/MSE/energy and sum diagnostics per PLAN section5, FP64 over logical region; zero denominator=null; near_cancellation if ratio<1e-6',
        stage_budgets=dict(d0_active_hours=4,max_build_repair_rounds=2,d1_active_workday=1,max_new_GPU_hours=2,max_findings=1,max_single_factor_controls=2),
        optional_timing_enabled=False,stop_rules=['D0 gate must pass','no D2 or training','no additional methods','at most one actionable signal and two localization controls','failure retained; incomplete results cannot establish dominance'],
        aggregate_comparability_status=qualification['aggregate_comparability_status'],qualification_sha256=sha(run/'baseline_qualification.json'))
    dump(run/'experiment_contract.json',contract)
    print(json.dumps(dict(status='frozen',method_count=len(methods),official_budgets=budgets,selection=selection,contract_sha256=sha(run/'experiment_contract.json'))))
if __name__=='__main__':main()
