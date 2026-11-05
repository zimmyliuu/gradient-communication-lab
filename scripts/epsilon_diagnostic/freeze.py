"""Freeze the single-factor contract before qualification/formal codec calls."""
import argparse,hashlib,json,shutil
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/bounded_discovery_v1/bounded_discovery_v1_20261104_001'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',required=True);a=ap.parse_args();run=(ROOT/a.run_dir).resolve();assert run.is_relative_to(ROOT/'results/epsilon_diagnostic_v1')
    assert not (run/'experiment_contract.json').exists()
    pairs=read(run/'input_randomness_manifest.json')['pairs'];assert len(pairs)==96
    assert all(r['original_pool_sha256']==r['regenerated_pool_sha256'] for r in pairs)
    builds={c:read(run/'build'/c/'build_manifest.json') for c in ['control_add1e7','diagnostic_add0']}
    assert all(v['status']=='compiled_and_loaded' for v in builds.values())
    files=[]
    def add(p,role,origin):
        p=Path(p).resolve();files.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),role=role,origin=origin))
    for folder in [ROOT/'scripts/epsilon_diagnostic',ROOT/'src/gradient_comm_lab/epsilon_diagnostic']:
        for p in folder.glob('*.py'):
            dest=run/'code'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
            add(dest,'frozen_execution_source',str(p));add(p,'workspace_execution_source',str(p))
    for c,b in builds.items():
        add(run/'build'/c/'build_manifest.json','encoder_build_manifest',c)
        add(b['module_path'],'derived_encoder_binary',c)
        for p in Path(b['source_directory']).rglob('*'):
            if p.is_file():add(p,'derived_encoder_source',c)
    decoder=read(BASE/'logs/official_build_a03.json')
    add(decoder['module_path'],'original_sealed_decoder_binary',decoder['official_commit'])
    add(BASE/'logs/official_build_a03.json','original_decoder_build_manifest',decoder['official_commit'])
    add(BASE/'experiment_contract.json','base_contract',str(BASE))
    add(BASE/'logs/d0_execution.json','qualification_reference_metadata',str(BASE))
    dump(run/'source_manifest.json',dict(schema_version='epsilon_sources_v1',files=files,base_commit=decoder['official_commit'],
        two_encoder_conditions=list(builds),decoder_unchanged=True))
    contract=dict(schema_version='epsilon_diagnostic_v1.0',run_id=run.name,revision=1,frozen_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        base_run=str(BASE),base_contract_sha256=sha(BASE/'experiment_contract.json'),upstream_commit=decoder['official_commit'],
        source_manifest_sha256=sha(run/'source_manifest.json'),decoder_binary_sha256=decoder['module_sha256'],
        encoder_binary_sha256={c:b['module_sha256'] for c,b in builds.items()},
        input_randomness_manifest_sha256=sha(run/'input_randomness_manifest.json'),region_manifest_sha256=sha(run/'region_manifest.json'),
        pair_ids=[r['pair_id'] for r in pairs],H=read(run/'region_manifest.json'),
        actual_kernel='scaling_compress_kernel_mee_hierarchical_bf16_opt<8>',
        conditions=dict(control_add1e7=1e-7,diagnostic_add0=0.0),mee_eps8=0.1,
        patch='Only optimized BF16 MEE raw_scale additive term selected via DIAG_B8_SCALE_ADD for NBITS==8; all other widths use original 1e-7f',
        flags={c:dict(extra_cflags=b['extra_cflags'],extra_cuda_cflags=b['extra_cuda_cflags']) for c,b in builds.items()},
        random_reuse='Read original 96 official b8 seeds; CPU FP32 torch.rand->BF16; exact full-P pool SHA must match. Never derive from new condition_id. A and B matched, uniform separate old streams.',
        decoder_process='Only original sealed module; never load encoder condition modules in decoder process',
        metric_definitions=dict(local='FP64 SSE/MSE/maxabs over logical full/H/rest; source_energy; paired delta=A-B before summarizing',
            one_shot='FP64 sum same four output vectors, original FP64 sum inputs, cross_term=aggregateSSE-sumlocalSSE',
            gap_closure='Only update100 positive mean(A-U): mean(A-B)/mean(A-U); H separately; no clipping',
            outcome_order=['qualification_failure=inconclusive','domain_failure=diagnostic_domain_failure','all decoded equal=no_effect_observed','gap_closure<=0=changes_without_closing_gap','0<gap_closure<1=partially_closes_gap','gap_closure>=1=closes_or_reverses_gap']),
        continuation_gate=dict(gap_closure_min=0.5,gap_closure_H_min=0.5,improved_workers_min=3,improved_snapshot_repeats_min=7,other_snapshot_relative_SSE_increase_max=0.02),
        budgets=dict(precheck_minutes=30,active_hours=4,gpu_minutes=30,condition_builds=2,repair_rounds=1),
        build_wall_seconds_upper_bound=sum(b.get('elapsed_seconds',0) for b in builds.values()),
        stop_rules=['No third constant or scale grid','Control 96 exact packet/decoded parity required before intervention','Domain failures retained and exclude causal partial-subset summaries','No D2 or new training','No changes to old runs'])
    dump(run/'experiment_contract.json',contract);print(sha(run/'experiment_contract.json'))
if __name__=='__main__':main()
