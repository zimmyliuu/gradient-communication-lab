"""Preserve completed revision 1 and freeze audit-only runner revision 2.

No input, method, byte budget, rounding rule or seed changes are permitted here.
"""
import argparse,sys,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from gradient_comm_lab.bounded_discovery.common import read,dump,sha,now

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',required=True);a=ap.parse_args();run=(ROOT/a.run_dir).resolve()
    assert run.is_relative_to((ROOT/'results/bounded_discovery_v1').resolve())
    old=read(run/'experiment_contract.json');assert old['revision']==1
    execution=read(run/'logs/d1_execution.json');assert execution['execution_complete']
    old_sha=sha(run/'experiment_contract.json');archive=(run/'revisions/contract_v1').resolve()
    assert archive.is_relative_to(run) and not archive.exists();archive.mkdir(parents=True)
    names=['experiment_contract.json','source_manifest.json','metrics_raw.jsonl','packet_accounting.jsonl','correctness.jsonl',
           'packets','decoded','code','logs/d1_start.json','logs/d1_execution.json','logs/d1_console.log']
    for name in names:
        src=(run/name).resolve();dst=(archive/name).resolve()
        assert src.is_relative_to(run) and dst.is_relative_to(archive) and src.exists() and not dst.exists()
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(src),str(dst))
    if (run/'issues.jsonl').exists():
        src=(run/'issues.jsonl').resolve();dst=(archive/'issues.jsonl').resolve()
        assert src.is_relative_to(run) and dst.is_relative_to(archive);shutil.move(str(src),str(dst))
    for name in ['input_manifest.json','baseline_qualification.json','environment.json']:shutil.copy2(run/name,archive/name)
    shutil.copytree(archive/'code',run/'code')
    sources=read(archive/'source_manifest.json');rows=[]
    for p in list((ROOT/'src/gradient_comm_lab/bounded_discovery').glob('*.py'))+list((ROOT/'scripts/bounded_discovery').glob('*.py')):
        dest=run/'code'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        rows.append(dict(path=dest.relative_to(run).as_posix(),original_path=str(p),bytes=p.stat().st_size,sha256=sha(p),role='execution_or_analysis_source'))
    sources.update(files=rows,frozen_at=now(),revision=2,previous_manifest_sha256=sha(archive/'source_manifest.json'))
    dump(run/'source_manifest.json',sources)
    reason='Independent code review after v1 launch: retain failed artifacts immediately, and require complete aggregate count plus normal completion. No observed v1 data failure; preserve and rerun unchanged matrix.'
    revised=dict(old);revised.update(revision=2,frozen_at=now(),source_manifest_sha256=sha(run/'source_manifest.json'),
        supersedes_contract_sha256=old_sha,revision_reason=reason,previous_execution_GPU_upper_bound_seconds=execution['gpu_execution_upper_bound_seconds'])
    for key in ['methods','input_manifest_sha256','selected_uniform_configurations','seed_derivation','comparison_budget_rule','padding_by_method']:
        assert revised[key]==old[key]
    dump(run/'experiment_contract.json',revised)
    dump(run/'logs/contract_revision_history.json',dict(revision=2,reason=reason,previous_contract_sha256=old_sha,
        current_contract_sha256=sha(run/'experiment_contract.json'),previous_results_path='revisions/contract_v1',
        numerical_method_or_budget_changed=False,previous_local=execution['successful_local'],previous_aggregate=execution['successful_aggregate'],
        previous_failed=execution['failed_records']))
    print(sha(run/'experiment_contract.json'))
if __name__=='__main__':main()
