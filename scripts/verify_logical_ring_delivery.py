"""Post-execution report clarification and independent exact archive verification."""
import argparse,json,hashlib,zipfile,shutil
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'results/gpu_codec_v1/gpu_codec_v1_20261104_001'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,ensure_ascii=False);f.write('\n')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',required=True);ap.add_argument('--postseal',action='store_true');a=ap.parse_args()
    run=(ROOT/a.run_dir).resolve();assert run.is_relative_to(ROOT/'results/logical_ring_v1')
    before=read(run/'logs/prior_seal_check.json')
    assert sha(OLD/'artifact_integrity.json')==before['prior_integrity_sha256']
    for entry in read(OLD/'artifact_integrity.json')['files']:
        assert sha(OLD/entry['path'])==entry['sha256']
    if not a.postseal:
        assert not (run/'artifact_integrity.json').exists()
        p=run/'REPORT.md';oldtext=p.read_text(encoding='utf-8');oldsha=sha(p)
        old='全部实际问题见 [issues.jsonl](issues.jsonl)，首次失败反例及 traceback 保留在相应尝试目录。'
        new='全部实际问题见 [issues.jsonl](issues.jsonl)。该临时 probe 保留了错误类型、消息、原因和修正记录，未单独保存完整 traceback；正式矩阵无失败。runner 在失败时保存反例的分支未因本轮正式失败而触发。'
        assert old in oldtext;p.write_text(oldtext.replace(old,new),encoding='utf-8')
        repro=run/'REPRODUCE.md';t=repro.read_text(encoding='utf-8')
        marker='./scripts/python.ps1 scripts/audit_logical_ring.py --run-dir <RUN> --output logs/audit_final.json'
        t=t.replace(marker,'./scripts/python.ps1 scripts/verify_logical_ring_delivery.py --run-dir <RUN>\n'+marker)
        t+='\nAfter sealing, run `./scripts/python.ps1 scripts/verify_logical_ring_delivery.py --run-dir <RUN> --postseal` to verify exact ZIP membership, every member SHA, and the old G1 integrity manifest itself. This writes only an adjacent verification file.\n'
        repro.write_text(t,encoding='utf-8')
        dest=run/'code/scripts'/Path(__file__).name;shutil.copy2(__file__,dest)
        write(run/'delivery_tools_manifest.json',dict(role='post_execution_document_correction_and_archive_verification',
            not_part_of_GPU_execution=True,files=[dict(path=dest.relative_to(run).as_posix(),original_path=str(Path(__file__).resolve()),sha256=sha(dest))]))
        write(run/'logs/report_review_amendment.json',dict(status='corrected',generated_report_sha256=oldsha,
            delivered_report_sha256=sha(p),reason='Describe exactly retained preflight failure evidence; no false traceback claim',
            prior_g1_integrity_sha256=before['prior_integrity_sha256'],formal_data_modified=False))
        print(json.dumps(dict(status='preseal_review_pass',old_g1_integrity_unchanged=True)));return
    integrity=read(run/'artifact_integrity.json');expected={e['path']:e for e in integrity['files']}
    assert len(expected)==len(integrity['files'])
    actual={p.relative_to(run).as_posix() for p in run.rglob('*') if p.is_file()}
    assert actual==set(expected)|{'artifact_integrity.json'}
    for name,e in expected.items():
        p=run/name;assert sha(p)==e['sha256'] and p.stat().st_size==e['bytes'],name
    archive=run.with_suffix('.zip')
    with zipfile.ZipFile(archive) as z:
        names=z.namelist();assert len(names)==len(set(names)) and set(names)==actual
        for name in names:
            content=z.read(name)
            assert hashlib.sha256(content).hexdigest()==sha(run/name)
            if name in expected:assert hashlib.sha256(content).hexdigest()==expected[name]['sha256'] and len(content)==expected[name]['bytes']
        assert z.testzip() is None
    original=read(run.parent/(run.name+'_postseal_verification.json'))
    assert sha(archive)==original['zip_sha256'] and sha(run/'artifact_integrity.json')==original['artifact_integrity_sha256']
    result=dict(status='pass',verified_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        actual_zip_entries=len(names),exact_member_set=True,no_duplicate_members=True,all_member_hashes_match_integrity=True,
        prior_g1_integrity_and_all_files_unchanged=True,zip_sha256=sha(archive),integrity_sha256=sha(run/'artifact_integrity.json'),
        verifier_sha256=sha(__file__))
    write(run.parent/(run.name+'_independent_postseal_verification.json'),result);print(json.dumps(result))
if __name__=='__main__':main()
