"""Freeze G2a execution sources before the full correctness matrix."""
import argparse,json,hashlib,shutil
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',required=True);args=ap.parse_args()
    run=(ROOT/args.run_dir).resolve()
    assert run.is_relative_to(ROOT/'results/logical_ring_v1')
    assert (run/'correctness_case_manifest.json').exists()
    assert not (run/'execution_registration.json').exists()
    paths=['src/gradient_comm_lab/'+f for f in ('__init__.py','cpu_reference.py','torch_codec.py','cuda_codec.py','logical_ring.py','ring_reference.py','ring_fixtures.py')]
    paths += ['scripts/'+f for f in ('prepare_logical_ring.py','check_logical_ring.py','audit_logical_ring.py','register_logical_ring.py','check_original_ring_parity.py','finalize_logical_ring.py','python.ps1')]
    paths += [p.relative_to(ROOT).as_posix() for p in (ROOT/'src/gradient_comm_lab/csrc').glob('*') if p.is_file()]
    rows=[];hashes={}
    for rel in paths:
        p=ROOT/rel;h=sha(p);dest=run/'code'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        hashes[rel]=h;rows.append(dict(path=dest.relative_to(run).as_posix(),original_path=str(p),sha256=h,bytes=p.stat().st_size,role='execution_source'))
    oldref=ROOT.parent/'spl_model_communication/src/splcomm'
    for p in sorted(oldref.glob('*.py')):
        dest=run/'code/reference/src/splcomm'/p.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        rows.append(dict(path=dest.relative_to(run).as_posix(),original_path=str(p),sha256=sha(p),bytes=p.stat().st_size,role='cpu_reference_dependency'))
    build=read(run/'logs/cuda_build_manifest.json');module=Path(build['module_path'])
    assert sha(module)==build['module_sha256'];dest=run/'code/build'/module.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(module,dest)
    rows.append(dict(path=dest.relative_to(run).as_posix(),original_path=str(module),sha256=sha(module),bytes=module.stat().st_size,role='accepted_binary_copy'))
    write(run/'source_manifest.json',dict(schema_version='logical_ring_v1.0',run_id=run.name,files=rows))
    write(run/'execution_registration.json',dict(schema_version='logical_ring_v1.0',run_id=run.name,
        frozen_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),phase='before_formal_correctness',
        contract_sha256=sha(run/'ring_contract.json'),input_manifest_sha256=sha(run/'input_manifest.json'),
        threshold_manifest_sha256=sha(run/'threshold_manifest.json'),correctness_case_manifest_sha256=sha(run/'correctness_case_manifest.json'),
        build_manifest_sha256=sha(run/'logs/cuda_build_manifest.json'),source_manifest_sha256=sha(run/'source_manifest.json'),sources=hashes))
    print(json.dumps(dict(status='registered',sources=len(rows),execution_sources=len(hashes))))
if __name__=='__main__':main()
