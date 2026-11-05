"""Collect reviewable source/config snapshots without touching historical runs."""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);args=p.parse_args();run=(ROOT/args.run_dir).resolve()
    if not run.is_relative_to(ROOT/'results/gpu_codec_v1') or (run/'artifact_integrity.json').exists():raise ValueError('New unsealed run required')
    registration=read(run/'execution_registration.json');files=[];names=set()
    def add(source,destination,role,execution_bound=False):
        source=Path(source);destination=Path(destination)
        if str(destination) in names:return
        names.add(str(destination));destination.parent.mkdir(parents=True,exist_ok=True)
        if source.resolve()!=destination.resolve():shutil.copyfile(source,destination)
        files.append({'path':destination.relative_to(run).as_posix(),'role':role,'sha256':sha(destination),
            'bytes':destination.stat().st_size,'original_path':str(source.resolve()),'execution_bound':execution_bound})
    for name,expected in registration['source_hashes'].items():
        assert sha(ROOT/name)==expected,name
        add(ROOT/name,run/'code'/name,registration['source_roles'][name],True)
    extras={
        'src/gradient_comm_lab/__init__.py':'gpu_source',
        'scripts/summarize_g1a.py':'report_generator',
        'scripts/write_g1a_report.py':'report_generator',
        'scripts/collect_g1a_sources.py':'report_generator',
        'scripts/seal_g1a_delivery.py':'test',
        'scripts/package_g1a_delivery.py':'report_generator',
        'scripts/probe_toolchain_v2.py':'build_script',
        'scripts/profile_cuda_trace_v2.py':'build_script',
        'scripts/cupti_trace_collector_v2.cpp':'gpu_source',
        'scripts/audit_cuda_trace_v2.py':'test',
        'scripts/freeze_g0_inputs.py':'build_script',
        'src/g0_probe.cpp':'gpu_source',
        'src/g0_probe.cu':'gpu_source',
        'TECHNICAL_PLAN.md':'config','DELIVERY_FORMAT.md':'config','README.md':'config',
        'reviews/gpu_codec_v1_20261101_001_review.md':'config',
        'handoff_templates/REPORT.md':'config','handoff_templates/summary.json':'config',
        'handoff_templates/codec_contract.json':'config','handoff_templates/environment.json':'config',
    }
    for script in (ROOT/'scripts').glob('*toolchain*'):
        if script.is_file() and script.suffix in ('.py','.ps1','.cmd'):extras[script.relative_to(ROOT).as_posix()]='build_script'
    for name,role in extras.items():add(ROOT/name,run/'code'/name,role)
    reference_root=Path('E:/dacomp/research/spl_model_communication')
    for item in read(run/'codec_contract.json')['cpu_reference_files']:
        source=Path(item['path']);assert sha(source)==item['sha256']
        add(source,run/'code/reference'/source.relative_to(reference_root),'cpu_reference',True)
    add(reference_root/'src/splcomm/__init__.py',run/'code/reference/src/splcomm/__init__.py','cpu_reference')
    for name in ('registry.py','training.py','global_qsgd.py'):
        add(reference_root/'src/splcomm'/name,run/'code/reference/src/splcomm'/name,'cpu_reference')
    for path in (run/'build').rglob('*'):
        if path.is_file() and (path.suffix in ('.cu','.cpp','.h','.ninja') or path.name in ('build.ninja',)):
            add(path,run/'code'/path.relative_to(run),'gpu_source' if path.suffix in ('.cu','.cpp','.h') else 'build_script')
    for name in ('codec_contract.json','benchmark_config.json','execution_registration.json','requirements.txt'):
        add(run/name,run/name,'config')
    try:
        commit=subprocess.run(['git','rev-parse','--show-toplevel'],cwd=ROOT,capture_output=True,text=True,timeout=10)
        if commit.returncode==0:
            git={'repository_root':commit.stdout.strip(),
                'commit':subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,timeout=10).stdout.strip(),
                'status_porcelain':subprocess.run(['git','status','--porcelain','--',str(ROOT)],cwd=ROOT,capture_output=True,text=True,timeout=10).stdout}
        else:git={'repository_root':None,'commit':None,'status_porcelain':None,'discovery_stderr':commit.stderr.strip()}
    except Exception as e:git={'commit':None,'error':str(e)}
    result={'schema_version':'gpu_codec_v1.0','run_id':run.name,'collected_at':datetime.now().astimezone().isoformat(),
        'git':git,'git_commit':git.get('commit'),'git_note':'Actual discovery only; source identity is the recorded SHA256, including execution registration before measurements.',
        'files':files,'external_toolchain_note':'Compiler packages and official URLs/hashes are in logs/toolchain_*.json; project-local binary tools are excluded from source archive.'}
    (run/'source_manifest.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'source_files':len(files),'git_commit':result['git_commit']}))

if __name__=='__main__':main()
