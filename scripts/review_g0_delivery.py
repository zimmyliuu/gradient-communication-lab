"""Apply reviewed documentation clarifications before sealing; retain drafts."""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',required=True)
    args=parser.parse_args();run=(ROOT/args.run_dir).resolve()
    if not run.is_relative_to(ROOT/'results/gpu_codec_v1'):raise ValueError('Invalid run directory')
    archive=run/'logs/prepublication_revision_01'
    archive.mkdir(exist_ok=False)
    names=['REPORT.md','REPRODUCE.md','codec_contract.json','cpu_reference_manifest.json',
        'summary.json','source_manifest.json','logs/toolchain_reference_sources.json']
    before={}
    for name in names:
        p=run/name;target=archive/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,target);before[name]=sha(p)
    for name,key in (('codec_contract.json','cpu_adapter'),('cpu_reference_manifest.json','adapter')):
        value=read(run/name)
        value[key].update(path_base='project_root',project_root=ROOT.as_posix(),
                          snapshot_path='code/src/gradient_comm_lab/cpu_reference.py',snapshot_path_base='run_directory')
        assert sha(run/value[key]['snapshot_path'])==value[key]['sha256']
        write(run/name,value)
    p=run/'REPORT.md'
    p.write_text(p.read_text(encoding='utf-8').replace('本次实际失败发生在调用编译器之前',
        '本次实际失败发生在实际源码编译前；此前仅进行过cl版本查询'),encoding='utf-8')
    p=run/'logs/toolchain_reference_sources.json';value=read(p)
    value['sources'][0]['interpretation']='Compatibility remains unverified; actual failure occurred before source compilation; cl was invoked only for a version query.'
    write(p,value)
    p=run/'REPRODUCE.md'
    with p.open('a',encoding='utf-8') as stream:
        stream.write('\n## 7 交付前审阅修订\n\n以下命令已实际执行，澄清适配器路径基准和失败阶段措辞；保留修订前草稿，不重跑或修改实验日志。\n\n```powershell\n'
            +f'.\\scripts\\python.ps1 scripts/review_g0_delivery.py --run-dir {args.run_dir}\n```\n')
    summary=read(run/'summary.json');summary['contract_sha256']=sha(run/'codec_contract.json')
    summary['reviewed_at']=datetime.now().astimezone().isoformat();write(run/'summary.json',summary)
    manifest=read(run/'source_manifest.json')
    for row in manifest['files']:
        if row['path']=='codec_contract.json':
            row.update(sha256=sha(run/row['path']),bytes=(run/row['path']).stat().st_size)
    source=Path(__file__);relative='code/scripts/review_g0_delivery.py'
    shutil.copyfile(source,run/relative)
    manifest['files'].append({'path':relative,'role':'report_generator','sha256':sha(run/relative),
        'bytes':(run/relative).stat().st_size,'original_path':str(source.resolve())})
    write(run/'source_manifest.json',manifest)
    write(run/'logs/prepublication_review_changes.json',{'created_at':datetime.now().astimezone().isoformat(),
        'scope':'documentation_and_path_metadata_only_no_experiment_or_CPU_semantic_change',
        'changes':['Explicit project-relative adapter source path and run-relative snapshot path',
                   'Distinguish cl version query from source compilation'],
        'before_sha256':before,'after_sha256':{name:sha(run/name) for name in names},
        'preserved_drafts':'logs/prepublication_revision_01','experiment_logs_modified':False})
    print(json.dumps({'documentation_review':'applied','contract_sha256':summary['contract_sha256'],
        'source_files':len(manifest['files']),'G0_status':summary['execution_status']}))


if __name__=='__main__':main()
