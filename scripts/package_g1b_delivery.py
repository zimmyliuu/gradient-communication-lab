"""Package a sealed run, excluding compiled cache while retaining its hashes."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);args=p.parse_args();run=(ROOT/args.run_dir).resolve()
    if not run.is_relative_to(ROOT/'results/gpu_codec_v1'):raise ValueError('Invalid run')
    integrity=json.loads((run/'artifact_integrity.json').read_text(encoding='utf-8'));included=[];excluded=[]
    for item in integrity['files']:
        path=run/item['path'];assert sha(path)==item['sha256'] and path.stat().st_size==item['bytes'],str(path)
        if item['path'].startswith('build/'):
            excluded.append({**item,'local_path':str(path),'reason':'Compiled/probe cache; original sources and build commands are packaged separately'})
        else:included.append(item)
    included.append({'path':'artifact_integrity.json','bytes':(run/'artifact_integrity.json').stat().st_size,'sha256':sha(run/'artifact_integrity.json')})
    archive=run.parent/(run.name+'.zip')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for item in included:z.write(run/item['path'],run.name+'/'+item['path'])
        z.writestr(run.name+'/PACKAGE_CONTENTS.json',json.dumps({'run_id':run.name,'full_run_local_path':str(run),
            'included':included,'excluded_local_build_files':excluded,
            'external_inputs':'See fixtures_manifest.json. Historical checkpoints and project toolchain binaries are not included.',
            'portability':'This is a review/source/evidence archive; rebuilding historical fixtures requires referenced checkpoint files and local path/environment setup.'},indent=2,ensure_ascii=False))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(included)+1 and len(set(z.namelist()))==len(z.namelist())
        for item in included:
            data=z.read(run.name+'/'+item['path'])
            assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256'],item['path']
        packaged=json.loads(z.read(run.name+'/PACKAGE_CONTENTS.json'))
        assert packaged['included']==included and packaged['excluded_local_build_files']==excluded
    digest=sha(archive);archive.with_suffix('.zip.sha256').write_text(digest+'  '+archive.name+'\n',encoding='utf-8')
    print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'sha256':digest,'included_files':len(included)+1,'external_build_files':len(excluded)}))

if __name__=='__main__':main()
