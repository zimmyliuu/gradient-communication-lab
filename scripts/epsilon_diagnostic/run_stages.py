"""Launch each condition encoder and the sealed decoder in fresh processes."""
import argparse,json,subprocess,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/bounded_discovery_v1/bounded_discovery_v1_20261104_001'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',required=True);a=ap.parse_args();run=(ROOT/a.run_dir).resolve()
    for phase in ['qualification','formal']:
        for condition in ['control_add1e7','diagnostic_add0']:
            for operation in ['encode','decode']:
                manifest=run/'build'/condition/'build_manifest.json' if operation=='encode' else BASE/'logs/official_build_a03.json'
                cmd=[sys.executable,'-X','utf8',str(ROOT/'scripts/epsilon_diagnostic/execute.py'),'--run-dir',str(run),'--operation',operation,'--condition',condition,'--phase',phase,'--build-manifest',str(manifest)]
                key=f'{phase}_{operation}_{condition}'
                with (run/'logs'/f'{key}_console.log').open('x',encoding='utf-8') as f:
                    result=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=300)
                print(json.dumps(dict(stage=key,returncode=result.returncode)),flush=True)
                if result.returncode:raise SystemExit(result.returncode)
                assert read(run/'logs'/f'{key}_completion.json')['status']=='pass'
    pairs=[]
    decoders={c:{r['pair_id']:r for r in rows(run/'logs'/f'formal_decode_{c}.jsonl')} for c in ['control_add1e7','diagnostic_add0']}
    for source in read(run/'input_randomness_manifest.json')['pairs']:
        item={k:source[k] for k in ['pair_id','snapshot_id','worker_id','repeat_id','input_id']}
        item['conditions']={c:decoders[c][item['pair_id']] for c in decoders};pairs.append(item)
    with (run/'execution_index.json').open('x',encoding='utf-8') as f:json.dump(dict(pairs=pairs,contract_sha256=hashlib.sha256((run/'experiment_contract.json').read_bytes()).hexdigest()),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status='pass',pairs=len(pairs))))
if __name__=='__main__':main()
