"""Verify only the twelve previously used development BF16 input tensors."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
import torch
from gradient_comm_lab.bounded_discovery.common import read,dump,sha,identity,now
def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',required=True);a=ap.parse_args();run=(ROOT/a.run_dir).resolve()
    origin=ROOT/'results/gpu_codec_v1/gpu_codec_v1_20261104_001/input_manifest.json';rows=[]
    for r in read(origin)['inputs']:
        if r['source_kind']!='real_gradient':continue
        assert r['training_seed']==1101 and r['update_step'] in (100,500,2500) and r['partition']=='development'
        assert sha(r['fixture_path'])==r['fixture_sha256']
        x=torch.load(r['fixture_path'],map_location='cpu',weights_only=False)['input_logical']
        assert x.dtype==torch.bfloat16 and x.numel()==818048 and identity(x)['sha256']==r['input_sha256']
        rows.append(dict(snapshot_id=f'seed1101_update{r["update_step"]:06d}',worker_id=r['rank'],input_id=r['input_id'],
            training_seed=1101,update_step=r['update_step'],partition='development',logical_numel=x.numel(),input_dtype='torch.bfloat16',
            source_path=r['source_path'],source_sha256=r['source_sha256'],fixture_path=r['fixture_path'],fixture_sha256=r['fixture_sha256'],
            input_identity=identity(x),transform_order=r['transform_order'],already_divided_by_world_size=True,
            world_size=4,loader='torch.load trusted accepted fixture, input_logical (no old padding, no second division)',padding_by_method={}))
    assert len(rows)==12 and {(r['update_step'],r['worker_id']) for r in rows}=={(s,w) for s in (100,500,2500) for w in range(4)}
    rows.sort(key=lambda r:(r['update_step'],r['worker_id']))
    dump(run/'input_manifest.json',dict(schema_version='bounded_discovery_v1.0',run_id=run.name,created_at=now(),
        source_manifest=str(origin),source_manifest_sha256=sha(origin),inputs=rows,old_holdout_opened=False,preliminary_until_method_padding_frozen=True))
    print('12 existing development inputs verified; no new data or padding consumed')
if __name__=='__main__':main()
