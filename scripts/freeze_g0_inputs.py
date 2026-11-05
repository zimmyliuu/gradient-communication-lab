"""Freeze CPU sources, LUTs and the twelve authorized development gradients.

This is a G0 source/fixture preparation, not GPU codec correctness evidence.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone,timedelta
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gradient_comm_lab.cpu_reference import REFERENCE_ROOT,lookup_table,pad_input,threshold_fixture,tensor_identity,wire_size
from splcomm.p2_replay import load_inputs


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''): digest.update(block)
    return digest.hexdigest()


def save_json(path,data):
    with path.open('x',encoding='utf-8') as stream:
        json.dump(data,stream,indent=2,ensure_ascii=False,allow_nan=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True)
    args=parser.parse_args()
    run=(ROOT/args.run_dir).resolve()
    if not run.is_relative_to(ROOT/'results/gpu_codec_v1') or not run.is_dir():
        raise ValueError('A unique existing local run directory is required')
    if (run/'input_manifest.json').exists(): raise FileExistsError('Preserve prior input manifests')
    torch.set_num_threads(4)
    directory=ROOT/'fixtures'/run.name
    directory.mkdir(parents=True,exist_ok=False)
    started=datetime.now(timezone(timedelta(hours=8))).isoformat()
    refs=['src/splcomm/dynamiq.py','src/splcomm/packing.py','tests/test_dynamiq.py',
          'tests/dynamiq_oracle.py','src/splcomm/p2_replay.py','src/splcomm/core.py','src/splcomm/ring.py']
    source=[{'path':str(REFERENCE_ROOT/p),'sha256':sha(REFERENCE_ROOT/p),
             'bytes':(REFERENCE_ROOT/p).stat().st_size,'role':'cpu_reference'} for p in refs]
    tables={b:lookup_table(b) for b in (4,8)}
    table_path=directory/'lookup_tables.pt';torch.save(tables,table_path)
    lut={str(b):tensor_identity(tables[b]) for b in tables}
    calibration=REFERENCE_ROOT/'results/p2/calibration_20261101_v1'
    inputs=load_inputs(calibration,'pilot')
    rows=[]
    for snapshot in inputs:
        for rank,gradient in enumerate(snapshot['gradients']):
            identity=f"real_{snapshot['input_id']}_rank{rank}"
            # Preserve BF16 conversion then BF16 division before any padding.
            logical=gradient.to(torch.bfloat16)/4
            padded=pad_input(logical)
            main_random,scale_random,generator=threshold_fixture(identity,padded.numel())
            path=directory/f'{identity}.pt'
            torch.save({'schema_version':'gpu_codec_v1.0','input_id':identity,
                'input_logical':logical,'input_padded':padded,'main_thresholds':main_random,
                'scale_thresholds':scale_random,'lookup_tables':tables,'source_input_id':snapshot['input_id'],
                'threshold_generator':generator},path)
            rows.append({'input_id':identity,'source_kind':'real_gradient','original_numel':logical.numel(),
                'padded_numel':padded.numel(),'dtype':'bfloat16','shape':[logical.numel()],'byte_order':'little',
                'input_sha256':tensor_identity(logical)['sha256'],'padded_input_sha256':tensor_identity(padded)['sha256'],
                'fixture_path':str(path),'fixture_sha256':sha(path),'fixture_bytes':path.stat().st_size,
                'generator_or_loader':"splcomm.p2_replay.load_inputs(calibration_root, 'pilot')",
                'generator_config':{'calibration_root':str(calibration),'thresholds':generator},
                'source_path':snapshot['checkpoint'],'source_sha256':snapshot['checkpoint_sha256'],
                'source_manifest_sha256':snapshot['manifest_sha256'],'source_input_id':snapshot['input_id'],
                'source_gradient_sha256':tensor_identity(gradient)['sha256'],'partition':'development',
                'index_semantics':'before_update_t_one_based','training_seed':snapshot['training_seed'],
                'update_step':snapshot['update'],'rank':rank,
                'transform_order':['FP32 saved worker gradient','to(BF16)','divide by 4 in BF16','pad zeros to 256'],
                'main_thresholds_sha256':generator['main']['sha256'],'scale_thresholds_sha256':generator['scale']['sha256'],
                'lut_sha256_by_bits':{b:entry['sha256'] for b,entry in lut.items()},
                'expected_packet_bytes_by_bits':{str(b):wire_size(padded.numel(),b) for b in (4,8)},
                'GPU_codec_executed':False})
    assert len(rows)==12 and {(r['update_step'],r['rank']) for r in rows}=={(s,r) for s in (100,500,2500) for r in range(4)}
    save_json(run/'input_manifest.json',{'schema_version':'gpu_codec_v1.0','template_only':False,
        'run_id':run.name,'created_at':started,'scope':'G0_real_input_availability_and_fixture_freeze_only',
        'inputs':rows,'missing_G1_fixture_categories':['edge','synthetic'],'holdout_opened':False})
    save_json(run/'fixtures_manifest.json',{'schema_version':'gpu_codec_v1.0','run_id':run.name,
        'local_root':str(directory),'large_files_retained_locally':True,
        'reconstruction_command':f'scripts/python.ps1 scripts/freeze_g0_inputs.py --run-dir <fresh-run-directory>',
        'reconstruction_note':'fresh output path; source identities and tensor hashes are deterministic, torch archive file bytes may vary with basename',
        'lookup_tables':{'path':str(table_path),'sha256':sha(table_path),'bytes':table_path.stat().st_size,'by_bits':lut},
        'files':[{'path':r['fixture_path'],'sha256':r['fixture_sha256'],'bytes':r['fixture_bytes']} for r in rows]})
    save_json(run/'cpu_reference_manifest.json',{'schema_version':'gpu_codec_v1.0','run_id':run.name,
        'upstream_commit':'96da2ee480bbfd8e2049de3fbd8595cc9d7a30b8','reference_files':source,
        'adapter':{'path':'src/gradient_comm_lab/cpu_reference.py','sha256':sha(ROOT/'src/gradient_comm_lab/cpu_reference.py')},
        'lookup_tables':lut,'reference_modified':False,'explicit_thresholds_do_not_advance_LCG':True,
        'CPU_or_GPU_codec_parity_test_executed':False})
    print(json.dumps({'run_id':run.name,'real_gradient_inputs_prepared':len(rows),
        'holdout_opened':False,'fixture_root':str(directory),'lookup_tables':lut}))


if __name__=='__main__': main()
