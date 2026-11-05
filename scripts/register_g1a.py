"""Freeze an actual eager CUDA implementation before formal acceptance/timing."""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'results/gpu_codec_v1/gpu_codec_v1_20261101_001'
SOURCES={
 'src/gradient_comm_lab/torch_codec.py':'gpu_source',
 'src/gradient_comm_lab/cpu_reference.py':'cpu_reference',
 'src/gradient_comm_lab/fixtures_v2.py':'config',
 'scripts/check_torch_codec.py':'test',
 'tests/test_torch_codec.py':'test',
 'scripts/benchmark_torch_codec.py':'build_script',
 'scripts/register_g1a.py':'build_script',
 'scripts/python.ps1':'build_script',
 'handoff_templates/benchmark_config.json':'config',
}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False)
def now():return datetime.now().astimezone().isoformat()

def verify_old():
    checks=[]
    for item in read(OLD/'artifact_integrity.json')['files']:
        p=OLD/item['path']
        assert sha(p)==item['sha256'] and p.stat().st_size==item['bytes'],str(p)
        checks.append(item['path'])
    contract=read(OLD/'codec_contract.json')
    for item in contract['cpu_reference_files']:
        assert sha(item['path'])==item['sha256'],item['path']
    assert sha(ROOT/contract['cpu_adapter']['path'])==contract['cpu_adapter']['sha256']
    return {'verified_at':now(),'old_run':OLD.name,'old_artifact_integrity_sha256':sha(OLD/'artifact_integrity.json'),
        'old_sealed_file_count':len(checks),'old_cpu_reference_file_count':len(contract['cpu_reference_files']),
        'status':'all_match','files':checks}

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);args=p.parse_args()
    run=(ROOT/args.run_dir).resolve()
    if not run.is_relative_to(ROOT/'results/gpu_codec_v1') or run==OLD:raise ValueError('New independent run required')
    if (run/'codec_contract.json').exists():raise FileExistsError('Do not overwrite a frozen contract')
    verification=verify_old();manifest=read(run/'input_manifest.json')
    for row in manifest['inputs']:
        assert sha(row['fixture_path'])==row['fixture_sha256'],row['input_id']
    oldreal={r['input_id']:r for r in read(OLD/'input_manifest.json')['inputs']}
    real=[r for r in manifest['inputs'] if r['source_kind']=='real_gradient']
    assert len(real)==12
    for row in real:
        previous=oldreal[row['input_id']]
        for key in ('input_sha256','padded_input_sha256','fixture_path','fixture_sha256','main_thresholds_sha256','scale_thresholds_sha256'):
            assert row[key]==previous[key],(row['input_id'],key)
    hashes={name:sha(ROOT/name) for name in SOURCES}
    review='reviews/gpu_codec_v1_20261101_001_review.md'
    contract=read(OLD/'codec_contract.json')
    contract.update(contract_id='dynamiq_local_repaired_explicit_thresholds_torch_cuda_eager_v1',run_id=run.name,
        stage='G1',substage='torch_cuda_baseline',freeze_status='frozen_actual_torch_cuda_implementation',
        inherited_format_contract={'run_id':OLD.name,'path':str(OLD/'codec_contract.json'),'sha256':sha(OLD/'codec_contract.json')},
        execution_dependency_amendment={'path':review,'sha256':sha(ROOT/review)},
        gpu_implementations=[{'method_id':'torch_cuda_repaired','source':'src/gradient_comm_lab/torch_codec.py',
            'source_sha256':hashes['src/gradient_comm_lab/torch_codec.py'],'runtime':'PyTorch 2.4.1+cu118 CUDA eager',
            'compiled_extension':False,'graph_mode':'off','compile_mode':'eager',
            'native_kernels':'Existing PyTorch binary CUDA kernels; no user nvcc invocation',
            'packet_storage':'Three independent tight GPU buffers; serialization order unchanged; validation concatenation excluded from timing',
            'workspace':'Full plan preallocation including integer packing indices, arithmetic arrays and scalar constants',
            'empty_input':'No arithmetic quantization; zero-byte packet and zero-length BF16 view'}],
        checked_api_implementation='TorchCodecPlan encode_into/decode_into/roundtrip_into checked=True; explicit device predicates synchronize',
        input_validation_boundary={'trusted_inputs':'Host tensor metadata checks retained, no device-value inspection; prior CPU and checked GPU parity required',
            'checked_api':'Includes finite/range/layout/alias/LUT checks, BF16 intermediate guards and packet alphabet checks',
            'api_allocating':'Create complete TorchCodecPlan and all workspace/indices/constants per call, then execute roundtrip',
            'alias_precondition':'External input/threshold/LUT tensors must not alias plan-owned storage; checked API rejects storage aliasing',
            'LUT_identity':'Prepared fixed CPU-reference LUT; checked API checks metadata/finite/ordering, fixture identity is verified by harness'},
        compiler_flags=[],changes_from_plan=[{'kind':'execution_dependency_only','source':review,
            'description':'Allow eager PyTorch CUDA G1a while custom CUDA compilation is separately investigated; full G1 still incomplete'}],
        missing_full_G1_methods=['cuda_optimized_repaired'],
        fixture_manifest_sha256=sha(run/'input_manifest.json'),
        future_contract_policy='Adding or changing a GPU method requires a new run and frozen source binding; retain this attempt unchanged')
    contract['bf16_rounding_semantics']['CPU_probe']=str(OLD/'logs/cpu_reference_semantics.json')
    contract['planned_GPU_compiler_policy']['note']='Applies to a future custom codec; this eager method uses the installed PyTorch binary. Toolchain smoke is not a codec implementation.'
    write(run/'codec_contract.json',contract)
    registration={'schema_version':'gpu_codec_v1.0','run_id':run.name,'registered_at':now(),
        'phase':'before_formal_correctness_and_performance','contract_sha256':sha(run/'codec_contract.json'),
        'input_manifest_sha256':sha(run/'input_manifest.json'),'source_hashes':hashes,'source_roles':SOURCES,
        'scope':'torch_cuda_repaired only; toolchain discovery/installation/probe cannot change this codec contract'}
    write(run/'execution_registration.json',registration)
    write(run/'logs/prior_run_integrity_before.json',verification)
    for name in SOURCES:
        dest=run/'code'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    print(json.dumps({'registered':run.name,'contract_sha256':registration['contract_sha256'],
        'frozen_sources':len(hashes),'inputs':len(manifest['inputs']),'unchanged_real_inputs':len(real)}))

if __name__=='__main__':main()
