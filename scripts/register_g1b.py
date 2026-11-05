"""Bind G1b's final development sources before formal correctness or timing."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'results/gpu_codec_v1/gpu_codec_v1_20261101_002'
BASELINE_SHA = '50bc272876be54db5d42327cce144022b7bf960d82073347479da2eb53f1d794'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p, value):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
def now(): return datetime.now(timezone(timedelta(hours=8))).isoformat()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    args=parser.parse_args(); run=(ROOT/args.run_dir).resolve()
    assert run.is_relative_to(ROOT/'results/gpu_codec_v1') and run != OLD
    assert not (run/'codec_contract.json').exists(), 'Do not rewrite frozen contract'
    assert sha(ROOT/'src/gradient_comm_lab/torch_codec.py') == BASELINE_SHA
    for item in read(OLD/'artifact_integrity.json')['files']:
        p=OLD/item['path']; assert sha(p)==item['sha256'] and p.stat().st_size==item['bytes'], str(p)
    for item in read(OLD/'source_manifest.json')['files']:
        if item.get('original_path'): assert sha(item['original_path'])==item['sha256'], item['original_path']
    inputs=read(run/'input_manifest.json')['inputs']
    oldinputs=read(OLD/'input_manifest.json')['inputs']
    assert inputs==oldinputs and len(inputs)==82
    for row in inputs: assert sha(row['fixture_path'])==row['fixture_sha256'], row['input_id']
    names={
        'src/gradient_comm_lab/torch_codec.py':'gpu_source',
        'src/gradient_comm_lab/cuda_codec.py':'gpu_source',
        'src/gradient_comm_lab/cpu_reference.py':'cpu_reference',
        'src/gradient_comm_lab/fixtures_v2.py':'config',
        'scripts/check_torch_codec.py':'test',
        'scripts/check_cuda_codec.py':'test',
        'tests/test_cuda_codec.py':'test',
        'configs/g1b_correctness_cases.json':'config',
        'scripts/build_cuda_codec.py':'build_script',
        'scripts/probe_toolchain_v2.py':'build_script',
        'scripts/benchmark_torch_codec.py':'build_script',
        'scripts/benchmark_cuda_pair.py':'build_script',
        'scripts/register_g1b.py':'build_script',
        'scripts/python.ps1':'build_script',
    }
    for p in (ROOT/'src/gradient_comm_lab/csrc').glob('*'):
        if p.is_file() and p.suffix in ('.cpp','.cu','.h','.cuh'):
            names[p.relative_to(ROOT).as_posix()]='gpu_source'
    hashes={name:sha(ROOT/name) for name in names}
    build=read(run/'logs/cuda_build_manifest.json')
    assert sha(build['module_path'])==build['module_sha256'], 'Actual extension binary changed'
    for path,expected in build['source_sha256'].items():
        assert sha(path)==expected,path
        name=Path(path).name
        original=(ROOT/'src/gradient_comm_lab/csrc'/name if name in ('codec.cpp','codec.cu') else
                  ROOT/'src/gradient_comm_lab'/name if name=='cuda_codec.py' else ROOT/'scripts'/name)
        assert sha(original)==expected,str(original)
    contract=read(OLD/'codec_contract.json')
    review='reviews/gpu_codec_v1_20261101_002_review_and_g1b_plan.md'
    contract.update(run_id=run.name, contract_id='dynamiq_local_repaired_explicit_thresholds_cuda_pair_v1',
        substage='G1b_cuda_optimized_paired', freeze_status='frozen_actual_dual_GPU_implementation',
        inherited_format_contract={'run_id':OLD.name,'path':str(OLD/'codec_contract.json'),'sha256':sha(OLD/'codec_contract.json')},
        execution_dependency_amendment={'path':review,'sha256':sha(ROOT/review)},
        missing_full_G1_methods=[],fixture_manifest_sha256=sha(run/'input_manifest.json'),
        changes_from_plan=[],
        checked_api_implementation='Each method implements checked=True separately; metadata rejection and checked device error states are validated by exact expected exception type and message.',
        optimized_build_manifest={'path':'logs/cuda_build_manifest.json','sha256':sha(run/'logs/cuda_build_manifest.json')},
        optimized_implementation_note='Native direct 5/9-bit payload ownership and local BF16 encode/decode; stream and ABI behavior qualified separately. All exact source hashes bound before formal execution.')
    contract['gpu_implementations'].append({
        'method_id':'cuda_optimized_repaired','source':'src/gradient_comm_lab/cuda_codec.py',
        'source_sha256':hashes['src/gradient_comm_lab/cuda_codec.py'],
        'native_sources':{p:h for p,h in hashes.items() if '/csrc/' in p},
        'compiled_extension':True,'graph_mode':'off','compile_mode':'eager',
        'build_manifest':'logs/cuda_build_manifest.json',
        'packet_storage':'Three independent tight GPU buffers in identical payload/uint8-scale/BF16-top order',
        'stream':'Calling PyTorch current CUDA stream on tensor device; independently tested with stream events',
        'rounding':'Round float arithmetic to BF16 at each frozen CPU operation boundary; explicit BF16 constants; no fast math or FMA substitution',
        'empty_input':'Outer adapter skips native arithmetic, empty packet and output'})
    contract['planned_GPU_compiler_policy'].update(note='Actual flags and binary identity are bound in optimized_build_manifest.',actual_codec_compiler_flags=build.get('extra_cuda_cflags',build.get('cuda_flags')))
    contract['compiler_flags']=build.get('extra_cuda_cflags',build.get('cuda_flags',[]))
    contract['input_validation_boundary']['api_allocating']='Both methods create complete per-call plan including output, packet and owned workspace; constructor and checked validation costs included by the selected mode.'
    write(run/'codec_contract.json',contract)
    registration={'schema_version':'gpu_codec_v1.0','run_id':run.name,'registered_at':now(),
        'phase':'before_formal_correctness_and_performance','contract_sha256':sha(run/'codec_contract.json'),
        'input_manifest_sha256':sha(run/'input_manifest.json'),'source_hashes':hashes,'source_roles':names,
        'optimized_build_manifest_sha256':sha(run/'logs/cuda_build_manifest.json'),
        'correctness_case_manifest_sha256':sha(run/'correctness_case_manifest.json'),
        'accepted_baseline_source_sha256':BASELINE_SHA,
        'methods':['torch_cuda_repaired','cuda_optimized_repaired'],
        'scope':'G1 fixed local codec only. G2 is not executed.'}
    write(run/'execution_registration.json',registration)
    for name in names:
        dest=run/'code'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    print(json.dumps({'registered':run.name,'contract_sha256':registration['contract_sha256'],
        'frozen_sources':len(hashes),'inputs':len(inputs),'baseline_source_sha256':BASELINE_SHA}))

if __name__=='__main__': main()
