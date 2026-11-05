"""Build exactly one epsilon diagnostic condition in a fresh Python process.

Copies the sealed, actually compiled source. Only the optimized BF16 MEE
compress raw-scale addend is parameterized, only for NBITS==8. No kernels run.
"""
from __future__ import annotations
import argparse,difflib,hashlib,json,os,platform,shutil,sys,time,traceback
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/bounded_discovery_v1/bounded_discovery_v1_20261104_001'
COMMIT='96da2ee480bbfd8e2049de3fbd8595cc9d7a30b8'
sys.path.insert(0,str(ROOT/'scripts'))
from probe_toolchain_v2 import configure,command

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def dump(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def now():return datetime.now().astimezone().isoformat()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',required=True)
    ap.add_argument('--condition',required=True,choices=('control_add1e7','diagnostic_add0'))
    args=ap.parse_args();run=(ROOT/args.run_dir).resolve()
    if not run.is_relative_to((ROOT/'results/epsilon_diagnostic_v1').resolve()) or run.is_relative_to(BASE.resolve()):
        raise ValueError('Output must be a new epsilon_diagnostic_v1 run, never a sealed baseline')
    if not run.is_dir():raise ValueError('Explicit new run directory must already exist')
    build=run/'build'/args.condition;manifest=run/'logs'/f'build_{args.condition}.json'
    if build.exists() or manifest.exists():raise FileExistsError('No overwrite of condition build or evidence')
    other_manifests=[p for p in (run/'logs').glob('build_*.json')
                     if p.stem in ('build_control_add1e7','build_diagnostic_add0')]
    if len(other_manifests)>=2:raise ValueError('The two-condition build limit is exhausted')
    literal={'control_add1e7':'1e-7f','diagnostic_add0':'0.0f'}[args.condition]
    prior_path=BASE/'logs/official_build_a03.json';prior=read(prior_path)
    assert prior['status']=='compiled_and_loaded' and prior['official_commit']==COMMIT
    assert sha(prior['module_path'])==prior['module_sha256']
    upstream=Path(prior['source_directory']).resolve();assert upstream.is_relative_to(BASE.resolve())
    base_contract=read(BASE/'experiment_contract.json')
    assert sha(BASE/'experiment_contract.json')=='97e8595532125d1fed44956c6a70da959031581884e753b449b64c1b3b195f47'
    base_sources={str(p.relative_to(upstream)):sha(p) for p in upstream.rglob('*') if p.is_file()}
    assert base_sources==prior['source_hashes']
    build.mkdir(parents=True,exist_ok=False);destination=build/'sources';shutil.copytree(upstream,destination)
    shutil.copy2(Path(__file__),build/'build_conditions.py')
    target=destination/'csrc/cuda_hierarchical_mee.cu';before=target.read_text(encoding='utf-8')
    anchor='template<int NBITS>\n__global__ void scaling_compress_kernel_mee_hierarchical_bf16_opt('
    assert before.count(anchor)==1 and sha(upstream/'csrc/cuda_hierarchical_mee.cu')=='fd25698435e7660be5d42a92b66bf091e3334d5dde93a8685a3f25edb2922b41'
    begin=before.index(anchor);end=before.index('\ntemplate<int NBITS, bool ADD>',begin)
    old='    chunk_scale = scaling_hier_positive_scale(chunk_scale / scaling_hier_positive_scale(chunk_max_mee) + 1e-7f);'
    new='    chunk_scale = scaling_hier_positive_scale(chunk_scale / scaling_hier_positive_scale(chunk_max_mee) + (NBITS == 8 ? DIAG_B8_SCALE_ADD : 1e-7f));'
    function=before[begin:end];assert function.count(old)==1
    macro='#ifndef DIAG_B8_SCALE_ADD\n#define DIAG_B8_SCALE_ADD 1e-7f\n#endif\n\n'
    after=before[:begin]+macro+function.replace(old,new)+before[end:]
    assert after.replace(macro,'',1).replace(new,old,1)==before
    target.write_text(after,encoding='utf-8',newline='\n')
    diff=build/'source_intervention.diff'
    diff.write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='sealed_actual/csrc/cuda_hierarchical_mee.cu',tofile='derived/csrc/cuda_hierarchical_mee.cu')),encoding='utf-8')
    source_hashes={str(p.relative_to(destination)):sha(p) for p in destination.rglob('*') if p.is_file()}
    changed=[k for k in base_sources if base_sources[k]!=source_hashes[k]]
    assert changed==['csrc\\cuda_hierarchical_mee.cu']
    flags=list(prior['extra_cuda_cflags'])+[f'-DDIAG_B8_SCALE_ADD={literal}']
    record=dict(schema_version='epsilon_condition_build_v1',run_id=run.name,condition_id=args.condition,
        started_at=now(),command=[sys.executable,*sys.argv],pid=os.getpid(),status='running',
        module_name='eden_utils',official_commit=COMMIT,derived_identity=True,
        origin='Sealed actual compiled source of bounded_discovery_v1; derived macro-parameterized rebuild, not an unmodified upstream binary',
        scale_add=1e-7 if args.condition=='control_add1e7' else 0.0,scale_add_literal=literal,
        nominal_api='eden_utils.scaling_compress',actual_kernel='scaling_compress_kernel_mee_hierarchical_bf16_opt<8>',
        expected_runtime_configuration=dict(dtype='bfloat16',nbits=8,chunk_size=16,strategy=3),
        base_run=str(BASE),base_contract_sha256=sha(BASE/'experiment_contract.json'),base_build_manifest=str(prior_path),base_build_manifest_sha256=sha(prior_path),
        base_module_sha256=prior['module_sha256'],base_source_directory=str(upstream),base_source_hashes=base_sources,
        source_directory=str(destination),source_hashes=source_hashes,source_diff_path=str(diff),source_diff_sha256=sha(diff),
        changed_source_files=changed,single_factor_patch_inverse_check=True,unmodified_other_bits_addend='1e-7f',mee_eps8_unchanged=0.1,
        unchanged_protections=['positive_scale 1e-30 floor','finite checks','BF16 super-scale storage','UINT8 ceil and clamp','main-code clamp','RNG interpretation','codebooks','decoder'],
        extra_cflags=prior['extra_cflags'],extra_cuda_cflags=flags,extra_include_paths=prior['extra_include_paths'],
        macro_is_only_flag_difference=True,official_math_flags_preserved=True,build_directory=str(build),
        module_loaded=False,kernels_called=False,repair_round=0,build_source_sha256=sha(__file__),toolchain_helper_sha256=sha(ROOT/'scripts/probe_toolchain_v2.py'))
    dump(manifest,record);start=time.perf_counter()
    try:
        config=configure()
        import torch
        from torch.utils.cpp_extension import load
        env=dict(captured_at=now(),pid=os.getpid(),python=sys.version,executable=sys.executable,platform=platform.platform(),
                 torch=torch.__version__,torch_cuda=torch.version.cuda,configuration=config,
                 compiler_versions={k:command([config['path_resolution'][k]]+v) for k,v in [('nvcc',['--version']),('cl',[]),('ninja',['--version'])]},
                 nvidia_smi=command(['nvidia-smi','--query-gpu=name,driver_version,compute_cap,memory.total','--format=csv,noheader']))
        dump(run/'logs'/f'build_environment_{args.condition}.json',env)
        assert str(Path(config['msvc_root'])/'bin/Hostx64/x64')==prior['extra_cuda_cflags'][prior['extra_cuda_cflags'].index('-ccbin')+1]
        for inc in prior['extra_include_paths']:assert (Path(inc)/'curand_kernel.h').is_file()
        sources=[destination/rel for rel in ('eden_utils.cpp','csrc/cuda_hadamard.cu','csrc/cuda_packing.cu','csrc/cuda_hierarchical_mee.cu')]
        module=load(name='eden_utils',sources=[str(p) for p in sources],build_directory=str(build),
                    extra_cflags=prior['extra_cflags'],extra_cuda_cflags=flags,extra_include_paths=prior['extra_include_paths'],with_cuda=True,verbose=True)
        record.update(status='compiled_and_loaded',module_loaded=True,module_path=module.__file__,module_sha256=sha(module.__file__))
        assert Path(module.__file__).resolve().is_relative_to(build.resolve())
    except Exception as exc:
        record.update(status='build_failed',exception=repr(exc),traceback=traceback.format_exc());raise
    finally:
        record.update(finished_at=now(),elapsed_seconds=time.perf_counter()-start);dump(manifest,record)
        print(json.dumps({k:record.get(k) for k in ('condition_id','status','pid','elapsed_seconds','module_path','module_sha256')},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
