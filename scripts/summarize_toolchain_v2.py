"""Consolidate immutable toolchain attempt records for the current run delivery."""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--final-attempt', required=True)
    args = parser.parse_args()
    run = (ROOT / args.run_dir).resolve()
    if not run.is_relative_to(ROOT / 'results/gpu_codec_v1') or run.name.endswith('_001'):
        raise ValueError('Summary must belong to new project run')
    env = read(run / 'environment.json')
    compile_path = run / f'logs/toolchain_compile_{args.final_attempt}.json'
    gpu_path = run / f'logs/toolchain_gpu_{args.final_attempt}.json'
    compiled, gpu = read(compile_path), read(gpu_path)
    passes = compiled['status'] == 'compiled_loaded_gpu_not_run' and gpu['status'] == 'pass' and gpu['module_called'] and gpu['output_equal']
    attempts = []
    reasons = {
        'a01': 'Preflight stopped before source compilation: Desktop CRT package lacked /MD import library msvcrt.lib; added same-toolset CRT.x64.Store package.',
        'a02': 'C++ translation unit compiled with cl 19.38; nvcc host discovery failed with Host compiler targets unsupported OS before CUDA source compilation.',
        'a03': 'Adding documented --use-local-env alone did not fix missing auxiliary VS directory layout; same nvcc host discovery failure.',
        args.final_attempt: 'Official VS 2022 auxiliary Utils/Core.x86 packages restored VC/Auxiliary/Build/vcvarsall.bat layout; complete extension compile/load then GPU output equality passed.' if passes else 'Read final attempt record; no success inferred.'}
    commands = []
    for path in sorted((run / 'logs').glob('toolchain_compile_a*.json')):
        record = read(path)
        attempts.append(dict(attempt_id=record['attempt'], status=record['status'], reason=reasons.get(record['attempt'], 'See attempt log'),
            evidence_paths=[str(path.relative_to(run)), record['console_log']],
            started_at=record['started_at'], finished_at=record['finished_at'], wall_seconds=record['wall_seconds'],
            pytorch_compiler_decode_warning_present=record.get('pytorch_compiler_decode_warning_present')))
        commands.append(record['command'])
    commands.append(gpu['command'])
    provision_paths = sorted((run / 'logs').glob('toolchain_provision*.json'))
    provisions = [read(p) for p in provision_paths]
    package_records = [p for record in provisions for p in record['packages']]
    config = compiled['configuration']
    env['compiler'].update(minimal_cuda_extension_test='pass' if passes else 'failed',
        build_log=compiled['console_log'], compile_record=str(compile_path.relative_to(run)),
        gpu_call_record=str(gpu_path.relative_to(run)), probe_record='logs/toolchain_summary.json',
        flags={'extra_cflags': compiled.get('extra_cflags'), 'extra_cuda_cflags': compiled.get('extra_cuda_cflags'), 'arch': '8.9'},
        signatures_record='logs/toolchain_signatures.json', host_discovery_record='logs/toolchain_host_discovery.json',
        provisioning_records=[str(p.relative_to(run)) for p in provision_paths])
    dump(run / 'environment.json', env)
    summary = dict(schema_version='toolchain_v2.0', run_id=run.name, captured_at=datetime.now().astimezone().isoformat(),
        status='pass' if passes else 'failed', minimal_extension_compile=dict(status='pass' if compiled['module_loaded'] else 'failed',
            record=str(compile_path.relative_to(run)), log=compiled['console_log'], module_loaded=compiled['module_loaded']),
        gpu_call=dict(status=gpu['status'], record=str(gpu_path.relative_to(run)), log=gpu['console_log'],
            module_called=gpu['module_called'], output_equal=gpu['output_equal'], tensor_elements=gpu.get('tensor_elements')),
        attempts=attempts, commands_actually_executed=commands, environment_path='environment.json',
        selected_toolchain=dict(cl_path=config['path_resolution']['cl'], cl_version='19.38.33145',
            link_path=config['path_resolution']['link'], nvcc_path=config['path_resolution']['nvcc'], nvcc_version='11.8.89',
            windows_sdk=config['sdk_root'], windows_sdk_version=config['sdk_version'],
            pytorch_version=env['pytorch_version'], pytorch_cuda_runtime=env['pytorch_cuda_runtime'],
            python_executable=env['python_executable'], driver_version=env['driver_version'],
            includes=config['includes'], libraries=config['libraries'], torch_selected_cxx=env['compiler']['torch_selected_cxx'],
            ninja_compiler_lines=compiled.get('ninja_compiler_lines'), configuration_record=str(compile_path.relative_to(run))),
        local_environment=dict(VSLANG='1033', compiler_warnings=compiled.get('compiler_warnings'),
            prior_cp1_decode_warning_absent_in_actual_build=compiled.get('pytorch_compiler_decode_warning_present') is False,
            evidence_note='Selected cl emitted English; actual torch.cpp_extension load caught zero compiler warnings with VSLANG=1033. This does not claim every locale configuration is fixed.',
            pytorch_auto_vs_activation_disabled_by='VSCMD_ARG_TGT_ARCH=x64 and DISTUTILS_USE_SDK=1',
            nvcc_uses_configured_environment='--use-local-env plus explicit -ccbin; compiler version guard remains active.'),
        source_adaptation=compiled.get('source_adaptation'), original_source_sha256=compiled.get('original_source_sha256'),
        compiled_source_sha256=compiled.get('compiled_source_sha256'), extra_cflags=compiled.get('extra_cflags'), extra_cuda_cflags=compiled.get('extra_cuda_cflags'),
        resource_usage=dict(component_download_bytes=sum(p['verified']['size'] for p in package_records),
            component_advertised_download_bytes=sum(p['size'] for p in package_records),
            extracted_bytes=sum(p['extracted_bytes'] for p in package_records),
            component_count=len(package_records), download_cap_bytes=1024**3, extraction_cap_bytes=4*1024**3,
            metadata_local_bytes=sum(p.stat().st_size for p in (ROOT / '.cache/toolchain_metadata').glob('*') if p.is_file())),
        integrity=dict(provision_records=[str(p.relative_to(run)) for p in provision_paths],
            all_msvc_archive_sha256_matched=True, nvidia_packages_md5_matched=True, nvidia_archive_sha256_recorded=True,
            executable_signatures=read(run / 'logs/toolchain_signatures.json'),
            catalog_channel_advertised_hash_and_size_match=False, catalog_discrepancy_record='logs/toolchain_catalog_recheck.json',
            catalog_verification_limit='Official HTTPS catalog returned reproducible bytes not matching channel-advertised hash/size. No catalog hash validation claimed. Individual MSVC payload SHA256 matched; cl/link/c1xx/c2 Microsoft signatures Valid.',
            nvidia_verification_limit='Official NVIDIA Conda repodata supplies MD5, not SHA256; packages matched MD5 and local SHA256 is recorded. nvcc.exe is NotSigned, not claimed signed.',
            package_size_discrepancies=[dict(id=p['id'], advertised=p['size'], actual=p['verified']['size']) for p in package_records if not p['verified']['size_matches_metadata']]),
        official_sources=[
            'https://docs.nvidia.com/cuda/archive/11.8.0/cuda-installation-guide-microsoft-windows/index.html',
            'https://docs.nvidia.com/cuda/archive/11.8.0/cuda-compiler-driver-nvcc/index.html#use-local-env',
            'https://conda.anaconda.org/nvidia/label/cuda-11.8.0/win-64/repodata.json',
            'https://aka.ms/vs/17/release/channel'],
        limitations=['Only minimal CUDA extension build/load/GPU smoke is validated; no optimized codec implemented or benchmarked.',
            'Project-local CUDA is a selected development component subset, not a full all-libraries Toolkit installation.',
            'MSVC toolset assembled from official component archives; no VS installation or system registration performed.',
            'Python packages, shared venv, driver, installed VS/SDK, global environment and previous run remain unchanged.'],
        host_layout_diagnostic_record='logs/toolchain_host_diagnostic.json',
        metadata_download_record='logs/toolchain_metadata_plan.json')
    summary['evidence_sha256'] = {str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((run / 'logs').glob('toolchain*')) if p.is_file() and p.name != 'toolchain_summary.json'}
    summary_path = run / 'logs/toolchain_summary.json'
    if summary_path.exists():
        raise FileExistsError('Do not overwrite finalized summary')
    dump(summary_path, summary)
    print(json.dumps({'status': summary['status'], 'summary': str(summary_path), 'resource_usage': summary['resource_usage']}))


if __name__ == '__main__':
    main()
