"""Probe project-local CUDA 11.8/MSVC 19.38 with a full, process-local SDK environment.

Compile/load and GPU invocation are separate phases to coordinate benchmark isolation.
Original G0 sources and prior runs remain untouched. No unsupported-compiler bypass.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
import warnings

ROOT = Path(__file__).resolve().parents[1]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(argv):
    result = subprocess.run(argv, capture_output=True, timeout=60)
    return dict(command=argv, returncode=result.returncode,
        stdout=result.stdout.decode('utf-8', errors='replace'), stderr=result.stderr.decode('utf-8', errors='replace'),
        stdout_sha256=hashlib.sha256(result.stdout).hexdigest(), stderr_sha256=hashlib.sha256(result.stderr).hexdigest())


def configure():
    cuda = ROOT / '.toolchains/cuda-11.8'
    msvc = ROOT / '.toolchains/msvc-14.38/VC/Tools/MSVC/14.38.33130'
    cldir = msvc / 'bin/Hostx64/x64'
    sdk = Path('C:/Program Files (x86)/Windows Kits/10')
    sdk_version = '10.0.26100.0'
    ninja = Path('C:/Program Files/Microsoft Visual Studio/18/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja')
    includes = [msvc / 'include'] + [sdk / 'Include' / sdk_version / p for p in ('ucrt', 'shared', 'um', 'winrt', 'cppwinrt')]
    libraries = [msvc / 'lib/x64'] + [sdk / 'Lib' / sdk_version / p / 'x64' for p in ('ucrt', 'um')]
    required = [cuda / 'bin/nvcc.exe', cuda / 'include/cuda_runtime.h', cuda / 'lib/x64/cudart.lib',
        cldir / 'cl.exe', cldir / 'link.exe', msvc / 'include/vector', msvc / 'lib/x64/msvcrt.lib',
        sdk / 'Include' / sdk_version / 'um/Windows.h', sdk / 'Lib' / sdk_version / 'um/x64/kernel32.lib',
        sdk / 'Lib' / sdk_version / 'ucrt/x64/ucrt.lib', ninja / 'ninja.exe']
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError('Required toolchain files missing: ' + repr(missing))
    additions = [cldir, cuda / 'bin', sdk / 'bin' / sdk_version / 'x64', ninja]
    values = dict(CUDA_HOME=str(cuda), CUDA_PATH=str(cuda), CXX=str(cldir / 'cl.exe'),
        INCLUDE=os.pathsep.join(str(p) for p in includes), LIB=os.pathsep.join(str(p) for p in libraries),
        LIBPATH=os.pathsep.join(str(p) for p in libraries),
        VSLANG='1033', VSCMD_ARG_TGT_ARCH='x64', VSCMD_ARG_HOST_ARCH='x64', DISTUTILS_USE_SDK='1', MSSdk='1',
        VCToolsInstallDir=str(msvc) + '\\', VCINSTALLDIR=str(msvc.parents[2]) + '\\',
        WindowsSdkDir=str(sdk) + '\\', WindowsSDKVersion=sdk_version + '\\', UCRTVersion=sdk_version,
        TORCH_CUDA_ARCH_LIST='8.9', MAX_JOBS='1', PYTHONDONTWRITEBYTECODE='1')
    values['PATH'] = os.pathsep.join(str(p) for p in additions) + os.pathsep + os.environ.get('PATH', '')
    os.environ.update(values)
    # Prevent inherited ad hoc flags from changing this probe's compiler/toolkit.
    removed = {k: os.environ.pop(k) for k in ('CC', 'PYTORCH_NVCC', 'NVCC_PREPEND_FLAGS', 'NVCC_APPEND_FLAGS', 'CL', '_CL_', 'LINK') if k in os.environ}
    return dict(cuda_root=str(cuda), msvc_root=str(msvc), sdk_root=str(sdk), sdk_version=sdk_version,
        configured_environment=values, removed_inherited_overrides=removed,
        includes=[str(p) for p in includes], libraries=[str(p) for p in libraries],
        required_files={str(p): True for p in required}, path_resolution={p: shutil.which(p) for p in ('cl', 'link', 'nvcc', 'ninja')},
        selection_note='VSCMD_ARG_TGT_ARCH suppresses torch cpp_extension default VS activation; CXX and nvcc -ccbin pin project-local cl.',
        global_environment_modified=False, driver_changed=False, package_installation_performed=False)


def snapshot(run, config):
    import torch
    from torch.utils import cpp_extension
    smi = command(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.free,compute_cap,power.limit,power.draw,clocks.current.graphics,clocks.current.memory,utilization.gpu', '--format=csv,noheader,nounits'])
    parts = smi['stdout'].strip().split(', ')
    def number(index):
        try:
            return float(parts[index])
        except (ValueError, IndexError):
            return None
    env = json.loads((ROOT / 'handoff_templates/environment.json').read_bytes())
    env.update(template_only=False, run_id=run.name, captured_at=datetime.now().astimezone().isoformat(),
        os=dict(platform=platform.platform(), release=platform.release(), version=platform.version(), machine=platform.machine()),
        python_executable=sys.executable, python_version=platform.python_version(), pytorch_version=torch.__version__,
        pytorch_cuda_runtime=torch.version.cuda, driver_version=parts[1] if len(parts) > 1 else None,
        nccl_available=torch.distributed.is_nccl_available(), cuda_available=torch.cuda.is_available(),
        bf16_available=None, dependency_lock_path='requirements.txt')
    env['gpu'].update(name=parts[0] if parts else None, memory_total_bytes=int(number(2) * 1024**2) if number(2) else None,
        compute_capability=[int(x) for x in parts[4].split('.')] if len(parts) > 4 else None,
        power_limit_watts=number(5), clock_settings=dict(locked_clocks=None, current_graphics_mhz=number(7), current_memory_mhz=number(8)),
        other_process_load=dict(utilization_gpu_percent=number(9), memory_free_mib=number(3), power_draw_watts=number(6), query=smi))
    versions = {key: command([config['path_resolution'][name]] + flags) for key, name, flags in (
        ('nvcc', 'nvcc', ['--version']), ('host', 'cl', []), ('link', 'link', []), ('ninja', 'ninja', ['--version']))}
    env['compiler'].update(nvcc_path=config['path_resolution']['nvcc'], nvcc_version=versions['nvcc'],
        host_compiler_path=config['path_resolution']['cl'], host_compiler_version=versions['host'],
        link_path=config['path_resolution']['link'], link_version=versions['link'], ninja_path=config['path_resolution']['ninja'],
        ninja_version=versions['ninja'], torch_detected_cuda_home=cpp_extension.CUDA_HOME,
        torch_selected_cxx=cpp_extension.get_cxx_compiler(), host_toolkit_compatibility='MSVC 193x in CUDA 11.8 support table; actual compile pending',
        configuration=config, provisioning_record='logs/toolchain_provision.json',
        metadata_integrity_note='VS catalog channel-advertised size/hash discrepancy retained in logs/toolchain_catalog_recheck.json; individual MSVC archives SHA256 checked, executable signatures checked separately.')
    env['notes'] = ['Fresh run002 snapshot; no previous environment snapshot copied.',
        'Shared Python, installed SDK and Ninja are reused read-only; CUDA and MSVC components extracted under this project.',
        'No driver upgrade, global environment edit, compiler guard bypass, or modification of old venv/run.',
        'GPU statistics are instantaneous discovery readings, not benchmark conditions.']
    dump(run / 'environment.json', env)
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--attempt', default='a01')
    parser.add_argument('--phase', choices=('compile', 'gpu'), default='compile')
    args = parser.parse_args()
    if not args.attempt.isalnum():
        raise ValueError('Attempt must be alphanumeric')
    run = (ROOT / args.run_dir).resolve()
    if not run.is_relative_to(ROOT / 'results/gpu_codec_v1') or run.name.endswith('_001'):
        raise ValueError('Probe requires a new run in this project')
    recordpath = run / f'logs/toolchain_{args.phase}_{args.attempt}.json'
    if recordpath.exists():
        raise FileExistsError('Use a fresh attempt; do not overwrite evidence')
    record = dict(run_id=run.name, phase=args.phase, attempt=args.attempt, started_at=datetime.now().astimezone().isoformat(),
        command=[sys.executable] + sys.argv, status='running', module_loaded=False, module_called=False, output_equal=None,
        unsupported_compiler_bypass=False, exception=None)
    start = time.perf_counter()
    env = None
    try:
        config = configure()
        record['configuration'] = config
        import torch
        from torch.utils.cpp_extension import load
        build = run / f'build/toolchain_{args.attempt}'
        name = 'gradient_comm_lab_g0_probe_v2'
        record.update(build_directory=str(build), module_name=name)
        if args.phase == 'compile':
            env = snapshot(run, config)
            build.mkdir(parents=True, exist_ok=True)
            original_cpp, original_cu = ROOT / 'src/g0_probe.cpp', ROOT / 'src/g0_probe.cu'
            cuda_source = original_cu.read_text(encoding='utf-8').replace('#include <ATen/cuda/CUDAContext.h>', '#include <c10/cuda/CUDAStream.h>').replace('at::cuda::getCurrentCUDAStream()', 'c10::cuda::getCurrentCUDAStream()')
            generated = build / 'g0_probe_v2.cu'
            generated.write_text(cuda_source, encoding='utf-8')
            sources = [original_cpp, generated]
            flags = ['-O3', '--fmad=false', '--ftz=false', '--prec-div=true', '--prec-sqrt=true', '--use-local-env', '-ccbin', str(Path(config['msvc_root']) / 'bin/Hostx64/x64')]
            record.update(original_source_sha256={str(p.relative_to(ROOT)): digest(p) for p in (original_cpp, original_cu)},
                compiled_source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources},
                source_adaptation='Only CUDA current-stream include/namespace narrowed to c10; same G0 increment kernel and validation, original files unchanged.',
                extra_cflags=['/O2'], extra_cuda_cflags=flags, with_cuda=True)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                try:
                    module = load(name=name, sources=[str(p) for p in sources], build_directory=str(build),
                        extra_cflags=['/O2'], extra_cuda_cflags=flags, with_cuda=True, verbose=True)
                finally:
                    record['compiler_warnings'] = [str(w.message) for w in caught]
                    record['pytorch_compiler_decode_warning_present'] = any('decode' in w.lower() or 'codec' in w.lower() for w in record['compiler_warnings'])
            record.update(module_loaded=True, status='compiled_loaded_gpu_not_run')
        else:
            env = json.loads((run / 'environment.json').read_bytes())
            prior = json.loads((run / f'logs/toolchain_compile_{args.attempt}.json').read_bytes())
            if prior['status'] != 'compiled_loaded_gpu_not_run':
                raise RuntimeError('Compile/load phase has not passed')
            filepath = build / (name + '.pyd')
            spec = importlib.util.spec_from_file_location(name, filepath)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            record['module_loaded'] = True
            x = torch.arange(256, device='cuda', dtype=torch.float32) / 8
            y = module.increment(x)
            torch.cuda.synchronize()
            record.update(module_called=True, output_equal=bool(torch.equal(y, x + 1)), tensor_elements=256)
            if not record['output_equal']:
                raise AssertionError('Minimal CUDA probe output mismatch')
            record['status'] = 'pass'
            env['bf16_available'] = torch.cuda.is_bf16_supported()
            env['gpu']['memory_total_bytes'] = torch.cuda.get_device_properties(0).total_memory
        ninja = build / 'build.ninja'
        if ninja.exists():
            record['build_ninja_sha256'] = digest(ninja)
            record['ninja_compiler_lines'] = [line for line in ninja.read_text(encoding='utf-8').splitlines() if line.startswith(('cxx =', 'nvcc =', 'cuda_cflags =', 'ldflags =', '  command ='))]
    except Exception as exc:
        record['status'] = 'failed'
        record['exception'] = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        traceback.print_exc()
    record.update(finished_at=datetime.now().astimezone().isoformat(), wall_seconds=time.perf_counter() - start,
        console_log=f'logs/toolchain_{args.phase}_{args.attempt}.log')
    dump(recordpath, record)
    if env is not None:
        env['compiler'].update(minimal_cuda_extension_test=record['status'], build_log=record['console_log'], probe_record=str(recordpath.relative_to(run)))
        if record['status'] == 'pass':
            env['compiler']['host_toolkit_compatibility'] = 'Actual CUDA 11.8 + MSVC 19.38 compile, load and GPU invocation passed without compiler guard bypass.'
        dump(run / 'environment.json', env)
    print(json.dumps({k: record[k] for k in ('phase', 'status', 'module_loaded', 'module_called', 'output_equal', 'exception')}), flush=True)
    return 0 if record['status'] in ('pass', 'compiled_loaded_gpu_not_run') else 2


if __name__ == '__main__':
    sys.exit(main())
