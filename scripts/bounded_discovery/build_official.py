"""Build the pinned upstream CUDA extension; retain every source and attempt."""
from __future__ import annotations
import argparse, difflib, hashlib, importlib.util, json, os, platform, shutil, subprocess, sys, time, traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'results/bounded_discovery_v1/bounded_discovery_v1_20261104_001'
UPSTREAM = ROOT / 'upstream/bounded_discovery_v1/dynamiq/cuda_kernels/src/srrcomp/eden_utils'
sys.path.insert(0, str(ROOT / 'scripts'))
from probe_toolchain_v2 import configure, command

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p, x): p.write_text(json.dumps(x, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', required=True, choices=['a01', 'a02', 'a03'])
    parser.add_argument('--remove-unused-posix-include', action='store_true')
    parser.add_argument('--isolated-curand-headers', action='store_true')
    args = parser.parse_args()
    attempt = RUN / 'build' / ('official_' + args.attempt)
    attempt.mkdir(parents=True, exist_ok=False)
    record_path = RUN / 'logs' / ('official_build_' + args.attempt + '.json')
    record = dict(attempt=args.attempt, started_at=datetime.now().astimezone().isoformat(),
                  command=[sys.executable] + sys.argv, status='running', module_name='eden_utils',
                  official_commit='96da2ee480bbfd8e2049de3fbd8595cc9d7a30b8',
                  source_directory=str(attempt / 'sources'), module_loaded=False,
                  repair='Python -X utf8 and host UTF-8 encoding; remove two unused POSIX includes' if args.remove_unused_posix_include else None)
    start = time.perf_counter()
    try:
        config = configure()
        import torch
        from torch.utils.cpp_extension import load
        env = dict(captured_at=datetime.now().astimezone().isoformat(), python=sys.version,
                   executable=sys.executable, platform=platform.platform(), torch=torch.__version__,
                   torch_cuda=torch.version.cuda, configuration=config,
                   compiler_versions={key: command([config['path_resolution'][key]] + extra) for key, extra in [('nvcc',['--version']),('cl',[]),('ninja',['--version'])]},
                   nvidia_smi=command(['nvidia-smi','--query-gpu=name,driver_version,compute_cap,memory.total','--format=csv,noheader']))
        dump(RUN / 'logs' / ('official_build_environment_' + args.attempt + '.json'), env)
        destination = attempt / 'sources'
        shutil.copytree(UPSTREAM, destination)
        shutil.copy2(Path(__file__), attempt / 'build_official.py')
        patches = []
        if args.remove_unused_posix_include:
            for rel in ('csrc/cuda_hadamard.cu','csrc/cuda_hierarchical_mee.cu'):
                source = destination / rel
                before = source.read_text(encoding='utf-8')
                assert before.count('#include <unistd.h>') == 1
                after = before.replace('#include <unistd.h>', '// Windows build glue: unused POSIX header omitted.')
                source.write_text(after, encoding='utf-8', newline='\n')
                patches.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile='upstream/' + rel, tofile='compiled/' + rel))
        (attempt / 'source_glue.diff').write_text(''.join(patches), encoding='utf-8')
        sources = [destination / rel for rel in ('eden_utils.cpp','csrc/cuda_hadamard.cu','csrc/cuda_packing.cu','csrc/cuda_hierarchical_mee.cu')]
        flags = ['-std=c++17','-O3','--use_fast_math','-lineinfo','--use-local-env','-ccbin',str(Path(config['msvc_root']) / 'bin/Hostx64/x64')]
        if args.remove_unused_posix_include:
            flags.extend(['-Xcompiler','/utf-8'])
        includes = [str(ROOT / '.toolchains/bounded_discovery_v1/curand-10.3.0.86/include')] if args.isolated_curand_headers else []
        if includes:
            assert (Path(includes[0]) / 'curand_kernel.h').is_file()
            record['repair'] += '; isolated official CUDA 11.8 cuRAND headers added to include search path'
        record.update(extra_cflags=['/O2','/std:c++17'] + (['/utf-8'] if args.remove_unused_posix_include else []), extra_cuda_cflags=flags, extra_include_paths=includes,
                      official_math_flags_preserved=True, cpp_optimization_port='/O2 replaces GCC -O3 on MSVC',
                      source_hashes={str(p.relative_to(destination)):sha(p) for p in destination.rglob('*') if p.is_file()},
                      source_glue_diff_sha256=sha(attempt / 'source_glue.diff'), build_directory=str(attempt))
        dump(record_path, record)
        module = load(name='eden_utils', sources=[str(p) for p in sources], build_directory=str(attempt),
                      extra_cflags=record['extra_cflags'], extra_cuda_cflags=flags, extra_include_paths=includes, with_cuda=True, verbose=True)
        record.update(status='compiled_and_loaded', module_loaded=True, module_path=module.__file__, module_sha256=sha(Path(module.__file__)))
    except Exception as error:
        record.update(status='build_failed', exception_type=type(error).__name__, exception=str(error), traceback=traceback.format_exc())
        raise
    finally:
        record.update(finished_at=datetime.now().astimezone().isoformat(), elapsed_seconds=time.perf_counter()-start)
        dump(record_path, record)
        print(json.dumps({k:record.get(k) for k in ['attempt','status','elapsed_seconds','module_path','module_sha256']},ensure_ascii=False), flush=True)

if __name__ == '__main__': main()
