"""Collect actual CUDA activity via CUPTI on the frozen b4/N65536 codec.

The PyTorch Windows wheel exposes CPU-only Kineto activities. This independent
collector preserves the old CPU-only trace and never alters frozen measurement
sources, contract, inputs, or raw benchmark blocks.
"""
from __future__ import annotations
import argparse
from collections import Counter
import ctypes
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path, value):
    with path.open('x', encoding='utf-8') as out:
        json.dump(value, out, indent=2, ensure_ascii=False, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--attempt', default='a01')
    args = parser.parse_args()
    run = (ROOT / args.run_dir).resolve()
    if not run.is_relative_to(ROOT / 'results/gpu_codec_v1') or not args.attempt.isalnum():
        raise ValueError('Invalid run or attempt')
    registration = read(run / 'execution_registration.json')
    def frozen():
        for name, expected in registration['source_hashes'].items():
            if sha(ROOT / name) != expected: raise ValueError('Frozen source changed: ' + name)
        for name, key in [('codec_contract.json', 'contract_sha256'), ('input_manifest.json', 'input_manifest_sha256')]:
            if sha(run / name) != registration[key]: raise ValueError('Frozen file changed: ' + name)
    frozen()
    if read(run / 'correctness_acceptance.json')['method_status'] != 'pass': raise ValueError('Correctness must pass')
    from probe_toolchain_v2 import configure
    config = configure()
    cuda = ROOT / '.toolchains/cupti-12.9'
    build = run / f'build/cupti_trace_{args.attempt}'
    build.mkdir(parents=True, exist_ok=True)
    raw = run / f'traces/cupti_activity_{args.attempt}.jsonl'
    trace = run / f'traces/cupti_roundtrip_b4_n65536_{args.attempt}.json'
    manifest_path = run / 'traces/gpu_activity_manifest.json'
    if raw.exists() or trace.exists() or manifest_path.exists(): raise FileExistsError('Preserve prior activity attempts')
    source = ROOT / 'scripts/cupti_trace_collector_v2.cpp'
    dll = build / 'cupti_trace_collector_v2.dll'
    command = [config['path_resolution']['cl'], '/nologo', '/O2', '/std:c++17', '/EHsc', '/MD', '/LD',
        '/I' + str(cuda / 'include'), str(source), '/Fo' + str(build / 'collector.obj'), '/link',
        '/LIBPATH:' + str(cuda / 'lib'), 'cupti.lib', '/OUT:' + str(dll)]
    compiled = subprocess.run(command, capture_output=True)
    build_log = run / f'logs/cupti_trace_build_{args.attempt}.log'
    build_log.write_bytes(compiled.stdout + compiled.stderr)
    if compiled.returncode: raise RuntimeError('CUPTI collector build failed: ' + str(build_log))
    directories = [os.add_dll_directory(str(cuda / 'lib')), os.add_dll_directory(str(Path(config['msvc_root']) / 'bin/Hostx64/x64'))]
    collector = ctypes.CDLL(str(dll))
    collector.trace_error.restype = ctypes.c_char_p
    collector.trace_timestamp.restype = ctypes.c_uint64
    collector.trace_start.argtypes = [ctypes.c_char_p]
    collector.trace_push.argtypes = [ctypes.c_uint64]
    def checked(value):
        if value: raise RuntimeError(collector.trace_error().decode('utf-8', errors='replace'))
    checked(collector.trace_start(str(raw).encode('utf-8')))
    import torch
    from gradient_comm_lab.cpu_reference import CpuReference, tensor_bytes
    from gradient_comm_lab.fixtures_v2 import load_fixture
    from gradient_comm_lab.torch_codec import TorchCodecPlan
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = False
    fallback = dict(torch_version=torch.__version__, kineto_available=torch.profiler.kineto_available(),
        supported_activities=[str(x) for x in torch.profiler.supported_activities()],
        cupti_dll=str(cuda / 'lib/cupti64_2025.2.1.dll'), cupti_dll_sha256=sha(cuda / 'lib/cupti64_2025.2.1.dll'),
        injection_environment={k:v for k,v in os.environ.items() if k=='CUDA_INJECTION64_PATH' or k.startswith(('CUPTI','KINETO','NSYS','NVPROF'))})
    input_manifest = read(run / 'input_manifest.json')
    row = next(r for r in input_manifest['inputs'] if r['source_kind'] == 'synthetic' and r['original_numel'] == 65536 and
        r['generator_config']['seed'] == 7301 and r['generator_config']['distribution'] == 'standard_normal')
    preparation_start = time.perf_counter()
    cpu = load_fixture(row)
    expected = CpuReference(4).roundtrip(cpu['input_logical'], cpu['main_thresholds'], cpu['scale_thresholds'])
    x, main, scale, lut = [cpu[k].cuda() for k in ('input_logical', 'main_thresholds', 'scale_thresholds')] + [cpu['lookup_tables'][4].cuda()]
    plan = TorchCodecPlan(65536, 4, device='cuda')
    out = plan.roundtrip_into(x, main, scale, lut, checked=True)
    torch.cuda.synchronize()
    actual = tensor_bytes(plan.packet.payload) + tensor_bytes(plan.packet.scale_codes) + tensor_bytes(plan.packet.top_scales)
    if actual != expected['wire'] or not torch.equal(out.cpu().view(torch.int16), expected['decoded'].view(torch.int16)):
        raise AssertionError('Trace fixture parity failed before capture window')
    fn = lambda: plan.roundtrip_into(x, main, scale, lut, checked=False)
    for _ in range(50): fn()
    torch.cuda.synchronize()
    preparation_seconds = time.perf_counter() - preparation_start
    markers = []
    window_start = collector.trace_timestamp()
    for index in range(3):
        checked(collector.trace_push(index + 1))
        start = collector.trace_timestamp()
        fn()
        end = collector.trace_timestamp()
        checked(collector.trace_pop())
        markers.append(dict(index=index, start=start, end=end))
    checked(collector.trace_push(100))
    sync_start = collector.trace_timestamp()
    torch.cuda.synchronize()
    sync_end = collector.trace_timestamp()
    checked(collector.trace_pop())
    window_end = collector.trace_timestamp()
    checked(collector.trace_stop())
    frozen()
    records = [json.loads(line) for line in raw.read_text(encoding='utf-8').splitlines()]
    final = [r for r in records if r['cat'] == 'collector_summary']
    if len(final) != 1 or final[0]['dropped_records_total'] or final[0]['error']:
        raise RuntimeError('Activity collection dropped records or reported an error')
    correlations = {r['correlation']: r['external_id'] for r in records if r['cat'] == 'external_correlation'}
    events = []
    for r in records:
        if 'start_ns' not in r or not (window_start <= r['start_ns'] <= r['end_ns'] <= window_end): continue
        external = correlations.get(r['correlation'])
        extra = {k: v for k, v in r.items() if k not in ('start_ns', 'end_ns', 'cat', 'name', 'pid', 'tid')}
        if external in (1, 2, 3): extra['operation_index'] = external - 1
        if external == 100: extra['boundary_synchronization'] = True
        extra['external_id'] = external
        events.append(dict(ph='X', cat=r['cat'], name=r['name'], ts=(r['start_ns']-window_start)/1000,
            dur=(r['end_ns']-r['start_ns'])/1000, pid=r.get('pid', 100000 + r.get('device', 0)),
            tid=r.get('tid', r.get('stream', 0)), args=extra))
    for marker in markers:
        events.append(dict(ph='X', cat='user_annotation', name='trusted_preallocated_roundtrip',
            ts=(marker['start']-window_start)/1000, dur=(marker['end']-marker['start'])/1000,
            pid=os.getpid(), tid=threading.get_native_id(), args={'operation_index': marker['index']}))
    events.append(dict(ph='X', cat='user_annotation', name='explicit_final_synchronization',
        ts=(sync_start-window_start)/1000, dur=(sync_end-sync_start)/1000, pid=os.getpid(),
        tid=threading.get_native_id(), args={'boundary_synchronization': True}))
    inventory = Counter(e['cat'] for e in events)
    kernels = [e for e in events if e['cat'] == 'kernel']
    if not kernels: raise RuntimeError('CUPTI produced no kernels in the capture window; missing data, not zero work')
    if any('operation_index' not in e['args'] for e in kernels): raise RuntimeError('Unassigned GPU kernel')
    if len({e['args']['operation_index'] for e in kernels}) != 3: raise RuntimeError('Incomplete operation attribution')
    write(trace, dict(traceEvents=sorted(events, key=lambda e: e['ts']), displayTimeUnit='ms',
        source='NVIDIA CUPTI 12.9 Activity API', capture_clock='CUPTI timestamps; ns converted to relative microseconds'))
    diagnostic = run / f'logs/cupti_trace_{args.attempt}.json'
    write(diagnostic, dict(status='pass', completed_at=datetime.now().astimezone().isoformat(),
        command=[sys.executable] + sys.argv, collector_build_command=command, collector_sha256=sha(source), dll_sha256=sha(dll),
        profiler_fallback_diagnosis=fallback, raw_activity_records=len(records), trace_inventory=dict(inventory),
        kernels_per_operation=dict(Counter(e['args']['operation_index'] for e in kernels)), dropped_records=0,
        window_start_ns=window_start, window_end_ns=window_end, preparation_seconds=preparation_seconds,
        CPU_reference_and_wire_parity_passed_before_capture=True, packet_sha256=hashlib.sha256(actual).hexdigest(),
        preparation_warmup_count=50, source_hashes=registration['source_hashes'],
        collector_sources={str(source.relative_to(ROOT)):sha(source),str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))}))
    artifact_paths = [trace, raw, diagnostic, build_log, source, Path(__file__).resolve()]
    artifacts = [dict(path=str(p.relative_to(run)).replace('\\','/'), sha256=sha(p), bytes=p.stat().st_size)
        for p in artifact_paths[:4]]
    write(manifest_path, dict(schema_version='gpu_activity_trace_v2.0', status='pass',
        tool='CUPTI 12.9 Activity API custom minimal collector (application remains torch cu118)', case_id=row['input_id']+'__b4__roundtrip__preallocated__trusted_inputs',
        method_id='torch_cuda_repaired', operation_count=3, trace=trace.name, sha256=sha(trace), bytes=trace.stat().st_size,
        artifacts=artifacts, contract_sha256=registration['contract_sha256'], benchmark_config_sha256=sha(run/'benchmark_config.json'),
        input_manifest_sha256=registration['input_manifest_sha256'], source_hashes=registration['source_hashes'],
        collector_sources={str(source.relative_to(ROOT)):sha(source),str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))},
        operation_attribution='CUPTI external correlation CUSTOM0 IDs 1,2,3 -> operation_index 0,1,2; ID100 marks final boundary synchronization.',
        raw_activity_note='Raw JSONL includes preparation/warmup. Chrome trace filters to explicit CUPTI timestamp window; actual GPU kernel attribution uses correlation, not CPU interval containment.',
        boundary='fixture/H2D/LUT/plan/CPU parity/50 warmups outside Chrome trace window; final explicit cuda synchronization retained.',
        independent_from_formal_timing=True, dropped_records=0, original_cpu_only_trace_preserved=True,
        official_reference='https://docs.nvidia.com/cupti/12.9.1/index.html'))
    print(json.dumps({'status':'pass','trace':str(trace),'inventory':dict(inventory),'kernel_count':len(kernels)}), flush=True)


if __name__ == '__main__': main()
