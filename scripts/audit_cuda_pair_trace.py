"""Verify CUPTI raw records and bind the collector sources to the GPU trace manifest."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p, v):
    with p.open('x', encoding='utf-8') as out: json.dump(v, out, indent=2, ensure_ascii=False, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    run = (ROOT / args.run_dir).resolve()
    if not run.is_relative_to(ROOT / 'results/gpu_codec_v1') or not args.attempt.isalnum(): raise ValueError('Invalid run/attempt')
    if (run / 'artifact_integrity.json').exists(): raise ValueError('Sealed run cannot be changed')
    manifest_path = run / f'traces/gpu_activity_manifest_{args.attempt}.json'
    manifest = read(manifest_path)
    audit_path = run / f'logs/cupti_trace_audit_{args.attempt}.json'
    initial_path = run / f'traces/gpu_activity_manifest_initial_{args.attempt}.json'
    if audit_path.exists() or initial_path.exists(): raise FileExistsError('Do not overwrite prior audit')
    for artifact in manifest['artifacts']:
        path = run / artifact['path']
        assert path.stat().st_size == artifact['bytes'] and sha(path) == artifact['sha256'], artifact['path']
    for name, expected in manifest['source_hashes'].items(): assert sha(ROOT / name) == expected, name
    for name, expected in manifest['collector_sources'].items(): assert sha(ROOT / name) == expected, name
    assert sha(run / 'codec_contract.json') == manifest['contract_sha256']
    assert sha(run / 'benchmark_config.json') == manifest['benchmark_config_sha256']
    raw_path = run / f'traces/cupti_activity_{args.attempt}.jsonl'
    diagnostic_path = run / f'logs/cupti_trace_{args.attempt}.json'
    raw = [json.loads(line) for line in raw_path.read_text(encoding='utf-8').splitlines()]
    diagnostic = read(diagnostic_path)
    enabled = sorted(r['kind'] for r in raw if r['cat'] == 'enabled_activity')
    assert enabled == [1, 2, 4, 5, 10, 38, 39], enabled
    errors = [r for r in raw if r['cat'] == 'collector_error']
    summaries = [r for r in raw if r['cat'] == 'collector_summary']
    assert not errors and len(summaries) == 1
    assert summaries[0]['dropped_records_total'] == 0 and not summaries[0]['error'] and not summaries[0]['first_error']
    assert sum(r['dropped_records'] for r in raw if r['cat'] == 'buffer_metadata') == 0
    correlations = defaultdict(set)
    for r in raw:
        if r['cat'] == 'external_correlation' and r['external_kind'] == 3:
            correlations[r['correlation']].add(r['external_id'])
    conflicts = {str(k): list(v) for k, v in correlations.items() if len(v) != 1}
    assert not conflicts, conflicts
    mapping = {k: next(iter(v)) for k, v in correlations.items()}
    measured = [r for r in raw if 'start_ns' in r and mapping.get(r['correlation']) in (1, 2, 3, 100)]
    invalid = [r for r in measured if not (0 < diagnostic['window_start_ns'] <= r['start_ns'] <= r['end_ns'] <= diagnostic['window_end_ns'])]
    assert not invalid, invalid[:3]
    assert all(r.get('return_value', 0) == 0 for r in measured), 'CUDA API failure in measured region'
    trace = read(run / 'traces' / manifest['trace'])
    inventory = Counter(r['cat'] for r in measured)
    chrome = [e for e in trace['traceEvents'] if e['cat'] != 'user_annotation']
    assert len(chrome) == len(measured), 'Chrome/raw activity mismatch'
    assert Counter(e['cat'] for e in chrome) == inventory
    raw_identity = Counter((r['cat'], r['correlation'], r['start_ns'], r['end_ns']) for r in measured)
    chrome_identity = Counter((e['cat'], e['args']['correlation'], round(e['ts'] * 1000) + diagnostic['window_start_ns'],
        round((e['ts'] + e['dur']) * 1000) + diagnostic['window_start_ns']) for e in chrome)
    assert raw_identity == chrome_identity, 'Chrome timestamps/identity do not match raw records'
    # Verify every activity field, not just its timing/correlation identity.
    expected_fields=[]
    for record in measured:
        external=mapping.get(record['correlation'])
        detail={k:v for k,v in record.items() if k not in ('start_ns','end_ns','cat','name','pid','tid')}
        if external in (1,2,3):detail['operation_index']=external-1
        if external==100:detail['boundary_synchronization']=True
        detail['external_id']=external
        expected_fields.append({'cat':record['cat'],'name':record['name'],
            'pid':record.get('pid',100000+record.get('device',0)),
            'tid':record.get('tid',record.get('stream',0)),'args':detail})
    expected_payload=Counter(json.dumps(r,sort_keys=True) for r in expected_fields)
    observed_payload=Counter(json.dumps({k:e[k] for k in ('cat','name','pid','tid','args')},sort_keys=True) for e in chrome)
    assert expected_payload==observed_payload,'Chrome activity fields differ from raw activity'
    kernels = [r for r in measured if r['cat'] == 'kernel']
    kernel_counts = Counter(mapping[r['correlation']] - 1 for r in kernels)
    assert set(kernel_counts) == {0, 1, 2} and min(kernel_counts.values()) > 0
    copy_kinds = {1: 'HtoD', 2: 'DtoH', 3: 'HtoA', 4: 'AtoH', 5: 'AtoA', 6: 'AtoD', 7: 'DtoA', 8: 'DtoD', 9: 'HtoH', 10: 'PtoP'}
    copies = [r for r in measured if r['cat'] == 'gpu_memcpy']
    direction_counts = Counter(copy_kinds.get(r['copy_kind'], 'unknown') for r in copies)
    assert 'unknown' not in direction_counts
    snapshot_dir = run / f'traces/collector_sources_{args.attempt}'
    snapshot_dir.mkdir()
    snapshots = []
    for name, expected in manifest['collector_sources'].items():
        target = snapshot_dir / Path(name).name
        shutil.copyfile(ROOT / name, target)
        assert sha(target) == expected
        snapshots.append(target)
    audit_source = snapshot_dir / Path(__file__).name
    shutil.copyfile(Path(__file__), audit_source); snapshots.append(audit_source)
    write(audit_path, dict(status='pass', completed_at=datetime.now().astimezone().isoformat(),
        command=[sys.executable] + sys.argv, raw_activity_path=str(raw_path.relative_to(run)).replace('\\','/'),
        enabled_activity_kinds=enabled, collection_errors=[], dropped_records=0,
        conflicting_external_correlations=0, external_kind=3, measured_timed_records=len(measured),
        measured_invalid_or_outside_window_records=0, chrome_records_exactly_match_measured_raw=True,
        activity_field_comparison='All fields including name, device, stream, grid, block, copy bytes/direction, API return value and operation attribution match; timestamps independently checked.',
        window_start_ns=diagnostic['window_start_ns'], window_end_ns=diagnostic['window_end_ns'],
        inventory=dict(inventory), kernel_count=len(kernels), kernels_per_operation=dict(kernel_counts),
        device_memcpy_direction_counts=dict(direction_counts), device_memcpy_bytes=sum(r['bytes'] for r in copies),
        device_HtoD_count=direction_counts['HtoD'], device_DtoH_count=direction_counts['DtoH'],
        scope='Three trusted preallocated roundtrips plus explicit final synchronization only; profiler durations are not formal benchmark samples.',
        zero_count_interpretation='Zero captured HtoD/DtoH/memset within this measured window is supported by enabled activity kinds, complete correlation/timestamp audit and zero dropped records; no claim about allocating API or setup.'))
    shutil.copyfile(manifest_path, initial_path)
    manifest.update(raw_activity_path=str(raw_path.relative_to(run)).replace('\\','/'), enabled_activity_kinds=enabled,
        collection_errors=[], dropped_records=0, audit_path=str(audit_path.relative_to(run)).replace('\\','/'),
        measured_invalid_or_outside_window_records=0, source_snapshots=[str(p.relative_to(run)).replace('\\','/') for p in snapshots],
        audit_source_sha256=sha(Path(__file__)), initial_manifest_preserved=str(initial_path.relative_to(run)).replace('\\','/'))
    for path in [audit_path, initial_path] + snapshots:
        manifest['artifacts'].append(dict(path=str(path.relative_to(run)).replace('\\','/'), sha256=sha(path), bytes=path.stat().st_size))
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(json.dumps({'status':'pass', 'kernel_count':len(kernels),'per_operation':dict(kernel_counts),
        'device_memcpy_directions':dict(direction_counts),'raw_and_chrome_records':len(measured),'audit':str(audit_path)}))


if __name__ == '__main__': main()
