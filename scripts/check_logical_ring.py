"""G2a frozen-case runner: CPU packet oracle, GPU audit comparison and ownership probes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gradient_comm_lab.cpu_reference import CpuReference, lookup_table, tensor_identity, threshold_fixture
from gradient_comm_lab.ring_reference import run_ring as cpu_ring

METHOD_CPU = 'cpu_reference'
METHOD_GPU = 'cuda_logical_ring'
VARIANTS = ('baseline', 'repeat', 'reverse')
CHECKS = ('prepared_input_parity', 'packet_parity', 'decode_parity', 'partial_sum_parity',
          'output_parity', 'rank_consistency', 'ledger', 'threshold_consumption', 'immutability')
PACKET_FIELDS = ('payload', 'scale_codes', 'top_scales')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raw(value):
    return value.detach().cpu().contiguous().view(torch.uint8).reshape(-1)


def identity(value):
    return tensor_identity(value.detach().cpu().contiguous())


def mismatch(left, right):
    if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
        return max(left.numel(), right.numel(), 1)
    a, b = raw(left), raw(right)
    return int((a != b).sum())


def packet_identity(packet):
    parts = [raw(packet[k]) for k in PACKET_FIELDS]
    wire = torch.cat(parts)
    result = {k: identity(packet[k]) for k in PACKET_FIELDS}
    result.update(wire_sha256=hashlib.sha256(wire.numpy().tobytes()).hexdigest(),
                  wire_bytes=wire.numel(), logical_numel=packet['logical_numel'],
                  padded_numel=packet['padded_numel'], bits=packet['bits'])
    return result


def jsonable(value):
    if isinstance(value, torch.Tensor):
        return identity(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def signature(result):
    return dict(outputs=[identity(v) for v in result['outputs']],
                events={r['event_id']: packet_identity(r['packet']) for r in result['events']},
                intermediates={r['intermediate_id']: identity(r['value']) for r in result['intermediates']})


class RecordingProvider:
    def __init__(self, provider):
        self.provider = provider
        self.calls = []
        self.tensors = []

    def __call__(self, event):
        value = self.provider(event)
        before = {k: identity(value[k + '_thresholds']) for k in ('main', 'scale')}
        self.calls.append(dict(event, event_key=value['event_key'], seed=value['seed'], **before))
        self.tensors.append((value['main_thresholds'], value['scale_thresholds'], before))
        return value

    def unchanged(self):
        return all(identity(a) == before['main'] and identity(b) == before['scale']
                   for a, b, before in self.tensors)

    def evidence(self):
        return [dict(event_key=call['event_key'], before=before,
                     after=dict(main=identity(a), scale=identity(b)))
                for call, (a, b, before) in zip(self.calls, self.tensors)]


class Writer:
    def __init__(self, directory, run, contract):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.run, self.contract, self.handles, self.rows = run, contract, {}, []

    def write(self, name, record):
        if name not in self.handles:
            path = self.directory / name
            if path.exists():
                raise ValueError(f'Refusing to overwrite attempt artifact: {path}')
            self.handles[name] = path.open('w', encoding='utf-8', newline='\n')
        row = dict(run_id=self.run.name, contract_sha256=self.contract, **record)
        self.handles[name].write(json.dumps(jsonable(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n')
        self.handles[name].flush()
        if name == 'correctness.jsonl':
            self.rows.append(row)

    def close(self):
        for value in self.handles.values():
            value.close()


def base(case, method, variant):
    return dict(case_id=case['case_id'], method_id=method, variant=variant,
                nominal_bits=case['bits'], world_size=case['world_size'],
                logical_numel=case['logical_numel'], entry_mode=case['entry_mode'])


def record_identity(method, variant, logical_id):
    return f'{method}/{variant}/{logical_id}'


def emit_result(writer, case, method, variant, result, expected, provider,
                inputs_before, inputs_after, lut_before, lut_after):
    common = base(case, method, variant)
    expected_events = {row['event_id']: row for row in expected['events']}
    expected_intermediates = {row['intermediate_id']: row for row in expected['intermediates']}
    for row in result['events']:
        exp = expected_events.get(row['event_id'])
        packet = packet_identity(row['packet'])
        expected_packet = packet_identity(exp['packet']) if exp else None
        diff = sum(mismatch(row['packet'][k], exp['packet'][k]) for k in PACKET_FIELDS) if exp else -1
        input_diff = mismatch(row['input'], exp['input']) if exp else -1
        out = {k: v for k, v in row.items() if k not in ('input', 'packet', 'threshold_info')}
        writer.write('quantization_events.jsonl', dict(common, **out,
            record_id=record_identity(method, variant, row['event_id']),
            encoding_input=identity(row['input']), expected_encoding_input=identity(exp['input']) if exp else None,
            encoding_input_bit_mismatch_count=input_diff, packet=packet, expected_packet=expected_packet,
            packet_byte_mismatch_count=diff, threshold_info=jsonable(row['threshold_info']),
            status='pass' if diff == 0 and input_diff == 0 else 'fail'))
    for row in result['messages']:
        out = {k: v for k, v in row.items() if k != 'packet'}
        writer.write('message_ledger.jsonl', dict(common, **out,
            record_id=record_identity(method, variant, row['message_id']),
            packet=packet_identity(row['packet']), status='pass'))
    for row in result['intermediates']:
        exp = expected_intermediates.get(row['intermediate_id'])
        diff = mismatch(row['value'], exp['value']) if exp else -1
        out = {k: v for k, v in row.items() if k != 'value'}
        writer.write('intermediate_checks.jsonl', dict(common, **out,
            record_id=record_identity(method, variant, row['intermediate_id']),
            expected=identity(exp['value']) if exp else None, actual=identity(row['value']),
            bit_mismatch_count=diff, reference_record=method == METHOD_CPU,
            status='pass' if diff == 0 else 'fail'))
    writer.write('input_output_checks.jsonl', dict(common,
        record_id=record_identity(method, variant, case['case_id'] + '/input_output'),
        inputs_before=inputs_before, inputs_after=inputs_after,
        prepared_inputs=[identity(v) for v in result['prepared_inputs']],
        outputs=[identity(v) for v in result['outputs']],
        expected_outputs=[identity(v) for v in expected['outputs']],
        lut_before=lut_before, lut_after=lut_after,
        threshold_consumption=provider.calls, threshold_immutability=provider.evidence(),
        metadata=jsonable(result['metadata']), status='pass' if inputs_before == inputs_after and provider.unchanged() and lut_before == lut_after else 'fail'))


def check_result(case, result, expected, provider, before, after, lut_before, lut_after):
    expected_events = {row['event_id']: row for row in expected['events']}
    actual_events = {row['event_id']: row for row in result['events']}
    expected_i = {row['intermediate_id']: row for row in expected['intermediates']}
    actual_i = {row['intermediate_id']: row for row in result['intermediates']}
    event_ids_ok = len(actual_events) == len(result['events']) and actual_events.keys() == expected_events.keys()
    intermediate_ids_ok = len(actual_i) == len(result['intermediates']) and actual_i.keys() == expected_i.keys()
    def diffs(kind):
        return sum(mismatch(actual_i[k]['value'], row['value']) for k, row in expected_i.items()
                   if row['kind'] == kind and k in actual_i)
    packet_diff = sum(sum(mismatch(actual_events[k]['packet'][field], row['packet'][field]) for field in PACKET_FIELDS)
                      + mismatch(actual_events[k]['input'], row['input']) for k, row in expected_events.items() if k in actual_events)
    n, d, bits = case['world_size'], case['logical_numel'], case['bits']
    active = n > 1 and d > 0
    length = ((d+n*1024-1)//(n*1024))*1024 if active else 0
    wanted_encodes = n*n if active else 0
    wanted_messages = 2*n*(n-1) if active else 0
    packet_bytes = length*(bits+1)//8 + length//16 + 2*(length//256)
    messages = result['messages']
    expected_m = {row['message_id']: row for row in expected['messages']}
    messages_ok = len(messages) == wanted_messages and len({r['message_id'] for r in messages}) == wanted_messages
    for row in messages:
        exp = expected_m.get(row['message_id'])
        fields = ('source_packet_id', 'phase', 'hop', 'sender', 'receiver', 'chunk', 'logical_numel', 'padded_numel', 'chunk_logical_numel', 'chunk_padded_numel', 'payload_bytes', 'scale_codes_bytes', 'top_scales_bytes', 'wire_bytes')
        messages_ok &= exp is not None and all(row[k] == exp[k] for k in fields)
        actual_bytes = sum(row['packet'][k].numel()*row['packet'][k].element_size() for k in PACKET_FIELDS)
        messages_ok &= actual_bytes == row['wire_bytes'] == packet_bytes
        messages_ok &= exp is not None and packet_identity(row['packet']) == packet_identity(exp['packet'])
    outputs_ok = len(result['outputs']) == n and all(mismatch(a, b) == 0 for a, b in zip(result['outputs'], expected['outputs']))
    prepared_ok = len(result['prepared_inputs']) == n and all(mismatch(a, b) == 0 for a, b in zip(result['prepared_inputs'], expected['prepared_inputs']))
    callkeys = [r['event_key'] for r in provider.calls]
    threshold_ok = len(callkeys) == wanted_encodes and len(set(callkeys)) == wanted_encodes and set(callkeys) == set(expected_events)
    return dict(prepared_input_parity=(prepared_ok, dict(count=n)),
        packet_parity=(event_ids_ok and packet_diff == 0, dict(expected_events=wanted_encodes, actual_events=len(result['events']), bit_or_byte_mismatch_count=packet_diff)),
        decode_parity=(intermediate_ids_ok and diffs('decode') == 0, dict(bit_mismatch_count=diffs('decode'))),
        partial_sum_parity=(intermediate_ids_ok and diffs('partial_sum') == 0, dict(bit_mismatch_count=diffs('partial_sum'))),
        output_parity=(outputs_ok, dict(outputs=[identity(v) for v in result['outputs']], expected_outputs=[identity(v) for v in expected['outputs']])),
        rank_consistency=(len(result['outputs']) == n and all(mismatch(result['outputs'][0], v) == 0 for v in result['outputs']), dict(rank_count=n)),
        ledger=(messages_ok, dict(expected_messages=wanted_messages, actual_messages=len(messages), expected_wire_bytes=wanted_messages*packet_bytes, actual_wire_bytes=sum(r['wire_bytes'] for r in messages))),
        threshold_consumption=(threshold_ok, dict(expected_count=wanted_encodes, actual_count=len(callkeys), event_keys=callkeys)),
        immutability=(before == after and provider.unchanged() and lut_before == lut_after, dict(inputs_before=before, inputs_after=after, lut_before=lut_before, lut_after=lut_after, thresholds_unchanged=provider.unchanged())))


def case_manifest(cases):
    result = []
    for case in cases:
        for variant in VARIANTS:
            for check in CHECKS:
                result.append(dict(base(case, METHOD_GPU, variant),
                    record_id=f"{METHOD_GPU}/{variant}/{case['case_id']}/{check}",
                    check_type=check, expected_result='pass'))
    for bits in (4, 8):
        for check in ('two_packets_in_flight', 'decode_output_reuse', 'cross_collective_isolation'):
            result.append(dict(record_id=f'buffer/b{bits}/{check}', case_id=f'buffer_b{bits}',
                               method_id=METHOD_GPU, variant='buffer', nominal_bits=bits,
                               check_type=check, expected_result='pass'))
    for spec in invalid_specs():
        result.append(dict(record_id='invalid/' + spec['name'], case_id='invalid/' + spec['name'],
                           method_id=METHOD_GPU, variant='invalid', nominal_bits=4,
                           check_type='expected_rejection', expected_result='pass',
                           expected_exception_type='ValueError', expected_message=spec['message']))
    return dict(schema_version='logical_ring_correctness_cases_v1',
                expected_record_count=len(result), cases=result,
                execution_roles=[dict(method_id=METHOD_CPU, variant='baseline')] +
                                [dict(method_id=METHOD_GPU, variant=v) for v in VARIANTS])


def invalid_specs():
    groups = {
        'RING_BITS: expected nominal b4/b8': ('bits2', 'bits_bool'),
        'RING_ENTRY_MODE: expected raw_fp32 or predivided_bf16': ('entry_mode',),
        'RING_WORKERS: expected a list or tuple with 1, 2 or 4 workers': ('workers_empty', 'workers_three'),
        'RING_CASE_ID: expected a nonempty string': ('case_id',),
        'RING_INPUT: expected a contiguous one-dimensional tensor': ('input_not_tensor', 'input_2d', 'input_noncontiguous'),
        'RING_INPUT_DTYPE: input dtype does not match entry mode': ('input_dtype',),
        'RING_INPUT_DEVICE: expected CUDA tensors on one device': ('input_cpu', 'input_mixed_device'),
        'RING_INPUT_LENGTH: worker lengths must match': ('input_length',),
        'RING_NONFINITE_INPUT: inputs must be finite': ('input_nan', 'input_inf'),
        'RING_RANK_ORDER: expected a permutation of worker ranks': ('rank_duplicate', 'rank_out_of_range', 'rank_bool'),
        'RING_THRESHOLDS: expected a callable provider or get method': ('provider_missing',),
        'RING_THRESHOLDS: provider must return thresholds for this event': ('threshold_result',),
        'RING_THRESHOLD_IDENTITY: event key differs from requested event': ('threshold_identity',),
        'RING_THRESHOLD_METADATA: expected contiguous BF16 tensors of canonical shape': ('threshold_shape', 'threshold_dtype', 'threshold_noncontiguous', 'late_threshold_shape'),
        'RING_THRESHOLD_DEVICE: expected CPU or the input CUDA device': ('threshold_device',),
        'RING_THRESHOLD_VALUES: expected finite values in [0,65536]': ('threshold_negative', 'threshold_too_large', 'threshold_nan'),
        'RING_LUT_METADATA: expected contiguous one-dimensional BF16 LUT': ('lut_shape', 'lut_dtype'),
        'RING_LUT_DEVICE: expected CPU or the input CUDA device': ('lut_device',),
        'RING_LUT_VALUES: LUT must be finite and strictly ascending': ('lut_nonmonotonic', 'lut_nan'),
        'RING_LUT_IDENTITY: LUT differs from frozen CPU source': ('lut_identity',),
        'RING_NONFINITE_PREPARED_INPUT: BF16 entry conversion overflowed': ('prepared_overflow',),
        'RING_BUILD: expected the accepted G1 a02 module identity': ('build_missing',),
    }
    return [dict(name=name, message=message) for message, names in groups.items() for name in names]


def run_case(writer, run, case, build_manifest, gpu_module, fixture_module):
    loaded = fixture_module.load_case(case, run)
    inputs = loaded['inputs'] if isinstance(loaded, dict) else loaded
    cpu_before = [identity(x) for x in inputs]
    cpu_provider = RecordingProvider(fixture_module.FrozenThresholdProvider(run, device='cpu'))
    expected = cpu_ring(inputs, case['bits'], case['entry_mode'], cpu_provider,
                        case_id=case['case_id'], audit=True)
    cpu_after = [identity(x) for x in inputs]
    cpu_lut_before = expected['metadata'].get('lut_before')
    cpu_lut_after = expected['metadata'].get('lut_after')
    emit_result(writer, case, METHOD_CPU, 'baseline', expected, expected, cpu_provider,
                cpu_before, cpu_after, cpu_lut_before, cpu_lut_after)
    source = torch.stack(inputs).double()
    ideal = source.mean(0) if case['entry_mode'] == 'raw_fp32' else source.sum(0)
    for variant in VARIANTS:
        values = [x.cuda() for x in inputs]
        provider = RecordingProvider(fixture_module.FrozenThresholdProvider(run, device='cuda'))
        lut = lookup_table(case['bits']).clone().cuda()
        before, lut_before = [identity(x) for x in values], identity(lut)
        order = list(reversed(range(case['world_size']))) if variant == 'reverse' else None
        try:
            actual = gpu_module.run_ring(values, case['bits'], case['entry_mode'], provider,
                case_id=case['case_id'], rank_order=order, audit=True,
                build_manifest=build_manifest, lut=lut)
            after, lut_after = [identity(x) for x in values], identity(lut)
            emit_result(writer, case, METHOD_GPU, variant, actual, expected, provider,
                        before, after, lut_before, lut_after)
            checks = check_result(case, actual, expected, provider, before, after, lut_before, lut_after)
            error = actual['outputs'][0].detach().cpu().double() - ideal
            diagnostic = dict(reference='original_fp32_fp64_mean' if case['entry_mode'] == 'raw_fp32' else 'predivided_bf16_fp64_sum',
                squared_error=float(error.square().sum()), mse=float(error.square().mean()) if error.numel() else None,
                max_abs_error=float(error.abs().max()) if error.numel() else None)
            failed = [name for name, (passed, _) in checks.items() if not passed]
            counterexample = None
            if failed:
                counterexample = f'counterexamples/{case["case_id"]}_{variant}.pt'
                target = writer.directory / counterexample
                target.parent.mkdir(parents=True, exist_ok=True)
                torch.save(dict(case=case, inputs=inputs, expected=expected, actual=actual, failed=failed), target)
            for check, (passed, details) in checks.items():
                writer.write('correctness.jsonl', dict(base(case, METHOD_GPU, variant),
                    record_id=f"{METHOD_GPU}/{variant}/{case['case_id']}/{check}", check_type=check,
                    status='pass' if passed else 'fail', details=details, diagnostic=diagnostic,
                    counterexample_path=counterexample))
        except Exception as exc:
            target = writer.directory / 'counterexamples' / f'{case["case_id"]}_{variant}_exception.json'
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(dict(exception_type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()), indent=2), encoding='utf-8')
            for check in CHECKS:
                writer.write('correctness.jsonl', dict(base(case, METHOD_GPU, variant),
                    record_id=f"{METHOD_GPU}/{variant}/{case['case_id']}/{check}", check_type=check,
                    status='fail', exception_type=type(exc).__name__, message=str(exc),
                    counterexample_path=str(target.relative_to(writer.directory))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--manifest-only', action='store_true')
    parser.add_argument('--formal', action='store_true')
    parser.add_argument('--case-id', action='append')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    inputs = json.loads((run / 'input_manifest.json').read_text(encoding='utf-8'))
    cases = inputs['cases']
    manifest = case_manifest(cases)
    if args.manifest_only:
        (run / 'correctness_case_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(dict(expected_record_count=manifest['expected_record_count'])))
        return
    if args.formal and args.case_id:
        raise ValueError('Formal execution must run the complete frozen matrix')
    if args.case_id:
        cases = [case for case in cases if case['case_id'] in args.case_id]
        if len(cases) != len(set(args.case_id)):
            raise ValueError('Unknown or repeated selected case')
    contract_path = run / 'ring_contract.json'
    contract = sha(contract_path) if contract_path.exists() else 'development_unregistered'
    build_manifest = run / 'logs' / 'cuda_build_manifest.json'
    if args.formal:
        registered = json.loads((run / 'execution_registration.json').read_text(encoding='utf-8'))
        if registered['contract_sha256'] != contract:
            raise ValueError('Registered ring contract mismatch')
        for name in ('input_manifest', 'threshold_manifest', 'correctness_case_manifest'):
            if sha(run / (name + '.json')) != registered[name + '_sha256']:
                raise ValueError(f'Registered {name} mismatch')
        if sha(build_manifest) != registered['build_manifest_sha256']:
            raise ValueError('Registered build manifest mismatch')
        for path, digest in registered['sources'].items():
            if sha(ROOT / path) != digest:
                raise ValueError(f'Registered source mismatch: {path}')
        if json.loads((run / 'correctness_case_manifest.json').read_text(encoding='utf-8')) != manifest:
            raise ValueError('Runtime expected cases differ from frozen manifest')
    from gradient_comm_lab import logical_ring as gpu_module
    from gradient_comm_lab import ring_fixtures as fixture_module
    from gradient_comm_lab import cuda_codec
    directory = run if args.formal else run / 'attempts' / args.attempt
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = dict(attempt=args.attempt, formal=args.formal,
                    started_utc=datetime.now(timezone.utc).isoformat(), argv=sys.argv,
                    contract_sha256=contract, build_manifest=json.loads(build_manifest.read_text(encoding='utf-8')),
                    build_manifest_sha256=sha(build_manifest),
                    source_hashes={str(p.relative_to(ROOT)): sha(p) for p in (Path(__file__), ROOT/'src/gradient_comm_lab/ring_reference.py', ROOT/'src/gradient_comm_lab/logical_ring.py', ROOT/'src/gradient_comm_lab/ring_fixtures.py')})
    snapshot_path = directory / 'execution_start.json'
    if snapshot_path.exists():
        raise ValueError('Attempt already exists')
    snapshot_path.write_text(json.dumps(snapshot, indent=2) + '\n', encoding='utf-8')
    cuda_codec.set_build_manifest(build_manifest)
    module = cuda_codec.load_extension()
    if sha(module.__file__) != snapshot['build_manifest']['module_sha256']:
        raise ValueError('Actually loaded module differs from recorded build artifact')
    writer = Writer(directory, run, contract)
    try:
        for index, case in enumerate(cases):
            print(f"case {index+1}/{len(cases)} {case['case_id']}", flush=True)
            run_case(writer, run, case, build_manifest, gpu_module, fixture_module)
        run_buffer_checks(writer, run, inputs['cases'], build_manifest, gpu_module, fixture_module)
        run_invalid_checks(writer, run, build_manifest, gpu_module)
    finally:
        writer.close()
    expected_ids = {r['record_id'] for r in manifest['cases']}
    actual_ids = [r['record_id'] for r in writer.rows]
    failed = [r['record_id'] for r in writer.rows if r['status'] != 'pass']
    complete = not args.case_id and len(set(actual_ids)) == len(actual_ids) and set(actual_ids) == expected_ids
    summary = dict(attempt=args.attempt, formal=args.formal, record_count=len(actual_ids),
                   expected_record_count=len(expected_ids), failed=failed,
                   status='pass' if complete and not failed else 'fail' if failed else 'incomplete',
                   contract_sha256=contract, loaded_module_path=module.__file__,
                   loaded_module_sha256=sha(module.__file__))
    (directory / 'runner_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary), flush=True)
    return 0 if not failed else 1


def run_buffer_checks(writer, run, cases, build_manifest, gpu_module, fixture_module):
    from gradient_comm_lab.cuda_codec import CudaCodecPlan
    for bits in (4, 8):
        lut = lookup_table(bits).cuda()
        a = ((torch.arange(1024, dtype=torch.float32) % 31 - 15) / 32).to(torch.bfloat16)
        b = (-a + torch.tensor(0.125, dtype=torch.bfloat16)).to(torch.bfloat16)
        ma, sa, meta_a = threshold_fixture(f'logical_ring_v1:ownership_A_b{bits}', 1024)
        mb, sb, meta_b = threshold_fixture(f'logical_ring_v1:ownership_B_b{bits}', 1024)
        oracle = CpuReference(bits)
        pa = oracle.encode_native(a, ma, sa)
        expected_a = oracle.decode_native(pa).clone()
        pb = oracle.encode_native(b, mb, sb)
        expected_b = oracle.decode_native(pb).clone()
        plan = CudaCodecPlan(1024, bits, 'cuda')
        owned_a = gpu_module.clone_packet(plan.encode_into(a.cuda(), ma.cuda(), sa.cuda(), lut, checked=True))
        before_a = packet_identity(gpu_module.packet_snapshot(owned_a))
        owned_b = gpu_module.clone_packet(plan.encode_into(b.cuda(), mb.cuda(), sb.cuda(), lut, checked=True))
        after_a = packet_identity(gpu_module.packet_snapshot(owned_a))
        pointers = {name: {k: getattr(packet, k).untyped_storage().data_ptr() for k in PACKET_FIELDS}
                    for name, packet in (('A', owned_a), ('B', owned_b), ('plan', plan.packet))}
        distinct = all(len({pointers[name][key] for name in pointers}) == 3 for key in PACKET_FIELDS)
        decoded_a_view = plan.decode_into(owned_a, lut, checked=True)
        saved_a = decoded_a_view.clone()
        decoded_a_ptr = decoded_a_view.data_ptr()
        decoded_b_view = plan.decode_into(owned_b, lut, checked=True)
        same_output_storage = decoded_b_view.data_ptr() == decoded_a_ptr
        saved_b = decoded_b_view.clone()
        expected_wire_a = hashlib.sha256(pa.wire).hexdigest()
        expected_wire_b = hashlib.sha256(pb.wire).hexdigest()
        checks = {
            'two_packets_in_flight': (distinct and before_a == after_a and after_a['wire_sha256'] == expected_wire_a
                and packet_identity(gpu_module.packet_snapshot(owned_b))['wire_sha256'] == expected_wire_b
                and mismatch(saved_a, expected_a) == 0,
                dict(packet_A_before=before_a, packet_A_after_B=after_a, packet_B=packet_identity(gpu_module.packet_snapshot(owned_b)),
                     expected_wire_A=expected_wire_a, expected_wire_B=expected_wire_b, storage_pointers=pointers,
                     independent_packet_storage=distinct, decoded_A=identity(saved_a), expected_decoded_A=identity(expected_a))),
            'decode_output_reuse': (same_output_storage and len({saved_a.data_ptr(), saved_b.data_ptr(), decoded_a_ptr}) == 3
                and mismatch(saved_a, expected_a) == 0 and mismatch(saved_b, expected_b) == 0
                and mismatch(saved_a, saved_b) > 0,
                dict(plan_output_storage_reused=same_output_storage, plan_output_pointer=decoded_a_ptr,
                     saved_A_pointer=saved_a.data_ptr(), saved_B_pointer=saved_b.data_ptr(),
                     decoded_B_plan_pointer=decoded_b_view.data_ptr(),
                     decoded_A_saved=identity(saved_a), decoded_B=identity(saved_b),
                     expected_A=identity(expected_a), expected_B=identity(expected_b),
                     saved_A_bit_mismatch_count=mismatch(saved_a, expected_a), decoded_B_bit_mismatch_count=mismatch(saved_b, expected_b))),
        }
        candidates = [c for c in cases if c['bits'] == bits and c['world_size'] == 4 and c['logical_numel'] == 1025 and c['entry_mode'] == 'predivided_bf16']
        if len(candidates) < 2:
            checks['cross_collective_isolation'] = (False, dict(reason='Missing two distinct frozen N4 D1025 predivided cases'))
        else:
            ca, cb = candidates[:2]
            def replay(case):
                values = fixture_module.load_case(case, run)
                result = gpu_module.run_ring([x.cuda() for x in values], bits, case['entry_mode'],
                    fixture_module.FrozenThresholdProvider(run, device='cuda'), case_id=case['case_id'],
                    audit=True, build_manifest=build_manifest)
                return result
            first = replay(ca)
            initial = signature(first)
            middle = replay(cb)
            after_b = signature(first)
            last = replay(ca)
            last_sig = signature(last)
            reference = cpu_ring(fixture_module.load_case(ca, run), bits, ca['entry_mode'],
                fixture_module.FrozenThresholdProvider(run), case_id=ca['case_id'])
            expected_sig = signature(reference)
            passed = initial == after_b == last_sig == expected_sig
            checks['cross_collective_isolation'] = (passed,
                dict(case_A=ca['case_id'], case_B=cb['case_id'], execution_order=['A', 'B', 'A'],
                     first_A=initial, first_A_after_B=after_b, second_A=last_sig, expected_A=expected_sig,
                     B_outputs=[identity(x) for x in middle['outputs']]))
        for check, (passed, details) in checks.items():
            counterexample = None
            if not passed:
                counterexample = f'counterexamples/buffer_b{bits}_{check}.pt'
                target = writer.directory / counterexample
                target.parent.mkdir(parents=True, exist_ok=True)
                torch.save(dict(input_A=a, input_B=b, main_A=ma, scale_A=sa,
                    main_B=mb, scale_B=sb, packet_A=gpu_module.packet_snapshot(owned_a),
                    packet_B=gpu_module.packet_snapshot(owned_b), expected_A=expected_a,
                    expected_B=expected_b, saved_A=saved_a, saved_B=saved_b,
                    details=details), target)
            row = dict(record_id=f'buffer/b{bits}/{check}', case_id=f'buffer_b{bits}',
                method_id=METHOD_GPU, variant='buffer', nominal_bits=bits, check_type=check,
                status='pass' if passed else 'fail', details=details,
                thresholds_A=meta_a, thresholds_B=meta_b, counterexample_path=counterexample)
            writer.write('buffer_lifetime_checks.jsonl', row)
            writer.write('correctness.jsonl', row)


def run_invalid_checks(writer, run, build_manifest, gpu_module):
    for spec in invalid_specs():
        name = spec['name']
        inputs = [torch.zeros(256, device='cuda', dtype=torch.bfloat16) for _ in range(2)]
        options = dict(bits=4, entry_mode='predivided_bf16', case_id='invalid/' + name,
                       build_manifest=build_manifest, lut=lookup_table(4).cuda())
        calls = []
        def provider(event):
            calls.append(event)
            main = torch.zeros((64, 16), dtype=torch.bfloat16)
            scale = torch.zeros((4, 16), dtype=torch.bfloat16)
            key = json.dumps(event, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
            if name == 'threshold_result': return None
            if name == 'threshold_identity': key = 'wrong-event'
            if name == 'threshold_shape' or name == 'late_threshold_shape' and len(calls) == 2: main = main[:1]
            if name == 'threshold_dtype': main = main.float()
            if name == 'threshold_noncontiguous': main = torch.zeros((64, 32), dtype=torch.bfloat16)[:, ::2]
            if name == 'threshold_device': main = torch.empty((64, 16), device='meta', dtype=torch.bfloat16)
            if name == 'threshold_negative': main[0, 0] = -1
            if name == 'threshold_too_large': main[0, 0] = 131072
            if name == 'threshold_nan': main[0, 0] = float('nan')
            return dict(main_thresholds=main, scale_thresholds=scale, event_key=key)
        selected_provider = provider
        if name == 'bits2': options['bits'] = 2
        elif name == 'bits_bool': options['bits'] = True
        elif name == 'entry_mode': options['entry_mode'] = 'raw_bf16'
        elif name == 'workers_empty': inputs = []
        elif name == 'workers_three': inputs.append(inputs[0].clone())
        elif name == 'case_id': options['case_id'] = ''
        elif name == 'input_not_tensor': inputs[0] = None
        elif name == 'input_2d': inputs[0] = inputs[0].reshape(16, 16)
        elif name == 'input_noncontiguous': inputs[0] = torch.zeros(512, device='cuda', dtype=torch.bfloat16)[::2]
        elif name == 'input_dtype': inputs[0] = inputs[0].float()
        elif name == 'input_cpu': inputs = [x.cpu() for x in inputs]
        elif name == 'input_mixed_device': inputs[1] = inputs[1].cpu()
        elif name == 'input_length': inputs[1] = inputs[1][:-1]
        elif name == 'input_nan': inputs[0][0] = float('nan')
        elif name == 'input_inf': inputs[0][0] = float('inf')
        elif name == 'rank_duplicate': options['rank_order'] = [0, 0]
        elif name == 'rank_out_of_range': options['rank_order'] = [0, 2]
        elif name == 'rank_bool': options['rank_order'] = [False, 1]
        elif name == 'provider_missing': selected_provider = None
        elif name == 'lut_shape': options['lut'] = options['lut'][:-1]
        elif name == 'lut_dtype': options['lut'] = options['lut'].float()
        elif name == 'lut_device': options['lut'] = torch.empty(33, device='meta', dtype=torch.bfloat16)
        elif name == 'lut_nonmonotonic': options['lut'][1] = options['lut'][0]
        elif name == 'lut_nan': options['lut'][0] = float('nan')
        elif name == 'lut_identity': options['lut'].mul_(2)
        elif name == 'prepared_overflow':
            options['entry_mode'] = 'raw_fp32'
            inputs = [torch.full((256,), torch.finfo(torch.float32).max, device='cuda') for _ in range(2)]
        elif name == 'build_missing': options['build_manifest'] = None
        returned = None
        observed_type = observed_message = None
        try:
            returned = gpu_module.run_ring(inputs, threshold_provider=selected_provider, **options)
        except Exception as exc:
            observed_type, observed_message = type(exc).__name__, str(exc)
        passed = observed_type == 'ValueError' and observed_message == spec['message'] and returned is None
        if name == 'late_threshold_shape': passed &= len(calls) == 2
        row = dict(record_id='invalid/' + name, case_id='invalid/' + name, method_id=METHOD_GPU,
            variant='invalid', nominal_bits=4, check_type='expected_rejection',
            expected_exception_type='ValueError', expected_message=spec['message'],
            observed_exception_type=observed_type, observed_exception_message=observed_message,
            returned_result=returned is not None, threshold_calls_before_rejection=len(calls),
            status='pass' if passed else 'fail')
        writer.write('correctness.jsonl', row)
        writer.write('invalid_checks.jsonl', row)


if __name__ == '__main__':
    raise SystemExit(main())
