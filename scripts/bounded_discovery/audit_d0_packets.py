"""Independent CPU/stdlib parser of actual official D0 hierarchical MEE packets.

No official adapter, tensor codec, torch, CUDA or local G1 codec is imported.
Offsets and little-endian bit packing follow the pinned upstream native source.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[2]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def audit_case(run, row):
    bits, logical, padded = row['bits'], row['logical_numel'], row['padded_numel']
    path = run / 'd0_cases' / f"{row['case_id']}_b{bits}.bin"
    wire = path.read_bytes()
    errors = []
    if bits not in (4, 8):
        raise ValueError('Unexpected D0 stored code width')
    if padded <= 0 or padded % 256 or padded != ((logical + 255) // 256) * 256:
        errors.append('Native padding differs from selected 256-element contract')
    group_payload = 16 * bits // 8
    block_bytes = 16 * (group_payload + 1) + 2
    if len(wire) != (padded // 256) * block_bytes:
        raise ValueError('Actual serialized packet length is inconsistent')
    if sha(wire) != row['packet_sha256']:
        errors.append('Packet hash differs from D0 execution record')
    payload, scales, units, codes, unit_values = bytearray(), bytearray(), bytearray(), [], []
    rebuilt = bytearray()
    for block in range(padded // 256):
        start = block * block_bytes
        for group in range(16):
            offset = start + group * (group_payload + 1)
            packed = wire[offset:offset + group_payload]
            payload.extend(packed)
            current = [(packed[(i * bits) // 8] >> ((i * bits) % 8)) & ((1 << bits) - 1)
                       for i in range(16)]
            codes.extend(current)
            code_bytes = bytearray(group_payload)
            for i, code in enumerate(current):
                code_bytes[(i * bits) // 8] |= code << ((i * bits) % 8)
            rebuilt.extend(code_bytes)
            scale = wire[offset + group_payload]
            scales.append(scale)
            rebuilt.append(scale)
        unit_bytes = wire[start + block_bytes - 2:start + block_bytes]
        unit_word = struct.unpack('<H', unit_bytes)[0]
        unit_values.append(struct.unpack('<f', struct.pack('<I', unit_word << 16))[0])
        units.extend(unit_bytes)
        rebuilt.extend(unit_bytes)
    illegal_codes = sum(code > (1 << bits) - 2 for code in codes)
    illegal_scales = sum(not 1 <= value <= 255 for value in scales)
    illegal_units = sum(not math.isfinite(value) or value <= 0 for value in unit_values)
    if illegal_codes:
        errors.append('Encoder emitted a reserved unsigned code')
    if illegal_scales:
        errors.append('Native UINT8 scale is outside [1,255]')
    if illegal_units:
        errors.append('Stored BF16 scale unit is not finite and positive')
    if bytes(rebuilt) != wire:
        errors.append('Independent low-bit-first reserialization changed bytes')
    counts = dict(payload_bytes=len(payload), scale_code_bytes=len(scales),
                  super_scale_bytes=len(units), scale_bytes=len(scales) + len(units),
                  header_bytes=0, alignment_bytes=0, packet_total_bytes=len(wire))
    for name in ('payload_bytes', 'scale_bytes', 'header_bytes', 'alignment_bytes', 'packet_total_bytes'):
        if counts[name] != row['accounting'][name]:
            errors.append('Accounting component differs: ' + name)
    if len(payload) + len(scales) + len(units) != len(wire):
        errors.append('Parsed byte components do not cover the wire exactly')
    return dict(case_id=row['case_id'], bits=bits, logical_numel=logical, padded_numel=padded,
                packet_path=path.relative_to(run).as_posix(), packet_sha256=sha(wire),
                status='pass' if not errors else 'fail', errors=errors,
                code_count=len(codes), code_min=min(codes), code_max=max(codes),
                illegal_code_count=illegal_codes, scale_code_count=len(scales),
                scale_code_min=min(scales), scale_code_max=max(scales),
                illegal_scale_count=illegal_scales, super_scale_count=len(units)//2,
                super_scale_min=min(unit_values), super_scale_max=max(unit_values),
                illegal_super_scale_count=illegal_units, actual_component_bytes=counts,
                component_sha256=dict(concatenated_payload=sha(payload), concatenated_scale_codes=sha(scales),
                                      concatenated_super_scale_units=sha(units)),
                exact_reserialization=True if bytes(rebuilt) == wire else False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    run = (ROOT / args.run_dir).resolve()
    execution_path = run / 'logs/d0_execution.json'
    source = json.loads(execution_path.read_text(encoding='utf-8'))
    rows, errors = [], []
    keys = [(r['case_id'], r['bits']) for r in source['cases']]
    if len(set(keys)) != len(keys):
        errors.append('Duplicate D0 case identity')
    wanted = {f'{case}_b{bits}.bin' for case, bits in keys}
    found = {p.name for p in (run/'d0_cases').glob('*.bin')}
    if wanted != found:
        errors.append(dict(message='Binary packet file coverage differs', missing=sorted(wanted-found), extra=sorted(found-wanted)))
    for row in source['cases']:
        try:
            parsed = audit_case(run, row)
            rows.append(parsed)
            if parsed['status'] != 'pass':
                errors.append(dict(case_id=row['case_id'], bits=row['bits'], errors=parsed['errors']))
        except Exception as exc:
            errors.append(dict(case_id=row['case_id'], bits=row['bits'], exception_type=type(exc).__name__, message=str(exc)))
    result = dict(schema_version='bounded_discovery_d0_packet_parse_review_v1', run_id=run.name,
                  status='pass' if not errors else 'fail', generated_at=datetime.now(timezone.utc).isoformat(),
                  execution_device='CPU stdlib only; no tensor/native codec imported',
                  upstream_commit='96da2ee480bbfd8e2049de3fbd8595cc9d7a30b8',
                  layout_source='cuda_hierarchical_mee.cu:175-202 and 273-343; native low-coordinate code occupies low bits',
                  script_sha256=sha(Path(__file__).read_bytes()), d0_execution_sha256=sha(execution_path.read_bytes()),
                  expected_packet_count=len(keys), parsed_packet_count=len(rows), errors=errors, packets=rows,
                  scope_limit='Validates packed alphabet, byte accounting and independent parse/repack; does not claim GPU execution or independent numeric reimplementation.')
    destination = run / 'logs/d0_packet_parse_review.json'
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], packets=len(rows), errors=errors)))
    return 0 if not errors else 1


if __name__ == '__main__':
    raise SystemExit(main())
