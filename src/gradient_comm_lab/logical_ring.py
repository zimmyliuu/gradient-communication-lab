"""Single-GPU fixed-codec logical ring, with round barriers and owned messages.

This is a numerical protocol simulator, not a network transport or a concurrent
multi-GPU collective. Inputs, messages, partial sums and output remain on CUDA;
audit records hold CUDA tensor snapshots. Hashing/copying them to CPU belongs to
the external checker. Checked scalar predicates may synchronize at API boundaries.
The accepted G1 sources and extension are reused without modification.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import torch

from . import cuda_codec
from .torch_codec import DevicePacket
from .cpu_reference import lookup_table

METHOD_ID = 'cuda_fixed_ring_repaired'
EXPECTED_MODULE_NAME = 'gradient_comm_lab_cuda_a02'
EXPECTED_MODULE_SHA256 = 'c4e37ff9394580bb15fe5d277115f616a439f2741528dff6826bc77e85f06b58'
INVALID_MESSAGES = {
    'bits': 'RING_BITS: expected nominal b4/b8',
    'entry_mode': 'RING_ENTRY_MODE: expected raw_fp32 or predivided_bf16',
    'workers': 'RING_WORKERS: expected a list or tuple with 1, 2 or 4 workers',
    'input_tensor': 'RING_INPUT: expected a contiguous one-dimensional tensor',
    'input_dtype': 'RING_INPUT_DTYPE: input dtype does not match entry mode',
    'input_device': 'RING_INPUT_DEVICE: expected CUDA tensors on one device',
    'input_length': 'RING_INPUT_LENGTH: worker lengths must match',
    'nonfinite_input': 'RING_NONFINITE_INPUT: inputs must be finite',
    'rank_order': 'RING_RANK_ORDER: expected a permutation of worker ranks',
    'case_id': 'RING_CASE_ID: expected a nonempty string',
    'provider': 'RING_THRESHOLDS: expected a callable provider or get method',
    'threshold_result': 'RING_THRESHOLDS: provider must return thresholds for this event',
    'threshold_identity': 'RING_THRESHOLD_IDENTITY: event key differs from requested event',
    'threshold_metadata': 'RING_THRESHOLD_METADATA: expected contiguous BF16 tensors of canonical shape',
    'threshold_device': 'RING_THRESHOLD_DEVICE: expected CPU or the input CUDA device',
    'threshold_values': 'RING_THRESHOLD_VALUES: expected finite values in [0,65536]',
    'lut_metadata': 'RING_LUT_METADATA: expected contiguous one-dimensional BF16 LUT',
    'lut_device': 'RING_LUT_DEVICE: expected CPU or the input CUDA device',
    'lut_values': 'RING_LUT_VALUES: LUT must be finite and strictly ascending',
    'lut_identity': 'RING_LUT_IDENTITY: LUT differs from frozen CPU source',
    'prepared_overflow': 'RING_NONFINITE_PREPARED_INPUT: BF16 entry conversion overflowed',
    'partial_overflow': 'RING_NONFINITE_PARTIAL_SUM: BF16 addition overflowed',
    'build': 'RING_BUILD: expected the accepted G1 a02 module identity',
}


def event_key(event):
    """Serialization only; routing is independently computed below."""
    return json.dumps(event, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def clone_packet(packet):
    """Independent payload/scales/top storages, safe across future plan reuse.

    All encoding packets are retained through collective completion. AllGather
    only borrows these immutable owned packets; it neither changes nor re-encodes
    them. recordStream alone would not provide this overwrite protection.
    """
    if not isinstance(packet, DevicePacket):
        raise ValueError('RING_PACKET: expected a DevicePacket')
    return DevicePacket(packet.payload.clone(), packet.scale_codes.clone(), packet.top_scales.clone(),
                        packet.logical_numel, packet.padded_numel, packet.bits)


def packet_snapshot(packet):
    """Read-only record of an already-owned packet; no additional tensor copy."""
    return dict(payload=packet.payload, scale_codes=packet.scale_codes, top_scales=packet.top_scales,
                logical_numel=packet.logical_numel, padded_numel=packet.padded_numel, bits=packet.bits)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _build_identity(path, *, load):
    if path is None:
        raise ValueError(INVALID_MESSAGES['build'])
    path = Path(path).resolve()
    try:
        manifest = json.loads(path.read_text(encoding='utf-8'))
        artifact = Path(manifest['module_path']).resolve()
        if (manifest['module_name'] != EXPECTED_MODULE_NAME or manifest['module_sha256'] != EXPECTED_MODULE_SHA256
                or _sha(artifact) != EXPECTED_MODULE_SHA256):
            raise ValueError(INVALID_MESSAGES['build'])
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(INVALID_MESSAGES['build']) from error
    identity = dict(manifest_path=str(path), manifest_sha256=_sha(path), module_path=str(artifact),
                    module_name=manifest['module_name'], module_sha256=manifest['module_sha256'],
                    module_loaded=False)
    if load:
        cuda_codec.set_build_manifest(path)
        module = cuda_codec.load_extension()
        loaded = Path(module.__file__).resolve()
        if loaded != artifact or _sha(loaded) != EXPECTED_MODULE_SHA256:
            raise ValueError(INVALID_MESSAGES['build'])
        identity.update(module_loaded=True, actually_loaded_module_path=str(loaded),
                        actually_loaded_module_sha256=_sha(loaded))
    return identity


def _validate_inputs(inputs, bits, entry_mode, case_id, rank_order):
    if type(bits) is not int or bits not in (4, 8):
        raise ValueError(INVALID_MESSAGES['bits'])
    if entry_mode not in ('raw_fp32', 'predivided_bf16'):
        raise ValueError(INVALID_MESSAGES['entry_mode'])
    if not isinstance(inputs, (list, tuple)) or len(inputs) not in (1, 2, 4):
        raise ValueError(INVALID_MESSAGES['workers'])
    if not isinstance(case_id, str) or not case_id:
        raise ValueError(INVALID_MESSAGES['case_id'])
    n = len(inputs)
    dtype = torch.float32 if entry_mode == 'raw_fp32' else torch.bfloat16
    device = None
    length = None
    for value in inputs:
        if not isinstance(value, torch.Tensor) or value.ndim != 1 or not value.is_contiguous():
            raise ValueError(INVALID_MESSAGES['input_tensor'])
        if value.dtype != dtype:
            raise ValueError(INVALID_MESSAGES['input_dtype'])
        if value.device.type != 'cuda' or (device is not None and value.device != device):
            raise ValueError(INVALID_MESSAGES['input_device'])
        if length is not None and value.numel() != length:
            raise ValueError(INVALID_MESSAGES['input_length'])
        device, length = value.device, value.numel()
        if not bool(torch.isfinite(value).all()):
            raise ValueError(INVALID_MESSAGES['nonfinite_input'])
    order = list(range(n)) if rank_order is None else rank_order
    if (not isinstance(order, (list, tuple)) or len(order) != n or
            any(type(r) is not int for r in order) or set(order) != set(range(n))):
        raise ValueError(INVALID_MESSAGES['rank_order'])
    return n, length, device, list(order)


@torch.no_grad()
def run_ring(inputs, bits, entry_mode, threshold_provider, *, case_id,
             rank_order=None, audit=True, build_manifest=None, lut=None):
    """Execute fixed b4/b8 RS + sink quantization + byte-forwarding AllGather.

    Inputs must already be CUDA tensors. A provider returns a dict containing
    main_thresholds, scale_thresholds, event_key and optional seed/main/scale
    identities. CPU thresholds/LUT may be supplied and are copied during setup;
    these copies are outside any claimed codec hot path. A callable provider is
    preferred; an object's get(event) or a threshold pair is also accepted.

    Failure raises and returns no aggregate/result/partial-message collection.
    There is no persistent collective state between calls. The returned owned
    packet dict intentionally keeps all encoded storages alive after completion.
    """
    n, d, device, order = _validate_inputs(inputs, bits, entry_mode, case_id, rank_order)
    bypass = n == 1 or d == 0
    build = _build_identity(build_manifest, load=not bypass)
    padded = 0 if d == 0 else (d if n == 1 else ((d + n*1024 - 1)//(n*1024))*(n*1024))
    length = 0 if bypass else padded//n
    metadata = dict(method_id=METHOD_ID, case_id=case_id, nominal_bits=bits, entry_mode=entry_mode,
                    world_size=n, logical_numel=d, padded_numel=padded, chunk_numel=length,
                    device=str(device), rank_order=order, audit=bool(audit), build=build,
                    bypass=bypass, bypass_reason=('empty' if d == 0 else 'one_worker' if n == 1 else None),
                    execution='single_gpu_sequential_round_synchronous',
                    network_transport_implemented=False, collective_state_persistent=False,
                    ownership='Every encode gets three cloned owned buffers retained until collective completion; decode outputs cloned before plan reuse.')
    if bypass:
        # In particular, raw FP32 N=1 never rounds to BF16, and D=0 never calls
        # the provider or codec. Entry dtype is preserved in either bypass.
        metadata.update(quantization_events=0, message_count=0, threshold_consumption_count=0,
                        decode_count=0, total_sent_wire_bytes=0)
        return dict(outputs=[x.clone() for x in inputs], prepared_inputs=[x.clone() for x in inputs],
                    events=[], messages=[], intermediates=[], threshold_consumption=[],
                    owned_packets={}, metadata=metadata)

    if callable(threshold_provider):
        provider = threshold_provider
    elif not isinstance(threshold_provider, dict) and callable(getattr(threshold_provider, 'get', None)):
        provider = threshold_provider.get
    else:
        raise ValueError(INVALID_MESSAGES['provider'])
    if lut is None:
        lut = lookup_table(bits)
    if (not isinstance(lut, torch.Tensor) or lut.dtype != torch.bfloat16 or
            tuple(lut.shape) != (2*(1 << bits)+1,) or not lut.is_contiguous()):
        raise ValueError(INVALID_MESSAGES['lut_metadata'])
    if lut.device.type != 'cpu' and lut.device != device:
        raise ValueError(INVALID_MESSAGES['lut_device'])
    if not bool(torch.isfinite(lut).all() & (lut[1:] > lut[:-1]).all()):
        raise ValueError(INVALID_MESSAGES['lut_values'])
    canonical_lut = lookup_table(bits).to(lut.device)
    if not torch.equal(lut.view(torch.int16), canonical_lut.view(torch.int16)):
        raise ValueError(INVALID_MESSAGES['lut_identity'])
    lut_device = lut.to(device)
    prepared = []
    for value in inputs:
        if entry_mode == 'raw_fp32':
            converted = value.to(torch.bfloat16)
            converted.div_(torch.tensor(n, dtype=torch.bfloat16, device=device))
        else:
            converted = value.clone()
        if not bool(torch.isfinite(converted).all()):
            raise ValueError(INVALID_MESSAGES['prepared_overflow'])
        prepared.append(converted)
    local = []
    for value in prepared:
        storage = torch.zeros(padded, dtype=torch.bfloat16, device=device)
        storage[:d].copy_(value)
        local.append(storage.reshape(n, length))

    # One plan is deliberately reused across all ranks. The explicit clones below
    # must protect every in-flight message and intermediate from that reuse.
    plan = cuda_codec.CudaCodecPlan(length, bits, device=device)
    events, messages, intermediates, consumed = [], [], [], []
    owned = {}
    decode_count = 0

    def checkpoint(kind, phase, hop, rank, chunk, source_packet_id, value):
        if audit:
            intermediates.append(dict(
                intermediate_id=f'{case_id}/b{bits}/{phase}/{kind}/h{hop}/r{rank}/c{chunk}',
                kind=kind, phase=phase, hop=hop, rank=rank, chunk=chunk,
                source_packet_id=source_packet_id, value=value.clone()))

    def decode(item, phase, hop, rank, chunk):
        nonlocal decode_count
        if item is None or item['chunk'] != chunk:
            raise RuntimeError('RING_INTERNAL_ROUTE: received chunk differs from scheduled chunk')
        # Never retain a view of plan.output across another decode.
        result = plan.decode_into(item['packet'], lut_device, checked=True).clone()
        decode_count += 1
        checkpoint('decode', phase, hop, rank, chunk, item['packet_id'], result)
        return result

    def encode(value, phase, hop, rank, chunk):
        descriptor = dict(case_id=case_id, nominal_bits=bits, phase=phase, rank=rank,
                          chunk=chunk, encoding_ordinal=hop)
        expected_key = event_key(descriptor)
        supplied = provider(dict(descriptor))
        if isinstance(supplied, (list, tuple)) and len(supplied) == 2:
            supplied = dict(main_thresholds=supplied[0], scale_thresholds=supplied[1], event_key=expected_key)
        if not isinstance(supplied, dict) or not all(k in supplied for k in ('main_thresholds', 'scale_thresholds')):
            raise ValueError(INVALID_MESSAGES['threshold_result'])
        key = supplied.get('event_key', expected_key)
        if key != expected_key:
            raise ValueError(INVALID_MESSAGES['threshold_identity'])
        if key in owned:
            raise RuntimeError('RING_INTERNAL_EVENT: duplicate encoding event')
        thresholds = []
        for tensor, shape in ((supplied['main_thresholds'], (length//16, 16)),
                              (supplied['scale_thresholds'], (length//256, 16))):
            if (not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.bfloat16 or
                    tuple(tensor.shape) != shape or not tensor.is_contiguous()):
                raise ValueError(INVALID_MESSAGES['threshold_metadata'])
            if tensor.device.type != 'cpu' and tensor.device != device:
                raise ValueError(INVALID_MESSAGES['threshold_device'])
            if not bool((torch.isfinite(tensor) & (tensor >= 0) & (tensor <= 65536)).all()):
                raise ValueError(INVALID_MESSAGES['threshold_values'])
            thresholds.append(tensor.to(device))
        main, scale = thresholds
        packet = clone_packet(plan.encode_into(value, main, scale, lut_device, checked=True))
        owned[key] = packet
        threshold_info = {k: supplied[k] for k in ('seed', 'main', 'scale') if k in supplied}
        consumed.append({**descriptor, 'event_id': key, 'event_key': key, **threshold_info})
        record = {**descriptor, 'event_id': key, 'event_key': key,
                  'threshold_info': threshold_info, 'packet': packet_snapshot(packet)}
        if audit:
            record['input'] = value.clone()
            record['main_thresholds'] = main.clone()
            record['scale_thresholds'] = scale.clone()
        events.append(record)
        return dict(packet=packet, packet_id=key, chunk=chunk)

    def send(item, phase, hop, sender, chunk, queue):
        receiver = (sender + 1) % n
        if queue[receiver] is not None or item['chunk'] != chunk:
            raise RuntimeError('RING_INTERNAL_ROUTE: invalid send or duplicate receiver')
        packet = item['packet']
        payload_bytes = packet.payload.numel()*packet.payload.element_size()
        scale_bytes = packet.scale_codes.numel()*packet.scale_codes.element_size()
        top_bytes = packet.top_scales.numel()*packet.top_scales.element_size()
        messages.append(dict(message_id=f'{case_id}/b{bits}/{phase}/h{hop}/r{sender}/c{chunk}',
            source_packet_id=item['packet_id'], phase=phase, hop=hop, sender=sender,
            receiver=receiver, chunk=chunk, logical_numel=d, padded_numel=padded,
            chunk_logical_numel=max(0, min(length, d - chunk*length)), chunk_padded_numel=length,
            payload_bytes=payload_bytes, scale_codes_bytes=scale_bytes, top_scales_bytes=top_bytes,
            scale_bytes=scale_bytes+top_bytes, header_bytes=0, alignment_bytes=0,
            wire_bytes=payload_bytes+scale_bytes+top_bytes, packet=packet_snapshot(packet)))
        # This is a logical directed transmission, not a CUDA memcpy or network
        # transfer. The receiving queue borrows the immutable owned packet.
        queue[receiver] = item

    incoming = [None]*n
    for hop in range(n-1):
        following = [None]*n
        for rank in order:
            chunk = (rank - hop) % n
            value = local[rank][chunk].clone()
            source_id = None
            if hop:
                received = decode(incoming[rank], 'reduce_scatter', hop, rank, chunk)
                source_id = incoming[rank]['packet_id']
                value.add_(received)
                if not bool(torch.isfinite(value).all()):
                    raise ValueError(INVALID_MESSAGES['partial_overflow'])
            checkpoint('partial_sum', 'reduce_scatter', hop, rank, chunk, source_id, value)
            packet = encode(value, 'reduce_scatter', hop, rank, chunk)
            send(packet, 'reduce_scatter', hop, rank, chunk, following)
        # Round barrier: no consumer can observe a packet from the current round
        # while any sender is still operating on the previous round's inbox.
        incoming = following

    output = [torch.empty((n, length), dtype=torch.bfloat16, device=device) for _ in range(n)]
    final = [None]*n
    for rank in order:
        chunk = (rank + 1) % n
        received = decode(incoming[rank], 'sink', n-1, rank, chunk)
        value = local[rank][chunk].clone().add_(received)
        if not bool(torch.isfinite(value).all()):
            raise ValueError(INVALID_MESSAGES['partial_overflow'])
        checkpoint('partial_sum', 'sink', n-1, rank, chunk, incoming[rank]['packet_id'], value)
        item = encode(value, 'sink', n-1, rank, chunk)
        final[rank] = item
        output[rank][chunk].copy_(decode(item, 'sink_self', n-1, rank, chunk))

    outgoing = final
    for hop in range(n-1):
        following = [None]*n
        for rank in order:
            chunk = (rank + 1 - hop) % n
            # AllGather sends exactly the original final packet, with no codec
            # encode or threshold access. Packet IDs remain the sink event IDs.
            send(outgoing[rank], 'all_gather', hop, rank, chunk, following)
        for rank in order:
            chunk = (rank - hop) % n
            output[rank][chunk].copy_(decode(following[rank], 'all_gather', hop, rank, chunk))
        outgoing = following

    results = [value.reshape(-1)[:d].clone() for value in output]
    if entry_mode == 'raw_fp32':
        results = [value.to(torch.float32) for value in results]
    events.sort(key=lambda r: (r['encoding_ordinal'], r['rank']))
    consumed.sort(key=lambda r: (r['encoding_ordinal'], r['rank']))
    messages.sort(key=lambda r: (r['phase'] == 'all_gather', r['hop'], r['sender']))
    phase_order = {'reduce_scatter': 0, 'sink': 1, 'sink_self': 2, 'all_gather': 3}
    intermediates.sort(key=lambda r: (phase_order[r['phase']], r['hop'], r['rank'], r['kind']))
    metadata.update(quantization_events=len(events), message_count=len(messages), decode_count=decode_count,
                    threshold_consumption_count=len(consumed),
                    total_sent_wire_bytes=sum(row['wire_bytes'] for row in messages))
    return dict(outputs=results, prepared_inputs=[x.clone() for x in prepared], events=events,
                messages=messages, intermediates=intermediates, threshold_consumption=consumed,
                owned_packets=owned, metadata=metadata)
