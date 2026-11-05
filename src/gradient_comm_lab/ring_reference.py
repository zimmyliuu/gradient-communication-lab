"""Independent CPU logical ring using the sealed original CPU packet codec.

Routing is written here from the reviewed protocol, independently of the GPU
implementation. No GPU implementation or GPU arithmetic is imported.
"""
from __future__ import annotations

import json
import torch

from .cpu_reference import CpuReference, tensor_identity


def event_descriptor(case_id, bits, phase, rank, chunk, ordinal):
    return dict(case_id=case_id, nominal_bits=bits, phase=phase, rank=rank,
                chunk=chunk, encoding_ordinal=ordinal)


def canonical_event_key(event):
    return json.dumps(event, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def byte_tensor(value):
    return torch.frombuffer(bytearray(value), dtype=torch.uint8).clone()


def packet_snapshot(packet):
    n = packet.length
    return dict(payload=byte_tensor(packet.payload),
                scale_codes=byte_tensor(packet.scales[:n // 16]),
                top_scales=torch.frombuffer(bytearray(packet.scales[n // 16:]),
                                            dtype=torch.bfloat16).clone(),
                logical_numel=n, padded_numel=n, bits=packet.bits)


def run_ring(inputs, bits, entry_mode, threshold_provider, *, case_id,
             rank_order=None, audit=True, build_manifest=None, lut=None):
    """Execute the CPU oracle; tensors in audit snapshots have independent storage."""
    if type(bits) is not int or bits not in (4, 8):
        raise ValueError('Ring requires nominal b4/b8')
    if entry_mode not in ('raw_fp32', 'predivided_bf16'):
        raise ValueError('Invalid ring entry mode')
    if not isinstance(inputs, (list, tuple)) or not inputs:
        raise ValueError('Ring inputs must be a nonempty sequence')
    n = len(inputs)
    dtype = torch.float32 if entry_mode == 'raw_fp32' else torch.bfloat16
    for value in inputs:
        if not isinstance(value, torch.Tensor) or value.device.type != 'cpu' or value.dtype != dtype or value.ndim != 1 or not value.is_contiguous():
            raise ValueError('CPU ring input metadata mismatch')
        if not bool(torch.isfinite(value).all()):
            raise ValueError('Nonfinite ring input')
    d = inputs[0].numel()
    if any(x.numel() != d for x in inputs):
        raise ValueError('Ring input lengths differ')
    order = list(range(n)) if rank_order is None else list(rank_order)
    if len(order) != n or any(type(r) is not int for r in order) or sorted(order) != list(range(n)):
        raise ValueError('Rank order must be a permutation')
    p = ((d + n * 1024 - 1) // (n * 1024)) * n * 1024 if d and n > 1 else d
    length = p // n if d and n > 1 else 0
    result = dict(outputs=[], events=[], messages=[], intermediates=[],
                  threshold_consumption=[], packets={}, prepared_inputs=[],
                  metadata=dict(world_size=n, logical_numel=d, padded_numel=p,
                                chunk_numel=length, bits=bits, entry_mode=entry_mode,
                                rank_order=order, method_id='cpu_reference'))
    if n == 1 or d == 0:
        result['outputs'] = [x.clone() for x in inputs]
        result['prepared_inputs'] = [x.clone() for x in inputs]
        return result
    prepared = [x.to(torch.bfloat16).div(n) if entry_mode == 'raw_fp32' else x.clone() for x in inputs]
    if any(not bool(torch.isfinite(x).all()) for x in prepared):
        raise ValueError('Nonfinite ring preprocessed input')
    result['prepared_inputs'] = [x.clone() for x in prepared]
    local = [torch.nn.functional.pad(x, (0, p-d)).reshape(n, length) for x in prepared]
    codec = CpuReference(bits)
    if lut is not None:
        if lut.device.type != 'cpu' or not torch.equal(lut.view(torch.int16), codec.impl.tables[bits].view(torch.int16)):
            raise ValueError('CPU oracle LUT differs from frozen source')
    result['metadata']['lut_before'] = tensor_identity(codec.impl.tables[bits])

    def intermediate(kind, phase, hop, rank, chunk, source_id, value):
        if audit:
            result['intermediates'].append(dict(
                intermediate_id=f'{case_id}/b{bits}/{phase}/{kind}/h{hop}/r{rank}/c{chunk}',
                kind=kind, phase=phase, hop=hop, rank=rank, chunk=chunk,
                source_packet_id=source_id, value=value.clone()))

    def encode(value, phase, hop, rank, chunk):
        descriptor = event_descriptor(case_id, bits, phase, rank, chunk, hop)
        provided = threshold_provider(descriptor) if callable(threshold_provider) else threshold_provider.get(descriptor)
        main, scale = provided['main_thresholds'], provided['scale_thresholds']
        key = provided.get('event_key', canonical_event_key(descriptor))
        before = dict(main=tensor_identity(main), scale=tensor_identity(scale))
        packet = codec.encode_native(value, main, scale)
        after = dict(main=tensor_identity(main), scale=tensor_identity(scale))
        info = {k: v for k, v in provided.items() if k not in ('main_thresholds', 'scale_thresholds')}
        info.update(before=before, after=after)
        result['threshold_consumption'].append(dict(**descriptor, event_key=key, **before))
        if audit:
            snapshot = packet_snapshot(packet)
            result['events'].append(dict(**descriptor, event_key=key, event_id=key,
                                        input=value.clone(), packet=snapshot, threshold_info=info))
            result['packets'][key] = snapshot
        return packet, key

    def decode(item, phase, hop, rank, chunk):
        packet, key = item
        value = codec.decode_native(packet).clone()
        intermediate('decode', phase, hop, rank, chunk, key, value)
        return value

    def message(item, phase, hop, sender, chunk):
        packet, key = item
        snapshot = packet_snapshot(packet)
        result['messages'].append(dict(
            message_id=f'{case_id}/b{bits}/{phase}/h{hop}/r{sender}/c{chunk}',
            source_packet_id=key, phase=phase, hop=hop, sender=sender,
            receiver=(sender+1) % n, chunk=chunk, logical_numel=d, padded_numel=p,
            chunk_logical_numel=max(0, min(length, d-chunk*length)), chunk_padded_numel=length,
            payload_bytes=len(packet.payload), scale_codes_bytes=length//16,
            top_scales_bytes=2*(length//256), wire_bytes=len(packet.wire), packet=snapshot))

    incoming = [None] * n
    for hop in range(n-1):
        outgoing = [None] * n
        for rank in order:
            chunk = (rank-hop) % n
            value = local[rank][chunk].clone()
            source_id = None
            if hop:
                source_id = incoming[rank][1]
                value.add_(decode(incoming[rank], 'reduce_scatter', hop, rank, chunk))
            intermediate('partial_sum', 'reduce_scatter', hop, rank, chunk, source_id, value)
            outgoing[rank] = encode(value, 'reduce_scatter', hop, rank, chunk)
            message(outgoing[rank], 'reduce_scatter', hop, rank, chunk)
        incoming = [outgoing[(rank-1) % n] for rank in range(n)]
    output = [torch.empty((n, length), dtype=torch.bfloat16) for _ in range(n)]
    owned = [None] * n
    for rank in order:
        chunk = (rank+1) % n
        value = local[rank][chunk].clone()
        value.add_(decode(incoming[rank], 'sink', n-1, rank, chunk))
        intermediate('partial_sum', 'sink', n-1, rank, chunk, incoming[rank][1], value)
        owned[rank] = encode(value, 'sink', n-1, rank, chunk)
        output[rank][chunk].copy_(decode(owned[rank], 'sink_self', n-1, rank, chunk))
    for hop in range(n-1):
        for rank in order:
            message(owned[rank], 'all_gather', hop, rank, (rank+1-hop) % n)
        received = [owned[(rank-1) % n] for rank in range(n)]
        for rank in order:
            chunk = (rank-hop) % n
            output[rank][chunk].copy_(decode(received[rank], 'all_gather', hop, rank, chunk))
        owned = received
    result['outputs'] = [x.reshape(-1)[:d].clone().to(dtype) for x in output]
    result['metadata']['lut_after'] = tensor_identity(codec.impl.tables[bits])
    return result
