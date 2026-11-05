"""CPU-only parser for frozen official interleaved hierarchical MEE packets.

Indices of codes/groups/units are in native padded coordinate order. Endpoint
codes are observed boundary codes, never treated as measured clamp events.
"""
import hashlib
import numpy as np


def parse_packet(wire, padded_numel, bits=8):
    if bits not in (4, 8) or padded_numel <= 0 or padded_numel % 256:
        raise ValueError('Illegal native packet configuration')
    payload_per_group=16*bits//8; block_bytes=16*(payload_per_group+1)+2
    if len(wire)!=padded_numel//256*block_bytes:raise ValueError('Packet byte count differs from native layout')
    blocks=np.frombuffer(wire,dtype=np.uint8).reshape(-1,block_bytes)
    groups=blocks[:,:-2].reshape(-1,16,payload_per_group+1)
    packed=groups[:,:,:payload_per_group]
    if bits==8:codes=packed.copy().reshape(-1)
    else:
        codes=np.empty((*packed.shape[:2],16),dtype=np.uint8)
        codes[:,:,0::2]=packed&15;codes[:,:,1::2]=packed>>4;codes=codes.reshape(-1)
    scales=groups[:,:,-1].copy().reshape(-1)
    words=blocks[:,-2].astype(np.uint16)|(blocks[:,-1].astype(np.uint16)<<8)
    values=(words.astype(np.uint32)<<16).view(np.float32)
    reserved=int(np.count_nonzero(codes==(1<<bits)-1))
    invalid_group=int(np.count_nonzero(scales==0))
    invalid_units=int(np.count_nonzero(~np.isfinite(values)|(values<=0)))
    return dict(main_codes=codes,group_scale_codes=scales,super_scale_words=words,
                super_scale_values=values,packet_sha256=hashlib.sha256(wire).hexdigest(),
                packet_bytes=len(wire),payload_bytes=padded_numel*bits//8,
                scale_bytes=len(scales)+2*len(words),reserved_code_count=reserved,
                invalid_group_scale_count=invalid_group,invalid_super_scale_count=invalid_units,
                invalid_scale_count=invalid_group+invalid_units,
                boundary_code_fraction=float(np.count_nonzero((codes==0)|(codes==(1<<bits)-2)))/len(codes),
                status='pass' if reserved+invalid_group+invalid_units==0 else 'fail')


def packet_summary(parsed):
    return {k:v for k,v in parsed.items() if not isinstance(v,np.ndarray)}
