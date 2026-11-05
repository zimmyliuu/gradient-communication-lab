"""G1b exact CPU parity, fixed rejection rules and stream/ABI qualification.

Derived from the sealed G1a harness; original files remain unchanged.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import traceback
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gradient_comm_lab.fixtures_v2 import prepare_fixtures,load_fixture,sha
from gradient_comm_lab.cpu_reference import CpuReference,QuantizedPacket,pack_unsigned,unpack_unsigned,tensor_identity,lookup_table,pad_input,threshold_fixture


METRICS=('code_mismatch_count','scale_code_mismatch_count','top_scale_bit_mismatch_count','packet_byte_mismatch_count',
 'decoded_bit_mismatch_count','expected_wire_bytes','actual_wire_bytes','quant_squared_error','quant_mse','quant_input_energy',
 'quant_normalized_squared_error','quant_max_abs_error','undefined_metric_reason','effective_bits')

def raw(t):return t.detach().cpu().contiguous().view(torch.uint8)
def byte_tensor(value):return torch.frombuffer(bytearray(value),dtype=torch.uint8).clone() if value else torch.empty(0,dtype=torch.uint8)
def mismatch(a,b):
    if a.shape!=b.shape:return max(a.numel(),b.numel())
    return int((a!=b).sum())
def gpu_wire(packet):return torch.cat((raw(packet.payload),raw(packet.scale_codes),raw(packet.top_scales)))
def bit_mismatch(a,b):return mismatch(a.detach().cpu().contiguous().view(torch.int16),b.detach().cpu().contiguous().view(torch.int16))

def difference_indices(a,b):
    a=a.detach().cpu().reshape(-1);b=b.detach().cpu().reshape(-1)
    if a.numel()!=b.numel():return {'left_numel':a.numel(),'right_numel':b.numel()}
    return torch.nonzero(a!=b).reshape(-1)[:64].tolist()

def distort(x,decoded,wire_bytes):
    n=x.numel()
    if not n:return {'quant_squared_error':0.0,'quant_mse':None,'quant_input_energy':0.0,
        'quant_normalized_squared_error':None,'quant_max_abs_error':None,'effective_bits':None,'undefined_metric_reason':'zero_length'}
    error=decoded.cpu().double()-x.double();sq=float(error.square().sum());energy=float(x.double().square().sum())
    return {'quant_squared_error':sq,'quant_mse':sq/n,'quant_input_energy':energy,
        'quant_normalized_squared_error':sq/energy if energy else None,'quant_max_abs_error':float(error.abs().max()),
        'effective_bits':8*wire_bytes/n,'undefined_metric_reason':None if energy else 'zero_input_energy'}

def base_record(run,contract,input_id,bits,length,check,method='torch_cuda_repaired'):
    return {'run_id':run.name,'contract_sha256':contract,'case_id':f'{input_id}_b{bits}_{check}',
        'input_id':input_id,'method_id':method,'reference_id':'cpu_ref_repaired','check_type':check,
        'nominal_bits':bits,'stored_bits':bits+1,'original_numel':length,'padded_numel':(length+255)//256*256,
        'status':'pass','failure_reason':None,'counterexample_path':None,**{k:None for k in METRICS}}

def qualify_fixture(row,bits,tc):
    cpu=load_fixture(row);x=cpu['input_logical'];main=cpu['main_thresholds'];scale=cpu['scale_thresholds'];lut=cpu['lookup_tables'][bits]
    reference=CpuReference(bits);expected=reference.roundtrip(x,main,scale)
    gpu={k:v.cuda() for k,v in (('x',x),('main',main),('scale',scale),('lut',lut))}
    before={k:raw(v).clone() for k,v in gpu.items()}
    plan=tc.TorchCodecPlan(x.numel(),bits,device='cuda')
    packet=plan.encode_into(gpu['x'],gpu['main'],gpu['scale'],gpu['lut'],checked=False)
    encoded=gpu_wire(packet)
    refpacket=expected['packet'];wanted=byte_tensor(expected['wire'])
    counts={'packet_byte_mismatch_count':mismatch(encoded,wanted),'expected_wire_bytes':len(expected['wire']),'actual_wire_bytes':encoded.numel()}
    if x.numel():
        audit=expected['audit']
        unsigned=tc.unpack_unsigned_cuda(packet.payload,bits+1,packet.padded_numel,checked=True,max_code=1<<bits).cpu()
        counts.update(code_mismatch_count=mismatch(unsigned,audit['codes']+(1<<(bits-1))),
            scale_code_mismatch_count=mismatch(packet.scale_codes.cpu(),audit['scale_codes']),
            top_scale_bit_mismatch_count=bit_mismatch(packet.top_scales,audit['top_scales']))
    else:counts.update(code_mismatch_count=0,scale_code_mismatch_count=0,top_scale_bit_mismatch_count=0)
    # Decode the CPU-produced legal packet, independently from the GPU encoder.
    if refpacket is not None:
        scales=refpacket.scales;n=refpacket.length
        canonical=tc.DevicePacket(payload=byte_tensor(refpacket.payload).cuda(),scale_codes=byte_tensor(scales[:n//16]).cuda(),
            top_scales=torch.frombuffer(bytearray(scales[n//16:]),dtype=torch.bfloat16).clone().cuda(),
            logical_numel=x.numel(),padded_numel=n,bits=bits)
    else:canonical=packet
    decoded=plan.decode_into(canonical,gpu['lut'],checked=True).cpu().clone()
    decode_count=bit_mismatch(decoded,expected['decoded'])
    roundtrip=plan.roundtrip_into(gpu['x'],gpu['main'],gpu['scale'],gpu['lut'],checked=True).cpu().clone()
    roundtrip_count=bit_mismatch(roundtrip,expected['decoded'])
    repeat_wire=gpu_wire(plan.packet)
    second=plan.roundtrip_into(gpu['x'],gpu['main'],gpu['scale'],gpu['lut'],checked=False).cpu().clone()
    repeat_count=bit_mismatch(second,roundtrip)+mismatch(gpu_wire(plan.packet),repeat_wire)
    changed=sum(mismatch(raw(gpu[k]),v) for k,v in before.items())
    details={'counts':counts,'decode_count':decode_count,'roundtrip_count':roundtrip_count,'repeat_count':repeat_count,
        'changed':changed,'quantization_events':plan.last_quantization_events,'quant':distort(x,roundtrip,encoded.numel()),'CPU_quant':distort(x,expected['decoded'],len(expected['wire']))}
    if any(counts[k] for k in ('code_mismatch_count','scale_code_mismatch_count','top_scale_bit_mismatch_count','packet_byte_mismatch_count')) or decode_count or roundtrip_count or repeat_count or changed:
        details['counterexample']={'fixture_path':row['fixture_path'],'cpu_packet':expected['wire'],'gpu_packet':encoded,
            'cpu_decoded':expected['decoded'],'gpu_decoded':decoded,'gpu_roundtrip':roundtrip,
            'cpu_audit':expected['audit'],'gpu_signed_codes':tc.unpack_unsigned_cuda(plan.packet.payload,bits+1,plan.padded_numel,checked=False).cpu().clone()-(1<<(bits-1)),
            'first_64_packet_difference_indices':difference_indices(encoded,wanted),
            'first_64_decoded_bit_difference_indices':difference_indices(decoded.view(torch.int16),expected['decoded'].view(torch.int16))}
    return details

def protocol_checks(tc):
    """Yield independently checked pack, decode and invalid input cases."""
    for bits in (4,8):
        width=bits+1;bound=1<<(bits-1)
        # Independent exhaustive scalar probe: preserve the BF16 epsilon operand
        # before the addition, including every finite positive/negative bit pattern.
        patterns=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
        finite=patterns[torch.isfinite(patterns)].contiguous()
        epsilon=torch.tensor(1e-23,dtype=torch.bfloat16)
        expected_add=finite+epsilon
        probe_plan=tc.TorchCodecPlan(0,bits,device='cuda')
        observed_add=tc.bf16_add_epsilon(finite.cuda()) if hasattr(tc,'bf16_add_epsilon') else finite.cuda()+probe_plan.epsilon
        yield bits,'epsilon_all_65280_finite_bf16','encode',finite.numel(),{
            'top_scale_bit_mismatch_count':bit_mismatch(observed_add,expected_add),
            'undefined_metric_reason':'arithmetic_probe_not_quantization'},None
        # Force group maxima to the legal LUT endpoint, so normalization scale is
        # exactly one. This tests actual lookup boundaries rather than only values
        # which were boundaries before an unrelated normalization step.
        table=lookup_table(bits);central=table[(1<<bits)-bound:(1<<bits)+bound+1]
        midpoint=((central[:-1].float()+central[1:].float())/2).to(torch.bfloat16)
        boundary=torch.cat((central,midpoint,torch.nextafter(midpoint,torch.full_like(midpoint,float('-inf'))),
                            torch.nextafter(midpoint,torch.full_like(midpoint,float('inf')))))
        groups=(boundary.numel()+14)//15
        boundary_input=central[-1].repeat(((groups+15)//16)*16,16)
        for i,value in enumerate(boundary):boundary_input[i//15,i%15+1]=value
        boundary_input=boundary_input.reshape(-1).contiguous()
        for threshold_value in (0.,65536.):
            m=torch.full((boundary_input.numel()//16,16),threshold_value,dtype=torch.bfloat16)
            s=torch.full((boundary_input.numel()//256,16),threshold_value,dtype=torch.bfloat16)
            cp=CpuReference(bits).roundtrip(boundary_input,m,s)
            plan=tc.TorchCodecPlan(boundary_input.numel(),bits,device='cuda')
            out=plan.roundtrip_into(boundary_input.cuda(),m.cuda(),s.cuda(),table.cuda(),checked=True)
            yield bits,f'normalized_lut_boundaries_threshold{int(threshold_value)}','roundtrip',boundary_input.numel(),{
                'code_mismatch_count':mismatch(tc.unpack_unsigned_cuda(plan.packet.payload,bits+1,plan.padded_numel,max_code=1<<bits).cpu()-bound,cp['audit']['codes']),
                'scale_code_mismatch_count':mismatch(plan.packet.scale_codes.cpu(),cp['audit']['scale_codes']),
                'top_scale_bit_mismatch_count':bit_mismatch(plan.packet.top_scales,cp['audit']['top_scales']),
                'packet_byte_mismatch_count':mismatch(gpu_wire(plan.packet),byte_tensor(cp['wire'])),
                'decoded_bit_mismatch_count':bit_mismatch(out,cp['decoded']),
                'normalization_scale_bit_mismatch_count':bit_mismatch(cp['audit']['scales_before'],torch.ones_like(cp['audit']['scales_before'])),
                'expected_wire_bytes':len(cp['wire']),'actual_wire_bytes':plan.packet.wire_bytes,
                **distort(boundary_input,out,len(cp['wire']))},None
        for count in (0,1,3,(1<<bits)+1,513):
            codes=(torch.arange(count,dtype=torch.int64)%((1<<bits)+1))
            packed=tc.pack_unsigned_cuda(codes.cuda(),width,checked=True,max_code=1<<bits)
            expected=pack_unsigned(codes,width)
            unpacked=tc.unpack_unsigned_cuda(packed,width,count,checked=True,max_code=1<<bits)
            yield bits,f'pack_all_legal_count{count}','pack',count,{'code_mismatch_count':mismatch(unpacked.cpu(),codes),
                'packet_byte_mismatch_count':mismatch(packed.cpu(),byte_tensor(expected)),
                'expected_wire_bytes':len(expected),'actual_wire_bytes':packed.numel()},None
        for tag,codes in [('negative',torch.tensor([-1])),('unused',torch.tensor([(1<<bits)+1])),('float_dtype',torch.tensor([1.]))]:
            yield bits,'pack_reject_'+tag,'invalid_input',codes.numel(),{},lambda c=codes,w=width,b=bits:tc.pack_unsigned_cuda(c.cuda(),w,checked=True,max_code=1<<b)
        # Odd count yields padding bits in the last byte; corruption must reject.
        good=byte_tensor(pack_unsigned(torch.tensor([1]),width)).cuda()
        corrupt=good.clone();corrupt[-1]|=128
        for tag,payload,count in [('truncated',good[:-1],1),('overlong',torch.cat((good,torch.zeros(1,dtype=torch.uint8,device='cuda'))),1),('padding_high_bit',corrupt,1)]:
            yield bits,'unpack_reject_'+tag,'invalid_input',count,{},lambda p=payload,w=width,c=count,b=bits:tc.unpack_unsigned_cuda(p,w,c,checked=True,max_code=1<<b)
        # All legal signed codes including b8 endpoints, qscale0 signed zero and positive scales.
        n=((1<<bits)+1+255)//256*256
        signed=torch.arange(n,dtype=torch.int64)%((1<<bits)+1)-bound
        lut=lookup_table(bits)
        for qscale_value in (0,1,255):
            qscale=torch.full((n//16,),qscale_value,dtype=torch.uint8);top=torch.ones(n//256,dtype=torch.bfloat16)
            cp=QuantizedPacket(pack_unsigned(signed+bound,width),bytes(qscale.tolist())+bytes(raw(top).tolist()),n,bits)
            cpu=CpuReference(bits).decode_native(cp)
            packet=tc.DevicePacket(byte_tensor(cp.payload).cuda(),qscale.cuda(),top.cuda(),n,n,bits)
            plan=tc.TorchCodecPlan(n,bits,device='cuda');out=plan.decode_into(packet,lut.cuda(),checked=True)
            yield bits,f'decode_all_legal_scale{qscale_value}','decode',n,{'decoded_bit_mismatch_count':bit_mismatch(out,cpu),
                'expected_wire_bytes':len(cp.wire),'actual_wire_bytes':gpu_wire(packet).numel()},None
        for top_name,top_value in [('minimum_subnormal',torch.finfo(torch.bfloat16).tiny/128),
                ('minimum_normal',torch.finfo(torch.bfloat16).tiny),('epsilon',1e-23),('large_finite',1e30)]:
            for qscale_value in (0,1,127,255):
                qscale=torch.full((n//16,),qscale_value,dtype=torch.uint8)
                top=torch.full((n//256,),top_value,dtype=torch.bfloat16)
                cp=QuantizedPacket(pack_unsigned(signed+bound,width),bytes(qscale.tolist())+bytes(raw(top).tolist()),n,bits)
                cpu=CpuReference(bits).decode_native(cp)
                packet=tc.DevicePacket(byte_tensor(cp.payload).cuda(),qscale.cuda(),top.cuda(),n,n,bits)
                out=tc.TorchCodecPlan(n,bits,device='cuda').decode_into(packet,lut.cuda(),checked=True)
                yield bits,f'decode_legal_top_{top_name}_qscale{qscale_value}','decode',n,{
                    'decoded_bit_mismatch_count':bit_mismatch(out,cpu),
                    'expected_wire_bytes':len(cp.wire),'actual_wire_bytes':gpu_wire(packet).numel()},None
        if float(lut[(1<<bits)+bound])>255:
            top=torch.full((n//256,),torch.finfo(torch.bfloat16).max,dtype=torch.bfloat16)
            qscale=torch.ones(n//16,dtype=torch.uint8)
            cp=QuantizedPacket(pack_unsigned(torch.full((n,),2*bound,dtype=torch.int64),width),bytes(qscale.tolist())+bytes(raw(top).tolist()),n,bits)
            try:CpuReference(bits).decode_native(cp)
            except (ValueError,RuntimeError,FloatingPointError):pass
            else:raise AssertionError('Output overflow probe must also be rejected by CPU reference')
            packet=tc.DevicePacket(byte_tensor(cp.payload).cuda(),qscale.cuda(),top.cuda(),n,n,bits)
            yield bits,'packet_reject_output_overflow_finite_decoded_scale','invalid_input',n,{},lambda v=packet,b=bits,l=lut.cuda():tc.decode(v,l,checked=True)
        n=256;lut=lut.cuda();x=torch.ones(n,dtype=torch.bfloat16,device='cuda');main=torch.zeros((16,16),dtype=torch.bfloat16,device='cuda');scale=torch.zeros((1,16),dtype=torch.bfloat16,device='cuda')
        plan=tc.TorchCodecPlan(n,bits,device='cuda')
        for tag,value in [('nan',float('nan')),('positive_inf',float('inf')),('negative_inf',float('-inf'))]:
            bad=x.clone();bad[0]=value
            yield bits,'encode_reject_'+tag,'invalid_input',n,{},lambda v=bad,p=plan,m=main,s=scale,l=lut:p.encode_into(v,m,s,l,checked=True)
        bad_inputs=[('float32',x.float()),('rank2',x.view(16,16)),('noncontiguous',torch.ones(512,dtype=torch.bfloat16,device='cuda')[::2]),('wrong_length',x[:-1])]
        for tag,bad in bad_inputs:
            yield bits,'encode_reject_'+tag,'invalid_input',n,{},lambda v=bad,p=plan,m=main,s=scale,l=lut:p.encode_into(v,m,s,l,checked=True)
        for tag,value in [('nan',float('nan')),('too_large',131072.),('negative',-1.)]:
            bad=main.clone();bad[0,0]=value
            yield bits,'threshold_reject_'+tag,'invalid_input',n,{},lambda v=bad,p=plan,x=x,s=scale,l=lut:p.encode_into(x,v,s,l,checked=True)
        yield bits,'threshold_reject_shape','invalid_input',n,{},lambda p=plan,x=x,m=main,s=scale,l=lut:p.encode_into(x,m.reshape(-1),s,l,checked=True)
        base=plan.encode_into(x,main,scale,lut,checked=True)
        for tag,value in [('zero',0.),('negative',-1.),('nan',float('nan')),('inf',float('inf')),('finite_intermediate_overflow',1e38)]:
            top=base.top_scales.clone();top.fill_(value)
            packet=tc.DevicePacket(base.payload.clone(),torch.full_like(base.scale_codes,255),top,n,n,bits)
            yield bits,'packet_top_reject_'+tag,'invalid_input',n,{},lambda p=plan,v=packet,l=lut:p.decode_into(v,l,checked=True)
        for tag,packet in [('truncated_payload',tc.DevicePacket(base.payload[:-1],base.scale_codes,base.top_scales,n,n,bits)),
                           ('truncated_scale',tc.DevicePacket(base.payload,base.scale_codes[:-1],base.top_scales,n,n,bits)),
                           ('unused_code',tc.DevicePacket(tc.pack_unsigned_cuda(torch.full((n,),1<<width-1,device='cuda',dtype=torch.int64)+1,width,checked=False),base.scale_codes,base.top_scales,n,n,bits))]:
            yield bits,'packet_reject_'+tag,'invalid_input',n,{},lambda p=plan,v=packet,l=lut:p.decode_into(v,l,checked=True)
        for tag,bad in [('nan',torch.full_like(lut,float('nan'))),('unsorted',lut.flip(0)),('duplicate',torch.zeros_like(lut))]:
            yield bits,'lut_reject_'+tag,'invalid_input',n,{},lambda v=bad,p=plan,x=x,m=main,s=scale:p.encode_into(x,m,s,v,checked=True)
        yield bits,'encode_reject_owned_storage_alias','invalid_input',n,{},lambda p=plan,m=main,s=scale,l=lut:p.encode_into(p.output,m,s,l,checked=True)
        # Finite inputs can cause a later BF16 arithmetic overflow. Retain the
        # reference classification; do not silently cast stages to FP32 to pass.
        max_input=torch.full((n,),torch.finfo(torch.bfloat16).max,dtype=torch.bfloat16)
        cm,cs=main.cpu(),scale.cpu();ref=CpuReference(bits)
        max_packet=ref.encode_native(max_input,cm,cs)
        observed=plan.encode_into(max_input.cuda(),main,scale,lut,checked=True)
        yield bits,'maxfinite_input_encode','encode',n,{
            'packet_byte_mismatch_count':mismatch(gpu_wire(observed),byte_tensor(max_packet.wire)),
            'expected_wire_bytes':len(max_packet.wire),'actual_wire_bytes':observed.wire_bytes,
            'undefined_metric_reason':'encode_only_decode_may_overflow'},None
        try:expected=ref.decode_native(max_packet)
        except (ValueError,RuntimeError,FloatingPointError):
            yield bits,'maxfinite_input_decode_reject_reference_overflow','invalid_input',n,{},lambda p=plan,v=observed,l=lut:p.decode_into(v,l,checked=True)
        else:
            out=plan.decode_into(observed,lut,checked=True)
            yield bits,'maxfinite_input_decode','decode',n,{'decoded_bit_mismatch_count':bit_mismatch(out,expected),
                **distort(max_input,out,observed.wire_bytes)},None


METHODS=('torch_cuda_repaired','cuda_optimized_repaired')
RULES_PATH=ROOT/'configs/g1b_correctness_cases.json'
FIXTURE_CHECKS=('roundtrip','encode','decode','repeatability','input_immutability')


def method_adapter(method):
    from types import SimpleNamespace
    if method=='torch_cuda_repaired':
        from gradient_comm_lab import torch_codec as module
        return module
    from gradient_comm_lab import cuda_codec as module
    result=SimpleNamespace(**{name:getattr(module,name) for name in ('DevicePacket','pack_unsigned_cuda','unpack_unsigned_cuda','encode','decode','roundtrip')})
    result.TorchCodecPlan=module.CudaCodecPlan
    if hasattr(module,'bf16_add_epsilon'):result.bf16_add_epsilon=module.bf16_add_epsilon
    result.load_extension=module.load_extension
    return result


def exact_rejection(operation, exception_type, message):
    """CUDA launch/OOM/illegal-memory errors never count as data rejection."""
    try:
        operation()
    except Exception as exc:
        full=str(exc)
        primary=full.split('\nException raised from ',1)[0] if exception_type=='RuntimeError' and message.startswith(('ABI_METADATA: ','ABI_BITS: ','ABI_ALIAS: ')) else full
        actual={'exception_type':type(exc).__name__,'message':full,'primary_message':primary,'matching':'exactprimaryhostABIwithpreservedC++diagnostics' if primary!=full else 'exactfullmessage'}
        if type(exc).__name__!=exception_type or primary!=message:
            raise AssertionError(f'Expected {exception_type}: {message}; observed {actual}') from exc
        # Any latent CUDA failure is surfaced separately and cannot be accepted.
        torch.cuda.current_stream().synchronize()
        return actual
    raise AssertionError(f'Expected {exception_type}: {message}; no exception raised')


def nondefault_stream_check(tc,bits):
    """Producer -> consumer -> result events, with no whole-device synchronize.

    Delayed producer overwrite makes an accidental default-stream launch read
    stale zeros. The only host wait is the consumer's result event.
    """
    n=257;x=torch.linspace(-2,2,n).to(torch.bfloat16)
    main,scale,_=threshold_fixture(f'g1b_stream_b{bits}',512)
    expected=CpuReference(bits).roundtrip(x,main,scale)
    gx=torch.zeros(n,dtype=torch.bfloat16,device='cuda')
    source=x.cuda();gm=main.cuda();gs=scale.cuda();lut=lookup_table(bits).cuda()
    plan=tc.TorchCodecPlan(n,bits,device='cuda')
    initial=torch.cuda.Event();initial.record()
    producer=torch.cuda.Stream();consumer=torch.cuda.Stream()
    ready=torch.cuda.Event();done=torch.cuda.Event()
    old_sync=torch.cuda.synchronize
    def forbidden(*args,**kwargs):raise AssertionError('Whole-device sync forbidden in nondefault stream check')
    torch.cuda.synchronize=forbidden
    try:
        with torch.cuda.stream(producer):
            producer.wait_event(initial)
            torch.cuda._sleep(50_000_000)
            gx.copy_(source)
            ready.record()
        with torch.cuda.stream(consumer):
            consumer.wait_event(ready)
            out=plan.roundtrip_into(gx,gm,gs,lut,checked=False).clone()
            wire=plan.packet.serialized().clone()
            done.record()
        done.synchronize()
        observed=out.cpu();observed_wire=wire.cpu()
    finally:torch.cuda.synchronize=old_sync
    return {'decoded_bit_mismatch_count':bit_mismatch(observed,expected['decoded']),
        'packet_byte_mismatch_count':mismatch(observed_wire,byte_tensor(expected['wire'])),
        'expected_wire_bytes':len(expected['wire']),'actual_wire_bytes':observed_wire.numel(),
        'producer_stream_id':producer.cuda_stream,'consumer_stream_id':consumer.cuda_stream,
        'whole_device_synchronization_used':False,'dependency':'producer event -> consumer wait -> codec -> result event host wait'}


def additional_checks(tc,bits):
    n=256;plan=tc.TorchCodecPlan(n,bits);lut=lookup_table(bits).cuda()
    x=torch.ones(n,dtype=torch.bfloat16,device='cuda')
    main=torch.zeros((16,16),dtype=torch.bfloat16,device='cuda')
    scale=torch.zeros((1,16),dtype=torch.bfloat16,device='cuda')
    packet=plan.encode_into(x,main,scale,lut,checked=True)
    # ABI-facing metadata and storage checks apply even when value checks are disabled.
    for identity,op,message in (
        ('abi_input_cpu',lambda:plan.encode_into(x.cpu(),main,scale,lut,checked=False),'input: expected contiguous tensor on cuda:0'),
        ('abi_lut_short',lambda:plan.encode_into(x,main,scale,lut[:-1],checked=False),'LUT: unexpected tensor dtype or shape'),
        ('abi_threshold_dtype',lambda:plan.encode_into(x,main.float(),scale,lut,checked=False),'main thresholds: unexpected tensor dtype or shape'),
        ('abi_packet_wrong_identity',lambda:plan.decode_into(replace(packet,bits=12-bits),lut,checked=False),'Packet identity differs from plan'),
        ('abi_packet_wrong_type',lambda:plan.decode_into(None,lut,checked=False),'Expected a DevicePacket'),
        ('abi_packet_output_alias',lambda:plan.decode_into(replace(packet,payload=plan.decoded_padded.view(torch.uint8)[:packet.payload.numel()]),lut,checked=True),'packet payload: aliases another plan-owned buffer'),
        ('abi_scale_threshold_nan',lambda:plan.encode_into(x,main,torch.full_like(scale,float('nan')),lut,checked=True),'Invalid explicit thresholds'),
    ):
        yield identity,'invalid_input',n,{},op,{'exception_type':'ValueError','message':message}
    yield 'nondefault_stream_dependency','roundtrip',257,nondefault_stream_check(tc,bits),None,None




def stream_lifetime_check(tc,bits,allocating):
    import gc
    n=65536;x=torch.linspace(-2,2,n).to(torch.bfloat16)
    main,scale,_=threshold_fixture(f'g1b_lifetime_{bits}',n)
    expected=CpuReference(bits).roundtrip(x,main,scale)
    producer=torch.cuda.Stream();consumer=torch.cuda.Stream();ready=torch.cuda.Event();done=torch.cuda.Event()
    with torch.cuda.stream(producer):
        gx=x.cuda();gm=main.cuda();gs=scale.cuda();lut=lookup_table(bits).cuda()
        plan=None if allocating else tc.TorchCodecPlan(n,bits)
        ready.record()
    input_ptrs={t.untyped_storage().data_ptr() for t in (gx,gm,gs,lut)}
    old_sync=torch.cuda.synchronize
    def forbidden(*a,**k):raise AssertionError('Whole-device synchronization forbidden')
    torch.cuda.synchronize=forbidden
    try:
        with torch.cuda.stream(consumer):
            consumer.wait_event(ready);torch.cuda._sleep(500_000_000)
            if allocating:packet,out=tc.roundtrip(gx,bits,gm,gs,lut,checked=False)
            else:out=plan.roundtrip_into(gx,gm,gs,lut,checked=False);packet=plan.packet
            # Clone return values so even original packet/output owners can be released.
            retained_output=out.clone();retained_wire=packet.serialized().clone();done.record()
        del gx,gm,gs,lut,plan,packet,out
        gc.collect()
        pending_before_churn=not done.query()
        churn=[];reused=set();unsafe_reuse=[];churn_observations=[]
        with torch.cuda.stream(producer):
            for size in (n,n//16,2*(1<<bits)+1):
                for repeat in range(8):
                    t=torch.full((size,),17,dtype=torch.bfloat16,device='cuda');churn.append(t)
                    ptr=t.untyped_storage().data_ptr();pending=not done.query()
                    matched=ptr in input_ptrs
                    churn_observations.append({'size':size,'pending_after_allocation':pending,'reused_input_pointer':matched})
                    if matched:
                        reused.add(ptr)
                        if pending:unsafe_reuse.append(ptr)
        done.synchronize()
        observed=retained_output.cpu();wire=retained_wire.cpu()
    finally:torch.cuda.synchronize=old_sync
    if not pending_before_churn:raise AssertionError('Lifetime probe completed before allocator churn; coverage not established')
    if unsafe_reuse:raise AssertionError('Input storage reused while result event still pending')
    return {'decoded_bit_mismatch_count':bit_mismatch(observed,expected['decoded']),
        'packet_byte_mismatch_count':mismatch(wire,byte_tensor(expected['wire'])),
        'pending_before_allocator_churn':pending_before_churn,'pending_input_storage_reused_count':len(unsafe_reuse),'completed_input_storage_reused_count':len(reused),'allocator_churn':churn_observations,
        'released_plan_and_inputs':True,'allocation_mode':'api_allocating' if allocating else 'preallocated',
        'whole_device_synchronization_used':False}


NATIVE_REJECTIONS={
 'native_input_cpu':'ABI_METADATA: input',
 'native_encode_bits':'ABI_BITS: expected nominal b4/b8',
 'native_encode_short_payload':'ABI_METADATA: payload',
 'native_encode_short_lut':'ABI_METADATA: LUT',
 'native_encode_status_dtype':'ABI_METADATA: status',
 'native_encode_status_length':'ABI_METADATA: status',
 'native_encode_payload_input_alias':'ABI_ALIAS: output overlaps input storage',
 'native_encode_output_alias':'ABI_ALIAS: outputs share storage',
 'native_decode_short_output':'ABI_METADATA: output',
 'native_pack_width':'ABI_BITS: expected stored 5/9 bits',
 'native_unpack_short_payload':'ABI_METADATA: payload',
}


def native_abi_checks(tc,bits):
    ext=tc.load_extension();n=256;plan=tc.TorchCodecPlan(n,bits)
    x=torch.ones(n,dtype=torch.bfloat16,device='cuda');m=torch.zeros((16,16),dtype=torch.bfloat16,device='cuda')
    sc=torch.zeros((1,16),dtype=torch.bfloat16,device='cuda');lut=lookup_table(bits).cuda();p=plan.packet
    st=plan.status
    def encode(**kw):
        d=dict(x=x,main=m,scale=sc,lut=lut,payload=p.payload,scales=p.scale_codes,top=p.top_scales,bits=bits,status=st)
        d.update(kw)
        return ext.encode_into(d['x'],d['main'],d['scale'],d['lut'],d['payload'],d['scales'],d['top'],d['bits'],d['status'],False)
    codes=torch.zeros(n,dtype=torch.int64,device='cuda')
    operations={
      'native_input_cpu':lambda:encode(x=x.cpu()),
      'native_encode_bits':lambda:encode(bits=6),
      'native_encode_short_payload':lambda:encode(payload=p.payload[:-1]),
      'native_encode_short_lut':lambda:encode(lut=lut[:-1]),
      'native_encode_status_dtype':lambda:encode(status=st.to(torch.int64)),
      'native_encode_status_length':lambda:encode(status=st[:0]),
      'native_encode_payload_input_alias':lambda:encode(payload=x.view(torch.uint8)[:p.payload.numel()]),
      'native_encode_output_alias':lambda:encode(scales=p.payload[:p.scale_codes.numel()]),
      'native_decode_short_output':lambda:ext.decode_into(p.payload,p.scale_codes,p.top_scales,lut,plan.decoded_padded[:-1],n,bits,st,False),
      'native_pack_width':lambda:ext.pack_into(codes,p.payload,6),
      'native_unpack_short_payload':lambda:ext.unpack_into(p.payload[:-1],codes,bits+1),
    }
    for identity,op in operations.items():
        yield identity,'invalid_input',n,{},op,{'exception_type':'RuntimeError','message':NATIVE_REJECTIONS[identity]}
    for allocating in (False,True):
        yield 'native_stream_lifetime_'+('allocating' if allocating else 'preallocated'),'roundtrip',65536,stream_lifetime_check(tc,bits,allocating),None,None
    # Canary guards around every native write buffer, including padded tail output.
    n=257;padded=512
    x=torch.linspace(-1,1,n).to(torch.bfloat16);m,sc,_=threshold_fixture('native_canary',padded)
    expected=CpuReference(bits).roundtrip(x,m,sc)
    def guarded(count,dtype):
        parent=torch.full((count+2,),37,dtype=dtype,device='cuda');return parent,parent[1:-1]
    pb,payload=guarded(padded*(bits+1)//8,torch.uint8)
    sb,scales=guarded(padded//16,torch.uint8);tb,top=guarded(padded//256,torch.bfloat16)
    ob,out=guarded(padded,torch.bfloat16);status=torch.zeros(1,device='cuda',dtype=torch.int32)
    ext.encode_into(x.cuda(),m.cuda(),sc.cuda(),lut,payload,scales,top,bits,status,False)
    ext.decode_into(payload,scales,top,lut,out,n,bits,status,False)
    torch.cuda.current_stream().synchronize()
    guard_errors=sum(int((parent[[0,-1]]!=37).sum()) for parent in (pb,sb,tb,ob))
    observed=torch.cat((payload,scales,top.view(torch.uint8))).cpu()
    yield 'native_tail_canary','roundtrip',n,{'packet_byte_mismatch_count':mismatch(observed,byte_tensor(expected['wire'])),
        'decoded_bit_mismatch_count':bit_mismatch(out[:n],expected['decoded']),
        'guard_mismatch_count':guard_errors,'padded_numel':padded},None,None


def expected_coverage(manifest, methods, limit, rules):
    result=[]
    for fixture in (manifest['inputs'][:limit] if limit else manifest['inputs']):
        for bits in (4,8):
            for method in methods:
                for check in FIXTURE_CHECKS:result.append((f"{fixture['input_id']}_b{bits}_{check}",method))
            result.append((f"{fixture['input_id']}_b{bits}_roundtrip",'cpu_ref_repaired'))
    if not limit:
        for method in methods:
            for case in rules['protocol_cases']:
                result.append((f"{case['input_id']}_b{case['nominal_bits']}_{case['check_type']}",method))
            for bits in (4,8):
                for name in ('abi_input_cpu','abi_lut_short','abi_threshold_dtype','abi_packet_wrong_identity',
                        'abi_packet_wrong_type','abi_packet_output_alias','abi_scale_threshold_nan'):
                    result.append((f'{name}_b{bits}_invalid_input',method))
                result.append((f'nondefault_stream_dependency_b{bits}_roundtrip',method))
                if method=='cuda_optimized_repaired':
                    result.extend((f'{identity}_b{bits}_invalid_input',method) for identity in NATIVE_REJECTIONS)
                    result.append((f'native_tail_canary_b{bits}_roundtrip',method))
                    result.extend((f'native_stream_lifetime_{mode}_b{bits}_roundtrip',method) for mode in ('allocating','preallocated'))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True);parser.add_argument('--output',default='correctness.jsonl')
    parser.add_argument('--limit',type=int,default=0);parser.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    parser.add_argument('--manifest-only',action='store_true');parser.add_argument('--contract-sha256')
    args=parser.parse_args();run=(ROOT/args.run_dir).resolve()
    if (run/'artifact_integrity.json').exists():raise ValueError('Sealed run cannot be changed')
    output=run/args.output
    if not output.is_relative_to(run):raise ValueError('Output must remain in run')
    output.parent.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((run/'input_manifest.json').read_text(encoding='utf-8'))
    rules=json.loads(RULES_PATH.read_text(encoding='utf-8'))
    expected=expected_coverage(manifest,args.methods,args.limit,rules)
    coverage={'run_id':run.name,'expected_cases':[dict(case_id=c,method_id=m) for c,m in expected],
        'fixture_inputs':len(manifest['inputs']),'inherited_protocol_case_count_per_method':len(rules['protocol_cases']),
        'rules_path':RULES_PATH.relative_to(ROOT).as_posix(),'rules_sha256':sha(RULES_PATH),
        'rule_origin':'002 observed concrete ValueError classes/messages frozen before G1b execution; numeric expectations unchanged',
        'new_checks':'7 Python ABI rejections +1stream perwidth/method;11exact native ABI metadata/alias rejections +1tailcanary +2crossstream lifetime perwidth optimized',
        'native_rejection_rules':{key:dict(exception_type='RuntimeError',message=value) for key,value in NATIVE_REJECTIONS.items()}}
    if args.manifest_only:
        (run/'correctness_case_manifest.json').write_text(json.dumps(coverage,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({'case_count':len(expected),'fixture_inputs':len(manifest['inputs'])}));return 0
    formal=args.output=='correctness.jsonl' and not args.limit and tuple(args.methods)==METHODS
    contract=args.contract_sha256 or sha(run/'codec_contract.json')
    if formal and args.contract_sha256:raise ValueError('Formal qualification uses frozen contract only')
    if formal:
        registration=json.loads((run/'execution_registration.json').read_text(encoding='utf-8'))
        assert registration['run_id']==run.name and registration['contract_sha256']==contract
        assert registration['input_manifest_sha256']==sha(run/'input_manifest.json')
        for path,digest in registration['source_hashes'].items():
            if sha(ROOT/path)!=digest:raise ValueError('Registered source changed: '+path)
        frozen_coverage=json.loads((run/'correctness_case_manifest.json').read_text(encoding='utf-8'))
        if frozen_coverage!=coverage:raise ValueError('Frozen correctness coverage differs')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.utils.deterministic.fill_uninitialized_memory=False
    attempt_sources=run/'logs'/(output.stem+'_sources');attempt_sources.mkdir(exist_ok=False)
    import shutil
    for source in (Path(__file__),ROOT/'tests/test_cuda_codec.py',RULES_PATH,ROOT/'src/gradient_comm_lab/cuda_codec.py'):
        shutil.copyfile(source,attempt_sources/source.name)
    build_identity=None
    if 'cuda_optimized_repaired' in args.methods:
        from gradient_comm_lab import cuda_codec
        extension=cuda_codec.load_extension()
        build_identity={'module_name':extension.__name__,'module_path':extension.__file__,'module_sha256':sha(extension.__file__),'manifest':json.loads((run/'logs/cuda_build_manifest.json').read_text())}
    start_identity={'sources':{p.name:sha(p) for p in attempt_sources.iterdir()},'loaded_build':build_identity}
    (run/'logs'/(output.stem+'_start_identity.json')).write_text(json.dumps(start_identity,indent=2)+'\n',encoding='utf-8')
    from collections import Counter,defaultdict
    counts=Counter();by_method=defaultdict(Counter);seen=[];started=datetime.now().astimezone().isoformat()
    def emit(stream,row):
        counts[row['status']]+=1;by_method[row['method_id']][row['status']]+=1
        seen.append((row['case_id'],row['method_id']))
        stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
    def failed_record(method,identity,bits,n,check,exc):
        row=base_record(run,contract,identity,bits,n,check,method)
        folder=run/'counterexamples';folder.mkdir(exist_ok=True)
        p=folder/f'{identity}_b{bits}_{method}_{output.stem}.json'
        p.write_text(json.dumps({'exception_type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()},indent=2),encoding='utf-8')
        row.update(status='fail',failure_reason=type(exc).__name__+': '+str(exc),counterexample_path=p.relative_to(run).as_posix())
        return row
    rejected={(r['nominal_bits'],r['input_id']):r for r in rules['rejection_rules']}
    with output.open('x',encoding='utf-8') as stream:
        for method_index,method in enumerate(args.methods):
            tc=method_adapter(method)
            for fixture in (manifest['inputs'][:args.limit] if args.limit else manifest['inputs']):
                for bits in (4,8):
                    try:
                        result=qualify_fixture(fixture,bits,tc)
                        counter=result.pop('counterexample',None);counterpath=None
                        if counter:
                            folder=run/'counterexamples';folder.mkdir(exist_ok=True)
                            p=folder/f"{fixture['input_id']}_b{bits}_{method}_{output.stem}.pt"
                            torch.save(counter,p);counterpath=p.relative_to(run).as_posix()
                        for check in FIXTURE_CHECKS:
                            row=base_record(run,contract,fixture['input_id'],bits,fixture['original_numel'],check,method)
                            row.update(result['counts']);row.update(result['quant']);row['quantization_events']=result['quantization_events']
                            value={'encode':sum(result['counts'][k] for k in ('code_mismatch_count','scale_code_mismatch_count','top_scale_bit_mismatch_count','packet_byte_mismatch_count')),
                                'decode':result['decode_count'],'roundtrip':result['roundtrip_count']+result['counts']['packet_byte_mismatch_count'],
                                'repeatability':result['repeat_count'],'input_immutability':result['changed']}[check]
                            row['decoded_bit_mismatch_count']=result['decode_count'] if check=='decode' else result['roundtrip_count']
                            row['mismatch_or_mutation_count']=value
                            if value:row.update(status='fail',failure_reason='Exact fixed-contract mismatch',counterexample_path=counterpath)
                            emit(stream,row)
                        if method_index==0:
                            row=base_record(run,contract,fixture['input_id'],bits,fixture['original_numel'],'roundtrip','cpu_ref_repaired')
                            row.update(result['CPU_quant'],expected_wire_bytes=result['counts']['expected_wire_bytes'],actual_wire_bytes=result['counts']['expected_wire_bytes'],
                                reference_qualification='CPU execution record, not independent GPU experiment')
                            emit(stream,row)
                    except Exception as exc:emit(stream,failed_record(method,fixture['input_id'],bits,fixture['original_numel'],'roundtrip',exc))
                    print(json.dumps({'method':method,'input':fixture['input_id'],'bits':bits,'failures':counts['fail']}),flush=True)
            if not args.limit:
                try:
                    for bits,identity,check,n,metrics,operation in protocol_checks(tc):
                        row=base_record(run,contract,identity,bits,n,check,method);row.update(metrics)
                        row['undefined_metric_reason']=row['undefined_metric_reason'] or 'protocol_or_arithmetic_check_has_no_original_quantization_input'
                        try:
                            if operation:
                                rule=rejected[(bits,identity)]
                                row['expected_rejection_rule']={k:rule[k] for k in ('exception_type','message')}
                                row['observed_rejection']=exact_rejection(operation,rule['exception_type'],rule['message'])
                            elif any(v for k,v in metrics.items() if k.endswith('mismatch_count')):raise AssertionError('Exact protocol mismatch')
                        except Exception as exc:row.update(failed_record(method,identity,bits,n,check,exc))
                        emit(stream,row)
                except Exception as exc:emit(stream,failed_record(method,'protocol_generator_failure',0,0,'roundtrip',exc))
                for bits in (4,8):
                    try:
                        from itertools import chain
                        checks=chain(additional_checks(tc,bits),native_abi_checks(tc,bits)) if method=='cuda_optimized_repaired' else additional_checks(tc,bits)
                        for identity,check,n,metrics,op,rule in checks:
                            row=base_record(run,contract,identity,bits,n,check,method);row.update(metrics)
                            try:
                                if op:
                                    row['expected_rejection_rule']=rule;row['observed_rejection']=exact_rejection(op,rule['exception_type'],rule['message'])
                                elif any(v for k,v in metrics.items() if k.endswith('mismatch_count')):raise AssertionError('Stream result mismatch')
                            except Exception as exc:row.update(failed_record(method,identity,bits,n,check,exc))
                            emit(stream,row)
                    except Exception as exc:emit(stream,failed_record(method,'additional_generator_failure',bits,0,'roundtrip',exc))
    missing=sorted(set(expected)-set(seen));extra=sorted(set(seen)-set(expected));duplicates=[k for k,v in Counter(seen).items() if v!=1]
    summary={'run_id':run.name,'contract_sha256':contract,'started_at':started,'finished_at':datetime.now().astimezone().isoformat(),
        'start_identity':start_identity,'counts':dict(counts),'method_statuses':{m:{'correctness_status':'pass' if not by_method[m]['fail'] and not missing and not extra and not duplicates and not args.limit else 'incomplete' if args.limit else 'fail','counts':dict(by_method[m])} for m in by_method},
        'correctness_status':'pass' if not counts['fail'] and not missing and not extra and not duplicates and not args.limit else 'incomplete' if args.limit else 'fail',
        'missing_coverage':missing,'unexpected_coverage':extra,'duplicate_coverage':duplicates,'expected_record_count':len(expected),
        'fixture_inputs':len(manifest['inputs']),'source_hashes':{str(Path(__file__).relative_to(ROOT)):sha(__file__),'configs/g1b_correctness_cases.json':sha(RULES_PATH)},
        'correctness_sha256':sha(output),'output':output.relative_to(run).as_posix(),'execution_registration_verified':formal,
        'deterministic_algorithms':True,'fill_uninitialized_memory':False,'scope':'local codec fixed configuration b4/b8 only'}
    summary['method_status']=summary['correctness_status']
    output.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    if formal:
        with (run/'correctness_acceptance.json').open('x',encoding='utf-8') as f:json.dump(summary,f,indent=2,allow_nan=False)
    print(json.dumps(summary));return 2 if counts['fail'] or missing or extra or duplicates else 0

if __name__=='__main__':sys.exit(main())

