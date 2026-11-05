"""Record every CPU/GPU comparison without relaxing fixed BF16 semantics."""
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
            'cpu_audit':expected['audit'],'gpu_signed_codes':plan.signed_codes.cpu().clone(),
            'gpu_scales':plan.scales.cpu().clone(),'gpu_decoded_scales':plan.decoded_scales.cpu().clone(),
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
        observed_add=finite.cuda()+probe_plan.epsilon
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
                'code_mismatch_count':mismatch(plan.signed_codes.cpu(),cp['audit']['codes']),
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

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',required=True)
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--output',default='correctness.jsonl')
    parser.add_argument('--contract-sha256');parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args();run=(ROOT/args.run_dir).resolve();torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory=False
    formal=not args.prepare_only and args.output=='correctness.jsonl' and not args.limit
    if formal:
        registration=json.loads((run/'execution_registration.json').read_text(encoding='utf-8'))
        if registration['run_id']!=run.name:raise ValueError('Registered run identity differs')
        if sha(run/'codec_contract.json')!=registration['contract_sha256']:raise ValueError('Registered contract changed')
        if args.contract_sha256 and args.contract_sha256!=registration['contract_sha256']:raise ValueError('Requested contract differs from registration')
        if sha(run/'input_manifest.json')!=registration['input_manifest_sha256']:raise ValueError('Registered input manifest changed')
        for path,digest in registration['source_hashes'].items():
            if sha(ROOT/path)!=digest:raise ValueError('Registered source changed: '+path)
    manifest=prepare_fixtures(run)
    if args.prepare_only:
        print(json.dumps({'inputs':len(manifest['inputs']),'counts':manifest['counts_by_source_kind']}));return 0
    from gradient_comm_lab import torch_codec as tc
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required')
    contract=args.contract_sha256 or sha(run/'codec_contract.json')
    output=run/args.output
    if not output.is_relative_to(run):raise ValueError('Output must remain in new run')
    output.parent.mkdir(parents=True,exist_ok=True)
    failed=0;counts={'pass':0,'fail':0,'skipped':0};started=datetime.now().astimezone().isoformat()
    def emit(stream,row):
        counts[row['status']]+=1;stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
    with output.open('x',encoding='utf-8') as stream:
        rows=manifest['inputs'][:args.limit] if args.limit else manifest['inputs']
        for fixture in rows:
            for bits in (4,8):
                try:
                    result=qualify_fixture(fixture,bits,tc)
                    counter=result.pop('counterexample',None);counterpath=None
                    if counter:
                        folder=run/'counterexamples';folder.mkdir(exist_ok=True)
                        counterfile=folder/f"{fixture['input_id']}_b{bits}_{output.stem}.pt";torch.save(counter,counterfile);counterpath=counterfile.relative_to(run).as_posix()
                    for check in ('roundtrip','encode','decode','repeatability','input_immutability'):
                        row=base_record(run,contract,fixture['input_id'],bits,fixture['original_numel'],check)
                        row.update(result['counts']);row.update(result['quant'])
                        row['quantization_events']=result['quantization_events']
                        value={'encode':sum(result['counts'][k] for k in ('code_mismatch_count','scale_code_mismatch_count','top_scale_bit_mismatch_count','packet_byte_mismatch_count')),
                            'decode':result['decode_count'],'roundtrip':result['roundtrip_count']+result['counts']['packet_byte_mismatch_count'],
                            'repeatability':result['repeat_count'],'input_immutability':result['changed']}[check]
                        row['decoded_bit_mismatch_count']=result['decode_count'] if check=='decode' else result['roundtrip_count']
                        row['mismatch_or_mutation_count']=value
                        if value:row.update(status='fail',failure_reason='Exact fixed-contract mismatch',counterexample_path=counterpath);failed+=1
                        emit(stream,row)
                    row=base_record(run,contract,fixture['input_id'],bits,fixture['original_numel'],'roundtrip','cpu_ref_repaired')
                    row.update(result['CPU_quant'],expected_wire_bytes=result['counts']['expected_wire_bytes'],actual_wire_bytes=result['counts']['expected_wire_bytes'],
                        reference_qualification='finite actual CPU encode/decode; not an independent implementation parity claim')
                    emit(stream,row)
                except Exception as exc:
                    row=base_record(run,contract,fixture['input_id'],bits,fixture['original_numel'],'roundtrip')
                    folder=run/'counterexamples';folder.mkdir(exist_ok=True)
                    p=folder/f"{fixture['input_id']}_b{bits}_{output.stem}.json"
                    p.write_text(json.dumps({'fixture':fixture,'exception':str(exc),'traceback':traceback.format_exc()},indent=2),encoding='utf-8')
                    row.update(status='fail',failure_reason=type(exc).__name__+': '+str(exc),counterexample_path=p.relative_to(run).as_posix());emit(stream,row);failed+=1
                print(json.dumps({'input':fixture['input_id'],'bits':bits,'failures_so_far':failed}),flush=True)
        if not args.limit:
            for bits,identity,check,length,metrics,operation in protocol_checks(tc):
                row=base_record(run,contract,identity,bits,length,check);row.update(metrics)
                if row['undefined_metric_reason'] is None and check in ('pack','decode','encode','invalid_input'):
                    row['undefined_metric_reason']='protocol_or_arithmetic_check_has_no_original_quantization_input'
                if operation:
                    try:operation();torch.cuda.synchronize();row.update(status='fail',failure_reason='Expected checked rejection was not raised');failed+=1
                    except (ValueError,TypeError,RuntimeError,FloatingPointError) as exc:row['expected_rejection']=type(exc).__name__+': '+str(exc)
                elif any(value for key,value in metrics.items() if key.endswith('mismatch_count')):
                    row.update(status='fail',failure_reason='Exact pack/decode mismatch');failed+=1
                emit(stream,row)
    summary={'run_id':run.name,'started_at':started,'finished_at':datetime.now().astimezone().isoformat(),
        'contract_sha256':contract,'output':output.relative_to(run).as_posix(),'counts':counts,'fixture_inputs':len(rows),
        'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
        'fill_uninitialized_memory':torch.utils.deterministic.fill_uninitialized_memory,
        'execution_registration_verified':formal,
        'method_status':'pass' if not failed and not args.limit else 'fail' if failed else 'partial_smoke_pass',
        'scope':'cpu_reference_and_torch_cuda_repaired_only_optimized_implementation_absent',
        'source_hashes':{str(ROOT/'src/gradient_comm_lab/torch_codec.py'):sha(ROOT/'src/gradient_comm_lab/torch_codec.py'),
            str(Path(__file__).resolve()):sha(__file__),str(ROOT/'src/gradient_comm_lab/fixtures_v2.py'):sha(ROOT/'src/gradient_comm_lab/fixtures_v2.py')},
        'correctness_sha256':sha(output)}
    with output.with_suffix('.summary.json').open('x',encoding='utf-8') as f:json.dump(summary,f,indent=2,ensure_ascii=False,allow_nan=False)
    if args.output=='correctness.jsonl' and not args.limit:
        with (run/'correctness_acceptance.json').open('x',encoding='utf-8') as f:json.dump(summary,f,indent=2,ensure_ascii=False,allow_nan=False)
    print(json.dumps(summary));return 2 if failed else 0

if __name__=='__main__':sys.exit(main())
