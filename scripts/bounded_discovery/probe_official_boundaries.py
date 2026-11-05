"""Small D0 diagnostic probes; observed extreme behavior is not altered."""
import hashlib, json, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'results/bounded_discovery_v1/bounded_discovery_v1_20261104_001'
sys.path.insert(0,str(ROOT/'src'))
from gradient_comm_lab.bounded_discovery import official_codec as codec
rows=[]
for bits in (4,8):
    for case_id,scale in [('large_1e30',1e30),('near_bf16_max',float(torch.finfo(torch.bfloat16).max)),('small_1e35',1e-35)]:
        value=torch.full((256,),scale,dtype=torch.bfloat16,device='cuda');value[1::2].neg_()
        before=value.clone();packet=codec.encode(value,bits,seed=1104);out=codec.decode(packet);torch.cuda.synchronize()
        row=dict(case_id=case_id,bits=bits,input_finite=bool(torch.isfinite(value).all()),output_nonfinite_count=int((~torch.isfinite(out)).sum()),
                 input_unchanged=bool(torch.equal(value,before)),packet_sha256=hashlib.sha256(packet.to_bytes()).hexdigest(),
                 output_max_abs_finite=float(out[torch.isfinite(out)].abs().max()) if bool(torch.isfinite(out).any()) else None,
                 evidence=f'd0_cases/{case_id}_b{bits}.pt',classification='observed_domain_boundary_not_a_new_discovery')
        torch.save(dict(input=value.cpu(),packet=packet.data.cpu(),decoded=out.cpu()),RUN/row['evidence']);rows.append(row)
    with torch.cuda.stream(torch.cuda.Stream()) as stream_context:
        value=torch.arange(257,dtype=torch.float32,device='cuda').sin().to(torch.bfloat16)
        packet=codec.encode(value,bits,seed=1105);out=codec.decode(packet)
        completed=torch.cuda.Event();completed.record()
    completed.synchronize()
    reference_packet=codec.encode(value,bits,seed=1105);reference=codec.decode(reference_packet);torch.cuda.synchronize()
    rows.append(dict(case_id='explicit_current_nondefault_stream',bits=bits,passed=bool(torch.equal(packet.data,reference_packet.data) and torch.equal(out,reference))))
for value in (float('inf'),float('nan')):
    source=torch.full((256,),value,dtype=torch.bfloat16,device='cuda')
    try:codec.encode(source,4,seed=1104)
    except ValueError as error:rows.append(dict(case_id='wrapper_nonfinite_rejection',value='nan' if value!=value else 'inf',passed=True,message=str(error),scope='Wrapper precondition, not a claimed native check'))
    else:raise AssertionError('Wrapper failed to reject nonfinite input')
(RUN/'logs/d0_boundary_probes.json').write_text(json.dumps(dict(module=codec.build_identity(),rows=rows),indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps(rows,allow_nan=False))
