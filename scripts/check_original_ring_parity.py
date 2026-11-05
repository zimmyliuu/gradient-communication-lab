"""Cross-check the new CPU oracle against the unchanged original _stream_ring."""
import argparse,sys,json,hashlib,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from gradient_comm_lab.cpu_reference import CpuReference,DynamiQAggregator,tensor_identity
from gradient_comm_lab.ring_fixtures import FrozenThresholdProvider,load_case,sha
from gradient_comm_lab.ring_reference import run_ring
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',required=True);a=ap.parse_args();run=(ROOT/a.run_dir).resolve()
    cases=json.loads((run/'input_manifest.json').read_text())['cases'];results=[]
    selected=[c for c in cases if (c['world_size'] in (2,4) and c['entry_mode']=='predivided_bf16' and c['logical_numel']==1025 and c['pattern'] in ('signed','cancellation','zeros','finite_extremes')) or c['source_kind']=='real_gradient']
    for c in selected:
        xs=load_case(c,run);b=c['bits'];n=c['world_size'];p=FrozenThresholdProvider(run);ref=CpuReference(b)
        original=object.__new__(DynamiQAggregator);original.n=n;original.device=torch.device('cpu')
        counts=[0]*n;packets={};sends=[]
        def encode(_self,value,bits,rank,chunk):
            ordinal=counts[rank];counts[rank]+=1
            event=dict(case_id=c['case_id'],nominal_bits=bits,phase='sink' if ordinal==n-1 else 'reduce_scatter',rank=rank,chunk=chunk,encoding_ordinal=ordinal)
            t=p(event);packet=ref.encode_native(value,t['main_thresholds'],t['scale_thresholds'])
            packets[t['event_key']]=dict(input=tensor_identity(value),wire_sha256=hashlib.sha256(packet.wire).hexdigest())
            return packet
        def decode(_self,packet):return ref.decode_native(packet)
        def record(_self,ledger,packet,sender,phase,hop,chunk,logical,step,bucket):
            sends.append(dict(sender=sender,phase=phase,hop=hop,chunk=chunk,wire_sha256=hashlib.sha256(packet.wire).hexdigest(),bytes=len(packet.wire)))
        original.encode=types.MethodType(encode,original);original.decode=types.MethodType(decode,original);original._record=types.MethodType(record,original)
        old=original._stream_ring(xs,b,None,0,0)
        new=run_ring(xs,b,c['entry_mode'],FrozenThresholdProvider(run),case_id=c['case_id'])
        for oldout,newout in zip(old,new['outputs']):assert torch.equal(oldout.view(torch.int16),newout.view(torch.int16)),c['case_id']
        for e in new['events']:
            wire=b''.join(e['packet'][k].contiguous().view(torch.uint8).numpy().tobytes() for k in ('payload','scale_codes','top_scales'))
            assert packets[e['event_id']]==dict(input=tensor_identity(e['input']),wire_sha256=hashlib.sha256(wire).hexdigest())
        assert len(sends)==2*n*(n-1) and sum(counts)==n*n
        results.append(dict(case_id=c['case_id'],status='pass',encoding_events=len(packets),sends=len(sends),outputs=[tensor_identity(v) for v in old],events=packets))
    log=dict(status='pass',kind='direct_original_private_stream_ring_CPU_cross_check',cases=len(results),
        source_path=str(ROOT.parent/'spl_model_communication/src/splcomm/dynamiq.py'),source_sha256=sha(ROOT.parent/'spl_model_communication/src/splcomm/dynamiq.py'),
        source_adapter='Only encode/decode threshold injection and ledger capture overridden; original routing method executed unchanged',results=results)
    target=run/'logs/original_stream_ring_crosscheck.json'
    with target.open('x',encoding='utf-8') as f:json.dump(log,f,indent=2)
    print(json.dumps(dict(status='pass',cases=len(results))))
if __name__=='__main__':main()
