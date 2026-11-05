"""Development-only direct pack then fused codec checks, preserving all attempts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gradient_comm_lab import cuda_codec as cc
from gradient_comm_lab.cpu_reference import CpuReference,pack_unsigned,unpack_unsigned,tensor_identity
from gradient_comm_lab.fixtures_v2 import load_fixture

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',required=True);ap.add_argument('--attempt',required=True);ap.add_argument('--all-fixtures',action='store_true');a=ap.parse_args()
    run=ROOT/a.run_dir;path=run/f'logs/cuda_dev_{a.attempt}.json';source=run/f'logs/cuda_dev_{a.attempt}_source.py'
    if path.exists() or source.exists():raise ValueError('Attempt exists')
    source.write_bytes(Path(__file__).read_bytes());cc.set_build_manifest(run/'logs/cuda_build_manifest.json')
    torch.use_deterministic_algorithms(True);torch.utils.deterministic.fill_uninitialized_memory=False
    rows=[];start=time.perf_counter()
    try:
        # First independently verify all legal symbols and odd/tail boundaries.
        for b in (4,8):
            for n in (0,1,3,7,8,15,16,17,255,256,257,513):
                codes=torch.arange(n,dtype=torch.int64)%((1<<b)+1)
                wanted=pack_unsigned(codes,b+1)
                packed=cc.pack_unsigned_cuda(codes.cuda(),b+1,max_code=1<<b)
                observed=bytes(packed.cpu().tolist())
                assert wanted==observed,(b,n,'pack')
                result=cc.unpack_unsigned_cuda(packed,b+1,n,max_code=1<<b)
                assert torch.equal(result.cpu(),codes),(b,n,'unpack')
                rows.append(dict(kind='direct_pack_unpack',bits=b,numel=n,status='pass'))
        fixtures=json.loads((run/'input_manifest.json').read_bytes())['inputs']
        if not a.all_fixtures:fixtures=[r for r in fixtures if r['original_numel']<=65536]
        for row in fixtures:
            f=load_fixture(row);x=f['input_logical'];m=f['main_thresholds'];s=f['scale_thresholds']
            for b in (4,8):
                ref=CpuReference(b).roundtrip(x,m,s);plan=cc.CudaCodecPlan(x.numel(),b)
                out=plan.roundtrip_into(x.cuda(),m.cuda(),s.cuda(),f['lookup_tables'][b].cuda(),checked=True)
                observed=bytes(plan.packet.serialized().cpu().tolist());expected=ref['wire']
                eq=observed==expected;bitseq=torch.equal(out.cpu().view(torch.int16),ref['decoded'].view(torch.int16))
                result=dict(kind='fused_roundtrip',input_id=row['input_id'],bits=b,numel=x.numel(),packet_equal=eq,decoded_bits_equal=bitseq,status='pass' if eq and bitseq else 'fail',workspace_bytes=plan.workspace_nbytes)
                rows.append(result);print(json.dumps(result),flush=True)
                if not eq or not bitseq:
                    counter=run/f'logs/cuda_dev_{a.attempt}_counterexample.pt'
                    torch.save(dict(fixture=row,ref=ref,gpu_wire=observed,gpu_output=out.cpu()),counter)
                    raise AssertionError('Fused codec mismatch')
        status='pass';error=None
    except Exception as exc:status='failed';error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc());traceback.print_exc()
    result=dict(status=status,scope='preliminary_not_formal',command=[sys.executable]+sys.argv,build=json.loads((run/'logs/cuda_build_manifest.json').read_bytes()),wrapper_sha256=hashlib.sha256(Path(cc.__file__).read_bytes()).hexdigest(),rows=rows,wall_seconds=time.perf_counter()-start,error=error)
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps(dict(status=status,checks=len(rows))))
    return 0 if status=='pass' else 1
if __name__=='__main__':sys.exit(main())
