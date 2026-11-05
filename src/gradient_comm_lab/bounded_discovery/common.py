"""Identity, seed and FP64 diagnostics for bounded discovery (no timing claims)."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone,timedelta
import torch

def now():return datetime.now(timezone(timedelta(hours=8))).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def dump(path,value,exclusive=True):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x' if exclusive else 'w',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def tensor_bytes(x):return x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
def identity(x):return dict(shape=list(x.shape),dtype=str(x.dtype),bytes=x.numel()*x.element_size(),sha256=hashlib.sha256(tensor_bytes(x)).hexdigest())
def seed_for(input_id,method_id,repeat_id):
    key=f'bounded_discovery_v1:{input_id}:{method_id}:{repeat_id}'
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')%(1<<63)
def local_metrics(x,y):
    assert x.shape==y.shape and x.dtype==y.dtype==torch.bfloat16
    nonfinite=int((~torch.isfinite(y)).sum())
    if nonfinite:return dict(status='fail',failure_reason='nonfinite decoded values',nonfinite_count=nonfinite)
    xd,yd=x.double(),y.double();e=yd-xd;sse=float(e.square().sum());energy=float(xd.square().sum());d=x.numel()
    return dict(status='pass',failure_reason=None,logical_numel=d,squared_error=sse,mse=sse/d if d else None,
        input_energy=energy,normalized_squared_error=sse/energy if energy else None,max_abs_error=float(e.abs().max()) if d else None,
        nonfinite_count=0,undefined_metric_reason='zero input energy' if not energy else None)
def aggregate_metrics(xs,ys,packet_bytes):
    assert len(xs)==len(ys)==len(packet_bytes)==4
    assert len({x.numel() for x in xs+ys})==1
    source=torch.stack(xs).double();decoded=torch.stack(ys).double()
    if not bool(torch.isfinite(decoded).all()):return dict(status='fail',failure_reason='nonfinite worker output')
    target=source.sum(0);actual=decoded.sum(0);err=actual-target;d=target.numel()
    sse=float(err.square().sum());source_energy=float(source.square().sum());target_energy=float(target.square().sum())
    local_sum=float((decoded-source).square().sum());ratio=target_energy/source_energy if source_energy else None
    return dict(status='pass',failure_reason=None,worker_count=4,logical_numel=d,aggregate_squared_error=sse,
        aggregate_mse=sse/d if d else None,source_energy=source_energy,target_energy=target_energy,
        source_normalized_error=sse/source_energy if source_energy else None,target_relative_error=sse/target_energy if target_energy else None,
        cancellation_ratio=ratio,near_cancellation=ratio is not None and ratio<1e-6,local_error_sum=local_sum,cross_term=sse-local_sum,
        aggregate_max_abs_error=float(err.abs().max()) if d else None,one_shot_packet_bytes_sum=sum(packet_bytes),
        undefined_metric_reason=('zero source energy' if not source_energy else 'zero target energy' if not target_energy else None))

