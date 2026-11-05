"""Frozen G2a inputs and event-addressed thresholds; no GPU execution."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import torch
from .cpu_reference import tensor_identity

def canonical_event_key(event):
    fields=('case_id','nominal_bits','phase','rank','chunk','encoding_ordinal')
    if set(event)!=set(fields): raise ValueError('Unexpected threshold event fields')
    return json.dumps(event,sort_keys=True,separators=(',',':'),ensure_ascii=True)

def make_thresholds(event,length):
    key=canonical_event_key(event)
    seed=int.from_bytes(hashlib.sha256(('logical_ring_v1:'+key).encode()).digest()[:8],'little')%(1<<63)
    g=torch.Generator(device='cpu').manual_seed(seed)
    constant=torch.tensor(65535,dtype=torch.bfloat16)
    main=torch.rand((length//16,16),generator=g).to(torch.bfloat16)*constant
    scale=torch.rand((length//256,16),generator=g).to(torch.bfloat16)*constant
    return main,scale,dict(event=event,event_key=key,seed=seed,length=length,
        main=tensor_identity(main),scale=tensor_identity(scale))

def case_events(case):
    n=case['worker_count'];d=case['logical_numel']
    if n==1 or d==0:return
    for hop in range(n-1):
        for rank in range(n):
            yield dict(case_id=case['case_id'],nominal_bits=case['nominal_bits'],phase='reduce_scatter',
                rank=rank,chunk=(rank-hop)%n,encoding_ordinal=hop)
    for rank in range(n):
        yield dict(case_id=case['case_id'],nominal_bits=case['nominal_bits'],phase='sink',
            rank=rank,chunk=(rank+1)%n,encoding_ordinal=n-1)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_case(case,run_dir):
    p=Path(run_dir)/case['fixture_path']
    if sha(p)!=case['fixture_sha256']:raise ValueError('Case fixture archive changed')
    obj=torch.load(p,map_location='cpu',weights_only=True)
    for x,row in zip(obj['inputs'],case['workers']):
        if tensor_identity(x)!=row['tensor']:raise ValueError('Case tensor changed')
    if len(obj['inputs'])!=case['worker_count']:raise ValueError('Worker count changed')
    return obj['inputs']

class FrozenThresholdProvider:
    def __init__(self,run_dir,device='cpu'):
        self.run_dir=Path(run_dir);self.device=device;self.cache={};self.consumed=[]
        entries=json.loads((self.run_dir/'threshold_manifest.json').read_text(encoding='utf-8'))['events']
        self.entries={r['event_key']:r for r in entries}
        if len(self.entries)!=len(entries):raise ValueError('Duplicate threshold keys')
    def describe(self,event):return self.entries[canonical_event_key(event)].copy()
    def __call__(self,event):
        key=canonical_event_key(event);row=self.entries[key]
        if key not in self.cache:
            a,b,meta=make_thresholds(event,row['length'])
            if meta!=row:raise ValueError('Threshold regeneration mismatch')
            self.cache[key]=(a.to(self.device),b.to(self.device))
        a,b=self.cache[key]
        self.consumed.append(key)
        return dict(row,main_thresholds=a,scale_thresholds=b)
    get=__call__

