"""Recompute paired epsilon diagnostics from saved packets and CPU BF16 outputs.

This program never imports a codec or calls CUDA. The first execution creates
new analysis files. --verify-only independently recomputes and compares without
writing, including the archive of all changed coordinate positions.
"""
import argparse,hashlib,json,math,statistics,sys
from collections import defaultdict
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from gradient_comm_lab.epsilon_diagnostic.packet import parse_packet,packet_summary

CONDITIONS=('control_add1e7','diagnostic_add0','uniform_reused')
REGIONS=('full','H','rest')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def tensor_sha(x):return hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def strict(text):
    def reject(x):raise ValueError('Nonfinite JSON constant '+x)
    def unique(pairs):
        d={}
        for k,v in pairs:
            if k in d:raise ValueError('Duplicate JSON key '+k)
            d[k]=v
        return d
    x=json.loads(text,parse_constant=reject,object_pairs_hook=unique)
    def finite(y):
        if isinstance(y,float):assert math.isfinite(y)
        elif isinstance(y,dict):
            for z in y.values():finite(z)
        elif isinstance(y,list):
            for z in y:finite(z)
    finite(x);return x
def read(p):return strict(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):return [strict(x) for x in Path(p).read_text(encoding='utf-8-sig').splitlines() if x.strip()]
def dump(p,value):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def stats(values):
    valid=[float(x) for x in values if x is not None]
    return dict(count=len(values),defined_count=len(valid),undefined_count=len(values)-len(valid),
                mean=statistics.mean(valid) if valid else None,sample_std=statistics.stdev(valid) if len(valid)>1 else None,
                min=min(valid) if valid else None,max=max(valid) if valid else None)
def ratio(a,b):return a/b if b else None
def rel(path,run):
    p=Path(path);return p if p.is_absolute() else run/p


class Analysis:
    def __init__(self,run,verify):
        self.run=run;self.verify=verify;self.checks=0;self.errors=[];self.raw=[];self.changes=[]
        self.contract=read(run/'experiment_contract.json');self.ch=sha(run/'experiment_contract.json')
        self.im=read(run/'input_randomness_manifest.json');self.regions=read(run/'region_manifest.json')
        self.index=read(run/'execution_index.json')
        self.identity=dict(run_id=run.name,contract_sha256=self.ch,input_randomness_manifest_sha256=sha(run/'input_randomness_manifest.json'),region_manifest_sha256=sha(run/'region_manifest.json'))
        self.path_hashes={p:sha(run/p) for p in ['experiment_contract.json','input_randomness_manifest.json','region_manifest.json','execution_index.json','control_parity.jsonl','correctness.jsonl']}
        self.domains=[];self.artifact_hash_cache={};self.tensor_cache={}
    def check(self,ok,message):
        self.checks+=1
        if not ok:raise ValueError(message)
    def file(self,path,digest):
        p=rel(path,self.run)
        if str(p) not in self.artifact_hash_cache:self.artifact_hash_cache[str(p)]=sha(p)
        self.check(self.artifact_hash_cache[str(p)]==digest,'Artifact SHA mismatch '+str(p));return p
    def tensor(self,path,digest,expected_hash=None):
        p=self.file(path,digest);x=torch.load(p,map_location='cpu',weights_only=True)
        self.check(isinstance(x,torch.Tensor) and x.dtype==torch.bfloat16 and tuple(x.shape)==(818048,), 'Expected logical CPU BF16 vector '+str(p))
        if expected_hash is not None:self.check(tensor_sha(x)==expected_hash,'Decoded tensor identity mismatch '+str(p))
        return x
    def setup(self):
        self.check(self.index['contract_sha256']==self.ch,'Execution index uses a different frozen contract')
        self.check(sha(self.run/'input_randomness_manifest.json')==self.contract['input_randomness_manifest_sha256'],'Frozen input/randomness identity differs')
        self.check(sha(self.run/'region_manifest.json')==self.contract['region_manifest_sha256'],'Frozen H regions identity differs')
        self.check(sha(self.run/'source_manifest.json')==self.contract['source_manifest_sha256'],'Frozen source manifest identity differs')
        for row in read(self.run/'source_manifest.json')['files']:
            self.check(sha(rel(row['path'],self.run))==row['sha256'],'Frozen execution source or module differs: '+row['path'])
        pairs=self.im['pairs'];self.check(len(pairs)==96,'Expected 96 original paired states')
        self.pairs={p['pair_id']:p for p in pairs};self.check(len(self.pairs)==96,'Duplicate original pair_id')
        self.new={p['pair_id']:p for p in self.index['pairs']};self.check(set(self.new)==set(self.pairs),'Execution index pair coverage differs')
        keys={(p['snapshot_id'],p['worker_id'],p['repeat_id']) for p in pairs};ss=sorted({p['snapshot_id'] for p in pairs})
        self.check(len(ss)==3 and len(keys)==96,'Bad snapshot/worker/repeat coverage')
        self.check(keys=={(s,w,k) for s in ss for w in range(4) for k in range(8)},'Missing planned key')
        self.snapshots=ss;self.grouped=defaultdict(list);self.masks={}
        regions={r['snapshot_id']:r for r in self.regions['regions']}
        for s in ss:
            ids=regions[s]['block_ids'];self.check(len(ids)==len(set(ids))==32,'H must contain 32 unique frozen blocks')
            mask=torch.zeros(818048,dtype=torch.bool)
            for bid in ids:
                self.check(type(bid) is int and 0<=bid<3196,'Illegal frozen H block')
                mask[bid*256:min((bid+1)*256,818048)]=True
            self.masks[s]={'full':slice(None),'H':mask,'rest':~mask}
        for p in pairs:
            self.check(p['already_divided_by_four'] is True and p['logical_numel']==818048 and p['padded_numel']==818176,'Input domain differs')
            self.check(p['original_pool_sha256']==p['regenerated_pool_sha256'],'Regenerated pool differs')
            self.check(p['original_method_id']=='official_dynamiq_fixed_codec_b8','Unexpected original method')
            n=self.new[p['pair_id']]
            self.check(all(n[k]==p[k] for k in ['snapshot_id','worker_id','repeat_id','input_id']),'Execution identity mismatch')
            self.check(set(n['conditions'])==set(CONDITIONS[:2]),'Unexpected condition set')
            for condition in CONDITIONS[:2]:self.check(n['conditions'][condition]['status']=='pass','Nonpassing encoder/decoder prevents comparison')
            self.grouped[p['snapshot_id'],p['repeat_id']].append(p)
        parity=rows(self.run/'control_parity.jsonl');self.check(len(parity)==96 and {r['pair_id'] for r in parity}==set(self.pairs),'Control parity coverage invalid')
        self.check(all(r['contract_sha256']==self.ch for r in parity),'Control parity contract identity mismatch')
        self.check(all(r['status']=='pass' and r['packet_byte_mismatch_count']==r['decoded_bit_mismatch_count']==0 and r['pool_hash_match'] and r['input_hash_match'] for r in parity),'Control parity failed')
        checks=rows(self.run/'correctness.jsonl');self.check(bool(checks) and all(r['status']=='pass' for r in checks),'Correctness qualification not uniformly passed')
        self.check(all(r['contract_sha256']==self.ch for r in checks),'Correctness contract identity mismatch')
        self.check(self.im.get('status','pass')=='pass','Precheck manifest not qualified')
    def source(self,p):
        iid=p['input_id']
        if iid not in self.tensor_cache:
            path=self.file(p['input_path'],p['input_file_sha256'])
            value=torch.load(path,map_location='cpu',weights_only=False)['input_logical']
            self.check(value.dtype==torch.bfloat16 and tuple(value.shape)==(818048,) and tensor_sha(value)==p['input_tensor_sha256'],'Logical source tensor differs')
            self.tensor_cache[iid]=value
        return self.tensor_cache[iid]
    def outputs(self,p):
        n=self.new[p['pair_id']];ys={};refs={}
        for c in CONDITIONS[:2]:
            r=n['conditions'][c];ys[c]=self.tensor(r['decoded_path'],r['decoded_sha256'],r.get('decoded_tensor_sha256'))
            refs[c]=dict(path=str(rel(r['decoded_path'],self.run)),sha256=r['decoded_sha256'],tensor_sha256=tensor_sha(ys[c]),reused=False)
        u=p['uniform_packet_or_output_reference'];ys['uniform_reused']=self.tensor(u['decoded_path'],u['decoded_sha256'],u.get('decoded_tensor_sha256'))
        refs['uniform_reused']=dict(path=str(rel(u['decoded_path'],self.run)),sha256=u['decoded_sha256'],tensor_sha256=tensor_sha(ys['uniform_reused']),reused=True,original_record_key=u['original_record_key'])
        old=self.tensor(p['original_decoded_path'],p['original_decoded_sha256'],p.get('original_decoded_tensor_sha256'))
        self.check(tensor_sha(old)==tensor_sha(ys['control_add1e7']),'Independently checked rebuilt control decoded bytes differ from sealed original')
        return ys,refs
    def packet(self,p,ys):
        n=self.new[p['pair_id']];par={};wires={};paths={}
        for c in CONDITIONS[:2]:
            r=n['conditions'][c];path=self.file(r['packet_path'],r['packet_sha256']);w=path.read_bytes();wires[c]=w
            par[c]=parse_packet(w,818176,8);paths[c]=dict(path=str(path),sha256=r['packet_sha256'])
        old=self.file(p['original_packet_path'],p['original_packet_sha256']).read_bytes()
        self.check(wires['control_add1e7']==old,'Control actual packet bytes differ from sealed packet')
        a,b=par['control_add1e7'],par['diagnostic_add0'];ya,yb=ys['control_add1e7'],ys['diagnostic_add0']
        locations=dict(packet_byte_indices=np.flatnonzero(np.frombuffer(wires['control_add1e7'],np.uint8)!=np.frombuffer(wires['diagnostic_add0'],np.uint8)).astype(np.uint32),
            main_code_indices=np.flatnonzero(a['main_codes']!=b['main_codes']).astype(np.uint32),
            group_scale_indices=np.flatnonzero(a['group_scale_codes']!=b['group_scale_codes']).astype(np.uint32),
            super_scale_indices=np.flatnonzero(a['super_scale_words']!=b['super_scale_words']).astype(np.uint32),
            decoded_logical_indices=torch.nonzero(ya.view(torch.int16)!=yb.view(torch.int16)).flatten().numpy().astype(np.uint32))
        location_path=self.run/'changes'/f"{p['pair_id']}.npz"
        if self.verify:
            with np.load(location_path,allow_pickle=False) as saved:
                self.check(set(saved.files)==set(locations),'Changed-position archive schema differs')
                for k,v in locations.items():self.check(np.array_equal(saved[k],v),'Changed-position archive differs: '+k)
        else:
            location_path.parent.mkdir(exist_ok=True);self.check(not location_path.exists(),'Position archive already exists')
            np.savez_compressed(location_path,**locations)
        na=int((~torch.isfinite(ya)).sum());nb=int((~torch.isfinite(yb)).sum())
        ok=a['status']==b['status']=='pass' and na==nb==0 and a['packet_bytes']==b['packet_bytes']==875704
        change=dict(**self.identity,pair_id=p['pair_id'],snapshot_id=p['snapshot_id'],worker_id=p['worker_id'],repeat_id=p['repeat_id'],
            condition_packet_paths_and_hashes=paths,packet_bytes_A=a['packet_bytes'],packet_bytes_B=b['packet_bytes'],
            changed_packet_bytes=len(locations['packet_byte_indices']),changed_main_codes=len(locations['main_code_indices']),main_code_count=818176,
            changed_group_scale_codes=len(locations['group_scale_indices']),group_scale_count=51136,
            changed_super_scale_units=len(locations['super_scale_indices']),super_scale_count=3196,
            changed_decoded_elements_logical=len(locations['decoded_logical_indices']),logical_numel=818048,
            boundary_code_fraction_A=a['boundary_code_fraction'],boundary_code_fraction_B=b['boundary_code_fraction'],
            nonfinite_output_count_A=na,nonfinite_output_count_B=nb,reserved_code_count=dict(A=a['reserved_code_count'],B=b['reserved_code_count']),
            invalid_scale_count=dict(A=a['invalid_scale_count'],B=b['invalid_scale_count']),clamp_event_count=None,
            denominator_scope='packet bytes/main codes/group scales/super-scales include native padding P=818176; decoded changes only logical D=818048',
            changes_path=location_path.relative_to(self.run).as_posix(),changes_sha256=sha(location_path),
            status='pass' if ok else 'fail',packet_domain_A=packet_summary(a),packet_domain_B=packet_summary(b))
        for field,total in [('changed_packet_bytes',875704),('changed_main_codes',818176),('changed_group_scale_codes',51136),('changed_super_scale_units',3196),('changed_decoded_elements_logical',818048)]:change[field+'_fraction']=change[field]/total
        change['region_change_counts']={}
        for region,mask in self.masks[p['snapshot_id']].items():
            h=mask if not isinstance(mask,slice) else torch.ones(818048,dtype=torch.bool)
            change['region_change_counts'][region]=int(torch.count_nonzero((ya.view(torch.int16)!=yb.view(torch.int16))[h]))
        self.changes.append(change)
        if not ok:self.domains.append(dict(pair_id=p['pair_id'],reason='Illegal packet or nonfinite decoded output'))
    def append_local(self,p,c,region,x,y,ref):
        mask=self.masks[p['snapshot_id']][region];x=x[mask].double();y=y[mask].double();e=y-x
        sse=float(e.square().sum());energy=float(x.square().sum());d=x.numel()
        row=dict(**self.identity,record_type='local',pair_id=p['pair_id'],condition_id=c,region_id=region,
            snapshot_id=p['snapshot_id'],worker_id=p['worker_id'],repeat_id=p['repeat_id'],input_id=p['input_id'],
            coordinate_count=d,logical_numel=818048,sse=sse,mse=ratio(sse,d),input_energy=energy,source_energy=energy,
            normalized_error=ratio(sse,energy),max_abs_error=float(e.abs().max()) if d else None,nonfinite_count=0,
            output_source=ref,reused=ref['reused'],original_record_key=ref.get('original_record_key'),status='pass',failure_reason=None,
            undefined_metric_reason='zero source energy' if not energy else None,packet_bytes=843628 if c=='uniform_reused' else 875704)
        self.raw.append(row)
    def append_aggregate(self,snapshot,repeat,c,region,xs,ys,refs,pairs):
        mask=self.masks[snapshot][region];source=torch.stack([x[mask].double() for x in xs]);decoded=torch.stack([y[mask].double() for y in ys])
        target=source.sum(0);actual=decoded.sum(0);e=actual-target;sse=float(e.square().sum());se=float(source.square().sum());te=float(target.square().sum());le=float((decoded-source).square().sum());d=target.numel()
        row=dict(**self.identity,record_type='one_shot',pair_id=f'{snapshot}__r{repeat:02d}',condition_id=c,region_id=region,
            snapshot_id=snapshot,worker_id=None,repeat_id=repeat,input_ids=[p['input_id'] for p in pairs],coordinate_count=d,logical_numel=818048,
            sse=sse,mse=ratio(sse,d),source_energy=se,target_energy=te,normalized_error=ratio(sse,se),target_relative_error=ratio(sse,te),
            cancellation_ratio=ratio(te,se),near_cancellation=se>0 and te/se<1e-6,local_error_sum=le,cross_term=sse-le,
            max_abs_error=float(e.abs().max()) if d else None,nonfinite_count=0,output_sources=refs,reused=c=='uniform_reused',
            original_record_key=[r.get('original_record_key') for r in refs] if c=='uniform_reused' else None,
            status='pass',failure_reason=None,undefined_metric_reason='zero source energy' if not se else 'zero target energy' if not te else None,
            one_shot_packet_bytes_sum=4*(843628 if c=='uniform_reused' else 875704))
        self.raw.append(row)
    def calculate(self):
        for (s,k),pairs in sorted(self.grouped.items()):
            pairs=sorted(pairs,key=lambda p:p['worker_id']);self.check([p['worker_id'] for p in pairs]==[0,1,2,3],'Incomplete worker group')
            xs=[];outputs={c:[] for c in CONDITIONS};references={c:[] for c in CONDITIONS}
            for p in pairs:
                x=self.source(p);ys,refs=self.outputs(p);self.packet(p,ys)
                if self.domains:continue
                xs.append(x)
                for c in CONDITIONS:
                    outputs[c].append(ys[c]);references[c].append(refs[c])
                    self.check(bool(torch.isfinite(ys[c]).all()),'Nonfinite output including reused reference')
                    for region in REGIONS:self.append_local(p,c,region,x,ys[c],refs[c])
            if not self.domains:
                for c in CONDITIONS:
                    for region in REGIONS:self.append_aggregate(s,k,c,region,xs,outputs[c],references[c],pairs)
        if self.domains:return
        self.check(len(self.raw)==1080,'Expected 864 local + 216 one-shot raw region records')
        keys={(r['record_type'],r['pair_id'],r['condition_id'],r['region_id']) for r in self.raw};self.check(len(keys)==1080,'Duplicate raw identity')
        lookup={(r['record_type'],r['pair_id'],r['condition_id'],r['region_id']):r for r in self.raw}
        for r in self.raw:
            if r['region_id']!='full':continue
            h=lookup[r['record_type'],r['pair_id'],r['condition_id'],'H'];rest=lookup[r['record_type'],r['pair_id'],r['condition_id'],'rest']
            self.check(math.isclose(r['sse'],h['sse']+rest['sse'],rel_tol=1e-12,abs_tol=1e-28),'H/rest SSE partition failed')
            self.check(r['coordinate_count']==h['coordinate_count']+rest['coordinate_count'],'H/rest coordinate partition failed')
    def summarize_group(self,record_type,snapshot,worker,region,triples):
        records=[]
        for k,values in sorted(triples.items()):
            a,b,u=(values[c] for c in CONDITIONS);delta=a['sse']-b['sse']
            records.append(dict(repeat_id=k,control_sse=a['sse'],diagnostic_sse=b['sse'],uniform_sse=u['sse'],paired_delta=delta,
                relative_reduction=ratio(delta,a['sse']),gap_before=a['sse']-u['sse'],gap_after=b['sse']-u['sse'],source_energy=a['source_energy'],
                normalized_control=ratio(a['sse'],a['source_energy']),normalized_diagnostic=ratio(b['sse'],a['source_energy']),normalized_uniform=ratio(u['sse'],a['source_energy'])))
        fields=[f for f in records[0] if f!='repeat_id'];statistics_by={f:stats([r[f] for r in records]) for f in fields}
        den=statistics_by['gap_before']['mean'];gc=ratio(statistics_by['paired_delta']['mean'],den) if snapshot.endswith('000100') and den>0 else None
        return dict(record_type=record_type,snapshot_id=snapshot,worker_id=worker,region_id=region,repeat_count=len(records),
            statistics=statistics_by,paired_repeats_improved=sum(r['paired_delta']>0 for r in records),paired_repeats_worsened=sum(r['paired_delta']<0 for r in records),
            paired_repeats_equal=sum(r['paired_delta']==0 for r in records),gap_closure=gc,
            gap_closure_definition='mean(paired_delta)/mean(gap_before), only update100 with positive denominator',
            gap_closure_undefined_reason=None if gc is not None else 'not update100 or mean control-uniform SSE gap is nonpositive',
            other_snapshots_relative_sse_change=ratio(statistics_by['diagnostic_sse']['mean']-statistics_by['control_sse']['mean'],statistics_by['control_sse']['mean']),
            per_repeat=records)
    def summaries(self):
        groups=defaultdict(lambda:defaultdict(dict));local=defaultdict(list)
        for r in self.raw:
            key=(r['record_type'],r['snapshot_id'],r['worker_id'],r['region_id']);groups[key][r['repeat_id']][r['condition_id']]=r
            if r['record_type']=='local':local[r['snapshot_id'],r['repeat_id'],r['condition_id'],r['region_id']].append(r)
        for (s,k,c,region),rs in local.items():
            self.check(len(rs)==4,'Local snapshot reduction lacks four workers')
            groups['snapshot_local',s,None,region][k][c]=dict(sse=math.fsum(r['sse'] for r in rs),source_energy=math.fsum(r['source_energy'] for r in rs))
        out=[]
        for (typ,s,w,region),reps in sorted(groups.items(),key=lambda x:str(x[0])):
            self.check(set(reps)==set(range(8)) and all(set(v)==set(CONDITIONS) for v in reps.values()),'Missing paired condition/repeat')
            out.append(self.summarize_group(typ,s,w,region,reps))
        look={(r['record_type'],r['snapshot_id'],r['worker_id'],r['region_id']):r for r in out}
        first=self.snapshots[0];full=look['snapshot_local',first,None,'full'];h=look['snapshot_local',first,None,'H']
        gc=full['gap_closure'];gch=h['gap_closure'];workers=sum(look['local',first,w,'full']['statistics']['paired_delta']['mean']>0 for w in range(4))
        repeats=full['paired_repeats_improved'];regressions={s:look['snapshot_local',s,None,'full']['other_snapshots_relative_sse_change'] for s in self.snapshots[1:]}
        all_same=all(c['changed_decoded_elements_logical']==0 for c in self.changes)
        same_sse=all(r['statistics']['paired_delta']['mean']==0 and r['statistics']['paired_delta']['min']==0 and r['statistics']['paired_delta']['max']==0 for r in out)
        if all_same:self.check(same_sse,'Identical output should imply exactly equal SSE')
        outcome='no_effect_observed' if all_same else 'inconclusive' if gc is None else 'changes_without_closing_gap' if gc<=0 else 'partially_closes_gap' if gc<1 else 'closes_or_reverses_gap'
        checks=dict(gap_closure_at_least_half=dict(value=gc,threshold=.5,pass_=gc is not None and gc>=.5),
            gap_closure_H_at_least_half=dict(value=gch,threshold=.5,pass_=gch is not None and gch>=.5),
            at_least_three_workers_improve=dict(value=workers,threshold=3,pass_=workers>=3),
            at_least_seven_paired_repeats_improve=dict(value=repeats,threshold=7,pass_=repeats>=7),
            other_snapshots_regression_at_most_two_percent=dict(values=regressions,threshold=.02,pass_=all(x is not None and x<=.02 for x in regressions.values())))
        for v in checks.values():v['passed']=v.pop('pass_')
        return dict(**self.identity,status='pass',outcome=outcome,continuation_gate='met' if all(v['passed'] for v in checks.values()) else 'not_met',
            continuation_checks=checks,update100=dict(gap_closure=gc,gap_closure_H=gch,workers_improved=workers,paired_repeats_improved=repeats),
            other_snapshots_relative_sse_change=regressions,all_formal_outputs_bitwise_identical=all_same,all_sse_identical=same_sse,
            groups=out,coverage=dict(pairs=96,local_raw_records=864,one_shot_raw_records=216,paired_groups=len(out),failed_pairs=0,missing_pairs=0),
            statistical_scope='Matched randomness only between control and diagnostic. Uniform remains independently generated old outputs; eight repeats are not independent training trajectories.',
            source_hashes=self.path_hashes)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',required=True);ap.add_argument('--verify-only',action='store_true');a=ap.parse_args()
    run=(ROOT/a.run_dir).resolve();assert run.is_relative_to(ROOT/'results/epsilon_diagnostic_v1')
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    analysis=Analysis(run,a.verify_only)
    try:
        analysis.setup();analysis.calculate()
        if analysis.domains:
            result=dict(**analysis.identity,status='fail',outcome='diagnostic_domain_failure',continuation_gate='not_evaluated',update100=None,
                        other_snapshots_relative_sse_change=None,domain_failures=analysis.domains,groups=None,source_hashes=analysis.path_hashes)
            analysis.raw=[] # No successful-subset effects are emitted as formal results.
        else:result=analysis.summaries()
        if a.verify_only:
            analysis.check(rows(run/'metrics_raw.jsonl')==analysis.raw,'Saved raw does not match independent recomputation')
            analysis.check(rows(run/'packet_changes.jsonl')==analysis.changes,'Saved packet changes do not match recomputation')
            analysis.check(read(run/'paired_summary.json')==result,'Saved paired summary does not match recomputation')
        else:
            for name,data in [('metrics_raw.jsonl',analysis.raw),('packet_changes.jsonl',analysis.changes)]:
                if data:
                    with (run/name).open('x',encoding='utf-8') as f:
                        for r in data:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
            dump(run/'paired_summary.json',result)
        report=dict(status='pass' if not analysis.domains else 'fail',outcome=result['outcome'],continuation_gate=result['continuation_gate'],
                    checks=analysis.checks,pairs=len(analysis.changes),raw_records=len(analysis.raw),verify_only=a.verify_only,
                    execution_device='CPU only; no codec imported',source_sha256=sha(__file__),packet_parser_sha256=sha(ROOT/'src/gradient_comm_lab/epsilon_diagnostic/packet.py'),
                    input_artifact_count=len(analysis.artifact_hash_cache),recorded_at=datetime.now(timezone.utc).isoformat())
        if not a.verify_only:dump(run/'logs/analysis_execution.json',report)
        print(json.dumps(report));return 0 if not analysis.domains else 2
    except Exception as exc:
        if not a.verify_only:
            path=run/'logs/analysis_failure.json'
            if not path.exists():dump(path,dict(status='fail',exception=repr(exc),checks=analysis.checks,domain_failures=analysis.domains))
        raise
if __name__=='__main__':raise SystemExit(main())
