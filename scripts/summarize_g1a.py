"""Recompute G1a statistics and figures from immutable records; CPU only.

Only derived benchmark_summary.jsonl, analysis.json and named figures are
overwritten. Invalid or incomplete evidence is reported and returns exit 2.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
METHOD='torch_cuda_repaired'
IDENTITY=('run_id','case_id','input_id','method_id','contract_sha256','benchmark_config_sha256',
          'nominal_bits','stored_bits','original_numel','padded_numel','operation','allocation_mode',
          'validation_mode','graph_mode','compile_mode','cache_mode','rng_work_included','coverage_group')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def reject_constant(value):raise ValueError('Nonfinite JSON constant: '+value)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'),parse_constant=reject_constant)
def read_lines(path):
    return [json.loads(line,parse_constant=reject_constant) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def finite_nonnegative(value):return type(value) in (float,int) and math.isfinite(value) and value>=0


class Checks:
    def __init__(self):self.errors=[];self.count=0
    def require(self,condition,code,**detail):
        if not condition:
            self.count+=1
            if len(self.errors)<500:self.errors.append({'code':code,**detail})
        return bool(condition)


def expected_cases(manifest):
    """Independent statement of the reviewed, single-method 104-case grid."""
    out={}
    def add(row,bits,operation,allocation,validation,group):
        case=f"{row['input_id']}__b{bits}__{operation}__{allocation}__{validation}"
        out[(case,METHOD)]={'case_id':case,'input_id':row['input_id'],'method_id':METHOD,
            'nominal_bits':bits,'stored_bits':bits+1,'original_numel':row['original_numel'],
            'padded_numel':row['padded_numel'],'operation':operation,'allocation_mode':allocation,
            'validation_mode':validation,'coverage_group':group,'graph_mode':'off','compile_mode':'eager',
            'cache_mode':'steady_state_reuse','rng_work_included':False,'input_dtype':'bfloat16',
            'output_dtype':'uint8_packet' if operation=='encode' else 'bfloat16'}
    for row in manifest['inputs']:
        if row['source_kind']=='synthetic' and row['generator_config']['seed']==7301:
            for b in (4,8):
                for op in ('encode','decode','roundtrip'):add(row,b,op,'preallocated','trusted_inputs','main')
                if row['generator_config']['distribution']=='standard_normal' and row['original_numel'] in (65536,1048576):
                    for validation in ('trusted_inputs','checked_api'):add(row,b,'roundtrip','api_allocating',validation,'api')
        elif row['source_kind']=='real_gradient':
            for b in (4,8):add(row,b,'roundtrip','preallocated','trusted_inputs','real_gradient')
    return out


def statistics(values):
    q=np.quantile(np.asarray(values,dtype=np.float64),[.25,.5,.75,.95],method='linear')
    return {name:float(value) for name,value in zip(('p25','median','p75','p95'),q)}


def summarize_group(rows,base):
    sessions=[]
    for session in sorted({r['session_id'] for r in rows}):
        current=[r for r in rows if r['session_id']==session]
        gpu=statistics([r['gpu_elapsed_ms']*1000/r['inner_repeats'] for r in current])
        wall=statistics([r['wall_elapsed_ms']*1000/r['inner_repeats'] for r in current])
        ratio=(gpu['p75']-gpu['p25'])/gpu['median'] if gpu['median'] else None
        sessions.append({'session_id':session,'valid_block_count':len(current),
            **{'gpu_'+k+'_us':v for k,v in gpu.items()},'wall_median_us':wall['median'],
            'gpu_iqr_over_median':ratio,'noise_flag':ratio is not None and ratio>.10,
            'p95_interpretation':'p95 of block-average per-call latency; not individual request tail latency',
            'inner_repeats':sorted({r['inner_repeats'] for r in current})})
    medians=[s['gpu_median_us'] for s in sessions]
    median=float(np.median(medians)) if medians else None
    n,l,b=base['original_numel'],base['padded_numel'],base['nominal_bits']
    wire=(l*(b+1)+7)//8+l//16+2*(l//256)
    missing=[]
    for session in range(5):
        ids={r['block_id'] for r in rows if r['session_id']==session}
        absent=sorted(set(range(20))-ids)
        if absent:missing.append({'session_id':session,'missing_block_ids':absent})
    return {**base,'valid_session_count':len(sessions),'valid_block_count':len(rows),'session_statistics':sessions,
        'gpu_session_median_us_median':median,'gpu_session_median_us_min':min(medians) if medians else None,
        'gpu_session_median_us_max':max(medians) if medians else None,
        'wall_session_median_us_median':float(np.median([s['wall_median_us'] for s in sessions])) if sessions else None,
        'original_input_bytes':n*2,'wire_bytes':wire,'effective_bits_per_logical_element':wire*8/n if n else None,
        'input_equivalent_gbps':n*2/(median*1e-6)/1e9 if median else None,
        'paired_reference_method':None,'paired_speedup_by_session':{},
        'noise_flag':any(s['noise_flag'] for s in sessions),'missing_coverage':missing,
        'interpretation':'Single PyTorch GPU baseline only; no optimized-method speedup or communication claim.',
        'quantile_implementation':'numpy.quantile(method=linear)',
        'event_timing_meaning':'CUDA event interval includes stream execution and possible eager launch gaps; not a sum of kernel durations.'}


def memory_summary(rows):
    keys=('input_bytes','threshold_bytes','lut_bytes','packet_buffer_bytes','decoded_output_bytes',
          'workspace_bytes','total_plan_owned_storage_bytes','allocated_before_bytes','allocated_after_bytes',
          'peak_allocated_bytes','reserved_before_bytes','peak_reserved_bytes','peak_extra_allocated_bytes')
    result=[]
    for key in sorted({(r['case_id'],r['method_id']) for r in rows}):
        current=[r for r in rows if (r['case_id'],r['method_id'])==key]
        result.append({'case_id':key[0],'method_id':key[1],'sessions':len(current),
            **{name:{'median':float(np.median([r[name] for r in current])),
                     'min':min(r[name] for r in current),'max':max(r[name] for r in current)} for name in keys},
            'baseline_meaning':'Already includes input, thresholds, LUT and the persistent preallocated plan.',
            'extra_meaning':'Peak allocated minus allocated before the isolated probe; zero does not mean zero workspace.',
            'api_allocating_meaning':'Persistent reference plan remains in baseline; an additional entire plan is created per API call.'})
    return result


def duration_diagnostics(groups,config):
    low,high=config.get('target_block_ms',[10,50]);all_times=[];cases=[]
    for (case,method),rows in sorted(groups.items()):
        times=[r['gpu_elapsed_ms'] for r in rows];all_times.extend(times)
        below=sum(t<low for t in times);above=sum(t>high for t in times)
        cases.append({'case_id':case,'method_id':method,'block_count':len(times),
            'elapsed_ms_min':min(times),'elapsed_ms_median':float(np.median(times)),'elapsed_ms_max':max(times),
            'below_target_count':below,'above_target_count':above,'outside_target_count':below+above})
    return {'target_block_ms':[low,high],'block_count':len(all_times),
        'elapsed_ms_min':min(all_times) if all_times else None,
        'elapsed_ms_median':float(np.median(all_times)) if all_times else None,
        'elapsed_ms_max':max(all_times) if all_times else None,
        'below_target_count':sum(t<low for t in all_times),'above_target_count':sum(t>high for t in all_times),
        'cases':cases,'interpretation':'Target-range misses are quality diagnostics only. K stays frozen and all valid observations are retained.'}


def verify_cupti_capture(run,manifest,events):
    """Check bound raw evidence before treating absent activity as a zero count."""
    if not manifest.get('raw_activity_path') or not manifest.get('audit_path'):
        return {'status':'unverified','reason':'No bound raw activity and coverage audit.'}
    try:
        artifacts={a['path'] for a in manifest.get('artifacts',[])}
        if not {manifest['raw_activity_path'],manifest['audit_path']}<=artifacts:
            raise ValueError('Raw activity/audit is not bound by artifact hash')
        raw=read_lines(run/manifest['raw_activity_path']);audit=read(run/manifest['audit_path'])
        enabled=sorted(r['kind'] for r in raw if r.get('cat')=='enabled_activity')
        required=[1,2,4,5,10,38,39]
        if enabled!=required or manifest.get('enabled_activity_kinds')!=required or audit.get('enabled_activity_kinds')!=required:
            raise ValueError('Required memcpy/memset/runtime/driver/kernel/sync/external activities were not all enabled')
        final=[r for r in raw if r.get('cat')=='collector_summary']
        if (len(final)!=1 or final[0].get('dropped_records_total')!=0 or final[0].get('error') or final[0].get('first_error') or
            any(r.get('cat')=='collector_error' for r in raw) or
            sum(r.get('dropped_records',0) for r in raw if r.get('cat')=='buffer_metadata')!=0 or
            manifest.get('dropped_records')!=0 or manifest.get('collection_errors')!=[] or audit.get('status')!='pass'):
            raise ValueError('Collector error, dropped activity, missing summary or failed audit')
        correlations=defaultdict(set)
        for r in raw:
            if r.get('cat')=='external_correlation' and r.get('external_kind')==3:
                correlations[r['correlation']].add(r['external_id'])
        if any(len(v)!=1 for v in correlations.values()):raise ValueError('Conflicting CUSTOM0 external correlations')
        mapping={key:next(iter(value)) for key,value in correlations.items()}
        measured=[r for r in raw if 'start_ns' in r and mapping.get(r['correlation']) in (1,2,3,100)]
        start,end=audit['window_start_ns'],audit['window_end_ns']
        if not measured or any(not(0<start<=r['start_ns']<=r['end_ns']<=end) or r.get('return_value',0)!=0 for r in measured):
            raise ValueError('Measured raw activity has invalid/out-of-window timestamps or failed CUDA API')
        chrome=[e for e in events if e.get('ph')=='X' and e.get('cat')!='user_annotation']
        raw_identity=Counter((r['cat'],r['correlation'],r['start_ns'],r['end_ns']) for r in measured)
        chrome_identity=Counter((e['cat'],e['args']['correlation'],round(e['ts']*1000)+start,
            round((e['ts']+e['dur'])*1000)+start) for e in chrome)
        if raw_identity!=chrome_identity:raise ValueError('Chrome activity does not exactly match measured raw activity')
        for e in chrome:
            external=mapping.get(e['args']['correlation'])
            if external in (1,2,3) and e['args'].get('operation_index')!=external-1:
                raise ValueError('Chrome operation attribution differs from raw external correlation')
        return {'status':'verified','raw_activity_path':manifest['raw_activity_path'],'audit_path':manifest['audit_path'],
            'enabled_activity_kinds':enabled,'dropped_records':0,'collection_errors':[],
            'measured_timed_records':len(measured),'chrome_records_exactly_match_measured_raw':True,
            'window_start_ns':start,'window_end_ns':end,
            'scope':'Three trusted preallocated roundtrips plus final synchronization only; zero counts do not describe setup or allocating API.'}
    except (KeyError,ValueError,TypeError) as exc:
        return {'status':'invalid','reason':str(exc)}


def analyze_trace(run,manifest_name='manifest.json'):
    manifest_path=run/'traces'/manifest_name
    if not manifest_path.exists():return {'status':'not_available','reason':'No trace manifest exists.'}
    manifest=read(manifest_path);path=(run/'traces'/manifest['trace']).resolve()
    if not path.is_relative_to(run) or not path.is_file():return {'status':'invalid','reason':'Manifest trace file is absent or outside run.'}
    if sha(path)!=manifest.get('sha256'):return {'status':'invalid','reason':'Trace SHA256 mismatch.'}
    if 'bytes' in manifest and path.stat().st_size!=manifest['bytes']:
        return {'status':'invalid','reason':'Trace size mismatch.'}
    for artifact in manifest.get('artifacts',[]):
        target=(run/artifact['path']).resolve()
        if not target.is_relative_to(run) or not target.is_file():
            return {'status':'invalid','reason':'Alternative trace artifact path is absent or outside run.'}
        if sha(target)!=artifact['sha256'] or target.stat().st_size!=artifact['bytes']:
            return {'status':'invalid','reason':'Alternative trace artifact hash/size mismatch.','artifact':artifact['path']}
    events=read(path).get('traceEvents',[])
    capture=verify_cupti_capture(run,manifest,events)
    if capture['status']=='invalid':return {'status':'invalid','reason':'CUPTI coverage audit: '+capture['reason']}
    complete=[e for e in events if e.get('ph')=='X']
    operations=sorted([e for e in complete if e.get('name')=='trusted_preallocated_roundtrip'],key=lambda e:e['ts'])
    kernels=[e for e in complete if 'kernel' in e.get('cat','').lower()]
    runtimes=[e for e in complete if any(c in e.get('cat','').lower() for c in ('cuda_runtime','cuda_driver'))]
    correlation={e.get('args',{}).get('correlation'):e for e in runtimes if 'correlation' in e.get('args',{})}
    per_operation=[{'operation_index':i,'kernel_count':0,'kernel_duration_sum_us':0.} for i in range(len(operations))]
    unassigned=0;top=defaultdict(lambda:{'count':0,'duration_sum_us':0.})
    for kernel in kernels:
        name=kernel['name'];duration=float(kernel.get('dur',0));top[name]['count']+=1;top[name]['duration_sum_us']+=duration
        launch=correlation.get(kernel.get('args',{}).get('correlation'));assigned=False
        explicit_operation=kernel.get('args',{}).get('operation_index')
        if type(explicit_operation) is int and 0<=explicit_operation<len(operations):
            per_operation[explicit_operation]['kernel_count']+=1
            per_operation[explicit_operation]['kernel_duration_sum_us']+=duration;assigned=True
        elif launch:
            for i,operation in enumerate(operations):
                if (launch.get('pid')==operation.get('pid') and launch.get('tid')==operation.get('tid') and
                        operation['ts']<=launch['ts']<=operation['ts']+operation.get('dur',0)):
                    per_operation[i]['kernel_count']+=1;per_operation[i]['kernel_duration_sum_us']+=duration;assigned=True;break
        if not assigned:unassigned+=1
    memcpy=[e for e in complete if 'gpu_memcpy' in e.get('cat','').lower()]
    memset=[e for e in complete if 'gpu_memset' in e.get('cat','').lower()]
    copy_runtime=[e for e in runtimes if 'memcpy' in e.get('name','').lower()]
    sync=[e for e in runtimes if 'synchronize' in e.get('name','').lower()]
    activity_sync=[e for e in complete if e.get('cat')=='cuda_sync']
    def inventory(items):
        totals=defaultdict(lambda:{'count':0,'duration_sum_us':0.})
        for e in items:totals[e['name']]['count']+=1;totals[e['name']]['duration_sum_us']+=float(e.get('dur',0))
        return [{'name':k,**v} for k,v in sorted(totals.items(),key=lambda pair:pair[1]['duration_sum_us'],reverse=True)]
    gpu_available=bool(kernels);runtime_available=bool(runtimes);coverage_verified=capture['status']=='verified'
    status='analyzed' if gpu_available and runtime_available and coverage_verified else ('partial_gpu' if gpu_available else 'cpu_only')
    copy_kinds={1:'HtoD',2:'DtoH',3:'HtoA',4:'AtoH',5:'AtoA',6:'AtoD',7:'DtoA',8:'DtoD',9:'HtoH',10:'PtoP'}
    directions=Counter(copy_kinds.get(e.get('args',{}).get('copy_kind'),'unknown') for e in memcpy)
    host_device_count=sum(directions[k] for k in ('HtoD','DtoH','HtoA','AtoH')) if coverage_verified and not directions['unknown'] else None
    if not gpu_available:
        per_operation=[{'operation_index':i,'kernel_count':None,'kernel_duration_sum_us':None} for i in range(len(operations))]
    return {'status':status,'trace_path':path.relative_to(run).as_posix(),'sha256':sha(path),
        'manifest_path':manifest_path.relative_to(run).as_posix(),'manifest_sha256':sha(manifest_path),
        'tool':manifest.get('tool','torch.profiler'),'gpu_activity_available':gpu_available,
        'cuda_runtime_activity_available':runtime_available,'capture_integrity':capture,
        'observed_event_categories':dict(Counter(e.get('cat','') for e in complete)),
        'cpu_operator_event_count':sum('cpu_op' in e.get('cat','').lower() for e in complete),
        'contract_sha256':manifest.get('contract_sha256'),'benchmark_config_sha256':manifest.get('benchmark_config_sha256'),
        'operation_count':len(operations),'expected_operation_count':manifest.get('operation_count'),
        'operation_count_matches':len(operations)==manifest.get('operation_count'),
        'kernel_count':len(kernels) if gpu_available else None,
        'kernel_count_per_operation_mean':len(kernels)/len(operations) if operations and gpu_available else None,
        'per_operation':per_operation,'unassigned_kernel_count':unassigned if gpu_available else None,
        'assignment_method':'Explicit kernel operation_index when present; otherwise GPU correlation id -> runtime launch inside CPU record_function interval on same pid/tid',
        'kernel_duration_sum_us':sum(float(k.get('dur',0)) for k in kernels) if gpu_available else None,
        'top_kernels':[{'name':name,**v,'mean_duration_us':v['duration_sum_us']/v['count']} for name,v in sorted(top.items(),key=lambda x:x[1]['duration_sum_us'],reverse=True)[:20]] if gpu_available else None,
        'device_memcpy_count':len(memcpy) if gpu_available else None,'device_memcpy_events':inventory(memcpy) if gpu_available else None,
        'device_memcpy_direction_counts':dict(directions) if gpu_available else None,
        'device_memcpy_bytes':sum(e.get('args',{}).get('bytes',0) for e in memcpy) if gpu_available else None,
        'device_HtoD_count':directions['HtoD'] if coverage_verified else None,
        'device_DtoH_count':directions['DtoH'] if coverage_verified else None,
        'device_host_transfer_count':host_device_count,
        'device_memset_count':len(memset) if gpu_available else None,'device_memset_events':inventory(memset) if gpu_available else None,
        'runtime_memcpy_count':len(copy_runtime) if runtime_available else None,'runtime_memcpy_events':inventory(copy_runtime) if runtime_available else None,
        'runtime_synchronization_count':len(sync) if runtime_available else None,'runtime_synchronization_events':inventory(sync) if runtime_available else None,
        'activity_synchronization_count':len(activity_sync) if coverage_verified else None,
        'runtime_boundary_synchronization_count':sum(bool(e.get('args',{}).get('boundary_synchronization')) for e in sync) if coverage_verified else None,
        'limitation':None if status=='analyzed' else 'GPU activity and/or collection coverage is incomplete. Missing activity is unknown, not zero; CPU operator counts and Self CUDA table values do not establish GPU kernel, copy or synchronization counts.',
        'host_copy_conclusion':'No host-device copy observed in the three-call window; all memcpy directions, activity enablement, zero drops and raw/Chrome identity were checked.' if host_device_count==0 else ('Host-device copies observed in the measured window.' if host_device_count else 'Unknown: complete GPU activity coverage or copy direction evidence is missing.'),
        'boundary':manifest.get('boundary'),'interpretation':'Independent profiler diagnostic. Explicit synchronization may be included; trace times are not formal latency samples. Missing GPU events cannot establish zero GPU work.'}


def figures(run,summary,memories,correctness,inputs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'figure.dpi':130})
    directory=run/'figures';directory.mkdir(exist_ok=True);paths=[]
    def save(fig,name):
        fig.tight_layout();fig.savefig(directory/name,bbox_inches='tight');plt.close(fig);paths.append('figures/'+name)
    main=[r for r in summary if r['coverage_group']=='main' and not r['missing_coverage']]
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
    for row,distribution in enumerate(('standard_normal','group_scaled_normal')):
        for col,bits in enumerate((4,8)):
            ax=axes[row,col]
            for operation in ('encode','decode','roundtrip'):
                values=sorted([r for r in main if r['nominal_bits']==bits and r['operation']==operation and inputs[r['input_id']]['generator_config']['distribution']==distribution],key=lambda r:r['original_numel'])
                if not values:continue
                x=[v['original_numel'] for v in values];y=np.array([v['gpu_session_median_us_median'] for v in values])
                error=np.array([[v['gpu_session_median_us_median']-v['gpu_session_median_us_min'] for v in values],
                                [v['gpu_session_median_us_max']-v['gpu_session_median_us_median'] for v in values]])
                ax.errorbar(x,y,yerr=error,marker='o',capsize=3,label=operation)
            ax.set(xscale='log',yscale='log',xlim=(200,5e6),title=f'b{bits} / {distribution}',xlabel='Logical BF16 elements',ylabel='Event interval per call (us)');ax.grid(alpha=.25);ax.legend()
    fig.suptitle('PyTorch eager baseline: median and range of 5 session medians',y=1.02)
    save(fig,'latency_vs_numel.png')
    normal=[r for r in main if r['operation']=='roundtrip' and inputs[r['input_id']]['generator_config']['distribution']=='standard_normal']
    for field,name,ylabel,title in [('effective_bits_per_logical_element','wire_effective_bits.png','Wire bits / logical element','Actual 5/9-bit payload plus hierarchical scales'),
                                  ('quant_normalized_squared_error','quantization_distortion.png','Squared error / input squared energy','Quantization distortion; CPU/GPU exact parity is a separate check')]:
        fig,ax=plt.subplots(figsize=(7,4))
        for bits in (4,8):
            values=sorted([r for r in normal if r['nominal_bits']==bits],key=lambda r:r['original_numel']);xs=[];ys=[]
            for value in values:
                if field=='effective_bits_per_logical_element':metric=value[field]
                else:
                    evidence=[c for c in correctness if c['input_id']==value['input_id'] and c['nominal_bits']==bits and c['method_id']==METHOD and c['check_type']=='roundtrip' and c['status']=='pass']
                    metric=evidence[0].get(field) if len(evidence)==1 else None
                if metric is not None:xs.append(value['original_numel']);ys.append(metric)
            ax.plot(xs,ys,marker='o',label=f'b{bits}')
        ax.set(xscale='log',xlabel='Logical BF16 elements',ylabel=ylabel,title=title+'\nStandard normal, seed 7301 only')
        if field=='quant_normalized_squared_error':ax.set_yscale('log')
        ax.grid(alpha=.25);ax.legend();save(fig,name)
    memory_map={(m['case_id'],m['method_id']):m for m in memories}
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    components=(('input_bytes','Input'),('threshold_bytes','Thresholds'),('lut_bytes','LUT'),('packet_buffer_bytes','Packet'),
                ('decoded_output_bytes','Output'),('workspace_bytes','Workspace'))
    for col,bits in enumerate((4,8)):
        values=sorted([r for r in normal if r['nominal_bits']==bits and (r['case_id'],r['method_id']) in memory_map],key=lambda r:r['original_numel'])
        positions=np.arange(len(values));bottom=np.zeros(len(values));ax=axes[0,col]
        for field,label in components:
            data=np.array([memory_map[(v['case_id'],v['method_id'])][field]['median']/2**20 for v in values])
            ax.bar(positions,data,bottom=bottom,label=label);bottom+=data
        ax.set_xticks(positions,[f"{v['original_numel']:,}" for v in values],rotation=35,ha='right')
        ax.set(ylabel='Owned/external tensor storage (MiB)',title=f'b{bits}: persistent preallocated storage')
        ax.legend(fontsize=7,ncol=2);ax.grid(axis='y',alpha=.2)
        ax=axes[1,col]
        for field,label in (('allocated_before_bytes','Allocator baseline (includes plan)'),('peak_extra_allocated_bytes','Peak extra during one call')):
            data=[memory_map[(v['case_id'],v['method_id'])][field]['median']/2**20 for v in values]
            ax.plot([v['original_numel'] for v in values],data,marker='o',label=label)
        ax.set(xscale='log',yscale='symlog',xlim=(200,5e6),xlabel='Logical BF16 elements',ylabel='PyTorch allocated memory (MiB)',title='Isolated memory probe, outside timing');ax.set_ylim(bottom=0);ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Roundtrip / standard normal / seed 7301; extra excludes persistent baseline',y=1.01)
    save(fig,'memory_vs_numel.png')
    return paths


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',required=True)
    args=parser.parse_args();run=(ROOT/args.run_dir).resolve()
    if not run.is_relative_to(ROOT/'results/gpu_codec_v1'):raise ValueError('Run must be inside the independent results tree')
    check=Checks();sources={};loaded={}
    for name in ('codec_contract.json','benchmark_config.json','input_manifest.json','correctness.jsonl','benchmark_raw.jsonl','memory.jsonl'):
        path=run/name
        if not check.require(path.exists(),'missing_input_file',path=name):continue
        try:loaded[name]=read_lines(path) if path.suffix=='.jsonl' else read(path);sources[name]=sha(path)
        except Exception as exc:check.require(False,'unreadable_input_file',path=name,error=str(exc))
    if len(loaded)!=6:
        write(run/'analysis.json',{'schema_version':'gpu_codec_v1.0','run_id':run.name,'torch_benchmark_validity':'incomplete',
            'validation_error_count':check.count,'validation_errors':check.errors,'source_sha256':sources,'derived_files':[]})
        print(json.dumps({'status':'incomplete','validation_errors':check.count}));return 2
    contract=sources['codec_contract.json'];config_hash=sources['benchmark_config.json'];config=loaded['benchmark_config.json']
    manifest=loaded['input_manifest.json'];raw=loaded['benchmark_raw.jsonl'];memory=loaded['memory.jsonl'];correctness=loaded['correctness.jsonl']
    expected=expected_cases(manifest);inputs={r['input_id']:r for r in manifest['inputs']}
    check.require(len(inputs)==len(manifest['inputs']),'duplicate_input_identity')
    for row in manifest['inputs']:
        n=row['original_numel']
        check.require(type(n) is int and n>=0 and row['padded_numel']==(n+255)//256*256,
            'noncanonical_input_length',input_id=row['input_id'])
    check.require(len(expected)==104,'expected_grid_not_104',observed=len(expected))
    check.require(len(raw)==10400,'raw_record_count',observed=len(raw),expected=10400)
    check.require(len(memory)==520,'memory_record_count',observed=len(memory),expected=520)
    check.require(config.get('expected_torch_cases')==104 and config.get('expected_torch_blocks')==10400,'config_count_mismatch')
    check.require(config.get('session_count_min')==5 and config.get('blocks_per_session_min')==20 and config.get('warmup_calls_min',0)>=50,'config_repetition_mismatch')
    for section in ('main_grid','api_grid','real_gradient_grid'):
        check.require(config.get(section,{}).get('methods')==[METHOD],'config_method_scope_mismatch',section=section)
    main_grid=config.get('main_grid',{});api_grid=config.get('api_grid',{})
    check.require(main_grid.get('numel')==[256,4096,65536,262144,1048576,4194304] and
        main_grid.get('nominal_bits')==[4,8] and main_grid.get('distributions')==['standard_normal','group_scaled_normal'] and
        main_grid.get('input_seed')==7301 and main_grid.get('operations')==['encode','decode','roundtrip'],
        'config_main_grid_mismatch')
    check.require(api_grid.get('numel')==[65536,1048576] and api_grid.get('nominal_bits')==[4,8] and
        api_grid.get('distribution')=='standard_normal' and api_grid.get('input_seed')==7301 and
        api_grid.get('validation_modes')==['trusted_inputs','checked_api'],'config_api_grid_mismatch')
    K={}
    for row in config.get('repeat_counts',[]):
        key=(row['case_id'],row['method_id'])
        check.require(key not in K,'duplicate_K_config',case_id=key[0]);K[key]=row['inner_repeats']
        check.require(type(row['inner_repeats']) is int and 1<=row['inner_repeats']<=10000,'invalid_frozen_K',case_id=key[0])
    check.require(set(K)==set(expected),'frozen_K_coverage_mismatch',missing=sorted(set(expected)-set(K)),unexpected=sorted(set(K)-set(expected)))
    pilot_path=run/'logs/K_pilot.jsonl'
    if pilot_path.exists():
        sources['logs/K_pilot.jsonl']=sha(pilot_path)
        check.require(sources['logs/K_pilot.jsonl']==config.get('pilot_sha256'),'frozen_pilot_hash_mismatch')
    groups=defaultdict(list);seen=set();memory_seen=set();valid_memory=[]
    def identity_valid(row,number,source):
        key=(row.get('case_id'),row.get('method_id'));base=expected.get(key)
        valid=check.require(base is not None,'unexpected_case',source=source,row=number,case_id=key[0])
        valid=check.require(row.get('run_id')==run.name and row.get('contract_sha256')==contract and row.get('benchmark_config_sha256')==config_hash,'identity_or_hash_mismatch',source=source,row=number) and valid
        if base:
            for field,value in base.items():valid=check.require(row.get(field)==value,'case_metadata_mismatch',source=source,row=number,field=field) and valid
        valid=check.require(type(row.get('session_id')) is int and 0<=row['session_id']<5,'invalid_session',source=source,row=number) and valid
        return valid,key
    for i,row in enumerate(raw):
        valid,key=identity_valid(row,i,'raw');joint=(*key,row.get('session_id'),row.get('block_id'))
        valid=check.require(joint not in seen,'duplicate_raw_joint_key',row=i,key=joint) and valid;seen.add(joint)
        valid=check.require(type(row.get('block_id')) is int and 0<=row['block_id']<20,'invalid_block',row=i) and valid
        valid=check.require(row.get('inner_repeats')==K.get(key) and type(row.get('inner_repeats')) is int and row['inner_repeats']>0,'K_changed_or_invalid',row=i) and valid
        valid=check.require(finite_nonnegative(row.get('gpu_elapsed_ms')) and finite_nonnegative(row.get('wall_elapsed_ms')),'invalid_elapsed_time',row=i) and valid
        valid=check.require(row.get('record_status')=='ok' and row.get('exclusion_reason') is None,'non_ok_raw_record',row=i,status=row.get('record_status')) and valid
        if key in expected:
            base=expected[key];l=base['padded_numel'];bits=base['nominal_bits'];payload=(l*(bits+1)+7)//8;scale=l//16+2*(l//256)
            valid=check.require(row.get('payload_bytes')==payload and row.get('scale_bytes')==scale and row.get('wire_bytes')==payload+scale and row.get('header_bytes')==0 and row.get('alignment_bytes')==0,'wire_formula_mismatch',row=i) and valid
        if valid:groups[key].append(row)
    for i,row in enumerate(memory):
        valid,key=identity_valid(row,i,'memory');joint=(*key,row.get('session_id'))
        valid=check.require(joint not in memory_seen,'duplicate_memory_joint_key',row=i,key=joint) and valid;memory_seen.add(joint)
        names=('input_bytes','threshold_bytes','lut_bytes','packet_buffer_bytes','decoded_output_bytes','workspace_bytes','total_plan_owned_storage_bytes',
               'allocated_before_bytes','allocated_after_bytes','peak_allocated_bytes','reserved_before_bytes','peak_reserved_bytes','peak_extra_allocated_bytes')
        valid=check.require(all(type(row.get(n)) is int and row[n]>=0 for n in names),'invalid_memory_bytes',row=i) and valid
        if valid:
            valid=check.require(row['peak_extra_allocated_bytes']==row['peak_allocated_bytes']-row['allocated_before_bytes'] and row['peak_allocated_bytes']>=row['allocated_after_bytes'] and row['peak_reserved_bytes']>=row['reserved_before_bytes'],'memory_peak_inconsistency',row=i) and valid
            valid=check.require(row['workspace_bytes']+row['packet_buffer_bytes']+row['decoded_output_bytes']==row['total_plan_owned_storage_bytes'],'memory_composition_inconsistency',row=i) and valid
            base=expected[key];n,l,b=base['original_numel'],base['padded_numel'],base['nominal_bits']
            valid=check.require(row['input_bytes']==n*2 and row['threshold_bytes']==2*l+l//8 and
                row['lut_bytes']==(2*(1<<b)+1)*2 and row['decoded_output_bytes']==l*2 and
                row['packet_buffer_bytes']==(l*(b+1)+7)//8+l//16+2*(l//256),
                'memory_component_formula_mismatch',row=i) and valid
        if valid:valid_memory.append(row)
    correctness_counts=Counter()
    for i,row in enumerate(correctness):
        check.require(row.get('run_id')==run.name and row.get('contract_sha256')==contract,'correctness_identity_mismatch',row=i)
        correctness_counts[(row.get('method_id'),row.get('status'))]+=1
    check.require(all(row.get('status')=='pass' for row in correctness) and bool(correctness),'correctness_not_all_pass')
    for input_id in inputs:
        for bits in (4,8):
            evidence=[r for r in correctness if r.get('input_id')==input_id and r.get('nominal_bits')==bits and
                r.get('method_id')==METHOD and r.get('check_type')=='roundtrip' and r.get('status')=='pass']
            check.require(len(evidence)==1,'correctness_fixture_roundtrip_coverage',input_id=input_id,bits=bits,observed=len(evidence))
    summary=[]
    for key,base in sorted(expected.items()):
        values=groups.get(key,[]);item=summarize_group(values,{**base,'run_id':run.name,'contract_sha256':contract,'benchmark_config_sha256':config_hash})
        check.require(item['valid_block_count']==100 and item['valid_session_count']==5 and not item['missing_coverage'],'missing_case_coverage',case_id=key[0],blocks=item['valid_block_count'])
        check.require(sum((r['case_id'],r['method_id'])==key for r in valid_memory)==5,'missing_memory_coverage',case_id=key[0])
        summary.append(item)
    memories=memory_summary(valid_memory);timing_error_count=check.count
    trace_attempts=[analyze_trace(run)]
    if (run/'traces/gpu_activity_manifest.json').exists():trace_attempts.append(analyze_trace(run,'gpu_activity_manifest.json'))
    for attempt in trace_attempts:
        check.require(attempt.get('status')!='invalid','trace_artifact_integrity_failure',reason=attempt.get('reason'))
        if 'trace_path' in attempt:
            check.require(attempt.get('contract_sha256')==contract and attempt.get('benchmark_config_sha256')==config_hash,
                'trace_identity_mismatch',trace=attempt['trace_path'])
            sources[attempt['trace_path']]=attempt['sha256'];sources[attempt['manifest_path']]=attempt['manifest_sha256']
    eligible=[t for t in trace_attempts if t.get('status')=='analyzed' and t.get('operation_count_matches') and
        t.get('operation_count')==3 and t.get('contract_sha256')==contract and t.get('benchmark_config_sha256')==config_hash]
    trace=eligible[-1] if eligible else trace_attempts[-1]
    trace_complete=bool(eligible) and check.count==timing_error_count
    benchmark_validity='invalid' if check.count else ('valid' if trace_complete else 'incomplete')
    summary_path=run/'benchmark_summary.jsonl'
    summary_path.write_text(''.join(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n' for row in summary),encoding='utf-8')
    generated=[]
    if not timing_error_count:generated=figures(run,summary,memories,correctness,inputs)
    analysis={'schema_version':'gpu_codec_v1.0','run_id':run.name,'stage':'G1','substage':'torch_cuda_baseline',
        'scope':'PyTorch eager GPU baseline only; full two-method G1 remains incomplete',
        'torch_timing_validity':'valid' if not timing_error_count else 'invalid',
        'torch_benchmark_validity':benchmark_validity,'trace_completeness':'complete' if trace_complete else ('invalid' if check.count>timing_error_count else 'incomplete'),
        'full_G1_benchmark_validity':'incomplete',
        'validation_error_count':check.count,'validation_errors':check.errors,'validation_errors_truncated':check.count>len(check.errors),
        'source_sha256':sources,'coverage':{'expected_cases':104,'summary_cases':len(summary),'raw_records':len(raw),
            'valid_raw_records':sum(len(v) for v in groups.values()),'memory_records':len(memory),'valid_memory_records':len(valid_memory),
            'sessions':5,'blocks_per_session':20,'missing_methods':['cuda_optimized_repaired']},
        'correctness_counts':[{'method_id':method,'status':status,'rows':count} for (method,status),count in sorted(correctness_counts.items())],
        'statistics':{'quantiles':'numpy.quantile(method=linear)','unit':'microseconds per call from elapsed_ms * 1000 / K',
            'overall':'Median, min and max of five session medians','p95':'Block-average per-call latency p95; not request tail latency',
            'throughput':'logical_numel * 2 / (overall_event_median_us * 1e-6) / 1e9',
            'noise_rule':'Any session GPU IQR/median > 0.10; slow observations are retained'},
        'noise_case_count':sum(r['noise_flag'] for r in summary),
        'noise_cases':[{'case_id':r['case_id'],'method_id':r['method_id'],'sessions':[s['session_id'] for s in r['session_statistics'] if s['noise_flag']]} for r in summary if r['noise_flag']],
        'paired_reference_method':None,'paired_speedup_by_session':{},'optimization_gain':'not_evaluated',
        'memory_case_statistics':memories,'block_duration_diagnostics':duration_diagnostics(groups,config),
        'trace':trace,'trace_attempts':trace_attempts,'figures':generated,
        'limitations':[] if trace_complete else ['Required GPU kernel/copy/synchronization trace evidence is incomplete. Formal event timing and its figures are reported independently; missing trace events are not zero counts.'],
        'figure_data_sources':{'latency_vs_numel.png':'benchmark_summary.jsonl; 5-session median range; main grid',
            'wire_effective_bits.png':'benchmark_summary.jsonl; standard_normal seed 7301 roundtrip only',
            'quantization_distortion.png':'correctness.jsonl; passing GPU roundtrip standard_normal seed 7301 only',
            'memory_vs_numel.png':'memory.jsonl; standard_normal seed 7301 preallocated roundtrip; medians across 5 sessions'},
        'reproduction_command':f'scripts/python.ps1 scripts/summarize_g1a.py --run-dir results/gpu_codec_v1/{run.name}',
        'derived_files':['benchmark_summary.jsonl','analysis.json',*generated],
        'generator_sha256':sha(__file__)}
    write(run/'analysis.json',analysis)
    print(json.dumps({'status':analysis['torch_benchmark_validity'],'timing_status':analysis['torch_timing_validity'],
        'trace_completeness':analysis['trace_completeness'],'cases':len(summary),'validation_errors':check.count,
        'noise_cases':analysis['noise_case_count'],'figures':generated}))
    return 0 if benchmark_validity=='valid' else 2


if __name__=='__main__':sys.exit(main())
