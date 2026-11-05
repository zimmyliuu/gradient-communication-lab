"""Registered, balanced paired G1b measurements; all failed observations retained."""
from __future__ import annotations
import argparse
from datetime import datetime
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
from gradient_comm_lab.cpu_reference import CpuReference,wire_size,tensor_bytes
from gradient_comm_lab.fixtures_v2 import load_fixture,sha
from gradient_comm_lab.torch_codec import TorchCodecPlan

METHODS=('torch_cuda_repaired','cuda_optimized_repaired')
BASELINE_SHA='50bc272876be54db5d42327cce144022b7bf960d82073347479da2eb53f1d794'

def plan_type(method):
    if method==METHODS[0]:return TorchCodecPlan
    from gradient_comm_lab.cuda_codec import CudaCodecPlan
    return CudaCodecPlan

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False)
def append(p,v):
    with Path(p).open('a',encoding='utf-8') as f:f.write(json.dumps(v,ensure_ascii=False,allow_nan=False)+'\n')
def now():return datetime.now().astimezone().isoformat()

def device_state():
    result={'captured_at':now(),'free_total_bytes':list(torch.cuda.mem_get_info()),
            'device_name':torch.cuda.get_device_name(),'torch_threads':torch.get_num_threads()}
    try:
        p=subprocess.run(['nvidia-smi','--query-gpu=driver_version,utilization.gpu,clocks.sm,clocks.mem,power.draw,power.limit,memory.used',
                          '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
        result['nvidia_smi']={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
    except Exception as e:result['nvidia_smi']={'error':str(e)}
    for key,args in (
        ('wddm_process_summary', ['nvidia-smi']),
        ('compute_processes', ['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'])):
        try:
            p=subprocess.run(args,capture_output=True,text=True,timeout=10)
            result[key]={'command':args,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
        except Exception as e:result[key]={'command':args,'error':str(e)}
    result['external_load_note']='Instantaneous WDDM desktop snapshot, not exclusive GPU reservation. Per-process N/A or permission-denied values are unavailable, not zero.'
    return result

def cases(manifest):
    selected=[]
    for row in manifest['inputs']:
        if row['source_kind']=='synthetic' and row['generator_config']['seed']==7301:
            for bits in (4,8):
                for op in ('encode','decode','roundtrip'):
                    selected.append((row,bits,op,'preallocated','trusted_inputs','main'))
                if row['generator_config']['distribution']=='standard_normal' and row['original_numel'] in (65536,1048576):
                    for validation in ('trusted_inputs','checked_api'):
                        selected.append((row,bits,'roundtrip','api_allocating',validation,'api'))
        elif row['source_kind']=='real_gradient':
            for bits in (4,8):selected.append((row,bits,'roundtrip','preallocated','trusted_inputs','real_gradient'))
    out=[]
    for row,bits,op,allocation,validation,group in selected:
        identity=f"{row['input_id']}__b{bits}__{op}__{allocation}__{validation}"
        out.append({'case_id':identity,'input_id':row['input_id'],'input_row':row,'nominal_bits':bits,
            'operation':op,'allocation_mode':allocation,'validation_mode':validation,'coverage_group':group})
    assert len(out)==104 and len({c['case_id'] for c in out})==104
    return out

def unique_bytes(tensors):
    found={}; visited=set()
    def visit(v):
        if id(v) in visited:return
        visited.add(id(v))
        if isinstance(v,torch.Tensor) and v.device.type=='cuda' and v.numel():
            storage=v.untyped_storage();found[(str(v.device),storage.data_ptr())]=storage.nbytes()
        elif isinstance(v,dict):
            for x in v.values():visit(x)
        elif isinstance(v,(list,tuple)):
            for x in v:visit(x)
        elif hasattr(v,'__dict__') and type(v).__module__.startswith('gradient_comm_lab'):
            for x in vars(v).values():visit(x)
    visit(tensors);return sum(found.values())

def prepare(case,run,phase,session,method):
    Plan=plan_type(method)
    start=time.perf_counter();row=case['input_row'];bits=case['nominal_bits']
    cpu=load_fixture(row);cpu_ms=(time.perf_counter()-start)*1000
    reference_start=time.perf_counter();reference=CpuReference(bits)
    expected=reference.roundtrip(cpu['input_logical'],cpu['main_thresholds'],cpu['scale_thresholds'])
    reference_ms=(time.perf_counter()-reference_start)*1000
    transfer=time.perf_counter()
    x=cpu['input_logical'].cuda();main=cpu['main_thresholds'].cuda();scale=cpu['scale_thresholds'].cuda();lut=cpu['lookup_tables'][bits].cuda()
    torch.cuda.synchronize();h2d_ms=(time.perf_counter()-transfer)*1000
    before=time.perf_counter();plan=Plan(row['original_numel'],bits,device='cuda')
    torch.cuda.synchronize();plan_ms=(time.perf_counter()-before)*1000
    output=plan.roundtrip_into(x,main,scale,lut,checked=True)
    torch.cuda.synchronize();validation_start=time.perf_counter()
    packet=plan.packet
    actual=tensor_bytes(packet.payload)+tensor_bytes(packet.scale_codes)+tensor_bytes(packet.top_scales)
    copied_output=output.cpu();d2h_ms=(time.perf_counter()-validation_start)*1000
    if actual!=expected['wire'] or not torch.equal(copied_output.view(torch.int16),expected['decoded'].view(torch.int16)):
        path=run/'logs'/f"benchmark_parity_failure_{time.time_ns()}.pt"
        torch.save({'case':case,'cpu':cpu,'expected_wire':expected['wire'],'actual_wire':actual,
                    'expected_decoded':expected['decoded'],'actual_decoded':copied_output},path)
        raise AssertionError('Benchmark preparation disagrees with CPU reference: '+str(path))
    sizes=wire_size(row['padded_numel'],bits)
    assert len(actual)==sizes['wire_bytes']
    if case['allocation_mode']=='api_allocating':
        checked=case['validation_mode']=='checked_api'
        def operation():
            current=Plan(row['original_numel'],bits,device='cuda')
            return current.roundtrip_into(x,main,scale,lut,checked=checked)
    elif case['operation']=='encode':operation=lambda:plan.encode_into(x,main,scale,lut,checked=False)
    elif case['operation']=='decode':operation=lambda:plan.decode_into(packet,lut,checked=False)
    else:operation=lambda:plan.roundtrip_into(x,main,scale,lut,checked=False)
    initialization={'method_id':method,'case_id':case['case_id'],'phase':phase,'session_id':session,'CPU_fixture_load_and_verify_ms':cpu_ms,
        'CPU_reference_ms':reference_ms,'fixture_H2D_ms':h2d_ms,'plan_prepare_ms':plan_ms,
        'D2H_validation_transfer_ms':d2h_ms,'correctness_checked_outside_timing':True,
        'packet_sha256':hashlib.sha256(actual).hexdigest(),
        'allocation_note':'Allocating mode creates an entire plan/workspace and output on every call; preallocated mode reuses a plan.'}
    append(run/'logs/benchmark_initialization.jsonl',initialization)
    del cpu,expected,reference
    return {'plan':plan,'x':x,'main':main,'scale':scale,'lut':lut,'packet':packet,'operation':operation,'sizes':sizes}

def time_block(function,K):
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    wall_start=time.perf_counter_ns();start.record()
    for _ in range(K):function()
    end.record();end.synchronize()
    wall_ms=(time.perf_counter_ns()-wall_start)/1e6
    return {'gpu_elapsed_ms':float(start.elapsed_time(end)),'wall_elapsed_ms':wall_ms}

def identity(case,run,contract,config,method):
    row=case['input_row'];return {'run_id':run.name,'contract_sha256':contract,'benchmark_config_sha256':config,
        'case_id':case['case_id'],'input_id':case['input_id'],'method_id':method,
        'nominal_bits':case['nominal_bits'],'stored_bits':case['nominal_bits']+1,
        'original_numel':row['original_numel'],'padded_numel':row['padded_numel'],
        'input_dtype':'bfloat16','output_dtype':'uint8_packet' if case['operation']=='encode' else 'bfloat16',
        'operation':case['operation'],'allocation_mode':case['allocation_mode'],'validation_mode':case['validation_mode'],
        'graph_mode':'off','compile_mode':'eager','cache_mode':'steady_state_reuse','rng_work_included':False,
        'coverage_group':case['coverage_group']}


def register(run,selected):
    protocol={'schema_version':'gpu_codec_v1.0','run_id':run.name,'created_at':now(),
        'methods':list(METHODS),'case_count':104,'session_count':5,'blocks_per_session':20,'warmup_calls':50,
        'case_seed_base':7501,'pair_seed_base':87501,'pilot_seed':97501,
        'pilot_rule':{'single_call_observations':3,'initial_K':'ceil(20 / median_single_call_ms), clamp 1..10000',
            'target_ms':20,'accept_interval_ms':[15,30],'max_actual_block_observations':4,
            'update':'ceil(K * 20 / observed_block_ms), clamp 1..10000; stop within [15,30], unchanged K, or fourth observed block',
            'selection':'K from final observed block; no fastest observation selection'},
        'comparison':'Same case/session/block_id; each method normalized by its own frozen K',
        'memory':'Separate single-method contexts measured after paired timing; not part of timed blocks',
        'timing_context':'Both implementations and their identical-input tensors resident during paired timing',
        'expected_raw_rows':20800,'expected_summary_rows':208,'expected_memory_rows':1040,
        'contract_sha256':sha(run/'codec_contract.json'),'input_manifest_sha256':sha(run/'input_manifest.json'),
        'baseline_source_sha256':BASELINE_SHA,'benchmark_source_sha256':sha(__file__)}
    write(run/'pilot_protocol.json',protocol)
    for session in range(5):
        ordered=list(selected);random.Random(7501+session).shuffle(ordered)
        pairs=[]
        for case in ordered:
            seed=int.from_bytes(hashlib.sha256(f"87501:{session}:{case['case_id']}".encode()).digest()[:8],'little')
            first=[0]*10+[1]*10;random.Random(seed).shuffle(first)
            pairs.append({'case_id':case['case_id'],'seed':seed,
                'orders':[list(METHODS) if index==0 else list(reversed(METHODS)) for index in first]})
        write(run/'logs'/f'benchmark_session_{session}_order.json',{'session_id':session,'case_seed':7501+session,
            'method_order':'Pre-generated balanced 10 AB / 10 BA per case/session',
            'case_order':[c['case_id'] for c in ordered],'paired_orders':pairs})
    orders=[{'case_id':c['case_id'],'method_id':m} for c in selected for m in METHODS]
    random.Random(97501).shuffle(orders)
    write(run/'logs/pilot_order.json',orders)
    write(run/'logs/benchmark_design_registration.json',{'protocol_sha256':sha(run/'pilot_protocol.json'),
        'order_sha256':{f'logs/benchmark_session_{s}_order.json':sha(run/'logs'/f'benchmark_session_{s}_order.json') for s in range(5)},
        'pilot_order_sha256':sha(run/'logs/pilot_order.json'),'registered_at':now()})
    print(json.dumps({'registered':True,'cases':104,'methods':list(METHODS)}),flush=True)


def memory_probe(case,run,session,method,contract,config_hash):
    context=prepare(case,run,'isolated_memory',session,method);fn=context['operation']
    for _ in range(50):fn()
    torch.cuda.synchronize()
    allocated=torch.cuda.memory_allocated();reserved=torch.cuda.memory_reserved();torch.cuda.reset_peak_memory_stats()
    fn();torch.cuda.synchronize()
    plan=context['plan'];packet=context['packet']
    owned=unique_bytes(plan);packet_bytes=unique_bytes([packet.payload,packet.scale_codes,packet.top_scales])
    output_bytes=unique_bytes(plan.output)
    record={**identity(case,run,contract,config_hash,method),'session_id':session,
        'input_bytes':context['x'].numel()*2,'threshold_bytes':context['main'].numel()*2+context['scale'].numel()*2,
        'lut_bytes':context['lut'].numel()*2,'packet_buffer_bytes':packet_bytes,'decoded_output_bytes':output_bytes,
        'workspace_bytes':owned-packet_bytes-output_bytes,'total_plan_owned_storage_bytes':owned,
        'total_named_device_storage_bytes':unique_bytes([plan,context['x'],context['main'],context['scale'],context['lut']]),
        'allocated_before_bytes':allocated,'allocated_after_bytes':torch.cuda.memory_allocated(),
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'reserved_before_bytes':reserved,
        'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'peak_extra_allocated_bytes':torch.cuda.max_memory_allocated()-allocated,
        'external_allocation_tracking':'PyTorch allocator only; context/driver/CUPTI or compiler allocations not included',
        'notes':'Only this method plan and inputs remain live in isolated probe. Baseline includes full persistent plan. API creates another whole plan each call; unique backing storages, not views, counted.'}
    append(run/'memory.jsonl',record)
    del fn,context,plan,packet;gc.collect();torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',required=True)
    parser.add_argument('--mode',choices=('register','pilot','measure'),required=True)
    args=parser.parse_args();run=(ROOT/args.run_dir).resolve()
    if not run.is_relative_to(ROOT/'results/gpu_codec_v1') or (run/'artifact_integrity.json').exists():
        raise ValueError('Requires unsealed project run')
    if sha(ROOT/'src/gradient_comm_lab/torch_codec.py')!=BASELINE_SHA:raise ValueError('Accepted baseline source changed')
    selected=cases(read(run/'input_manifest.json'));by_case={c['case_id']:c for c in selected}
    if args.mode=='register':return register(run,selected)
    registration=read(run/'execution_registration.json');contract=sha(run/'codec_contract.json')
    def assert_frozen():
        for p,h in registration['source_hashes'].items():
            if sha(ROOT/p)!=h:raise ValueError('Frozen execution source changed: '+p)
        if sha(run/'codec_contract.json')!=registration['contract_sha256']:raise ValueError('Frozen contract changed')
        if sha(run/'input_manifest.json')!=registration['input_manifest_sha256']:raise ValueError('Frozen inputs changed')
        if sha(run/'logs/cuda_build_manifest.json')!=registration['optimized_build_manifest_sha256']:raise ValueError('Frozen build manifest changed')
        build=read(run/'logs/cuda_build_manifest.json')
        if sha(build['module_path'])!=build['module_sha256']:raise ValueError('Frozen loaded CUDA artifact changed')
        design=read(run/'logs/benchmark_design_registration.json')
        if sha(run/'pilot_protocol.json')!=design['protocol_sha256']:raise ValueError('Pilot rules changed')
        for p,h in design['order_sha256'].items():
            if sha(run/p)!=h:raise ValueError('Pre-generated pair order changed')
        if sha(run/'logs/pilot_order.json')!=design['pilot_order_sha256']:raise ValueError('Pilot order changed')
        protocol=read(run/'pilot_protocol.json')
        if protocol['benchmark_source_sha256']!=sha(__file__):raise ValueError('Harness changed after design registration')
        if protocol['contract_sha256']!=contract:raise ValueError('Pilot contract differs')
    assert_frozen()
    acceptance=read(run/'correctness_acceptance.json')
    for method in METHODS:
        status=acceptance.get('method_statuses',{}).get(method,{})
        if isinstance(status,dict):status=status.get('correctness_status')
        if status!='pass':raise ValueError('Current-run correctness missing: '+method)
    if acceptance['contract_sha256']!=contract:raise ValueError('Correctness contract mismatch')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True);torch.utils.deterministic.fill_uninitialized_memory=False
    # Extension loads once here, outside any plan initialization, memory probe or timing.
    from gradient_comm_lab.cuda_codec import load_extension,set_build_manifest
    set_build_manifest(run/'logs/cuda_build_manifest.json')
    load_extension()
    protocol=read(run/'pilot_protocol.json');benchmark_path=run/'benchmark_config.json'
    if args.mode=='pilot':
        if benchmark_path.exists() or (run/'logs/K_pilot.jsonl').exists():raise FileExistsError('Pilot evidence exists; preserve attempt')
        config=read(ROOT/'handoff_templates/benchmark_config.json')
        config.update(template_only=False,run_id=run.name,scope='G1b_paired_fixed_local_codec',method_comparison_performed=True,
            quantile_implementation='numpy.quantile(method=linear)',
            wall_timing_boundary='synchronize before timer; start-event record, K calls, end-event record/synchronization inside wall timer',
            K_selection=protocol['pilot_rule'],pilot_protocol_sha256=sha(run/'pilot_protocol.json'),
            design_registration_sha256=sha(run/'logs/benchmark_design_registration.json'),
            outer_padding_included=True,api_allocation_boundary='both methods construct complete plan/workspace plus roundtrip each call',
            deterministic_algorithms=True,fill_uninitialized_memory=False,
            gpu_timing_interpretation='CUDA event interval includes eager launch gaps, checked synchronization and allocator gaps; not kernel duration sum',
            expected_cases=104,expected_methods=2,expected_raw_blocks=20800,expected_summary_rows=208,expected_memory_rows=1040,
            baseline_source_sha256=BASELINE_SHA,measurement_pairing=protocol['comparison'],timing_residency=protocol['timing_context'])
        config['repeat_counts']=[]
        append(run/'logs/benchmark_device_states.jsonl',{'phase':'pilot','state':device_state()})
        orders=read(run/'logs/pilot_order.json')
        for index,pair in enumerate(orders):
            case=by_case[pair['case_id']];method=pair['method_id'];context=prepare(case,run,'pilot',None,method);fn=context['operation']
            for _ in range(50):fn()
            singles=[time_block(fn,1) for _ in range(3)];median=sorted(v['gpu_elapsed_ms'] for v in singles)[1]
            K=max(1,min(10000,math.ceil(20/max(median,1e-9))));observations=[]
            for attempt in range(4):
                measured=time_block(fn,K);observations.append({'inner_repeats':K,**measured})
                elapsed=measured['gpu_elapsed_ms'];newK=max(1,min(10000,math.ceil(K*20/max(elapsed,1e-9))))
                if 15<=elapsed<=30 or newK==K or attempt==3:break
                K=newK
            record={**pair,'single_call_pilots':singles,'initial_selection_median_ms':median,
                'actual_block_observations':observations,'inner_repeats':K,'reason':'last_observed_K_from_registered_bounded_adjustment'}
            append(run/'logs/K_pilot.jsonl',record);config['repeat_counts'].append({k:record[k] for k in ('case_id','method_id','inner_repeats','reason')})
            if index%10==0:print(json.dumps({'pilot_method_cases_done':index+1,'total':208}),flush=True)
            del fn,context;gc.collect();torch.cuda.empty_cache()
        config['K_frozen_at']=now();config['pilot_sha256']=sha(run/'logs/K_pilot.jsonl');write(benchmark_path,config)
        print(json.dumps({'pilot':'complete','method_cases':208,'benchmark_config_sha256':sha(benchmark_path)}),flush=True)
        return
    config=read(benchmark_path);config_hash=sha(benchmark_path)
    K_by_pair={(r['case_id'],r['method_id']):r['inner_repeats'] for r in config['repeat_counts']}
    if config['pilot_sha256']!=sha(run/'logs/K_pilot.jsonl'):raise ValueError('Frozen pilot changed')
    raw=run/'benchmark_raw.jsonl';memory=run/'memory.jsonl'
    if raw.exists() or memory.exists():raise FileExistsError('Retain previous measurement attempt; no overwrite')
    started=time.perf_counter()
    try:
        for session in range(5):
            assert_frozen();design=read(run/'logs'/f'benchmark_session_{session}_order.json')
            pair_orders={r['case_id']:r['orders'] for r in design['paired_orders']}
            append(run/'logs/benchmark_device_states.jsonl',{'phase':'measurement','session_id':session,'state':device_state()})
            for index,case_id in enumerate(design['case_order']):
                case=by_case[case_id];contexts={m:prepare(case,run,'paired_measurement',session,m) for m in METHODS}
                for method in METHODS:
                    for _ in range(50):contexts[method]['operation']()
                torch.cuda.synchronize()
                for block,method_order in enumerate(pair_orders[case_id]):
                    for position,method in enumerate(method_order):
                        context=contexts[method];K=K_by_pair[(case_id,method)]
                        measurement=time_block(context['operation'],K)
                        append(raw,{**identity(case,run,contract,config_hash,method),'session_id':session,'block_id':block,
                            'paired_block_id':f'{case_id}__s{session}__p{block:02d}','pair_position':position,'pair_order':method_order,
                            'inner_repeats':K,**measurement,**context['sizes'],'header_bytes':0,'alignment_bytes':0,
                            'record_status':'ok','exclusion_reason':None})
                del context,contexts;gc.collect();torch.cuda.empty_cache()
                for method in METHODS:memory_probe(case,run,session,method,contract,config_hash)
                if index%10==0:print(json.dumps({'session':session,'paired_cases_done':index+1,'total':104,
                    'elapsed_seconds':time.perf_counter()-started}),flush=True)
            append(run/'logs/benchmark_device_states.jsonl',{'phase':'session_end','session_id':session,'state':device_state()})
        assert_frozen()
        if sha(benchmark_path)!=config_hash:raise ValueError('Frozen benchmark config changed during measurement')
        write(run/'logs/benchmark_execution.json',{'status':'complete','finished_at':now(),'case_count':104,'method_count':2,
            'session_count':5,'blocks_per_session':20,'raw_blocks':20800,'memory_rows':1040,
            'elapsed_seconds':time.perf_counter()-started,'benchmark_config_sha256':config_hash,'raw_sha256':sha(raw),
            'no_profiler_during_measurement':True,'timing_residency':protocol['timing_context']})
        print(json.dumps({'measurement':'complete','raw_blocks':20800}),flush=True)
    except Exception as exc:
        import traceback
        append(run/'logs/benchmark_failure.jsonl',{'time':now(),'exception_type':type(exc).__name__,'message':str(exc),
            'traceback':traceback.format_exc(),'elapsed_seconds':time.perf_counter()-started,
            'raw_records_retained':sum(1 for _ in raw.open()) if raw.exists() else 0})
        raise

if __name__=='__main__':main()
