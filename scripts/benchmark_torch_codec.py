"""Five-session G1a eager-CUDA measurements; no cross-method speedup claim."""
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

METHOD='torch_cuda_repaired'

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
    found={}
    def visit(v):
        if isinstance(v,torch.Tensor) and v.device.type=='cuda' and v.numel():
            storage=v.untyped_storage();found[(str(v.device),storage.data_ptr())]=storage.nbytes()
        elif isinstance(v,dict):
            for x in v.values():visit(x)
        elif isinstance(v,(list,tuple)):
            for x in v:visit(x)
        elif hasattr(v,'__dict__'):
            for x in vars(v).values():visit(x)
    visit(tensors);return sum(found.values())

def prepare(case,run,phase,session):
    start=time.perf_counter();row=case['input_row'];bits=case['nominal_bits']
    cpu=load_fixture(row);cpu_ms=(time.perf_counter()-start)*1000
    reference_start=time.perf_counter();reference=CpuReference(bits)
    expected=reference.roundtrip(cpu['input_logical'],cpu['main_thresholds'],cpu['scale_thresholds'])
    reference_ms=(time.perf_counter()-reference_start)*1000
    transfer=time.perf_counter()
    x=cpu['input_logical'].cuda();main=cpu['main_thresholds'].cuda();scale=cpu['scale_thresholds'].cuda();lut=cpu['lookup_tables'][bits].cuda()
    torch.cuda.synchronize();h2d_ms=(time.perf_counter()-transfer)*1000
    before=time.perf_counter();plan=TorchCodecPlan(row['original_numel'],bits,device='cuda')
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
            current=TorchCodecPlan(row['original_numel'],bits,device='cuda')
            return current.roundtrip_into(x,main,scale,lut,checked=checked)
    elif case['operation']=='encode':operation=lambda:plan.encode_into(x,main,scale,lut,checked=False)
    elif case['operation']=='decode':operation=lambda:plan.decode_into(packet,lut,checked=False)
    else:operation=lambda:plan.roundtrip_into(x,main,scale,lut,checked=False)
    initialization={'case_id':case['case_id'],'phase':phase,'session_id':session,'CPU_fixture_load_and_verify_ms':cpu_ms,
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

def identity(case,run,contract,config):
    row=case['input_row'];return {'run_id':run.name,'contract_sha256':contract,'benchmark_config_sha256':config,
        'case_id':case['case_id'],'input_id':case['input_id'],'method_id':METHOD,
        'nominal_bits':case['nominal_bits'],'stored_bits':case['nominal_bits']+1,
        'original_numel':row['original_numel'],'padded_numel':row['padded_numel'],
        'input_dtype':'bfloat16','output_dtype':'uint8_packet' if case['operation']=='encode' else 'bfloat16',
        'operation':case['operation'],'allocation_mode':case['allocation_mode'],'validation_mode':case['validation_mode'],
        'graph_mode':'off','compile_mode':'eager','cache_mode':'steady_state_reuse','rng_work_included':False,
        'coverage_group':case['coverage_group']}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--mode',choices=('pilot','measure','profile'),required=True)
    args=parser.parse_args();run=(ROOT/args.run_dir).resolve()
    if not run.is_relative_to(ROOT/'results/gpu_codec_v1'):raise ValueError('Invalid independent run')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    # Every trusted workspace is written before read; sentinel tails use zeros.
    # Avoid counting deterministic-debug fills as part of allocating API cost.
    torch.utils.deterministic.fill_uninitialized_memory=False
    contract=sha(run/'codec_contract.json');manifest=read(run/'input_manifest.json');selected=cases(manifest)
    registration=read(run/'execution_registration.json')
    def assert_frozen():
        for p,h in registration['source_hashes'].items():
            if sha(ROOT/p)!=h:raise ValueError('Frozen execution source changed: '+p)
        if sha(run/'codec_contract.json')!=registration['contract_sha256']:raise ValueError('Frozen contract changed')
        if sha(run/'input_manifest.json')!=registration['input_manifest_sha256']:raise ValueError('Frozen inputs changed')
    assert_frozen()
    acceptance=read(run/'correctness_acceptance.json')
    if acceptance.get('method_status')!='pass':raise ValueError('GPU method correctness must pass before measurement')
    if acceptance['contract_sha256']!=contract:raise ValueError('Correctness contract mismatch')
    benchmark_path=run/'benchmark_config.json'
    if args.mode=='pilot':
        if benchmark_path.exists():raise FileExistsError('K configuration is already frozen')
        if (run/'logs/K_pilot.jsonl').exists():raise FileExistsError('Preserve existing pilot; explicit recovery required')
        config=read(ROOT/'handoff_templates/benchmark_config.json')
        config.update(template_only=False,run_id=run.name,scope='G1a_torch_cuda_baseline_only',
            required_but_missing_methods=['cuda_optimized_repaired'],method_comparison_performed=False,
            quantile_implementation='numpy.quantile(method=linear)',
            wall_timing_boundary='synchronize before timer; events and K calls and end-event synchronization inside wall timer',
            K_selection='median of 3 single-call pilot intervals, ceil(20ms/median), clamp1..10000; one selected-K observation; no best-run selection',
            outer_padding_included=True,api_allocation_boundary='construct full plan/workspace and roundtrip per call',
            deterministic_algorithms=True,fill_uninitialized_memory=False,
            gpu_timing_interpretation='CUDA event stream interval including eager launch gaps, checked synchronization and allocation-path gaps; not summed kernel-active time',
            expected_torch_cases=104,expected_torch_blocks=10400)
        for name in ('main_grid','api_grid','real_gradient_grid'):config[name]['methods']=[METHOD]
        config['repeat_counts']=[]
        append(run/'logs/benchmark_device_states.jsonl',{'phase':'pilot','state':device_state()})
        for index,case in enumerate(selected):
            context=prepare(case,run,'pilot',None);fn=context['operation']
            for _ in range(50):fn()
            singles=[time_block(fn,1) for _ in range(3)]
            median=sorted(v['gpu_elapsed_ms'] for v in singles)[1]
            K=max(1,min(10000,math.ceil(20/max(median,1e-9))))
            observed=time_block(fn,K)
            reason='slow_single_call_K1' if median>50 else 'target_20ms_from_median_single_call'
            record={'case_id':case['case_id'],'method_id':METHOD,'single_call_pilots':singles,
                'selection_median_ms':median,'inner_repeats':K,'reason':reason,'selected_K_observation':observed}
            append(run/'logs/K_pilot.jsonl',record)
            config['repeat_counts'].append({k:record[k] for k in ('case_id','method_id','inner_repeats','reason')})
            if index%10==0:print(json.dumps({'pilot_cases_done':index+1,'total':len(selected)}),flush=True)
            del fn,context;gc.collect();torch.cuda.empty_cache()
        config['K_frozen_at']=now();config['pilot_sha256']=sha(run/'logs/K_pilot.jsonl')
        write(benchmark_path,config)
        print(json.dumps({'pilot':'complete','cases':len(selected),'benchmark_config_sha256':sha(benchmark_path)}),flush=True)
        return
    config=read(benchmark_path);config_hash=sha(benchmark_path)
    K_by_case={r['case_id']:r['inner_repeats'] for r in config['repeat_counts']}
    if args.mode=='measure':
        raw=run/'benchmark_raw.jsonl';memory=run/'memory.jsonl'
        if raw.exists() or memory.exists():raise FileExistsError('Retain previous measurement attempts; no silent overwrite')
        started=time.perf_counter()
        for session in range(5):
            assert_frozen()
            order=list(selected);random.Random(7501+session).shuffle(order)
            append(run/'logs/benchmark_device_states.jsonl',{'phase':'measurement','session_id':session,'state':device_state()})
            write(run/'logs'/f'benchmark_session_{session}_order.json',{'session_id':session,'seed':7501+session,
                'methods':[METHOD],'method_randomization':'single method only; no between-method comparison',
                'case_order':[c['case_id'] for c in order]})
            for index,case in enumerate(order):
                context=prepare(case,run,'measurement',session);fn=context['operation'];base=identity(case,run,contract,config_hash)
                for _ in range(50):fn()
                torch.cuda.synchronize()
                # Independently measure allocations outside any timed block.
                allocated=torch.cuda.memory_allocated();reserved=torch.cuda.memory_reserved();torch.cuda.reset_peak_memory_stats()
                fn();torch.cuda.synchronize()
                plan=context['plan'];packet=context['packet']
                owned=unique_bytes(plan);packet_bytes=unique_bytes([packet.payload,packet.scale_codes,packet.top_scales])
                output_bytes=unique_bytes(plan.output)
                append(memory,{**base,'session_id':session,
                    'input_bytes':context['x'].numel()*2,'threshold_bytes':context['main'].numel()*2+context['scale'].numel()*2,
                    'lut_bytes':context['lut'].numel()*2,'packet_buffer_bytes':packet_bytes,'decoded_output_bytes':output_bytes,
                    'workspace_bytes':owned-packet_bytes-output_bytes,'total_plan_owned_storage_bytes':owned,
                    'allocated_before_bytes':allocated,'allocated_after_bytes':torch.cuda.memory_allocated(),
                    'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'reserved_before_bytes':reserved,
                    'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                    'peak_extra_allocated_bytes':torch.cuda.max_memory_allocated()-allocated,
                    'external_allocation_tracking':'PyTorch allocator only; CUDA context/driver allocations not included',
                    'notes':'Base includes input, thresholds, LUT and one persistent plan; allocating mode creates an additional plan during each call. Unique storages, not views, counted. Memory probe outside timing.'})
                for block in range(20):
                    measurement=time_block(fn,K_by_case[case['case_id']])
                    append(raw,{**base,'session_id':session,'block_id':block,'inner_repeats':K_by_case[case['case_id']],
                        **measurement,**context['sizes'],'header_bytes':0,'alignment_bytes':0,
                        'record_status':'ok','exclusion_reason':None})
                if index%10==0:print(json.dumps({'session':session,'cases_done':index+1,'total':len(order),
                    'elapsed_seconds':time.perf_counter()-started}),flush=True)
                del fn,context,plan,packet;gc.collect();torch.cuda.empty_cache()
            append(run/'logs/benchmark_device_states.jsonl',{'phase':'session_end','session_id':session,'state':device_state()})
        write(run/'logs/benchmark_execution.json',{'status':'complete','finished_at':now(),
            'case_count':104,'session_count':5,'blocks_per_session':20,'raw_blocks':10400,
            'elapsed_seconds':time.perf_counter()-started,'benchmark_config_sha256':config_hash,'raw_sha256':sha(raw),
            'memory_rows':520,'only_method':METHOD,'no_profiler_during_measurement':True})
        print(json.dumps({'measurement':'complete','raw_blocks':10400}),flush=True)
    else:
        target=next(c for c in selected if c['coverage_group']=='main' and c['operation']=='roundtrip' and
            c['nominal_bits']==4 and c['input_row']['original_numel']==65536 and
            c['input_row']['generator_config']['distribution']=='standard_normal')
        directory=run/'traces';directory.mkdir(exist_ok=True)
        trace=directory/'torch_cuda_roundtrip_b4_n65536.json'
        if trace.exists():raise FileExistsError('Profiler trace already exists')
        context=prepare(target,run,'profile',None);fn=context['operation']
        for _ in range(50):fn()
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,profile_memory=True,with_stack=False) as prof:
            for _ in range(3):
                with torch.profiler.record_function('trusted_preallocated_roundtrip'):fn()
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(trace))
        (directory/'torch_cuda_roundtrip_table.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=60),encoding='utf-8')
        write(directory/'manifest.json',{'case_id':target['case_id'],'method_id':METHOD,'operation_count':3,
            'trace':trace.name,'sha256':sha(trace),'bytes':trace.stat().st_size,'independent_from_formal_timing':True,
            'boundary':'fixture, H2D, correctness, warmup outside trace; one final explicit cuda synchronization retained',
            'contract_sha256':contract,'benchmark_config_sha256':config_hash})
        print(json.dumps({'profile':'complete','trace':str(trace)}),flush=True)


if __name__=='__main__':main()
