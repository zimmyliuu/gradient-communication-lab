"""Execute the frozen development matrix once; retain packets and decoded tensors.

Resource windows are conservative wall time including CUDA transfers/checks, not
kernel latency benchmarks. No speed claim is made from them.
"""
import argparse,hashlib,json,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
import torch
from gradient_comm_lab.bounded_discovery import official_codec as official,uniform_codec as uniform
from gradient_comm_lab.bounded_discovery.common import now,sha,read,dump,identity,tensor_bytes,seed_for,local_metrics,aggregate_metrics

def append(path,row):
    with path.open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',required=True);args=ap.parse_args();run=(ROOT/args.run_dir).resolve()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    contract=read(run/'experiment_contract.json');manifest=read(run/'input_manifest.json');sources=read(run/'source_manifest.json')
    qualification=read(run/'baseline_qualification.json');assert qualification['gate_decision']=='pass'
    ch=sha(run/'experiment_contract.json');ih=sha(run/'input_manifest.json')
    assert ih==contract['input_manifest_sha256'] and sha(run/'source_manifest.json')==contract['source_manifest_sha256']
    assert sha(run/'baseline_qualification.json')==contract['qualification_sha256']
    assert sha(run/'logs/official_build_a03.json')==sources['build_manifest_sha256']
    for r in sources['files']:
        assert sha(Path(r['original_path']))==r['sha256'] and sha(run/r['path'])==r['sha256'],r
    assert sha(sources['binary']['path'])==sources['binary']['sha256']
    assert not any((run/p).exists() for p in ['metrics_raw.jsonl','packet_accounting.jsonl','correctness.jsonl','logs/d1_execution.json'])
    for name in ['packets','decoded']:(run/name).mkdir(exist_ok=False)
    official.set_build_manifest(run/'logs/official_build_a03.json')
    actual_module=official.build_identity()
    assert actual_module['module_sha256']==sources['binary']['sha256'] and Path(actual_module['module_path'])==Path(sources['binary']['path'])
    assert actual_module['build_manifest_sha256']==sources['build_manifest_sha256']
    began=now();wall_start=time.perf_counter();gpu_wall=0.0;failed=0;local_count=0;aggregate_count=0;seeds=set();normal_completion=False
    prior_gpu_upper=qualification['resource_usage']['wall_window_seconds']+read(run/'logs/budget_selection.json')['freeze_gpu_wall_seconds']+contract.get('previous_execution_GPU_upper_bound_seconds',0)
    common=dict(run_id=run.name,contract_sha256=ch,input_manifest_sha256=ih)
    audit=dict(started_at=began,contract_sha256=ch,input_manifest_sha256=ih,method_count=len(contract['methods']),
        resource_measure='CUDA operation synchronized wall windows including transfers and GPU validation; upper bound, not performance timing',
        cpu_num_threads=4,actual_loaded_module=actual_module,prior_GPU_upper_bound_seconds=prior_gpu_upper,
        prior_GPU_upper_bound_definition='Entire D0 wall window plus CUDA budget-selection and previous revision execution windows; conservative, includes CPU/build idle time',
        raw_paths=['metrics_raw.jsonl','packet_accounting.jsonl','correctness.jsonl'])
    dump(run/'logs/d1_start.json',audit)
    # D0 synthetic evidence was generated before the comparison contract existed.
    d0=read(run/'logs/d0_execution.json')
    for typ,evidence in [('reference_agreement','logs/d0_execution.json'),('repeatability','logs/d0_execution.json'),
        ('tail_handling','logs/d0_execution.json'),('zero_case','logs/d0_execution.json'),('pack/unpack','logs/uniform_selfcheck_a02.json')]:
        append(run/'correctness.jsonl',dict(**common,check_id='d0_evidence_'+typ.replace('/','_'),method_id='D0_applicable_methods',input_id=None,
            check_type=typ,status='pass',reference_id='pre-contract D0 qualification',difference_measure=None,
            expected_behavior='Preserve official reference tolerances or uniform exact format contract as recorded',
            observed_behavior='Qualified D0 evidence, not a post-contract rerun',evidence_path=evidence,failure_reason=None))
    try:
        for snapshot in sorted({r['snapshot_id'] for r in manifest['inputs']}):
            entries=sorted([r for r in manifest['inputs'] if r['snapshot_id']==snapshot],key=lambda r:r['worker_id'])
            assert [r['worker_id'] for r in entries]==list(range(4))
            xs=[]
            for entry in entries:
                assert entry['already_divided_by_world_size'] is True and entry['partition']=='development'
                assert sha(entry['fixture_path'])==entry['fixture_sha256']
                x=torch.load(entry['fixture_path'],map_location='cpu',weights_only=False)['input_logical'].contiguous().clone()
                assert identity(x)==entry['input_identity'] and x.numel()==818048
                xs.append(x)
            for method in contract['methods']:
                mid=method['method_id'];family=method['family']
                for repeat in range(method['repeats']):
                    ys=[];packet_ids=[];sizes=[];state_ids=[]
                    for entry,x in zip(entries,xs):
                        if prior_gpu_upper+gpu_wall>=7200:raise RuntimeError('Cumulative conservative GPU cap reached')
                        iid=entry['input_id'];pid=f"{iid}__{mid}__r{repeat:02d}";seed=None if method['deterministic'] else seed_for(iid,mid,repeat)
                        if seed is not None:assert seed not in seeds;seeds.add(seed)
                        rng_id='deterministic' if seed is None else f'{mid}/{iid}/r{repeat:02d}/seed{seed}'
                        base=dict(**common,record_type='local',snapshot_id=snapshot,worker_id=entry['worker_id'],method_id=mid,repeat_id=repeat,
                            input_id=iid,rng_state_id=rng_id,deterministic=method['deterministic'],budget_role=method['budget_role'])
                        wire=None;y=None;y_independent=None
                        try:
                            before=identity(x)
                            if family=='official_dynamiq_fixed_codec':
                                torch.cuda.synchronize();gpu_start=time.perf_counter()
                                try:
                                    xg=x.cuda();pool=official.make_random_pool(method['padded_numel'],seed);pool_hash=identity(pool)['sha256']
                                    p=official.encode(xg,method['stored_code_bits'],rand_pool=pool);wire=p.to_bytes();y=official.decode(p).cpu()
                                    restored=official.OfficialPacket.from_bytes(wire,logical_numel=x.numel(),bits=method['stored_code_bits'])
                                    y_independent=official.decode(restored).cpu()
                                    assert identity(xg)==before and identity(pool)['sha256']==pool_hash
                                    account=p.accounting();metadata=dict(seed=seed,pool_sha256=pool_hash,rng_recipe=method['rng_mode'],draw_count=method['padded_numel'])
                                    account['component_shapes_dtypes']=dict(payload=dict(shape=[account['payload_bytes']],dtype='uint8',layout='interleaved bit-packed codes'),
                                        group_scales=dict(shape=[method['padded_numel']//16],dtype='uint8'),
                                        super_scale_units=dict(shape=[method['padded_numel']//256],dtype='bfloat16'),header=dict(shape=[0],dtype='uint8'))
                                finally:
                                    torch.cuda.synchronize();gpu_wall+=time.perf_counter()-gpu_start
                            elif family=='block_uniform':
                                p=uniform.encode(x,method['stored_code_bits'],method['group_size'],seed);wire=p.to_bytes()
                                y=uniform.decode(p);y_independent=uniform.decode(bytes(bytearray(wire)));account=p.accounting();metadata=p.metadata
                            elif family=='bf16_identity':
                                wire=tensor_bytes(x);y=torch.frombuffer(bytearray(wire),dtype=torch.bfloat16).clone();y_independent=y.clone()
                                account=dict(payload_bytes=len(wire),scale_bytes=0,header_bytes=0,alignment_bytes=0,packet_total_bytes=len(wire),
                                    per_input_auxiliary_bytes=0,setup_or_persistent_state_bytes=0,
                                    component_shapes_dtypes=dict(payload=dict(shape=[x.numel()],dtype='bfloat16')),
                                    byte_measurement_method='len actual raw BF16 bytes; torch.save wrapper excluded');metadata=dict(rng_mode='deterministic')
                            else:raise ValueError(family)
                            assert identity(x)==before,'input changed'
                            assert torch.equal(y,y_independent),'independent decode mismatch'
                            assert y.shape==x.shape and y.dtype==x.dtype
                            assert len(wire)==account['packet_total_bytes']==method['packet_total_bytes']
                            assert sum(account[k] for k in ['payload_bytes','scale_bytes','header_bytes','alignment_bytes','per_input_auxiliary_bytes'])==len(wire)
                            metrics=local_metrics(x,y);assert metrics['status']=='pass',metrics
                            if family=='bf16_identity':assert metrics['squared_error']==0
                            packet_path=run/'packets'/f'{pid}.bin';packet_path.write_bytes(wire)
                            output_path=run/'decoded'/f'{pid}.pt';torch.save(y,output_path)
                            reconstruction=dict(seed=seed,metadata=metadata,packet_path=packet_path.relative_to(run).as_posix(),
                                packet_sha256=hashlib.sha256(wire).hexdigest(),decoded_output_path=output_path.relative_to(run).as_posix(),
                                decoded_output_sha256=sha(output_path),decoded_tensor_identity=identity(y),
                                native_static_configuration=dict(bits=method['stored_code_bits'],group_size=method['group_size'],strategy=3 if family=='official_dynamiq_fixed_codec' else None,
                                    logical_numel=x.numel(),padded_numel=method['padded_numel']))
                            account.update(common);account.update(reconstruction)
                            account.update(packet_id=pid,snapshot_id=snapshot,input_id=iid,worker_id=entry['worker_id'],
                                method_id=mid,repeat_id=repeat,logical_numel=x.numel(),padded_numel=method['padded_numel'],
                                effective_bits_per_logical_element=8*len(wire)/x.numel(),independent_decode_pass=True)
                            append(run/'packet_accounting.jsonl',account)
                            append(run/'metrics_raw.jsonl',dict(**base,**metrics,packet_id=pid,packet_total_bytes=len(wire),
                                decoded_output_path=reconstruction['decoded_output_path'],decoded_output_sha256=reconstruction['decoded_output_sha256']))
                            for typ in ['independent_decode','input_immutability','pack/unpack','tail_handling']:
                                append(run/'correctness.jsonl',dict(**common,check_id=f'{pid}/{typ}',method_id=mid,input_id=iid,check_type=typ,status='pass',
                                    reference_id=pid,difference_measure=0,expected_behavior='Exact byte length and shape; original input preserved; decoded wire equals packet output',
                                    observed_behavior='All asserted invariants passed; logical region only',evidence_path=reconstruction['packet_path'],failure_reason=None))
                            ys.append(y);sizes.append(len(wire));packet_ids.append(pid);state_ids.append(rng_id);local_count+=1
                        except Exception as exc:
                            fail_dir=run/'failed_cases'/pid;fail_dir.mkdir(parents=True,exist_ok=False)
                            torch.save(x,fail_dir/'input.pt')
                            if wire is not None:(fail_dir/'packet.bin').write_bytes(wire)
                            for label,value in [('decoded',y),('independent_decoded',y_independent)]:
                                if value is not None:torch.save(value,fail_dir/f'{label}.pt')
                            dump(fail_dir/'failure.json',dict(**base,seed=seed,packet_id=pid,failure_reason=repr(exc),
                                source_fixture=entry['fixture_path'],source_fixture_sha256=entry['fixture_sha256'],
                                captured_artifacts=[p.name for p in fail_dir.iterdir()]))
                            failed+=1;append(run/'metrics_raw.jsonl',dict(**base,status='fail',failure_reason=repr(exc),traceback=traceback.format_exc()))
                            append(run/'correctness.jsonl',dict(**common,check_id=f'{pid}/failed',method_id=mid,input_id=iid,check_type='execution',status='fail',
                                reference_id=pid,difference_measure=None,expected_behavior='All frozen invariants',observed_behavior=repr(exc),evidence_path='metrics_raw.jsonl',failure_reason=repr(exc)))
                            append(run/'issues.jsonl',dict(stage='D1',packet_id=pid,status='unresolved',failure_reason=repr(exc)))
                        if prior_gpu_upper+gpu_wall>=7200:raise RuntimeError('Cumulative conservative GPU cap reached')
                    ab=dict(**common,record_type='one_shot_codec_sum',snapshot_id=snapshot,worker_id=None,method_id=mid,repeat_id=repeat,
                        input_ids=[r['input_id'] for r in entries],rng_state_id=state_ids,packet_ids=packet_ids,deterministic=method['deterministic'],budget_role=method['budget_role'])
                    if len(ys)==4:
                        metrics=aggregate_metrics(xs,ys,sizes)
                        if family=='bf16_identity':assert metrics['aggregate_squared_error']==0
                        append(run/'metrics_raw.jsonl',dict(**ab,**metrics,aggregate_comparability_status='valid' if metrics['status']=='pass' else 'invalid'))
                        if metrics['status']=='pass':aggregate_count+=1
                        else:failed+=1
                    else:append(run/'metrics_raw.jsonl',dict(**ab,status='fail',failure_reason='Missing successful worker packets',aggregate_comparability_status='incomplete'));failed+=1
                    if prior_gpu_upper+gpu_wall>7200:raise RuntimeError('D1 conservative GPU window exhausted two-hour cap')
                print(json.dumps(dict(snapshot=snapshot,method_id=mid,successful_local=local_count,successful_aggregate=aggregate_count,failed_records=failed,gpu_window_seconds=gpu_wall)),flush=True)
        normal_completion=True
    except Exception as exc:
        dump(run/'logs/d1_outer_failure.json',dict(status='fail',failure_reason=repr(exc),traceback=traceback.format_exc()))
        raise
    finally:
        expected_local=len(manifest['inputs'])*sum(m['repeats'] for m in contract['methods'])
        expected_aggregate=len({r['snapshot_id'] for r in manifest['inputs']})*sum(m['repeats'] for m in contract['methods'])
        audit.update(finished_at=now(),wall_seconds=time.perf_counter()-wall_start,gpu_execution_upper_bound_seconds=gpu_wall,
            cumulative_GPU_upper_bound_seconds=prior_gpu_upper+gpu_wall,
            successful_local=local_count,successful_aggregate=aggregate_count,failed_records=failed,distinct_random_seeds=len(seeds),
            expected_local=expected_local,expected_aggregate=expected_aggregate,normal_completion=normal_completion,
            execution_complete=normal_completion and local_count==expected_local and aggregate_count==expected_aggregate and failed==0)
        dump(run/'logs/d1_execution.json',audit)
    print(json.dumps(audit),flush=True)
if __name__=='__main__':main()
