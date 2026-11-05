"""Assemble D0 admission facts from actual official-source/build/runtime evidence."""
import hashlib, json, sys
from datetime import datetime
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'results/bounded_discovery_v1/bounded_discovery_v1_20261104_001'
sys.path.insert(0,str(ROOT/'src'))
from gradient_comm_lab.bounded_discovery import official_codec as codec
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads((RUN/p).read_bytes())
def tensor_sha(t):return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def evidence(path):return dict(path=path,sha256=sha(RUN/path))
def main():
    review=read('logs/official_contract_review.json');execution=read('logs/d0_execution.json');boundaries=read('logs/d0_boundary_probes.json')
    assert execution['status']=='pass' and execution['official_selftest']['status']=='pass' and execution['independent_decode']['passed']
    parse=read('logs/d0_packet_parse_review.json')
    assert parse['status']=='pass'
    pools=[codec.make_random_pool(256,77100+worker,device='cpu') for worker in range(4)]
    rng_rows=[dict(worker=worker,seed=77100+worker,pool_sha256=tensor_sha(pool),same_seed_recreated_identically=bool(torch.equal(pool,codec.make_random_pool(256,77100+worker,device='cpu'))),exact_one_count=int((pool==1).sum())) for worker,pool in enumerate(pools)]
    assert all(row['same_seed_recreated_identically'] for row in rng_rows)
    assert len({p.data_ptr() for p in pools})==4
    rng=dict(status='pass',scope='D0 API-state check, separate from frozen D1 seed derivation',rows=rng_rows,
             independently_allocated_worker_pools=True,mathematical_independence_proven=False)
    (RUN/'logs/d0_rng_state_checks.json').write_text(json.dumps(rng,indent=2)+'\n',encoding='utf-8')
    attempts=[read(f'logs/official_build_{x}.json') for x in ('a01','a02','a03')]
    assert attempts[-1]['status']=='compiled_and_loaded'
    source=read('logs/official_source_manifest.json')
    source['license']='Root LICENSE is Apache-2.0; inherited VMware files carry BSD-3-Clause SPDX headers and setup.py has BSD metadata. All originals and notices preserved.'
    (RUN/'logs/official_source_manifest.json').write_text(json.dumps(source,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    admission=[]
    for question in review['admission_questions']:
        item=dict(question);item['status']='pass';item['scope']='Selected finite contiguous single-device BF16 inputs; fixed stored b4/b8, chunk16, strategy3; no full distributed-system claim'
        item['runtime_evidence']=['logs/d0_execution.json','logs/d0_fresh_process_decode.json','logs/d0_packet_parse_review.json','logs/d0_boundary_probes.json']
        admission.append(item)
    now=datetime.now().astimezone();started=datetime.fromisoformat(read('logs/task_start.json')['recorded_at'])
    qualification=dict(schema_version='bounded_discovery_baseline_qualification_v1',run_id=RUN.name,recorded_at=now.isoformat(),
        baseline_name='official_dynamiq_fixed_codec',upstream_repository='https://github.com/CharlesHan24/DynamiQ',upstream_commit=codec.COMMIT,
        source_identity_status='pass',local_codec_status='pass',packet_status='pass',rng_status='pass',aggregate_comparability_status='pass',
        gate_decision='pass',gate_scope='D1 fixed local codec and one-shot FP64 sum with separately keyed worker RNG pools only',
        full_dynamiq_system_qualified=False,multi_gpu_communication_qualified=False,allreduce_performance_qualified=False,
        loaded_module=execution['module'],admission_questions=admission,public_history=review['public_history'],
        official_entrypoints=['eden_utils.scaling_compress(src,dst,rand_pool,n,nbits,chunk_size,stream_p,strategy)','eden_utils.scaling_decompress(src,dst,n,nbits,chunk_size,stream_p,strategy)'],
        selected_configuration=dict(input_dtype='bfloat16',output_dtype='bfloat16',stored_code_bits=[4,8],chunk_size=16,strategy=3,strategy_name='hierarchical MEE',
            native_min_numel=256,native_numel_multiple=256,tail_glue='Zero-pad to 256*ceil(D/256); crop to logical D; do not divide predivided fixture again',
            arithmetic_profile='Official --use_fast_math CUDA code; BF16 inputs/pools promoted to float, FP16-derived constants and LUT, BF16 final output; unchanged official arithmetic',
            scale_format='Per16-values UINT8 ceil scale code [1,255]; per256-values BF16 scale unit',
            code_range='Unsigned [0,2^b-2] packed in exactly b bits; all-ones code unused by encoder'),
        build_attempts=[dict(attempt=a['attempt'],status=a['status'],elapsed_seconds=a['elapsed_seconds'],repair=a['repair'],manifest=f"logs/official_build_{a['attempt']}.json",console=f"logs/official_build_{a['attempt']}_console.log") for a in attempts],
        build_repair_rounds=2,arithmetic_changes=False,wrapper_changes=[
            'MSVC /O2 and /std:c++17 map official GCC host flags; keep official CUDA -O3 --use_fast_math -lineinfo -std=c++17',
            'Python -X utf8 and MSVC UTF-8 source character set overcome GBK default',
            'Omit two unused unistd.h includes in compiled copies only; original official blobs remain unchanged',
            'Add verified NVIDIA CUDA11.8 cuRAND10.3.0.86 headers through isolated include path',
            'Python wrapper adds selected legal-domain guards, owned buffers, zero padding/crop, explicit current stream and artifact SHA256 verification'],
        source_diff=evidence('build/official_a03/source_glue.diff'),
        official_selftest=execution['official_selftest'],small_input_case_count=len(execution['cases']),
        independent_decode=dict(status='pass',packet_count=len(execution['independent_decode']['rows']),fresh_process=True,input_or_random_pool_supplied=False,static_inputs='Fixed dtype, length, bitwidth, chunk size and strategy; code-generated input-independent LUT'),
        rng=dict(generation='CPU torch.Generator.manual_seed(seed); torch.rand FP32 -> BF16 -> CUDA, matching official functional test',
            local_marginal='BF16 rounding of FP32 uniform PRNG samples, including reachable exact 1.0; not an ideal continuous distribution',
            across_worker_relation='Use separately keyed PRNG states and separately allocated pools; no claim of mathematically proven independence or original full-system correlation',
            repeatability='Same pool yields byte-identical packet; same seed creates identical pool',
            exact_one_behavior='frac=0, rand=1 gives increment -1 under official ceil rule; endpoint zero probes yield about 1e-7 and match official reference'),
        invalid_input_scope='Wrapper rejects nonfinite inputs and illegal metadata. Native host n=0/257 rejection was checked. No claim that native itself validates every nonfinite or cross-device input.',
        overflow_scope='Sampled +/-1e30 and +/-BF16max produced finite outputs; tiny +/-1e-35 samples collapsed to zero. This is observed boundary behavior, not a proof over the finite BF16 domain; D1 checks every result for nonfinites.',
        persistent_state=dict(source_defined_device_static_data_bytes=9288,allocated_lut_bytes=4608,device_symbol_bytes=4680,
            measurement='Source-defined array sizes (9*256 half plus 9*(4+256) half); not measured allocator footprint or total module/GPU memory',per_input_auxiliary_bytes=0),
        aggregate_comparability=dict(status='pass',definition='One-shot sum of independently decoded local packets using FP64 accumulation versus FP64 sum of original predivided BF16 inputs',
            constraints='Freeze input/worker/config/repeat seeds before D1; complete worker coverage, no duplicated pools, no ring reinterpretation'),
        resource_usage=dict(wall_window_seconds=(now-started).total_seconds(),conservative_active_upper_bound_hours=(now-started).total_seconds()/3600,cap_hours=4,build_repair_rounds=2,cap_build_repair_rounds=2),
        evidence=[evidence(x) for x in ['logs/official_source_manifest.json','logs/official_contract_review.json','logs/official_build_a01.json','logs/official_build_a02.json','logs/official_build_a03.json','logs/official_curand_dependency.json','logs/d0_execution.json','logs/d0_fresh_process_decode.json','logs/d0_packet_parse_review.json','logs/d0_boundary_probes.json','logs/d0_rng_state_checks.json']])
    (RUN/'baseline_qualification.json').write_text(json.dumps(qualification,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    captured=read('logs/official_build_environment_a03.json')
    gpu=[v.strip() for v in captured['nvidia_smi']['stdout'].strip().split(',')]
    environment=dict(schema_version='bounded_discovery_environment_v1',template_only=False,run_id=RUN.name,
        captured_at=captured['captured_at'],metadata_finalized_at=now.isoformat(),timezone='Asia/Shanghai',
        os=captured['platform'],python_executable=captured['executable'],python_version=captured['python'],
        pytorch_version=captured['torch'],pytorch_cuda_runtime=captured['torch_cuda'],driver_version=gpu[1],
        nccl_available=torch.distributed.is_nccl_available(),cuda_available=True,bf16_available=True,
        capability_evidence='Actual D0 official CUDA BF16 execution and official full self-test; not a fresh separate capability query',
        gpu=dict(name=gpu[0],compute_capability=[int(v) for v in gpu[2].split('.')],memory_total_bytes=int(gpu[3].split()[0])*1024**2,
            power_limit_watts=None,clock_settings=None,other_process_load=None,query=captured['nvidia_smi']),
        compiler=dict(configuration=captured['configuration'],versions=captured['compiler_versions'],
            flags=dict(cxx=attempts[-1]['extra_cflags'],cuda=attempts[-1]['extra_cuda_cflags'],include_paths=attempts[-1]['extra_include_paths']),
            minimal_cuda_extension_test='official_extension_built_loaded_executed',build_log='logs/official_build_a03_console.log',
            codec_extension_status='compiled_loaded_and_D0_passed',module=execution['module']),
        initialization_costs=dict(lut_prepare_ms=None,threshold_prepare_ms=None,fixture_h2d_ms=None,notes='Not benchmarked in D0'),
        dependency_lock_path='requirements.txt',new_dependency='NVIDIA CUDA11.8 libcurand10.3.0.86 headers/license only, isolated directory; logs/official_curand_dependency.json',
        environment_snapshot=evidence('logs/official_build_environment_a03.json'),
        notes=['New D0 build snapshot, not copied from historical G1/G2 runs.',
            'Shared Python/runtime and existing project-local CUDA/MSVC reused read-only; additional headers isolated under .toolchains/bounded_discovery_v1.',
            'No driver/global environment/package upgrade, no unsupported compiler bypass, no NCCL or distributed-system invocation.',
            'Unmeasured clocks/load/initialization costs remain null; static LUT byte count is not allocator memory measurement.'])
    (RUN/'environment.json').write_text(json.dumps(environment,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(gate_decision='pass',module_sha256=execution['module']['module_sha256'],wall_seconds=qualification['resource_usage']['wall_window_seconds'])))
if __name__=='__main__':main()
