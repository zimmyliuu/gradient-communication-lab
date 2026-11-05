"""Generate the eight-section G1a report from completed evidence, CPU only.

This writes only derived REPORT.md, summary.json, issues.jsonl and REPRODUCE.md.
It refuses missing/failed evidence and sealed runs; never invents measurements.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
METHOD='torch_cuda_repaired'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def lines(path):return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def fmt(value,digits=3):return 'null' if value is None else f'{value:,.{digits}f}'
def sci(value):return 'null' if value is None else f'{value:.6g}'
def cell(value):return str(value).replace('|','\\|').replace('\n',' ')
def table(header,rows):
    return '| '+' | '.join(header)+' |\n|'+ '|'.join('---' for _ in header)+'|\n'+''.join('| '+' | '.join(cell(v) for v in row)+' |\n' for row in rows)
def command_text(command):
    if isinstance(command,list):return subprocess.list2cmdline([str(v) for v in command])
    return str(command)
def commands_from(value):
    if isinstance(value,list):return value
    return value.get('commands_actually_executed',value.get('commands',[]))
def now():return datetime.now().astimezone().isoformat()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',required=True)
    args=parser.parse_args();run=(ROOT/args.run_dir).resolve()
    if not run.is_relative_to(ROOT/'results/gpu_codec_v1') or run.name!='gpu_codec_v1_20261101_002':raise ValueError('Use the new G1a run only')
    if (run/'artifact_integrity.json').exists():raise ValueError('Do not modify a sealed run')
    names=('analysis.json','benchmark_summary.jsonl','correctness.jsonl','correctness_acceptance.json',
        'logs/correctness_analysis.json','logs/correctness_commands.json','logs/execution_commands.json',
        'logs/benchmark_execution.json','logs/toolchain_summary.json','environment.json','codec_contract.json',
        'benchmark_config.json','input_manifest.json','execution_registration.json','requirements.txt',
        'traces/manifest.json','logs/pytest_torch_codec_attempt01_failure_analysis.json','logs/pytest_torch_codec_attempt02.xml')
    for name in names:
        if not (run/name).is_file():raise FileNotFoundError('Completed evidence required: '+name)
    analysis=read(run/'analysis.json');bench=lines(run/'benchmark_summary.jsonl');correct=lines(run/'correctness.jsonl')
    acceptance=read(run/'correctness_acceptance.json');quant=read(run/'logs/correctness_analysis.json')
    execution=read(run/'logs/benchmark_execution.json');environment=read(run/'environment.json')
    toolchain=read(run/'logs/toolchain_summary.json');contract=read(run/'codec_contract.json')
    config=read(run/'benchmark_config.json');manifest=read(run/'input_manifest.json');registration=read(run/'execution_registration.json')
    contract_hash=sha(run/'codec_contract.json');config_hash=sha(run/'benchmark_config.json')
    if analysis['torch_benchmark_validity']!='valid' or analysis['validation_error_count']:
        raise ValueError('Completed valid PyTorch measurements are required; retain failed analysis separately')
    if acceptance['method_status']!='pass' or execution['status']!='complete':raise ValueError('Formal correctness and measurements must complete')
    if not acceptance['execution_registration_verified'] or not acceptance['deterministic_algorithms'] or acceptance['fill_uninitialized_memory']:
        raise ValueError('Formal deterministic configuration differs')
    assert acceptance['contract_sha256']==registration['contract_sha256']==contract_hash
    assert execution['benchmark_config_sha256']==config_hash
    assert sha(run/'correctness.jsonl')==acceptance['correctness_sha256']
    assert analysis['generator_sha256']==sha(ROOT/'scripts/summarize_g1a.py')
    for name,digest in analysis['source_sha256'].items():assert sha(run/name)==digest,name
    for name,digest in registration['source_hashes'].items():assert sha(ROOT/name)==digest,name
    assert len(bench)==104 and execution['raw_blocks']==10400 and analysis['coverage']['valid_memory_records']==520
    assert Counter(r['status'] for r in correct)=={'pass':1101}
    assert all(v==0 for v in quant['mismatch_sums'].values())
    trace=analysis['trace'];assert trace['status']=='analyzed' and trace['operation_count']==3 and trace['kernel_count']>0
    assert analysis.get('trace_completeness')=='complete'
    assert trace.get('capture_integrity',{}).get('status')=='verified'
    success_trace_manifest=trace['manifest_path']
    assert success_trace_manifest=='traces/gpu_activity_manifest.json' and (run/success_trace_manifest).is_file()
    activity_manifest=read(run/success_trace_manifest)
    activity_audit_path=activity_manifest['audit_path'];activity_audit=read(run/activity_audit_path)
    assert activity_audit['status']=='pass' and activity_audit['chrome_records_exactly_match_measured_raw']
    assert activity_audit['dropped_records']==0 and not activity_audit['collection_errors']
    names=(*names,success_trace_manifest,activity_audit_path)
    assert len(analysis['figures'])==4 and all((run/path).is_file() for path in analysis['figures'])
    inputs={r['input_id']:r for r in manifest['inputs']}
    def normal(row):
        source=inputs[row['input_id']]
        return source['source_kind']=='synthetic' and source['generator_config']['distribution']=='standard_normal' and source['generator_config']['seed']==7301
    def select(n,b,operation='roundtrip',allocation='preallocated',validation='trusted_inputs'):
        matches=[r for r in bench if normal(r) and r['original_numel']==n and r['nominal_bits']==b and r['operation']==operation and r['allocation_mode']==allocation and r['validation_mode']==validation]
        assert len(matches)==1,(n,b,operation,allocation,validation)
        return matches[0]
    chosen=[select(n,b) for n in (256,65536,1048576,4194304) for b in (4,8)]
    chosen += [select(4194304,b,op) for b in (4,8) for op in ('encode','decode')]
    performance_rows=[[r['nominal_bits'],f"{r['original_numel']:,}",r['operation'],fmt(r['gpu_session_median_us_median']),
        'null',f"{fmt(r['gpu_session_median_us_min'])}–{fmt(r['gpu_session_median_us_max'])}",'null', '是' if r['noise_flag'] else '否'] for r in chosen]
    api=[select(n,b,'roundtrip','api_allocating',validation) for n in (65536,1048576) for b in (4,8) for validation in ('trusted_inputs','checked_api')]
    api_rows=[[r['original_numel'],r['nominal_bits'],r['validation_mode'],fmt(r['gpu_session_median_us_median']),
        f"{fmt(r['gpu_session_median_us_min'])}–{fmt(r['gpu_session_median_us_max'])}"] for r in api]
    memory={r['case_id']:r for r in analysis['memory_case_statistics']}
    memory_cases=[select(n,b) for n in (65536,4194304) for b in (4,8)]
    storage_rows=[];allocator_rows=[]
    for row in memory_cases:
        m=memory[row['case_id']]
        storage_rows.append([row['original_numel'],row['nominal_bits'],*[fmt(m[k]['median']/2**20,4) for k in
            ('input_bytes','threshold_bytes','lut_bytes','packet_buffer_bytes','decoded_output_bytes','workspace_bytes')]])
        allocator_rows.append([row['original_numel'],row['nominal_bits'],fmt(m['allocated_before_bytes']['median']/2**20,4),
            fmt(m['peak_allocated_bytes']['max']/2**20,4),fmt(m['peak_extra_allocated_bytes']['max']/2**20,4),
            fmt(m['reserved_before_bytes']['median']/2**20,4),fmt(m['peak_reserved_bytes']['max']/2**20,4)])
    duration=analysis['block_duration_diagnostics'];outside=duration['below_target_count']+duration['above_target_count']
    noisy_sessions=sum(s['noise_flag'] for r in bench for s in r['session_statistics'])
    qs=quant['real_gradient_quantization_distortion']
    quant_rows=[[q['bits'],q['inputs'],'0 字节差异','0 BF16 位差异',f"{sci(q['quant_mse_min'])}–{sci(q['quant_mse_max'])}",
        f"{sci(q['normalized_squared_error_min'])}–{sci(q['normalized_squared_error_max'])}",q['wire_bytes_unique'][0],fmt(q['effective_bits_unique'][0],6)] for q in qs]
    tool_status=toolchain.get('status','unknown');compile_info=toolchain.get('minimal_extension_compile',{});gpu_info=toolchain.get('gpu_call',{})
    compile_status=compile_info.get('status','unknown');gpu_status=gpu_info.get('status','unknown')
    tool_pass=compile_status in ('pass','passed','complete','success') and gpu_status in ('pass','passed','complete','success')
    selected_toolchain=toolchain.get('selected_toolchain',{})
    link_probe=environment.get('compiler',{}).get('link_version',{})
    link_match=re.search(r'Version\s+(\d+(?:\.\d+)+)',str(link_probe.get('stdout',''))+' '+str(link_probe.get('stderr','')),re.IGNORECASE)
    link_version=selected_toolchain.get('link_version') or (link_match.group(1) if link_match else '见environment.json实际输出')
    attempts=toolchain.get('attempts',[])
    failed_attempts=[a for a in attempts if a.get('status') in ('failed','fail','error','blocked')]
    next_action='optimize_codec' if tool_pass else 'fix_environment'
    counts_by_kind=manifest['counts_by_source_kind'];K=[r['inner_repeats'] for r in config['repeat_counts']]
    all_protocol=[r for r in correct if r['input_id'] not in inputs]
    protocol_counts=Counter(r['check_type'] for r in all_protocol)
    def issue(identifier,category,severity,status,description,evidence,reproduction,next_step):
        return {'issue_id':identifier,'category':category,'severity':severity,'status':status,'description':description,
            'evidence_paths':evidence,'reproduction':reproduction,'fix_or_next_step':next_step}
    issues=[issue('G1A-MISSING-OPTIMIZED','coverage','high','open',
        '仅完成 torch_cuda_repaired；cuda_optimized_repaired 及两实现配对比较缺失。完整 G1 不能通过。',
        ['codec_contract.json','analysis.json','benchmark_config.json'],None,'新 run 实现融合 CUDA codec，复用固定语义和全部夹具，通过后执行完整双实现矩阵。'),
        issue('G1A-PYTEST-HELPER-001','test_harness','low','resolved',
        '首次 pytest 18 项在测试助手 bytes_of 对二维阈值生成嵌套列表时 TypeError，未进入 codec 比较；仅将助手 uint8 视图展平后 38 项全通过。',
        ['logs/pytest_torch_codec_attempt01.xml','logs/pytest_torch_codec_attempt01_failure_analysis.json','logs/pytest_torch_codec_attempt02.xml'],
        '原始失败命令与后续通过命令见 logs/correctness_commands.json；原始失败 JUnit 保留。','已修复测试助手；CPU参照、阈值、期望结果和容差未改。')]
    if analysis['noise_case_count']:
        issues.append(issue('G1A-BENCH-NOISE','measurement_noise','medium','observed',
            f"{analysis['noise_case_count']}/104 个 case 至少一个 session 的 GPU IQR/median>0.10，共 {noisy_sessions}/520 个 case-session。所有观测保留，不按快慢剔除。",
            ['analysis.json','benchmark_raw.jsonl','benchmark_summary.jsonl','logs/benchmark_device_states.jsonl'],
            analysis['reproduction_command'],'保留当前数据；后续优化比较采用相同环境与配对会话，必要时在新run调查背景负载和时钟。'))
    if outside:
        issues.append(issue('G1A-BLOCK-DURATION','timing_quality','medium','observed',
            f"固定 K 后 {outside}/10400 块不在10–50 ms目标内：低于目标 {duration['below_target_count']}，高于目标 {duration['above_target_count']}。全部原始块仍计入汇总。",
            ['analysis.json','benchmark_config.json','logs/K_pilot.jsonl','benchmark_raw.jsonl'],
            analysis['reproduction_command'],'目标区间用于诊断，不事后调整K或挑选快块；后续run可改进独立pilot设计。'))
    if failed_attempts:
        issues.append(issue('G1A-TOOLCHAIN-ATTEMPTS','toolchain','medium','resolved' if tool_pass else 'open',
            f"工具链保留 {len(failed_attempts)} 个非通过尝试；最终编译={compile_status}，GPU调用={gpu_status}。每次原因见 toolchain_summary attempts。",
            ['logs/toolchain_summary.json',*[p for a in failed_attempts for p in a.get('evidence_paths',[])]],
            '实际命令见 logs/toolchain_summary.json.commands_actually_executed',
            '最小扩展只证明工具链可用，不能替代优化 codec 的实现、验收和性能测量。' if tool_pass else '按保留的首个真实编译/调用错误继续修复工具链；PyTorch基线证据独立有效。'))
    if not tool_pass and not failed_attempts:
        issues.append(issue('G1A-TOOLCHAIN-INCOMPLETE','toolchain','high','open',f'最小扩展编译={compile_status}、GPU调用={gpu_status}。',
            ['logs/toolchain_summary.json','environment.json'],None,'修复并验证最小扩展后再实现优化codec。'))
    if (run/'logs/analysis_attempt01_cpu_only_trace_misclassification.json').exists():
        issues.append(issue('G1A-PROFILER-CPU-ONLY','profiling_validation','medium','resolved',
            '首轮Windows PyTorch Kineto仅输出CPU事件，初版汇总错误地将缺失GPU活动视为0并判valid；复核后改成unknown/null与trace incomplete，保留错误汇总和CPU-only trace。CUPTI11.8连独立cudart活动探针也未完成采集；仅追加CUPTI12.9 profiling组件后取得真实GPU/runtime活动并通过raw覆盖审计。',
            ['logs/analysis_attempt01_cpu_only_trace_misclassification.json','logs/analysis_attempt02_trace_incomplete.json',
             'traces/manifest.json','logs/cupti_diagnostic_cudart_only.json',success_trace_manifest,'logs/cupti_trace_audit_a03.json'],
            '实际profile/collector/audit/summarize命令见REPRODUCE及logs/execution_commands.json',
            '已修正分析器的GPU活动可用性与覆盖门槛；新采集不改Torch cu118、固定codec或原始10400个计时块。'))
    small=select(256,4);large4=select(4194304,4);large8=select(4194304,8)
    findings=[f"正式局部codec正确性1101项通过，五类码/尺度/完整包/输出差异均为0；82输入、164 input-bit 对。",
        f"PyTorch基线104case×5session×20block=10400块有效，520份独立显存记录；完整G1仍缺优化GPU实现。",
        f"标准正态seed7301直接roundtrip：N=256 b4中位 {small['gpu_session_median_us_median']:.3f} us；N=4194304 b4/b8中位 {large4['gpu_session_median_us_median']:.3f}/{large8['gpu_session_median_us_median']:.3f} us。",
        f"独立trace中{trace['operation_count']}次roundtrip共{trace['kernel_count']}个GPU kernel，平均{trace['kernel_count_per_operation_mean']:.1f}个/次；融合与缩小workspace需要新实现验证。",
        f"噪声标记{analysis['noise_case_count']}case，目标块时长偏离{outside}块；全部保留，无优化版加速比。"]
    summary=read(ROOT/'handoff_templates/summary.json')
    summary.update(template_only=False,run_id=run.name,stage='G1',substage='torch_cuda_baseline',
        started_at=registration['registered_at'],finished_at=now(),execution_status='incomplete',contract_sha256=contract_hash,
        benchmark_config_sha256=config_hash,correctness_status='incomplete',benchmark_validity='incomplete',
        optimization_gain='not_evaluated',g1_acceptance='incomplete',next_action=next_action,
        coverage={'nominal_bits_completed':[4,8],'methods_completed':[METHOD],'correctness_cases_passed':1101,
            'correctness_cases_failed':0,'correctness_cases_skipped':0,'real_gradient_inputs_completed':12,
            'benchmark_cases_completed':104,'raw_timing_blocks':10400,'memory_records':520,
            'fixture_inputs':82,'protocol_checks':117,'missing_items':['cuda_optimized_repaired','two_GPU_implementation_paired_comparison']},
        claims={'local_codec_parity_validated':True,'gpu_codec_performance_measured':True,
            'full_dynamiq_gpu_validated':False,'training_quality_newly_validated':False,'multigpu_speedup_measured':False},
        method_statuses={METHOD:{'execution_status':'complete','correctness_status':'pass','benchmark_validity':'valid',
            'correctness_rows':937,'benchmark_cases':104,'raw_blocks':10400},
            'cpu_ref_repaired':{'correctness_status':'pass','qualification':'actual_CPU_reference_execution_not_independent_GPU_parity',
                'roundtrip_rows':164,'benchmark_validity':'not_run'},
            'cuda_optimized_repaired':{'execution_status':'not_started','correctness_status':'not_run','benchmark_validity':'not_run'}},
        key_findings=findings,open_issues=[i['issue_id'] for i in issues if i['status']!='resolved'],
        noise_case_count=analysis['noise_case_count'],outside_target_duration_block_count=outside,
        toolchain_status=tool_status,toolchain_compile_status=compile_status,toolchain_gpu_call_status=gpu_status,
        paired_speedup=None,report_generator={'path':'scripts/write_g1a_report.py','sha256':sha(__file__),
            'source_sha256':{name:sha(run/name) for name in names if name not in ('logs/execution_commands.json',)}})
    summary['artifacts'].update(analysis='analysis.json',correctness_acceptance='correctness_acceptance.json',
        trace_manifest=success_trace_manifest,gpu_activity_manifest=success_trace_manifest,
        retained_cpu_only_trace_manifest='traces/manifest.json',fixtures='fixtures_manifest.json',requirements='requirements.txt',
        toolchain_summary='logs/toolchain_summary.json',figures=analysis['figures'])
    state_rows=[['环境与最小CUDA扩展',f"{environment['gpu']['name']}；编译 {compile_status} / GPU调用 {gpu_status}",'environment.json；logs/toolchain_summary.json'],
        ['b=4 固定配置','CPU/GPU完整包、整数码、尺度和BF16输出逐位通过','correctness.jsonl'],
        ['b=8 固定配置','完整合法码集合与修复索引逐位通过','correctness.jsonl'],
        ['真实梯度','12输入×2位宽局部codec通过；无新训练结果','input_manifest.json；logs/correctness_analysis.json'],
        ['GPU性能与显存','PyTorch子矩阵valid：104case、10400块、520显存记录','analysis.json；benchmark_raw.jsonl；memory.jsonl'],
        ['完整G1验收','incomplete：缺cuda_optimized_repaired及配对比较','summary.json']]
    attempt_rows=[[a.get('attempt_id'),a.get('status'),a.get('reason',''),', '.join(a.get('evidence_paths',[]))] for a in attempts]
    kernel_rows=[[k['name'],k['count'],fmt(k['duration_sum_us']),fmt(k['mean_duration_us'])] for k in trace['top_kernels'][:5]]
    issue_rows=[[i['issue_id'],i['description'],'；'.join(i['evidence_paths'][:3]),i['fix_or_next_step']] for i in issues]
    report=f'''# GPU 低比特编解码实施报告

run_id：`{run.name}`。生成时间：{summary['finished_at']}。技术方案版本：1.0；复核依据：`reviews/gpu_codec_v1_20261101_001_review.md`。

## 1 结论与完成范围

已完成 G1a 的纯 PyTorch CUDA eager 局部 codec 基线。正式正确性 **1101项通过、0失败、0跳过**，其中GPU检查937项、CPU参照实际roundtrip164项；两者的码值、尺度、紧凑码流和BF16输出一致。104个基线性能case完成5个session、每session20个计时块，汇总校验通过。标准正态seed7301的N=4,194,304直接roundtrip中位延迟为b4 **{fmt(large4['gpu_session_median_us_median'])} us**、b8 **{fmt(large8['gpu_session_median_us_median'])} us**。本次没有优化版codec，也没有配对加速比，因此完整G1及其顶层执行/正确性/性能验收状态均为`incomplete`。

{table(['项目','实际状态','证据路径'],state_rows)}
## 2 实现与契约

沿用已接受G0的CPU参照及上游commit `{contract['upstream_commit']}`，没有改动旧项目或旧run。CPU核心 `dynamiq.py` SHA256为 `{contract['cpu_reference_files'][0]['sha256']}`；GPU实现 `src/gradient_comm_lab/torch_codec.py` SHA256为 `{registration['source_hashes']['src/gradient_comm_lab/torch_codec.py']}`。本次契约SHA256为 `{contract_hash}`，9个执行源文件在正式验收前冻结，正式检查与计时均核验注册哈希。

输入/输出是连续一维BF16；逻辑长度向上补齐到256，尾部补零，解码裁切回原长。空输入跳过量化、零字节包。名义b4/b8实际载荷为5/9位，低位先行；16元素一组的uint8尺度码和256元素一组的BF16顶层尺度紧随payload，额外header/alignment为0。完整包大小是`ceil(L*(b+1)/8)+L/16+2*L/256`。LUT与显式CPU阈值均有原始字节哈希；阈值按固定CPU生成器先main后scale生成，codec内不推进LCG。

各步保留BF16舍入，包括`BF16(1e-23)`、`BF16(65535)=65536`、`qscale_BF16*top/255`；查表用`right=True`，阈值比较严格`<`，b8查表偏移在整数域完成，负零保留。`qscale=0`合法；checked API拒绝非有限输入/中间量/输出、非法码流与尺度。trusted路径保留形状/dtype/device等主机元数据检查，不读取GPU值；其输入已经CPU和checked GPU验收。checked路径包含显式GPU谓词同步，单独计时。外部输入/阈值/LUT不能别名到plan-owned storage。

GPU基线使用已安装PyTorch {environment['pytorch_version']} 的CUDA {environment['pytorch_cuda_runtime']} 二进制算子，没有编译自定义codec；CUDA Graph关闭、compile=eager、无快速数学替换。环境为{environment['os']['platform']}，GPU为{environment['gpu']['name']}，驱动{environment['driver_version']}。复核允许把原先的编译依赖拆开，先交PyTorch基线；这一调整没有降低数值与计时验收标准。工具链状态见第6节，最小扩展通过也不代表优化codec已经存在。

## 3 正确性结果

82个输入由36个合成、34个边界、12个原样复用真实梯度构成。合成覆盖6个性能长度、标准正态/组间尺度不同正态、种子7301/7302/7303，先CPU FP32生成再转BF16。边界覆盖外层长度0/1/15/16/255/256/257/1023/1024/1025，全零、正负常数、负零、subnormal、尺度差异、LUT端点/中点/相邻可表示数和阈值0/65536。

每个输入×位宽执行编码、独立解码、roundtrip、重复性和输入保护五类GPU检查；输入、两类阈值和LUT在操作前保存独立CPU字节快照，输出和重复码流也使用独立副本。117条补充协议检查包括{protocol_counts['pack']}条pack、{protocol_counts['decode']}条decode、{protocol_counts['encode']}条encode算术探针、{protocol_counts['roundtrip']}条精确归一化LUT边界roundtrip和{protocol_counts['invalid_input']}条预期拒绝。涵盖b4/b8全部合法码、截断/unused/尾部填充位、65,280个有限BF16 epsilon复算、最小subnormal顶层尺度、qscale0/1/127/255以及内部与最终输出溢出。五类差异累计全部为0。

真实梯度来自seed1101、更新前100/500/2500、rank0–3的12份开发夹具，保持原始来源与阈值身份；使用`x=(g_i.to(BF16)/4)`后再补齐，没有重新训练、反向传播或打开留出分区。量化失真用FP64累积且排除padding，量化误差与实现差异分开。12个真实输入结果为：

{table(['b','输入数','完整包','解码','MSE范围','归一化SSE范围','wire B','实际bits/逻辑元素'],quant_rows)}
所有164对CPU/GPU量化指标逐字段相等。零长度的每元素指标和零能量的归一化指标使用`null`并注明原因。正式检查没有反例；早期pytest曾有18个测试助手TypeError，修复字节展平后38项通过，旧JUnit与归因文件保留。早期smoke与两次完整preliminary日志标记`preliminary_not_formal`，不替代正式契约证据。

## 4 GPU 性能结果

主矩阵为6长度×2分布×2位宽×encode/decode/roundtrip=72case；另有标准正态N=65,536/1,048,576的8个allocating API case和24个真实梯度roundtrip case，共104。每case warmup至少50次；5个session分别用固定种子7501–7505打乱case顺序，每case每session20块。独立pilot以3次单调用中位数选择接近20ms的K，固定范围为{min(K)}–{max(K)}，正式计时后不更改。全部10,400块保留。

CUDA Event记录的是stream区间，包含eager发射空隙以及checked/API路径中的同步或分配空隙；不是单纯kernel执行时长之和。wall计时边界为预同步之后开始，包含event、K次调用和末尾event同步。主矩阵复用预分配plan；所有case的外部输入、LUT和阈值生成/文件读取/H2D、CPU参照与正确性检查在计时外。allocating API每次创建整个plan/index/workspace及3个BF16标量常数，这部分初始化及可能的CPU→GPU常量拷贝/同步包含在被测factory成本中。Graph关闭、eager执行、steady-state reuse；确定性算法开启、未初始化内存填充关闭。roundtrip来自直接测量，没有把encode和decode相加代替。

测量运行在Windows WDDM工作站，不能声称独占或空载：`logs/benchmark_external_processes_before.json`保留测量前桌面/GPU进程快照，部分进程显存为N/A或因权限不能读取；`logs/benchmark_device_states.jsonl`保留会话边界设备状态。没有把观察到的噪声归因于某个具体应用，也没有把瞬时环境快照当作整个实验的恒定负载。

下表为标准正态seed7301；中位数是5个session中位数的中位数，范围是这5个中位数的最小–最大。优化版与加速比均为`null`，不能据此判定任何大小上的优化收益。

{table(['b','逻辑N','操作','PyTorch中位us','优化版us','session中位范围us','配对加速比','噪声标记'],performance_rows)}
`noise_flag`只检查session内的IQR/median，无标记不代表跨session稳定。例如N=4,194,304的b8 roundtrip未触发该标记，但5个session中位数仍分布在{fmt(large8['gpu_session_median_us_min']/1000)}–{fmt(large8['gpu_session_median_us_max']/1000)}ms，必须连同范围解读中位数。

API调用每次构造完整plan/workspace后执行roundtrip，因此同时包含分配和索引/常量准备；checked模式额外包含验证同步。其独立结果为：

{table(['N','b','验证模式','allocating中位us','session中位范围us'],api_rows)}
**噪声和目标时长诊断不作剔除：**{analysis['noise_case_count']}/104个case出现至少一个session的GPU IQR/median>0.10，共{noisy_sessions}/520个case-session；低于10ms目标的块{duration['below_target_count']}个，高于50ms的块{duration['above_target_count']}个，共{outside}/10,400。实际块时长min/median/max为{fmt(duration['elapsed_ms_min'])}/{fmt(duration['elapsed_ms_median'])}/{fmt(duration['elapsed_ms_max'])}ms。详单、每session统计和原始观测均保留在`analysis.json`、`benchmark_summary.jsonl`和`benchmark_raw.jsonl`。p95描述的是块平均每调用延迟，不能解释为单次请求尾延迟。

![桶大小与直接测量延迟](figures/latency_vs_numel.png)

## 5 字节、显存与性能剖析

可手算示例：N=L=256时，b4 payload=160B、uint8尺度=16B、BF16顶层尺度=2B，完整包178B，即5.5625bits/元素；b8分别为288+16+2=306B，即9.5625bits/元素。真实梯度的实际bits/逻辑元素略高，因为最后一块padding也占wire空间。wire大小不等于GPU临时存储。

下表按独立显存探针跨5个session的中位数列出标准正态预分配roundtrip的张量存储，单位MiB；按唯一storage计数，不重复计算view。

{table(['N','b','输入','阈值','LUT','包','输出','workspace'],storage_rows)}
分配器基线已经包含输入、阈值、LUT及整个持久plan；下表基线为session中位数，peak/extra取session最大值。`extra=peak_allocated-allocated_before`，extra为零不等于工作内存为零。reserved是PyTorch缓存分配器保留空间，不能当作wire或纯codec张量大小。

{table(['N','b','allocated基线MiB','peak allocated MiB','peak extra MiB','reserved基线MiB','peak reserved MiB'],allocator_rows)}
allocating API的baseline仍有一份用于参照的持久plan，每调用再建立一份新plan，其增量单独记录于`memory.jsonl`。这些统计只覆盖PyTorch allocator，不包括CUDA context/driver的外部分配。显存探针在正式计时之外执行。

独立GPU活动trace为`{trace['trace_path']}`，成功清单是`{success_trace_manifest}`，对应b4/N=65,536的3次可信输入预分配roundtrip。CUPTI12.9仅用作分析组件，应用与正式基线仍使用PyTorch cu118。共{trace['kernel_count']}个kernel，平均{fmt(trace['kernel_count_per_operation_mean'],1)}个/次；无法关联到操作的kernel为{trace['unassigned_kernel_count']}。设备memcpy事件{trace['device_memcpy_count']}，runtime memcpy调用{trace['runtime_memcpy_count']}，设备memset事件{trace['device_memset_count']}，runtime同步事件{trace['runtime_synchronization_count']}。trace边界包含末尾显式同步，不能把该同步计作codec内部验证，也不能以trace时长替换正式样本。这份trace只覆盖preallocated模式，不能据此声称allocating API没有host copy。主要kernel按累计设备时长排序：

{table(['kernel','次数','累计us','平均us'],kernel_rows)}
采集覆盖已核验：raw窗口内记录与Chrome trace逐项相等，dropped records和采集错误均为0。方向审计记录DtoD {activity_audit['device_memcpy_direction_counts'].get('DtoD',0)}次、合计{activity_audit['device_memcpy_bytes']:,}B，HtoD {activity_audit['device_HtoD_count']}次、DtoH {activity_audit['device_DtoH_count']}次。这里的0有已启用活动类型、关联/时间窗完整性和零丢记录支撑，结论只限这3次preallocated roundtrip的捕获窗口；不覆盖准备阶段或allocating API。审计证据为`{activity_audit_path}`。

trace支持的下一步是减少当前多算子链的kernel发射与中间workspace：优先尝试融合16元素scale/归一化、LUT区间与显式随机舍入，并把紧凑打包接入同一实现。现有结果只证明该开销存在，融合收益需要新run逐位验收和配对测量，不能预填速度提升。

首次Windows PyTorch Kineto trace只有CPU事件，原始文件与`traces/manifest.json`仍保留；其中GPU kernel/copy/sync计数应为unknown/null，不是0。首轮分析误判valid的快照及纠正后的trace-incomplete快照也保留在`logs/analysis_attempt01_cpu_only_trace_misclassification.json`与`logs/analysis_attempt02_trace_incomplete.json`。CUPTI11.8探针失败也保留，成功证据来自另行采集的CUPTI12.9活动及raw→Chrome覆盖审计；没有用CPU算子数量替代GPU kernel数量。

![wire实际位数](figures/wire_effective_bits.png)

![量化失真](figures/quantization_distortion.png)

![显存组成与增量](figures/memory_vs_numel.png)

## 6 未解决问题与下一步

工具链最终状态：`{tool_status}`；最小扩展编译=`{compile_status}`，GPU调用=`{gpu_status}`。实际工具为nvcc {selected_toolchain.get('nvcc_version','见环境记录')}、cl {selected_toolchain.get('cl_version','见环境记录')}、link {link_version}、Windows SDK {selected_toolchain.get('windows_sdk_version','见环境记录')}。最小GPU调用使用{gpu_info.get('tensor_elements','见日志')}元素验证y==x+1。项目独立的CUDA/MSVC准备没有改动旧run或原CPU实现；具体编译命令、日志与实际环境见`logs/toolchain_summary.json`和`environment.json`。下表a04的`compiled_loaded_gpu_not_run`是编译阶段快照，后续单独GPU调用已通过，以最终`gpu_call.status`为准。所有尝试保留：

供应记录存在明确边界：取得的官方VS catalog与channel宣告的大小/哈希不一致，未将catalog本身声称为哈希验证通过。单个MSVC包的SHA256及关键已签名MSVC二进制的签名检查通过；NVIDIA CUDA11.8组件按官方MD5核对并记录本地SHA256，`nvcc`的unsigned状态如实保留。详情见[工具链来源与校验记录](logs/toolchain_summary.json)。这些记录不改变上面的实际编译与GPU调用结果。

{table(['尝试','状态','原因','证据'],attempt_rows)}
{table(['问题','影响/状态','证据','建议处理'],issue_rows)}
建议下一动作：`{next_action}`。{'工具链最小闭环已通过，可在新的run实现并验证优化codec；完成双实现一致性和配对性能之后再复核完整G1。' if tool_pass else '先完成最小编译与GPU调用闭环，再在新的run实现优化codec。现有纯PyTorch基线仍可独立复核。'}本轮不自动进入G2，不新增训练或多GPU结论。

## 7 复现与结论边界

实际命令入口：[REPRODUCE.md](REPRODUCE.md)。完整结果目录：`{run.as_posix()}`。`fixtures_manifest.json`记录全部夹具路径、字节数与SHA256，`{success_trace_manifest}`记录成功GPU活动、raw记录和采集/审计源码快照；旧`traces/manifest.json`仅对应保留的CPU-only尝试。真实梯度夹具仍引用本机已接受G0文件。跨机器复核需取得这些明确列出的历史夹具/checkpoint、CPU参照及其导入依赖，并改配解释器和绝对路径；也可在具有这些依赖的环境按生成器重建新夹具目录。本交付不宣称是脱离原数据的portable standalone包。生成脚本是`scripts/summarize_g1a.py`和`scripts/write_g1a_report.py`，图表均可从JSONL重建。

本轮证明固定配置局部codec在这台GPU、这个PyTorch/CUDA环境及显式阈值下的正确性与成本。它没有实现完整DynamiQ的动态控制、LCG状态推进、误差反馈或ring多跳过程；没有新的训练质量、多GPU网络吞吐或训练加速证据，也没有证明压缩小包等于端到端更快。完整G1保留`incomplete`，只将已测PyTorch子矩阵标为complete/pass/valid。

## 8 交付文件核对

- [x] summary、环境、契约和基准配置使用实际数据，template_only=false；整体和逐实现状态分开。
- [x] CPU参考、GPU执行源、配置、输入与夹具哈希已记录，正式源冻结后未改动。
- [x] 1101条正确性、10400条原始计时、104个汇总和520份显存记录相互对应。
- [x] 延迟、wire/量化失真、显存三组共4张图可由原始JSONL重新生成。
- [x] 源码、测试、依赖与实际命令可取得；旧pytest失败、工具链失败与preliminary记录保留。
- [x] 未完成优化codec、噪声、目标时长偏离和本机外部夹具依赖均明确记录。

完整文件SHA封存由最后的独立交付审计生成`artifact_integrity.json`；该封存不是新增性能或正确性实验。
'''
    command_records=[]
    for group in ('logs/correctness_commands.json','logs/execution_commands.json'):
        for item in commands_from(read(run/group)):
            command_records.append((group,item))
    for item in toolchain.get('commands_actually_executed',[]):command_records.append(('logs/toolchain_summary.json',item))
    for source in ('logs/cupti_trace_a03.json',activity_audit_path):
        record=read(run/source)
        if record.get('command'):
            command_records.append((source,{'command':record['command'],'result':record.get('status'),
                'evidence_paths':[source,success_trace_manifest]}))
    command_records.append(('当前报告生成进程',{'command':f'./scripts/python.ps1 scripts/write_g1a_report.py --run-dir results/gpu_codec_v1/{run.name}',
        'result':'读取已完成证据并生成本报告、summary、issues、REPRODUCE；成功退出后生效'}))
    reproduced=[]
    for source,item in command_records:
        if isinstance(item,dict):
            text=command_text(item.get('command',item.get('argv',item.get('commands',''))))
            evidence=item.get('evidence_paths',item.get('evidence',item.get('output',item.get('result',''))))
            if item.get('status') in ('planned','pending','not_run'):continue
        else:text=command_text(item);evidence='详见来源日志的退出状态与输出'
        if not text:continue
        reproduced.append(f'来源：`{source}`。结果/证据：{cell(evidence)}。\n\n```powershell\n{text}\n```\n')
    reproduce=f'''# G1a 实际执行记录与复现边界

工作目录：`{ROOT.as_posix()}`。Shell：PowerShell。解释器：`{environment['python_executable']}`，Python {environment['python_version']}、PyTorch {environment['pytorch_version']}。始终经`scripts/python.ps1`设置本项目缓存并复用已验证解释器；没有在旧项目目录执行测试。

以下命令来自实际执行台账、工具链退出记录和当前报告生成进程；不把计划命令写成已完成。列表按证据来源分组，并非时间顺序或可直接从上到下重放的顺序。正式证据必须使用`execution_registration.json`中冻结的源、契约和输入；注册必须在正式正确性之前完成。阶段顺序为环境/工具链发现与准备、夹具生成、pytest和preliminary、注册、正式正确性、K pilot、5-session measurement、独立profile、CPU汇总作图、报告。

下列裸Python路径形式保留的是日志中的已执行子进程argv，不表示跳过项目启动环境。复现时以`scripts/python.ps1`启动相同的脚本与参数，保证使用本项目缓存配置；即用`./scripts/python.ps1`替换原argv中的Python可执行文件部分，后续参数保持不变。原始命令展示只用于追溯实际调用，不能向已封存run原地写入。

{''.join(reproduced)}
## 输出与重跑注意

正式正确性见`correctness.jsonl`和`correctness_acceptance.json`；主测量见`benchmark_raw.jsonl`、`memory.jsonl`和`logs/benchmark_execution.json`；成功的独立GPU活动见`{success_trace_manifest}`，旧`traces/manifest.json`对应CPU-only失败诊断；汇总与四张图见`analysis.json`列出的derived_files。正式源快照保存在`code/`，依赖清单是`requirements.txt`。

记录中的路径是本次真实执行路径，不应对已封存run原地重跑注册、prepare、正确性或性能命令；这些入口会拒绝覆盖关键证据。需要新实验时建立新run并重新绑定输入/源/契约。CPU汇总与报告属于派生产物，封存前可以从同一不可变原始记录重建。

本报告生成时，最后的seal属于后续交付步骤；只有实际执行并产出审计/完整文件清单后才构成封存证据，不将它预记为已完成实验。优化CUDA codec入口尚未实现；G2状态推进/多跳与训练验收也未执行。

初次pytest失败、preliminary日志及工具链各尝试均保留原记录。夹具以连续张量原始字节哈希验证；重新torch.save归档的文件哈希可能因文件名/序列化版本不同，应同时比较input/LUT/main/scale内容身份。真实梯度复用的12个旧文件路径见`fixtures_manifest.json`，无留出数据读取。
'''
    # All required evidence was checked before any derived delivery file is written.
    (run/'REPORT.md').write_text(report,encoding='utf-8')
    write(run/'summary.json',summary)
    (run/'issues.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n' for row in issues),encoding='utf-8')
    (run/'REPRODUCE.md').write_text(reproduce,encoding='utf-8')
    print(json.dumps({'run_id':run.name,'report':'REPORT.md','overall_G1':'incomplete','torch_correctness':'pass',
        'torch_benchmark':'valid','issues':len(issues),'next_action':next_action,'toolchain_status':tool_status},ensure_ascii=False))
    return 0


if __name__=='__main__':sys.exit(main())
