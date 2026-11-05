"""Generate G1b delivery report from complete measured evidence; CPU only."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime,timezone,timedelta
import hashlib,json
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[1]
METHODS=('torch_cuda_repaired','cuda_optimized_repaired')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rows(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def fmt(v,d=3):return 'null' if v is None else f'{v:,.{d}f}'
def table(header,data):
    def cell(v):return str(v).replace('|','\\|').replace('\n',' ')
    return '| '+' | '.join(header)+' |\n|'+'|'.join('---' for _ in header)+'|\n'+''.join('| '+' | '.join(cell(v) for v in r)+' |\n' for r in data)
def now():return datetime.now(timezone(timedelta(hours=8))).isoformat()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',required=True);args=parser.parse_args()
    run=(ROOT/args.run_dir).resolve()
    if not run.is_relative_to(ROOT/'results/gpu_codec_v1') or (run/'artifact_integrity.json').exists():raise ValueError('New unsealed project run required')
    names=('analysis.json','benchmark_summary.jsonl','benchmark_raw.jsonl','memory.jsonl','correctness.jsonl','correctness_acceptance.json',
        'correctness_case_manifest.json','environment.json','codec_contract.json','benchmark_config.json','pilot_protocol.json',
        'input_manifest.json','execution_registration.json','logs/cuda_build_manifest.json','logs/correctness_development_attempts.json',
        'logs/execution_commands.json','traces/g1b_trace_index.json','logs/pytest_cuda_codec_a01.xml','logs/cuda_implementation_review.json','logs/cuda_build_a01.json','logs/cuda_build_a02.json','logs/toolchain_summary.json')
    for name in names:
        if not (run/name).is_file():raise FileNotFoundError(name)
    a=read(run/'analysis.json');accept=read(run/'correctness_acceptance.json');correct=rows(run/'correctness.jsonl')
    bench=rows(run/'benchmark_summary.jsonl');raw=rows(run/'benchmark_raw.jsonl');memory=rows(run/'memory.jsonl')
    config=read(run/'benchmark_config.json');registration=read(run/'execution_registration.json')
    manifest=read(run/'input_manifest.json');inputs={r['input_id']:r for r in manifest['inputs']}
    build=read(run/'logs/cuda_build_manifest.json');trace_index=read(run/'traces/g1b_trace_index.json');env=read(run/'environment.json')
    import xml.etree.ElementTree as ET
    pytest_suite=ET.parse(run/'logs/pytest_cuda_codec_a01.xml').getroot().find('testsuite')
    pytest_count=int(pytest_suite.attrib['tests']);assert int(pytest_suite.attrib['failures'])==int(pytest_suite.attrib['errors'])==0
    assert a['benchmark_validity']=='valid' and a['validation_error_count']==0 and a['trace_completeness']=='complete'
    assert accept['correctness_status']=='pass' and accept['execution_registration_verified']
    assert sha(run/'correctness.jsonl')==accept['correctness_sha256']
    assert len(correct)==len(read(run/'correctness_case_manifest.json')['expected_cases'])
    assert Counter(r['status'] for r in correct)=={'pass':len(correct)}
    assert sha(run/'codec_contract.json')==registration['contract_sha256']==accept['contract_sha256']
    assert len(bench)==208 and len(raw)==20800 and len(memory)==1040
    assert trace_index['status']=='pass' and len(trace_index['traces'])==8
    for p,h in registration['source_hashes'].items():assert sha(ROOT/p)==h,p
    for p,h in a['source_sha256'].items():assert sha(run/p)==h,p
    assert sha(ROOT/'scripts/summarize_g1b.py')==a['generator_sha256']
    by={(r['case_id'],r['method_id']):r for r in bench}
    main=[r for r in bench if r['method_id']==METHODS[1] and r['allocation_mode']=='preallocated' and r['validation_mode']=='trusted_inputs'
        and inputs[r['input_id']]['source_kind']=='synthetic' and inputs[r['input_id']]['generator_config'].get('distribution')=='standard_normal'
        and inputs[r['input_id']]['generator_config'].get('seed')==7301]
    if not main: # Generator configurations can encode the same fixed seed in the identity.
        main=[r for r in bench if r['method_id']==METHODS[1] and r['allocation_mode']=='preallocated' and r['validation_mode']=='trusted_inputs'
            and r['input_id'].startswith('synthetic_standard_normal_seed7301_')]
    selected=sorted([r for r in main if r['original_numel'] in (256,65536,4194304)],key=lambda r:(r['original_numel'],r['nominal_bits'],r['operation']))
    perf=[]
    for r in selected:
        ref=by[(r['case_id'],METHODS[0])]
        perf.append([r['nominal_bits'],r['original_numel'],r['operation'],fmt(ref['gpu_session_median_us_median']),fmt(r['gpu_session_median_us_median']),
            f"{fmt(ref['wall_session_median_us_median'])} / {fmt(r['wall_session_median_us_median'])}",
            f"{fmt(ref['gpu_session_median_us_min'])}–{fmt(ref['gpu_session_median_us_max'])} / {fmt(r['gpu_session_median_us_min'])}–{fmt(r['gpu_session_median_us_max'])}",
            ', '.join(fmt(s['speedup'],2) for s in r['paired_speedup_by_session'])])
    api_rows=[]
    for r in sorted([r for r in bench if r['method_id']==METHODS[1] and r['allocation_mode']=='api_allocating'],key=lambda r:(r['original_numel'],r['nominal_bits'],r['validation_mode'])):
        ref=by[(r['case_id'],METHODS[0])]
        api_rows.append([r['nominal_bits'],r['original_numel'],r['validation_mode'],fmt(ref['gpu_session_median_us_median']),fmt(r['gpu_session_median_us_median']),fmt(ref['wall_session_median_us_median']),fmt(r['wall_session_median_us_median']),', '.join(fmt(x['speedup'],2) for x in r['paired_speedup_by_session'])])
    large=[r for r in main if r['original_numel']==4194304 and r['operation']=='roundtrip']
    finding_rows=[];memory_rows=[]
    for r in sorted(large,key=lambda x:x['nominal_bits']):
        ref=by[(r['case_id'],METHODS[0])]
        finding_rows.append(f"N=4194304、b{r['nominal_bits']} roundtrip：PyTorch {fmt(ref['gpu_session_median_us_median'])} us，自定义 CUDA {fmt(r['gpu_session_median_us_median'])} us；五个 session 的配对比值 {fmt(r['paired_speedup_min'],2)}–{fmt(r['paired_speedup_max'],2)}，中位数 {fmt(r['paired_speedup_median'],2)}。")
        for method in METHODS:
            samples=[m for m in memory if m['case_id']==r['case_id'] and m['method_id']==method]
            med=lambda key:statistics.median(m[key] for m in samples)
            memory_rows.append([r['nominal_bits'],method,int(med('input_bytes')),int(med('threshold_bytes')),int(med('lut_bytes')),
                int(med('packet_buffer_bytes')),int(med('decoded_output_bytes')),int(med('workspace_bytes')),int(med('peak_extra_allocated_bytes')),int(sum(med(k) for k in ('input_bytes','threshold_bytes','lut_bytes','packet_buffer_bytes','decoded_output_bytes','workspace_bytes')))])
    real=[r for r in correct if r['method_id']==METHODS[1] and r['check_type']=='roundtrip' and r['input_id'] in inputs and inputs[r['input_id']]['source_kind']=='real_gradient']
    real_table=[]
    for bits in (4,8):
        group=[r for r in real if r['nominal_bits']==bits]
        real_table.append([bits,len(group),'0','0',f"{min(r['quant_normalized_squared_error'] for r in group):.9g}–{max(r['quant_normalized_squared_error'] for r in group):.9g}",group[0]['actual_wire_bytes']])
    ccounts=Counter(r['method_id'] for r in correct)
    noise=a['noise_case_counts_by_method'];duration={k:v for k,v in a['block_duration_diagnostics'].items() if k!='cases'}
    trace_rows=[]
    for t in trace_index['traces']:
        trace_rows.append([t['method_id'],t['nominal_bits'],t['original_numel'],str(t['kernels_per_operation']),t['kernel_count'],
            json.dumps(t['device_memcpy_direction_counts']),t['device_memcpy_bytes'],t['manifest_path']])
    drift=[]
    for method in METHODS:
        group=[r for r in bench if r['method_id']==method]
        maximum=max(group,key=lambda r:r['session_median_max_over_min'])
        drift.append(f"{method} 最大 session_median_max_over_min={fmt(maximum['session_median_max_over_min'],3)}，对应 `{maximum['case_id']}`。")
    gains=a['gain_classification_counts']
    provenance=read(run/'logs/correctness_development_attempts.json')
    issue_records=[
        dict(issue_id='G1B_DEVELOPMENT_HOST_ERROR_MATCH',category='test_harness',severity='low',status='resolved',description='首轮预实验22个native ABI错误字符串因附加C++栈诊断而未匹配，保留失败记录；只精确匹配已登记ABI主消息并保留完整栈，CUDA/OOM错误仍失败。',evidence_paths=['logs/correctness_preliminary_a01.jsonl','logs/correctness_development_attempts.json','logs/correctness_harness_changes.diff'],reproduction='See preserved preliminary a01 source and counterexamples.',fix_or_next_step='Final exact rejection helper and regression tests pass; no numeric expectations changed.'),
        dict(issue_id='G1B_LIFETIME_PROBE_POST_COMPLETION_REUSE',category='test_harness',severity='low',status='resolved',description='第二轮预实验2个额外检查生成器失败：探针把完成后的合法内存复用也当成错误；修正为每次分配后核验结果事件是否仍pending，保留原失败及4个未完成用例身份。',evidence_paths=['logs/correctness_preliminary_a02.jsonl','logs/correctness_preliminary_a02.summary.json','logs/correctness_development_attempts.json'],reproduction='See preserved a02 helper snapshot.',fix_or_next_step='a03 and formal run fully pass exact CPU comparisons and pending-lifetime tests.'),
        dict(issue_id='G1B_CROSS_STREAM_LIFETIME',category='implementation',severity='medium',status='resolved',description='a01 development build did not explicitly record allocator use on foreign streams; a02 records every kernel-used storage on current stream. Arithmetic CUDA source unchanged.',evidence_paths=['logs/cuda_build_a01.json','logs/cuda_build_a02.json','correctness.jsonl'],reproduction='Native lifetime cases release producer-allocated tensors and plan before pending consumer completes, then churn producer allocator.',fix_or_next_step='a02 is the frozen formal artifact; current-stream dependency and lifetime tests pass.'),
        dict(issue_id='G1B_BENCHMARK_NOISE',category='benchmark',severity='medium',status='open',description=f"{a['noise_case_count']}/208 method-cases and {a['noise_session_count']}/1040 case-sessions have IQR/median >10%. All observations retained; cross-session ratios reported separately.",evidence_paths=['benchmark_raw.jsonl','benchmark_summary.jsonl','analysis.json'],reproduction='Run summarize_g1b.py on retained raw records.',fix_or_next_step='Interpret all five paired ratios and session spread; do not discard slower data. Any retest requires new preregistered run.'),
        dict(issue_id='G1B_EXTERNAL_DEPENDENCIES',category='portability',severity='low',status='open',description='Small ZIP relies on external local fixture archives and toolchains. Prior toolchain catalog/channel metadata discrepancy remains a documented supply-record limitation.',evidence_paths=['fixtures_manifest.json','environment.json','REPRODUCE.md'],reproduction='Check listed external hashes and paths before reproducing.',fix_or_next_step='Provide fixtures/toolchain components separately or reconstruct from recorded sources; no full supply-chain verification claim.')]
    outside=duration['below_target_count']+duration['above_target_count']
    if outside:
        issue_records.append(dict(issue_id='G1B_BLOCK_DURATION_TARGET_MISS',category='benchmark',severity='low',status='open',description=f'{outside}/{len(raw)} blocks outside target10–50ms; K frozen before measurement, all data retained.',evidence_paths=['analysis.json','benchmark_raw.jsonl','benchmark_config.json'],reproduction='Recompute block_duration_diagnostics from raw.',fix_or_next_step='Diagnostic only, no posthoc K adjustment; any repeat uses new preregistered run.'))
    (run/'issues.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in issue_records),encoding='utf-8')
    source_hashes={name:sha(run/name) for name in names}
    summary=read(ROOT/'handoff_templates/summary.json')
    summary.update(template_only=False,run_id=run.name,stage='G1',substage='custom_cuda_paired',started_at=registration['registered_at'],finished_at=now(),
        execution_status='complete',contract_sha256=sha(run/'codec_contract.json'),benchmark_config_sha256=sha(run/'benchmark_config.json'),
        correctness_status='pass',benchmark_validity='valid',optimization_gain=a['optimization_gain'],g1_acceptance='pass',
        next_action='review_required',method_statuses=accept['method_statuses'],report_generator=dict(path='scripts/write_g1b_report.py',sha256=sha(__file__),source_sha256=source_hashes))
    summary['coverage'].update(nominal_bits_completed=[4,8],methods_completed=list(METHODS),correctness_cases_passed=len(correct),correctness_cases_failed=0,correctness_cases_skipped=0,
        real_gradient_inputs_completed=12,benchmark_cases_completed=104,benchmark_method_cases_completed=208,raw_blocks=len(raw),paired_blocks=10400,memory_records=len(memory),gpu_trace_configurations=8,missing_items=[])
    summary['claims'].update(local_codec_parity_validated=True,gpu_codec_performance_measured=True)
    summary['key_findings']=[f"{len(correct)} correctness/reference records pass; counts {dict(ccounts)}. These are not independent GPU experiments.",*finding_rows,
        f"Paired descriptive gain classes: {gains}; noisy method-cases: {noise}.",
        'Custom CUDA eliminates O(N) global int64 packing indices and arithmetic intermediates; actual memory components and trace counts retained.',
        'Full G1 local-codec deliverable complete; G2 has not been executed.']
    summary['artifacts'].update(analysis='analysis.json',figures=a['figures'],correctness_case_manifest='correctness_case_manifest.json',trace_index='traces/g1b_trace_index.json',toolchain_summary='logs/toolchain_summary.json',cuda_implementation_review='logs/cuda_implementation_review.json',cuda_build_manifest='logs/cuda_build_manifest.json')
    summary['open_issues']=[r['issue_id'] for r in issue_records if r['status']=='open']
    write(run/'summary.json',summary)
    finding_list='\n'.join('- '+x for x in finding_rows)
    report=f'''# GPU 低比特编解码实施报告

run_id：`{run.name}`。执行日期：2026-11-04，北京时间。技术方案版本：1.0；实施依据：[G1b 复核与执行要求](../../../reviews/gpu_codec_v1_20261101_002_review_and_g1b_plan.md)。

## 1 结论与完成范围

完成 b4/b8 **局部 codec 固定配置** 的自定义 CUDA 编码、解码和直接 5/9 位打包/解包，在同一新 run 与已验收 PyTorch 实现进行配对比较。正式正确性 {len(correct)} 条记录全部通过，20800 个计时块、208 个方法汇总、1040 个显存记录及小/大输入两位宽两实现的 8 份 GPU trace 完整。执行者判断完整 G1 验收通过，提交复核；本轮止于 G1。

{finding_list}
配对收益分类为 `{a['optimization_gain']}`，各 case 的描述性分类计数为 `{gains}`；分类按五个 session 的四分位区间关系生成，不作显著性检验，也不把计时块视为独立训练重复。

{table(['项目','状态','证据'],[['工具链与实际加载产物','编译/加载/真实GPU调用通过','logs/cuda_build_a02.json；logs/cuda_build_manifest.json'],['b4/b8固定语义','CPU码流与BF16位模式逐项通过','correctness.jsonl'],['真实梯度回放','12输入×2位宽×2GPU方法','input_manifest.json'],['性能/显存/trace','valid；所有记录保留','analysis.json'],['完整G1','pass，待接收方复核','summary.json']])}
## 2 实现与契约

冻结契约 SHA256：`{summary['contract_sha256']}`。CPU 参照仍为原项目 `index_repaired_bf16` 修复路径；上游 commit `96da2ee480bbfd8e2049de3fbd8595cc9d7a30b8`。CPU 原文件、LUT、阈值和 82 个夹具身份逐项记录；002 输入与原源码保持只读。旧 GPU 基线 `torch_codec.py` SHA256 固定为 `50bc272876be54db5d42327cce144022b7bf960d82073347479da2eb53f1d794`。

新实现位于 `src/gradient_comm_lab/cuda_codec.py` 与 `csrc/codec.cpp/.cu`。每个 256 元素顶层组由一个 256 线程 CTA 处理，16 元素尺度在半 warp 中归约，尺度和码值暂存在共享内存；每个输出字节有唯一线程写入。编码包含分组尺度、LUT 搜索、显式阈值舍入、尺度量化和真实紧凑打包；解码包含解包、尺度恢复和反量化。热路径每次 encode/decode 各调用一个自定义内核，roundtrip 顺序调用二者；实际次数以第 5 节 trace 为准。直接 pack/unpack 是已测试的开发模块，没有另设一个“只换打包”的主比较方法。

每一步浮点运算先以受控 FP32 执行，再在 CPU 参照规定边界舍入到 BF16；常数先取 BF16，严格 `<` 比较、整数 LUT 偏移和负零均保持。编译使用 CUDA 11.8、MSVC 19.38，实际选项 `{build['extra_cuda_cflags']}`；无 fast-math。冻结实际加载模块 `{build['module_name']}`，SHA256 `{build['module_sha256']}`。原始命令、ninja 文件、编译器输出及 a01/a02 源码快照保留在 build/logs。

`CudaCodecPlan` 预分配 packet/output/status/epsilon；`encode_into`、`decode_into`、`roundtrip_into` 默认 trusted。三个 packet buffer 依次为低位优先 payload、uint8 scale codes、little-endian BF16 top scales，逻辑序列化与 CPU 完全一致。N 补齐至 256 倍数，编码在内核中读逻辑范围并补零，解码裁回 N；N=0 跳过原生 codec，零字节、零量化事件。

checked API 单独检查 dtype/shape/device/连续性/别名、有限性及阈值范围；内核通过显式 status word 反馈算术问题。错误索引在加载 LUT 之前限界，非法包不会先越界再依赖异常拒绝。底层 ABI 在任何模式都检查元数据与输出别名；其已登记 host `ABI_*` 错误和 CUDA 启动/非法访问/OOM 错误严格区分。使用张量设备上的 PyTorch 当前 stream 与 CUDAGuard，并记录 allocator 跨 stream 使用，延长临时 storage 生命周期。trusted 模式要求已资格验证的数值输入，不作同步数值检查。两方法 checked API 各自的检查成本属于包含分配的独立 API 测量。

## 3 正确性结果

正式记录按独立清单核对，共 {len(correct)} 条：PyTorch {ccounts[METHODS[0]]} 条、自定义 CUDA {ccounts[METHODS[1]]} 条、CPU 参照执行 {ccounts['cpu_ref_repaired']} 条，全部 pass，fail=0、skipped=0、缺失/重复=0。这个总数是检查和参照执行记录数，不是独立 GPU 实验次数。

82 输入含 36 合成、34 边界、12 真实开发梯度；每个 GPU 方法继承跨 b4/b8 的全部 117 条协议检查；每个输入/位宽另含 5 类检查。GPU 解码独立接收 CPU 产生的包；完整码流、主码、尺度码、顶层尺度和输出 BF16 位模式逐项一致，容差为零。输入/阈值/LUT 通过独立字节副本保护检查。所有 164 个 CPU 输入位宽组合与两种 GPU 方法的量化失真字段一致。

新增逐用例固定异常规则：继承的 60 个非法输入明确要求 ValueError 和精确消息；11 个底层 ABI 用例/位宽要求精确 host RuntimeError 主消息，附加 C++ 栈完整保留。意外 CUDA 错误会失败。另有非默认 stream 生产依赖、预分配及含分配临时生命周期、输出/码流 canary 边界检查。生命周期探针在 consumer 仍 pending 时释放输入与 plan，并在 producer stream 分配扰动；仅等待结果 event，不使用全设备同步。没有执行 compute-sanitizer，不将这些用例升级为工具级内存安全证明。

真实梯度来自 seed 1101、update 100/500/2500 的更新前、rank 0..3；原 FP32 长度 818048，转换次序为 FP32→BF16→BF16 除以 4→补零到 818176。未使用新训练或 holdout。

{table(['位宽','真实输入数','包差异字节','输出位差异','归一化平方误差范围','单包B'],real_table)}
预实验失败全部保留：a01 的 22 条失败来自 host ABI 错误消息附带栈诊断；a02 的 2 条生成器失败来自生命周期探针误拒绝已完成后合法内存复用，导致后续 4 个身份未覆盖。两处仅修正测试助手，未修改参考输出、随机阈值或数值容差。a03 预实验 {len(correct)} 项通过，正式冻结后再次 {len(correct)} 项通过；pytest {pytest_count}/{pytest_count} 通过。a01→a02 native 改动为 allocator stream 生命周期记录，算术 CUDA 源码未变。详情见 `logs/correctness_development_attempts.json`、助手 diff/hash 和完整反例文件。

## 4 GPU 性能结果

主矩阵为 N=256/4096/65536/262144/1048576/4194304，两种分布、b4/b8、encode/decode/roundtrip，共 72 case；另有 8 个包含分配/checked API case 和 24 个真实梯度 roundtrip case。104 case 均对两种方法测量 5 session、每 session 20 块，逐 case/session 使用预先生成的平衡交替顺序及配对 block ID。每个 `(case_id, method_id)` 的 K 在正式测量前按固定 pilot 规则冻结，各自块时除以自身 K；同一次新 run 的 PyTorch 是收益分母。

每个 case/session 两方法独立预热并在开始前清空待执行工作；主路径预分配、trusted、eager、Graph off、steady-state reuse。输入、LUT、阈值生成和 H2D 不计入主计时，初始化成本另记。包含分配模式每次重建完整 plan，比较边界一致；checked 检查与同步只计入单列的 checked API。正式计时关闭未初始化内存调试填充，开启确定性算法，全部 workspace 写后读。

GPU CUDA event 与带边界同步的墙钟同时记录。event 区间可包含 eager 发射间隙及 API 分配/同步等待，不等于 kernel 活跃时间之和。单次延迟来自 elapsed/K；session 线性分位数与五个 session 中位数的中位数、范围分别报告，p95 是“块平均单次延迟 p95”。以下为标准正态 seed 7301；范围列为 PyTorch / CUDA 的 session 中位数范围，配对比依 session 0..4 展示。

{table(['b','N','操作','PyTorch GPU us','CUDA GPU us','墙钟 PyTorch/CUDA us','两方法GPU session范围 us','5个配对比'],perf)}
包含分配的 8 个 API case 单列如下，均为直接 roundtrip；两边每次创建完整 plan。checked 与 trusted 独立，不与主矩阵预分配结果混合。

{table(['b','N','检查模式','PyTorch GPU us','CUDA GPU us','PyTorch wall us','CUDA wall us','5个配对比'],api_rows)}
噪声：PyTorch {noise[METHODS[0]]}/104、自定义 CUDA {noise[METHODS[1]]}/104 case 至少一个 session 的 IQR/median>10%；合计 {a['noise_session_count']}/1040 case-session。保留全部数据，不据此剔除。{''.join(drift)} 会话内 noise_flag=false 不等于跨会话稳定。pilot 规则、K、时钟/负载/WDDM 快照、块长偏离和全部墙钟值见 config/raw/analysis；块长诊断为 `{json.dumps(duration,ensure_ascii=False)}`。

收益分类由各 session 的四分位区间分离情况描述；差异不足以覆盖波动的 case 标为 inconclusive。不得只挑最大加速比，也不对不同输入/API 边界合并成一个“整体加速”。图表可由 `scripts/summarize_g1b.py` 重建：

![延迟](figures/latency_vs_numel.png)

![五个session配对收益](figures/paired_speedup_vs_numel.png)

## 5 字节、显存与性能剖析

对补齐长度 L，完整 wire bytes=`ceil(L*(b+1)/8)+L/16+2*L/256`；b4/b8 实际存储 5/9 位主码。L=256 时分别 160+16+2=178 B、288+16+2=306 B，即 5.5625/9.5625 bits/element。尾块按真实逻辑 N 作有效位数分母，padding 已包含在 payload/scale 中，不另加第二份 padding。roundtrip 的 wire 大小指一个包，不代表网络传输。

下表为 N=4194304、标准正态、预分配 roundtrip，5 个 session 显存独立测量中位数（单位 B）。workspace 包含唯一 storage，views 不重复计数；原 PyTorch 的 O(N) int64 索引/临时 word/算术数组被 CUDA 寄存器/共享内存局部计算取代。

{table(['b','方法','输入','阈值','LUT','包','输出','workspace','peak extra allocated','列示device张量总计'],memory_rows)}
allocator allocated/reserved 与唯一张量存储组成分别记录。预分配基线已包括整个 plan，所以 peak extra=0 也不代表没有工作内存；寄存器/CTA共享内存和驱动内部资源不计入 PyTorch tensor storage。ptxas 编译记录显示 encode 每CTA共享内存 580 B，b4/b8 分别每线程 30/29 个寄存器，decode 为 16 个寄存器，无 spill；这些是编译资源记录，不与全局张量 workspace 相加。配对计时同时驻留两套各自相同输入与 plan，可能影响缓存及显存压力；独立显存探针只持有当前方法，故其 allocator 基线不是配对计时时总占用。CUDA workspace 包含持久 status 和 epsilon，不宣称零工作区。详细 allocated/reserved、峰值、输入/阈值/LUT/包/输出与总 storage 见 `memory.jsonl`。

分别剖析 N=65536 和 4194304、b4/b8、两种方法，每份三次 roundtrip。GPU 活动来源为项目局部 CUPTI 12.9 collector，运行中的 PyTorch CUDA runtime 仍为 11.8。原始活动、Chrome 窗口、external correlation 和审计日志逐项对应，所有 trace 必须无丢记录、无采集错误、所有 kernel 归属明确才进入以下统计。profiler 时间不作为正式延迟样本。

{table(['方法','b','N','每次kernel数','3次kernel总数','copy方向次数','copy总B','manifest'],trace_rows)}
源码和显存测量确认减少了全局中间张量；两个长度的实际 trace 用于核对内核调用及复制活动变化，延迟收益来自第 4 节的独立配对测量。本 collector 未测 DRAM 事务量，不能仅凭内核次数断言大输入的唯一瓶颈，也不能将输入等效 GB/s 解释为 DRAM 带宽或网络吞吐。无主机传输的判断只覆盖已明确划定的三次调用窗口，初始化/H2D 在窗口外。

![实际位数](figures/wire_effective_bits.png)

![量化失真](figures/quantization_distortion.png)

![显存](figures/memory_vs_numel.png)

## 6 未解决问题与下一步

{table(['问题','影响','证据','建议'],[['WDDM桌面噪声及跨session漂移','小收益可能不确定，所有样本保留','raw/summary/analysis','按全session配对比及波动解释，需要复测则建新run'],['外部夹具与本机工具链依赖','小ZIP不是跨机器独立环境','fixtures_manifest/REPRODUCE','按清单取得外部文件；不改历史目录'],['旧供应catalog/channel不一致记录','不能声明完整供应链全部验证','environment与002工具链日志','沿用明确边界，不影响已核验的本地模块调用']])}

建议 `review_required`：复核完整 G1 证据后再决定 G2。后续若进入 G2，应单独恢复跨 worker 聚合、随机状态及训练更新语义；这些未在本轮实现。没有“必须快多少”的验收门槛，当前收益只对本机和实测实现比较成立。

## 7 复现与结论边界

[REPRODUCE.md](REPRODUCE.md) 提供实际执行命令及日志。完整目录：`{run.as_posix()}`。源码清单、夹具外部路径与 SHA256、编译产物身份及原始 trace 均保留；小型 ZIP 可排除编译缓存和外部大夹具，但应带明确清单。

本轮证明指定 GPU、输入、固定数值契约下局部 codec 的逐位正确性和成本。它不是完整 DynamiQ GPU 复现、先进 codec 优越性证明、AllReduce 加速或大模型训练加速；新训练质量、多 GPU 网络吞吐和训练加速字段仍为 false。旧训练结果未纳入本轮新增验证。

## 8 交付文件核对

- [x] summary、环境、冻结契约、基准配置填写完整，template_only=false。
- [x] CPU/GPU源码、实际编译模块、测试、配置、输入与夹具身份可核验。
- [x] 正确性清单、20800计时块、208汇总、1040显存与配对关系完整。
- [x] 5张图从原始记录生成；8组小/大输入GPUtrace及原始活动审计完整。
- [x] 实际复现命令、构建日志、失败尝试、测试助手改动及原始反例保留。
- [x] G2及以上未执行，结果边界和开放问题已明确。最终封存完整性与ZIP核对由独立seal/package脚本生成。
'''
    (run/'REPORT.md').write_text(report,encoding='utf-8')
    commands=read(run/'logs/execution_commands.json')
    reproduce=f'''# G1b 复现与审计入口

工作目录：`{ROOT.as_posix()}`。解释器：`E:/dacomp/.venv/Scripts/python.exe`，通过 `scripts/python.ps1` 只读复用，所有缓存写本项目 `.cache/`；没有升级旧环境或驱动。正式运行使用 CUDA 11.8/MSVC 19.38/SDK 10.0.26100.0、RTX 4000 Ada、PyTorch 2.4.1+cu118，精确当前环境以 environment.json 为准。

## 实际执行命令

以下结构直接取自 `logs/execution_commands.json`，表示本轮执行记录，包含开发失败。原 console/raw/JSON 均可检索。

```json
{json.dumps(commands,indent=2,ensure_ascii=False)}
```

## 冻结与正确性

正式顺序为：新run引用002的82夹具及哈希 → build a01/开发检查 → a02加入跨stream allocator记录 → 预实验a01/a02失败与a03通过 → pytest{pytest_count}通过 → 注册冻结{len(registration['source_hashes'])}执行源码/正确性清单/契约 → 正式{len(correct)}项通过 → pilot冻结K → 双方法配对计时 → 独立GPUtrace → 汇总作图 → 报告/源码收集/独立审计封存。

`correctness_acceptance.json` 绑定每个正式结果的契约哈希；`logs/correctness_start_identity.json` 记录实际加载模块和起始源码快照。`logs/correctness_harness_changes.diff/json` 记录测试助手差异，不改原阈值、输出或容差。重新执行不能回写已封存run，需生成唯一新目录。冻结的 `check_cuda_codec.py` 默认 CUDA loader 指向本次run，而非自动跟随 --run-dir；本次正式结果已核对实际加载a02，因此结果身份不受影响。新run必须先调用 `gradient_comm_lab.cuda_codec.set_build_manifest(new_run / "logs/cuda_build_manifest.json")` 再调用 `check_cuda_codec.main()`。另提供新入口 `scripts/check_cuda_codec_replay.py --run-dir <new_run> ...` 自动做此选择；该shim在正式矩阵之后添加，仅执行过CPU --help检查，未声称完整重跑。复制封存源码时保留相对目录布局和旧参考依赖。

## 夹具与依赖

`input_manifest.json` 和 `fixtures_manifest.json` 引用本机已有 `.pt` 夹具，逐文件与张量哈希校验；本轮未重新生成或改变它们。生成逻辑在封存 `fixtures_v2.py`，真实梯度还依赖原项目已有 checkpoint/回放来源；生成入口是该模块 `prepare_fixtures(new_run)`，本轮执行的是读取和校验既有夹具，不能把重建命令写成已验证跨机器复现。

CPU 参照原目录 `E:/dacomp/research/spl_model_communication/` 为只读外部依赖；其完整路径/哈希见契约及源码清单。不要运行原 `tests/test_dynamiq.py`，该旧测试存在写 provenance 的fixture。CUDA/MSVC/CUPTI组件位于本项目 `.toolchains/`，Windows SDK/Ninja依赖已安装系统组件。ZIP不携带完整环境；外部文件不可得时须先获取或重建，不能仅解压声称可独立运行。

## 测量与复算

`benchmark_cuda_pair.py` 分 register/pilot/measure 阶段，pilot规则先冻结、K只在正式测量前决定；正式数据不调K、不删慢块。两种实现同case/session交替，逐块配对ID在 raw。`summarize_g1b.py` 自动复算统计、paired speedups、噪声、漂移、显存和5张图。`collect_g1b_traces.py` 对8配置分别调用profile/audit，窗口剔除初始化和warmup但保留原始活动，核验零丢记录及逐项raw/Chrome对应。图和trace只是解释性证据，正式性能来自无profiler记录。

`seal_g1b_delivery.py --seal` 独立复核原始结果、源码、夹具、统计、trace与旧封存文件；`package_g1b_delivery.py` 创建含清单的小ZIP。封存后的run不可修改，再测必须另建run。
'''
    (run/'REPRODUCE.md').write_text(reproduce,encoding='utf-8')
    write(run/'logs/report_generation.json',dict(run_id=run.name,generated_at=now(),command=[sys.executable,*sys.argv],generator_sha256=sha(__file__),source_sha256=source_hashes,
        outputs={n:sha(run/n) for n in ('REPORT.md','summary.json','issues.jsonl','REPRODUCE.md')}))
    print(json.dumps(dict(run_id=run.name,status='complete',correctness_records=len(correct),benchmark_records=len(raw),optimization_gain=a['optimization_gain'])))
if __name__=='__main__':main()
