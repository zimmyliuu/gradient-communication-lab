"""Generate the bounded, negative-result delivery from independently verified D1.

This post-comparison reporting source does not change the frozen experiment.
"""
import argparse,json,sys,shutil,statistics
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from gradient_comm_lab.bounded_discovery.common import read,dump,sha,now

def rows(path):return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
def fmt(v):return 'null' if v is None else f'{v:.6g}'
def stat(r,key):
    s=r['statistics'][key]
    return fmt(s['mean'])+(' ± '+fmt(s['sample_std']) if s['sample_std'] is not None else '（确定性）')
def write(path,text):
    with path.open('x',encoding='utf-8') as f:f.write(text)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',required=True);a=ap.parse_args();run=(ROOT/a.run_dir).resolve()
    analysis=read(run/'summary_analysis.json');assert analysis['status']=='valid' and analysis['validation_error_count']==0
    assert analysis['tensor_recomputation']['local_rows_checked']==780 and analysis['tensor_recomputation']['aggregate_rows_checked']==195
    audit=read(run/'logs/revision_equivalence_audit.json');assert audit['status']=='pass'
    contract=read(run/'experiment_contract.json');q=read(run/'baseline_qualification.json');execution=read(run/'logs/d1_execution.json')
    assert execution['execution_complete'] and execution['failed_records']==0
    summary_rows=rows(run/'metrics_summary.jsonl');raw=rows(run/'metrics_raw.jsonl');packets=rows(run/'packet_accounting.jsonl');checks=rows(run/'correctness.jsonl')
    assert len(raw)==975 and len(packets)==780 and all(r['status']=='pass' for r in checks)
    by={(r['record_type'],r['snapshot_id'],r['method_id']):r for r in summary_rows if r['record_type']!='local'}
    snapshots=sorted({r['snapshot_id'] for r in raw});methods=contract['methods'];method_map={m['method_id']:m for m in methods}
    def mean(typ,s,m,field):return by[typ,s,m]['statistics'][field]['mean']
    comparisons=[]
    for s in snapshots:
        low='official_dynamiq_fixed_codec_b4';high='official_dynamiq_fixed_codec_b8'
        comparisons.append(dict(snapshot_id=s,
            local_low=mean('snapshot_local',s,low,'weighted_normalized_squared_error'),
            aggregate_low=mean('one_shot_codec_sum',s,low,'source_normalized_error'),
            local_uniform_low_minus=mean('snapshot_local',s,'uniform_g128_q4','weighted_normalized_squared_error'),
            local_uniform_low_plus=mean('snapshot_local',s,'uniform_g128_q5','weighted_normalized_squared_error'),
            high_uniform_to_official_local_ratio=mean('snapshot_local',s,'uniform_g128_q8','weighted_normalized_squared_error')/mean('snapshot_local',s,high,'weighted_normalized_squared_error'),
            high_uniform_to_official_aggregate_ratio=mean('one_shot_codec_sum',s,'uniform_g128_q8','source_normalized_error')/mean('one_shot_codec_sum',s,high,'source_normalized_error')))
    assert all(c['local_low']<c['local_uniform_low_minus'] and c['local_low']>c['local_uniform_low_plus'] for c in comparisons)
    assert comparisons[0]['high_uniform_to_official_local_ratio']<1 and all(c['high_uniform_to_official_local_ratio']>1 for c in comparisons[1:])
    reason='有效完成既定有限比较；低精度存在实际字节与误差取舍，高精度均匀参照只在update100占优、另两快照反转，尚无具体触发机制与可行动干预证据；本次有限诊断未找到可推进线索。'
    decision=dict(decision='no_actionable_signal',reason=reason,comparisons=comparisons,
        final_contract_sha256=sha(run/'experiment_contract.json'),raw_sha256=sha(run/'metrics_raw.jsonl'),summary_sha256=sha(run/'metrics_summary.jsonl'),
        candidates_meeting_all_five_requirements=0,diagnostic_controls_executed=0,d2='not_run',
        screening=[dict(id='low_budget_tradeoff',outcome='not_actionable',reason='Lower uniform neighbors lose accuracy; higher g128q5 neighbor uses more bytes. Not equal-budget victory.'),
            dict(id='high_precision_crossover',outcome='not_actionable',reason='g128q8 wins only update100, reverses on update500/2500. Consistent with possible distribution/scale/representative tradeoffs; cause not isolated and no localized defect identified.'),
            dict(id='official_rng_endpoint',outcome='D0_boundary_only',reason='BF16 rand=1 zero endpoint matches official reference. No practical real-gradient effect isolated; not promoted to a finding.')],
        statistical_scope='Three fixed development snapshots, four workers each, eight PRNG repetitions. No independent trajectory confirmation or cross-method matched randomness.')
    dump(run/'logs/decision_evidence.json',decision)
    # Preserve historical failures explicitly, with their resolution and evidence.
    issues=[
        dict(issue_id='build_a01_encoding',stage='D0',status='resolved',kind='build_failure',detail='GBK source decoding failed before compilation; a02 used Python UTF-8 and host UTF-8.',evidence='logs/official_build_a01.json'),
        dict(issue_id='build_a02_missing_curand',stage='D0',status='resolved',kind='build_failure',detail='Missing curand_kernel.h; a03 used verified isolated NVIDIA headers. No further repair rounds.',evidence='logs/official_build_a02.json'),
        dict(issue_id='uniform_a01_empty_hash_harness',stage='D0',status='resolved',kind='test_harness_failure',detail='18 empty-output identity checks failed on dtype view of stride-zero empty tensors; harness fixed, quantizer unchanged; a02 205 checks passed.',evidence='logs/uniform_selfcheck_a01.json'),
        dict(issue_id='runner_v1_audit_guards',stage='D1',status='resolved',kind='audit_robustness',detail='Preserve failures and require complete aggregate count. Entire v1 retained; unchanged matrix rerun under contract v2; packets and numerical results equivalent.',evidence='logs/contract_revision_history.json'),
        dict(issue_id='rng_endpoint_scope',stage='D0',status='documented',kind='official_semantic_boundary',detail='Legal FP32-to-BF16 random pool may contain 1; zero endpoint behavior agrees with official reference. No general unbiasedness claim.',evidence='logs/d0_execution.json'),
        dict(issue_id='resource_clock_scope',stage='delivery',status='documented',kind='measurement_limit',detail='Resource upper bounds include wall windows; no exact total kernel clock or algorithm speed comparison measured.',evidence='logs/d1_execution.json')]
    with (run/'issues.jsonl').open('x',encoding='utf-8') as f:
        for r in issues:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    # All 108 fixed-input distributions, including deterministic std=null.
    appendix=['# 每输入随机结果\n','来源：metrics_summary.jsonl 的 local 记录；均值 ± sample std，确定性方法 std=null；另列 min/max。所有输入为 development。\n',
        '| 快照/worker | 方法 | SSE | MSE | SSE/energy | norm min–max | max abs error | valid/failed/missing |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summary_rows:
        if r['record_type']!='local':continue
        st=r['statistics']['normalized_squared_error']
        appendix.append(f"| {r['snapshot_id']}/{r['worker_id']} | {r['method_id']} | {stat(r,'squared_error')} | {stat(r,'mse')} | {stat(r,'normalized_squared_error')} | {fmt(st['min'])}–{fmt(st['max'])} | {stat(r,'max_abs_error')} | {r['valid_repeats']}/{r['failed_repeats']}/{r['missing_repeats']} |")
    appendix+=['','完整 input_id 与能量、包字节、每次repeat见 input_manifest.json、metrics_summary.jsonl、metrics_raw.jsonl。稳定键为 local + snapshot_id + worker_id + method_id + repeat_id。']
    write(run/'PER_INPUT_RESULTS.md','\n'.join(appendix)+'\n')
    budget_table=['| 方法 | 实际码宽 | P | payload B | scale B | header B | 总 B/worker | 相对位置 |','|---|---:|---:|---:|---:|---:|---:|---|']
    for m in methods:
        p=next(r for r in packets if r['method_id']==m['method_id'])
        budget_table.append(f"| {m['method_id']} | {m['stored_code_bits']} | {p['padded_numel']} | {p['payload_bytes']} | {p['scale_bytes']} | {p['header_bytes']} | {p['packet_total_bytes']} | {m['budget_role']} |")
    results=[]
    for s in snapshots:
        results+=['',f'### {s}','','| 方法 | B/worker | 局部 ΣSSE/Σenergy | one-shot SSE/source_energy |','|---|---:|---:|---:|']
        for m in methods:
            mid=m['method_id'];l=by['snapshot_local',s,mid];g=by['one_shot_codec_sum',s,mid]
            results.append(f"| {mid} | {m['packet_total_bytes']} | {stat(l,'weighted_normalized_squared_error')} | {stat(g,'source_normalized_error')} |")
    energies=['| 快照 | source_energy | target_energy | cancellation_ratio | official b4 aggregate SSE | b4 target_relative_error | b8 aggregate SSE | b8 target_relative_error |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in snapshots:
        b4=by['one_shot_codec_sum',s,'official_dynamiq_fixed_codec_b4'];b8=by['one_shot_codec_sum',s,'official_dynamiq_fixed_codec_b8']
        energies.append(f"| {s} | {fmt(b4['statistics']['source_energy']['mean'])} | {fmt(b4['statistics']['target_energy']['mean'])} | {fmt(b4['statistics']['cancellation_ratio']['mean'])} | {stat(b4,'aggregate_squared_error')} | {stat(b4,'target_relative_error')} | {stat(b8,'aggregate_squared_error')} | {stat(b8,'target_relative_error')} |")
    finished=now();d0_hours=q['resource_usage']['conservative_active_upper_bound_hours']
    d1_hours=(datetime.fromisoformat(finished)-datetime.fromisoformat(read(run/'logs/task_start.json')['recorded_at'])).total_seconds()/3600
    summary=read(run/'summary.json');summary.update(execution_status='complete',d0_status='pass',d1_status='complete',correctness_status='pass',
        packet_accounting_status='pass',local_comparison_status='valid',aggregate_comparison_status='valid',timing_status='not_run',decision='no_actionable_signal',
        decision_reason=reason,contract_sha256=sha(run/'experiment_contract.json'),completed_at=finished,contract_revision=2)
    summary['baseline'].update(executed_commit=contract['upstream_commit'],entrypoints=contract['official_entrypoints'],arithmetic_modified=False,
        actual_module_sha256=q['loaded_module']['module_sha256'],qualification_evidence='baseline_qualification.json')
    summary['coverage'].update(completed_snapshots=3,completed_worker_inputs=12,method_configuration_count=9,valid_local_records=780,valid_aggregate_records=195,failed_records=0,missing_records=0,missing_coverage=[])
    summary['claims']['official_local_codec_executed']=True
    summary['finding'].update(candidate_count=0,selected_finding_id=None,finding_path=None,nearest_work_check='logs/nearest_work_source_review.json',confirmation_plan_path=None)
    summary['resource_usage'].update(d0_active_work_hours=d0_hours,d1_active_work_hours=d1_hours,
        gpu_execution_hours=execution['cumulative_GPU_upper_bound_seconds']/3600,build_repair_rounds=2,
        measurement_note='Conservative elapsed upper bounds, not exact active/kernel time or summed parallel labor. D1 charges the entire task window to include preparation overlapping D0. GPU upper bound charges entire D0 wall window plus synchronized D1 windows and freeze; includes both revisions.',
        d1_final_execution_wall_seconds=execution['wall_seconds'],d1_final_GPU_window_seconds=execution['gpu_execution_upper_bound_seconds'],
        d1_prior_GPU_window_seconds=contract['previous_execution_GPU_upper_bound_seconds'])
    summary.update(d2_status='not_run',diagnostic_controls_status='not_run',diagnostic_controls_count=0,
        not_produced=['findings/FINDING.md (no qualifying finding)','diagnostic_config.json / diagnostic_raw.jsonl (not_run)',
                      'benchmark_config.json / benchmark_raw.jsonl / benchmark_summary.jsonl (not_run)','D2 confirmation plan (not applicable to no_actionable_signal)'],
        evidence_paths=['REPORT.md','PER_INPUT_RESULTS.md','baseline_qualification.json','experiment_contract.json','source_manifest.json','input_manifest.json','environment.json',
            'correctness.jsonl','packet_accounting.jsonl','metrics_raw.jsonl','metrics_summary.jsonl','summary_analysis.json','logs/revision_equivalence_audit.json',
            'logs/decision_evidence.json','REPRODUCE.md','integrity_manifest.json'],next_action='stop_after_report; review required for any new scope')
    dump(run/'summary.json',summary,exclusive=False)
    high_ratios=', '.join(f"{s.rsplit('update',1)[1]}: {c['high_uniform_to_official_local_ratio']:.4f}" for s,c in zip(snapshots,comparisons))
    text=f'''# 低比特梯度通信诊断报告

状态：已执行、完整交付。run_id：{run.name}；日期：2026-11-04。完成 **D0 + D1**；单因素定位 0 项，D2 未执行。

## 1 结论和下一步

**decision = no_actionable_signal。** {reason}

官方固定 codec 资格通过；最终修订完成 3 个开发快照、每快照 4 个 worker、9 个配置、8 次随机重复（BF16 identity 1 次）。780 条局部记录及 195 条 one_shot_codec_sum 记录全部有效，无失败或缺失。独立复算确认原始数值与保存的 BF16 输出相符。

本轮没有达到计划五项条件的问题线索，不追加新数据或干预来寻找正结果，也不编写 D2 确认方案。结论只适用于这条小型 Transformer 训练轨迹的固定组件诊断，不能推出不存在独立贡献，也不能否定完整 DynamiQ 的动态配置、通信协议或训练效果。

## 2 官方基线资格

仓库：https://github.com/CharlesHan24/DynamiQ；commit `{contract['upstream_commit']}`。实际入口 `eden_utils.scaling_compress/scaling_decompress`，固定 BF16、strategy=3、chunk=16，主码实际存储 4/8 位。它与 G1 本地 5/9 位 profile 有不同语义和预算，本轮未把本地实现更名为官方。

构建二进制 SHA256：`{q['loaded_module']['module_sha256']}`。a01 被 GBK 读源码阻断；a02 在 UTF-8 与移除两个未用 POSIX include 后遇到 cuRAND 头缺失；a03 加入项目隔离的 NVIDIA cuRAND 10.3.0.86 头文件后成功。正好两轮修复，官方 `--use_fast_math`、尺度/主码/舍入/输出算术未改。原始源码、许可证、diff、失败日志均保留。共享 PyTorch、驱动及旧封存环境未升级。

原官方 48 配置功能自测通过，保留原容差；7 类输入×2 档共 14 组小型检查、新进程仅包+静态配置解码、CPU 独立逐字节拆包/repack 均通过。uniform 合成检查最终 205 项通过；a01 的 18 个空张量哈希测试框架失败及修复均保留。详见 [资格核验](baseline_qualification.json)、[独立源码核验](logs/official_contract_review.json)、[问题记录](issues.jsonl)。

解码静态状态包含 dtype、位宽、逻辑/补齐长度、chunk/strategy 及源码生成 LUT。官方 9,288 B 是源码定义的设备表数据量，单列且非每包传输；它不是 allocator 实测或总显存。编码端随机池为临时本地状态，不用于解码。极值检查只证明已测样例；合法 BF16 RNG 可能取到 1，全零端点探针的约 1e-7 输出符合官方参考，未伪装成“全零必精确”或全域无偏证明。

## 3 冻结比较条件

仅使用 seed=1101、update=100/500/2500 的已验收 development 输入，共 12 份。来源及文件/逻辑张量 SHA256 见 [input_manifest](input_manifest.json)。读取 `input_logical`：原始 FP32→BF16→BF16 除以4 的结果，D=818048，不再次预除，不把旧尾部当成输入。官方 P=818176；本轮 uniform 的 D 可整除 g，无新增尾部。所有最终输出为 BF16，指标只算逻辑部分。

最终契约 revision=2，SHA256 `{sha(run/'experiment_contract.json')}`。修订只补足失败反例保存和完整结束判定，保留 [v1 全部证据](revisions/contract_v1)。9 个配置、输入、预算和 seed 映射与 v1 完全相同；重跑的实际包、BF16 输出及误差已逐项核对。没有观察误差后重新选择预算。

预算先由实际编码字节确定：C_low=466616、C_high=875704 B/worker。uniform q 为实际码宽，FP32 scale 使用 FP64 覆盖检查与 nextafter 修正，FP32 归一化/概率及乘法后 BF16 RNE；16 B 真实头部计入预算。below/above 预算关系不等于等预算比较，所有 alignment/per-input auxiliary=0。

{chr(10).join(budget_table)}

随机 seed = SHA256(`bounded_discovery_v1:input_id:method_id:repeat_id`) 前8字节小端 mod 2^63。768 个随机任务均为独立构造且 seed 无重复。官方沿原自测 CPU FP32 rand→BF16→CUDA 路径；uniform 使用 CPU FP32 U∈[0,1)。按输入配对，不声称跨方法 matched randomness，不把 PRNG 重复当作独立训练轨迹。

## 4 数值和字节结果

以下为均值 ± sample std；确定性 std=null。局部每次先汇总四 worker SSE 和 energy 再求比，未平均四份相对误差。one-shot 用同四份解码结果与输入分别在 FP64 求和，不追加量化调用。其字节为表中 B/worker 的4倍，字段 `one_shot_packet_bytes_sum`，不是网络总流量或 ring AllReduce。

{chr(10).join(results)}

### 输入能量与绝对误差

{chr(10).join(energies)}

三个 cancellation_ratio 均远大于1e-6，未触发 near_cancellation。cross_term 与全部 SSE/MSE/max_abs、逐重复 min/max 均在原始和汇总文件中；单次非零 cross_term 不解释为已证明相关噪声。BF16 identity 的 12 条局部及3条 one-shot SSE 严格为0，仅检验本次 FP64 量化诊断，不意味着 BF16 网络归约无加法舍入。

[每输入完整表](PER_INPUT_RESULTS.md)覆盖108个输入×方法分布，含离散与 min/max；[机器汇总](metrics_summary.jsonl)含162组统计。稳定 raw 键：`record_type + snapshot_id + worker_id + method_id + repeat_id`，聚合 worker_id=null。每份包与解码结果可由 packet_id 定位 `packets/`、`decoded/`，并绑定契约/input_manifest 哈希。没有筛掉不利 worker 或重复。

![实际字节与局部误差](figures/local_error_vs_actual_bytes.png)

![实际字节与一次编码聚合误差](figures/one_shot_error_vs_actual_bytes.png)

性能测量：not_run。Python uniform 与官方 CUDA 的实现成熟度不同，未作速度竞赛、网络加速或完整系统吞吐结论。

## 5 问题线索和定位

筛查记录见 [decision_evidence](logs/decision_evidence.json)。没有选定主要发现，也未执行单因素对照。

1. 低位档：官方 b4 的三个局部归一化误差约0.01221/0.01417/0.01396；g128q4 下邻误差更大，g128q5 上邻误差更小但多付约15.05%字节。g16q3 则比官方 b4 包大且误差更大。这些显示位宽、组粒度与非均匀代表值的成本取舍，不是统一预算下的缺陷证据。
2. 高位档：g128q8 比官方 b8 少3.66%字节，局部误差比（uniform/官方）为 {high_ratios}。update100 上它在所测均值上支配官方点，且所测8次重复方向一致、区间不重叠，已超过本轮观察到的重复离散范围；但另两快照误差反转，one-shot 方向一致。本轮没有给出update100特有的可复现特征触发条件、出现范围、算术归因及带成本干预。分布适配和尺度粒度仍是未排除的解释，未声称已证明机制；这个实际点优势未自动升级为问题线索。
3. RNG=1 的零端点是 D0 中记录的官方有限精度边界，原参考亦如此。本轮未隔离其在真实梯度上的实际效应，不能把这个玩具端点或论文与代码表述差异写成已成立的研究线索。

判断为 no_actionable_signal：未形成满足计划§6五项条件的具体候选，也未进入有待解决的机制定位；跨快照点优势只作为描述性筛查记录，未据此主张某个待确认机制。规定的比较覆盖、正确性和字节账本均已有效完成，不存在未修复的比较缺口。该结论不声称已解释或排除所有微观误差机制。

## 6 最接近工作与贡献判断

检查了 [DynamiQ 原论文 arXiv v3](https://arxiv.org/pdf/2602.08923v3)：p5 §3.3 介绍非均匀代表值及层次量化，p6 §3.5 给出系统预算，p11 Table6已有组件消融。论文层次尺度的随机量化论述与本次冻结 CUDA 的确定性 ceil 尺度路径不是同一算术契约；其理论无偏论证未直接套用于本实现。ACM正式版页面未能读取，采用已核实的作者 v3，并明确这一限制。

[官方量化说明](https://github.com/CharlesHan24/DynamiQ/blob/96da2ee480bbfd8e2049de3fbd8595cc9d7a30b8/cuda_kernels/documentation/scaling_quantization_algorithm.md)与[固定 CUDA 源码](https://github.com/CharlesHan24/DynamiQ/blob/96da2ee480bbfd8e2049de3fbd8595cc9d7a30b8/cuda_kernels/src/srrcomp/eden_utils/csrc/cuda_hierarchical_mee.cu)用于核对实际尺度重建与码值路径。公开分支/commit/issue/PR 检查未发现冻结版本之后的公开 codec 修复；不声称覆盖未公开信息。具体位置与访问限制见 [原文定位](logs/nearest_work_source_review.json)。

本次新增的是可达的官方局部组件、真实字节邻点与可复算开发数据证据；它们属于复现和诊断交付，尚不是独立算法贡献。完整系统的动态位宽、前置统计、跨 worker 相关协议、多跳通信与训练质量均未在本轮验证。

## 7 执行成本和交付完整性

D0 主动工作保守上界 {d0_hours:.4f} h（整个阶段墙钟），构建修复2轮。D1 至交付准备结束的工作窗口上界 {d1_hours:.4f} h，从本轮任务开始计入以覆盖与D0重叠的准备，包含两次完整执行及独立复核；不是多个并行代理工时之和。最终执行墙钟 {execution['wall_seconds']:.3f} s，最终 CUDA 操作窗口 {execution['gpu_execution_upper_bound_seconds']:.3f} s；两次 D1 窗口均已计入累计额度。把整个 D0 墙钟也全部计入后的 GPU 保守上界 {execution['cumulative_GPU_upper_bound_seconds']:.3f} s（{execution['cumulative_GPU_upper_bound_seconds']/3600:.4f} h），低于2h。该上界包含 CPU/build 等非GPU时间，不冒充精确kernel执行时间。

独立汇总/复算通过 {analysis['check_count']} 项检查，重新读取12份逻辑输入、780份解码输出并复算195份聚合；{len(checks)}条最终正确性记录均通过。跨修订审计见 [revision_equivalence_audit](logs/revision_equivalence_audit.json)；两次为同种子复现，不累计成16次独立重复。源代码与所加载二进制由 source_manifest 绑定，报告程序等后处理源另列 delivery_sources.json；其加入不改变冻结的实验算术或指标。

交付保留两次完整执行、实际包与解码结果、官方小型反例/边界样例、构建失败与修复日志，以及严格JSON。原始官方源码与许可证已保留；没有复制旧G1/G2a整包、环境或模型。使用完整结果目录交付，未另生成ZIP。最终 [integrity_manifest](integrity_manifest.json)覆盖目录文件（排除自身），外部封存回执记录清单哈希。复算与历史执行入口见 [REPRODUCE](REPRODUCE.md)。

本轮在此结束。D2、新训练、新数据采集、G2b–G4及旧候选均未执行。
'''
    write(run/'REPORT.md',text)
    print(json.dumps(dict(status='written',decision=summary['decision'],raw_records=len(raw),summary_records=len(summary_rows),correctness_records=len(checks)),ensure_ascii=False))
if __name__=='__main__':main()
