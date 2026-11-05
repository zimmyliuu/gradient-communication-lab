"""Post-experiment report only: no codec calls, no changes to frozen analysis."""
import argparse,json,hashlib,shutil
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def dump(p,v,mode='x'):
    with p.open(mode,encoding='utf-8') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def fmt(v):return 'null' if v is None else f'{v:.8g}'
def stat(r,f):
    s=r['statistics'][f];return fmt(s['mean'])+' ± '+fmt(s['sample_std'])
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',required=True);a=ap.parse_args();run=(ROOT/a.run_dir).resolve()
    p=read(run/'paired_summary.json');c=read(run/'experiment_contract.json');pre=read(run/'scale_precheck.json');rg=read(run/'region_manifest.json')
    assert p['status']=='pass' and read(run/'logs/independent_effect_audit.json')['status']=='pass'
    assert read(run/'logs/recompute_verification.json')['status']=='pass'
    parity=rows(run/'control_parity.jsonl');checks=rows(run/'correctness.jsonl');changes=rows(run/'packet_changes.jsonl')
    assert len(parity)==96 and all(r['status']=='pass' for r in parity+checks+changes)
    groups=p['groups'];lookup={(r['record_type'],r['snapshot_id'],r['worker_id'],r['region_id']):r for r in groups}
    snapshots=sorted({r['snapshot_id'] for r in groups});first=snapshots[0]
    complete=[read(f) for f in (run/'logs').glob('*_completion.json')];assert len(complete)==8 and all(x['status']=='pass' for x in complete)
    now=datetime.now(timezone(timedelta(hours=8)));start=datetime.fromisoformat(read(run/'logs/task_start.json')['started_at'])
    last_gpu=datetime.fromtimestamp(max(f.stat().st_mtime for f in (run/'logs').glob('*_completion.json')),tz=timezone(timedelta(hours=8)))
    frozen=datetime.fromisoformat(c['frozen_at']);gpu_upper=c['build_wall_seconds_upper_bound']+(last_gpu-frozen).total_seconds()
    resources=dict(active_work_hours=(now-start).total_seconds()/3600,gpu_execution_minutes=gpu_upper/60,condition_builds=2,repair_rounds=1,
        measurement='Elapsed upper bounds, not sum of parallel labor or kernel performance. GPU bound charges both complete build wall windows plus entire freeze-to-last-execution window.',
        synchronized_GPU_windows_seconds=sum(x['gpu_window_seconds'] for x in complete),build_wall_seconds=c['build_wall_seconds_upper_bound'],
        precheck_elapsed_upper_bound_minutes=(datetime.fromisoformat(pre['created_at'])-start).total_seconds()/60)
    s=read(run/'summary.json');s.update(execution_status='complete',stages=dict(E0='pass',E1='pass',E2='pass'),contract_sha256=sha(run/'experiment_contract.json'),
        outcome=p['outcome'],outcome_reason='去掉固定加数部分降低误差；update100整体闭合55.17%，固定H仅34.11%，未满足H至少50%的投入门槛。按预设规则结束当前假说，不追加条件。',
        update100=p['update100'],continuation_gate=p['continuation_gate'],resource_usage=resources,completed_at=now.isoformat())
    s['identity'].update(control_binary_sha256=c['encoder_binary_sha256']['control_add1e7'],diagnostic_binary_sha256=c['encoder_binary_sha256']['diagnostic_add0'],
        decoder_binary_sha256=c['decoder_binary_sha256'],control_parity_status='pass',matched_pool_status='pass',single_factor_source_status='pass')
    s['coverage'].update(completed_control_pairs=96,completed_diagnostic_pairs=96,valid_diagnostic_pairs=96,failed_pairs=0,missing_pairs=0)
    s['other_snapshots_relative_sse_change']={f'update{int(k.rsplit("update",1)[1])}':v for k,v in p['other_snapshots_relative_sse_change'].items()}
    s['continuation_checks']={k:v['passed'] for k,v in p['continuation_checks'].items()};s['continuation_measurements']=p['continuation_checks']
    s['claims']['counterfactual_executed']=True;s.update(not_produced=['performance measurements (not_run)','D2 and new training (not_run)','causal evidence card for continuation (gate not_met; not required)'],
        evidence_paths=['REPORT.md','paired_summary.json','metrics_raw.jsonl','packet_changes.jsonl','control_parity.jsonl','correctness.jsonl','experiment_contract.json','source_manifest.json','environment.json','input_randomness_manifest.json','scale_precheck.json','region_manifest.json','logs/independent_effect_audit.json','logs/recompute_verification.json','REPRODUCE.md','integrity_manifest.json'],
        next_action='stop_current_hypothesis; review required for any new scope',D2='not_run')
    dump(run/'summary.json',s,'w')
    issue_rows=[dict(issue_id='build_preflight_guard',status='resolved',kind='implementation_entrypoint',evidence='logs/build_preflight_guard_repair.json',
        details='Build count guard initially included environment records; corrected before second compile. Two condition compilations, no arithmetic change; conservatively count one repair round.'),
        dict(issue_id='C8_estimated',status='documented',kind='uncertainty',evidence='scale_precheck.json',details='574.0/0x607c is explicit FP32 simulation then FP16, not exported actual host constant. Error bound unknown.'),
        dict(issue_id='wrapper_inherited_docstring',status='documented',kind='identity',evidence='source_manifest.json',details='Thin wrapper code copied unchanged from base; its inherited description does not make the loaded diagnostic encoder unmodified official code. Derived identities are in manifests and contract.'),
        dict(issue_id='fixed_gate_not_met',status='closed_current_hypothesis',kind='research_stop',evidence='paired_summary.json',details='H gap closure <0.5; no further constants, regions, data or D2 pursued.')]
    assert not (run/'issues.jsonl').exists()
    with (run/'issues.jsonl').open('x',encoding='utf-8') as f:
        for r in issue_rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    tables=['| 快照/区域 | A SSE | B SSE | U SSE | 配对Δ mean ± std | Δ min/max | relative reduction均值 | gap_before → gap_after | 改善次数 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for snap in snapshots:
        for region in ['full','H','rest']:
            r=lookup['snapshot_local',snap,None,region];t=r['statistics']
            tables.append(f"| {snap}/{region} | {stat(r,'control_sse')} | {stat(r,'diagnostic_sse')} | {stat(r,'uniform_sse')} | {stat(r,'paired_delta')} | {fmt(t['paired_delta']['min'])}/{fmt(t['paired_delta']['max'])} | {fmt(t['relative_reduction']['mean'])} | {fmt(t['gap_before']['mean'])} → {fmt(t['gap_after']['mean'])} | {r['paired_repeats_improved']}/8 |")
    workers=['| 快照/worker | Δ mean ± std | Δ min/max | 改善重复数 |','|---|---:|---:|---:|']
    for snap in snapshots:
        for w in range(4):
            r=lookup['local',snap,w,'full'];st=r['statistics']['paired_delta']
            workers.append(f"| {snap}/{w} | {stat(r,'paired_delta')} | {fmt(st['min'])}/{fmt(st['max'])} | {r['paired_repeats_improved']}/8 |")
    ag=['| 快照 | source_energy | target_energy | A one-shot SSE | B one-shot SSE | U one-shot SSE | 配对Δ mean ± std |','|---|---:|---:|---:|---:|---:|---:|']
    raw=rows(run/'metrics_raw.jsonl')
    for snap in snapshots:
        r=lookup['one_shot',snap,None,'full'];row=next(x for x in raw if x['record_type']=='one_shot' and x['snapshot_id']==snap and x['region_id']=='full')
        ag.append(f"| {snap} | {fmt(row['source_energy'])} | {fmt(row['target_energy'])} | {stat(r,'control_sse')} | {stat(r,'diagnostic_sse')} | {stat(r,'uniform_sse')} | {stat(r,'paired_delta')} |")
    ct=['| 快照 | 包字节变化 | 主码变化 | UINT8尺度变化 | BF16超尺度变化 | 逻辑输出变化 | 边界码 A/B |','|---|---:|---:|---:|---:|---:|---:|']
    for snap in snapshots:
        rr=[r for r in changes if r['snapshot_id']==snap]
        vals=[sum(x[f] for x in rr)/(len(rr)*den) for f,den in [('changed_packet_bytes',875704),('changed_main_codes',818176),('changed_group_scale_codes',51136),('changed_super_scale_units',3196),('changed_decoded_elements_logical',818048)]]
        ba=sum(x['boundary_code_fraction_A'] for x in rr)/32;bb=sum(x['boundary_code_fraction_B'] for x in rr)/32
        ct.append('| '+snap+' | '+' | '.join(f'{v:.6%}' for v in vals)+f' | {ba:.6%}/{bb:.6%} |')
    gate=['| 冻结门槛 | 实测 | 通过 |','|---|---|---|']
    for key,value in p['continuation_checks'].items():gate.append(f"| {key} | {json.dumps(value,ensure_ascii=False)} | {'是' if value['passed'] else '否'} |")
    h=lookup['snapshot_local',first,None,'H'];f=lookup['snapshot_local',first,None,'full'];hshare=h['statistics']['paired_delta']['mean']/f['statistics']['paired_delta']['mean']
    totalfrac=pre['all_nonzero_group_add_fraction']
    report=f'''# 固定尺度加数单因素诊断报告

已完成 E0、E1、E2；run_id：{run.name}；执行日期：2026-11-05。

## 1 结论

**outcome = {p['outcome']}；continuation_gate = {p['continuation_gate']}。** 原固定加数对这批开发输入的误差有实际作用，去掉它部分闭合update100相对旧uniform的误差劣势：整体闭合 **{p['update100']['gap_closure']:.4%}**，预先冻结高能量区域H闭合 **{p['update100']['gap_closure_H']:.4%}**。后者未达50%门槛，因此按原计划结束本假说，不改门槛、不扫描加数、不重选区域。

update100四个worker均值均改善，8次快照配对均改善；update500/2500均值SSE分别变化 **{p['other_snapshots_relative_sse_change'][snapshots[1]]:.4%} / {p['other_snapshots_relative_sse_change'][snapshots[2]]:.4%}**（负号表示改善），无均值回退。全部96对有效，无非法包、尺度或非有限输出。假说得到“部分解释差距”的支持，未证明解释了高贡献区域的主要差距；未验证部署收益、泛化或独立贡献，**D2未执行**。

## 2 来源与唯一干预

基线为封存 bounded_discovery_v1_20261104_001，commit `{c['upstream_commit']}`；旧合同 `{c['base_contract_sha256']}`。本轮合同 `{sha(run/'experiment_contract.json')}`，在资格/正式调用之前冻结。

实际路径为 `scaling_compress_kernel_mee_hierarchical_bf16_opt<8>`，BF16、strategy3、chunk16；只将原raw_scale语句的加数用编译宏条件化。两个源码树相同，仅 `-DDIAG_B8_SCALE_ADD=1e-7f/0.0f` 不同；其他位宽仍1e-7f。MEE epses[8]=0.1、host FP32构表/FP16 LUT、positive_scale的1e-30下限、非有限值保护、BF16 super-scale、UINT8 ceil/clamp、主码范围和随机舍入均未改。见 [路径资格](logs/source_path_qualification.json)、[构建审计](logs/build_pair_audit.json)、build目录中的逐项diff。

| 身份 | SHA256 |
|---|---|
| 派生控制encoder | `{c['encoder_binary_sha256']['control_add1e7']}` |
| 派生零加数encoder | `{c['encoder_binary_sha256']['diagnostic_add0']}` |
| 封存原decoder | `{c['decoder_binary_sha256']}` |

两个encoder与原decoder均通过显式manifest在独立进程加载，8个阶段的PID/模块/manifest哈希已保存。decoder进程不加载条件encoder。两构建不能冒称未修改官方算法；commit仅表示来源。未执行旧build_official.py、未覆盖旧模块或共享环境。

原96个seed/pool按CPU FP32 rand→BF16复现，全P字节哈希与旧记录一致，保留合法的1。A/B使用matched randomness，uniform仍是原独立随机流。控制96包与封存包逐字节相同，原decoder解出的BF16逐位相同；这96次就是控制组，没有重复生成额外控制组。资格还覆盖零、常数、正负混合、稀疏、257尾部、真实输入及一个未改b4非零夹具，两条件全部通过。详见 [control_parity](control_parity.jsonl)、[correctness](correctness.jsonl)。

## 3 量级预查

C8 **estimated={pre['C8']['value']}，FP16位模式{pre['C8']['fp16_bits_hex']}**。来自封存host构表源码的显式FP32递推并转FP16，未直接导出实际host常数，误差边界未知；Python double功能测试未作为严格oracle。

12输入共{pre['logical_group_count']}个逻辑16元素组，零组{pre['zero_group_count']}，另{pre['total_padding_only_groups']}个padding-only组单列。add_fraction普通中位数{totalfrac['quantiles']['0.5']:.6%}、均值{totalfrac['mean']:.6%}；输入能量加权中位数{totalfrac['energy_weighted_quantiles']['0.5']:.6%}、均值{totalfrac['energy_weighted_mean']:.6%}。完整分位数、每输入分布、max_abs/energy、逐组NPZ及原包实际UINT8/BF16尺度见 [scale_precheck](scale_precheck.json)。实际存储尺度经过量化，不能当raw_scale。

各快照共同256坐标块按四worker输入能量之和排序，取32块、tie按block_id，结果在 [region_manifest](region_manifest.json) 冻结。旧审计只存数量/占比而未列ID，本轮按同规则重建并核对能量占比。所有误差裁掉尾部；估计占比不能证明CUDA fast math结果不变，E0未按量级早停。

## 4 配对结果

A=重建原加数控制，B=零加数诊断，U=旧uniform_g128_q8复用。以下均为四worker局部SSE总和在8次配对中的均值±sample std。Δ=A−B，正号改善；先算每次差再汇总，std不是两个独立std相加。A/B同为875704 B/worker，U为843628 B，所有结果以BF16输出、FP64逻辑坐标误差比较，D=818048，P=818176，不再次除4。

{chr(10).join(tables)}

gap_closure严格按mean(A−B)/mean(A−U)，不取每次比例均值、不裁剪。仅update100正分母作闭合解释；其他快照报告gap_before/after，不使用负分母解释比例。relative_reduction列为逐次(A−B)/A的均值，summary其他快照变化则严格为(meanB−meanA)/meanA。

### 四worker方向与重复离散

{chr(10).join(workers)}

各worker、各区域、各快照全部54组mean/std/min/max及逐重复值见 [paired_summary](paired_summary.json)，源能量与归一化误差均保留，不筛选坏结果或最好seed。raw联合键为 record_type + pair_id + condition_id + region_id；one_shot的pair_id为snapshot__rNN，worker_id=null。

### 一次编码求和

{chr(10).join(ag)}

复用同四份输出做FP64求和，target为原输入FP64之和；cross_term=aggregateSSE−ΣlocalSSE保存在raw。one-shot仅每worker一次编码，不是多跳AllReduce；包和为A/B 3502816 B、U 3374512 B，不套ring系数，不报告网络速度。

## 5 变化位置和有效域

{chr(10).join(ct)}

包/主码/尺度的分母含P尾部；输出变化仅D逻辑坐标。boundary_code_fraction只是码0/254的比例，不是实际clamp或饱和次数，后者未测为null。每对完整变化位置保存为changes/*.npz，入口及哈希在 [packet_changes](packet_changes.jsonl)。所有reserved code、非法尺度与非有限输出计数为0；原始新包及解码张量完整保留。

update100的H贡献了总配对改善的{hshare:.4%}，H与rest分别改善{fmt(h['statistics']['paired_delta']['mean'])}、{fmt(lookup['snapshot_local',first,None,'rest']['statistics']['paired_delta']['mean'])} SSE。H原净差距约占94.4%，但只闭合34.11%，故整体超过50%不能替代H门槛。源码单因素对照支持这项加数对最终误差的作用；尺度、主码、输出同时变化仍不能据此分解每个中间环节的单独因果份额，E0估计关联也不是额外机制证明。

## 6 继续或停止

{chr(10).join(gate)}

**唯一未通过项是H gap_closure≥0.5；continuation_gate=not_met。** outcome描述部分实测效应，gate描述后续资源投入，二者不混为“无效”或“成功改进”。50%/2%均为事前工程门槛，不是统计显著性、普遍正确性或科研新颖性标准。8次重复只反映同一开发轨迹的随机状态；本轮没有独立输入、训练质量、性能或多卡证据。B与U预算不同，误差劣势减少不能称字节与误差同时支配U。

两条件构建完成，实际CUDA编译修复0次；一次编译前计数保护误计环境日志的入口修复已保留diff与失败日志，保守计repair_rounds=1。E0保守窗口{resources['precheck_elapsed_upper_bound_minutes']:.3f}分钟，总主动工作按经过窗口计上界{resources['active_work_hours']:.4f}小时；不是并行代理工时之和。两build全墙钟加冻结到最后GPU阶段的窗口作为GPU保守上界{resources['gpu_execution_minutes']:.3f}分钟，低于30分钟；实际同步GPU操作窗口合计{resources['synchronized_GPU_windows_seconds']:.3f}秒，不作为性能测量。

数值由保存包/输出重算，且通过独立实现复核 [independent_effect_audit](logs/independent_effect_audit.json)；只读复算入口及实际执行顺序见 [REPRODUCE](REPRODUCE.md)。全目录及旧外部依赖由 [integrity_manifest](integrity_manifest.json) 记录，新数据未只留汇总，旧3.57GB交付未复制。没有新增图表。到报告交付结束，停止本假说；D2、新训练、新方法及额外参数扫描均未执行。
'''
    with (run/'REPORT.md').open('x',encoding='utf-8') as f:f.write(report)
    dest=run/'code/delivery'/Path(__file__).name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(__file__,dest)
    dump(run/'delivery_source.json',dict(role='Post-experiment reporting, no frozen numerical source changes',path=str(dest),sha256=sha(dest),bytes=dest.stat().st_size))
    print(json.dumps(dict(status='complete',outcome=p['outcome'],continuation_gate=p['continuation_gate'],resource_usage=resources),ensure_ascii=False))
if __name__=='__main__':main()
