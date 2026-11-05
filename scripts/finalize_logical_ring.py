"""Generate reviewable G2a delivery from records, then seal and verify ZIP."""
import argparse,json,hashlib,shutil,zipfile
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'results/gpu_codec_v1/gpu_codec_v1_20261104_001'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rows(p):return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x]
def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def now():return datetime.now(timezone(timedelta(hours=8))).isoformat()
def audit_proof(run,name):
    proof=read(run/name);registration=read(run/'execution_registration.json')
    assert proof['status']=='pass' and proof['run_id']==run.name
    assert proof['contract_sha256']==sha(run/'ring_contract.json')
    assert proof['execution_registration_sha256']==sha(run/'execution_registration.json')
    assert proof['auditor_sha256']==registration['sources']['scripts/audit_logical_ring.py']
    assert proof['audited_files'], 'Unbound audit proof'
    for name,digest in proof['audited_files'].items():assert sha(run/name)==digest,name
    return proof
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',required=True);ap.add_argument('--audit-proof');ap.add_argument('--seal',action='store_true');a=ap.parse_args()
    run=(ROOT/a.run_dir).resolve();assert run.is_relative_to(ROOT/'results/logical_ring_v1')
    if a.seal:
        assert read(run/'summary.json')['g2a_acceptance']=='pass'
        proof=audit_proof(run,a.audit_proof)
        for r in read(OLD/'artifact_integrity.json')['files']:
            assert sha(OLD/r['path'])==r['sha256'],r['path']
        for r in read(run/'source_manifest.json')['files']:
            assert sha(run/r['path'])==r['sha256'];assert sha(r['original_path'])==r['sha256']
        assert not (run/'artifact_integrity.json').exists()
        files=[dict(path=p.relative_to(run).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(run.rglob('*')) if p.is_file()]
        write(run/'artifact_integrity.json',dict(schema_version='logical_ring_v1.0',run_id=run.name,
            created_at=now(),status='sealed_G2a_complete_full_G2_incomplete',prior_g1_seal_unchanged=True,files=files))
        archive=run.with_suffix('.zip');assert not archive.exists()
        with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in sorted(run.rglob('*')):
                if p.is_file():z.write(p,p.relative_to(run).as_posix())
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            for name in z.namelist():assert z.read(name)==(run/name).read_bytes(),name
        final=dict(status='pass',verified_at=now(),sealed_files=len(files),zip_entries=len(files)+1,
            artifact_integrity_sha256=sha(run/'artifact_integrity.json'),zip_path=str(archive),
            zip_sha256=sha(archive),zip_bytes=archive.stat().st_size,prior_g1_seal_unchanged=True)
        write(run.parent/(run.name+'_postseal_verification.json'),final);print(json.dumps(final));return
    passed=False
    if a.audit_proof:
        proof=audit_proof(run,a.audit_proof);passed=True
    correctness=rows(run/'correctness.jsonl');assert correctness and all(r['status']=='pass' for r in correctness)
    rs=read(run/'runner_summary.json');assert rs['status']=='pass'
    manifest=read(run/'input_manifest.json');cases=manifest['cases'];byid={c['case_id']:c for c in cases}
    ledger=rows(run/'message_ledger.jsonl');events=rows(run/'quantization_events.jsonl');inter=rows(run/'intermediate_checks.jsonl');buffers=rows(run/'buffer_lifetime_checks.jsonl')
    real=[]
    for c in cases:
        if c['source_kind']!='real_gradient':continue
        sends=[r for r in ledger if r['case_id']==c['case_id'] and r['method_id']=='cuda_logical_ring' and r['variant']=='baseline']
        check=next(r for r in correctness if r['case_id']==c['case_id'] and r['variant']=='baseline' and r['check_type']=='output_parity')
        real.append(dict(case_id=c['case_id'],logical_numel=c['logical_numel'],padded_numel=c['padded_numel'],chunk_numel=c['chunk_numel'],
            nominal_bits=c['bits'],message_count=len(sends),sent_wire_bytes=sum(r['wire_bytes'] for r in sends),diagnostic=check['diagnostic']))
    baseline_ledger=[r for r in ledger if r['method_id']=='cuda_logical_ring' and r['variant']=='baseline']
    summary=dict(schema_version='logical_ring_v1.0',template_only=False,run_id=run.name,stage='G2a',
        scope='fixed_b4_b8_explicit_thresholds_single_gpu_logical_ring',execution_status='complete',
        correctness_status='pass',message_ledger_status='pass' if passed else 'incomplete',
        buffer_lifetime_status='pass' if passed else 'incomplete',g2a_acceptance='pass' if passed else 'incomplete',full_g2_acceptance='incomplete',
        claims=dict(logical_ring_parity_validated=passed,full_dynamiq_gpu_validated=False,training_quality_newly_validated=False,multigpu_speedup_measured=False),
        next_action='review_required',generated_at=now(),contract_sha256=sha(run/'ring_contract.json'),
        source_manifest_sha256=sha(run/'source_manifest.json'),audit_proof=a.audit_proof,
        coverage=dict(collective_cases=len(cases),real_collective_cases=len(real),real_snapshots=3,
            correctness_records=len(correctness),correctness_status_counts=dict(Counter(r['status'] for r in correctness)),
            message_ledger_records=len(ledger),quantization_event_records=len(events),intermediate_records=len(inter),buffer_lifetime_records=len(buffers),
            frozen_threshold_events=len(read(run/'threshold_manifest.json')['events']),
            gpu_baseline_sends=len(baseline_ledger),gpu_baseline_sent_wire_bytes=sum(r['wire_bytes'] for r in baseline_ledger)),
        real_replays=real,measurement_note='CPU baseline plus GPU baseline/repeat/reverse. Record counts are not independent experiments. No network timing or training run.',
        artifacts=dict(report='REPORT.md',reproduce='REPRODUCE.md',correctness='correctness.jsonl',ledger='message_ledger.jsonl',events='quantization_events.jsonl',intermediate='intermediate_checks.jsonl',buffers='buffer_lifetime_checks.jsonl'))
    if (run/'summary.json').exists():
        assert passed and not (run/'logs/summary_before_audit.json').exists()
        shutil.copy2(run/'summary.json',run/'logs/summary_before_audit.json');shutil.copy2(run/'REPORT.md',run/'logs/REPORT_before_audit.md')
    write(run/'summary.json',summary)
    table='\n'.join(f"| {r['case_id']} | {r['nominal_bits']} | {r['message_count']} | {r['sent_wire_bytes']:,} | {r['diagnostic']['mse']:.9g} |" for r in real)
    status='通过' if passed else '待独立账本/交付审计'
    report=f'''# G2a 固定配置单 GPU 逻辑 ring 交付报告

日期：2026-11-04，北京时间。run_id：`{run.name}`。**G2a：{status}；完整 G2：incomplete。** 按用户“按照原计划完成，不要暂缓版本”的明确指令执行，授权记录见 [execution_authority.json](execution_authority.json)。未修改旧项目或 G1 封存结果。

## 结果

174 个集体用例覆盖 N=1/2/4、b4/b8、raw_fp32/predivided_bf16 两入口、全部指定长度，以及零值、正负值、抵消、有限大值与 subnormal。真实部分为三个快照各两个位宽，共六个 collective。

- 正确性记录：{len(correctness)}，全部 pass；计数包含分项检查，不是独立实验。
- 码流/量化记录：{len(events)}；有向消息记录：{len(ledger)}；中间态记录：{len(inter)}；所有权回归记录：{len(buffers)}。
- 每个输入执行独立 CPU 参照和 GPU baseline/repeat/reverse；CPU 包、编码输入、每次解码、BF16 部分和与各 rank 输出逐位核对。
- N=4 非空输入每次有 16 次编码、24 次发送，sink 再量化计入编码而不增加发送；AllGather 仅转发最终包。
- N=1 与空流零量化、零消息，保留入口 dtype；非空零流仍执行协议。

## 真实梯度回放

D=818048，新的 P=819200，每 chunk L=204800；从已预除 4 的 BF16 逻辑输入读取，没有二次预除。

| 集体输入 | 位宽 | 有向发送数 | 实际发送字节 | 对 BF16 输入 FP64 和的 MSE |
|---|---:|---:|---:|---:|
{table}

误差仅作量化诊断；通过条件是与 CPU 协议逐位一致，不是误差为零。原始 FP32 入口的诊断参照另为原输入 FP64 平均值。旧三个开发快照不支持训练泛化或方法排序。

## 实现与证据

GPU 路由使用逐轮队列屏障；单个 codec plan 可复用，但每次编码克隆三段紧凑消息，decode 输出也克隆。最终包保持至 AllGather 完成。详见 [BUFFER_OWNERSHIP.md](BUFFER_OWNERSHIP.md)。消息计数与字节来自实际载体张量，审计哈希及路由记录不计为网络头。

阈值按 case、位宽、phase、rank、chunk、encoding_ordinal 独立定位，共冻结 1216 个事件；CPU/GPU 使用同一阈值字节。原 LCG 和随机池未推进。b4/b8 主码仍实际存储 5/9 位。

显式加载并核对已验收 a02 模块 SHA256 `c4e37ff9394580bb15fe5d277115f616a439f2741528dff6826bc77e85f06b58`；新入口的实际加载与结果 smoke 已记录。输入、阈值、LUT 保护、逆序 host rank、重复调用、两包在途与解码输出复用均有独立检查。

源码、用例、阈值身份在正式矩阵前登记于 [execution_registration.json](execution_registration.json)。账本由不导入两种 ring 的独立审计程序重新推导。审计状态以 [summary.json](summary.json) 和其 audit_proof 为准。

## 失败记录及限制

开发尝试保留在 attempts/；临时有限极值 probe 曾误把整条补齐流送入单 chunk 阈值，触发明确 ValueError，修正 probe 后通过，未修改 codec 或数据。全部实际问题见 [issues.jsonl](issues.jsonl)，首次失败反例及 traceback 保留在相应尝试目录。

没有重跑 G1 性能矩阵，没有新增训练、动态 b2/4/8 控制、随机状态闭环或网络通信。没有测量网络 AllReduce 延迟或多卡加速；本阶段不等于完整 DynamiQ GPU 移植。G2b 待交付复核后确定，当前 `full_g2_acceptance=incomplete`。

复现入口、环境依赖与封存核验见 [REPRODUCE.md](REPRODUCE.md)。
'''
    (run/'REPORT.md').write_text(report,encoding='utf-8')
    reproduction='''# Reproduction

The executed environment is Windows, Python 3.10.3 / PyTorch 2.4.1+cu118 and the accepted CUDA 11.8 a02 binary. See environment.json and logs/cuda_build_manifest.json. This is not a portable wheel or a network backend.

Run from E:/dacomp/research/gradient_comm_lab. Preserve the sealed directory; use a fresh RUN path under results/logical_ring_v1. All commands below use scripts/python.ps1 (project-local caches).

```powershell
./scripts/python.ps1 scripts/prepare_logical_ring.py --run-dir <RUN>
./scripts/python.ps1 scripts/check_logical_ring.py --run-dir <RUN> --attempt preparation --manifest-only
./scripts/python.ps1 scripts/register_logical_ring.py --run-dir <RUN>
./scripts/python.ps1 scripts/check_logical_ring.py --run-dir <RUN> --attempt formal_a01 --formal
./scripts/python.ps1 scripts/finalize_logical_ring.py --run-dir <RUN>
./scripts/python.ps1 scripts/audit_logical_ring.py --run-dir <RUN> --output logs/audit_initial.json
./scripts/python.ps1 scripts/finalize_logical_ring.py --run-dir <RUN> --audit-proof logs/audit_initial.json
./scripts/python.ps1 scripts/audit_logical_ring.py --run-dir <RUN> --output logs/audit_final.json
./scripts/python.ps1 scripts/finalize_logical_ring.py --run-dir <RUN> --audit-proof logs/audit_final.json --seal
```

Before the audit, include the unchanged BUFFER_OWNERSHIP.md and issues.jsonl describing actual execution issues; include the report generator's source snapshot. Do not fabricate pass results for a new attempt. The original actual command line is recorded in execution_start.json; development attempts have their own immutable execution_start.json and source hashes.

The input tensors are bundled in fixtures/. The source archives and their tensor identities bind real inputs; for full regeneration the accepted G1 and original CPU project at E:/dacomp/research/spl_model_communication must remain available. Copies of reference Python modules and accepted binary are provided under code/reference and code/build. The unchanged CPU adapter retains its original absolute dependency path. On relocation, explicitly rebind this path and the build manifest in a NEW run, then requalify; never edit a sealed manifest. Thresholds regenerate from the frozen event keys and exact PyTorch/BF16 operation order. Serialization archive hashes can differ in a new path even when tensor bytes match.

The archive includes all development evidence and fixtures. artifact_integrity.json binds every pre-seal file; adjacent postseal_verification.json binds the integrity file and ZIP, whose members were checked byte-for-byte. No compute-sanitizer, NCCL, physical multi-GPU or new training validation is claimed.
'''
    (run/'REPRODUCE.md').write_text(reproduction,encoding='utf-8')
    print(json.dumps(dict(status='final' if passed else 'draft',coverage=summary['coverage'])))
if __name__=='__main__':main()
