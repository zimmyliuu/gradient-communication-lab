"""Produce an evidence-bound G0 blocker delivery, without claiming G1 work."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone,timedelta
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(2**20),b''): h.update(part)
    return h.hexdigest()


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def write(path,value):
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False)


def stamp(): return datetime.now(timezone(timedelta(hours=8))).isoformat()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True)
    args=parser.parse_args()
    run=(ROOT/args.run_dir).resolve()
    if not run.is_relative_to(ROOT/'results/gpu_codec_v1'): raise ValueError('Invalid run directory')
    if (run/'summary.json').exists(): raise FileExistsError('Prior deliveries are immutable; use a new run')
    env=read(run/'environment.json'); probe=read(run/'logs/minimal_cuda_probe.json')
    refs=read(run/'cpu_reference_manifest.json'); inputs=read(run/'input_manifest.json')
    fixtures=read(run/'fixtures_manifest.json'); semantics=read(run/'logs/cpu_reference_semantics.json')
    assert probe['status']=='blocked' and probe['module_called'] is False
    assert env['compiler']['nvcc_path'] is None
    assert len(inputs['inputs'])==12 and not inputs['holdout_opened']
    # Keep the documentation and templates as supplied, and freeze their identities.
    supplied=[ROOT/'TECHNICAL_PLAN.md',ROOT/'DELIVERY_FORMAT.md',*sorted((ROOT/'handoff_templates').glob('*'))]
    write(run/'specification_manifest.json',{'run_id':run.name,'files':[
        {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in supplied if p.is_file()]})
    contract=read(ROOT/'handoff_templates/codec_contract.json')
    contract.update(template_only=False,run_id=run.name,
        contract_id='dynamiq_local_repaired_explicit_thresholds_G0_reference_v1',
        scope='局部 codec 固定配置 / local_codec_only',
        freeze_status='CPU_reference_and_format_frozen_GPU_implementation_binding_pending',
        cpu_reference_files=refs['reference_files'],cpu_adapter=refs['adapter'],lookup_tables=refs['lookup_tables'],
        gpu_implementations=[],compiler_flags=[],
        checked_api_implementation=None,
        trusted_input_core_preconditions=['contiguous one-dimensional BF16','positive native L divisible by 256',
            'CPU reference successfully encodes and decodes this actual input with the identical explicit thresholds',
            'threshold dtype/shape/range verified; LUT copied bitwise from CPU',
            'finite input and positive finite top alone do not imply finite BF16 intermediates'],
        bf16_rounding_semantics={
            'R':'each individual BF16 operation rounds before the following operation; no fused arithmetic substitution',
            'epsilon_scalar':'BF16(1e-23) = 9.977866138920896e-24',
            'random_multiplier':'BF16(65535) = 65536',
            'scale':'R(R(maxabs16 / LUT[3*2^(b-1)]) + BF16(1e-23))',
            'normalization':'R(x / scale)',
            'probability':'R(R(work-left) / R(right-left))',
            'stochastic_comparison':'threshold < R(probability * BF16(65535)); strict <; upper_bound/right=True',
            'top':'R(max16(scales) + BF16(1e-23))',
            'scale_levels':'R(R(scales/top) * BF16(255))',
            'scale_probability':'R(levels - floor(levels)); then threshold < R(probability * BF16(65535))',
            'decoded_scale':'R(R(BF16(qscale) * top) / BF16(255))',
            'index':'int32(signed_code) + 2^b; do not add offset in BF16',
            'decoded_value':'R(LUT[index] * decoded_scale); preserve signed zero',
            'finite_guard':'reject nonfinite normalized input, probability, received scales and output as the CPU reference does',
            'CPU_probe':'logs/cpu_reference_semantics.json','CPU_probe_sha256':sha(run/'logs/cpu_reference_semantics.json')},
        planned_GPU_compiler_policy={'fast_math':False,'fmad':False,'ftz':False,'precise_division':True,
            'actual_codec_compiler_flags':None,'note':'G0 probe flags are recorded separately; no codec has been compiled'},
        future_contract_policy='G1 must create a new full contract binding actual GPU sources and flags; do not modify this closed attempt')
    write(run/'codec_contract.json',contract)
    bench=read(ROOT/'handoff_templates/benchmark_config.json')
    bench.update(template_only=False,run_id=run.name,execution_status='not_run',
        freeze_status='planned_matrix_only_no_K_pilot_or_sessions_executed',
        initialization_costs_measured=False)
    write(run/'benchmark_config.json',bench)
    command=[sys.executable,'-m','pip','freeze','--all']
    result=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',errors='replace')
    if result.returncode: raise RuntimeError(result.stderr)
    (run/'requirements.txt').write_text(result.stdout,encoding='utf-8')
    write(run/'logs/dependency_capture.json',{'created_at':stamp(),'command':command,
        'returncode':result.returncode,'stderr':result.stderr,'role':'read_only_snapshot_of_shared_interpreter_not_an_install',
        'requirements_sha256':sha(run/'requirements.txt')})
    guide='https://docs.nvidia.com/cuda/archive/11.8.0/cuda-installation-guide-microsoft-windows/index.html'
    write(run/'logs/toolchain_reference_sources.json',{'checked_at':stamp(),'sources':[{
        'url':guide,'section':'1.1 System Requirements, Table 2 Windows Compiler Support in CUDA 11.8',
        'fact':'CUDA 11.8 Windows table lists MSVC 193x, 192x, 191x; observed MSVC 19.51 is not in this table',
        'interpretation':'Compatibility remains unverified; current actual failure occurred before compiler invocation'}]})
    issues=[
        {'issue_id':'G0-ENV-001','category':'toolchain','severity':'blocking','status':'open',
         'description':'CUDA Toolkit/nvcc was not found in checked accessible installation roots; CUDA_HOME is unset; actual extension load failed before compilation.',
         'evidence_paths':['environment.json','logs/environment_discovery.json','logs/minimal_cuda_probe.json','logs/minimal_cuda_build.log'],
         'reproduction':'scripts/python.ps1 scripts/probe_environment.py --run-dir <fresh-run>; then scripts/python.ps1 scripts/minimal_cuda_probe.py --run-dir <fresh-run>',
         'fix_or_next_step':'Locate or provision a complete compatible Toolkit; initialize a supported host compiler in a project-scoped process; rerun the real build and GPU call in a fresh run.'},
        {'issue_id':'G0-ENV-002','category':'compiler_compatibility','severity':'medium','status':'unverified',
         'description':'Detected MSVC 19.51.36260 is outside the CUDA 11.8 guide compiler table. No Toolkit compile has yet tested this combination.',
         'evidence_paths':['environment.json','logs/toolchain_reference_sources.json'],
         'reproduction':'Compare recorded compiler version with the frozen-runtime Toolkit requirements; then perform an actual build.',
         'fix_or_next_step':'Select a supported host toolset alongside existing tools; do not infer compatibility from cl.exe existence or suppress compiler guards to claim validation.'},
        {'issue_id':'G0-ENV-003','category':'environment_probe_warning','severity':'low','status':'open',
         'description':'PyTorch compiler-version probing emitted a localized cl output decoding warning; raw warning retained. This is not the primary CUDA_HOME failure.',
         'evidence_paths':['logs/minimal_cuda_build.log','environment.json'],
         'reproduction':'Inspect the retained minimal build log.',
         'fix_or_next_step':'Resolve process-local compiler output decoding when validating the complete toolchain; retain the original attempt.'},
        {'issue_id':'G1-NOT-RUN','category':'coverage','severity':'blocking','status':'not_started',
         'description':'G1 GPU implementations, parity matrix, performance grid, memory measurements and profiler trace were not executed because G0 compilation did not pass.',
         'evidence_paths':['summary.json','logs/minimal_cuda_probe.json'],
         'reproduction':None,'fix_or_next_step':'After G0 passes, implement b4 then b8 and execute the prescribed G1 matrix without changing the CPU semantics.'}]
    with (run/'issues.jsonl').open('x',encoding='utf-8') as stream:
        for issue in issues: stream.write(json.dumps(issue,ensure_ascii=False,allow_nan=False)+'\n')
    missing={name:'Not generated: G0 CUDA extension compilation blocked; G1 not started.' for name in
             ('correctness.jsonl','benchmark_raw.jsonl','benchmark_summary.jsonl','memory.jsonl','traces/','figures/')}
    summary=read(ROOT/'handoff_templates/summary.json')
    summary.update(template_only=False,run_id=run.name,stage='G0',
        stage_name='局部 codec 固定配置：环境、CPU 参考与输入契约准备',
        started_at=datetime.fromtimestamp(run.stat().st_ctime,timezone(timedelta(hours=8))).isoformat(),
        finished_at=stamp(),execution_status='blocked',contract_sha256=sha(run/'codec_contract.json'),
        benchmark_config_sha256=sha(run/'benchmark_config.json'),correctness_status='not_run',
        benchmark_validity='not_run',optimization_gain='not_evaluated',g1_acceptance='not_ready',
        next_action='fix_environment',g0_gate='blocked_at_minimal_CUDA_extension_build',
        cpu_semantic_probe_status='passed_CPU_only_not_GPU_parity',prepared_real_gradient_inputs=12,
        missing_artifacts=missing,key_findings=[
            'GPU CUDA runtime and BF16 are usable; MSVC, Ninja and CMake exist.',
            'CUDA Toolkit/nvcc not discovered in checked accessible locations; actual minimal extension build blocked before compilation.',
            '12 authorized development gradient inputs, CPU LUTs and explicit thresholds are frozen; no holdout read.',
            'BF16 operation ordering, scalar rounding, internal nonfinite rejection and signed zero must be preserved; G1 remains untested.'],
        open_issues=[i['issue_id'] for i in issues])
    summary['coverage']['missing_items']=['G0 minimal CUDA extension compile and GPU call',
        'PyTorch CUDA and optimized CUDA/C++ local codec b4/b8','edge and synthetic fixtures',
        'complete CPU/GPU bytewise and BF16 parity coverage','12 real-gradient GPU roundtrip cases per method/bit',
        'five-session performance matrix with frozen K and raw blocks','independent memory measurements and representative profiler trace','three figure groups']
    for key in ('correctness','benchmark_raw','benchmark_summary','memory'): summary['artifacts'][key]=None
    summary['artifacts'].update(fixtures='fixtures_manifest.json',cpu_reference='cpu_reference_manifest.json',
        cpu_semantics='logs/cpu_reference_semantics.json',requirements='requirements.txt')
    write(run/'summary.json',summary)
    reproduce=f'''# G0 实际复现记录

工作目录：`{ROOT.as_posix()}`。解释器：`{env['python_executable']}`。本轮只读复用已有环境，未安装或升级包；缓存与临时文件由 `scripts/python.ps1` 隔离到本项目。

以下命令已实际执行。`R` 代表 `{args.run_dir}`；历史尝试受保护，重新运行须建立新的唯一 run 目录，不能覆盖本目录。

```powershell
.\\scripts\\python.ps1 scripts/probe_environment.py --run-dir {args.run_dir}
.\\scripts\\python.ps1 scripts/minimal_cuda_probe.py --run-dir {args.run_dir}
.\\scripts\\python.ps1 scripts/freeze_g0_inputs.py --run-dir {args.run_dir}
.\\scripts\\python.ps1 scripts/probe_reference_semantics.py --run-dir {args.run_dir}
.\\scripts\\python.ps1 scripts/finalize_g0_delivery.py --run-dir {args.run_dir}
```

环境探针成功写入 `environment.json`；最小扩展探针退出 2，原始错误见 `logs/minimal_cuda_build.log`。没有 nvcc 编译命令被真正执行，也没有生成可调用扩展。实际尝试的 build API、源码、参数和 traceback 在 `logs/minimal_cuda_probe.json`；仅为构建子进程加入现有 cl/Ninja 路径，没有更改全局环境。

夹具脚本调用原 `splcomm.p2_replay.load_inputs(calibration_root, 'pilot')`，仅加载 seed1101 的更新前100/500/2500及每个输入4个worker。先转BF16、再BF16除以4，最后补零。固定生成器和main→scale顺序见 `input_manifest.json`。大文件留在 `fixtures_manifest.json` 列出的本机路径；重建要求所列历史checkpoint及相同参考来源可取得，文件级和tensor级SHA分别记录。

CPU语义探针已执行并通过，但不是GPU正确性矩阵。完整G1正确性入口、性能入口、显存/trace和汇总作图尚未实现或执行，没有可声称已复现的相关命令。不要直接运行旧项目 `tests/test_dynamiq.py`，其fixture可能写回旧P1证据。

依赖快照通过实际命令 `{sys.executable} -m pip freeze --all` 获取，见 `requirements.txt` 和 `logs/dependency_capture.json`。它是旧解释器当前包清单，不代表在本项目已创建新venv或完成可移植安装。若未来需安装/升级包，按技术方案创建本项目独立venv。

后续环境修复尚未执行：补齐完整Toolkit并选择其支持的宿主工具链，在新run中先通过最小构建与真实GPU调用，再实施G1。没有自动进入G2。
'''
    (run/'REPRODUCE.md').write_text(reproduce,encoding='utf-8')
    report=f'''# GPU 低比特编解码实施报告

run_id：`{run.name}`。执行日期：2026-11-01，北京时间。技术方案版本：1.0。对象：**局部 codec 固定配置**。

## 1 结论与完成范围

本轮完成环境核验、CPU参考适配与语义探针、数值/格式契约冻结，以及12个指定真实开发梯度夹具准备。**G0最小CUDA扩展构建受阻，G1未启动，不能标为通过。** 已找到MSVC/Ninja/CMake，但在已检查且可访问的安装路径中未发现CUDA Toolkit/nvcc；PyTorch检测`CUDA_HOME=None`，实际构建调用在准备链接参数时失败。未产生GPU codec正确性或性能结果。

| 项目 | 实际状态 | 证据路径 |
|---|---|---|
| 环境核验 | 已执行：RTX4000 Ada / SM8.9 / driver596.72 / Torch2.4.1+cu118 / BF16可用 | [environment.json](environment.json) |
| 最小CUDA扩展编译与调用 | blocked；未完成编译，未调用GPU扩展 | [探针](logs/minimal_cuda_probe.json)、[日志](logs/minimal_cuda_build.log) |
| CPU参考与固定契约 | 已冻结；CPU-only语义探针通过，GPU绑定尚缺 | [契约](codec_contract.json)、[CPU探针](logs/cpu_reference_semantics.json) |
| b=4 / b=8 GPU固定配置 | 未实现、未验证 | [summary.json](summary.json) |
| 真实梯度 | 12个CPU夹具已准备；GPU回放未执行 | [输入清单](input_manifest.json)、[夹具清单](fixtures_manifest.json) |
| GPU性能、显存和trace | 未执行，指标为null，不生成空测量文件 | [summary.json](summary.json) |
| 完整G1验收 | not_ready | [summary.json](summary.json) |

## 2 实现与契约

直接复用已冻结`dynamiq.py`及`packing.py`，上游commit为`96da2ee480bbfd8e2049de3fbd8595cc9d7a30b8`；新增适配器只调用CPU参照，未重写近似公式。源码身份见 [CPU来源](cpu_reference_manifest.json) 与 [交付源码清单](source_manifest.json)。配置为`index_repaired_bf16`、名义b4/b8、真实5/9位码、16元素尺度组和256元素顶层组。包序为payload、全部uint8尺度码、全部小端BF16顶层尺度；不新增头部。

CPU探针确认BF16(65535)=65536，BF16(1e-23)=9.977866138920896e-24。将epsilon加法改为FP32字面常数后才cast，在全部有限BF16位模式中会得到1527个不同结果。编码/解码必须逐操作保留舍入，使用right=True与严格小于；b8先做整数索引偏移。合法qscale=0不允许统一填正零，负LUT值乘零的符号须保留。

带检查GPU API尚未实现；契约已明确要求形状、dtype、连续性、阈值、码值/包长、top正有限及中间非有限检测。有限输入或正有限top并不足以保证全BF16计算可表示，CPU探针保留内部溢出被拒绝的案例。可信输入性能网格须先用实际参照roundtrip校验，不能用其延迟冒充带检查API开销。

本次冻结的是CPU参考和格式部分。GPU源码及编译参数未绑定；环境恢复后应新建完整契约与新run，不回写本尝试。G0探针使用的编译选项保留在构建日志，不冒充已验证的codec编译选项。

## 3 正确性结果

完整G1正确性状态为`not_run`。没有CPU/GPU码流或BF16逐位一致性结论，也没有填写测量MSE。

| 输入类别 | 实际执行 | 完整GPU包一致/解码一致 | 备注 |
|---|---|---|---|
| CPU语义边界与适配器 | 已通过CPU探针 | 未测 | 常数、全零、内部溢出、阈值顺序、输入不变、无LCG推进、空输入 |
| 12个真实梯度 | 来源/哈希/变换与阈值准备 | 未测 | 仅1101更新前100/500/2500，4 worker；未读holdout |
| 全部G1合成与非法包矩阵 | 未执行 | 未测 | 不用CPU探针冒充完整G1覆盖 |

## 4 GPU性能结果

未进行GPU codec测量。计划矩阵保留在 [benchmark_config.json](benchmark_config.json)，K尚未预试验冻结，session和计时块均未开始。`benchmark_raw.jsonl`、`benchmark_summary.jsonl`不生成；延迟、p95、吞吐和加速比均未定义。不能将这次环境查询的GPU负载视为基准结果。

## 5 字节、显存与性能剖析

L=256时，b4局部包为160+16+2=178 B，即5.5625 bit/元素；b8为288+16+2=306 B，即9.5625 bit/元素。这是参照格式及CPU零输入探针结果，不是已完成GPU打包或网络发送。真实梯度逻辑长度818048，补齐长度818176；每个输入与阈值的实际内存字节列在输入/夹具清单。

GPU工作显存分项、allocated/reserved测量和性能trace尚未执行。没有基于trace的优化结论，也没有生成无数据图表。固定语义与CPU参照字节不能替代GPU紧凑device buffer的验收。

## 6 未解决问题与下一步

| 问题 | 影响 | 证据 | 建议处理 |
|---|---|---|---|
| 未找到可用Toolkit/nvcc；CUDA_HOME未设置 | 阻止实际扩展构建，G0门槛未通过 | [发现记录](logs/environment_discovery.json)、[构建日志](logs/minimal_cuda_build.log) | 补齐完整编译工具链后在新run重做最小编译与调用 |
| MSVC19.51与目标Toolkit组合未验证 | 仅有cl.exe不代表CUDA可编译 | [环境](environment.json)、[官方要求记录](logs/toolchain_reference_sources.json) | 选匹配宿主toolset并实际验证，保留现有环境 |
| cl版本输出解码警告 | 需在完整工具链核验时解决；不是当前主阻碍 | [构建日志](logs/minimal_cuda_build.log) | 检查进程内输出编码/语言环境 |
| G1实现及矩阵未执行 | 完整G1 not_ready | [问题列表](issues.jsonl) | G0通过后按b4→b8推进，保留同阈值和BF16契约 |

搜索覆盖PATH、CUDA注册表及记录中的常见/非标准安装目录；两处目录访问被拒绝，不能声称进行了完整全盘排查。现有MSVC为19.51.36260，CUDA11.8官方Windows支持表列出193x、192x、191x，当前版本不在该表中；这是兼容性待验证事项，本次实际失败发生在调用编译器之前。[NVIDIA CUDA11.8 Windows要求]({guide})

选择下一步：**fix_environment**。本次没有安装软件、改旧Python环境或更新全局工具链。单独填写CUDA_HOME不能代替真实Toolkit文件；修复后以真实扩展编译及GPU调用结果为准。G2–G4不在本轮执行范围。

## 7 复现与结论边界

实际入口与命令见 [REPRODUCE.md](REPRODUCE.md)。完整目录：`{run.as_posix()}`。源码快照在`code/`；大夹具留在本机并提供SHA，见 [fixtures_manifest.json](fixtures_manifest.json)。失败日志与未执行项全部保留。

本轮仅形成G0环境、CPU参考和输入准备证据。局部GPU codec一致性与性能、完整DynamiQ GPU路径、新训练质量、多卡吞吐及训练加速均未验证。此前项目结果没有修改，也未被计入本轮新成果。

## 8 交付文件核对

- [x] summary、环境、CPU部分契约、计划基准配置已填写，template_only=false；明确GPU绑定尚缺。
- [x] CPU参考、适配器、最小构建探针、输入、LUT、阈值与本地夹具哈希已记录。
- [x] 源码快照、依赖快照、实际命令、失败构建日志与问题清单可取得。
- [x] 未生成的文件与原因在summary.missing_artifacts逐项列出，未用空文件冒充执行。
- [ ] G1正确性、原始计时块、性能汇总与显存记录：未执行。
- [ ] profiler trace、三组数据图表及完整G1验收：未执行。
'''
    (run/'REPORT.md').write_text(report,encoding='utf-8')
    # Snapshot actual source bytes; original historical reference files stay read-only.
    source_files=[]
    own=sorted({p for folder in ('src','scripts') for p in (ROOT/folder).rglob('*')
                if p.is_file() and p.suffix in ('.py','.ps1','.cpp','.cu')})
    for p in own:
        relative=Path('code')/p.relative_to(ROOT);target=run/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
        role='cpu_reference' if p.name in ('cpu_reference.py','__init__.py') else (
            'test' if p.name=='probe_reference_semantics.py' else 'report_generator' if p.name==Path(__file__).name else 'build_script')
        source_files.append({'path':relative.as_posix(),'role':role,'sha256':sha(target),'bytes':target.stat().st_size,'original_path':str(p)})
    reference_root=Path('E:/dacomp/research/spl_model_communication')
    for entry in refs['reference_files']:
        p=Path(entry['path']); assert sha(p)==entry['sha256']
        relative=Path('code/reference')/p.relative_to(reference_root);target=run/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
        source_files.append({'path':relative.as_posix(),'role':'cpu_reference','sha256':sha(target),'bytes':target.stat().st_size,'original_path':str(p)})
    for name in ('codec_contract.json','benchmark_config.json','requirements.txt'):
        p=run/name;source_files.append({'path':name,'role':'config','sha256':sha(p),'bytes':p.stat().st_size})
    write(run/'source_manifest.json',{'schema_version':'gpu_codec_v1.0','run_id':run.name,
        'git_commit':None,'git_note':'File manifests used; no repository commit asserted.',
        'files':source_files,'probe_is_not_GPU_codec':True})
    for entry in fixtures['files']+[fixtures['lookup_tables']]: assert sha(Path(entry['path']))==entry['sha256']
    print(json.dumps({'run_id':run.name,'execution_status':summary['execution_status'],
        'g1_acceptance':summary['g1_acceptance'],'report':str(run/'REPORT.md'),
        'summary':str(run/'summary.json'),'source_files':len(source_files),'prepared_real_inputs':12}))


if __name__=='__main__': main()
