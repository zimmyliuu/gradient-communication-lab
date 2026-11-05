# G1a 实际执行记录与复现边界

工作目录：`E:/dacomp/research/gradient_comm_lab`。Shell：PowerShell。解释器：`E:\dacomp\.venv\Scripts\python.exe`，Python 3.10.3、PyTorch 2.4.1+cu118。始终经`scripts/python.ps1`设置本项目缓存并复用已验证解释器；没有在旧项目目录执行测试。

以下命令来自实际执行台账、工具链退出记录和当前报告生成进程；不把计划命令写成已完成。列表按证据来源分组，并非时间顺序或可直接从上到下重放的顺序。正式证据必须使用`execution_registration.json`中冻结的源、契约和输入；注册必须在正式正确性之前完成。阶段顺序为环境/工具链发现与准备、夹具生成、pytest和preliminary、注册、正式正确性、K pilot、5-session measurement、独立profile、CPU汇总作图、报告。

下列裸Python路径形式保留的是日志中的已执行子进程argv，不表示跳过项目启动环境。复现时以`scripts/python.ps1`启动相同的脚本与参数，保证使用本项目缓存配置；即用`./scripts/python.ps1`替换原argv中的Python可执行文件部分，后续参数保持不变。原始命令展示只用于追溯实际调用，不能向已封存run原地写入。

来源：`logs/correctness_commands.json`。结果/证据：82 fixtures: 36 synthetic, 34 edge, 12 exact reused real-gradient fixtures。

```powershell
./scripts/python.ps1 scripts/check_torch_codec.py --run-dir results/gpu_codec_v1/gpu_codec_v1_20261101_002 --prepare-only
```
来源：`logs/correctness_commands.json`。结果/证据：36 passed; preliminary smoke only。

```powershell
./scripts/python.ps1 scripts/check_torch_codec.py --run-dir results/gpu_codec_v1/gpu_codec_v1_20261101_002 --output logs/correctness_smoke_01.jsonl --contract-sha256 preliminary_not_formal --limit 3
```
来源：`logs/correctness_commands.json`。结果/证据：20 passed, 18 test-helper failures; retained。

```powershell
./scripts/python.ps1 -m pytest tests/test_torch_codec.py -q --junitxml=results/gpu_codec_v1/gpu_codec_v1_20261101_002/logs/pytest_torch_codec_attempt01.xml
```
来源：`logs/correctness_commands.json`。结果/证据：38 passed。

```powershell
./scripts/python.ps1 -m pytest tests/test_torch_codec.py -q --junitxml=results/gpu_codec_v1/gpu_codec_v1_20261101_002/logs/pytest_torch_codec_attempt02.xml
```
来源：`logs/correctness_commands.json`。结果/证据：1068 passed; retained preliminary evidence before supplemental decoder top-scale grid。

```powershell
./scripts/python.ps1 scripts/check_torch_codec.py --run-dir results/gpu_codec_v1/gpu_codec_v1_20261101_002 --output logs/correctness_preliminary_full01.jsonl --contract-sha256 preliminary_not_formal *> results/gpu_codec_v1/gpu_codec_v1_20261101_002/logs/correctness_preliminary_full01.log
```
来源：`logs/correctness_commands.json`。结果/证据：1101 passed; retained preliminary evidence before registration verification and deterministic-setting additions。

```powershell
./scripts/python.ps1 scripts/check_torch_codec.py --run-dir results/gpu_codec_v1/gpu_codec_v1_20261101_002 --output logs/correctness_preliminary_full02.jsonl --contract-sha256 preliminary_not_formal *> results/gpu_codec_v1/gpu_codec_v1_20261101_002/logs/correctness_preliminary_full02.log
```
来源：`logs/correctness_commands.json`。结果/证据：1101 passed, 0 failed, 0 skipped; registration verified; deterministic algorithms true; fill uninitialized memory false。

```powershell
./scripts/python.ps1 scripts/check_torch_codec.py --run-dir results/gpu_codec_v1/gpu_codec_v1_20261101_002 --contract-sha256 068400a4067849390aec27a01d9f46811e395f0765f63ccb67fdeda82e470e7d *> results/gpu_codec_v1/gpu_codec_v1_20261101_002/logs/correctness_formal.log
```
来源：`logs/execution_commands.json`。结果/证据：['execution_registration.json', 'codec_contract.json', 'logs/prior_run_integrity_before.json']。

```powershell
./scripts/python.ps1 scripts/register_g1a.py --run-dir results/gpu_codec_v1/gpu_codec_v1_20261101_002
```
来源：`logs/execution_commands.json`。结果/证据：['logs/benchmark_pilot_console.log', 'logs/K_pilot.jsonl', 'benchmark_config.json']。

```powershell
./scripts/python.ps1 scripts/benchmark_torch_codec.py --run-dir results/gpu_codec_v1/gpu_codec_v1_20261101_002 --mode pilot *> results/gpu_codec_v1/gpu_codec_v1_20261101_002/logs/benchmark_pilot_console.log
```
来源：`logs/execution_commands.json`。结果/证据：['requirements.txt']。

```powershell
./scripts/python.ps1 -m pip freeze --all > results/gpu_codec_v1/gpu_codec_v1_20261101_002/requirements.txt
```
来源：`logs/execution_commands.json`。结果/证据：['logs/benchmark_measure_console.log', 'logs/benchmark_execution.json', 'benchmark_raw.jsonl', 'memory.jsonl']。

```powershell
./scripts/python.ps1 scripts/benchmark_torch_codec.py --run-dir results/gpu_codec_v1/gpu_codec_v1_20261101_002 --mode measure
```
来源：`logs/execution_commands.json`。结果/证据：['logs/benchmark_profile_console.log', 'traces/manifest.json']。

```powershell
./scripts/python.ps1 scripts/benchmark_torch_codec.py --run-dir results/gpu_codec_v1/gpu_codec_v1_20261101_002 --mode profile
```
来源：`logs/execution_commands.json`。结果/证据：['logs/analysis_attempt01_cpu_only_trace_misclassification.json', 'logs/benchmark_summary_attempt01.log', 'logs/summarize_g1a_attempt01.py']。

```powershell
./scripts/python.ps1 scripts/summarize_g1a.py --run-dir results/gpu_codec_v1/gpu_codec_v1_20261101_002
```
来源：`logs/execution_commands.json`。结果/证据：['logs/benchmark_summary_final.log', 'analysis.json', 'benchmark_summary.jsonl']。

```powershell
./scripts/python.ps1 scripts/summarize_g1a.py --run-dir results/gpu_codec_v1/gpu_codec_v1_20261101_002
```
来源：`logs/toolchain_summary.json`。结果/证据：详见来源日志的退出状态与输出。

```powershell
E:\dacomp\.venv\Scripts\python.exe scripts/probe_toolchain_v2.py --run-dir results/gpu_codec_v1/gpu_codec_v1_20261101_002 --attempt a01 --phase compile
```
来源：`logs/toolchain_summary.json`。结果/证据：详见来源日志的退出状态与输出。

```powershell
E:\dacomp\.venv\Scripts\python.exe scripts/probe_toolchain_v2.py --run-dir results/gpu_codec_v1/gpu_codec_v1_20261101_002 --attempt a02 --phase compile
```
来源：`logs/toolchain_summary.json`。结果/证据：详见来源日志的退出状态与输出。

```powershell
E:\dacomp\.venv\Scripts\python.exe scripts/probe_toolchain_v2.py --run-dir results/gpu_codec_v1/gpu_codec_v1_20261101_002 --attempt a03 --phase compile
```
来源：`logs/toolchain_summary.json`。结果/证据：详见来源日志的退出状态与输出。

```powershell
E:\dacomp\.venv\Scripts\python.exe scripts/probe_toolchain_v2.py --run-dir results/gpu_codec_v1/gpu_codec_v1_20261101_002 --attempt a04 --phase compile
```
来源：`logs/toolchain_summary.json`。结果/证据：详见来源日志的退出状态与输出。

```powershell
E:\dacomp\.venv\Scripts\python.exe scripts/probe_toolchain_v2.py --run-dir results/gpu_codec_v1/gpu_codec_v1_20261101_002 --attempt a04 --phase gpu
```
来源：`logs/cupti_trace_a03.json`。结果/证据：['logs/cupti_trace_a03.json', 'traces/gpu_activity_manifest.json']。

```powershell
E:\dacomp\.venv\Scripts\python.exe scripts/profile_cuda_trace_v2.py --run-dir results/gpu_codec_v1/gpu_codec_v1_20261101_002 --attempt a03
```
来源：`logs/cupti_trace_audit_a03.json`。结果/证据：['logs/cupti_trace_audit_a03.json', 'traces/gpu_activity_manifest.json']。

```powershell
E:\dacomp\.venv\Scripts\python.exe scripts/audit_cuda_trace_v2.py --run-dir results/gpu_codec_v1/gpu_codec_v1_20261101_002 --attempt a03
```
来源：`当前报告生成进程`。结果/证据：读取已完成证据并生成本报告、summary、issues、REPRODUCE；成功退出后生效。

```powershell
./scripts/python.ps1 scripts/write_g1a_report.py --run-dir results/gpu_codec_v1/gpu_codec_v1_20261101_002
```

## 输出与重跑注意

正式正确性见`correctness.jsonl`和`correctness_acceptance.json`；主测量见`benchmark_raw.jsonl`、`memory.jsonl`和`logs/benchmark_execution.json`；成功的独立GPU活动见`traces/gpu_activity_manifest.json`，旧`traces/manifest.json`对应CPU-only失败诊断；汇总与四张图见`analysis.json`列出的derived_files。正式源快照保存在`code/`，依赖清单是`requirements.txt`。

记录中的路径是本次真实执行路径，不应对已封存run原地重跑注册、prepare、正确性或性能命令；这些入口会拒绝覆盖关键证据。需要新实验时建立新run并重新绑定输入/源/契约。CPU汇总与报告属于派生产物，封存前可以从同一不可变原始记录重建。

本报告生成时，最后的seal属于后续交付步骤；只有实际执行并产出审计/完整文件清单后才构成封存证据，不将它预记为已完成实验。优化CUDA codec入口尚未实现；G2状态推进/多跳与训练验收也未执行。

初次pytest失败、preliminary日志及工具链各尝试均保留原记录。夹具以连续张量原始字节哈希验证；重新torch.save归档的文件哈希可能因文件名/序列化版本不同，应同时比较input/LUT/main/scale内容身份。真实梯度复用的12个旧文件路径见`fixtures_manifest.json`，无留出数据读取。
