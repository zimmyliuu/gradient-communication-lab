# 固定尺度加数诊断复现说明

实际工作目录：`E:/dacomp/research/gradient_comm_lab`。解释器为只读复用的 `E:/dacomp/.venv/Scripts/python.exe`，经 `scripts/python.ps1` 将缓存放到本项目。PyTorch/CUDA/MSVC/UTF-8/隔离cuRAND头文件及全部flags见 environment.json、两份 build_manifest.json。没有修改旧结果、原decoder、共享PyTorch或驱动。

## 从保存包和输出复算

以下命令已实际执行并通过，记录在 logs/recompute_verification.json。它只读本轮实际包/输出、既有输入和旧uniform输出，重算1,080条full/H/rest指标、96组包变化位置、54组配对统计及五项门槛；不加载codec或调用GPU，也不写入封存目录。

```powershell
.\scripts\python.ps1 -X utf8 scripts/epsilon_diagnostic/analyze.py --run-dir results/epsilon_diagnostic_v1/epsilon_diagnostic_v1_20261105_001 --verify-only
```

数值还经独立CPU实现验证，未导入上述分析公式或codec；15,009项检查通过。其实际入口如下，结果为 logs/independent_effect_audit.json，源码哈希在该JSON内。该入口会写审计文件，历史记录保留后不需要重跑覆盖。

```powershell
.\scripts\python.ps1 -X utf8 results/epsilon_diagnostic_v1/epsilon_diagnostic_v1_20261105_001/logs/independent_effect_audit.py results/epsilon_diagnostic_v1/epsilon_diagnostic_v1_20261105_001
```

## 已执行入口与阶段顺序

E0 CPU预查与构建曾并行；每份日志保留实际时间，下面按阶段依赖列出。以下会写入的历史入口不要在已封存目录重放。新建任务需使用新的run和明确的计划，当前不安排新运行。

```powershell
.\scripts\python.ps1 -X utf8 scripts/epsilon_diagnostic/precheck.py --run-dir results/epsilon_diagnostic_v1/epsilon_diagnostic_v1_20261105_001
.\scripts\python.ps1 -X utf8 scripts/epsilon_diagnostic/build_conditions.py --run-dir results/epsilon_diagnostic_v1/epsilon_diagnostic_v1_20261105_001 --condition control_add1e7
.\scripts\python.ps1 -X utf8 scripts/epsilon_diagnostic/build_conditions.py --run-dir results/epsilon_diagnostic_v1/epsilon_diagnostic_v1_20261105_001 --condition diagnostic_add0
```

第二条件入口曾在编译前因计数guard把环境日志误算为构建次数而停止；修复后用同一命令重试。实际只编译两份条件，未新增第三个加数。失败console、入口diff和修复说明分别在 logs/build_diagnostic_add0_console.log、logs/build_entry_guard_repair.diff、logs/build_preflight_guard_repair.json；成功第二条件console为 logs/build_diagnostic_add0_a01_console.log。两个实际编译器源入口快照位于各build目录中，不用最新入口冒充历史源。

构建manifest使用显式新输出路径，未调用写死旧RUN的build_official.py。两个encoder来源于旧实际编译源码，编译选项只差DIAG_B8_SCALE_ADD；module、source、flags、PID和耗时见 build/control_add1e7/build_manifest.json、build/diagnostic_add0/build_manifest.json。新构建进程不运行kernel。

完成来源核验、冻结96个池和H、构建及分析程序后，实际执行：

```powershell
.\scripts\python.ps1 -X utf8 scripts/epsilon_diagnostic/freeze.py --run-dir results/epsilon_diagnostic_v1/epsilon_diagnostic_v1_20261105_001
.\scripts\python.ps1 -X utf8 scripts/epsilon_diagnostic/run_stages.py --run-dir results/epsilon_diagnostic_v1/epsilon_diagnostic_v1_20261105_001
```

run_stages实际启动8个独立Python进程，顺序是：qualification A encode→原decoder decode→qualification B encode→原decoder decode→formal A encode→原decoder decode及96控制parity→formal B encode→原decoder decode。每个进程通过execute.py的 `--operation`、`--condition`、`--phase`、`--build-manifest` 显式限定；完整参数生成器在冻结源run_stages.py，进程实际模块/manifest哈希与PID在logs/*_process.json，结果在对应jsonl和completion记录。运行器不通过同一进程更换sys.modules来切换条件。

正式A/B各96次编码，资格各6个b8夹具和1个b4夹具。A正式96次直接作为控制组；无额外重建控制矩阵。每份A包逐字节匹配旧包，所有A解码逐位匹配旧BF16输出后，才执行正式B。codec.py是旧薄wrapper的代码副本，默认描述不代表新加载的派生encoder未改算术，实际身份以本次contract及manifest为准。

随后从保存数据计算，再独立复核、只读重算并生成报告：

```powershell
.\scripts\python.ps1 -X utf8 scripts/epsilon_diagnostic/analyze.py --run-dir results/epsilon_diagnostic_v1/epsilon_diagnostic_v1_20261105_001
.\scripts\python.ps1 -X utf8 scripts/epsilon_diagnostic/analyze.py --run-dir results/epsilon_diagnostic_v1/epsilon_diagnostic_v1_20261105_001 --verify-only
.\scripts\python.ps1 -X utf8 scripts/epsilon_diagnostic/deliver.py --run-dir results/epsilon_diagnostic_v1/epsilon_diagnostic_v1_20261105_001
```

## 数据与身份

input_randomness_manifest.json列96个配对键、原文件与张量hash、原seed/pool SHA、重建pool SHA及旧包/输出和uniform引用。读取既有input_logical，不重新预除，不复用旧padding；每方法使用D=818048、P=818176。真实输入不复制整套历史夹具，路径与哈希用于验收读取。

新A/B包与输出都保存在packets/formal和decoded/formal；资格数据在qualification子目录。所有输入、池、输出合法性通过，无域失败反例。本次保留每对全部变化位置NPZ，并明确本机文件外壳不是875704B通信预算。uniform直接复用旧96输出，没有重新编码或改参数，也不属于A/B共同随机数设计。

source_manifest.json在调用前冻结runner、包解析、统计程序、两份派生源码/二进制和原decoder身份；旧条件来源保持只读。code/保留这些执行源快照；报告源单独记录在delivery_source.json，独立审计源码及hash在logs中。E0逐组数据与估计C8的不确定性保存于scale_precheck.json和logs/e0_groups_*.npz。

## 完整性

完整结果目录交付，不另生成ZIP或复制旧3.57GB结果包。integrity_manifest.json覆盖所有新文件并排除自身，旧包/输出/输入/源码依赖列为external_references且逐项验证hash。外部回执位于相邻的epsilon_diagnostic_v1_20261105_001.integrity_verification.json，保存清单hash。封存工具自身在code/delivery/seal.py；`--seal`为本次实际封存入口，省略时只读验证。

```powershell
.\scripts\python.ps1 -X utf8 results/epsilon_diagnostic_v1/epsilon_diagnostic_v1_20261105_001/code/delivery/seal.py --run-dir results/epsilon_diagnostic_v1/epsilon_diagnostic_v1_20261105_001 --seal
```

本轮已到停止条件。D2、训练、其他加数、输入缩放或区域重选均未执行。
