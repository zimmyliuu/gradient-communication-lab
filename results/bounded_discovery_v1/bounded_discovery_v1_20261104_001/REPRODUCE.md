# 复算与执行记录

工作目录：`E:/dacomp/research/gradient_comm_lab`。实际解释器 `E:/dacomp/.venv/Scripts/python.exe`，Python 3.10.3、PyTorch 2.4.1+cu118；项目包装入口 `scripts/python.ps1` 只复用解释器，把缓存放在本项目。GPU 为 RTX 4000 Ada，CUDA 11.8、MSVC 14.38、Windows SDK 10.0.26100.0；完整路径、版本和编译参数见 environment.json、logs/official_build_a03.json。

## 验收只需复算摘要

以下命令已经执行，退出码 0；`logs/raw_summary_replay.json` 中 status=valid、root_existing_summary_matches=true。它不调用 GPU、不启动 codec、不写回封存结果，只从 raw 重建统计并核对已有汇总和唯一键。

```powershell
.\scripts\python.ps1 -X utf8 scripts/bounded_discovery/summarize.py --run-dir results/bounded_discovery_v1/bounded_discovery_v1_20261104_001 --verify-only
```

完整独立数值复算也已执行一次，加载已保存的 780 个 CPU BF16 张量与 12 份逻辑输入，以 FP64 重新计算所有局部/聚合误差，产出两张图及 metrics_summary.jsonl。命令与完整输出在 logs/summary_revision2_console.txt、summary_analysis.json：

```powershell
.\scripts\python.ps1 -X utf8 scripts/bounded_discovery/summarize.py --run-dir results/bounded_discovery_v1/bounded_discovery_v1_20261104_001 --recompute-decoded
```

此第二命令再次执行会写到 `summary_revisions/` 新目录；封存后无需运行写入式重算。需要无写入的二次全张量验证，可组合现有 CLI 选项，但该组合未在本轮执行，不列为历史结果。图表字体、样式和数值均在冻结的 summarize.py 中；最多两张主要图。

## 已执行阶段与依赖顺序

以下列出实际 Python 入口。独立 CPU 读源码/均匀量化检查曾并行进行，精确时间以各日志时间戳为准；D1 两次矩阵严格顺序执行，没有并行 GPU 比较。构建 a01/a02 的失败是历史证据，不要求验收重新触发。涉及写入的历史命令不要指向已封存目录重放。

1. 取得官方 Git tree 元数据及固定 commit 原始 blobs，Git blob SHA1 和本地 SHA256 双重校验；git clone 曾因网络 reset 失败，后续使用不可变官方 raw URL。原 API 响应、30份原始文件、许可证、来源 URL 在 logs/official_git_tree.json、logs/official_source_manifest.json、code/upstream/。

   ```powershell
   .\scripts\python.ps1 -X utf8 scripts/bounded_discovery/fetch_official.py
   .\scripts\python.ps1 -X utf8 scripts/bounded_discovery/prepare_inputs.py --run-dir results/bounded_discovery_v1/bounded_discovery_v1_20261104_001
   ```

2. 构建尝试 a01 → 第一轮修复 a02 → 下载验证隔离 cuRAND 头文件 → 第二轮修复 a03。a01 使用缺省编码失败，后续 UTF-8 开关见日志。

   ```powershell
   .\scripts\python.ps1 scripts/bounded_discovery/build_official.py --attempt a01
   .\scripts\python.ps1 -X utf8 scripts/bounded_discovery/build_official.py --attempt a02 --remove-unused-posix-include
   .\scripts\python.ps1 -X utf8 scripts/bounded_discovery/fetch_curand_headers.py
   .\scripts\python.ps1 -X utf8 scripts/bounded_discovery/build_official.py --attempt a03 --remove-unused-posix-include --isolated-curand-headers
   ```

   NVIDIA 归档 URL、官方 SHA256、提取头文件清单见 logs/official_curand_dependency.json；原目录 `.toolchains/bounded_discovery_v1/curand-10.3.0.86`。只有本项目隔离目录新增头文件，未修改共享解释器/驱动。构建复用只读 `scripts/probe_toolchain_v2.py`，其副本和哈希在 delivery_sources.json。CUDA/MSVC/SDK与整个解释器不复制到交付目录。成功的 eden_utils.pyd 及实际编译源码/diff 在 build/official_a03/；失败尝试均保留。

3. uniform 自检 a01 保留空张量哈希辅助函数的18项失败；修复辅助函数后执行 a02。量化算术未改变。旧测试源码在 logs/uniform_a01_code/，最终205检查通过。

   ```powershell
   .\scripts\python.ps1 -X utf8 scripts/bounded_discovery/check_uniform.py --output results/bounded_discovery_v1/bounded_discovery_v1_20261104_001/logs/uniform_selfcheck_a01.json
   .\scripts\python.ps1 -X utf8 scripts/bounded_discovery/check_uniform.py --output results/bounded_discovery_v1/bounded_discovery_v1_20261104_001/logs/uniform_selfcheck_a02.json
   ```

4. 官方48配置自测和14组小样本；qualify_official.py 内部实际启动 `--fresh-decode` 新进程解码14包。随后极值/非默认stream检查、CPU拆包复核、资格定稿：

   ```powershell
   .\scripts\python.ps1 -X utf8 scripts/bounded_discovery/qualify_official.py
   .\scripts\python.ps1 -X utf8 scripts/bounded_discovery/probe_official_boundaries.py
   .\scripts\python.ps1 -X utf8 scripts/bounded_discovery/audit_d0_packets.py --run-dir results/bounded_discovery_v1/bounded_discovery_v1_20261104_001
   .\scripts\python.ps1 -X utf8 scripts/bounded_discovery/finalize_qualification.py
   ```

5. D0通过后，实际零输入包只用来选择 q，未看误差选配置。冻结v1，完整执行；独立代码审查发现两个审计容错缺口后修正runner，保留v1全部资料并冻结v2，保持输入、方法、预算、种子和算术不变重跑。

   ```powershell
   .\scripts\python.ps1 -X utf8 scripts/bounded_discovery/freeze_comparison.py --run-dir results/bounded_discovery_v1/bounded_discovery_v1_20261104_001
   .\scripts\python.ps1 -X utf8 scripts/bounded_discovery/run_comparison.py --run-dir results/bounded_discovery_v1/bounded_discovery_v1_20261104_001
   .\scripts\python.ps1 -X utf8 scripts/bounded_discovery/revise_execution.py --run-dir results/bounded_discovery_v1/bounded_discovery_v1_20261104_001
   .\scripts\python.ps1 -X utf8 scripts/bounded_discovery/run_comparison.py --run-dir results/bounded_discovery_v1/bounded_discovery_v1_20261104_001
   ```

   当前源码是v2；v1的实际源文件在 revisions/contract_v1/code/。两版契约、原始记录、包和BF16输出完整保留，不能用当前runner冒称重放v1源码。两版属于同seed复现，统计重复数仍为8。

6. 已执行本页开头的完整独立复算；另外运行逐包/逐输出跨修订审计，26,613检查通过：

   ```powershell
   .\scripts\python.ps1 results/bounded_discovery_v1/bounded_discovery_v1_20261104_001/logs/revision_equivalence_audit_source.py results/bounded_discovery_v1/bounded_discovery_v1_20261104_001
   .\scripts\python.ps1 -X utf8 scripts/bounded_discovery/write_delivery.py --run-dir results/bounded_discovery_v1/bounded_discovery_v1_20261104_001
   ```

   随后执行开头 `--verify-only` 入口，确认现有 summary 与raw复算完全一致。报告程序是后处理；其源码与只读构建辅助依赖另外封存在 delivery_sources.json，不修改已冻结的实验源 manifest。

## 输入、实际包与随机状态

input_manifest.json 明确12份既有development夹具的绝对来源、文件哈希和逻辑BF16哈希；原始旧项目只读，未打开旧holdout、未采集新数据。取 `input_logical`，不重新除4。未重复打包模型、旧G1/G2a整包及原始历史夹具。

packets/ 保存780份真实字节，decoded/ 保存780份torch.save(CPU BF16 Tensor)。torch.save文件外壳不计入通信预算。packet_accounting.jsonl 保存真实包长度、分组件字节、packet SHA、输入/方法/repeat/seed、官方随机池哈希、uniform初末generator状态哈希、解码输出文件和原始张量哈希。官方包通过已声明的静态配置解码，uniform所有逐包元数据已在16B header计费；BF16参照就是裸BF16字节。

采用完整结果目录交付。integrity_manifest.json列全部交付文件的大小和SHA256，排除清单本身；其外部封存回执为相邻的 `bounded_discovery_v1_20261104_001.integrity_verification.json`。没有ZIP。D2及任何新训练命令均未执行，也未安排自动继续。

## 完整目录封存

实际封存/逐文件复验入口如下。封存程序先校验冻结源码及二进制、严格JSON、契约与输入身份，再建立清单，并重新读取所有文件验证大小、SHA256、无额外或缺失文件。外部回执保存清单哈希，不写进清单本身。

```powershell
.\scripts\python.ps1 -X utf8 results/bounded_discovery_v1/bounded_discovery_v1_20261104_001/code/delivery_tools/seal_delivery.py --run-dir results/bounded_discovery_v1/bounded_discovery_v1_20261104_001 --seal
```

封存后的只读核验使用同一入口但省略 `--seal`；该核验模式不写文件。
