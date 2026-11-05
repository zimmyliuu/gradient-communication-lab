# 梯度通信工程项目数据位置索引

核对日期：2026-11-05。项目根目录：`E:/dacomp/research/gradient_comm_lab/`。本清单只读既有交付和复核材料，不包含新实验。记录数包含同一输入的检查分项和随机重复，不等于独立训练样本数。

项目已完成本地低比特 CUDA 编解码、单 GPU 逻辑 ring 验证、官方固定编码组件的预算比较，以及固定尺度加数的单因素诊断。最新诊断确认开发输入上的同预算误差改善；原继续门槛未通过，当前论文路线停止追加实验，独立研究贡献尚未确立。当前工程成果适合整理为可复现案例。

## 阶段和权威结果位置

每个结果目录均包含 `REPORT.md`、`summary.json` 和 `REPRODUCE.md`。表中路径是原始本机位置；GitHub 中的精简副本不能替代被排除的大型数据。

| 阶段 | 结果目录 | 已确认结果与状态 |
|---|---|---|
| G0 环境和参考准备 | `E:/dacomp/research/gradient_comm_lab/results/gpu_codec_v1/gpu_codec_v1_20261101_001/` | 当时 CUDA Toolkit/nvcc 缺失，扩展构建 blocked；冻结 CPU 参考、契约和 12 份真实输入。没有 GPU codec 正确性或性能数据。此阻碍随后解决，保留为历史交付 |
| G1a PyTorch CUDA 基线 | `E:/dacomp/research/gradient_comm_lab/results/gpu_codec_v1/gpu_codec_v1_20261101_002/` | 最小扩展编译/调用通过；1101 项正确性记录通过；104 个性能 case、10400 个计时块、520 条显存记录。只有 eager 基线，完整 G1 当时仍 incomplete |
| 完整 G1 自定义 CUDA 与同轮配对基线 | `E:/dacomp/research/gradient_comm_lab/results/gpu_codec_v1/gpu_codec_v1_20261104_001/` | G1 pass；2098 项正确性记录；104 case × 2 方法，20800 个计时块，1040 条显存记录，8 组 GPU trace |
| G2a 单 GPU 逻辑 ring | `E:/dacomp/research/gradient_comm_lab/results/logical_ring_v1/logical_ring_v1_20261104_001/` | G2a pass；174 个 collective，其中 6 个真实梯度用例；4741 条正确性、4864 条量化、6848 条有向消息、13152 条中间态记录。完整 G2 incomplete |
| D0/D1 官方固定组件有限比较 | `E:/dacomp/research/gradient_comm_lab/results/bounded_discovery_v1/bounded_discovery_v1_20261104_001/` | 官方资格 pass；3 快照 × 4 worker、9 配置，随机方法各 8 次重复，780 条局部和 195 条 one-shot 聚合记录。原决策保留 `no_actionable_signal`；复核后仅另立一次 epsilon 定位任务 |
| E0/E1/E2 固定尺度加数诊断 | `E:/dacomp/research/gradient_comm_lab/results/epsilon_diagnostic_v1/epsilon_diagnostic_v1_20261105_001/` | 96 组 A/B 配对全部有效、全部局部 SSE 改善。`partially_closes_gap`；`continuation_gate=not_met`，唯一未通过项为固定高能量区 H 闭合不足 50%。D2 未执行 |

旧阶段的 `next_action` 和 incomplete/blocked 是当时状态，不应当成项目当前状态。G1a 的单独计时不能拿来充当 G1 最终配对收益的分母；最终比较使用同轮重测的 PyTorch 基线。

## 可以用于项目展示的核心结果

### 本地 CUDA 编解码

名义 b4/b8 的本地修复 profile 实际主码为 **5/9 位**，不能与后面的官方 4/8 位编码器混称。指定 BF16 运算顺序、码流、LUT 索引和解码结果通过 CPU/GPU 一致性核验。

在 RTX 4000 Ada、N=4194304（8 MiB BF16）、标准正态 seed7301、预分配 trusted roundtrip 条件下：

| 配置 | 同轮 PyTorch GPU 中位延迟 | 自定义 CUDA GPU 中位延迟 | 五 session 配对比值中位数 |
|---|---:|---:|---:|
| 名义 b4，实际主码 5 位 | 7304.440 μs | 200.428 μs | 36.44 倍 |
| 名义 b8，实际主码 9 位 | 9251.784 μs | 233.630 μs | 39.41 倍 |

在已测 trace 配置中，每次 roundtrip 从 52 个 kernel 减为 2 个。该收益仅比较指定本地实现与 PyTorch eager，不代表领先成熟 codec，也不是物理 AllReduce 或训练加速。所有 WDDM 噪声和原始块均保留。

### 单 GPU 逻辑 ring

覆盖 worker 数 1/2/4、两位宽、raw FP32 与已预除 BF16 两入口，检查逐跳解码、BF16 部分和、消息所有权和最终各 rank 输出。四 worker 真实回放的每个 collective 有 24 次有向发送，名义 b4/b8 的实际发送字节分别为 3417600/5875200 B。网络时延、多卡吞吐及通信计算重叠均未测量。

### 官方组件诊断与可避免误差

官方来源：`https://github.com/CharlesHan24/DynamiQ`，冻结 commit：`96da2ee480bbfd8e2049de3fbd8595cc9d7a30b8`。固定 BF16、strategy=3、chunk=16，主码实际 4/8 位。官方 b8 为 875704 B/worker，uniform_g128_q8 为 843628 B/worker，后者少 3.6629%。

三个快照中 uniform/官方 b8 的局部 SSE 比分别为约 0.78150、1.16829、1.08921，显示开发输入上的排序反转，并非 uniform 普遍更优。

后续唯一干预为 b8 raw scale 的 `+1e-7f` 改为 `+0.0f`，原 `1e-30` 下限、其他保护和数值路径均保留。两条件共用原随机池，控制 96 包及 BF16 输出与封存结果完全一致，均由原解码器解码。

| 快照 | 按均值之比计算的局部 SSE 降幅 |
|---|---:|
| update100 | 12.053945% |
| update500 | 14.814818% |
| update2500 | 12.838681% |

全部 96 个局部配对均改善；包大小不变。update100 对 uniform 的差距整体闭合 55.166468%，固定 H 闭合 34.107726%。闭合比例和 SSE 降幅的分母不同，不能把 55.17% 写成自身误差下降。该结果支持当前开发输入上的因果作用，不证明部署安全性、泛化、训练精度或速度收益，也不构成新的尺度保护算法。

## 各项数据到哪里找

以下相对文件名均以第一节对应的结果目录为根；文件名已在本机核对存在。

| 需要查找的内容 | 所属阶段与文件 |
|---|---|
| G0 环境阻碍、CPU 语义、输入来源 | G0：`environment.json`、`cpu_reference_manifest.json`、`codec_contract.json`、`input_manifest.json`、`fixtures_manifest.json`、`logs/cpu_reference_semantics.json`、`logs/minimal_cuda_build.log` |
| G1a 早期基线原始计时 | G1a：`benchmark_raw.jsonl`、`benchmark_summary.jsonl`、`analysis.json`、`memory.jsonl`；该数据不用于最终 G1 加速比分母 |
| G1 最终配对计时与显存 | 完整 G1：`benchmark_raw.jsonl`、`benchmark_summary.jsonl`、`benchmark_config.json`、`pilot_protocol.json`、`analysis.json`、`memory.jsonl` |
| G1 正确性、编译和 trace | 完整 G1：`correctness.jsonl`、`correctness_case_manifest.json`、`logs/cuda_build_manifest.json`、`traces/g1b_trace_index.json`、`traces/`、`figures/` |
| G2a 实际消息字节和中间态 | G2a：`message_ledger.jsonl`、`quantization_events.jsonl`、`intermediate_checks.jsonl`、`input_output_checks.jsonl`、`buffer_lifetime_checks.jsonl`、`BUFFER_OWNERSHIP.md` |
| G2a 用例、输入和阈值 | G2a：`input_manifest.json`、`correctness_case_manifest.json`、`threshold_manifest.json`、`fixtures/`、`ring_contract.json` |
| D0/D1 数值、实际包和输出 | D0/D1：`metrics_raw.jsonl`、`metrics_summary.jsonl`、`PER_INPUT_RESULTS.md`、`packet_accounting.jsonl`、`packets/`、`decoded/` |
| 官方资格、源码与预算身份 | D0/D1：`baseline_qualification.json`、`experiment_contract.json`、`input_manifest.json`、`source_manifest.json`、`code/upstream/`、`build/`、`logs/official_contract_review.json` |
| epsilon 的 A/B/U 误差与变化位置 | E0/E1/E2：`metrics_raw.jsonl`、`paired_summary.json`、`packet_changes.jsonl`、`changes/`、`packets/formal/`、`decoded/formal/` |
| epsilon 输入、随机池、H 和尺度预查 | E0/E1/E2：`input_randomness_manifest.json`、`region_manifest.json`、`scale_precheck.json`、`logs/e0_groups_*.npz` |
| epsilon 单因素与控制复现 | E0/E1/E2：`experiment_contract.json`、`source_manifest.json`、`control_parity.jsonl`、`correctness.jsonl`、`build/`、`logs/build_pair_audit.json`、`logs/independent_effect_audit.json` |
| 交付完整性 | G0/G1a/G1/G2a：各自 `artifact_integrity.json`；D0/D1 与 epsilon：各自 `integrity_manifest.json`。各阶段的 `REPRODUCE.md` 解释清单、自身哈希回执及外部依赖 |

## 原始梯度、夹具及外部依赖

真实输入来自旧项目的一个小型 Transformer 开发轨迹，训练 seed=1101，更新前 100/500/2500，4 个 worker。逻辑维度 D=818048。数据变换为 FP32 梯度转 BF16，再在 BF16 中除以 4。后续读取 `input_logical` 时不得再次预除，也不得把旧 padding 当作新逻辑输入。

| 内容 | 本机原始位置 |
|---|---|
| 原始训练快照 | `E:/dacomp/research/spl_model_communication/results/p2/calibration_20261101_v1/seed_1101/snapshots/update_000100.pt`、`update_000500.pt`、`update_002500.pt` |
| 12 份接受的真实梯度夹具 | `E:/dacomp/research/gradient_comm_lab/fixtures/gpu_codec_v1_20261101_001/`；各具体文件名、文件/张量 SHA256 见 G0、G1 和 D0/D1 的 `input_manifest.json` |
| G1 合成和边界夹具 | `E:/dacomp/research/gradient_comm_lab/fixtures/gpu_codec_v1_20261101_002/`；完整 G1 的 `fixtures_manifest.json` 指向这些既有夹具 |
| G2a 自带夹具 | `E:/dacomp/research/gradient_comm_lab/results/logical_ring_v1/logical_ring_v1_20261104_001/fixtures/` |
| 历史 CPU 参考模块 | `E:/dacomp/research/spl_model_communication/src/splcomm/`，包括 `dynamiq.py`、`packing.py`、`core.py`、`ring.py`、`p2_replay.py`；精确身份见 G0 `cpu_reference_manifest.json`，各 run 的 `code/` 保留实际快照 |
| 共享已执行 Python | `E:/dacomp/.venv/Scripts/python.exe`；项目入口 `E:/dacomp/research/gradient_comm_lab/scripts/python.ps1` 负责项目缓存位置 |
| 局部工具链和缓存 | `E:/dacomp/research/gradient_comm_lab/.toolchains/`、`.cache/`；不应当作普通 Git 源文件上传 |

已执行环境为 Windows、Python 3.10.3、PyTorch 2.4.1+cu118、CUDA Toolkit 11.8、MSVC 19.38、RTX 4000 Ada；每轮实际版本、flags 和二进制身份以对应 `environment.json` 和 build manifest 为准。现有脚本仍有本机路径依赖，不是异机一键安装包。

epsilon 复用 D0/D1 的 96 份 uniform 输出和旧输入，不重复复制全部依赖。仅下载 epsilon 的精简报告无法重算张量级结果，必须同时具备其 `input_randomness_manifest.json` 与 `integrity_manifest.json` 所列外部文件。

## 当前代码入口

代码根：`E:/dacomp/research/gradient_comm_lab/src/gradient_comm_lab/`。

- `torch_codec.py`、`cuda_codec.py`、`csrc/codec.cpp`、`csrc/codec.cu`：本地 5/9 位 profile 的 eager 基线与自定义 CUDA。
- `cpu_reference.py`、`ring_reference.py`、`logical_ring.py`：参考适配、CPU 协议和单 GPU 逻辑 ring。
- `bounded_discovery/official_codec.py`、`bounded_discovery/uniform_codec.py`：官方固定编码器入口与显式预算均匀参照。
- `epsilon_diagnostic/codec.py`、`epsilon_diagnostic/packet.py`：派生条件入口与包解析。

任务入口根：`E:/dacomp/research/gradient_comm_lab/scripts/`。G1 包括 `build_cuda_codec.py`、`check_cuda_codec.py`、`benchmark_cuda_pair.py`、`summarize_g1b.py`；G2a 包括 `prepare_logical_ring.py`、`check_logical_ring.py`、`audit_logical_ring.py`；两轮诊断分别在 `bounded_discovery/` 与 `epsilon_diagnostic/` 子目录。

历史可复现身份应优先使用各 run 的 `code/` 快照。当前源码可继续维护，但不应覆盖封存目录或把新代码冒称历史执行代码。只读复算和会生成新结果的执行命令分别见对应 `REPRODUCE.md`。

## 研究审查与阶段修订位置

| 材料 | 本机位置与结论 |
|---|---|
| G0/G1a/G1/G2a 独立验收 | `E:/dacomp/research/gradient_comm_lab/reviews/` 中对应 run 名的 `.md`；依次记录历史接受范围和下一阶段条件 |
| D0/D1 复核与已有输出分解 | `E:/dacomp/research/gradient_comm_lab/reviews/bounded_discovery_v1_20261104_001_review.md`；补充数据在同级 `bounded_discovery_v1_20261104_001_audit/saved_block_analysis.json`，脚本为 `analyze_saved_blocks.py` |
| 早期论文定位草案 | `E:/dacomp/research/gradient_comm_lab/reports/spl_positioning_20261104_001/` 和 `spl_positioning_20261104_002/`；002 替代 001 的当时规划，均不是已成立贡献 |
| T0 理论可行性 | `E:/dacomp/research/gradient_comm_lab/reports/t0_theory_20261104_001/REPORT.md`、`THEORY.md`、`summary.json`；限定模型下可计算，方法属精确变量消元的特化，未建立独立优势，因此停止进入实验 |
| 其他选题审查 | `E:/dacomp/research/gradient_comm_lab/reports/topic_screen_20261104_001/REPORT.md`、`CANDIDATE_REVIEWS.md`、`summary.json`；整数聚合范围和旋转选择两个对象均未立项，未跑实验 |
| 后两轮执行约束 | `E:/dacomp/research/gradient_comm_lab/plans/bounded_discovery_v1/` 与 `plans/epsilon_diagnostic_v1/`，各含 `PLAN.md`、交付要求和模板 |

项目 README 和总报告已更新为最新诊断状态；六个封存结果目录的报告和摘要保持原样。早期计划中的 G2b–G4、独立输入确认或训练验证均未完成。

## 归档与上传边界

本机已存在三个小型 ZIP：G1a `results/gpu_codec_v1/gpu_codec_v1_20261101_002.zip`、完整 G1 `results/gpu_codec_v1/gpu_codec_v1_20261104_001.zip`、G2a `results/logical_ring_v1/logical_ring_v1_20261104_001.zip`。ZIP 不自动等于包含全部外部梯度、环境和编译依赖。

适合 GitHub 的内容是项目汇总、源码、方案、精简结果和身份清单。大型 `.pt`、逐包 `.npz`、编译缓存和工具链可保留本机，但必须在数据索引中列明位置、依赖关系和 SHA256 清单入口。保持“下载代码可阅读和核对摘要”与“取得全部外部数据后可复算完整结果”的区别。

本项目可支持 CUDA 编码实现、通信字节账本、协议一致性验证、配对数值诊断等工程能力的陈述。未验证 NCCL 加速、真实多卡性能、大模型训练质量提升或独立论文贡献。


## GitHub 精简归档中的数据

精简归档实际包含的文件及原始位置由根目录 `ARCHIVE_MANIFEST.json` 逐项列出；六轮结果的完整本机文件目录、大小和是否收录，由 `LOCAL_DATA_CATALOG.json` 列出。`local_only` 文件没有上传，其本机路径不是公共下载链接。历史报告内指向这些文件的相对链接仅在完整本机归档中可用。

最新诊断的 `metrics_raw.jsonl` 在仓库内无损压缩为 `metrics_raw.jsonl.gz`；逐项数据不改变，原始文件 SHA256 保存在归档清单。仓库的 `tools/verify_epsilon_summary.py` 支持直接读取该 gzip 文件，只复算记录层面的统计，不调用 GPU、不读取原始张量。
