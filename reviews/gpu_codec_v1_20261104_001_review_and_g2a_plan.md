# G1 验收结论与 G2a 逻辑聚合实施要求

日期：2026-11-04，北京时间。复核对象：[本轮报告](../results/gpu_codec_v1/gpu_codec_v1_20261104_001/REPORT.md)、[机器摘要](../results/gpu_codec_v1/gpu_codec_v1_20261104_001/summary.json)、封存源码、原始记录与同名 ZIP。验收依据：[上一轮 G1b 要求](gpu_codec_v1_20261101_002_review_and_g1b_plan.md)。

> **执行范围修订：用户随后明确以 SPL 投稿为主要目标，需控制工作量。G1 验收保持有效；下文 G2a 是候选技术方案，暂缓启动。** 先确定论文的一项核心贡献及最小充分验证，再决定是否需要执行或裁剪 G2a；完整 G2、G3、G4 不作为默认前置条件。当前优先级见[技术方案顶部修订](../TECHNICAL_PLAN.md)。四页限制约束论文范围，不减少支持核心主张所需的证据。

**接受完整 G1：固定 b4/b8 局部 CUDA codec 的正确性、受控性能比较和交付证据满足要求。原工程路线的下一项候选是 G2a，固定 b4/b8 的单 GPU 逻辑 ring。** 本次复核重新读取源码并独立复算原始数据、文件身份和 GPU trace，没有重跑正式 GPU 矩阵；历史目录保持封存。

本项目已经完成一个有实测依据的 CUDA 工程成果。下一步的价值是验证这个组件能否正确参与多跳聚合，不需要继续以提高局部加速比作为主要目标。

## 1 G1 验收依据

| 项目 | 独立复核结果 |
|---|---|
| 正确性 | 2098 条记录：953 条 PyTorch、981 条 CUDA、164 条 CPU；身份无重复，差异计数为零 |
| 参照独立性 | GPU decoder 接收独立 CPU 包；重复结果与输入保护使用独立副本 |
| 错误处理 | 170 条预期拒绝符合固定规则：148 条 ValueError、22 条精确 host ABI RuntimeError；意外 CUDA 错误不算通过 |
| CUDA 实现 | 逐步 BF16 舍入、LUT 索引限界、唯一字节写入、当前 stream、设备保护和 allocator 生命周期记录均有实现 |
| 补充检查 | pytest 50 项通过；非默认 stream、四个 CUDA 生命周期探针、两个 canary 检查有对应记录 |
| 性能 | 20800 条 raw、208 条逐方法汇总、10400 组配对；顺序符合预注册的每 case/session 10 AB 加 10 BA |
| 统计 | 分位数、session 汇总、墙钟、配对收益和显存汇总可从原始记录复算，无差异 |
| 显存 | 1040 条记录；唯一 storage、持久 plan 和峰值增量的口径一致 |
| 身份 | 完整性清单 344 项、源码清单 68 项、正式执行前冻结的 16 个源码文件均匹配；各清单有重叠 |
| 构建 | a02 实际模块、构建源码快照和 ninja 哈希匹配；正式结果绑定该加载产物 |
| ZIP | 305 个文件与本地逐字节相同，另有 PACKAGE_CONTENTS.json；排除的 build 文件有清单 |
| GPU trace | 8 个配置均从 CUPTI raw 重建并与 Chrome 对应；无丢记录、采集错误或未归属窗口事件 |

2098 是检查与参照执行记录数，不是独立实验重复数。源码和探针支持已测路径的工程正确性，未执行 compute-sanitizer 不应改写为工具级内存安全验证通过。

## 2 性能结果的含义

以下均为本机、同一新 run、标准正态 seed 7301、N=4194304，即 8 MiB BF16 输入，预分配 trusted roundtrip。主码实际存储宽度分别为 5/9 位。

| 配置 | PyTorch 中位延迟 | CUDA 中位延迟 | 五个 session 配对比中位数 | 配对比范围 |
|---|---:|---:|---:|---:|
| 名义 b4 | 7.304440 ms | 0.200428 ms | 36.44 倍 | 34.99–37.03 倍 |
| 名义 b8 | 9.251784 ms | 0.233630 ms | 39.41 倍 | 38.53–41.17 倍 |

收益分母是本轮重测的 PyTorch 实现。配对比先按同 session 两种方法的中位数计算，再汇总五个比值；不能拿两个总体中位数的商替代这一统计量。

CUDA 版本消除了 PyTorch 基线约 542.75/702.75 MiB 的持久全局张量工作区，剩余列入 workspace 的张量为 status 和 epsilon，共 6 B。**这不等于整个 codec 只占 6 B 显存**：输入、阈值、LUT、包和输出仍占空间，b4/b8 列示 device 张量总计分别为 28,606,536/30,704,648 B；寄存器、共享内存和驱动资源另有口径。

小输入 N=65536 与大输入 N=4194304、两种位宽的实测 trace 均显示，每次 roundtrip 从 52 个 kernel 降为 2 个。PyTorch 每份三次调用窗口有 6 次 DtoD；CUDA 窗口无复制活动。正式延迟来自关闭 profiler 的配对测量，trace 解释实际调用与复制变化，没有测 DRAM 事务量。

噪声仍存在：PyTorch 76/104、CUDA 87/104 case 有会话内噪声标记，437/1040 case-session 的 IQR/median 超过 10%；9/20800 块超过 50 ms，全部保留。五个 session 的配对差距支持相对这个基线的局部优化收益，无需为让图更平滑而重测整个 G1。

可用于履历的表述：

> 在 RTX 4000 Ada 上实现保持 CPU 逐位语义的 BF16 梯度低比特 CUDA 编解码，采用真实 5/9 位紧凑码流；通过协议与真实梯度回放检查，并在 8 MiB 输入的配对基准中较 PyTorch eager 基线取得约 36/39 倍 roundtrip 加速，将已剖析调用的内核数从 52 降至 2。

该结果证明实现优化；尚不能写成先进 codec 对比优势、AllReduce 加速或大模型训练加速。

## 3 下一阶段的范围

[原技术方案](../TECHNICAL_PLAN.md)中的完整 G2 涉及逻辑 ring、随机状态、均值与能量同步、动态位宽和训练更新。将它拆为两个明确子阶段：

| 子阶段 | 范围 | 本次建议 |
|---|---|---|
| G2a | 固定 b4/b8，显式阈值，逻辑 ReduceScatter 与 AllGather，逐消息与逐 rank 核验 | 下一轮仅执行这一项 |
| G2b | b2 的 3 位主码、原随机池与 LCG、预处理、统计同步、动态控制、完整更新语义 | G2a 交付后再定实施细节 |

G2a 是完整 G2 的子阶段，通过后仍标记 `full_g2_acceptance=incomplete`。它采用既有 ring 的路由和量化位置，作为独立的固定配置验证协议，不冒充原完整动态算法。原算法 `target_bits=4/8` 仍可能产生 b2 流；不能仅更改标签就宣布完成移植。

G2a 不引入新量化算法、误差反馈或随机耦合候选，不要求新训练和网络环境。继续使用已验收的工具链与 codec；将 GPU 数据缓冲区作为逻辑消息载体，在单卡上顺序模拟 worker。

## 4 G2a 实施契约

### 4.1 输入与 CPU 参照

新增独立的固定配置适配器。核验并引用原项目 [DynamiQAggregator 的 codec 与 stream ring](../../spl_model_communication/src/splcomm/dynamiq.py)，保存本轮实际引用文件的 SHA256；不修改旧项目源码或结果。CPU oracle 使用原 encode/decode，可复用已验收的 [CpuReference](../results/gpu_codec_v1/gpu_codec_v1_20261104_001/code/src/gradient_comm_lab/cpu_reference.py)显式阈值适配器，避免为局部 oracle 构造原随机池。GPU 实现不能同时充当自身的期望结果。

固定两个输入入口，并在清单中区分：

- `raw_fp32`：每个 worker 的原始 FP32 梯度；先转 BF16，再以 BF16 除以 worker 数 N。各自的舍入和顺序保持一致。
- `predivided_bf16`：已转 BF16 且已除 N 的输入；核心 ring 直接求和，不能再次除 N。

核心 ring 只接受同长度的一维 BF16 输入，不做均值移除、能量统计或动态分流。非空流逻辑长度 D 补到 `P=N*1024*ceil(D/(N*1024))`，每个 chunk 长 `L=P/N`；最终裁回 D。N=1 在入口直接返回原输入，D=0 返回空输出，两者零消息、零量化。N=1 的 raw FP32 入口不应额外降精度。

核心输出固定为 BF16；raw_fp32 外层适配器在核心完成后转 FP32，predivided_bf16 入口保留 BF16。N=1 和空输入的外层输出遵守各自入口 dtype，不调用原私有 `_stream_ring` 的 N>1 路由。非空全零输入仍执行规定量化与发送，不作为空流跳过。

真实回放必须按快照分组：已有 12 个 rank 输入是 update 100/500/2500 的 **3 组四 worker collective**。这些 G1 夹具已经除以 4；读取逻辑长度 818048，不能把原局部补齐长度误当新的 D，也不能二次预除。若改读原 FP32 快照，则走 raw_fp32 入口，并核对得到的 BF16 输入与已有夹具一致。

### 4.2 路由与量化事件

采用右邻 `(r+1) mod N` 的 ring，显式区分轮次：

1. ReduceScatter 的 h=0，rank r 编码本地 chunk r，发送给右邻。
2. h=1 到 N-2，rank r 只消费上一轮收到的包，解码后与本地 chunk `(r-h) mod N` 做 BF16 加法，再编码发送。
3. 最终 rank r 解码收到的部分和，加到本地 chunk `(r+1) mod N`；最终和还要编码一次，并将这个最终包解码为自身输出。
4. AllGather 共 N-1 轮，只转发最终包的原始字节并在接收端解码；不能再次编码。该轮发送 chunk `(r+1-h) mod N`，接收 chunk `(r-h) mod N`。

CPU 和 GPU 都保留每个中间 BF16 和、编码输入、码流及最终输出的身份。审计模式可拷贝到 CPU 计算哈希；这些核验操作不属于 GPU 热路径。不要用一次全局求和后量化替代逐跳协议。

显式阈值按稳定事件键冻结：至少包含 case、位宽、阶段、rank、chunk 和该次编码序号。序号由协议 hop 或逻辑事件 ID 决定，不得使用依赖 host rank 遍历顺序的全局调用计数。为每次编码提供形状为 `(L/16,16)` 和 `(L/256,16)` 的 BF16 阈值，沿用 G1 的生成与 BF16 常数规则；保存种子推导和张量哈希。CPU/GPU 读取相同事件的同一阈值，AllGather 不消耗阈值。此阶段不推进原 LCG，不把这些独立夹具宣称为原相关随机池。

### 4.3 消息所有权

[CudaCodecPlan](../results/gpu_codec_v1/gpu_codec_v1_20261104_001/code/src/gradient_comm_lab/cuda_codec.py)的 `packet` 和 `output` 是可复用存储。同一 plan 下一次编码会覆盖上一包，下一次解码也会覆盖其输出；Python 对象引用不是独立消息副本。

发送队列必须拥有在接收消费完成前不会改写的 payload、scale codes 和 top scales。可使用独立槽位或 ping-pong 缓冲，保留最终 AllGather 包直到转发完成。明确列出每个缓冲区的生产、消费和可复用时刻；不得依赖 Python rank 遍历顺序偶然正确。`recordStream` 保护生命周期，不阻止调用者主动覆盖仍在使用的存储。

至少加入两包重叠在途的复用回归：生成 A 后再生成 B，A 在 B 完成后仍可正确接收；同时检查 decode 输出复用。先用单 stream 建立正确路由，不把模拟 rank 数强行等同于 CUDA stream 数。

新入口显式选择 build manifest，并记录实际加载模块身份。当前封存 loader 默认绑定旧 run；已有 replay shim 仅做过接口检查，新入口应做一次实际加载与结果 smoke check，避免新目录悄悄加载错误产物。

## 5 字节账本与验收检查

每次有向发送独立记账，记录消息 ID、来源包 ID、phase、hop、sender、receiver、chunk、逻辑与补齐长度、payload/scales 字节及包哈希。GPU 账本从实际发送缓冲区的元素数与 dtype 取得字节，不能只填预期公式。审计哈希与路由信息不是已实现网络头；若实现发送额外头部，应另计协议字节。

对一个非空固定流，N>1，chunk 包长为：

`B = L*(b+1)/8 + L/16 + 2*L/256`。

可独立核查：ReduceScatter 和 AllGather 各有 `N*(N-1)` 次发送，总发送字节 `2*N*(N-1)*B`；每个 chunk 经 N 次编码，全流共 `N*N` 次 codec 编码调用。最终 sink 编码是量化事件，但不是额外一轮发送。N=4 时应为 16 次编码、24 次发送，而不是二者相等。

必要检查：

- N=1、2、4；b4、b8；空流、全零、正负值、明显抵消、跨 chunk 尾部、合法极值和多次连续调用。
- 长度覆盖 0、1、255、256、257、1023、1024、1025，以及 `N*1024-1`、`N*1024`、`N*1024+1`，去重后冻结清单；另加一个常规长度 65536。
- 三组真实四 worker 梯度，每组两个位宽；这只有六个固定输入集体回放，不宣称统计泛化。
- CPU/GPU 每个编码包、每次解码、BF16 部分和、最终各 rank 输出逐位一致；所有 rank 的最终结果一致。
- AllGather 来源包字节不变，消息数、实际字节、量化事件和阈值消费数符合独立公式；N=1/空流各自满足零事件规则。
- 相同事件和输入重复执行一致；改变 host 端 rank 处理顺序后仍符合按轮次路由的期望，防止读取本轮尚未完成的数据。
- 输入/阈值/LUT 不被修改；不同 collective 之间不残留上次消息；非法输入明确失败，不输出可被误用的部分聚合结果。

与 FP64 平均值的误差只作量化诊断，不能要求其为零，也不能代替 CPU 协议逐位一致。对 raw_fp32 入口可给出原始 FP32 的 FP64 平均值；对只有 predivided_bf16 的夹具，仅据其 BF16 输入构造 FP64 和，并明确参照身份。

G2a 以正确性与账本为验收重点，不要求重跑完整 G1 性能矩阵。若附单 GPU 顺序模拟耗时，字段用 `single_gpu_logical_replay_*` 并说明包含内容；不能标成网络 AllReduce 延迟。真正的网络成本、计算通信重叠和调度预测留给后续阶段。

## 6 G2a 交付格式

新目录建议为 `results/logical_ring_v1/<唯一run_id>/`，不回写已验收 G1。沿用原项目交付风格，并至少提供：

| 文件 | 内容 |
|---|---|
| REPORT.md 与 summary.json | 完成范围、通过/失败、真实回放结果、未解决问题 |
| ring_contract.json | 输入阶段、预除、padding、路由、量化位置、阈值事件键、buffer 所有权、源码与模块身份 |
| input_manifest.json 与 threshold_manifest.json | case/worker 分组、逻辑长度、输入来源、种子与哈希 |
| correctness_case_manifest.json 与 correctness.jsonl | 冻结用例、逐项结果、CPU/GPU 位差异、重复/缺失身份检查 |
| message_ledger.jsonl | 每个有向发送的真实包字节、来源与接收关系 |
| quantization_events.jsonl | 每次实际编码、阈值身份及输入/输出哈希，包括 sink 编码 |
| intermediate_checks.jsonl | 逐 hop 解码与部分和比较、AllGather 不改包、rank 一致性 |
| buffer_lifetime_checks.jsonl | 在途复用与跨调用污染的检查证据 |
| code、environment、REPRODUCE、issues | 源码快照、环境、实际命令和失败反例；完整性清单与 ZIP |

摘要至少增加以下独立字段，运行前的模板使用 null，不把未知写成通过：

```json
{
  "schema_version": "logical_ring_v1.0",
  "template_only": true,
  "stage": "G2a",
  "scope": "fixed_b4_b8_explicit_thresholds_single_gpu_logical_ring",
  "execution_status": "not_started",
  "correctness_status": null,
  "message_ledger_status": null,
  "buffer_lifetime_status": null,
  "g2a_acceptance": null,
  "full_g2_acceptance": "incomplete",
  "claims": {
    "logical_ring_parity_validated": false,
    "full_dynamiq_gpu_validated": false,
    "training_quality_newly_validated": false,
    "multigpu_speedup_measured": false
  },
  "next_action": "review_required"
}
```

通过条件是覆盖清单完整、逐位一致、真实消息账本正确、在途缓冲区行为明确，且记录能复算。发现失败时保留第一个可复现反例，不靠改阈值、放宽容差或省略 sink 量化绕过问题。交付 G2a 后，再决定恢复完整动态路径的次序。
