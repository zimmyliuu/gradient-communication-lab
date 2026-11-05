# 固定尺度加数的单因素诊断方案

版本：1.0。日期：2026-11-04。执行对象：用户或后续实施助手。本次仅编写方案，尚未启动诊断。

**唯一问题：官方 b8 编码原始尺度中的固定加数 1e-7，是否实质造成了 update100 上相对 uniform 的误差劣势及其集中区域？** 只比较原加数与零加数，使用既有数据和完全相同的随机池。不扫描其他加数，不做输入缩放网格，不增加训练、模型或方法族。

依据：[上一轮交付](../../results/bounded_discovery_v1/bounded_discovery_v1_20261104_001/REPORT.md)、[独立复核](../../reviews/bounded_discovery_v1_20261104_001_review.md)及[保存输出的块分解](../../reviews/bounded_discovery_v1_20261104_001_audit/saved_block_analysis.json)。配套：[交付要求](DELIVERY_FORMAT.md)、[报告模板](REPORT_TEMPLATE.md)、[摘要模板](summary.template.json)。

## 1 当前证据和假说

上一轮 uniform_g128_q8 比官方 b8 少用 3.66% 字节；其局部 SSE 在 update100 低 21.85%，在 update500/2500 高 16.83%/8.92%。各快照四个 worker 方向一致。update100 中，输入能量最高的 32 个共同坐标块贡献了约 94.4% 的两方法净 SSE 差。现象可信，原因尚未确定。

实际执行的官方 BF16 MEE optimized compress 路径使用：

```text
raw_scale = positive_scale(block_absmax / positive_scale(chunk_max_mee) + scale_add)
scale_add = 1e-7f
```

固定加数可能改变相对尺度，以及随后存储尺度、UINT8 ceil 和主码舍入的结果。本轮验证它在这些输入上的作用，不预称它为 bug，也不预设去掉它一定更好。

必须区分两个参数：本轮改变的是 scale_add=1e-7；官方码表的 MEE 参数 epses[8]=0.1 完全不改。旧本地 G1 的 0.08、BF16 LUT 及 5/9 位契约均不用于本轮。

## 2 固定范围和投入上限

| 项目 | 固定要求 |
|---|---|
| 输入 | 旧 seed1101 的 update100/500/2500，各四个 worker，逻辑 D=818048 |
| 输入入口 | 原 input_logical，已 BF16 预除4；不再预除，不复用旧 padding |
| 编码 | 官方 b8、BF16、strategy=3、chunk=16；P=818176 |
| 随机状态 | 原官方 b8 的 96 个 seed/pool；每输入8次，逐项核对旧 pool SHA256 |
| 两个条件 | rebuilt_control_add1e7；diagnostic_b8_add0 |
| 解码 | 两条件的包均由封存、未修改的原官方 decoder 解码 |
| uniform | 直接复用旧 g128q8 的96份输出，不重新编码或修改参数 |
| 正式规模 | 96个配对键；控制条件96次编码，干预条件96次编码；另有少量资格检查 |
| 聚合 | 复用解码结果做四 worker 的 FP64 one-shot sum，不调用 ring |

量级预查主动工作不超过30分钟；实施、构建、执行和解释合计不超过半个工作日，新增 GPU 执行累计不超过30分钟。最多两份条件构建，再允许一轮有明确原因的同语义构建修复；不新增第三种加数。

沿用上一轮已成功的 Python/PyTorch、CUDA11.8、MSVC、UTF-8、隔离 cuRAND 头文件和 --use_fast_math。不得升级共享环境或覆盖原模块。超出范围交付 blocked/inconclusive，不扩展平台。

## 3 E0 来源和量级预查

读取旧 input_manifest、packet_accounting、合同、构建日志和实际源码，核对哈希。确认 BF16 + strategy3 + chunk16 进入 scaling_compress_kernel_mee_hierarchical_bf16_opt<8>；不要误改普通、FP16、AEE 或 fused decompress-accumulate-recompress 路径。

尺度常数 C8=chunk_max_mee 来自 cuda_hadamard.cu 的 build_lookup_table_host：host float 逐次乘法构表，各代表值转 FP16，取 lut[8].table[254]，launcher 再转 float。保存相关源码与常数来源。

可低成本导出实际 host 常数时记录数值和 FP16 位模式；否则用显式 FP32 仿真作为 estimated 常数并标明误差边界未知。官方功能测试 mee_params 使用 Python double 递推，也不能自动当成严格常数 oracle。不要为预查另建完整浮点模拟器；常数估计的不确定性不阻止下一阶段直接使用真实内核对照。

对12份输入的每个16元素组，统计 max_abs、输入能量，以及估计的加数占比：

```text
base_scale = max_abs / C8
add_fraction = 1e-7 / (base_scale + 1e-7)
```

零组单列，避免大量零能量组扭曲统计。报告普通分位数和按输入能量加权的统计。解析旧包的 UINT8 group scale 和 BF16 super-scale unit，作为实际已存尺度参照；它们不等于未量化的 raw_scale。

按已有块分解规则冻结每个快照的高能量区域 H：四个 worker 对相同256坐标范围的输入能量相加，取能量最高的32块，tie 按 block_id。计算误差时仍裁掉 padding。保存块 ID，后续不能根据干预结果重选。

**量级小不构成无影响证明。** 后续存在离散阈值，CPU 运算也不等于 CUDA fast math。E0只帮助判断预期和解释结果，不以add_fraction小于任意阈值作科学早停，也不扩展为算术不变性证明工程。来源和路径资格通过后统一进入固定96项对照；来源或路径无法确定时因资格不足停止。

## 4 E1 构建干预和资格验证

### 4.1 只改变一个实际因素

从上轮实际编译源码复制到新目录。目标是 cuda_hierarchical_mee.cu 中 optimized BF16 MEE compress 模板的原始尺度计算，冻结源中位于第303行。通过函数名、原始语句和源码哈希共同定位，不能仅依赖行号。

允许用编译常量 DIAG_B8_SCALE_ADD，仅 NBITS==8 时使用它；其他位宽仍使用1e-7f。两个构建分别取1e-7f和0.0f，其余源码、工具链和编译选项一致。不要对整个文件全局替换1e-7。

以下全部保留：positive_scale 的1e-30下限、非有限值处理、码表及其生成、BF16 super-scale存储、UINT8 ceil/clamp、主码限界、随机池及随机舍入、最终 BF16 输出。零加数仍然保留这些保护，不是“去除所有尺度保护”。两份新构建均单独记录源码diff、flags、二进制哈希及派生身份，干预不能标为未修改官方算法。

旧 build_official.py 写死了 RUN、UPSTREAM 等路径，不能原样执行后写回旧 run。新入口必须显式接收新输出目录，并检查其不在封存目录内。

### 4.2 防止模块或随机状态混淆

扩展名 eden_utils 和 wrapper 的模块缓存会影响加载。控制、干预、原 decoder 分别在独立 Python 进程中加载显式 build manifest，记录 PID、实际模块路径和SHA256。不要在同一进程替换 sys.modules 来切换条件。两个 encoder 产出包后，再由只加载原官方模块的 decoder 进程解码。

从旧 packet_accounting.jsonl 读取 official_dynamiq_fixed_codec_b8 的 seed，按旧 CPU FP32 rand → BF16 → CUDA 生成完整 P 长度随机池；每个池的原始字节哈希必须与旧记录一致。新 method_id 不参与 seed 派生。两个条件使用同一配对键的同一池，不裁掉或修正 BF16 随机数中的1。

### 4.3 正确性与停止条件

先检查全零、非零常数/正负混合、稀疏输入、一个257长度尾部输入，以及一个真实输入。干预只需满足其声明的包和数值域，不要求与原输出相等。增加一个小型 b4 非零夹具作为未改位宽的对照，不扩展 b4 矩阵。

正式比较前，rebuilt control 必须在全部96个旧 b8 配对键上与封存包逐字节一致，并由原 decoder 得到逐位相同 BF16 输出。这96次编码就是正式控制组，不重复生成另一套控制数据。不能用官方自测的0.08容差替代身份检查；不一致时停止因果解释，先报告构建或入口差异。

干预包应仍为875704 B/worker，独立解码，输入和随机池不被修改。检查未使用码值255、尺度合法性、输出非有限值及尾部行为。统计边界码0/254的比例，但称 boundary_code_fraction；码落在端点不等于真实发生饱和，不能把它当 clamp 次数。

若出现非法包、非有限输出或反事实不满足数值域要求，保存具体输入/池/包，判 diagnostic_domain_failure；不要丢弃坏样本后报告改善，也不要另改尺度下限或限界规则挽救这个变体。

## 5 E2 配对执行和指标

主指标是局部 SSE，全部以 FP64 在逻辑坐标上计算。对同一快照t、重复k，令 S_A 为四worker控制组 SSE 总和，S_B 为零加数组总和，S_U 为旧 uniform 总和。

```text
paired_delta = S_A - S_B
relative_reduction = (S_A - S_B) / S_A
gap_before = S_A - S_U
gap_after = S_B - S_U
```

正 paired_delta 表示去掉加数降低误差。报告逐worker及快照的mean/std/min/max、改善重复数、四worker方向；先算配对差再汇总，不能用两个独立标准差替代配对差的波动。

对 update100 的预先选定问题，定义覆盖比例：

```text
gap_closure = mean(S_A - S_B) / mean(S_A - S_U)
```

只在分母为正时解释；不裁剪到[0,1]。负值表示差距扩大，超过1表示误差劣势被反转。对固定区域H计算同样的 gap_closure_H。其他两个快照直接报告 gap_before/gap_after，不套用负分母的“解释比例”。同时保留 source_energy 和归一化误差，避免只看绝对尺度。

两条件同字节，因此 paired_delta 是同包预算的实现干预效应。uniform 仍少用字节；B的误差低于uniform，只表示原误差劣势消除，不表示B同时在字节和误差上支配uniform。

补充记录包、主码、UINT8尺度、BF16 super-scale、BF16输出的变化比例和变化位置；只从包中可见的内容做判断。按冻结H和其余区域分解paired_delta，不重新选择最有利块。E0估计量与结果的关联只作描述，不能单凭相关性解释因果链。

one-shot继续由同一快照、同一重复的四份输出做FP64求和，使用旧参照 Σx_r，记录聚合SSE及配对差。沿用 source_energy、target_energy、cross_term 口径，不新增量化或多跳实验。

控制与干预之间现在确实使用 matched randomness；uniform仍是原来的另一组随机流，不能声称三方法共同随机数配对。8次重复只反映固定开发输入上的随机性，不是8条独立训练轨迹。

## 6 如何作出结论

结果与后续投入分别判断，不要求输出正面发现：

| outcome | 含义 |
|---|---|
| no_effect_observed | 全部正式配对输出逐位相同，SSE完全相同；仅说明当前输入/状态未观察到效应 |
| changes_without_closing_gap | 输出改变，但update100的mean gap_closure≤0，含净差距不变或扩大 |
| partially_closes_gap | update100的mean gap_closure在0与1之间；不把微小改善称主要解释 |
| closes_or_reverses_gap | update100的mean gap_closure≥1；同时报告worker/重复稳定性和其他快照回退 |
| diagnostic_domain_failure | 原基线有效，但干预出现非法包/尺度或非有限输出等域失败 |
| inconclusive | 控制不匹配、构建阻碍、干预超范围、缺失记录或其他不可判读条件 |

按唯一顺序分类：先判断资格不足或域失败；正式结果完整有效时，先检查是否全部输出相同，否则按update100的mean gap_closure≤0、介于0与1、至少1依次归类。无法定义闭合比例时为inconclusive。worker/重复方向混合及其他快照回退单独报告并进入continuation_gate，不改变按均值定义的outcome。

冻结一个仅用于限制后续投入的门槛：只有 update100 的 gap_closure 和 gap_closure_H 均至少0.5、至少3/4 worker均值改善、至少7/8次快照配对改善，且另外两快照均值SSE不增加超过2%，才标 continuation_gate=met。50%表达“至少解释当前差距的一半才值得追加投入”；2%是本轮允许的开发集回退上限。这些不是统计显著性、普遍正确性或科研新颖性的标准，不能看完结果后修改。

未过门槛时结束当前假说：如实报告局部作用或回退，不换加数、不换数据、不扫描新因素。门槛满足也只交回一份因果证据卡，说明改了什么、哪些区域变化、代价与适用范围；不自动启动D2、训练或新方法。对固定输入干预有效，仍可能只是常规实现修复，独立贡献需要另行判断。

## 7 交付和范围结束

新目录使用 results/epsilon_diagnostic_v1/<唯一run_id>/。原 bounded_discovery_v1、旧审计输出及其源码保持只读。控制结果可引用旧包和输出，不复制整个旧交付；新反事实包、解码结果和必要反例保留以便复算。

最少回传 REPORT.md、summary.json、结果目录或ZIP。报告必须明确：假说是否支持、解释了多少差距、是否发生回退/域失败、是否满足投入门槛。交付后本轮结束，按[结果格式](DELIVERY_FORMAT.md)接受复核。
