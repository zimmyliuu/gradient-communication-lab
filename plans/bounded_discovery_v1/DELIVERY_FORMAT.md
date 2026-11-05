# 下一轮诊断结果格式

本格式仅对应[本轮任务](PLAN.md)。沿用项目 UTF-8、严格 JSON、失败保留和唯一 run_id 的规则；不继承 G1 的完整性能矩阵、GPU trace 和所有附加测试要求。所有未测量值写 null，状态写 not_run，不填 0 或 pass。

## 1 目录和阶段交付

建议 run_id 为 bounded_discovery_v1_YYYYMMDD_NNN，日期以实际执行日为准。

```text
results/bounded_discovery_v1/<run_id>/
  REPORT.md
  summary.json
  baseline_qualification.json
  experiment_contract.json
  input_manifest.json
  environment.json
  source_manifest.json
  correctness.jsonl
  packet_accounting.jsonl
  metrics_raw.jsonl
  metrics_summary.jsonl
  issues.jsonl
  REPRODUCE.md
  integrity_manifest.json
  code/
  logs/
  figures/
```

D0 阻碍时只交实际产生的材料：报告、摘要、资格核验、环境/来源、日志、问题记录和复现说明；没有运行的指标文件在摘要列为 not_produced，不创建空文件冒充完成。

线索定位完成时追加 findings/FINDING.md、diagnostic_config.json、diagnostic_raw.jsonl 及最小反例输入清单。没有线索不要求这些文件。计时仅在计划规定的条件成立时追加 benchmark_config.json、benchmark_raw.jsonl、benchmark_summary.jsonl。

## 2 冻结和身份

D0 可先做环境及小型自测。D1 正式比较前冻结 experiment_contract.json，包含以下内容，计算 SHA256；原始记录引用该哈希。冻结后变更生成新 revision 和理由，保留受影响的旧记录，不能边看误差边修改预算配置。

```text
schema_version, run_id, revision, frozen_at, scope,
upstream_repository, upstream_commit, source_manifest_sha256,
official_entrypoints, official_configurations, wrapper_changes,
methods, comparison_budget_rule, selected_uniform_configurations,
input_manifest_sha256, logical_input_dtype, decoded_output_dtype,
padding_by_method, rng_by_method, repeats, seed_derivation,
aggregation_definition, metric_definitions, stage_budgets,
optional_timing_enabled, stop_rules
```

methods 每项须有 method_id、family、implementation_origin、nominal_parameter、stored_code_bits、group_size、scale_dtype、arithmetic_profile、output_conversion、packet_format、shared_state、rng_mode、timing_eligible。未知字段保留 null 并注明原因。

baseline_qualification.json 至少包括各准入问题、pass/fail/unknown、依据路径、构建尝试、算术变更、独立解码检查、局部及聚合比较资格。给出 source_identity_status、local_codec_status、packet_status、rng_status、aggregate_comparability_status 及 gate_decision；不能只写一个总 pass。

source_manifest.json 记录原始官方代码、实际 wrapper、uniform、指标生成程序和所加载二进制的路径/字节数/SHA256；二进制未构建则说明。若代码不在 Git，用文件哈希，不虚构 commit。

input_manifest.json 区分 snapshot_id、worker_id、input_id；记录来源文件/张量哈希、training_seed、update_step、partition=development、logical_numel、input_dtype、transform_order、already_divided_by_world_size=true、各方法 padding 和加载方式。附参数映射时记录其来源，不改变原始扁平顺序。

## 3 正确性和包账本

correctness.jsonl 每行至少包含：

```text
run_id, contract_sha256, check_id, method_id, input_id,
check_type, status, reference_id, difference_measure,
expected_behavior, observed_behavior, evidence_path, failure_reason
```

check_type 覆盖本轮实际接口需要的 pack/unpack、独立 decode、repeatability、input_immutability、tail_handling、zero_case 和 reference_agreement。不同方法之间不要求码流一致；CPU/GPU 逐位一致只在其契约本来要求相同算术时使用。保留 failed/skipped，解释为何不影响或阻止后续比较。

packet_accounting.jsonl 每行是一份实际独立解码包，至少包含：

```text
run_id, contract_sha256, packet_id, snapshot_id, input_id, worker_id,
method_id, repeat_id, logical_numel, padded_numel,
payload_bytes, scale_bytes, header_bytes, alignment_bytes,
packet_total_bytes, effective_bits_per_logical_element,
packet_sha256, component_shapes_dtypes,
per_input_auxiliary_bytes, setup_or_persistent_state_bytes,
independent_decode_pass, byte_measurement_method
```

packet_total_bytes 是实际序列化组成之和，包含必须的逐输入辅助信息；setup_or_persistent_state_bytes 另列并说明生命周期。若 auxiliary 已归入 header 等组成，不再重复加算。padding 已在 payload/scales 中，不能再加一份。

effective_bits_per_logical_element=8×packet_total_bytes/logical_numel。按张量原始存储取字节，不能按 torch.save 文件大小计算；文件序列化外壳不是通信包。至少保存可重建的包/随机状态清单与哈希，无法重建时保留所需原始包。

## 4 原始指标和汇总

metrics_raw.jsonl 的 record_type 为 local 或 one_shot_codec_sum。共同键为 record_type + snapshot_id + worker_id + method_id + repeat_id；聚合记录 worker_id=null。所有身份还包含 run_id、contract_sha256、input_manifest_sha256、input_id 或 input_ids、rng_state_id、status、failure_reason。

局部字段：

```text
logical_numel, squared_error, mse, input_energy,
normalized_squared_error, max_abs_error, nonfinite_count,
packet_id, packet_total_bytes, undefined_metric_reason
```

聚合字段：

```text
worker_count, logical_numel, aggregate_squared_error, aggregate_mse,
source_energy, target_energy, source_normalized_error,
target_relative_error, cancellation_ratio, near_cancellation,
local_error_sum, cross_term, aggregate_max_abs_error,
one_shot_packet_bytes_sum, packet_ids,
aggregate_comparability_status, undefined_metric_reason
```

全部误差和能量用 FP64 累计，裁掉 padding。聚合引用同一方法、同一快照和重复的四份包；保证 worker 完整且不重复。deterministic 方法 repeat_id=0，注明 deterministic，不伪造 8 份独立结果。BF16 identity 的局部和 one-shot 量化 SSE 应为零，作为指标实现检查。

metrics_summary.jsonl 由程序从 raw 生成。按输入或快照和方法给出有效/失败重复数、均值、sample std、min/max、实际字节和缺失项；每个输出标明统计对象。确定性配置 std=null。若只有部分随机重复有效，展示失败并标 incomplete，不用筛掉的样本得出优势结论。

比较表必须同时显示实际字节和误差；对于 uniform 上邻明确标 higher_budget。择优点标 development_selected，不把开发集择优胜出当作留出结果。没有配对随机变量就不用 matched_randomness 名称，不把所有坐标当作独立试验计算显著性。

## 5 发现卡和条件性后续任务

只有一张主要发现卡，内容包括：

1. 一句话问题；官方组件、输入范围和精度范围。
2. 支持和不支持该现象的全部开发快照、随机离散程度、绝对和归一化效应量、真实字节。
3. 已排除的实现错误和不公平因素；尚未排除的解释。
4. 最多两项单因素对照及其额外成本；oracle 结果与可部署结果分开。
5. 最接近原始工作和官方说明的具体位置，哪些属于已知取舍，可能增加什么证据或改进。
6. 一个可执行的干预设想，以及 D2 最小确认的冻结主张、输入独立性、主要指标、失败条件和预计资源。D2 标 planned_not_executed。

D0 通过且 D1 有效完成本轮适用的比较，仍没有线索时，报告用 no_actionable_signal，并说明有限范围。基线阻碍使用 baseline_blocked，未解决的无效或不完整比较使用 inconclusive，同时保留具体阶段状态；不能把缺少结果当成科学上的负结果。不能用模型太小或期刊要求高替代对本次数据的判断，也不要求必须交出正面发现。

## 6 摘要状态

从 [summary.template.json](summary.template.json)复制开始，实际执行后 template_only=false。状态采用：

| 字段 | 允许值 |
|---|---|
| execution_status | not_started / running / complete / incomplete / blocked / failed |
| d0_status | not_run / pass / blocked / fail |
| d1_status | not_run / complete / incomplete / invalid |
| correctness_status、packet_accounting_status | not_run / pass / fail / incomplete |
| local_comparison_status、aggregate_comparison_status | not_run / valid / invalid / incomplete |
| timing_status | not_run / valid / invalid / incomplete |
| decision | null / baseline_blocked / no_actionable_signal / inconclusive / candidate_for_confirmation |

预算内按规则停止并交付可核验的负结果，可以是 execution_status=complete、decision=no_actionable_signal。D0 构建阻碍用 blocked，不把它写成算法失效。完整性通过不等于研究发现成立。

## 7 复现和回传

REPRODUCE.md 只列实际执行过的命令和顺序，标明解释器、工作目录、CUDA/编译器、输入来源及输出位置。未来建议命令单独标 not_executed。提供一个从 raw 重建表图和检查身份唯一性的实际入口，验收只需复算摘要，不重跑全部历史实验。

报告内所有关键数值可追到 raw 的稳定记录键。integrity_manifest.json 覆盖交付文件，避免把自身哈希写回自身；ZIP 在封存后建立，单独保存其哈希。大官方仓库、环境、模型及历史夹具不重复打包，但保存可取得来源、哈希和必要重建方式；最小反例随包。

回传格式：

```text
已完成：D0 / D0+D1 / D0+D1+单因素定位
主报告：<REPORT.md 绝对路径>
摘要：<summary.json 绝对路径>
完整结果目录或 ZIP：<绝对路径>
官方基线资格：pass / blocked / fail
比较有效性：valid / incomplete / invalid / not_run
研究判断：baseline_blocked / no_actionable_signal / inconclusive / candidate_for_confirmation
最重要发现或停止原因：<一句话>
D2：未执行
```
