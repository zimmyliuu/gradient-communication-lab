# GPU 编解码结果格式与回传要求

本格式对应 [技术方案](TECHNICAL_PLAN.md) 的 G0/G1。所有结果文件为 UTF-8；结构化记录使用严格 JSON，不使用 NaN、Infinity 或注释。未测量的值为 `null`，不能填 0。路径在交付包内使用相对路径；引用本机历史数据时另存绝对路径、文件哈希及重建方式。

## 1 交付目录

每次完整尝试使用唯一 `run_id`，例如 `gpu_codec_v1_20261103_001`。示例日期不是计划完成期限。不要覆盖旧尝试，也不要把失败记录从最终包中删除。

```text
results/gpu_codec_v1/<run_id>/
  REPORT.md
  summary.json
  environment.json
  codec_contract.json
  benchmark_config.json
  input_manifest.json
  source_manifest.json
  correctness.jsonl
  benchmark_raw.jsonl
  benchmark_summary.jsonl
  memory.jsonl
  issues.jsonl
  REPRODUCE.md
  requirements.txt
  code/                      # 最小可运行源码、测试、配置；或同等可取得的源码归档
  fixtures_manifest.json     # 大夹具位置、哈希与重建命令
  traces/                    # 至少一个代表性 trace，可留本地并在清单引用
  figures/                   # 主报告引用的曲线
  logs/                      # 环境预检、编译、测试和运行日志
```

环境阻碍时允许只提交已有文件；`summary.json` 要明确哪些文件未产生以及原因。不要用空记录文件冒充执行通过。执行成功且没有问题时，`issues.jsonl` 可以为空，并在报告中注明。

`handoff_templates/` 提供可填写模板，不是实验输出。实施时复制到新 run 目录，不修改模板本身。复制并开始实际执行记录时，将 JSON 的 `template_only` 改为 `false`；G0 阻碍报告也属于实际记录。性能/正确性 JSONL 由程序生成，不手填测量结果。

## 2 总摘要与状态

按 [summary.json 模板](handoff_templates/summary.json) 填写。关键状态分开判断：

| 字段 | 允许值 | 含义 |
|---|---|---|
| `execution_status` | `not_started/running/complete/incomplete/blocked/failed` | 本次执行情况 |
| `correctness_status` | `not_run/pass/fail/incomplete` | 要求的数值与码流测试 |
| `benchmark_validity` | `not_run/valid/invalid/incomplete` | 覆盖范围、计时方法和原始数据是否合格 |
| `optimization_gain` | `not_evaluated/observed/not_observed/mixed/inconclusive` | 相对相同语义 GPU PyTorch 基线的实测表现 |
| `g1_acceptance` | `not_ready/pass/fail/incomplete` | 是否满足本轮完整交付要求 |
| `next_action` | `fix_environment/fix_correctness/complete_measurement/optimize_codec/prepare_g2/review_required` | 执行者建议，接收方复核后再推进 |

`pass` 要求两种位宽、正确性矩阵、GPU 两种实现、规定性能网格与交付文件齐全。`optimization_gain` 不决定验收能否通过。实测“无收益”可以与 `g1_acceptance=pass` 同时成立。

`observed` 或 `mixed` 必须引用具体配对 case、session 的延迟比和离散程度，不能只写总体百分比。这里不预设显著性检验，也不把很多计时块当作很多独立训练重复。

以下字段在 G1 固定为 `false`：`full_dynamiq_gpu_validated`、`training_quality_newly_validated`、`multigpu_speedup_measured`。旧训练结果可以引用，但不算本轮新完成。

## 3 冻结契约与来源

### codec_contract.json

使用 [契约模板](handoff_templates/codec_contract.json)，补全参考文件哈希、LUT/阈值身份、GPU 实现、BF16 舍入语义、编译参数和输入检查模式。冻结后计算该文件字节的 SHA256，写入 `summary.json` 和每条结果的 `contract_sha256`。不要把契约自己的哈希写回契约导致循环变化。

增加全 FP32 中间运算、改变尺度、加入误差反馈、把 5/9 位改为 4/8 位等都属于新契约，不得在同一方法 ID 下混合。

### environment.json

使用 [环境模板](handoff_templates/environment.json)。缺失工具写 `null` 并说明发现路径；运行时 CUDA、编译 Toolkit 和驱动分别记录。记录 GPU 显存、compute capability、时钟/功率设置和其他进程负载；未读取的动态指标写 `null`。

### benchmark_config.json

可从 [基准配置模板](handoff_templates/benchmark_config.json) 开始。记录性能长度、输入分布及使用的种子、方法和操作列表、warmup 次数、session 数、块数、K 的选择规则、graph/compile 模式、缓存模式、检查模式和预分配/含分配模式。另存每个 `(case_id, method_id)` 冻结后的 K，跨 session 保持；不同实现不要求使用相同 K。计算该文件哈希，所有性能记录引用它。

### source_manifest.json

根对象包含 `schema_version/run_id/files`。`files` 每项至少有 `path/role/sha256/bytes`，其中 role 是 `cpu_reference/gpu_source/test/config/build_script/report_generator` 之一。Git 仓库另记 commit 与工作区是否有修改；没有仓库就用文件清单，不虚构 commit。保存实际编译日志和编译命令。

### input_manifest.json

根对象包含 `schema_version/run_id/inputs`。每个输入至少记录：

```text
input_id, source_kind, original_numel, padded_numel, dtype,
input_sha256, fixture_path, generator_or_loader, generator_config,
source_path, source_sha256, training_seed, update_step, rank,
transform_order, main_thresholds_sha256, scale_thresholds_sha256,
lut_sha256_by_bits
```

`source_kind` 为 `edge/synthetic/real_gradient/captured_codec_event`。不适用的训练信息为 null。哈希以连续张量的原始字节计算，同时记录 shape、dtype、字节序；完整夹具文件的哈希另存。完整可复现数据至少可在本机取得，不能只提供一个不存在的路径。

## 4 正确性记录

`correctness.jsonl` 每行对应一个输入、位宽、实现与检查。共同身份字段：

```text
run_id, contract_sha256, case_id, input_id, method_id, reference_id,
check_type, nominal_bits, stored_bits, original_numel, padded_numel,
status, failure_reason, counterexample_path
```

`check_type` 为 `pack/decode/encode/roundtrip/invalid_input/repeatability/input_immutability`；`status` 为 `pass/fail/skipped`。必测项 skipped 时整体为 incomplete，除非该 case 在契约中明确不适用，例如空输入的原生 encode。

各检查相应输出以下字段，不适用时 null：

| 字段 | 定义 |
|---|---|
| `code_mismatch_count` | 相同输入与阈值下整数码差异数 |
| `scale_code_mismatch_count` | uint8 尺度码差异数 |
| `top_scale_bit_mismatch_count` | 顶层尺度 BF16 位模式差异数 |
| `packet_byte_mismatch_count` | 完整紧凑包不同字节数；长度不同时另报长度失败 |
| `decoded_bit_mismatch_count` | BF16 输出位模式差异数 |
| `expected_wire_bytes/actual_wire_bytes` | CPU 参照和 GPU 紧凑格式字节数 |
| `quant_squared_error/quant_mse` | 解码相对原始输入的量化失真，FP64 累计，排除 padding |
| `quant_input_energy/quant_normalized_squared_error` | 原始输入能量及归一化失真 |
| `quant_max_abs_error` | 量化绝对误差最大值 |
| `undefined_metric_reason` | 零长度/零能量等导致 null 的解释 |

量化失真不等于 CPU/GPU 不一致。完整码流不同，即便 `quant_mse` 相近，仍不能通过逐位复现。异常输入测试通过的含义是检测并拒绝预期错误，不能把它当作有效输入失败后忽略。

## 5 原始性能记录

`benchmark_raw.jsonl` **每行对应一个计时块**，必需字段如下：

```text
run_id, contract_sha256, benchmark_config_sha256,
case_id, input_id, method_id, nominal_bits, stored_bits,
original_numel, padded_numel, input_dtype, output_dtype,
operation, allocation_mode, validation_mode,
graph_mode, compile_mode, cache_mode, rng_work_included,
session_id, block_id, inner_repeats,
gpu_elapsed_ms, wall_elapsed_ms,
payload_bytes, scale_bytes, header_bytes, alignment_bytes, wire_bytes,
record_status, exclusion_reason
```

字段约定：

- `method_id` 推荐为 `cpu_ref_repaired`、`torch_cuda_repaired`、`cuda_optimized_repaired`；位宽单独存储。
- `operation` 主项为 `encode/decode/roundtrip`；其他分项单列。
- `allocation_mode` 为 `preallocated/api_allocating`，不能合并统计。
- `validation_mode` 为 `trusted_inputs/checked_api`。主内核基准使用已验收的可信输入；带运行时检查的 API 时间另外记录。
- `graph_mode=off`、`compile_mode=eager`、`cache_mode=steady_state_reuse` 为默认条件；这里 eager 指调用方式，不代表 CUDA/C++ 未编译。
- G1 显式阈值模式的 `rng_work_included=false`；LUT/阈值准备耗时在环境或配置的初始化成本中单列。
- `gpu_elapsed_ms` 与 `wall_elapsed_ms` 都是 K 次操作的总时间，不是单次时间。CPU 参考的 GPU 时间为 null。
- `wire_bytes = payload_bytes + scale_bytes + header_bytes + alignment_bytes`。payload 和 scale 已包含补齐元素，不能再加一份 padding 字节。
- decode/roundtrip 的 wire_bytes 指一个压缩包的大小，不代表该操作发生了网络传输。
- `record_status=ok/diagnostic_invalid/failed`。无效或失败记录保留，不进入有效性能汇总，解释依据。不得按慢快选择性排除。

`case_id` 应由输入、位宽、操作及计时模式确定，**不包含 method_id**，以便不同实现配对；联合键为 `case_id + method_id + session_id + block_id`。

## 6 汇总性能和显存

`benchmark_summary.jsonl` 从 raw 自动生成。每行对应一个 `case_id + method_id`，包含：

```text
run_id, case_id, method_id, contract_sha256, benchmark_config_sha256,
valid_session_count, valid_block_count, session_statistics,
gpu_session_median_us_median, gpu_session_median_us_min, gpu_session_median_us_max,
wall_session_median_us_median,
original_input_bytes, wire_bytes, effective_bits_per_logical_element,
input_equivalent_gbps, paired_reference_method, paired_speedup_by_session,
noise_flag, missing_coverage, interpretation
```

`session_statistics` 每项记录 session ID、块数、单次 GPU 延迟的 median/p25/p75/p95、墙钟 median；所有单次值由 block elapsed/K 得到。p95 的完整名称为块平均延迟 p95。默认分位数使用线性插值并记录实现。

`input_equivalent_gbps = original_input_bytes / (gpu_session_median_us_median * 1e-6) / 1e9`。原始字节为逻辑元素数乘 BF16 的 2 B，padding 不放入吞吐分子。

`paired_speedup_by_session` 的值为同 session 下 PyTorch GPU 中位延迟/优化 GPU 中位延迟。分母为零或未测时为 null，不做无限大收益报告。每个 session 的 IQR/median 超过 10% 时标 `noise_flag=true` 并调查；这只是数据质量提示，不是删数据条件。

`memory.jsonl` 独立测量，不把显存重置与查询放入性能计时块。每行至少包含：

```text
run_id, case_id, method_id, operation, allocation_mode,
input_bytes, threshold_bytes, lut_bytes, packet_buffer_bytes,
decoded_output_bytes, workspace_bytes,
allocated_before_bytes, allocated_after_bytes, peak_allocated_bytes,
reserved_before_bytes, peak_reserved_bytes,
peak_extra_allocated_bytes, external_allocation_tracking, notes
```

`peak_extra_allocated_bytes=peak_allocated_bytes-allocated_before_bytes`，并说明基线是否已包含预分配缓冲。为零不代表总工作内存为零。外部 CUDA 分配未纳入统计时明确说明，不能假装全覆盖。

## 7 报告与图表

按 [REPORT 模板](handoff_templates/REPORT.md) 回答以下问题：实现对象是什么；哪些语义通过；数据格式到底占多少字节；什么大小的桶上优化有效或无效；当前主要开销在哪里；下一步应修复什么。

至少提供三组可由数据重建的图表：

1. 桶大小与 encode/decode/roundtrip 延迟，分开显示位宽和实现；用 session 间范围或其他明确说明的误差条。
2. 桶大小与实际每元素位数、量化失真；两者分图，避免暗示减少字节等于加速。
3. 桶大小与显存组成或峰值增量。

不规定结果必须好看、必须快于基线或必须支持继续研究。图表的数据路径、生成脚本和命令必须保留。

`issues.jsonl` 每项记录 `issue_id/category/severity/status/description/evidence_paths/reproduction/fix_or_next_step`。记录工具链阻碍、数值反例、基准噪声、缺失输入等；不得只在聊天里口头说明。

## 8 复现和回传

`REPRODUCE.md` 写实际使用过的命令、工作目录、解释器和预期输出位置。至少分别说明环境检查、夹具生成、正确性检查、性能测量、汇总作图；标出尚未实现的入口。不能写一组未经执行的命令并声称已复现。

完整归档留在本机。回传小包优先包含主报告、摘要、所有小型 JSON/JSONL、环境、配置、源码、测试与复现说明；大 checkpoint、虚拟环境、编译缓存不打入小包。大 trace 和夹具提供本机路径与 SHA256。若跨机器接收方无法取得历史文件，必须能重建夹具或按需补传。

回传消息可直接使用以下格式：

```text
已完成阶段：G0 / G1（或实际完成到的部分）
主报告：<REPORT.md 的绝对路径或附件>
摘要：<summary.json 的绝对路径或附件>
结果目录或压缩包：<绝对路径或附件>
正确性：pass / fail / incomplete
性能测量：valid / invalid / incomplete
最重要的发现：<一句话>
未解决的问题：<一句话；没有则写无>
```

接收后先核查状态与证据，再讨论下一阶段；不需要先把失败包装成成功。
