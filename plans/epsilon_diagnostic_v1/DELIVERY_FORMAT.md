# 固定尺度加数诊断交付要求

对应[任务方案](PLAN.md)。只记录本轮实际执行，UTF-8、严格JSON，不使用NaN/Infinity。未测写null，未执行写not_run；不改写封存目录。

## 1 最小目录

```text
results/epsilon_diagnostic_v1/<run_id>/
  REPORT.md
  summary.json
  experiment_contract.json
  source_manifest.json
  environment.json
  input_randomness_manifest.json
  scale_precheck.json
  region_manifest.json
  control_parity.jsonl
  correctness.jsonl
  packet_changes.jsonl
  metrics_raw.jsonl
  paired_summary.json
  issues.jsonl
  REPRODUCE.md
  integrity_manifest.json
  code/
  build/
  packets/
  decoded/
  logs/
```

不要求性能测试或GPU trace。报告用表格即可，图最多一张。阻碍时交已有材料及失败日志，摘要列出未产生文件，不建空文件冒充完整。允许引用旧控制输出/包，必须有可读取路径及哈希；新反事实数据不可只留汇总。

## 2 实验前冻结

experiment_contract.json 固定 base_run、upstream_commit、旧合同/原模块哈希、实际kernel、两个scale_add值、源码patch、编译选项、随机复用规则、96个配对键、H区域、指标公式、工程门槛、预算和停止规则。其SHA256写入全部原始记录，不能把自己的哈希写回自身。

source_manifest.json 包含两份encoder源码/模块、原decoder、runner、包解析和统计程序。逐项记录origin、role、path、sha256、bytes。每个进程记录实际加载模块、manifest哈希、PID、入口配置；基线来源commit与修改后源码身份分开。

input_randomness_manifest.json 每项至少包含：

```text
pair_id, snapshot_id, worker_id, repeat_id, input_id,
input_path, input_file_sha256, input_tensor_sha256,
logical_numel, padded_numel, dtype, already_divided_by_four,
original_method_id, original_seed, original_pool_sha256,
regenerated_pool_sha256, original_packet_path, original_packet_sha256,
original_decoded_path, original_decoded_sha256,
uniform_packet_or_output_reference
```

pair_id与条件无关。全部96条必须来自旧official_b8键；不按新方法名派生seed。seed相同不够，还要核对实际pool字节哈希。

region_manifest.json 写各快照32个block_id、坐标起止和输入能量、选取规则、旧审计来源哈希。scale_precheck.json写C8来源/位模式/confirmed或estimated、码表参数0.1、FP32仿真方法、zero-group统计及普通/能量加权add_fraction分布。未取得真实常数时明确不能据估计证明无影响。

## 3 控制和合法性

control_parity.jsonl 每个pair一行：packet_byte_mismatch_count、decoded_bit_mismatch_count、pool_hash_match、input_hash_match、source/binary身份和status。必须覆盖96键且全部零差异；控制检查已完成的数据直接作为控制组。

correctness.jsonl 分开记录语义复现和反事实合法性，含condition_id、pair_id、check、status、details、counterexample_path。干预包与控制不同不是失败；输入被改写、包不可独立解码或非法值才是失败。域失败保留，不删坏样本算“成功子集”均值。

packet_changes.jsonl 每配对键记录：

```text
pair_id, condition_packet_paths_and_hashes, packet_bytes_A, packet_bytes_B,
changed_packet_bytes, changed_main_codes, main_code_count,
changed_group_scale_codes, group_scale_count,
changed_super_scale_units, super_scale_count,
changed_decoded_elements_logical, logical_numel,
boundary_code_fraction_A, boundary_code_fraction_B,
nonfinite_output_count_A, nonfinite_output_count_B,
reserved_code_count, invalid_scale_count, status
```

说明padding是否包含在每个计数分母中；误差和输出变化主项只看逻辑坐标。包均875704B。边界码比例不可标成真实饱和概率；未测clamp次数写null。

## 4 数值和配对汇总

metrics_raw.jsonl 的联合键为 record_type + pair_id + condition_id + region_id。record_type为local/one_shot，condition为control_add1e7/diagnostic_add0/uniform_reused，region为full/H/rest；聚合行worker_id=null，用snapshot+repeat标识。uniform复用项标记reused=true及原记录键。

每行记录snapshot、worker、repeat、区域坐标数、SSE、MSE、input/source_energy、归一化误差、max_abs、输出来源/哈希、status。one_shot另外记录target_energy、cross_term和四个来源输出身份。FP64累计，裁掉padding。

paired_summary.json由raw生成，包含各worker和快照的配对delta mean/std/min/max、relative_reduction、gap_before/after、改善重复数；update100另报gap_closure和gap_closure_H。闭合比例按“均值之比”计算，不误写成“每次比值的均值”；分母为零或不适用时null并说明。

分开给 outcome 和 continuation_gate：前者描述实测效应，后者是否达到PLAN的资源分配门槛。保存五项门槛的测量值与布尔判断，不能只填met。8次随机重复不能当独立训练样本；uniform不加入共同随机数配对声明。

summary中的other_snapshots_relative_sse_change按(mean(S_B)−mean(S_A))/mean(S_A)计算，正值表示回退。continuation_gate取not_evaluated/met/not_met。execution_status取not_started/running/complete/incomplete/blocked/failed；阶段与身份检查使用not_run/pass/fail/incomplete/blocked。控制失配或域失败时不得把不完整有效子集填成正式效果，相关汇总写null并链接失败记录。

## 5 报告和复现

按[报告模板](REPORT_TEMPLATE.md)写结论、来源/干预、量级预查、配对结果、尺度/主码变化、高能量区域、回退与域失败、投入门槛和未解决问题。给出原始记录键，不把源码假说写成实测机制。

REPRODUCE.md只把实际执行过的命令列为已执行。至少有一个从保存包/输出复算表格的入口，不需要重新运行编码或训练。需要新建/修复的入口在实现前不得写成可直接运行的现成命令。

完整性清单排除自身，记录所有新文件，旧依赖列为外部引用并保留哈希。可用完整目录交付，不强制ZIP；不再复制旧3.57GB交付包。

回传三项：REPORT.md、summary.json、结果目录或ZIP，并附outcome、continuation_gate、是否有失败、D2未执行。
