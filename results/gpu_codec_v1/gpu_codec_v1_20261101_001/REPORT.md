# GPU 低比特编解码实施报告

run_id：`gpu_codec_v1_20261101_001`。执行日期：2026-11-01，北京时间。技术方案版本：1.0。对象：**局部 codec 固定配置**。

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

搜索覆盖PATH、CUDA注册表及记录中的常见/非标准安装目录；两处目录访问被拒绝，不能声称进行了完整全盘排查。现有MSVC为19.51.36260，CUDA11.8官方Windows支持表列出193x、192x、191x，当前版本不在该表中；这是兼容性待验证事项，本次实际失败发生在实际源码编译前；此前仅进行过cl版本查询。[NVIDIA CUDA11.8 Windows要求](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-installation-guide-microsoft-windows/index.html)

选择下一步：**fix_environment**。本次没有安装软件、改旧Python环境或更新全局工具链。单独填写CUDA_HOME不能代替真实Toolkit文件；修复后以真实扩展编译及GPU调用结果为准。G2–G4不在本轮执行范围。

## 7 复现与结论边界

实际入口与命令见 [REPRODUCE.md](REPRODUCE.md)。完整目录：`E:/dacomp/research/gradient_comm_lab/results/gpu_codec_v1/gpu_codec_v1_20261101_001`。源码快照在`code/`；大夹具留在本机并提供SHA，见 [fixtures_manifest.json](fixtures_manifest.json)。失败日志与未执行项全部保留。

本轮仅形成G0环境、CPU参考和输入准备证据。局部GPU codec一致性与性能、完整DynamiQ GPU路径、新训练质量、多卡吞吐及训练加速均未验证。此前项目结果没有修改，也未被计入本轮新成果。

## 8 交付文件核对

- [x] summary、环境、CPU部分契约、计划基准配置已填写，template_only=false；明确GPU绑定尚缺。
- [x] CPU参考、适配器、最小构建探针、输入、LUT、阈值与本地夹具哈希已记录。
- [x] 源码快照、依赖快照、实际命令、失败构建日志与问题清单可取得。
- [x] 未生成的文件与原因在summary.missing_artifacts逐项列出，未用空文件冒充执行。
- [ ] G1正确性、原始计时块、性能汇总与显存记录：未执行。
- [ ] profiler trace、三组数据图表及完整G1验收：未执行。
