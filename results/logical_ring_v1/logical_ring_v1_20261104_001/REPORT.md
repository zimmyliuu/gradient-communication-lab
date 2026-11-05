# G2a 固定配置单 GPU 逻辑 ring 交付报告

日期：2026-11-04，北京时间。run_id：`logical_ring_v1_20261104_001`。**G2a：通过；完整 G2：incomplete。** 按用户“按照原计划完成，不要暂缓版本”的明确指令执行，授权记录见 [execution_authority.json](execution_authority.json)。未修改旧项目或 G1 封存结果。

## 结果

174 个集体用例覆盖 N=1/2/4、b4/b8、raw_fp32/predivided_bf16 两入口、全部指定长度，以及零值、正负值、抵消、有限大值与 subnormal。真实部分为三个快照各两个位宽，共六个 collective。

- 正确性记录：4741，全部 pass；计数包含分项检查，不是独立实验。
- 码流/量化记录：4864；有向消息记录：6848；中间态记录：13152；所有权回归记录：6。
- 每个输入执行独立 CPU 参照和 GPU baseline/repeat/reverse；CPU 包、编码输入、每次解码、BF16 部分和与各 rank 输出逐位核对。
- N=4 非空输入每次有 16 次编码、24 次发送，sink 再量化计入编码而不增加发送；AllGather 仅转发最终包。
- N=1 与空流零量化、零消息，保留入口 dtype；非空零流仍执行协议。

## 真实梯度回放

D=818048，新的 P=819200，每 chunk L=204800；从已预除 4 的 BF16 逻辑输入读取，没有二次预除。

| 集体输入 | 位宽 | 有向发送数 | 实际发送字节 | 对 BF16 输入 FP64 和的 MSE |
|---|---:|---:|---:|---:|
| real_update000100_n4_b4 | 4 | 24 | 3,417,600 | 9.05359261e-09 |
| real_update000100_n4_b8 | 8 | 24 | 5,875,200 | 7.09403708e-11 |
| real_update000500_n4_b4 | 4 | 24 | 3,417,600 | 2.0606021e-08 |
| real_update000500_n4_b8 | 8 | 24 | 5,875,200 | 1.21611983e-10 |
| real_update002500_n4_b4 | 4 | 24 | 3,417,600 | 4.68444688e-08 |
| real_update002500_n4_b8 | 8 | 24 | 5,875,200 | 2.7013008e-10 |

误差仅作量化诊断；通过条件是与 CPU 协议逐位一致，不是误差为零。原始 FP32 入口的诊断参照另为原输入 FP64 平均值。旧三个开发快照不支持训练泛化或方法排序。

## 实现与证据

GPU 路由使用逐轮队列屏障；单个 codec plan 可复用，但每次编码克隆三段紧凑消息，decode 输出也克隆。最终包保持至 AllGather 完成。详见 [BUFFER_OWNERSHIP.md](BUFFER_OWNERSHIP.md)。消息计数与字节来自实际载体张量，审计哈希及路由记录不计为网络头。

阈值按 case、位宽、phase、rank、chunk、encoding_ordinal 独立定位，共冻结 1216 个事件；CPU/GPU 使用同一阈值字节。原 LCG 和随机池未推进。b4/b8 主码仍实际存储 5/9 位。

显式加载并核对已验收 a02 模块 SHA256 `c4e37ff9394580bb15fe5d277115f616a439f2741528dff6826bc77e85f06b58`；新入口的实际加载与结果 smoke 已记录。输入、阈值、LUT 保护、逆序 host rank、重复调用、两包在途与解码输出复用均有独立检查。

源码、用例、阈值身份在正式矩阵前登记于 [execution_registration.json](execution_registration.json)。账本由不导入两种 ring 的独立审计程序重新推导。审计状态以 [summary.json](summary.json) 和其 audit_proof 为准。

## 失败记录及限制

开发尝试保留在 attempts/；临时有限极值 probe 曾误把整条补齐流送入单 chunk 阈值，触发明确 ValueError，修正 probe 后通过，未修改 codec 或数据。全部实际问题见 [issues.jsonl](issues.jsonl)。该临时 probe 保留了错误类型、消息、原因和修正记录，未单独保存完整 traceback；正式矩阵无失败。runner 在失败时保存反例的分支未因本轮正式失败而触发。

没有重跑 G1 性能矩阵，没有新增训练、动态 b2/4/8 控制、随机状态闭环或网络通信。没有测量网络 AllReduce 延迟或多卡加速；本阶段不等于完整 DynamiQ GPU 移植。G2b 待交付复核后确定，当前 `full_g2_acceptance=incomplete`。

复现入口、环境依赖与封存核验见 [REPRODUCE.md](REPRODUCE.md)。
