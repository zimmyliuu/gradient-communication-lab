# G0 交付复核与后续执行补充

日期：2026-11-01，北京时间。复核对象：[主报告](../results/gpu_codec_v1/gpu_codec_v1_20261101_001/REPORT.md)及其完整结果目录。

结论：**接受为可信的 G0 阻碍交付；G0 自定义扩展编译门槛未通过，G1 未完成，暂不进入 G2。继续当前方向。** 现有结果没有提供 GPU codec 的正面或负面性能证据，也没有显示显卡算力不足。

本补充只调整技术方案 1.0 的执行依赖：纯 PyTorch CUDA eager 基线可以在编译工具链尚未就绪时推进；自定义 CUDA 分支仍须先通过真实编译与调用。完整 G1 的正确性和性能验收标准保持不变。

## 1 独立复核结果

| 项目 | 复核结果 |
|---|---|
| 文件完整性 | artifact_integrity 47 项、source_manifest 21 项、fixtures 13 项、specification 8 项及摘要引用的契约哈希全部匹配；这些清单有交叉，不相加为独立文件数 |
| CPU 参考 | 7 份原参考文件与记录和快照一致 |
| 真实梯度 | 三个指定开发 checkpoint 的 12 个梯度，FP32→BF16→BF16 除以 4→补零的变换逐一匹配 |
| 阈值与 LUT | 张量身份匹配；按 input_id 独立生成阈值，12/12 一致 |
| CPU 常数探针 | 对 65,280 个有限 BF16 位模式独立复算，FP32 字面 epsilon 加法后 cast 有 1,527 个差异，显式 BF16 标量与参照差异为 0 |
| 构建失败 | 错误发生在 CUDA_HOME 链接路径准备；尚未执行 nvcc 编译或调用 GPU 扩展 |
| 缺失结果 | 正确性、性能、显存和 trace 未执行，报告如实标记，没有用空文件伪装完成 |

原交付可继续作为后续输入和数值参照。CPU 探针揭示的是移植需要遵守的数值语义，不单独承担算法创新或实际训练收益结论。

源码快照仍依赖原项目绝对路径，因此当前是本机可复现交付。未来将其作为公开履历项目发布时，应将 reference root 参数化或提供完整依赖获取步骤；这不影响本次 G0 交付的接受。

## 2 阻碍定位

[原始构建日志](../results/gpu_codec_v1/gpu_codec_v1_20261101_001/logs/minimal_cuda_build.log)记录的直接错误为：

```text
OSError: CUDA_HOME environment variable is not set.
```

它发生于 `torch.utils.cpp_extension.load()` 准备 Toolkit 的 `lib/x64` 链接目录时。现有证据只证明检查范围内未找到可用 Toolkit，不能推出整个磁盘上一定没有，也不能证明当前 MSVC 已经实际编译失败。cl 输出解码警告是另外一个待处理项。

本次复核还用现有 `E:/dacomp/.venv/Scripts/python.exe` 做了少量内置 CUDA 算子检查：BF16 `searchsorted`、`take` 和 int64 位操作均成功，`CUDA_HOME` 仍为 null。该检查只证明相关基础算子能运行，不是完整 codec 一致性或性能验收。

已安装的 CUDA 版 PyTorch 可以运行其预编译算子；自定义 CUDA 扩展另需本地编译工具链。因此，原方案把最小扩展编译设为所有 G1 工作的前置条件过于严格，应改为只约束自定义扩展分支。

## 3 下一轮分支 A 纯 PyTorch GPU 基线

在新 run 中推进，保留原输入夹具、CPU 参考、5/9 位格式、显式阈值和 BF16 分步舍入语义。建议依次完成：

1. 实现 GPU 位打包/解包和解码，测试全部合法码值及 b=8 整数索引修复。
2. 实现 PyTorch CUDA eager 编码及 roundtrip。紧凑 payload、尺度和输出留在 GPU；校验时才传回 CPU。
3. 完成方案规定的边界、合成和 12 个真实梯度输入对照。相同输入、阈值、LUT 下，码流和 BF16 位模式必须与 CPU 一致。
4. 通过数值验收后，测 PyTorch 基线的局部成本；可以先提交部分规定网格，但必须标记覆盖范围，不能提前声称完整 G1 通过。

本分支不使用自定义 extension、`torch.compile` 或依赖本地 CUDA 编译的替代路径。GPU 基线是否完整通过仍由测试决定，不能由几项基础算子成功推断。

这是 G1 的中间交付，可命名为 G1a，但机器字段仍用 `stage="G1"` 并加 `substage="torch_cuda_baseline"`。示例状态仅说明口径：

```json
{
  "stage": "G1",
  "substage": "torch_cuda_baseline",
  "execution_status": "incomplete",
  "g1_acceptance": "incomplete",
  "optimization_gain": "not_evaluated",
  "claims": {
    "multigpu_speedup_measured": false
  }
}
```

其余字段沿用 summary 模板。完整两实现矩阵未完成时，顶层 `correctness_status` 和 `benchmark_validity` 仍为 `incomplete`；可添加逐实现状态，明确 PyTorch 基线中已经通过的部分。不得填写不存在的优化版本或加速比。

## 4 下一轮分支 B 自定义 CUDA 工具链

优先保留当前 PyTorch 2.4.1+cu118，定位或补齐完整 CUDA Toolkit 11.8，包含 nvcc、头文件及开发库，并选用明确版本的兼容 MSVC 工具集与 Windows SDK。

CUDA 11.8 官方表列出 MSVC 193x、192x、191x；当前 19.51 不在此支持表内。193x 对应 19.30–19.39，不能只写“安装最新版 v143”而不检查实际 cl 版本。该支持表用于选择候选组合，最终仍以实际编译与调用为准。[NVIDIA CUDA 11.8 Windows 要求](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-installation-guide-microsoft-windows/index.html)

环境验收顺序：

1. 核实 Toolkit 实体文件和 `nvcc --version`，再将进程内 CUDA_HOME 指向真实根目录。
2. 初始化选定 toolset 的完整开发环境，核对实际 cl/link、INCLUDE、LIB 和 Windows SDK；仅把 cl.exe 加入 PATH 不代表环境齐全。
3. 记录 PyTorch 子进程最终选用的编译器，防止它重新激活另一套默认 VS 环境。
4. 在新 run 中重新执行最小扩展构建、加载和 GPU 调用，核对输出，再开始自定义 codec。可在构建进程内尝试 `VSLANG=1033` 处理输出语言问题，是否有效要记录实测。

不要用 `--allow-unsupported-compiler` 绕过版本检查后当作兼容性证明。当前 GPU runtime 能运行，没有证据需要重装驱动。依赖安装和工具集配置应避免改坏旧实验环境；本次复核没有执行安装或环境修改。[PyTorch 扩展说明](https://docs.pytorch.org/docs/stable/cpp_extension.html)

## 5 下一次回传

新 run 应引用本次交付为输入来源，重新绑定实际实现代码和配置哈希。G1a 可先回传，不必等编译器问题解决后才提交任何 GPU 证据。

至少提供：

- REPORT.md、summary.json 和新 run 路径。
- PyTorch GPU 源码、固定契约、CPU/GPU 正确性逐 case 记录；失败时附最小反例。
- 如已计时，附原始块记录和覆盖范围；尚无优化版时不提供跨实现加速结论。
- 工具链修复的实际记录、最小编译/调用结果；若未成功则保留失败日志。

下一次决策重点是 GPU 数值移植能否成立，以及已有 PyTorch 实现的主要成本在哪里。完整 G1 通过后，再讨论逻辑 ring 集成与通信调度。
