# Gradient Communication Lab

低比特梯度通信工程项目：CUDA 编解码、单 GPU 逻辑 ring、实际字节预算比较及尺度数值诊断。

截至 2026-11-05，六轮实验与诊断已归档。最新结果在固定开发输入上确认了同包预算下约 12%–15% 的量化 SSE 改善；预设继续门槛未通过，当前论文路线暂停追加实验。项目未声称真实多卡 AllReduce 加速、训练精度提升或已成立的独立论文贡献。

## 阅读入口

- **[项目汇总 PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)**：目标、阶段成果、核心数字和结论边界。
- **[数据位置 DATA_INDEX.md](DATA_INDEX.md)**：所有主要数据、报告、源码和外部依赖的本机位置。
- GitHub 精简归档另附 `ARCHIVE_MANIFEST.json`（上传文件、来源与 SHA256）及 `LOCAL_DATA_CATALOG.json`（完整六轮结果文件位置与收录状态）。本机原始项目目录不要求存在这两个发布清单。

## 已完成的工作

| 工作 | 验证结果 | 证据 |
|---|---|---|
| 自定义 CUDA 编解码 | 2098 项正确性记录；指定 8 MiB BF16 预分配 roundtrip 条件下，相对同轮 PyTorch eager 的配对提速中位数 36.44× / 39.41× | [G1 报告](results/gpu_codec_v1/gpu_codec_v1_20261104_001/REPORT.md) |
| 单 GPU 逻辑 ring | 174 个 collective、4741 项正确性记录；逐跳算术、消息所有权和实际字节账本通过 | [G2a 报告](results/logical_ring_v1/logical_ring_v1_20261104_001/REPORT.md) |
| 官方固定组件比较 | 实际 4/8 位码流；780 条局部与 195 条 one-shot 误差记录；观察到输入相关的排序反转 | [D0/D1 报告](results/bounded_discovery_v1/bounded_discovery_v1_20261104_001/REPORT.md) |
| 固定尺度加数诊断 | 96/96 配对改善，三快照 SSE 降 12.05% / 14.81% / 12.84%，字节数不变 | [诊断报告](results/epsilon_diagnostic_v1/epsilon_diagnostic_v1_20261105_001/REPORT.md) |

本地 G1/G2a 名义 b4/b8 profile 实际主码为 **5/9 位**；官方诊断组件实际主码为 **4/8 位**。两个实验对象的数字不能混用。速度结果只针对局部 codec 与指定 eager 基线；逻辑 ring 没有测量物理网络。

## 无 GPU 核验最新摘要

在 GitHub 精简归档根目录，使用 Python 3.10 或更新版本运行：

```bash
python tools/verify_epsilon_summary.py --run-dir results/epsilon_diagnostic_v1/epsilon_diagnostic_v1_20261105_001
```

归档提供 `metrics_raw.jsonl.gz`、`paired_summary.json` 和 `summary.json`。此入口仅使用标准库，核对记录覆盖、配对汇总、误差降幅和继续门槛；不会运行 CUDA，不验证未随仓库上传的原始张量。原始本机项目中的同一工具位于 `publication_tools/verify_epsilon_summary.py`。

## 代码与复现范围

- `src/gradient_comm_lab/`：PyTorch/CUDA 编码器、逻辑 ring、官方组件 wrapper 和包解析。
- `scripts/`：历史构建、执行、统计和验收入口。
- `plans/`、`reports/`、`reviews/`：任务边界、理论筛选和结果复核。
- `results/`：GitHub 中为精简证据；本机为完整封存交付。
- `upstream/bounded_discovery_v1/dynamiq/`：固定上游来源副本；许可见其 LICENSE。

已执行环境：Windows、Python 3.10.3、PyTorch 2.4.1+cu118、CUDA Toolkit 11.8、MSVC 19.38、RTX 4000 Ada。部分历史脚本仍依赖 `E:/dacomp` 和旧 CPU 参考路径，仓库不是跨平台一键复现包。历史报告里的原始张量、包、trace 和工具链链接可能只在完整本机归档中存在；具体位置和状态见数据索引及发布清单。

会写入结果的历史入口不能直接在封存 run 上重放。新的执行需要另建输出目录并核对实际依赖；本次归档没有重跑训练或扩展实验范围。

## 来源

官方组件来自 [DynamiQ](https://github.com/CharlesHan24/DynamiQ)，commit `96da2ee480bbfd8e2049de3fbd8595cc9d7a30b8`。精简归档中的 `THIRD_PARTY.md` 说明上游许可与诊断补丁身份。项目自身没有在本次归档中新增开源许可。
