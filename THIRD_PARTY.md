# 第三方来源与许可

`upstream/bounded_discovery_v1/dynamiq/` 是 DynamiQ 固定来源副本，原仓库 https://github.com/CharlesHan24/DynamiQ ，commit `96da2ee480bbfd8e2049de3fbd8595cc9d7a30b8`。Apache-2.0 许可证原文保留在该目录的 LICENSE。

`patches/epsilon_b8_scale_add.diff` 是针对冻结编译源的诊断改动，目标为 `cuda_hierarchical_mee.cu`；其编译宏分别取 `1e-7f` 和 `0.0f`，仅作用于 b8。它不是已经验证所有输入域的部署补丁。

自有工程代码、结果与文档在本次归档中没有新增开源许可授权。CUDA、cuRAND、CUPTI、MSVC 和 Python 环境没有打包上传。CPU 参考的本机外部依赖见 DATA_INDEX.md。
