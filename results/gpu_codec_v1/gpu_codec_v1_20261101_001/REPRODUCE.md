# G0 实际复现记录

工作目录：`E:/dacomp/research/gradient_comm_lab`。解释器：`E:\dacomp\.venv\Scripts\python.exe`。本轮只读复用已有环境，未安装或升级包；缓存与临时文件由 `scripts/python.ps1` 隔离到本项目。

以下命令已实际执行。`R` 代表 `results/gpu_codec_v1/gpu_codec_v1_20261101_001`；历史尝试受保护，重新运行须建立新的唯一 run 目录，不能覆盖本目录。

```powershell
.\scripts\python.ps1 scripts/probe_environment.py --run-dir results/gpu_codec_v1/gpu_codec_v1_20261101_001
.\scripts\python.ps1 scripts/minimal_cuda_probe.py --run-dir results/gpu_codec_v1/gpu_codec_v1_20261101_001
.\scripts\python.ps1 scripts/freeze_g0_inputs.py --run-dir results/gpu_codec_v1/gpu_codec_v1_20261101_001
.\scripts\python.ps1 scripts/probe_reference_semantics.py --run-dir results/gpu_codec_v1/gpu_codec_v1_20261101_001
.\scripts\python.ps1 scripts/finalize_g0_delivery.py --run-dir results/gpu_codec_v1/gpu_codec_v1_20261101_001
```

环境探针成功写入 `environment.json`；最小扩展探针退出 2，原始错误见 `logs/minimal_cuda_build.log`。没有 nvcc 编译命令被真正执行，也没有生成可调用扩展。实际尝试的 build API、源码、参数和 traceback 在 `logs/minimal_cuda_probe.json`；仅为构建子进程加入现有 cl/Ninja 路径，没有更改全局环境。

夹具脚本调用原 `splcomm.p2_replay.load_inputs(calibration_root, 'pilot')`，仅加载 seed1101 的更新前100/500/2500及每个输入4个worker。先转BF16、再BF16除以4，最后补零。固定生成器和main→scale顺序见 `input_manifest.json`。大文件留在 `fixtures_manifest.json` 列出的本机路径；重建要求所列历史checkpoint及相同参考来源可取得，文件级和tensor级SHA分别记录。

CPU语义探针已执行并通过，但不是GPU正确性矩阵。完整G1正确性入口、性能入口、显存/trace和汇总作图尚未实现或执行，没有可声称已复现的相关命令。不要直接运行旧项目 `tests/test_dynamiq.py`，其fixture可能写回旧P1证据。

依赖快照通过实际命令 `E:\dacomp\.venv\Scripts\python.exe -m pip freeze --all` 获取，见 `requirements.txt` 和 `logs/dependency_capture.json`。它是旧解释器当前包清单，不代表在本项目已创建新venv或完成可移植安装。若未来需安装/升级包，按技术方案创建本项目独立venv。

后续环境修复尚未执行：补齐完整Toolkit并选择其支持的宿主工具链，在新run中先通过最小构建与真实GPU调用，再实施G1。没有自动进入G2。

## 7 交付前审阅修订

以下命令已实际执行，澄清适配器路径基准和失败阶段措辞；保留修订前草稿，不重跑或修改实验日志。

```powershell
.\scripts\python.ps1 scripts/review_g0_delivery.py --run-dir results/gpu_codec_v1/gpu_codec_v1_20261101_001
```
