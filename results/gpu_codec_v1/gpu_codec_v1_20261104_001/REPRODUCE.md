# G1b 复现与审计入口

工作目录：`E:/dacomp/research/gradient_comm_lab`。解释器：`E:/dacomp/.venv/Scripts/python.exe`，通过 `scripts/python.ps1` 只读复用，所有缓存写本项目 `.cache/`；没有升级旧环境或驱动。正式运行使用 CUDA 11.8/MSVC 19.38/SDK 10.0.26100.0、RTX 4000 Ada、PyTorch 2.4.1+cu118，精确当前环境以 environment.json 为准。

## 实际执行命令

以下结构直接取自 `logs/execution_commands.json`，表示本轮执行记录，包含开发失败。原 console/raw/JSON 均可检索。

```json
{
  "run_id": "gpu_codec_v1_20261104_001",
  "working_directory": "E:\\dacomp\\research\\gradient_comm_lab",
  "interpreter": "E:\\dacomp\\.venv\\Scripts\\python.exe",
  "launcher": "scripts/python.ps1",
  "recorded_at": "2026-11-04T18:04:07.890078+08:00",
  "commands_actually_executed": [
    {
      "phase": "cuda_build_a01",
      "command": [
        "E:\\dacomp\\.venv\\Scripts\\python.exe",
        "scripts/build_cuda_codec.py",
        "--run-dir",
        "results/gpu_codec_v1/gpu_codec_v1_20261104_001",
        "--attempt",
        "a01"
      ],
      "status": "compiled_loaded",
      "evidence": "logs/cuda_build_a01.json"
    },
    {
      "phase": "cuda_dev_a01",
      "command": [
        "E:\\dacomp\\.venv\\Scripts\\python.exe",
        "scripts/dev_cuda_codec.py",
        "--run-dir",
        "results/gpu_codec_v1/gpu_codec_v1_20261104_001",
        "--attempt",
        "a01"
      ],
      "status": "pass",
      "evidence": "logs/cuda_dev_a01.json"
    },
    {
      "phase": "cuda_build_a02",
      "command": [
        "E:\\dacomp\\.venv\\Scripts\\python.exe",
        "scripts/build_cuda_codec.py",
        "--run-dir",
        "results/gpu_codec_v1/gpu_codec_v1_20261104_001",
        "--attempt",
        "a02"
      ],
      "status": "compiled_loaded",
      "evidence": "logs/cuda_build_a02.json"
    },
    {
      "command": [
        "scripts/python.ps1",
        "scripts/check_cuda_codec.py",
        "--run-dir",
        "results/gpu_codec_v1/gpu_codec_v1_20261104_001",
        "--output",
        "logs/correctness_preliminary_a01.jsonl",
        "--contract-sha256",
        "068400a4067849390aec27a01d9f46811e395f0765f63ccb67fdeda82e470e7d"
      ],
      "started_at": "2026-11-04T17:32:42.612786+08:00",
      "finished_at": "2026-11-04T17:33:34.544708+08:00",
      "observed_shell_exit_code": 1,
      "log": "logs/correctness_preliminary_a01_console.log",
      "status": "fail",
      "scope": "preliminary_not_formal",
      "note": "PowerShell host reported1 on failedharness; harness main returns2 forfailures."
    },
    {
      "command": [
        "scripts/python.ps1",
        "scripts/check_cuda_codec.py",
        "--run-dir",
        "results/gpu_codec_v1/gpu_codec_v1_20261104_001",
        "--output",
        "logs/correctness_preliminary_a02.jsonl",
        "--contract-sha256",
        "068400a4067849390aec27a01d9f46811e395f0765f63ccb67fdeda82e470e7d"
      ],
      "started_at": "2026-11-04T17:36:00.579994+08:00",
      "finished_at": "2026-11-04T17:36:46.116346+08:00",
      "observed_shell_exit_code": 1,
      "log": "logs/correctness_preliminary_a02_console.log",
      "status": "fail",
      "scope": "preliminary_not_formal",
      "note": "PowerShell host reported1 on failedharness; harness main returns2 forfailures."
    },
    {
      "command": [
        "scripts/python.ps1",
        "scripts/check_cuda_codec.py",
        "--run-dir",
        "results/gpu_codec_v1/gpu_codec_v1_20261104_001",
        "--output",
        "logs/correctness_preliminary_a03.jsonl",
        "--contract-sha256",
        "068400a4067849390aec27a01d9f46811e395f0765f63ccb67fdeda82e470e7d"
      ],
      "started_at": "2026-11-04T17:37:54.772853+08:00",
      "finished_at": "2026-11-04T17:38:41.506261+08:00",
      "observed_shell_exit_code": 0,
      "log": "logs/correctness_preliminary_a03_console.log",
      "status": "pass",
      "scope": "preliminary_not_formal",
      "note": "PowerShell host reported1 on failedharness; harness main returns2 forfailures."
    },
    {
      "command": [
        "scripts/python.ps1",
        "scripts/check_cuda_codec.py",
        "--run-dir",
        "results/gpu_codec_v1/gpu_codec_v1_20261104_001",
        "--manifest-only"
      ],
      "observed_shell_exit_code": 0,
      "status": "complete",
      "note": "CPU-only coverage generation executedaftereach coverage change; final2098manifest beforefreeze."
    },
    {
      "command": [
        "scripts/python.ps1",
        "-m",
        "pytest",
        "tests/test_cuda_codec.py",
        "-q",
        "--junitxml=results/gpu_codec_v1/gpu_codec_v1_20261104_001/logs/pytest_cuda_codec_a01.xml"
      ],
      "observed_shell_exit_code": 0,
      "status": "pass",
      "log": "logs/pytest_cuda_codec_a01.log",
      "tests": 50,
      "failures": 0
    },
    {
      "command": [
        "scripts/python.ps1",
        "scripts/check_cuda_codec.py",
        "--run-dir",
        "results/gpu_codec_v1/gpu_codec_v1_20261104_001"
      ],
      "started_at": "2026-11-04T17:40:13.169452+08:00",
      "finished_at": "2026-11-04T17:41:00.302763+08:00",
      "observed_shell_exit_code": 0,
      "status": "pass",
      "log": "logs/correctness_formal_console.log",
      "scope": "formal_frozen_sources_and_contract"
    },
    {
      "command": [
        "scripts/python.ps1",
        "scripts/check_cuda_codec_replay.py",
        "--help"
      ],
      "observed_shell_exit_code": 0,
      "status": "pass",
      "log": "logs/check_cuda_codec_replay_help.log",
      "scope": "CPU_interface_only_not_full_replay"
    },
    {
      "phase": "pilot",
      "started_at": "2026-11-04T17:42:02.432064+08:00",
      "command": "./scripts/python.ps1 scripts/benchmark_cuda_pair.py --run-dir results/gpu_codec_v1/gpu_codec_v1_20261104_001 --mode pilot",
      "console_log": "logs/benchmark_pilot_console.log",
      "status": "complete",
      "exit_code": 0,
      "finished_at": "2026-11-04T17:43:06.645775+08:00"
    },
    {
      "phase": "measure",
      "started_at": "2026-11-04T17:43:06.645775+08:00",
      "command": "./scripts/python.ps1 scripts/benchmark_cuda_pair.py --run-dir results/gpu_codec_v1/gpu_codec_v1_20261104_001 --mode measure",
      "console_log": "logs/benchmark_measure_console.log",
      "status": "complete",
      "exit_code": 0,
      "finished_at": "2026-11-04T18:00:21.612778+08:00"
    },
    {
      "phase": "register_g1b",
      "command": [
        "E:\\dacomp\\.venv\\Scripts\\python.exe",
        "scripts/register_g1b.py",
        "--run-dir",
        "results/gpu_codec_v1/gpu_codec_v1_20261104_001"
      ],
      "status": "complete",
      "exit_code": 0,
      "evidence": "execution_registration.json"
    },
    {
      "phase": "benchmark_cuda_pair",
      "command": [
        "E:\\dacomp\\.venv\\Scripts\\python.exe",
        "scripts/benchmark_cuda_pair.py",
        "--run-dir",
        "results/gpu_codec_v1/gpu_codec_v1_20261104_001",
        "--mode",
        "register"
      ],
      "status": "complete",
      "exit_code": 0,
      "evidence": "logs/benchmark_design_registration.json"
    },
    {
      "phase": "collect_g1b_traces",
      "command": [
        "E:\\dacomp\\.venv\\Scripts\\python.exe",
        "scripts/collect_g1b_traces.py",
        "--run-dir",
        "results/gpu_codec_v1/gpu_codec_v1_20261104_001"
      ],
      "status": "complete",
      "exit_code": 0,
      "evidence": "traces/g1b_trace_index.json"
    },
    {
      "phase": "summarize_g1b",
      "command": [
        "E:\\dacomp\\.venv\\Scripts\\python.exe",
        "scripts/summarize_g1b.py",
        "--run-dir",
        "results/gpu_codec_v1/gpu_codec_v1_20261104_001"
      ],
      "status": "complete",
      "exit_code": 0,
      "evidence": "analysis.json"
    },
    {
      "command": [
        "E:\\dacomp\\.venv\\Scripts\\python.exe",
        "E:\\dacomp\\research\\gradient_comm_lab\\scripts\\profile_cuda_pair.py",
        "--run-dir",
        "E:\\dacomp\\research\\gradient_comm_lab\\results\\gpu_codec_v1\\gpu_codec_v1_20261104_001",
        "--attempt",
        "c4la01",
        "--method",
        "cuda_optimized_repaired",
        "--bits",
        "4",
        "--numel",
        "4194304"
      ],
      "started_at": "2026-11-04T18:00:50.328348+08:00",
      "finished_at": "2026-11-04T18:00:54.374854+08:00",
      "returncode": 0,
      "console_log": "logs/trace_c4la01_0_console.log",
      "console_sha256": "8f8dd85320808a5a2ae19a65ef314d4f855f9d174239968103a369b8493c918c"
    },
    {
      "command": [
        "E:\\dacomp\\.venv\\Scripts\\python.exe",
        "E:\\dacomp\\research\\gradient_comm_lab\\scripts\\audit_cuda_pair_trace.py",
        "--run-dir",
        "E:\\dacomp\\research\\gradient_comm_lab\\results\\gpu_codec_v1\\gpu_codec_v1_20261104_001",
        "--attempt",
        "c4la01"
      ],
      "started_at": "2026-11-04T18:00:54.382506+08:00",
      "finished_at": "2026-11-04T18:00:54.540241+08:00",
      "returncode": 0,
      "console_log": "logs/trace_c4la01_1_console.log",
      "console_sha256": "55d93639dc9f577f9d2cc7eea26de0046435707bc67d1e84fcc0d3f3dce85608"
    },
    {
      "command": [
        "E:\\dacomp\\.venv\\Scripts\\python.exe",
        "E:\\dacomp\\research\\gradient_comm_lab\\scripts\\profile_cuda_pair.py",
        "--run-dir",
        "E:\\dacomp\\research\\gradient_comm_lab\\results\\gpu_codec_v1\\gpu_codec_v1_20261104_001",
        "--attempt",
        "c4sa01",
        "--method",
        "cuda_optimized_repaired",
        "--bits",
        "4",
        "--numel",
        "65536"
      ],
      "started_at": "2026-11-04T18:00:46.849418+08:00",
      "finished_at": "2026-11-04T18:00:50.166969+08:00",
      "returncode": 0,
      "console_log": "logs/trace_c4sa01_0_console.log",
      "console_sha256": "1991a26a51e34465ecc0482cbef36fab16fdc6408b5ad2a863d581fae7761226"
    },
    {
      "command": [
        "E:\\dacomp\\.venv\\Scripts\\python.exe",
        "E:\\dacomp\\research\\gradient_comm_lab\\scripts\\audit_cuda_pair_trace.py",
        "--run-dir",
        "E:\\dacomp\\research\\gradient_comm_lab\\results\\gpu_codec_v1\\gpu_codec_v1_20261104_001",
        "--attempt",
        "c4sa01"
      ],
      "started_at": "2026-11-04T18:00:50.173058+08:00",
      "finished_at": "2026-11-04T18:00:50.316316+08:00",
      "returncode": 0,
      "console_log": "logs/trace_c4sa01_1_console.log",
      "console_sha256": "fb88d6c198d3c7e1ffa835fab3e42ae3c4afef6c0104b4f791cda091d5c63f39"
    },
    {
      "command": [
        "E:\\dacomp\\.venv\\Scripts\\python.exe",
        "E:\\dacomp\\research\\gradient_comm_lab\\scripts\\profile_cuda_pair.py",
        "--run-dir",
        "E:\\dacomp\\research\\gradient_comm_lab\\results\\gpu_codec_v1\\gpu_codec_v1_20261104_001",
        "--attempt",
        "c8la01",
        "--method",
        "cuda_optimized_repaired",
        "--bits",
        "8",
        "--numel",
        "4194304"
      ],
      "started_at": "2026-11-04T18:00:58.017249+08:00",
      "finished_at": "2026-11-04T18:01:02.340570+08:00",
      "returncode": 0,
      "console_log": "logs/trace_c8la01_0_console.log",
      "console_sha256": "71ee72bb6f30cd33f590a9e7bd4fa825b3b7823b786260f6c8d6cedcc8e734a6"
    },
    {
      "command": [
        "E:\\dacomp\\.venv\\Scripts\\python.exe",
        "E:\\dacomp\\research\\gradient_comm_lab\\scripts\\audit_cuda_pair_trace.py",
        "--run-dir",
        "E:\\dacomp\\research\\gradient_comm_lab\\results\\gpu_codec_v1\\gpu_codec_v1_20261104_001",
        "--attempt",
        "c8la01"
      ],
      "started_at": "2026-11-04T18:01:02.346878+08:00",
      "finished_at": "2026-11-04T18:01:02.510985+08:00",
      "returncode": 0,
      "console_log": "logs/trace_c8la01_1_console.log",
      "console_sha256": "92e807afca9a84e548e0787750a412753789f136257825e9efbe4266d3decbbf"
    },
    {
      "command": [
        "E:\\dacomp\\.venv\\Scripts\\python.exe",
        "E:\\dacomp\\research\\gradient_comm_lab\\scripts\\profile_cuda_pair.py",
        "--run-dir",
        "E:\\dacomp\\research\\gradient_comm_lab\\results\\gpu_codec_v1\\gpu_codec_v1_20261104_001",
        "--attempt",
        "c8sa01",
        "--method",
        "cuda_optimized_repaired",
        "--bits",
        "8",
        "--numel",
        "65536"
      ],
      "started_at": "2026-11-04T18:00:54.554103+08:00",
      "finished_at": "2026-11-04T18:00:57.854139+08:00",
      "returncode": 0,
      "console_log": "logs/trace_c8sa01_0_console.log",
      "console_sha256": "7cfaffa3114c3713881a05abcdb281c3dd1ff61630da94873556458ba49650cd"
    },
    {
      "command": [
        "E:\\dacomp\\.venv\\Scripts\\python.exe",
        "E:\\dacomp\\research\\gradient_comm_lab\\scripts\\audit_cuda_pair_trace.py",
        "--run-dir",
        "E:\\dacomp\\research\\gradient_comm_lab\\results\\gpu_codec_v1\\gpu_codec_v1_20261104_001",
        "--attempt",
        "c8sa01"
      ],
      "started_at": "2026-11-04T18:00:57.859916+08:00",
      "finished_at": "2026-11-04T18:00:58.003457+08:00",
      "returncode": 0,
      "console_log": "logs/trace_c8sa01_1_console.log",
      "console_sha256": "5fe6471a54072c05b5dd64f5ce62863c1e6b50e326c921304488da0cf82b7b47"
    },
    {
      "command": [
        "E:\\dacomp\\.venv\\Scripts\\python.exe",
        "E:\\dacomp\\research\\gradient_comm_lab\\scripts\\profile_cuda_pair.py",
        "--run-dir",
        "E:\\dacomp\\research\\gradient_comm_lab\\results\\gpu_codec_v1\\gpu_codec_v1_20261104_001",
        "--attempt",
        "t4la01",
        "--method",
        "torch_cuda_repaired",
        "--bits",
        "4",
        "--numel",
        "4194304"
      ],
      "started_at": "2026-11-04T18:00:32.938385+08:00",
      "finished_at": "2026-11-04T18:00:37.569439+08:00",
      "returncode": 0,
      "console_log": "logs/trace_t4la01_0_console.log",
      "console_sha256": "230b47a88f78f3b71b89c3ea803371d5301106b174a6a74c5dbdda72b4289b25"
    },
    {
      "command": [
        "E:\\dacomp\\.venv\\Scripts\\python.exe",
        "E:\\dacomp\\research\\gradient_comm_lab\\scripts\\audit_cuda_pair_trace.py",
        "--run-dir",
        "E:\\dacomp\\research\\gradient_comm_lab\\results\\gpu_codec_v1\\gpu_codec_v1_20261104_001",
        "--attempt",
        "t4la01"
      ],
      "started_at": "2026-11-04T18:00:37.575099+08:00",
      "finished_at": "2026-11-04T18:00:37.820251+08:00",
      "returncode": 0,
      "console_log": "logs/trace_t4la01_1_console.log",
      "console_sha256": "8ef6f672bab683e409a5fcf883da57d5a2aa12d0e28fedf50147857fd32b2222"
    },
    {
      "command": [
        "E:\\dacomp\\.venv\\Scripts\\python.exe",
        "E:\\dacomp\\research\\gradient_comm_lab\\scripts\\profile_cuda_pair.py",
        "--run-dir",
        "E:\\dacomp\\research\\gradient_comm_lab\\results\\gpu_codec_v1\\gpu_codec_v1_20261104_001",
        "--attempt",
        "t4sa01",
        "--method",
        "torch_cuda_repaired",
        "--bits",
        "4",
        "--numel",
        "65536"
      ],
      "started_at": "2026-11-04T18:00:27.840043+08:00",
      "finished_at": "2026-11-04T18:00:32.670323+08:00",
      "returncode": 0,
      "console_log": "logs/trace_t4sa01_0_console.log",
      "console_sha256": "c6eab3773896a75c712de38e2b1a8348f8362fffd95757ff3e4e338d75ecdad0"
    },
    {
      "command": [
        "E:\\dacomp\\.venv\\Scripts\\python.exe",
        "E:\\dacomp\\research\\gradient_comm_lab\\scripts\\audit_cuda_pair_trace.py",
        "--run-dir",
        "E:\\dacomp\\research\\gradient_comm_lab\\results\\gpu_codec_v1\\gpu_codec_v1_20261104_001",
        "--attempt",
        "t4sa01"
      ],
      "started_at": "2026-11-04T18:00:32.675988+08:00",
      "finished_at": "2026-11-04T18:00:32.925163+08:00",
      "returncode": 0,
      "console_log": "logs/trace_t4sa01_1_console.log",
      "console_sha256": "337be60ab41ba091584ca675ed6f49fa9a72c112e649d77a8822ae1e43e62ebe"
    },
    {
      "command": [
        "E:\\dacomp\\.venv\\Scripts\\python.exe",
        "E:\\dacomp\\research\\gradient_comm_lab\\scripts\\profile_cuda_pair.py",
        "--run-dir",
        "E:\\dacomp\\research\\gradient_comm_lab\\results\\gpu_codec_v1\\gpu_codec_v1_20261104_001",
        "--attempt",
        "t8la01",
        "--method",
        "torch_cuda_repaired",
        "--bits",
        "8",
        "--numel",
        "4194304"
      ],
      "started_at": "2026-11-04T18:00:41.657596+08:00",
      "finished_at": "2026-11-04T18:00:46.578260+08:00",
      "returncode": 0,
      "console_log": "logs/trace_t8la01_0_console.log",
      "console_sha256": "b115ca57fb5b207adaeccbfe42d935dc343ed07a9a31070e192274012245b898"
    },
    {
      "command": [
        "E:\\dacomp\\.venv\\Scripts\\python.exe",
        "E:\\dacomp\\research\\gradient_comm_lab\\scripts\\audit_cuda_pair_trace.py",
        "--run-dir",
        "E:\\dacomp\\research\\gradient_comm_lab\\results\\gpu_codec_v1\\gpu_codec_v1_20261104_001",
        "--attempt",
        "t8la01"
      ],
      "started_at": "2026-11-04T18:00:46.585219+08:00",
      "finished_at": "2026-11-04T18:00:46.835663+08:00",
      "returncode": 0,
      "console_log": "logs/trace_t8la01_1_console.log",
      "console_sha256": "93ad0e54da6cbdbb9264de686e432048ad88c6db8266af77121fc036334b2fbe"
    },
    {
      "command": [
        "E:\\dacomp\\.venv\\Scripts\\python.exe",
        "E:\\dacomp\\research\\gradient_comm_lab\\scripts\\profile_cuda_pair.py",
        "--run-dir",
        "E:\\dacomp\\research\\gradient_comm_lab\\results\\gpu_codec_v1\\gpu_codec_v1_20261104_001",
        "--attempt",
        "t8sa01",
        "--method",
        "torch_cuda_repaired",
        "--bits",
        "8",
        "--numel",
        "65536"
      ],
      "started_at": "2026-11-04T18:00:37.833455+08:00",
      "finished_at": "2026-11-04T18:00:41.388345+08:00",
      "returncode": 0,
      "console_log": "logs/trace_t8sa01_0_console.log",
      "console_sha256": "4cc9a51d7f70de7d67e3028a861b2be8938680182db5e368e064a6105a82902a"
    },
    {
      "command": [
        "E:\\dacomp\\.venv\\Scripts\\python.exe",
        "E:\\dacomp\\research\\gradient_comm_lab\\scripts\\audit_cuda_pair_trace.py",
        "--run-dir",
        "E:\\dacomp\\research\\gradient_comm_lab\\results\\gpu_codec_v1\\gpu_codec_v1_20261104_001",
        "--attempt",
        "t8sa01"
      ],
      "started_at": "2026-11-04T18:00:41.395963+08:00",
      "finished_at": "2026-11-04T18:00:41.644050+08:00",
      "returncode": 0,
      "console_log": "logs/trace_t8sa01_1_console.log",
      "console_sha256": "a445028154a3f1a841bdfd62a604b14a0681c756f75277fe87f26327c382b130"
    }
  ],
  "evidence_sha256": {
    "logs/cuda_build_a01.json": "dbd69254d87e3339e8ae37e1e445fa965ef3d52e55a240777890924349a4a3d0",
    "logs/cuda_dev_a01.json": "42e38b416b205b7674b322bb03872cb59a126baef6533501acdf0a8da5e21bdd",
    "logs/cuda_build_a02.json": "0a2f3dc7650e97ee6c0909103f87ed6370e36feabac5e6dbb5b062be68e1d491",
    "logs/correctness_commands.json": "f6d038396cae5b42e5f5eeeb25f9c3b5a094029dc2813ac882f184e66d29325c",
    "logs/benchmark_commands.json": "5c398a2a8148d0e7a1e7be575f1853faf6eec9fdb4fb9b1215aec745ab35e19d",
    "execution_registration.json": "304eab7af5459b653b4fb54c17afb6ced0a49a19befc8cd3a124cf17ba9c0e82",
    "logs/benchmark_design_registration.json": "c3ed512a4002f8c44b96b77ee2ef8fcd166c1016bac4cd6f5f061f608b02b7ce",
    "traces/g1b_trace_index.json": "6f37bea91e10b0fb24e58c07d574af65aed854ce061a382e1d4488e6ef4de807",
    "analysis.json": "65862fd84a00b20119c2bf0b581af7661e8281281428cedf7265b8521cf8778d",
    "logs/trace_c4la01_execution.json": "6ac01490a1f715209f7005ebe27c8c176d4d51391b46a7268290beab640f8a0e",
    "logs/trace_c4sa01_execution.json": "79cf386eae15abc7d54d7c42ba93797bfe600d9f1a09d802b4c8f5fb891fc1f2",
    "logs/trace_c8la01_execution.json": "c516cd828725d58301ceb3cf73a6498d3b81f0ba93b62ebe8fc5f0f82c7410d1",
    "logs/trace_c8sa01_execution.json": "9510e19356edfe27a53c181122af223477b222e41dbdca868b562f9da80cf606",
    "logs/trace_t4la01_execution.json": "66cd9075af618d34ddeb5b64e157d1da8f8fa074647ec0c7ae5dd83eae900298",
    "logs/trace_t4sa01_execution.json": "7bbfdc9b3c00999b6e409408e9d4dc2a94dd2ad5b7f427cb1533fc4475893a17",
    "logs/trace_t8la01_execution.json": "87f303b5b0bbc5056c2a2a563567e0ffbb2ecbbf9415cc221c1a4c559740a8e9",
    "logs/trace_t8sa01_execution.json": "c27d95e6a4a8ebb27bfb7f1708440ddb0287c98030cba08ed5733d1aca1a6283"
  },
  "initialization": "New directory created by inline Python; all207 prior sealed files and55 original sources verified;82 original fixtures reused unchanged. Evidence: logs/prior_run_integrity_before.json.",
  "scope": "Completed execution evidence only; additional commands in REPRODUCE are labeled replay/audit entrypoints."
}
```

## 冻结与正确性

正式顺序为：新run引用002的82夹具及哈希 → build a01/开发检查 → a02加入跨stream allocator记录 → 预实验a01/a02失败与a03通过 → pytest50通过 → 注册冻结16执行源码/正确性清单/契约 → 正式2098项通过 → pilot冻结K → 双方法配对计时 → 独立GPUtrace → 汇总作图 → 报告/源码收集/独立审计封存。

`correctness_acceptance.json` 绑定每个正式结果的契约哈希；`logs/correctness_start_identity.json` 记录实际加载模块和起始源码快照。`logs/correctness_harness_changes.diff/json` 记录测试助手差异，不改原阈值、输出或容差。重新执行不能回写已封存run，需生成唯一新目录。冻结的 `check_cuda_codec.py` 默认 CUDA loader 指向本次run，而非自动跟随 --run-dir；本次正式结果已核对实际加载a02，因此结果身份不受影响。新run必须先调用 `gradient_comm_lab.cuda_codec.set_build_manifest(new_run / "logs/cuda_build_manifest.json")` 再调用 `check_cuda_codec.main()`。另提供新入口 `scripts/check_cuda_codec_replay.py --run-dir <new_run> ...` 自动做此选择；该shim在正式矩阵之后添加，仅执行过CPU --help检查，未声称完整重跑。复制封存源码时保留相对目录布局和旧参考依赖。

## 夹具与依赖

`input_manifest.json` 和 `fixtures_manifest.json` 引用本机已有 `.pt` 夹具，逐文件与张量哈希校验；本轮未重新生成或改变它们。生成逻辑在封存 `fixtures_v2.py`，真实梯度还依赖原项目已有 checkpoint/回放来源；生成入口是该模块 `prepare_fixtures(new_run)`，本轮执行的是读取和校验既有夹具，不能把重建命令写成已验证跨机器复现。

CPU 参照原目录 `E:/dacomp/research/spl_model_communication/` 为只读外部依赖；其完整路径/哈希见契约及源码清单。不要运行原 `tests/test_dynamiq.py`，该旧测试存在写 provenance 的fixture。CUDA/MSVC/CUPTI组件位于本项目 `.toolchains/`，Windows SDK/Ninja依赖已安装系统组件。ZIP不携带完整环境；外部文件不可得时须先获取或重建，不能仅解压声称可独立运行。

## 测量与复算

`benchmark_cuda_pair.py` 分 register/pilot/measure 阶段，pilot规则先冻结、K只在正式测量前决定；正式数据不调K、不删慢块。两种实现同case/session交替，逐块配对ID在 raw。`summarize_g1b.py` 自动复算统计、paired speedups、噪声、漂移、显存和5张图。`collect_g1b_traces.py` 对8配置分别调用profile/audit，窗口剔除初始化和warmup但保留原始活动，核验零丢记录及逐项raw/Chrome对应。图和trace只是解释性证据，正式性能来自无profiler记录。

`seal_g1b_delivery.py --seal` 独立复核原始结果、源码、夹具、统计、trace与旧封存文件；`package_g1b_delivery.py` 创建含清单的小ZIP。封存后的run不可修改，再测必须另建run。
