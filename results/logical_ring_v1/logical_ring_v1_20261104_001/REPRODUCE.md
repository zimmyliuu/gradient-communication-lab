# Reproduction

The executed environment is Windows, Python 3.10.3 / PyTorch 2.4.1+cu118 and the accepted CUDA 11.8 a02 binary. See environment.json and logs/cuda_build_manifest.json. This is not a portable wheel or a network backend.

Run from E:/dacomp/research/gradient_comm_lab. Preserve the sealed directory; use a fresh RUN path under results/logical_ring_v1. All commands below use scripts/python.ps1 (project-local caches).

```powershell
./scripts/python.ps1 scripts/prepare_logical_ring.py --run-dir <RUN>
./scripts/python.ps1 scripts/check_logical_ring.py --run-dir <RUN> --attempt preparation --manifest-only
./scripts/python.ps1 scripts/register_logical_ring.py --run-dir <RUN>
./scripts/python.ps1 scripts/check_logical_ring.py --run-dir <RUN> --attempt formal_a01 --formal
./scripts/python.ps1 scripts/finalize_logical_ring.py --run-dir <RUN>
./scripts/python.ps1 scripts/audit_logical_ring.py --run-dir <RUN> --output logs/audit_initial.json
./scripts/python.ps1 scripts/finalize_logical_ring.py --run-dir <RUN> --audit-proof logs/audit_initial.json
./scripts/python.ps1 scripts/verify_logical_ring_delivery.py --run-dir <RUN>
./scripts/python.ps1 scripts/audit_logical_ring.py --run-dir <RUN> --output logs/audit_final.json
./scripts/python.ps1 scripts/finalize_logical_ring.py --run-dir <RUN> --audit-proof logs/audit_final.json --seal
```

Before the audit, include the unchanged BUFFER_OWNERSHIP.md and issues.jsonl describing actual execution issues; include the report generator's source snapshot. Do not fabricate pass results for a new attempt. The original actual command line is recorded in execution_start.json; development attempts have their own immutable execution_start.json and source hashes.

The input tensors are bundled in fixtures/. The source archives and their tensor identities bind real inputs; for full regeneration the accepted G1 and original CPU project at E:/dacomp/research/spl_model_communication must remain available. Copies of reference Python modules and accepted binary are provided under code/reference and code/build. The unchanged CPU adapter retains its original absolute dependency path. On relocation, explicitly rebind this path and the build manifest in a NEW run, then requalify; never edit a sealed manifest. Thresholds regenerate from the frozen event keys and exact PyTorch/BF16 operation order. Serialization archive hashes can differ in a new path even when tensor bytes match.

The archive includes all development evidence and fixtures. artifact_integrity.json binds every pre-seal file; adjacent postseal_verification.json binds the integrity file and ZIP, whose members were checked byte-for-byte. No compute-sanitizer, NCCL, physical multi-GPU or new training validation is claimed.

After sealing, run `./scripts/python.ps1 scripts/verify_logical_ring_delivery.py --run-dir <RUN> --postseal` to verify exact ZIP membership, every member SHA, and the old G1 integrity manifest itself. This writes only an adjacent verification file.
