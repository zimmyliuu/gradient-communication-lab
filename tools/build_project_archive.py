"""Create the curated project archive; never rewrite sealed experiment runs."""
from pathlib import Path
import gzip
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
DEST = Path('E:/dacomp/deliverables/gradient-communication-lab')
RUNS = [
    ('G0', 'gpu_codec_v1/gpu_codec_v1_20261101_001'),
    ('G1a', 'gpu_codec_v1/gpu_codec_v1_20261101_002'),
    ('G1', 'gpu_codec_v1/gpu_codec_v1_20261104_001'),
    ('G2a', 'logical_ring_v1/logical_ring_v1_20261104_001'),
    ('D0_D1', 'bounded_discovery_v1/bounded_discovery_v1_20261104_001'),
    ('E0_E1_E2', 'epsilon_diagnostic_v1/epsilon_diagnostic_v1_20261105_001'),
]
TEXT_EXTS = {'.md', '.py', '.ps1', '.cpp', '.cu', '.h', '.cuh', '.json', '.jsonl', '.yaml', '.yml', '.toml', '.txt', '.diff', '.csv'}
SKIP_PARTS = {'__pycache__', '.cache', '.pytest_cache', '.toolchains', '.git'}
records = []

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def copy(source, relative=None):
    relative = Path(relative) if relative else source.relative_to(ROOT)
    target = DEST / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    records.append({'path': relative.as_posix(), 'origin': source.as_posix(), 'transform': 'exact_copy', 'source_sha256': sha(source)})

def write(relative, text):
    target = DEST / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding='utf-8')
    records.append({'path': relative, 'origin': 'archive_documentation', 'transform': 'new_document'})

def main():
    if DEST.exists() and any(DEST.iterdir()):
        raise SystemExit(f'Refusing to overwrite existing nonempty archive: {DEST}')
    DEST.mkdir(parents=True, exist_ok=True)

    inventory = (ROOT / 'reports/project_summary_20261105_inventory.md').read_text(encoding='utf-8')
    inventory = inventory.replace('# 梯度通信工程项目阶段结果与数据位置清单', '# 梯度通信工程项目数据位置索引', 1)
    inventory = inventory.replace('截至本清单编写时，README 顶部仍有“epsilon 尚未执行”的历史描述，应由项目汇总维护者更新；六个结果目录的报告和摘要保持原样。不得把早期计划中的 G2b–G4、独立输入确认或训练验证写成已完成工作。', '项目 README 和总报告已更新为最新诊断状态；六个封存结果目录的报告和摘要保持原样。早期计划中的 G2b–G4、独立输入确认或训练验证均未完成。')
    inventory += '\n\n## GitHub 精简归档中的数据\n\n精简归档实际包含的文件及原始位置由根目录 `ARCHIVE_MANIFEST.json` 逐项列出；六轮结果的完整本机文件目录、大小和是否收录，由 `LOCAL_DATA_CATALOG.json` 列出。`local_only` 文件没有上传，其本机路径不是公共下载链接。历史报告内指向这些文件的相对链接仅在完整本机归档中可用。\n\n最新诊断的 `metrics_raw.jsonl` 在仓库内无损压缩为 `metrics_raw.jsonl.gz`；逐项数据不改变，原始文件 SHA256 保存在归档清单。仓库的 `tools/verify_epsilon_summary.py` 支持直接读取该 gzip 文件，只复算记录层面的统计，不调用 GPU、不读取原始张量。\n'
    (ROOT / 'DATA_INDEX.md').write_text(inventory, encoding='utf-8')

    for name in ('PROJECT_SUMMARY.md', 'DATA_INDEX.md', 'README.md', 'TECHNICAL_PLAN.md', 'DELIVERY_FORMAT.md'):
        copy(ROOT / name)
    for folder in ('src', 'scripts', 'configs', 'tests', 'plans', 'reports', 'reviews', 'handoff_templates'):
        for p in sorted((ROOT / folder).rglob('*')):
            if not p.is_file() or set(p.parts) & SKIP_PARTS or p.suffix.lower() not in TEXT_EXTS:
                continue
            if p.name == 'project_summary_20261105_inventory.md':
                continue
            if p.stat().st_size <= 2_000_000:
                copy(p)
    upstream = ROOT / 'upstream/bounded_discovery_v1/dynamiq'
    for p in sorted(upstream.rglob('*')):
        if p.is_file() and not (set(p.parts) & SKIP_PARTS) and (p.suffix.lower() in TEXT_EXTS or p.name == 'LICENSE'):
            copy(p)

    selected_names = {
        'summary.json', 'environment.json', 'analysis.json', 'artifact_integrity.json',
        'integrity_manifest.json', 'codec_contract.json', 'ring_contract.json',
        'experiment_contract.json', 'source_manifest.json', 'input_manifest.json',
        'input_randomness_manifest.json', 'fixtures_manifest.json', 'cpu_reference_manifest.json',
        'benchmark_config.json', 'benchmark_summary.jsonl', 'metrics_summary.jsonl',
        'paired_summary.json', 'region_manifest.json', 'scale_precheck.json',
        'baseline_qualification.json', 'packet_budget.json', 'pilot_protocol.json',
        'threshold_manifest.json', 'correctness_case_manifest.json', 'issues.jsonl',
    }
    catalog = []
    for stage, run in RUNS:
        base = ROOT / 'results' / run
        for p in sorted(base.iterdir()):
            if p.is_file() and (p.name in selected_names or p.suffix.lower() == '.md') and p.stat().st_size <= 2_000_000:
                copy(p)
        for p in sorted((base / 'figures').glob('*')):
            if p.is_file() and p.suffix.lower() in {'.png', '.svg'}:
                copy(p)
        for name in ('independent_effect_audit.json', 'build_pair_audit.json', 'source_path_qualification.json', 'recompute_verification.json', 'official_contract_review.json'):
            p = base / 'logs' / name
            if p.is_file() and p.stat().st_size <= 300_000:
                copy(p)
        for p in sorted(base.rglob('*')):
            if not p.is_file():
                continue
            rel = p.relative_to(ROOT).as_posix()
            catalog.append({'stage': stage, 'run': run, 'relative_path': rel, 'local_path': p.as_posix(), 'bytes': p.stat().st_size, 'availability': 'packaged' if (DEST / rel).exists() else 'local_only'})

    epsilon = ROOT / 'results/epsilon_diagnostic_v1/epsilon_diagnostic_v1_20261105_001'
    raw = epsilon / 'metrics_raw.jsonl'
    relative = raw.relative_to(ROOT).as_posix() + '.gz'
    target = DEST / relative
    target.write_bytes(gzip.compress(raw.read_bytes(), compresslevel=9, mtime=0))
    records.append({'path': relative, 'origin': raw.as_posix(), 'transform': 'gzip_lossless_mtime0', 'source_sha256': sha(raw), 'uncompressed_bytes': raw.stat().st_size})
    for item in catalog:
        if item['local_path'] == raw.as_posix():
            item['availability'] = 'packaged_gzip'
            item['archive_path'] = relative
    copy(epsilon / 'build/diagnostic_add0/source_intervention.diff', 'patches/epsilon_b8_scale_add.diff')
    copy(ROOT / 'publication_tools/verify_epsilon_summary.py', 'tools/verify_epsilon_summary.py')
    copy(Path(__file__).resolve(), 'tools/build_project_archive.py')
    write('LOCAL_DATA_CATALOG.json', json.dumps({'original_root': ROOT.as_posix(), 'note': 'Only rows marked packaged or packaged_gzip are included. Source data and compiled artifacts are local-only.', 'files': catalog}, ensure_ascii=False, indent=2) + '\n')
    write('.gitignore', '__pycache__/\n*.pyc\n.venv/\n.cache/\n.pytest_cache/\n.toolchains/\n*.pt\n*.npz\n*.pyd\n*.obj\n*.lib\n*.exp\n*.dll\n*.zip\n')
    write('.gitattributes', '# Preserve archived bytes and their SHA256 values.\n* -text\n')
    write('THIRD_PARTY.md', '''# 第三方来源与许可\n\n`upstream/bounded_discovery_v1/dynamiq/` 是 DynamiQ 固定来源副本，原仓库 https://github.com/CharlesHan24/DynamiQ ，commit `96da2ee480bbfd8e2049de3fbd8595cc9d7a30b8`。Apache-2.0 许可证原文保留在该目录的 LICENSE。\n\n`patches/epsilon_b8_scale_add.diff` 是针对冻结编译源的诊断改动，目标为 `cuda_hierarchical_mee.cu`；其编译宏分别取 `1e-7f` 和 `0.0f`，仅作用于 b8。它不是已经验证所有输入域的部署补丁。\n\n自有工程代码、结果与文档在本次归档中没有新增开源许可授权。CUDA、cuRAND、CUPTI、MSVC 和 Python 环境没有打包上传。CPU 参考的本机外部依赖见 DATA_INDEX.md。\n''')
    write('patches/README.md', '''# 固定尺度加数诊断补丁\n\n此 diff 记录已执行诊断中的唯一源码干预。来源与适用 commit 见根目录 THIRD_PARTY.md。\n\n两条件源码相同，通过 `DIAG_B8_SCALE_ADD=1e-7f` 或 `0.0f` 区分，其他位宽保持原加数；原下限和有限值保护保留。历史 Windows 构建还移除了两个未使用的 POSIX include，完整构建身份见本机原始交付。\n\n本补丁用于追溯实验，未验证完整通信协议及任意 BF16 输入，不应直接替换部署系统。\n''')
    for record in records:
        p = DEST / record['path']
        record['bytes'] = p.stat().st_size
        record['sha256'] = sha(p)
    manifest = {'schema_version': 'gradient_comm_project_archive_v1', 'as_of': '2026-11-05', 'source_root': ROOT.as_posix(), 'files': records, 'file_count_excluding_this_manifest': len(records), 'total_bytes_excluding_this_manifest': sum(r['bytes'] for r in records), 'excluded': ['raw tensors and packets', 'compiled binaries', 'toolchains and caches', 'full timing and message ledgers', 'all model/training datasets'], 'sealed_results_modified': False}
    (DEST / 'ARCHIVE_MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'destination': str(DEST), 'files': len(records)+1, 'bytes': manifest['total_bytes_excluding_this_manifest'], 'catalog_entries': len(catalog)}, ensure_ascii=False))

if __name__ == '__main__':
    main()
