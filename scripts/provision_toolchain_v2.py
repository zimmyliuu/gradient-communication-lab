"""Download official CUDA/MSVC components into this project; never install globally.

Requires previously captured official NVIDIA repodata and Microsoft VS catalog.
The plan is written before downloading. Every binary archive is hash checked before
safe, bounded extraction. No existing Conda environment or VS instance is modified.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import urllib.request
import urllib.parse
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_CAP = 1024**3
EXTRACT_CAP = 4 * 1024**3


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def selected_packages(supplement_store=False, supplement_utils=False):
    metadata = ROOT / '.cache/toolchain_metadata'
    nvidia = json.loads((metadata / 'nvidia_win64.json').read_bytes())
    names = {'cuda-nvcc', 'cuda-cudart', 'cuda-cudart-dev', 'cuda-cccl', 'cuda-profiler-api'}
    selected = []
    for filename, package in nvidia['packages'].items():
        if package['name'] in names:
            selected.append(dict(id=package['name'], version=package['version'], filename=filename,
                url='https://conda.anaconda.org/nvidia/label/cuda-11.8.0/win-64/' + filename,
                size=package['size'], md5=package.get('md5'), sha256=package.get('sha256'),
                kind='tar', target='.toolchains/cuda-11.8', dependencies=package.get('depends'),
                reason='CUDA 11.8 compiler and runtime development files matching torch cu118'))
    catalog = json.loads((metadata / 'vs17_catalog.json').read_bytes())
    ids = {'Microsoft.VC.14.38.17.8.Tools.HostX64.TargetX64.base',
           'Microsoft.VC.14.38.17.8.Tools.HostX64.TargetX64.Res.base',
           'Microsoft.VC.14.38.17.8.CRT.Headers.base',
           'Microsoft.VC.14.38.17.8.CRT.x64.Desktop.base'}
    for package in catalog['packages']:
        if package['id'] in ids and package.get('language') in (None, 'en-US'):
            for p in package['payloads']:
                selected.append(dict(id=package['id'], version=package['version'], filename=p['fileName'],
                    url=p['url'], size=p['size'], sha256=p['sha256'], md5=None, kind='zip',
                    target='.toolchains/msvc-14.38', dependencies=package.get('dependencies'),
                    language=package.get('language'), advertised_signer=p.get('signer'),
                    reason='MSVC 19.38 x64, CRT and English resources within CUDA 11.8 supported 193x family'))
    if supplement_store or supplement_utils:
        ids = {'Microsoft.VC.14.38.17.8.CRT.x64.Store.base'} if supplement_store else {'Microsoft.VisualCpp.Tools.Common.Utils', 'Microsoft.VisualCpp.Tools.Core.x86'}
        reason = ('Contains MSVC shared CRT import libraries required by PyTorch /MD; Desktop package has static libraries only.' if supplement_store else
                  'Official VS 2022 auxiliary vcvars layout for nvcc host toolset recognition; compile uses --use-local-env with pinned MSVC 19.38 and SDK.')
        result = []
        for package in catalog['packages']:
            if package['id'] not in ids:
                continue
            for p in package['payloads']:
                result.append(dict(id=package['id'], version=package['version'], filename=p['fileName'], url=p['url'],
                    size=p['size'], sha256=p['sha256'], md5=None, kind='zip', target='.toolchains/msvc-14.38', reason=reason))
        return result
    if len(selected) != 9:
        raise RuntimeError(f'Expected 9 component packages, found {len(selected)}')
    return selected


def download(package):
    path = ROOT / '.cache/toolchain_downloads' / package['filename']
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        partial = path.with_name(path.name + '.partial')
        with urllib.request.urlopen(package['url'], timeout=60) as response, partial.open('wb') as out:
            if urllib.parse.urlparse(response.url).hostname not in ('conda.anaconda.org', 'download.visualstudio.microsoft.com'):
                raise ValueError('Unexpected download redirect')
            size = 0
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                out.write(block)
                size += len(block)
                if size > max(package['size'] * 1.1, package['size'] + 1024 * 1024):
                    raise RuntimeError('Archive exceeds bounded advertised size tolerance')
        partial.rename(path)
    actual = {'size': path.stat().st_size}
    for algorithm in ('sha256', 'md5'):
        h = hashlib.new(algorithm)
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(block)
        actual[algorithm] = h.hexdigest()
        if package.get(algorithm) and actual[algorithm] != package[algorithm]:
            raise RuntimeError(f'{algorithm} mismatch for {path.name}')
    actual['size_matches_metadata'] = actual['size'] == package['size']
    if not actual['size_matches_metadata'] and not package.get('sha256'):
        raise RuntimeError(f'Size mismatch without strong expected hash for {path.name}')
    print('Verified ' + path.name, flush=True)
    return dict(package, archive=str(path.relative_to(ROOT)), verified=actual)


def destination(base, name):
    parts = PurePosixPath(name.replace('\\', '/')).parts
    if not parts or '..' in parts or ':' in name or name.startswith(('/', '\\')):
        raise ValueError('Unsafe archive path: ' + name)
    path = (base / Path(*parts)).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError('Archive path escapes destination')
    return path


def extract(package, budget):
    path = ROOT / package['archive']
    base = ROOT / package['target']
    extracted = 0
    entries = []
    if package['kind'] == 'tar':
        with tarfile.open(path, 'r:bz2') as archive:
            for entry in archive.getmembers():
                if not entry.isfile():
                    if entry.issym() or entry.islnk():
                        raise ValueError('Links are not permitted')
                    continue
                # Package metadata is retained in archives, not merged across packages.
                if entry.name.startswith('info/'):
                    continue
                entries.append((entry, entry.name, entry.size))
            extracted = sum(size for _, _, size in entries)
            if extracted > budget:
                raise ValueError('Extraction budget exceeded')
            for entry, name, _ in entries:
                target = destination(base, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(entry) as source, target.open('wb') as out:
                    shutil.copyfileobj(source, out)
    else:
        with zipfile.ZipFile(path) as archive:
            for entry in archive.infolist():
                name = entry.filename.replace('\\', '/')
                if entry.is_dir() or not name.startswith('Contents/'):
                    continue
                name = name[len('Contents/'):]
                entries.append((entry, name, entry.file_size))
            extracted = sum(size for _, _, size in entries)
            if extracted > budget:
                raise ValueError('Extraction budget exceeded')
            for entry, name, _ in entries:
                target = destination(base, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(entry) as source, target.open('wb') as out:
                    shutil.copyfileobj(source, out)
    package['extracted_bytes'] = extracted
    package['extracted_files'] = len(entries)
    print('Extracted ' + path.name + f' ({extracted} bytes)', flush=True)
    return extracted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--supplement-store', action='store_true')
    parser.add_argument('--supplement-utils', action='store_true')
    args = parser.parse_args()
    run = (ROOT / args.run_dir).resolve()
    if not run.is_relative_to(ROOT / 'results/gpu_codec_v1'):
        raise ValueError('Run must remain in this project')
    packages = selected_packages(args.supplement_store, args.supplement_utils)
    plan = dict(created_at=datetime.now().astimezone().isoformat(), packages=packages,
        download_bytes=sum(p['size'] for p in packages), download_cap_bytes=DOWNLOAD_CAP,
        extracted_cap_bytes=EXTRACT_CAP, estimated_extracted_bytes=600_000_000,
        metadata_paths=['.cache/toolchain_metadata/nvidia_win64.json', '.cache/toolchain_metadata/vs17_catalog.json'],
        existing_sdk_read_only='C:/Program Files (x86)/Windows Kits/10/10.0.26100.0',
        excluded_dependencies_note='MSBuild props/servicing registration are not needed by direct cl/ninja; no global installer invoked.',
        global_installation=False, modify_old_venv=False, driver_change=False)
    if plan['download_bytes'] > DOWNLOAD_CAP:
        raise ValueError('Download budget exceeded')
    suffix = '_store' if args.supplement_store else ('_utils' if args.supplement_utils else '')
    planpath = run / f'logs/toolchain_download_plan{suffix}.json'
    if not planpath.exists():
        dump(planpath, plan)
    print(json.dumps({'planned_download_bytes': plan['download_bytes'], 'packages': len(packages)}), flush=True)
    if not args.download:
        return
    resultpath = run / f'logs/toolchain_provision{suffix}.json'
    if resultpath.exists():
        raise FileExistsError('Do not overwrite a prior provisioning result')
    with ThreadPoolExecutor(max_workers=3) as pool:
        verified = list(pool.map(download, packages))
    total = 0
    for package in verified:
        total += extract(package, EXTRACT_CAP - total)
    dump(resultpath, dict(created_at=datetime.now().astimezone().isoformat(), status='extracted_not_yet_compiled',
        packages=verified, total_download_bytes=sum(p['verified']['size'] for p in verified), total_extracted_bytes=total,
        global_installation=False, modify_old_venv=False, driver_change=False))


if __name__ == '__main__':
    main()
