"""Add official CUDA 11.8 cuRAND headers in a new isolated task directory."""
import hashlib, json, urllib.request, zipfile
from datetime import datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'results/bounded_discovery_v1/bounded_discovery_v1_20261104_001'
catalog = RUN / 'logs/cuda_redistrib_11.8.0.json'
metadata = json.loads(catalog.read_bytes())['libcurand']
record = metadata['windows-x86_64']
url = 'https://developer.download.nvidia.com/compute/cuda/redist/' + record['relative_path']
destination = ROOT / '.toolchains/bounded_discovery_v1/curand-10.3.0.86'
destination.mkdir(parents=True, exist_ok=True)
archive = destination / Path(record['relative_path']).name
if not archive.exists():
    with urllib.request.urlopen(url, timeout=120) as response, archive.open('wb') as output:
        while chunk := response.read(1024 * 1024): output.write(chunk)
assert archive.stat().st_size == int(record['size'])
assert hashlib.sha256(archive.read_bytes()).hexdigest() == record['sha256']
files = []
with zipfile.ZipFile(archive) as package:
    for name in package.namelist():
        relative = Path(*Path(name).parts[1:])
        if not relative.parts or name.endswith('/'):
            continue
        if relative.parts[0] == 'include' or 'license' in name.lower():
            target = (destination / relative).resolve()
            assert target.is_relative_to(destination.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(package.read(name))
            files.append(dict(path=str(target.relative_to(ROOT)), bytes=target.stat().st_size, sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
result = dict(captured_at=datetime.now().astimezone().isoformat(),url=url,metadata=record,
              catalog_sha256=hashlib.sha256(catalog.read_bytes()).hexdigest(), archive=str(archive.relative_to(ROOT)),
              archive_sha256=record['sha256'],package_version=metadata['version'],files=files,
              action='Extract headers and license only to new isolated directory; do not modify prior CUDA/toolchain/shared environment')
(RUN / 'logs/official_curand_dependency.json').write_text(json.dumps(result, indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status='verified',files=len(files),include_directory=str(destination/'include'))))
