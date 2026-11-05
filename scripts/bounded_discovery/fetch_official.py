"""Fetch immutable official Git blobs and verify Git SHA1 plus local SHA256."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone,timedelta
import hashlib,json,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'results/bounded_discovery_v1/bounded_discovery_v1_20261104_001'
DEST=ROOT/'upstream/bounded_discovery_v1/dynamiq'
COMMIT='96da2ee480bbfd8e2049de3fbd8595cc9d7a30b8'
tree=json.loads((RUN/'logs/official_git_tree.json').read_bytes())
selected=[r for r in tree['tree'] if r['type']=='blob' and (r['path'].startswith('cuda_kernels/') or r['path'] in ('LICENSE','README.md','ENVIRONMENT_REQUIREMENTS.md','requirements.txt') or r['path'].startswith('testbed_evaluation/compression/') or r['path'] in ('testbed_evaluation/new_comm_hooks/comm_hooks.py','testbed_evaluation/build_eden_utils_llm.zsh'))]
def fetch(row):
 p=DEST/row['path'];p.parent.mkdir(parents=True,exist_ok=True)
 url=f'https://raw.githubusercontent.com/CharlesHan24/DynamiQ/{COMMIT}/'+row['path']
 if p.exists():data=p.read_bytes()
 else:
  for attempt in range(3):
   try:
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'gradient-comm-lab-d0'}),timeout=40) as response:data=response.read()
    break
   except Exception:
    if attempt==2:raise
    time.sleep(1)
  p.write_bytes(data)
 blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
 assert blob==row['sha'] and len(data)==row['size'],row['path']
 return dict(upstream_path=row['path'],path=str(p.relative_to(ROOT)).replace('\\','/'),url=url,bytes=len(data),github_blob_sha1=blob,sha256=hashlib.sha256(data).hexdigest(),git_blob_verified=True)
with ThreadPoolExecutor(max_workers=4) as pool:files=list(pool.map(fetch,selected))
record=dict(repository='https://github.com/CharlesHan24/DynamiQ',commit=COMMIT,retrieved_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
    mechanism='GitHub official Git tree API plus immutable raw blob URLs; git clone initially failed with Recv failure: Connection was reset',
    tree_sha=tree['sha'],tree_json_sha256=hashlib.sha256((RUN/'logs/official_git_tree.json').read_bytes()).hexdigest(),files=files,
    license='Root LICENSE is Apache-2.0; inherited VMware files carry BSD-3-Clause SPDX headers and setup.py has BSD metadata. All originals and notices preserved.',scope='Selected original official sources; not a complete Git working tree')
(RUN/'logs/official_source_manifest.json').write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps({'status':'verified','files':len(files),'bytes':sum(r['bytes'] for r in files),'commit':COMMIT}))
