"""Read-only G0 discovery; no package installation or global environment changes."""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import winreg

ROOT = Path(__file__).resolve().parents[1]


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    args=parser.parse_args()
    run=(ROOT/args.run_dir).resolve()
    if not run.is_relative_to(ROOT/'results/gpu_codec_v1'):
        raise ValueError('G0 output must belong to this project run')
    logs=run/'logs'; logs.mkdir(parents=True,exist_ok=True)
    commands=[]
    def command(argv, timeout=60, encoding='utf-8'):
        started=time.perf_counter()
        try:
            result=subprocess.run(argv,capture_output=True,text=True,encoding=encoding,errors='replace',timeout=timeout)
            entry={'command':argv,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
        except (subprocess.TimeoutExpired, OSError) as exc:
            entry={'command':argv,'returncode':None,'stdout':None,'stderr':str(exc),'failure':type(exc).__name__}
        entry['wall_seconds']=time.perf_counter()-started
        commands.append(entry)
        return entry
    path_tools={name:shutil.which(name) for name in ('nvcc','cl','ninja','cmake','nvidia-smi','conda','rg')}
    vswhere=Path('C:/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe')
    vs_installations=[]
    if vswhere.is_file():
        result=command([str(vswhere),'-all','-products','*','-format','json','-utf8'])
        if result['returncode']==0:
            vs_installations=json.loads(result['stdout'])
    compiler_paths=[]; ninja_paths=[]; cmake_paths=[]
    for item in vs_installations:
        base=Path(item['installationPath'])
        compiler_paths.extend(str(p) for p in base.glob('VC/Tools/MSVC/*/bin/Hostx64/x64/cl.exe'))
        ninja_paths.extend(str(p) for p in base.glob('Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe'))
        cmake_paths.extend(str(p) for p in base.glob('Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe'))
    registry_toolkit=[]; registry_products=[]
    for view in (winreg.KEY_WOW64_64KEY, winreg.KEY_WOW64_32KEY):
        keyname=r'SOFTWARE\NVIDIA Corporation\GPU Computing Toolkit\CUDA'
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,keyname,0,winreg.KEY_READ|view) as key:
                values={}; index=0
                while True:
                    try:
                        name,value,_=winreg.EnumValue(key,index); values[name]=value; index+=1
                    except OSError: break
                registry_toolkit.append({'view':view,'key':keyname,'values':values,'exists':True})
        except FileNotFoundError:
            registry_toolkit.append({'view':view,'key':keyname,'exists':False})
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall',0,winreg.KEY_READ|view) as key:
            index=0
            while True:
                try: name=winreg.EnumKey(key,index); index+=1
                except OSError: break
                try:
                    with winreg.OpenKey(key,name) as sub:
                        display=winreg.QueryValueEx(sub,'DisplayName')[0]
                        if not any(v in display.lower() for v in ('cuda','visual studio','build tools','nvidia')): continue
                        item={'display_name':display,'registry_view':view}
                        for field in ('DisplayVersion','InstallLocation'):
                            try: item[field]=winreg.QueryValueEx(sub,field)[0]
                            except FileNotFoundError: item[field]=None
                        registry_products.append(item)
                except OSError: pass
    roots=[Path(x) for x in ('C:/Program Files','C:/Program Files (x86)',
        'C:/Users/admin/AppData/Local/Programs','C:/Users/admin/scoop',
        'C:/CUDA','D:/CUDA','E:/CUDA','C:/BuildTools','E:/visual','A:/devcpp')]
    conda=path_tools['conda']
    if conda: roots.append(Path(conda).resolve().parents[1])
    roots.append(Path('C:/Users/admin/.conda'))
    for key,value in os.environ.items():
        if key.startswith('CUDA_PATH') or key=='CUDA_HOME': roots.append(Path(value))
    roots=list(dict.fromkeys(roots)); discoveries=[]
    rg=path_tools['rg']
    if rg:
        for base in roots:
            if base.is_dir():
                result=command([rg,'--files','--hidden','--no-ignore','-g','nvcc.exe','-g','cl.exe',
                    '-g','ninja.exe','-g','cmake.exe',str(base)], timeout=90)
                if result['stdout']:
                    discoveries.extend(result['stdout'].splitlines())
    nvcc_paths=list(dict.fromkeys(([path_tools['nvcc']] if path_tools['nvcc'] else [])+
        [p for p in discoveries if Path(p).name.lower()=='nvcc.exe']))
    host=path_tools['cl'] or (compiler_paths[-1] if compiler_paths else None)
    ninja=path_tools['ninja'] or (ninja_paths[0] if ninja_paths else None)
    cmake=path_tools['cmake'] or (cmake_paths[0] if cmake_paths else None)
    versions={}
    for key,path,arg in (('nvcc',nvcc_paths[0] if nvcc_paths else None,'--version'),('host',host,None),('ninja',ninja,'--version'),('cmake',cmake,'--version')):
        versions[key]=command([path]+([arg] if arg else [])) if path else None
    smi=path_tools['nvidia-smi']
    gpu_query=command([smi,'--query-gpu=name,driver_version,memory.total,memory.free,compute_cap,power.limit,power.draw,clocks.current.graphics,clocks.current.memory,utilization.gpu','--format=csv,noheader,nounits']) if smi else None
    processes=command([smi,'--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader']) if smi else None
    if smi: command([smi])
    import torch
    from torch.utils.cpp_extension import CUDA_HOME
    cuda=torch.cuda.is_available()
    properties=torch.cuda.get_device_properties(0) if cuda else None
    numbers=gpu_query['stdout'].strip().split(', ') if gpu_query and gpu_query['returncode']==0 else []
    def number(index):
        try: return float(numbers[index])
        except (ValueError,IndexError): return None
    env=json.loads((ROOT/'handoff_templates/environment.json').read_text(encoding='utf-8'))
    env.update(template_only=False,run_id=run.name,captured_at=datetime.now().astimezone().isoformat(),
        os={'platform':platform.platform(),'release':platform.release(),'version':platform.version(),'machine':platform.machine()},
        python_executable=sys.executable,python_version=platform.python_version(),pytorch_version=torch.__version__,
        pytorch_cuda_runtime=torch.version.cuda,driver_version=numbers[1] if len(numbers)>1 else None,
        nccl_available=torch.distributed.is_nccl_available(),cuda_available=cuda,
        bf16_available=torch.cuda.is_bf16_supported() if cuda else None)
    env['gpu'].update(name=properties.name if properties else None,memory_total_bytes=properties.total_memory if properties else None,
        compute_capability=list(torch.cuda.get_device_capability(0)) if cuda else None,power_limit_watts=number(5),
        clock_settings={'locked_clocks':None,'current_graphics_mhz':number(7),'current_memory_mhz':number(8),'note':'Read-only instantaneous clocks; no clock or power settings changed.'},
        other_process_load={'utilization_gpu_percent':number(9),'memory_free_mib':number(3),'power_draw_watts':number(6),
            'query':processes,'note':'WDDM reports some process memory as N/A and some names inaccessible; no processes stopped.'})
    env['compiler'].update(nvcc_path=nvcc_paths[0] if nvcc_paths else None,nvcc_version=versions['nvcc'],
        host_compiler_path=host,host_compiler_version=versions['host'],flags=[],
        minimal_cuda_extension_test='not_run',build_log=None,ninja_path=ninja,ninja_version=versions['ninja'],cmake_path=cmake,cmake_version=versions['cmake'],
        torch_detected_cuda_home=CUDA_HOME,host_toolkit_compatibility='not_verified_without_actual_toolkit_compile')
    env['dependency_lock_path']='requirements.txt'
    env['missing_tools']=[x for x,available in (('CUDA Toolkit nvcc',nvcc_paths),('MSVC x64 cl',host),('Ninja executable',ninja)) if not available]
    env['notes']=['Shared interpreter and packages were only read; no installs/upgrades performed.',
        'Missing CUDA compiler means not discovered in PATH, registry and enumerated installation roots; this is not a claim about every filesystem path.',
        'NVIDIA driver registry entries can be stale; driver_version is the live nvidia-smi query.',
        'Current GPU load is not a benchmark condition or a performance result.']
    discovery={'captured_at':env['captured_at'],'path_tools':path_tools,'cuda_environment':{k:v for k,v in os.environ.items() if k.startswith('CUDA_PATH') or k=='CUDA_HOME'},
        'vswhere':str(vswhere),'vs_installations':vs_installations,'msvc_candidates':compiler_paths,'ninja_candidates':ninja_paths,'cmake_candidates':cmake_paths,
        'cuda_registry':registry_toolkit,'installed_relevant_products':registry_products,'search_roots':[{'path':str(p),'exists':p.exists()} for p in roots],
        'discovered_tool_files':discoveries,'nvcc_candidates':nvcc_paths,'torch_cuda_home':CUDA_HOME,
        'python_ninja_module_available':importlib.util.find_spec('ninja') is not None,'python_cmake_module_available':importlib.util.find_spec('cmake') is not None,
        'commands':commands}
    dump(logs/'environment_discovery.json',discovery); dump(run/'environment.json',env)
    print(json.dumps({'run_id':run.name,'environment':'environment.json','log':'logs/environment_discovery.json','nvcc':env['compiler']['nvcc_path'],'cl':host,'ninja':ninja,'cuda_runtime':torch.version.cuda,'gpu':env['gpu']['name'],'missing_tools':env['missing_tools']}))


if __name__=='__main__': main()
