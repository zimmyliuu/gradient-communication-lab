# Read-only reuse of the verified interpreter; all caches belong to this project.
$ErrorActionPreference = 'Stop'
$GradientLabRoot = Split-Path -Parent $PSScriptRoot
$GradientLabPython = 'E:\dacomp\.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $GradientLabPython)) { throw 'Verified interpreter is missing; no implicit fallback.' }
$env:PYTHONNOUSERSITE = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = ''
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:OMP_NUM_THREADS = '4'
$env:PIP_CACHE_DIR = Join-Path $GradientLabRoot '.cache\pip'
$env:TORCH_HOME = Join-Path $GradientLabRoot '.cache\torch'
$env:TORCH_EXTENSIONS_DIR = Join-Path $GradientLabRoot '.cache\torch_extensions'
$env:CUDA_CACHE_PATH = Join-Path $GradientLabRoot '.cache\cuda'
$env:MPLCONFIGDIR = Join-Path $GradientLabRoot '.cache\matplotlib'
$env:XDG_CACHE_HOME = Join-Path $GradientLabRoot '.cache\xdg'
$env:PYTHONUSERBASE = Join-Path $GradientLabRoot '.cache\python-user'
$env:TEMP = Join-Path $GradientLabRoot '.cache\tmp'
$env:TMP = $env:TEMP
foreach ($GradientLabCache in @($env:PIP_CACHE_DIR,$env:TORCH_HOME,$env:TORCH_EXTENSIONS_DIR,$env:CUDA_CACHE_PATH,$env:MPLCONFIGDIR,$env:XDG_CACHE_HOME,$env:PYTHONUSERBASE,$env:TEMP)) {
    New-Item -ItemType Directory -Path $GradientLabCache -Force | Out-Null
}
if ($MyInvocation.ExpectingInput) { $input | & $GradientLabPython @args } else { & $GradientLabPython @args }
exit $LASTEXITCODE
