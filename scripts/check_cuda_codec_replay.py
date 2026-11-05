"""Future-run replay shim: select the requested run's CUDA artifact explicitly.

This entrypoint was added after formal G1b qualification. It is not claimed as
having executed the formal matrix; --help is a CPU-only interface check.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import check_cuda_codec

def main():
    if '--help' in sys.argv[1:] or '-h' in sys.argv[1:]:
        return check_cuda_codec.main()
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--run-dir',required=True)
    args,_=parser.parse_known_args()
    run=(ROOT/args.run_dir).resolve()
    if not run.is_relative_to(ROOT/'results/gpu_codec_v1'):
        raise ValueError('Replay requires a run under project results/gpu_codec_v1')
    from gradient_comm_lab import cuda_codec
    cuda_codec.set_build_manifest(run/'logs/cuda_build_manifest.json')
    return check_cuda_codec.main()

if __name__=='__main__':sys.exit(main())
