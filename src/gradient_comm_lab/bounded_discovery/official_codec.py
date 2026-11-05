"""Thin, allocating adapter around the pinned official DynamiQ CUDA entrypoints.

The upstream arithmetic and RNG interpretation are unchanged. This adapter only
validates the selected single-GPU BF16 domain, pads, owns byte packets, passes the
current stream, and crops output. It is not the local G1 codec.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[3]
COMMIT = '96da2ee480bbfd8e2049de3fbd8595cc9d7a30b8'
DEFAULT_MANIFEST = ROOT / 'results/bounded_discovery_v1/bounded_discovery_v1_20261104_001/logs/official_build_a03.json'
_manifest_path = DEFAULT_MANIFEST
_module = None
_identity = None
_dll_handles = []

def set_build_manifest(path):
    global _manifest_path
    path = Path(path).resolve()
    if _module is not None and path != _manifest_path:
        raise RuntimeError('Cannot replace the official module identity after loading')
    _manifest_path = path

def extension():
    global _module, _identity
    if _module is None:
        record = json.loads(_manifest_path.read_bytes())
        if record['status'] != 'compiled_and_loaded' or record['official_commit'] != COMMIT:
            raise ValueError('Official build manifest has no qualified pinned module')
        binary = Path(record['module_path'])
        actual_hash = hashlib.sha256(binary.read_bytes()).hexdigest()
        if actual_hash != record['module_sha256']:
            raise ValueError('Official module SHA256 differs from build manifest')
        if os.name == 'nt':
            for folder in (ROOT / '.toolchains/cuda-11.8/bin', Path(torch.__file__).parent / 'lib'):
                _dll_handles.append(os.add_dll_directory(str(folder)))
        spec = importlib.util.spec_from_file_location(record['module_name'], binary)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _module = module
        sys.modules[record['module_name']] = module
        _identity = dict(commit=COMMIT, module_path=str(binary), module_sha256=actual_hash,
                         build_manifest=str(_manifest_path), build_manifest_sha256=hashlib.sha256(_manifest_path.read_bytes()).hexdigest())
    return _module

def build_identity():
    extension()
    return dict(_identity)

def padded_numel(n):
    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        raise ValueError('Official selected codec requires a positive logical length')
    return ((n + 255) // 256) * 256

def required_bytes(n, bits):
    if bits not in (4, 8):
        raise ValueError('Official selected codec supports stored bits 4 or 8')
    if n <= 0 or n % 256:
        raise ValueError('Official selected native length must be positive and divisible by 256')
    return (n // 256) * (16 * (16 * bits // 8 + 1) + 2)

def make_random_pool(n, seed, device='cuda'):
    """Exactly the official functional-test RNG recipe (including BF16 1.0)."""
    generator = torch.Generator(device='cpu').manual_seed(int(seed))
    return torch.rand(n, generator=generator, dtype=torch.float32).to(torch.bfloat16).to(device)

@dataclass(frozen=True)
class OfficialPacket:
    data: torch.Tensor
    logical_numel: int
    padded_numel: int
    bits: int
    chunk_size: int = 16
    strategy: int = 3
    dtype: torch.dtype = torch.bfloat16

    @property
    def payload(self): return self.data

    def to_bytes(self):
        return self.data.detach().cpu().contiguous().numpy().tobytes()

    @classmethod
    def from_bytes(cls, wire, *, logical_numel, bits, device='cuda'):
        n = padded_numel(logical_numel)
        if len(wire) != required_bytes(n, bits):
            raise ValueError('Official serialized byte length differs from static packet configuration')
        data = torch.frombuffer(bytearray(wire), dtype=torch.uint8).clone().to(device)
        return cls(data, logical_numel, n, bits)

    def accounting(self):
        blocks = self.padded_numel // 256
        return dict(payload_bytes=self.padded_numel * self.bits // 8,
                    scale_bytes=blocks * 18, header_bytes=0, alignment_bytes=0,
                    packet_total_bytes=self.data.numel() * self.data.element_size(),
                    per_input_auxiliary_bytes=0, setup_or_persistent_state_bytes=9288,
                    metadata='bits, dtype, logical/native lengths, chunk=16, strategy=3 are agreed experiment configuration; no serialized header',
                    persistent_state='Official per-device FP16 device LUT allocation 4608 B plus static device lookup_table[9] 4680 B; input-independent and initialized from code',
                    byte_measurement_method='uint8 tensor numel*element_size; payload and scales partition the same interleaved storage')

def _input_ok(value):
    if not isinstance(value, torch.Tensor) or value.ndim != 1 or not value.is_contiguous():
        raise ValueError('Official input must be a contiguous 1D tensor')
    if value.dtype != torch.bfloat16 or value.device.type != 'cuda':
        raise ValueError('Official selected input must be CUDA BF16')
    padded_numel(value.numel())
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError('Official adapter requires finite input')

def encode(value, bits, *, seed=None, rand_pool=None):
    _input_ok(value)
    n = padded_numel(value.numel())
    byte_count = required_bytes(n, bits)
    if (seed is None) == (rand_pool is None):
        raise ValueError('Provide exactly one official RNG seed or random pool')
    if rand_pool is None:
        rand_pool = make_random_pool(n, seed, value.device)
    if not isinstance(rand_pool, torch.Tensor) or rand_pool.device != value.device or rand_pool.dtype != value.dtype or rand_pool.ndim != 1 or not rand_pool.is_contiguous() or rand_pool.numel() != n:
        raise ValueError('Official random pool must match native length, dtype and CUDA device')
    if not bool((torch.isfinite(rand_pool) & (rand_pool >= 0) & (rand_pool <= 1)).all().item()):
        raise ValueError('Official random pool must be finite within [0,1] after BF16 conversion')
    source = torch.zeros(n, dtype=value.dtype, device=value.device)
    source[:value.numel()].copy_(value)
    encoded = torch.empty(byte_count, dtype=torch.uint8, device=value.device)
    with torch.cuda.device(value.device):
        stream = torch.cuda.current_stream(value.device).cuda_stream
        extension().scaling_compress(source, encoded, rand_pool, n, bits, 16, stream, 3)
    return OfficialPacket(encoded, value.numel(), n, bits)

def decode(packet, *, padded=False):
    if not isinstance(packet, OfficialPacket):
        raise ValueError('Expected an OfficialPacket')
    data = packet.data
    if packet.dtype != torch.bfloat16 or packet.chunk_size != 16 or packet.strategy != 3:
        raise ValueError('Packet configuration differs from the selected official codec')
    if packet.padded_numel != padded_numel(packet.logical_numel):
        raise ValueError('Packet logical and native lengths disagree')
    if data.dtype != torch.uint8 or data.device.type != 'cuda' or data.ndim != 1 or not data.is_contiguous() or data.numel() != required_bytes(packet.padded_numel, packet.bits):
        raise ValueError('Official packet storage metadata is invalid')
    output = torch.empty(packet.padded_numel, dtype=packet.dtype, device=data.device)
    with torch.cuda.device(data.device):
        stream = torch.cuda.current_stream(data.device).cuda_stream
        extension().scaling_decompress(data, output, packet.padded_numel, packet.bits, 16, stream, 3)
    return output if padded else output[:packet.logical_numel].clone()

def roundtrip(value, bits, *, seed=None, rand_pool=None):
    packet = encode(value, bits, seed=seed, rand_pool=rand_pool)
    return packet, decode(packet)
