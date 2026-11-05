"""Simple CPU reference for bounded_discovery_v1 PLAN section 4.2.

This is an audit implementation, not a performance baseline.  A packet contains
every input-dependent decode quantity.  Its 16-byte header is actually transmitted
and counted: no implicit free length, group size, or code-width metadata is used.
Only the fixed format/version, little-endian interpretation and arithmetic rules
are shared protocol knowledge.  The encoder RNG state is not needed by decoding.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import struct
from typing import Any

import numpy as np
import torch


FORMAT_ID = "bounded_discovery_uniform_v1"
HEADER = struct.Struct("<4sBBHQ")  # magic, version, q, group_size, logical_numel
MAGIC = b"BDU1"
VERSION = 1
GROUP_SIZES = (16, 128)
CODE_WIDTHS = tuple(range(2, 11))


def _configuration(logical_numel: int, q: int, g: int) -> tuple[int, int]:
    if type(q) is not int or q not in CODE_WIDTHS:
        raise ValueError("UNIFORM_Q: expected an integer in [2,10]")
    if type(g) is not int or g not in GROUP_SIZES:
        raise ValueError("UNIFORM_GROUP: expected 16 or 128")
    if type(logical_numel) is not int or not 0 <= logical_numel < 2**64:
        raise ValueError("UNIFORM_LENGTH: expected an unsigned 64-bit logical length")
    padded = g * ((logical_numel + g - 1) // g)
    return padded, (1 << (q - 1)) - 1


def pack_unsigned(values: np.ndarray, q: int) -> bytes:
    """Pack integer values LSB-first, including the last partial byte if any.

    This generic storage helper permits the reserved value; the codec encoder
    never produces it and the decoder rejects it.  It makes malformed-packet
    tests possible without adding a second wire format.
    """
    _configuration(0, q, 16)
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in "ui":
        raise ValueError("UNIFORM_PACK: expected a one-dimensional integer array")
    if values.size and (np.any(values < 0) or np.any(values >= (1 << q))):
        raise ValueError("UNIFORM_PACK: unsigned value exceeds code width")
    unsigned = values.astype(np.uint16, copy=False)
    bits = ((unsigned[:, None] >> np.arange(q, dtype=np.uint16)) & 1).astype(np.uint8)
    return np.packbits(bits.reshape(-1), bitorder="little").tobytes()


def unpack_unsigned(payload: bytes, q: int, count: int) -> np.ndarray:
    """Independent byte-index/shift reader; does not use np.unpackbits."""
    _configuration(count, q, 16)
    if not isinstance(payload, bytes) or len(payload) != (count*q + 7)//8:
        raise ValueError("UNIFORM_PAYLOAD: incorrect packed payload length")
    data = np.frombuffer(payload, dtype=np.uint8)
    if count*q % 8 and data.size and int(data[-1]) >> (count*q % 8):
        raise ValueError("UNIFORM_PAYLOAD: nonzero unused high bits")
    codes = np.zeros(count, dtype=np.uint16)
    offsets = np.arange(count, dtype=np.int64)*q
    for bit in range(q):
        positions = offsets + bit
        selected = data[positions // 8].astype(np.uint16)
        codes |= ((selected >> (positions % 8)) & 1).astype(np.uint16) << bit
    return codes


def parse_header(header: bytes) -> dict[str, int]:
    if not isinstance(header, bytes) or len(header) != HEADER.size:
        raise ValueError("UNIFORM_HEADER: expected exactly 16 bytes")
    magic, version, q, g, logical = HEADER.unpack(header)
    if magic != MAGIC or version != VERSION:
        raise ValueError("UNIFORM_HEADER: unsupported magic or version")
    padded, maximum = _configuration(logical, q, g)
    return dict(logical_numel=logical, padded_numel=padded, q=q, group_size=g,
                maximum_integer=maximum, payload_bytes=(padded*q+7)//8,
                scale_bytes=4*(padded//g))


@dataclass(frozen=True)
class UniformPacket:
    """Wire components plus optional, explicitly non-transmitted audit metadata."""
    header: bytes
    payload: bytes
    scales: bytes
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    def to_bytes(self) -> bytes:
        self.validate_components()
        return self.header + self.payload + self.scales

    def validate_components(self) -> dict[str, int]:
        configuration = parse_header(self.header)
        if not isinstance(self.payload, bytes) or len(self.payload) != configuration["payload_bytes"]:
            raise ValueError("UNIFORM_PAYLOAD: incorrect packed payload length")
        if not isinstance(self.scales, bytes) or len(self.scales) != configuration["scale_bytes"]:
            raise ValueError("UNIFORM_SCALES: incorrect FP32 scale byte length")
        return configuration

    @classmethod
    def from_bytes(cls, wire: bytes) -> "UniformPacket":
        if not isinstance(wire, bytes) or len(wire) < HEADER.size:
            raise ValueError("UNIFORM_PACKET: expected complete immutable packet bytes")
        header = wire[:HEADER.size]
        configuration = parse_header(header)
        boundary = HEADER.size + configuration["payload_bytes"]
        if len(wire) != boundary + configuration["scale_bytes"]:
            raise ValueError("UNIFORM_PACKET: wire byte length differs from header")
        return cls(header, wire[HEADER.size:boundary], wire[boundary:])

    def accounting(self) -> dict[str, Any]:
        configuration = self.validate_components()
        wire = self.to_bytes()
        logical = configuration["logical_numel"]
        configuration.update(payload_bytes=len(self.payload), scale_bytes=len(self.scales))
        return dict(format_id=FORMAT_ID, **configuration,
            stored_code_bits=configuration["q"], header_bytes=len(self.header), alignment_bytes=0,
            packet_total_bytes=len(wire),
            effective_bits_per_logical_element=8*len(wire)/logical if logical else None,
            packet_sha256=hashlib.sha256(wire).hexdigest(),
            per_input_auxiliary_bytes=0, setup_or_persistent_state_bytes=0,
            component_shapes_dtypes={
                "header": {"shape": [len(self.header)], "dtype": "uint8",
                           "fields": "little-endian <4sBBHQ: magic/version/q/g/logical_numel"},
                "payload": {"shape": [len(self.payload)], "dtype": "uint8",
                            "layout": "q-bit unsigned offset code u=k+M; LSB-first"},
                "scales": {"shape": [configuration["padded_numel"]//configuration["group_size"]],
                           "dtype": "float32", "byte_order": "little"}},
            shared_state="Only the fixed versioned wire specification; all per-packet q/g/D are in the counted header.",
            encoder_rng_state="Ephemeral local CPU generator; never required or transmitted for decoding.",
            byte_measurement_method="len of actual serialized header, packed payload and little-endian FP32 scales")


def _state_sha(generator: torch.Generator) -> str:
    return hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest()


@torch.no_grad()
def encode(x: torch.Tensor, q: int, g: int, seed: int) -> UniformPacket:
    """Encode finite, contiguous, one-dimensional CPU BF16 input.

    Each call owns a fresh CPU generator.  U is drawn in FP32 for every padded
    coordinate (including zero blocks), so calls do not share mutable RNG state.
    """
    if not isinstance(x, torch.Tensor) or x.device.type != "cpu":
        raise ValueError("UNIFORM_INPUT: expected a CPU tensor")
    if x.dtype != torch.bfloat16 or x.ndim != 1 or not x.is_contiguous():
        raise ValueError("UNIFORM_INPUT: expected contiguous one-dimensional BF16")
    padded, maximum = _configuration(x.numel(), q, g)
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("UNIFORM_SEED: expected an integer in [0,2^63)")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("UNIFORM_INPUT: nonfinite input")
    value = torch.zeros(padded, dtype=torch.float32, device="cpu")
    value[:x.numel()].copy_(x.to(torch.float32))
    blocks = value.reshape(-1, g)
    maxima = blocks.abs().amax(dim=1)
    scales = maxima / torch.tensor(maximum, dtype=torch.float32)
    upward_steps = torch.zeros(scales.shape, dtype=torch.int64)
    # FP64 products are exact here: FP32 scale mantissa plus at most nine
    # significant integer bits fits within the FP64 mantissa.
    uncovered = scales.to(torch.float64)*maximum < maxima.to(torch.float64)
    while bool(uncovered.any()):
        next_scale = torch.nextafter(scales, torch.full_like(scales, float("inf")))
        if not bool(((next_scale > scales) & torch.isfinite(next_scale))[uncovered].all()):
            raise ValueError("UNIFORM_SCALE: cannot advance finite FP32 scale to cover block maximum")
        scales = torch.where(uncovered, next_scale, scales)
        upward_steps += uncovered.to(torch.int64)
        uncovered = scales.to(torch.float64)*maximum < maxima.to(torch.float64)
    zero_blocks = maxima == 0
    scales = torch.where(zero_blocks, torch.zeros_like(scales), scales)
    denominator = torch.where(zero_blocks, torch.ones_like(scales), scales)
    normalized = blocks / denominator[:, None]
    lower = torch.floor(normalized)
    probability = normalized - lower
    if not bool(torch.isfinite(normalized).all() & ((probability >= 0) & (probability < 1)).all()):
        raise ValueError("UNIFORM_ARITHMETIC: invalid FP32 normalization or rounding probability")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    initial_state = _state_sha(generator)
    uniform = torch.rand(blocks.shape, dtype=torch.float32, device="cpu", generator=generator)
    integer = (lower + (uniform < probability).to(torch.float32)).clamp(-maximum, maximum).to(torch.int16)
    integer[zero_blocks] = 0
    unsigned = (integer.to(torch.int32) + maximum).reshape(-1).numpy().astype(np.uint16)
    payload = pack_unsigned(unsigned, q)
    scale_bytes = scales.numpy().astype("<f4", copy=False).tobytes()
    header = HEADER.pack(MAGIC, VERSION, q, g, x.numel())
    metadata = dict(format_id=FORMAT_ID, seed=seed, rng_mode="fresh torch.Generator(device=cpu) per packet",
        rng_initial_state_sha256=initial_state, rng_final_state_sha256=_state_sha(generator),
        rng_draw_count=padded, rng_draw_dtype="torch.float32", rng_interval="[0,1)",
        rng_scope="Independent seeds must be assigned to each input/worker/configuration/repeat by the experiment contract.",
        input_dtype="torch.bfloat16", arithmetic_profile="FP32 max/divide/normalize/floor/probability; FP64 scale coverage check",
        decode_profile="signed integer -> FP32 multiply by stored FP32 scale -> BF16 RNE -> crop logical length",
        scale_nextafter_steps_total=int(upward_steps.sum()),
        scale_nextafter_steps_max=int(upward_steps.max()) if upward_steps.numel() else 0,
        nonzero_blocks=int((~zero_blocks).sum()), zero_blocks=int(zero_blocks.sum()),
        reserved_unsigned_code=(1 << q)-1,
        audit_metadata_transmitted=False, audit_metadata_required_for_decode=False)
    packet = UniformPacket(header, payload, scale_bytes, metadata)
    packet.validate_components()
    return packet


@torch.no_grad()
def decode(packet: UniformPacket | bytes) -> torch.Tensor:
    """Decode from packet bytes only; never reads input, seed, or audit metadata."""
    if isinstance(packet, bytes):
        packet = UniformPacket.from_bytes(packet)
    if not isinstance(packet, UniformPacket):
        raise ValueError("UNIFORM_PACKET: expected UniformPacket or immutable wire bytes")
    configuration = packet.validate_components()
    q, g = configuration["q"], configuration["group_size"]
    padded, logical = configuration["padded_numel"], configuration["logical_numel"]
    maximum = configuration["maximum_integer"]
    unsigned = unpack_unsigned(packet.payload, q, padded)
    if np.any(unsigned == (1 << q)-1):
        raise ValueError("UNIFORM_CODE: reserved unsigned code encountered")
    integer = unsigned.astype(np.int16) - maximum
    scales_np = np.frombuffer(packet.scales, dtype="<f4").copy()
    if not np.isfinite(scales_np).all() or np.any(np.signbit(scales_np)):
        raise ValueError("UNIFORM_SCALE: expected finite nonnegative scales with positive zero")
    if np.any(integer[logical:] != 0):
        raise ValueError("UNIFORM_TAIL: padded positions must encode integer zero")
    grouped = integer.reshape(-1, g)
    if np.any(grouped[scales_np == 0] != 0):
        raise ValueError("UNIFORM_ZERO: a zero-scale block must encode integer zero")
    codes = torch.from_numpy(grouped.copy()).to(torch.float32)
    scales = torch.from_numpy(scales_np).to(torch.float32)
    decoded_fp32 = (codes * scales[:, None]).reshape(-1)
    output = decoded_fp32.to(torch.bfloat16)[:logical].clone()
    if not bool(torch.isfinite(output).all()):
        raise ValueError("UNIFORM_DECODE: nonfinite decoded BF16 output")
    return output
