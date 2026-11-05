"""Small CPU-only synthetic checks; does not qualify D0 or select D1 settings."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
import sys
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from gradient_comm_lab.bounded_discovery.uniform_codec import (
    UniformPacket, encode, decode, pack_unsigned, unpack_unsigned)


def digest(value):
    if value.numel() == 0:
        return hashlib.sha256(b"").hexdigest()
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def same(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape and digest(a) == digest(b)


def scalar_unpack(payload, q, count):
    """Third implementation: scalar byte/bit reads, no codec storage helper."""
    result = []
    for index in range(count):
        code = 0
        for bit in range(q):
            position = index*q + bit
            code |= ((payload[position//8] >> (position % 8)) & 1) << bit
        result.append(code)
    return np.asarray(result, dtype=np.int16)


def independent_decode(wire):
    """Parse only actual wire bytes with independently computed component sizes."""
    magic, version, q, g, logical = struct.unpack("<4sBBHQ", wire[:16])
    assert magic == b"BDU1" and version == 1
    padded = ((logical + g - 1)//g)*g
    payload_size = (padded*q + 7)//8
    assert len(wire) == 16 + payload_size + 4*(padded//g)
    signed = scalar_unpack(wire[16:16+payload_size], q, padded) - ((1 << (q-1))-1)
    scales = np.asarray(struct.unpack("<" + "f"*(padded//g), wire[16+payload_size:]), dtype=np.float32)
    before_bf16 = signed.astype(np.float32).reshape(-1, g)*scales[:, None]
    output = torch.from_numpy(before_bf16.reshape(-1).copy()).to(torch.bfloat16)[:logical].clone()
    return output, signed, scales, (q, g, logical, padded)


def synthetic(name, g):
    tiny = torch.finfo(torch.bfloat16).tiny
    if name == "empty": value = torch.empty(0)
    elif name == "zeros": value = torch.zeros(g+3)
    elif name == "constant_positive": value = torch.full((g+3,), 0.75)
    elif name == "constant_negative": value = torch.full((g+3,), -0.75)
    elif name == "mixed":
        pattern = torch.tensor([-1., -.75, -.3, -0., 0., .1, .35, .5, .9, 1., 2.25, -3.5])
        value = pattern.repeat((2*g+3+len(pattern)-1)//len(pattern))[:2*g+3]
    elif name == "sparse_outlier":
        value = torch.zeros(g+1); value[0] = 32.; value[-1] = .25
    elif name == "random":
        value = torch.randn(2*g+3, generator=torch.Generator(device="cpu").manual_seed(44120))
    elif name == "subnormal":
        pattern = torch.tensor([tiny/128, -tiny/128, tiny/64, -tiny/64, tiny, -tiny, 0.])
        value = pattern.repeat((g+3+len(pattern)-1)//len(pattern))[:g+3]
    elif name == "finite_extremes":
        largest = torch.finfo(torch.bfloat16).max
        pattern = torch.tensor([largest, -largest, tiny, -tiny, 0.])
        value = pattern.repeat((g+3+len(pattern)-1)//len(pattern))[:g+3]
    else: raise ValueError(name)
    return value.to(torch.bfloat16).contiguous()


def packet_checks(name, q, g):
    x = synthetic(name, g)
    input_before = digest(x)
    namespace = f"bounded_discovery_v1:uniform_selfcheck:{name}:q{q}:g{g}"
    seed = int.from_bytes(hashlib.sha256(namespace.encode()).digest()[:8], "little") % 2**63
    global_state = torch.random.get_rng_state().clone()
    packet = encode(x, q, g, seed)
    assert digest(x) == input_before
    same(torch.random.get_rng_state(), global_state)
    repeated = encode(x.clone(), q, g, seed)
    assert packet.to_bytes() == repeated.to_bytes() and packet.metadata == repeated.metadata
    wire = packet.to_bytes()
    independent, integer, scales, config = independent_decode(wire)
    actual = decode(UniformPacket.from_bytes(wire))
    same(actual, independent)
    same(decode(replace(packet, metadata={"incorrect": "decoder must ignore all audit metadata"})), actual)
    assert config[:3] == (q, g, x.numel())
    padded = config[3]
    maximum = (1 << (q-1))-1
    assert integer.size == padded and np.all(integer >= -maximum) and np.all(integer <= maximum)
    assert np.all(integer[x.numel():] == 0)
    values = np.zeros(padded, dtype=np.float32)
    values[:x.numel()] = x.float().numpy()
    blocks = values.reshape(-1, g)
    maxima = np.max(np.abs(blocks), axis=1)
    assert np.all(scales.astype(np.float64)*maximum >= maxima.astype(np.float64))
    assert np.all(scales[maxima == 0] == 0) and not np.signbit(scales[maxima == 0]).any()
    assert np.all(integer.reshape(-1, g)[maxima == 0] == 0)
    initial = maxima / np.float32(maximum)
    minimal = initial.copy()
    under = minimal.astype(np.float64)*maximum < maxima.astype(np.float64)
    adjustments = 0
    while under.any():
        minimal[under] = np.nextafter(minimal[under], np.float32(np.inf))
        adjustments += int(under.sum())
        under = minimal.astype(np.float64)*maximum < maxima.astype(np.float64)
    assert np.array_equal(minimal.view(np.uint32), scales.view(np.uint32))
    assert adjustments == packet.metadata["scale_nextafter_steps_total"]
    # Independently reproduce the specified FP32 stochastic decision using the
    # stored scales, NumPy arithmetic and a fresh local torch FP32 random stream.
    denominator = np.where(scales == 0, np.float32(1), scales)
    normalized = blocks / denominator[:, None]
    lower = np.floor(normalized)
    probability = normalized - lower
    generator = torch.Generator(device="cpu").manual_seed(seed)
    expected_initial_state = digest(generator.get_state())
    draws = torch.rand((padded//g, g), generator=generator, dtype=torch.float32).numpy()
    expected_integer = np.clip(lower + (draws < probability).astype(np.float32), -maximum, maximum).astype(np.int16).reshape(-1)
    assert np.array_equal(integer, expected_integer)
    assert packet.metadata["rng_draw_count"] == padded
    assert packet.metadata["rng_initial_state_sha256"] == expected_initial_state
    assert packet.metadata["rng_final_state_sha256"] == digest(generator.get_state())
    accounting = packet.accounting()
    assert accounting["payload_bytes"] == len(packet.payload) == (padded*q+7)//8
    assert accounting["scale_bytes"] == len(packet.scales) == 4*(padded//g)
    assert accounting["header_bytes"] == len(packet.header) == 16
    assert accounting["packet_total_bytes"] == len(wire) == sum(
        accounting[k] for k in ("payload_bytes", "scale_bytes", "header_bytes", "alignment_bytes"))
    assert accounting["packet_sha256"] == hashlib.sha256(wire).hexdigest()
    assert accounting["per_input_auxiliary_bytes"] == 0
    # Destroy the source; a new packet constructed from the copied wire is still
    # sufficient to decode, without an input tensor, scales cache or RNG state.
    x.fill_(0)
    same(decode(bytes(bytearray(wire))), independent)
    return dict(logical_numel=config[2], padded_numel=padded, packet_bytes=len(wire),
                upward_scale_adjustments=adjustments, independent_decode_pass=True,
                source_protected=True, global_rng_protected=True)


def expect_value_error(function):
    try: function()
    except ValueError as error: return dict(exception_type="ValueError", message=str(error))
    raise AssertionError("Expected ValueError was not raised")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New JSON report within results/bounded_discovery_v1")
    args = parser.parse_args()
    records = []
    def check(name, operation, check_type="synthetic_contract"):
        try:
            details = operation()
            records.append(dict(check_id=name, check_type=check_type, status="pass", details=details))
        except Exception as error:
            records.append(dict(check_id=name, check_type=check_type, status="fail",
                exception_type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))

    for q in range(2, 11):
        def storage(q=q):
            values = np.arange(1 << q, dtype=np.uint16)
            payload = pack_unsigned(values, q)
            assert len(payload) == (len(values)*q+7)//8
            assert np.array_equal(unpack_unsigned(payload, q, len(values)), values)
            assert np.array_equal(scalar_unpack(payload, q, len(values)), values)
            for count in (0, 1, 3, 7, 9):
                sample = values[:count]
                packed = pack_unsigned(sample, q)
                assert np.array_equal(scalar_unpack(packed, q, len(sample)), sample)
                assert np.array_equal(unpack_unsigned(packed, q, len(sample)), sample)
            return dict(exhaustive_stored_codes=len(values), reserved_code_storage_only=(1 << q)-1)
        check(f"pack_unpack/q{q}", storage, "pack_unpack")
        for g in (16, 128):
            for name in ("empty", "zeros", "constant_positive", "constant_negative", "mixed",
                         "sparse_outlier", "random", "subnormal", "finite_extremes"):
                check(f"packet/q{q}/g{g}/{name}", lambda q=q, g=g, name=name: packet_checks(name, q, g))
    x = synthetic("mixed", 16)
    invalid_inputs = {
        "fp32": lambda: encode(x.float(), 4, 16, 1),
        "two_dimensional": lambda: encode(torch.zeros((2, 16), dtype=torch.bfloat16), 4, 16, 1),
        "noncontiguous": lambda: encode(torch.zeros(32, dtype=torch.bfloat16)[::2], 4, 16, 1),
        "non_cpu": lambda: encode(torch.empty(16, device="meta", dtype=torch.bfloat16), 4, 16, 1),
        "nan": lambda: encode(torch.tensor([float("nan")], dtype=torch.bfloat16), 4, 16, 1),
        "inf": lambda: encode(torch.tensor([float("inf")], dtype=torch.bfloat16), 4, 16, 1),
        "q_below": lambda: encode(x, 1, 16, 1), "q_above": lambda: encode(x, 11, 16, 1),
        "q_bool": lambda: encode(x, True, 16, 1), "group": lambda: encode(x, 4, 32, 1),
        "seed_negative": lambda: encode(x, 4, 16, -1), "seed_too_large": lambda: encode(x, 4, 16, 2**63),
        "seed_bool": lambda: encode(x, 4, 16, True),
    }
    for name, operation in invalid_inputs.items():
        check("invalid_input/" + name, lambda op=operation: expect_value_error(op), "expected_rejection")
    packet = encode(x, 4, 16, 98123)
    wire = packet.to_bytes()
    malformed = {
        "truncated": wire[:-1], "extra_byte": wire+b"\0",
        "magic": b"FAIL"+wire[4:], "version": wire[:4]+b"\x02"+wire[5:],
        "header_q": wire[:5]+b"\x0b"+wire[6:],
    }
    for name, value in malformed.items():
        check("invalid_packet/" + name, lambda value=value: expect_value_error(lambda: decode(value)), "expected_rejection")
    for q in range(2, 11):
        p = encode(x, q, 16, 1)
        unsigned = scalar_unpack(p.payload, q, p.validate_components()["padded_numel"])
        unsigned[0] = (1 << q)-1
        bad = replace(p, payload=pack_unsigned(unsigned, q))
        check(f"reserved_code/q{q}", lambda bad=bad: expect_value_error(lambda: decode(bad)), "expected_rejection")
    for name, scale in (("nan_scale", float("nan")), ("infinite_scale", float("inf")),
                        ("negative_scale", -1.), ("negative_zero_scale", -0.)):
        bad = replace(packet, scales=struct.pack("<f", scale)+packet.scales[4:])
        check("invalid_packet/" + name, lambda bad=bad: expect_value_error(lambda: decode(bad)), "expected_rejection")
    zero = encode(torch.zeros(1, dtype=torch.bfloat16), 4, 16, 1)
    codes = scalar_unpack(zero.payload, 4, 16)
    codes[0] = 8
    bad_zero = replace(zero, payload=pack_unsigned(codes, 4))
    check("invalid_packet/nonzero_code_zero_scale", lambda: expect_value_error(lambda: decode(bad_zero)), "expected_rejection")
    codes = scalar_unpack(packet.payload, 4, packet.validate_components()["padded_numel"])
    codes[x.numel()] = 8
    bad_tail = replace(packet, payload=pack_unsigned(codes, 4))
    check("invalid_packet/nonzero_tail", lambda: expect_value_error(lambda: decode(bad_tail)), "expected_rejection")
    def stochastic_variation():
        first = encode(x, 4, 16, 9921).to_bytes()
        second = encode(x, 4, 16, 9922).to_bytes()
        assert first != second
        return dict(distinct_local_states_produced_distinct_packets=True,
                    unbiasedness_or_statistical_independence_claimed=False)
    check("different_local_states", stochastic_variation, "repeatability")
    failures = [r for r in records if r["status"] != "pass"]
    adjustments = sum(r.get("details", {}).get("upward_scale_adjustments", 0) for r in records)
    if not adjustments:
        records.append(dict(check_id="scale_upward_branch_exercised", status="fail",
                            message="Synthetic inputs did not exercise an upward scale adjustment"))
        failures = [r for r in records if r["status"] != "pass"]
    report = dict(schema_version="bounded_discovery_uniform_selfcheck_v1",
        scope="CPU synthetic implementation checks only; not D0 qualification or D1 comparison",
        status="pass" if not failures else "fail", real_inputs_read=False, gpu_executed=False,
        d1_configurations_selected=False, created_at=datetime.now(timezone.utc).isoformat(),
        command=[sys.executable, *sys.argv], python=sys.version, torch=str(torch.__version__), numpy=str(np.__version__),
        source_sha256={path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in
            (Path(__file__), ROOT/"src/gradient_comm_lab/bounded_discovery/uniform_codec.py")},
        check_count=len(records), failed_count=len(failures), scale_upward_adjustments_observed=adjustments,
        records=records)
    if args.output:
        output = args.output.resolve()
        if not output.is_relative_to(ROOT/"results/bounded_discovery_v1"):
            raise ValueError("Selfcheck output must be within the new bounded_discovery_v1 results")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, ensure_ascii=False, allow_nan=False))
    if failures: print(json.dumps(failures, ensure_ascii=False, allow_nan=False))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
