"""CPU-only independent audit of a frozen G2a logical-ring evidence directory.

This checker does not import either ring implementation.  Protocol identities,
routing, packet sizes and coverage are derived here directly from the contract.
Missing evidence is an error; a reported ``pass`` never substitutes for evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
EXECUTIONS = (("cpu_reference", "baseline"), ("cuda_logical_ring", "baseline"),
              ("cuda_logical_ring", "repeat"), ("cuda_logical_ring", "reverse"))
COMPONENTS = ("payload", "scale_codes", "top_scales")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
CHECK_TYPES = {"prepared_input_parity", "packet_parity", "decode_parity",
               "partial_sum_parity", "output_parity", "rank_consistency",
               "ledger", "threshold_consumption", "immutability"}
REJECTION_GROUPS = {
    "RING_BITS: expected nominal b4/b8": ("bits2", "bits_bool"),
    "RING_ENTRY_MODE: expected raw_fp32 or predivided_bf16": ("entry_mode",),
    "RING_WORKERS: expected a list or tuple with 1, 2 or 4 workers": ("workers_empty", "workers_three"),
    "RING_CASE_ID: expected a nonempty string": ("case_id",),
    "RING_INPUT: expected a contiguous one-dimensional tensor": ("input_not_tensor", "input_2d", "input_noncontiguous"),
    "RING_INPUT_DTYPE: input dtype does not match entry mode": ("input_dtype",),
    "RING_INPUT_DEVICE: expected CUDA tensors on one device": ("input_cpu", "input_mixed_device"),
    "RING_INPUT_LENGTH: worker lengths must match": ("input_length",),
    "RING_NONFINITE_INPUT: inputs must be finite": ("input_nan", "input_inf"),
    "RING_RANK_ORDER: expected a permutation of worker ranks": ("rank_duplicate", "rank_out_of_range", "rank_bool"),
    "RING_THRESHOLDS: expected a callable provider or get method": ("provider_missing",),
    "RING_THRESHOLDS: provider must return thresholds for this event": ("threshold_result",),
    "RING_THRESHOLD_IDENTITY: event key differs from requested event": ("threshold_identity",),
    "RING_THRESHOLD_METADATA: expected contiguous BF16 tensors of canonical shape":
        ("threshold_shape", "threshold_dtype", "threshold_noncontiguous", "late_threshold_shape"),
    "RING_THRESHOLD_DEVICE: expected CPU or the input CUDA device": ("threshold_device",),
    "RING_THRESHOLD_VALUES: expected finite values in [0,65536]": ("threshold_negative", "threshold_too_large", "threshold_nan"),
    "RING_LUT_METADATA: expected contiguous one-dimensional BF16 LUT": ("lut_shape", "lut_dtype"),
    "RING_LUT_DEVICE: expected CPU or the input CUDA device": ("lut_device",),
    "RING_LUT_VALUES: LUT must be finite and strictly ascending": ("lut_nonmonotonic", "lut_nan"),
    "RING_LUT_IDENTITY: LUT differs from frozen CPU source": ("lut_identity",),
    "RING_NONFINITE_PREPARED_INPUT: BF16 entry conversion overflowed": ("prepared_overflow",),
    "RING_BUILD: expected the accepted G1 a02 module identity": ("build_missing",),
}
REJECTIONS = {name: message for message, names in REJECTION_GROUPS.items() for name in names}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def json_safe(value):
    if isinstance(value, set): return sorted((json_safe(x) for x in value), key=str)
    if isinstance(value, (tuple, list)): return [json_safe(x) for x in value]
    if isinstance(value, dict): return {str(k): json_safe(v) for k, v in value.items()}
    return value


def strict_json(text):
    def reject(value):
        raise ValueError("Nonfinite JSON token: " + value)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result
    result = json.loads(text, parse_constant=reject, object_pairs_hook=unique)
    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Nonfinite JSON number")
        if isinstance(value, dict):
            for item in value.values(): finite(item)
        elif isinstance(value, list):
            for item in value: finite(item)
    finite(result)
    return result


def read(path):
    return strict_json(Path(path).read_text(encoding="utf-8"))


def rows(path):
    return [strict_json(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def dimensions(case):
    n, d, b = case["world_size"], case["logical_numel"], case["nominal_bits"]
    active = n > 1 and d > 0
    p = n * 1024 * ((d + n * 1024 - 1) // (n * 1024)) if active else d
    return n, d, b, p, p // n if active else 0, active


def event_descriptor(case, phase, rank, chunk, ordinal):
    return {"case_id": case["case_id"], "nominal_bits": case["nominal_bits"],
            "phase": phase, "rank": rank, "chunk": chunk, "encoding_ordinal": ordinal}


def expected_protocol(case):
    """Derive every identity independently, with a barrier between each hop."""
    n, d, b, p, length, active = dimensions(case)
    events, messages, intermediates = {}, {}, {}
    if not active:
        return events, messages, intermediates
    prefix = f"{case['case_id']}/b{b}"
    sizes = {"payload_bytes": length * (b + 1) // 8,
             "scale_codes_bytes": length // 16, "top_scales_bytes": 2 * (length // 256)}
    sizes["wire_bytes"] = sum(sizes.values())
    def event(phase, rank, chunk, ordinal):
        desc = event_descriptor(case, phase, rank, chunk, ordinal)
        key = canonical(desc)
        events[key] = desc
        return key
    def intermediate(phase, kind, hop, rank, chunk, source):
        key = f"{prefix}/{phase}/{kind}/h{hop}/r{rank}/c{chunk}"
        intermediates[key] = {"intermediate_id": key, "phase": phase, "kind": kind,
                              "hop": hop, "rank": rank, "chunk": chunk,
                              "source_packet_id": source}
    def message(phase, hop, sender, chunk, source):
        key = f"{prefix}/{phase}/h{hop}/r{sender}/c{chunk}"
        messages[key] = {"message_id": key, "phase": phase, "hop": hop,
                         "sender": sender, "receiver": (sender + 1) % n,
                         "chunk": chunk, "source_packet_id": source,
                         "logical_numel": d, "padded_numel": p,
                         "chunk_logical_numel": max(0, min(length, d - chunk * length)),
                         "chunk_padded_numel": length, **sizes}
    for hop in range(n - 1):
        for rank in range(n):
            chunk = (rank - hop) % n
            source = None if hop == 0 else canonical(event_descriptor(
                case, "reduce_scatter", (rank - 1) % n, chunk, hop - 1))
            if hop:
                intermediate("reduce_scatter", "decode", hop, rank, chunk, source)
            intermediate("reduce_scatter", "partial_sum", hop, rank, chunk, source)
            packet = event("reduce_scatter", rank, chunk, hop)
            message("reduce_scatter", hop, rank, chunk, packet)
    for rank in range(n):
        chunk = (rank + 1) % n
        source = canonical(event_descriptor(case, "reduce_scatter", (rank - 1) % n,
                                             chunk, n - 2))
        intermediate("sink", "decode", n - 1, rank, chunk, source)
        intermediate("sink", "partial_sum", n - 1, rank, chunk, source)
        packet = event("sink", rank, chunk, n - 1)
        intermediate("sink_self", "decode", n - 1, rank, chunk, packet)
    for hop in range(n - 1):
        for rank in range(n):
            chunk = (rank + 1 - hop) % n
            packet = canonical(event_descriptor(case, "sink", (chunk - 1) % n,
                                                 chunk, n - 1))
            message("all_gather", hop, rank, chunk, packet)
            intermediate("all_gather", "decode", hop, (rank + 1) % n, chunk, packet)
    assert len(events) == n * n
    assert len(messages) == 2 * n * (n - 1)
    assert len(intermediates) == 3 * n * n - n
    return events, messages, intermediates


class Audit:
    def __init__(self, run):
        self.run = run
        self.errors = []
        self.check_count = 0
        self.counts = {}

    def check(self, condition, message, context=None):
        self.check_count += 1
        if not condition:
            self.errors.append({"message": message, "context": context})
        return bool(condition)

    def equal(self, actual, expected, message, context=None):
        return self.check(actual == expected, message,
                          {"identity": json_safe(context), "actual": json_safe(actual), "expected": json_safe(expected)})

    def stage(self, name, function):
        try:
            function()
        except Exception as error:
            self.errors.append({"message": "Audit stage could not finish: " + name,
                                "exception_type": type(error).__name__, "detail": str(error)})

    def identity(self, value, context, *, dtype=None, shape=None):
        if not self.check(isinstance(value, dict), "Missing tensor identity", context):
            return
        self.check(isinstance(value.get("sha256"), str) and bool(SHA256.fullmatch(value["sha256"])),
                   "Invalid tensor SHA256", context)
        dims = value.get("shape")
        self.check(isinstance(dims, list) and all(type(x) is int and x >= 0 for x in dims),
                   "Invalid tensor shape", context)
        self.check(type(value.get("bytes")) is int and value["bytes"] >= 0,
                   "Invalid tensor byte count", context)
        width = {"torch.bfloat16": 2, "torch.float32": 4, "torch.uint8": 1}.get(value.get("dtype"))
        self.check(width is not None, "Unexpected identity dtype", context)
        if width and isinstance(dims, list):
            self.equal(value.get("bytes"), math.prod(dims) * width, "Tensor byte formula", context)
        if dtype: self.equal(value.get("dtype"), dtype, "Tensor dtype", context)
        if shape is not None: self.equal(dims, shape, "Tensor shape", context)
        self.equal(value.get("byte_order"), "little", "Tensor byte order", context)

    def identity_equal(self, actual, expected, context):
        self.identity(actual, context)
        self.identity(expected, context)
        self.equal(actual, expected, "Tensor bit identity differs", context)

    def packet(self, packet, length, bits, context):
        for name, width, count in (("payload", "torch.uint8", length * (bits + 1) // 8),
                                   ("scale_codes", "torch.uint8", length // 16),
                                   ("top_scales", "torch.bfloat16", length // 256)):
            self.identity(packet[name], (context, name), dtype=width)
            self.equal(math.prod(packet[name]["shape"]), count, "Packet component elements", (context, name))
        self.equal(packet["wire_bytes"], sum(packet[name]["bytes"] for name in COMPONENTS),
                   "Packet total actual bytes", context)
        self.check(isinstance(packet.get("wire_sha256"), str) and bool(SHA256.fullmatch(packet["wire_sha256"])),
                   "Packet concatenated-byte SHA256 missing", context)
        self.equal((packet["logical_numel"], packet["padded_numel"], packet["bits"]),
                   (length, length, bits), "Packet metadata", context)

    def unique_index(self, records, key, label):
        result = {}
        for row in records:
            identity = key(row)
            self.check(identity not in result, "Duplicate " + label, identity)
            result[identity] = row
        return result

    def common(self, records, filename):
        self.unique_index(records, lambda r: r["record_id"], filename + " record_id")
        for row in records:
            self.equal(row.get("run_id"), self.run.name, "Row run identity", (filename, row.get("record_id")))
            self.equal(row.get("contract_sha256"), self.contract_sha256,
                       "Row contract identity", (filename, row.get("record_id")))
            self.equal(row.get("status"), "pass", "Row did not pass", (filename, row.get("record_id")))

    def registration(self):
        self.contract_sha256 = sha(self.run / "ring_contract.json")
        self.contract = read(self.run / "ring_contract.json")
        self.registration_data = read(self.run / "execution_registration.json")
        self.inputs = read(self.run / "input_manifest.json")
        self.thresholds = read(self.run / "threshold_manifest.json")
        self.case_manifest = read(self.run / "correctness_case_manifest.json")
        self.summary = read(self.run / "summary.json")
        registration = self.registration_data
        for name, field in (("ring_contract.json", "contract_sha256"),
                            ("input_manifest.json", "input_manifest_sha256"),
                            ("threshold_manifest.json", "threshold_manifest_sha256"),
                            ("correctness_case_manifest.json", "correctness_case_manifest_sha256")):
            self.equal(sha(self.run / name), registration.get(field), "Frozen manifest changed", name)
        self.check(bool(registration.get("sources")), "Missing frozen source manifest")
        for name, digest in registration["sources"].items():
            self.equal(sha(ROOT / name), digest, "Live frozen source changed", name)
            self.equal(sha(self.run / "code" / name), digest, "Frozen source copy changed", name)
        for filename in ("scripts/audit_logical_ring.py", "src/gradient_comm_lab/logical_ring.py",
                         "src/gradient_comm_lab/ring_fixtures.py"):
            self.check(filename in registration["sources"], "Required source not frozen", filename)
        build_path = self.run / self.contract["build_manifest"]
        build = read(build_path)
        self.equal(sha(build_path), registration["build_manifest_sha256"], "Registered build manifest changed")
        self.equal(sha(build_path), self.contract["build_manifest_sha256"], "Contract build manifest changed")
        self.equal(sha(build["module_path"]), build["module_sha256"], "Actual accepted binary changed")
        self.equal(build["module_sha256"], self.contract["module_sha256"], "Contract binary differs")
        self.equal(build["module_sha256"], "c4e37ff9394580bb15fe5d277115f616a439f2741528dff6826bc77e85f06b58",
                   "Unexpected G1 module binary")
        self.equal(sha(self.contract["cpu_reference_path"]), self.contract["cpu_reference_sha256"],
                   "Original CPU reference changed")
        self.equal(self.contract["cpu_reference_sha256"],
                   "9296231d3a29d3fb4cfa1d12a2ffe5a1a4d2922e62c51003fb214c59303e8ea0",
                   "Original ring reference is not the accepted source")

    def manifests(self):
        cases = self.inputs["cases"]
        if "inputs" in self.inputs: self.equal(self.inputs["inputs"], cases, "Input aliases disagree")
        self.cases = self.unique_index(cases, lambda r: r["case_id"], "input case")
        self.counts["input_cases"] = len(cases)
        for case in cases:
            n, d, b, p, length, active = dimensions(case)
            key = case["case_id"]
            self.check(n in (1, 2, 4) and b in (4, 8) and type(d) is int and d >= 0,
                       "Unsupported case dimensions", key)
            self.equal(case.get("worker_count", n), n, "Worker count aliases", key)
            self.equal(case.get("bits", b), b, "Bit width aliases", key)
            self.check(case["entry_mode"] in ("raw_fp32", "predivided_bf16"), "Entry mode", key)
            self.equal(len(case["workers"]), n, "Input worker count", key)
            for rank, worker in enumerate(case["workers"]):
                self.equal(worker["rank"], rank, "Worker rank/order", key)
                identity = worker["tensor"]
                self.identity(identity, (key, rank), dtype="torch.float32" if case["entry_mode"] == "raw_fp32"
                              else "torch.bfloat16", shape=[d])
            fixture = self.run / case["fixture_path"]
            self.equal(sha(fixture), case["fixture_sha256"], "Input fixture archive changed", key)
            self.equal(fixture.stat().st_size, case["fixture_bytes"], "Input fixture archive bytes", key)
            self.equal(case["padded_numel"], p, "Manifest ring padding formula", key)
            self.equal(case["chunk_numel"], length, "Manifest ring chunk length", key)
        observed = {(c["world_size"], c["nominal_bits"], c["logical_numel"], c["entry_mode"]) for c in cases}
        required = {(n, b, d, mode) for n in (1, 2, 4) for b in (4, 8)
                    for d in {0, 1, 255, 256, 257, 1023, 1024, 1025, n*1024-1, n*1024, n*1024+1, 65536}
                    for mode in ("raw_fp32", "predivided_bf16")}
        self.equal(sorted(required - observed), [], "Required length/entry grid missing")
        for n in (1, 2, 4):
            for b in (4, 8):
                for mode in ("raw_fp32", "predivided_bf16"):
                    patterns = {c.get("pattern") for c in cases if c["world_size"] == n
                                and c["nominal_bits"] == b and c["entry_mode"] == mode}
                    self.check({"signed", "zeros", "cancellation", "finite_extremes"} <= patterns,
                               "Required numerical input families missing", (n, b, mode))
        real = [c for c in cases if c["source_kind"] == "real_gradient"]
        self.equal(len(real), 6, "Expected six fixed real collective replays")
        real_keys = []
        for case in real:
            key = case["case_id"]
            self.equal((case["world_size"], case["entry_mode"], case["logical_numel"],
                        case["padded_numel"], case["chunk_numel"]),
                       (4, "predivided_bf16", 818048, 819200, 204800), "Real replay collective dimensions", key)
            steps = {w["source"]["update_step"] for w in case["workers"]}
            seeds = {w["source"]["training_seed"] for w in case["workers"]}
            self.equal(len(steps), 1, "Mixed snapshots within a real collective", key)
            self.equal(seeds, {1101}, "Real replay training seed", key)
            real_keys.append((next(iter(steps)), case["nominal_bits"]))
            for rank, worker in enumerate(case["workers"]):
                source = worker["source"]
                self.equal((source["rank"], source["already_divided_by"], source["original_numel"]),
                           (rank, 4, 818048), "Real source rank/predivision/length", (key, rank))
                self.equal(sha(source["fixture_path"]), source["fixture_sha256"], "Real source archive changed", (key, rank))
        self.equal(Counter(real_keys), Counter((step, b) for step in (100, 500, 2500) for b in (4, 8)),
                   "Real snapshot/bit coverage or duplicate group")
        self.threshold_by_key = self.unique_index(self.thresholds["events"], lambda r: r["event_key"], "threshold event")
        expected = {}
        self.protocols = {}
        for key, case in self.cases.items():
            protocol = expected_protocol(case)
            self.protocols[key] = protocol
            expected.update(protocol[0])
        self.equal(set(self.threshold_by_key), set(expected), "Threshold event coverage")
        for key, descriptor in expected.items():
            record = self.threshold_by_key[key]
            self.equal(record["event"], descriptor, "Threshold event descriptor", key)
            self.equal(key, canonical(record["event"]), "Threshold key not canonical", key)
            self.check(type(record["seed"]) is int and 0 <= record["seed"] < 2**63,
                       "Threshold seed invalid", key)
            expected_seed = int.from_bytes(hashlib.sha256(("logical_ring_v1:" + key).encode()).digest()[:8],
                                           "little") % 2**63
            self.equal(record["seed"], expected_seed, "Threshold seed derivation", key)
            length = dimensions(self.cases[descriptor["case_id"]])[4]
            self.identity(record["main"], key, dtype="torch.bfloat16", shape=[length//16, 16])
            self.identity(record["scale"], key, dtype="torch.bfloat16", shape=[length//256, 16])
        self.counts["frozen_threshold_events"] = len(expected)

    def fixture_tensors(self):
        # Explicit CPU fixture reads/arithmetic; no ring or CUDA module import.
        import torch
        def identity(tensor):
            value = tensor.detach().cpu().contiguous()
            return {"sha256": hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest(),
                    "shape": list(value.shape), "dtype": str(value.dtype), "byte_order": "little",
                    "bytes": value.numel()*value.element_size()}
        self.prepared_inputs = {}
        for key, case in self.cases.items():
            fixture = torch.load(self.run / case["fixture_path"], map_location="cpu", weights_only=True)
            values = fixture["inputs"]
            self.equal(len(values), case["world_size"], "Actual fixture worker count", key)
            prepared = []
            for rank, (value, worker) in enumerate(zip(values, case["workers"])):
                self.identity_equal(identity(value), worker["tensor"], (key, rank, "fixture"))
                if case["world_size"] > 1 and case["logical_numel"] > 0 and case["entry_mode"] == "raw_fp32":
                    core = value.to(torch.bfloat16)
                    core = (core / torch.tensor(case["world_size"], dtype=torch.bfloat16)).to(torch.bfloat16)
                else:
                    core = value
                prepared.append(identity(core))
                if case["source_kind"] == "real_gradient":
                    source = torch.load(worker["source"]["fixture_path"], map_location="cpu", weights_only=False)
                    self.identity_equal(identity(source["input_logical"]), identity(value), (key, rank, "real_source"))
            self.prepared_inputs[key] = prepared
        for key, record in self.threshold_by_key.items():
            length = dimensions(self.cases[record["event"]["case_id"]])[4]
            generator = torch.Generator(device="cpu").manual_seed(record["seed"])
            for name, shape in (("main", (length//16, 16)), ("scale", (length//256, 16))):
                value = torch.rand(shape, generator=generator, dtype=torch.float32).to(torch.bfloat16)
                value = value * torch.tensor(65535, dtype=torch.bfloat16)
                self.identity_equal(identity(value), record[name], (key, name, "regenerated_threshold"))

    def raw_records(self):
        self.raw = {}
        for filename in ("quantization_events.jsonl", "message_ledger.jsonl", "intermediate_checks.jsonl",
                         "input_output_checks.jsonl", "correctness.jsonl", "buffer_lifetime_checks.jsonl"):
            self.raw[filename] = rows(self.run / filename)
            self.common(self.raw[filename], filename)
            self.counts[filename.removesuffix(".jsonl") + "_rows"] = len(self.raw[filename])
        for filename in ("quantization_events.jsonl", "message_ledger.jsonl", "intermediate_checks.jsonl",
                         "input_output_checks.jsonl"):
            for row in self.raw[filename]:
                key = row["case_id"]
                self.check(key in self.cases, "Unknown raw case", (filename, key))
                case = self.cases[key]
                for field in ("nominal_bits", "world_size", "logical_numel", "entry_mode"):
                    self.equal(row[field], case[field], "Raw row case metadata", (filename, key, field))
                self.check((row["method_id"], row["variant"]) in EXECUTIONS,
                           "Unknown execution role", (filename, row["record_id"]))

    def events(self):
        self.event_rows = self.unique_index(self.raw["quantization_events.jsonl"],
            lambda r: (r["method_id"], r["variant"], r["event_id"]), "quantization event")
        wanted = {(m, v, key) for protocol in self.protocols.values() for key in protocol[0]
                  for m, v in EXECUTIONS}
        self.equal(set(self.event_rows), wanted, "Complete quantization event identity coverage")
        for (method, variant, key), row in self.event_rows.items():
            case = self.cases[row["case_id"]]
            length = dimensions(case)[4]
            descriptor = self.protocols[row["case_id"]][0][key]
            for field, expected in descriptor.items():
                self.equal(row[field], expected, "Quantization event descriptor", (key, field))
            self.equal(row["event_key"], key, "Event/threshold identity", key)
            self.equal(row["record_id"], f"{method}/{variant}/{key}", "Event record identity", key)
            self.identity(row["encoding_input"], key, dtype="torch.bfloat16", shape=[length])
            self.packet(row["packet"], length, case["nominal_bits"], key)
            cpu = self.event_rows[("cpu_reference", "baseline", key)]
            self.identity_equal(row["encoding_input"], cpu["encoding_input"], (key, "CPU encode input"))
            self.identity_equal(row["expected_encoding_input"], cpu["encoding_input"], (key, "expected encode input"))
            self.equal(row["packet"], cpu["packet"], "CPU/GPU/repeat/reverse packet identity", (method, variant, key))
            self.equal(row["expected_packet"], cpu["packet"], "Expected packet is not independent CPU packet", key)
            self.equal(row["encoding_input_bit_mismatch_count"], 0, "Encode input mismatches", key)
            self.equal(row["packet_byte_mismatch_count"], 0, "Encoded packet mismatches", key)
            threshold = self.threshold_by_key[key]
            for name in ("main", "scale"):
                self.identity_equal(row["threshold_info"][name], threshold[name], (key, name))
                if method == "cuda_logical_ring":
                    self.identity_equal(row[name + "_thresholds"], threshold[name], (key, name, "actual encode threshold"))
                else:
                    for moment in ("before", "after"):
                        self.identity_equal(row["threshold_info"][moment][name], threshold[name], (key, name, moment))
            self.equal(row["threshold_info"]["seed"], threshold["seed"], "Encode threshold seed", key)

    def messages(self):
        self.message_rows = self.unique_index(self.raw["message_ledger.jsonl"],
            lambda r: (r["method_id"], r["variant"], r["message_id"]), "directed message")
        wanted = {(m, v, key) for protocol in self.protocols.values() for key in protocol[1]
                  for m, v in EXECUTIONS}
        self.equal(set(self.message_rows), wanted, "Complete directed message identity coverage")
        for (method, variant, key), row in self.message_rows.items():
            expected = self.protocols[row["case_id"]][1][key]
            for field, value in expected.items():
                self.equal(row[field], value, "Directed route/actual bytes formula", (key, field))
            self.equal(row["record_id"], f"{method}/{variant}/{key}", "Message record identity", key)
            packet = row["packet"]
            source = self.event_rows[(method, variant, expected["source_packet_id"])]
            self.equal(packet, source["packet"], "Message is not its original immutable source packet", (method, variant, key))
            cpu = self.message_rows[("cpu_reference", "baseline", key)]
            self.equal(packet, cpu["packet"], "Message byte identity differs from independent CPU route", key)
            for component in COMPONENTS:
                self.equal(row[component + "_bytes"], packet[component]["bytes"], "Ledger versus actual buffer bytes", (key, component))
            self.equal(row["wire_bytes"], packet["wire_bytes"], "Ledger wire bytes", key)
            if method == "cuda_logical_ring":
                self.equal((row["header_bytes"], row["alignment_bytes"]), (0, 0), "Unexpected uncounted protocol overhead", key)
                self.equal(row["scale_bytes"], row["scale_codes_bytes"] + row["top_scales_bytes"], "Scale byte sum", key)

    def intermediates(self):
        self.intermediate_rows = self.unique_index(self.raw["intermediate_checks.jsonl"],
            lambda r: (r["method_id"], r["variant"], r["intermediate_id"]), "intermediate")
        wanted = {(m, v, key) for protocol in self.protocols.values() for key in protocol[2]
                  for m, v in EXECUTIONS}
        self.equal(set(self.intermediate_rows), wanted, "Complete partial-sum/decode identity coverage")
        decoded = {}
        for (method, variant, key), row in self.intermediate_rows.items():
            expected = self.protocols[row["case_id"]][2][key]
            for field, value in expected.items():
                self.equal(row[field], value, "Intermediate protocol identity", (key, field))
            self.equal(row["record_id"], f"{method}/{variant}/{key}", "Intermediate record identity", key)
            length = dimensions(self.cases[row["case_id"]])[4]
            self.identity(row["actual"], key, dtype="torch.bfloat16", shape=[length])
            cpu = self.intermediate_rows[("cpu_reference", "baseline", key)]
            self.identity_equal(row["actual"], cpu["actual"], (method, variant, key))
            self.identity_equal(row["expected"], cpu["actual"], (key, "independent expectation"))
            self.equal(row["bit_mismatch_count"], 0, "Intermediate mismatch count", key)
            self.equal(row["reference_record"], method == "cpu_reference", "CPU reference role", key)
            if row["kind"] == "partial_sum":
                event = canonical(event_descriptor(self.cases[row["case_id"]], row["phase"], row["rank"],
                                                    row["chunk"], row["hop"]))
                self.identity_equal(row["actual"], self.event_rows[(method, variant, event)]["encoding_input"],
                                    (key, "sum-to-encode linkage"))
            else:
                source_key = (method, variant, row["source_packet_id"])
                if source_key in decoded:
                    self.identity_equal(row["actual"], decoded[source_key], (key, "same packet decode"))
                else:
                    decoded[source_key] = row["actual"]

    def input_outputs(self):
        self.io_rows = self.unique_index(self.raw["input_output_checks.jsonl"],
            lambda r: (r["case_id"], r["method_id"], r["variant"]), "input/output execution")
        self.equal(set(self.io_rows), {(c, m, v) for c in self.cases for m, v in EXECUTIONS},
                   "Complete input/output execution coverage")
        for (key, method, variant), row in self.io_rows.items():
            case = self.cases[key]
            n, d, b, p, length, active = dimensions(case)
            original = [w["tensor"] for w in case["workers"]]
            self.equal(row["inputs_before"], original, "Inputs differ from frozen fixture", (key, method, variant))
            self.equal(row["inputs_after"], original, "Inputs modified", (key, method, variant))
            self.equal(row["prepared_inputs"], self.prepared_inputs[key], "Independent entry conversion/predivision", (key, method, variant))
            cpu = self.io_rows[(key, "cpu_reference", "baseline")]
            self.equal(row["outputs"], cpu["outputs"], "Output parity/repeat/reverse", (key, method, variant))
            self.equal(row["expected_outputs"], cpu["outputs"], "Expected outputs not from CPU record", (key, method, variant))
            self.equal(len(row["outputs"]), n, "Output rank count", key)
            self.equal(row["record_id"], f"{method}/{variant}/{key}/input_output", "Input/output record identity", key)
            for rank, output in enumerate(row["outputs"]):
                self.identity(output, (key, rank), dtype="torch.float32" if case["entry_mode"] == "raw_fp32" else "torch.bfloat16", shape=[d])
                self.equal(output, row["outputs"][0], "Ranks do not agree", (key, rank))
                if not active: self.equal(output, original[rank], "Bypass lost identity/dtype", (key, rank))
            if method == "cuda_logical_ring" or active:
                self.identity_equal(row["lut_before"], self.contract["luts"][str(b)], (key, "LUT before"))
                self.identity_equal(row["lut_after"], self.contract["luts"][str(b)], (key, "LUT after"))
            else:
                self.equal((row["lut_before"], row["lut_after"]), (None, None), "CPU bypass unexpectedly used a LUT", key)
            calls = self.unique_index(row["threshold_consumption"], lambda r: r["event_key"], "actual threshold call")
            wanted = self.protocols[key][0]
            self.equal(set(calls), set(wanted), "Actual threshold consumption (AllGather must consume zero)", (key, method, variant))
            for event, call in calls.items():
                for field, value in wanted[event].items(): self.equal(call[field], value, "Consumed threshold descriptor", (event, field))
                frozen = self.threshold_by_key[event]
                for field in ("seed", "main", "scale"):
                    self.equal(call[field], frozen[field], "Consumed threshold identity", (event, field))
            immutability = self.unique_index(row["threshold_immutability"], lambda r: r["event_key"], "threshold immutability")
            self.equal(set(immutability), set(wanted), "Threshold immutability coverage", (key, method, variant))
            for event, record in immutability.items():
                for moment in ("before", "after"):
                    for field in ("main", "scale"):
                        self.identity_equal(record[moment][field], self.threshold_by_key[event][field], (event, moment, field))
            metadata = row["metadata"]
            for field, value in (("world_size", n), ("logical_numel", d), ("padded_numel", p),
                                 ("chunk_numel", length), ("entry_mode", case["entry_mode"])):
                self.equal(metadata[field], value, "Execution metadata dimensions", (key, field))
            order = list(reversed(range(n))) if variant == "reverse" else list(range(n))
            self.equal(metadata["rank_order"], order, "Host rank traversal order", (key, method, variant))
            if method == "cuda_logical_ring":
                expected_counts = {"quantization_events": n*n if active else 0,
                                   "message_count": 2*n*(n-1) if active else 0,
                                   "threshold_consumption_count": n*n if active else 0,
                                   "decode_count": 2*n*n-n if active else 0,
                                   "total_sent_wire_bytes": sum(m["wire_bytes"] for m in self.protocols[key][1].values())}
                for field, value in expected_counts.items(): self.equal(metadata[field], value, "Actual execution count", (key, field))
                build = metadata["build"]
                self.equal(build["manifest_sha256"], self.registration_data["build_manifest_sha256"], "Loaded build manifest", key)
                self.equal(build["module_sha256"], self.contract["module_sha256"], "Loaded module identity", key)
                self.equal(build["module_loaded"], active, "Bypass/active module load", key)
                if active:
                    self.equal(build["actually_loaded_module_sha256"], self.contract["module_sha256"], "Actual loaded module SHA", key)
                    self.equal(str(Path(build["actually_loaded_module_path"]).resolve()), str(Path(build["module_path"]).resolve()), "Actual loaded module path", key)
                self.equal(metadata["network_transport_implemented"], False, "Unexpected network claim", key)

    def correctness(self):
        manifest = self.unique_index(self.case_manifest["cases"], lambda r: r["record_id"], "frozen check")
        actual = self.unique_index(self.raw["correctness.jsonl"], lambda r: r["record_id"], "correctness check")
        wanted = {f"cuda_logical_ring/{v}/{key}/{check}" for key in self.cases
                  for v in ("baseline", "repeat", "reverse") for check in CHECK_TYPES}
        wanted |= {f"buffer/b{b}/{check}" for b in (4, 8)
                   for check in ("two_packets_in_flight", "decode_output_reuse", "cross_collective_isolation")}
        wanted |= {"invalid/" + name for name in REJECTIONS}
        self.equal(set(manifest), wanted, "Frozen correctness identities are incomplete/unexpected")
        self.equal(set(actual), wanted, "Actual correctness identities are incomplete/unexpected")
        self.equal(self.case_manifest["expected_record_count"], len(wanted), "Frozen correctness record count")
        self.equal({(r["method_id"], r["variant"]) for r in self.case_manifest["execution_roles"]},
                   set(EXECUTIONS), "Frozen execution roles")
        for key, spec in manifest.items():
            self.equal(spec["expected_result"], "pass", "Frozen check expectation", key)
            row = actual[key]
            for field in ("case_id", "method_id", "variant", "nominal_bits", "check_type"):
                self.equal(row[field], spec[field], "Correctness record differs from frozen identity", (key, field))
            if row["variant"] not in ("baseline", "repeat", "reverse"): continue
            case = self.cases[row["case_id"]]
            n, d, b, p, length, active = dimensions(case)
            check = row["check_type"]
            details = row["details"]
            io = self.io_rows[(row["case_id"], row["method_id"], row["variant"])]
            if check == "prepared_input_parity":
                self.equal(details["count"], n, "Prepared input check count", key)
            elif check == "packet_parity":
                for field in ("expected_events", "actual_events"):
                    self.equal(details[field], n*n if active else 0, "Packet event check count", key)
                self.equal(details["bit_or_byte_mismatch_count"], 0, "Packet/input differences", key)
            elif check in ("decode_parity", "partial_sum_parity"):
                self.equal(details["bit_mismatch_count"], 0, "Intermediate differences", key)
            elif check == "output_parity":
                for field in ("outputs", "expected_outputs"):
                    self.equal(details[field], io[field], "Output check versus raw IO", (key, field))
            elif check == "rank_consistency":
                self.equal(details["rank_count"], n, "Rank consistency check count", key)
            elif check == "ledger":
                count = 2*n*(n-1) if active else 0
                wire = sum(m["wire_bytes"] for m in self.protocols[row["case_id"]][1].values())
                for field in ("expected_messages", "actual_messages"):
                    self.equal(details[field], count, "Ledger check count", key)
                for field in ("expected_wire_bytes", "actual_wire_bytes"):
                    self.equal(details[field], wire, "Ledger check wire sum", key)
            elif check == "threshold_consumption":
                count = n*n if active else 0
                for field in ("expected_count", "actual_count"):
                    self.equal(details[field], count, "Threshold check count", key)
                self.equal(Counter(details["event_keys"]), Counter(self.protocols[row["case_id"]][0].keys()),
                           "Threshold check consumed identities", key)
            elif check == "immutability":
                for field in ("inputs_before", "inputs_after", "lut_before", "lut_after"):
                    self.equal(details[field], io[field], "Immutability check raw evidence", (key, field))
                self.equal(details["thresholds_unchanged"], True, "Threshold immutability flag", key)
            diagnostic = row["diagnostic"]
            reference = "original_fp32_fp64_mean" if case["entry_mode"] == "raw_fp32" else "predivided_bf16_fp64_sum"
            self.equal(diagnostic["reference"], reference, "Diagnostic reference identity", key)
            self.check(isinstance(diagnostic["squared_error"], (int, float)) and diagnostic["squared_error"] >= 0,
                       "Invalid diagnostic squared error", key)
            if d == 0:
                self.equal((diagnostic["squared_error"], diagnostic["mse"], diagnostic["max_abs_error"]),
                           (0.0, None, None), "Empty diagnostic convention", key)
            else:
                self.check(diagnostic["mse"] >= 0 and diagnostic["max_abs_error"] >= 0, "Negative diagnostic error", key)
                self.check(math.isclose(diagnostic["squared_error"]/d, diagnostic["mse"], rel_tol=1e-12, abs_tol=1e-30),
                           "Diagnostic MSE denominator", key)
            self.equal(diagnostic, actual[f"cuda_logical_ring/baseline/{row['case_id']}/output_parity"]["diagnostic"],
                       "Repeated result diagnostic differs", key)
        self.correctness_rows = actual
        self.counts["expected_correctness_records"] = len(wanted)

    def rejections(self):
        invalid = rows(self.run / "invalid_checks.jsonl")
        self.common(invalid, "invalid_checks.jsonl")
        indexed = self.unique_index(invalid, lambda r: r["record_id"], "invalid check")
        self.equal(set(indexed), {"invalid/" + name for name in REJECTIONS}, "Invalid-input required coverage")
        frozen = {r["record_id"]: r for r in self.case_manifest["cases"]}
        for name, message in REJECTIONS.items():
            key = "invalid/" + name
            row = indexed[key]
            self.equal(row, self.correctness_rows[key], "Invalid check dual records differ", key)
            for record in (frozen[key], row):
                self.equal(record["expected_exception_type"], "ValueError", "Expected rejection class", key)
                self.equal(record["expected_message"], message, "Frozen exact rejection rule", key)
            self.equal(row["observed_exception_type"], "ValueError", "Unexpected error cannot count as a rejection pass", key)
            self.equal(row["observed_exception_message"], message, "Exact rejection message mismatch", key)
            self.equal(row["returned_result"], False, "Invalid collective returned a usable partial result", key)
            calls = 2 if name == "late_threshold_shape" else 1 if name.startswith("threshold_") else 0
            self.equal(row["threshold_calls_before_rejection"], calls, "Rejection stage/threshold count", key)
        self.counts["expected_rejections"] = len(REJECTIONS)

    def lifetimes(self):
        records = self.unique_index(self.raw["buffer_lifetime_checks.jsonl"], lambda r: r["record_id"], "buffer check")
        wanted = {f"buffer/b{b}/{check}" for b in (4, 8)
                  for check in ("two_packets_in_flight", "decode_output_reuse", "cross_collective_isolation")}
        self.equal(set(records), wanted, "Required buffer lifetime evidence coverage")
        for key, row in records.items():
            self.equal(row, self.correctness_rows[key], "Lifetime check dual records differ", key)
            details = row["details"]
            bits = row["nominal_bits"]
            for label in ("A", "B"):
                threshold = row["thresholds_" + label]
                seed = int.from_bytes(hashlib.sha256(
                    f"gpu_codec_v1:logical_ring_v1:ownership_{label}_b{bits}".encode()).digest()[:8], "little") % 2**63
                self.equal(threshold["seed"], seed, "Lifetime threshold seed identity", (key, label))
                self.identity(threshold["main"], (key, label), dtype="torch.bfloat16", shape=[64, 16])
                self.identity(threshold["scale"], (key, label), dtype="torch.bfloat16", shape=[4, 16])
            if row["check_type"] == "two_packets_in_flight":
                self.equal(details["packet_A_before"], details["packet_A_after_B"], "In-flight A overwritten by B", key)
                for label, field in (("A", "packet_A_after_B"), ("B", "packet_B")):
                    self.packet(details[field], 1024, bits, (key, label))
                    self.equal(details[field]["wire_sha256"], details["expected_wire_" + label], "Lifetime CPU packet identity", (key, label))
                pointers = details["storage_pointers"]
                self.equal(set(pointers), {"A", "B", "plan"}, "Packet storage pointer coverage", key)
                flattened = [pointers[label][component] for label in ("A", "B", "plan") for component in COMPONENTS]
                self.check(all(type(p) is int and p > 0 for p in flattened), "Invalid device storage pointer", key)
                self.equal(len(set(flattened)), 9, "Owned packet buffers alias live storage", key)
                self.equal(details["independent_packet_storage"], True, "Packet storage independent flag", key)
                self.identity_equal(details["decoded_A"], details["expected_decoded_A"], (key, "A receive after B"))
            elif row["check_type"] == "decode_output_reuse":
                self.equal(details["plan_output_storage_reused"], True, "Output reuse probe did not exercise reuse", key)
                self.equal(details["plan_output_pointer"], details["decoded_B_plan_pointer"], "Plan output pointer not reused", key)
                pointers = [details["plan_output_pointer"], details["saved_A_pointer"], details["saved_B_pointer"]]
                self.check(all(type(p) is int and p > 0 for p in pointers), "Invalid saved output pointer", key)
                self.equal(len(set(pointers)), 3, "Saved decode output aliases reusable output", key)
                self.identity_equal(details["decoded_A_saved"], details["expected_A"], (key, "saved A"))
                self.identity_equal(details["decoded_B"], details["expected_B"], (key, "B"))
                self.check(details["decoded_A_saved"]["sha256"] != details["decoded_B"]["sha256"], "Decode reuse fixture is not discriminating", key)
                self.equal((details["saved_A_bit_mismatch_count"], details["decoded_B_bit_mismatch_count"]), (0, 0), "Decode reuse differences", key)
            else:
                a, b = details["case_A"], details["case_B"]
                self.check(a != b and a in self.cases and b in self.cases, "Cross-collective distinct cases missing", key)
                self.equal(details["execution_order"], ["A", "B", "A"], "Cross-collective execution order", key)
                for field in ("first_A", "first_A_after_B", "second_A"):
                    self.equal(details[field], details["expected_A"], "Cross-collective pollution", (key, field))
                expected_signature = {
                    "outputs": self.io_rows[(a, "cpu_reference", "baseline")]["outputs"],
                    "events": {event: self.event_rows[("cpu_reference", "baseline", event)]["packet"] for event in self.protocols[a][0]},
                    "intermediates": {item: self.intermediate_rows[("cpu_reference", "baseline", item)]["actual"] for item in self.protocols[a][2]}}
                self.equal(details["expected_A"], expected_signature, "Cross-collective expected signature not from frozen CPU case", key)
                self.equal(details["B_outputs"], self.io_rows[(b, "cpu_reference", "baseline")]["outputs"], "Cross-collective B output parity", key)

    def delivery(self):
        registration = self.registration_data
        self.equal(sha(self.run / "source_manifest.json"), registration["source_manifest_sha256"],
                   "Frozen source manifest changed")
        source_manifest = read(self.run / "source_manifest.json")
        self.unique_index(source_manifest["files"], lambda r: r["path"], "source manifest file")
        for item in source_manifest["files"]:
            self.equal(sha(self.run / item["path"]), item["sha256"], "Bundled source/binary changed", item["path"])
            self.equal((self.run / item["path"]).stat().st_size, item["bytes"], "Bundled source/binary size", item["path"])
            self.equal(sha(item["original_path"]), item["sha256"], "Original frozen dependency changed", item["path"])
        start = read(self.run / "execution_start.json")
        runner = read(self.run / "runner_summary.json")
        self.equal(start["formal"], True, "Formal evidence was only a development attempt")
        self.equal(runner["formal"], True, "Runner summary was only a development attempt")
        self.equal(runner["status"], "pass", "Formal runner did not finish successfully")
        self.equal(runner["failed"], [], "Formal runner preserved failed checks")
        self.equal(runner["record_count"], len(self.correctness_rows), "Runner reported record count")
        self.equal(runner["expected_record_count"], len(self.correctness_rows), "Runner expected record count")
        self.equal(runner["attempt"], start["attempt"], "Formal attempt identity")
        self.equal(runner["contract_sha256"], self.contract_sha256, "Runner contract identity")
        self.equal(start["contract_sha256"], self.contract_sha256, "Execution start contract identity")
        self.equal(start["build_manifest_sha256"], registration["build_manifest_sha256"], "Start build identity")
        self.equal(runner["loaded_module_sha256"], self.contract["module_sha256"], "Runner actually loaded binary")
        self.equal(sha(runner["loaded_module_path"]), self.contract["module_sha256"], "Loaded module file changed")
        self.equal(start["build_manifest"], read(self.run / self.contract["build_manifest"]), "Start build snapshot")
        for name, digest in start["source_hashes"].items():
            name = Path(name).as_posix()
            self.equal(digest, registration["sources"][name], "Execution start source identity", name)
        self.check(datetime.fromisoformat(start["started_utc"]) >= datetime.fromisoformat(registration["frozen_at"]),
                   "Formal execution preceded source freeze")
        self.equal(registration["phase"], "before_formal_correctness", "Registration phase")
        self.check("--formal" in start["argv"] and "--case-id" not in start["argv"], "Formal command used a subset")
        summary = self.summary
        for field, expected in (("schema_version", "logical_ring_v1.0"), ("template_only", False),
                                ("run_id", self.run.name), ("stage", "G2a"),
                                ("scope", "fixed_b4_b8_explicit_thresholds_single_gpu_logical_ring"),
                                ("execution_status", "complete"), ("correctness_status", "pass"),
                                ("full_g2_acceptance", "incomplete"), ("next_action", "review_required"),
                                ("contract_sha256", self.contract_sha256),
                                ("source_manifest_sha256", sha(self.run / "source_manifest.json"))):
            self.equal(summary[field], expected, "Delivery summary identity/scope", field)
        acceptance = summary["g2a_acceptance"]
        self.check(acceptance in ("incomplete", "pass"), "Unexpected G2a acceptance state")
        for field in ("message_ledger_status", "buffer_lifetime_status"):
            self.equal(summary[field], acceptance, "Summary acceptance states disagree", field)
        self.equal(summary["claims"]["logical_ring_parity_validated"], acceptance == "pass", "Logical parity claim state")
        for field in ("full_dynamiq_gpu_validated", "training_quality_newly_validated", "multigpu_speedup_measured"):
            self.equal(summary["claims"][field], False, "Out-of-scope result claimed", field)
        baseline = [r for r in self.raw["message_ledger.jsonl"]
                    if r["method_id"] == "cuda_logical_ring" and r["variant"] == "baseline"]
        counts = {"collective_cases": len(self.cases), "real_collective_cases": 6, "real_snapshots": 3,
                  "correctness_records": len(self.correctness_rows),
                  "correctness_status_counts": {"pass": len(self.correctness_rows)},
                  "message_ledger_records": len(self.raw["message_ledger.jsonl"]),
                  "quantization_event_records": len(self.raw["quantization_events.jsonl"]),
                  "intermediate_records": len(self.raw["intermediate_checks.jsonl"]),
                  "buffer_lifetime_records": len(self.raw["buffer_lifetime_checks.jsonl"]),
                  "frozen_threshold_events": len(self.threshold_by_key),
                  "gpu_baseline_sends": len(baseline),
                  "gpu_baseline_sent_wire_bytes": sum(r["wire_bytes"] for r in baseline)}
        self.equal(summary["coverage"], counts, "Summary coverage differs from independent raw recount")
        self.counts.update(counts)
        replays = self.unique_index(summary["real_replays"], lambda r: r["case_id"], "summary real replay")
        self.equal(set(replays), {k for k, c in self.cases.items() if c["source_kind"] == "real_gradient"},
                   "Summary real replay coverage")
        for key, record in replays.items():
            case = self.cases[key]
            for field in ("logical_numel", "padded_numel", "chunk_numel", "nominal_bits"):
                self.equal(record[field], case[field], "Real replay summary dimensions", (key, field))
            messages = self.protocols[key][1]
            self.equal(record["message_count"], len(messages), "Real replay summary message count", key)
            self.equal(record["sent_wire_bytes"], sum(m["wire_bytes"] for m in messages.values()), "Real replay summary wire bytes", key)
            self.equal(record["diagnostic"], self.correctness_rows[f"cuda_logical_ring/baseline/{key}/output_parity"]["diagnostic"],
                       "Real replay diagnostic source", key)
        if acceptance == "pass":
            self.check(isinstance(summary["audit_proof"], str), "Accepted summary lacks prior audit proof")
            proof = read(self.run / summary["audit_proof"])
            self.equal(proof["status"], "pass", "Prior audit did not pass")
            self.equal(proof["validation_error_count"], 0, "Prior audit has errors")
            self.equal(proof["auditor_sha256"], sha(__file__), "Prior audit generator changed")
            self.equal(proof["run_id"], self.run.name, "Prior audit run differs")
        else:
            self.equal(summary["audit_proof"], None, "Draft summary should not claim an audit proof")
        prior = ROOT / "results/gpu_codec_v1/gpu_codec_v1_20261104_001"
        for item in read(prior / "artifact_integrity.json")["files"]:
            self.equal(sha(prior / item["path"]), item["sha256"], "Sealed G1 file changed", item["path"])
            self.equal((prior / item["path"]).stat().st_size, item["bytes"], "Sealed G1 file size changed", item["path"])
        self.counts["prior_g1_files_rechecked"] = len(read(prior / "artifact_integrity.json")["files"])
        # Strictly parse all run JSON, including preserved failures. Development
        # failures remain evidence and do not masquerade as formal pass rows.
        for path in self.run.rglob("*.json"): read(path)
        for path in self.run.rglob("*.jsonl"): rows(path)

    def execute(self):
        required = ("REPORT.md", "summary.json", "ring_contract.json", "input_manifest.json",
                    "threshold_manifest.json", "correctness_case_manifest.json", "execution_registration.json",
                    "correctness.jsonl", "message_ledger.jsonl", "quantization_events.jsonl",
                    "intermediate_checks.jsonl", "buffer_lifetime_checks.jsonl", "input_output_checks.jsonl",
                    "environment.json", "REPRODUCE.md", "issues.jsonl", "invalid_checks.jsonl",
                    "source_manifest.json", "execution_start.json", "runner_summary.json", "BUFFER_OWNERSHIP.md")
        for name in required: self.check((self.run / name).is_file(), "Required file missing", name)
        self.stage("registration", self.registration)
        self.stage("manifests", self.manifests)
        self.stage("actual CPU fixtures and thresholds", self.fixture_tensors)
        self.stage("raw records", self.raw_records)
        self.stage("quantization events", self.events)
        self.stage("directed message ledger", self.messages)
        self.stage("intermediate values", self.intermediates)
        self.stage("inputs/outputs/immutability", self.input_outputs)
        self.stage("correctness coverage", self.correctness)
        self.stage("exact input rejections", self.rejections)
        self.stage("buffer ownership/lifetime", self.lifetimes)
        self.stage("delivery/source/summary", self.delivery)
        audited_files = {path.relative_to(self.run).as_posix(): sha(path)
                         for path in sorted(self.run.rglob("*")) if path.is_file()}
        return {"schema_version": "logical_ring_audit_v1.0", "run_id": self.run.name,
                "status": "pass" if not self.errors else "fail", "validation_error_count": len(self.errors),
                "errors": self.errors, "check_count": self.check_count, "counts": self.counts,
                "auditor_sha256": sha(__file__), "generated_at": datetime.now(timezone.utc).isoformat(),
                "contract_sha256": getattr(self, "contract_sha256", None),
                "execution_registration_sha256": sha(self.run / "execution_registration.json")
                    if (self.run / "execution_registration.json").is_file() else None,
                "audited_summary_acceptance": getattr(self, "summary", {}).get("g2a_acceptance"),
                "audited_files": audited_files,
                "audit_execution_device": "CPU only; no ring or CUDA implementation imported",
                "command": [sys.executable, *sys.argv]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", help="New audit JSON path relative to run; existing files are never overwritten")
    args = parser.parse_args()
    run = (ROOT / args.run_dir).resolve()
    if not run.is_relative_to(ROOT / "results/logical_ring_v1"):
        raise ValueError("Expected a new logical_ring_v1 run")
    result = Audit(run).execute()
    if args.output:
        destination = (run / args.output).resolve()
        if not destination.is_relative_to(run): raise ValueError("Audit output escapes run")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    raise SystemExit(0 if result["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
