"""Independently validate and summarize bounded D1 records; never run a codec.

No input tensors or decoded tensors are loaded by default.  Raw FP64 algebra,
coverage, actual packet accounting, and four-worker linkage are checked here.
Existing outputs are never overwritten: subsequent invocations create revisions.
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
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
SHA = re.compile(r"^[0-9a-f]{64}$")
SOURCE_FILES = ("metrics_raw.jsonl", "packet_accounting.jsonl", "experiment_contract.json", "input_manifest.json")
LOCAL_FIELDS = ("squared_error", "mse", "input_energy", "normalized_squared_error", "max_abs_error")
AGG_FIELDS = ("aggregate_squared_error", "aggregate_mse", "source_energy", "target_energy",
              "source_normalized_error", "target_relative_error", "cancellation_ratio",
              "local_error_sum", "cross_term", "aggregate_max_abs_error", "one_shot_packet_bytes_sum")
PASS = "pass"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""): h.update(block)
    return h.hexdigest()


def strict_json(text):
    def reject(token): raise ValueError("Nonfinite JSON value: " + token)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result: raise ValueError("Duplicate JSON object key: " + key)
            result[key] = value
        return result

    value = json.loads(text, parse_constant=reject, object_pairs_hook=unique)
    def finite(item):
        if isinstance(item, float) and not math.isfinite(item): raise ValueError("Nonfinite JSON number")
        if isinstance(item, dict):
            for child in item.values(): finite(child)
        elif isinstance(item, list):
            for child in item: finite(child)
    finite(value)
    return value


def read(path): return strict_json(Path(path).read_text(encoding="utf-8"))
def rows(path): return [strict_json(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
def identity_family(method): return method["family"] == "bf16_identity" or method["method_id"] == "bf16_identity"
def finite_number(value):
    try: return type(value) in (int, float) and math.isfinite(value)
    except OverflowError: return False
def safe(value):
    if isinstance(value, (set, tuple, list)): return [safe(x) for x in sorted(value, key=str)] if isinstance(value, set) else [safe(x) for x in value]
    if isinstance(value, dict): return {str(k): safe(v) for k, v in value.items()}
    return value


def statistics_of(values):
    defined = [float(value) for value in values if value is not None]
    return dict(valid_value_count=len(defined), undefined_value_count=len(values)-len(defined),
                mean=statistics.mean(defined) if defined else None,
                sample_std=statistics.stdev(defined) if len(defined)>1 else None,
                min=min(defined) if defined else None, max=max(defined) if defined else None)


class Analysis:
    def __init__(self, run):
        self.run = run
        self.errors, self.missing, self.failed = [], [], []
        self.checks = 0
        self.source_sha256 = {name: sha(run/name) for name in SOURCE_FILES}
        self.contract = read(run/"experiment_contract.json")
        self.manifest = read(run/"input_manifest.json")
        self.raw = rows(run/"metrics_raw.jsonl")
        self.accounting = rows(run/"packet_accounting.jsonl")
        self.contract_sha256 = self.source_sha256["experiment_contract.json"]
        self.input_manifest_sha256 = self.source_sha256["input_manifest.json"]
        self.optional_artifacts = {}

    def check(self, condition, message, key=None):
        self.checks += 1
        if not condition: self.errors.append(dict(message=message, identity=safe(key)))
        return bool(condition)

    def equal(self, actual, expected, message, key=None):
        return self.check(actual == expected, message, dict(key=safe(key), actual=safe(actual), expected=safe(expected)))

    def close(self, actual, expected, message, key=None):
        if expected is None: return self.equal(actual, None, message, key)
        return self.check(finite_number(actual) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-300),
                          message, dict(key=safe(key), actual=actual, expected=expected))

    def index(self, records, key_function, label):
        indexed = {}
        for record in records:
            key = key_function(record)
            self.check(key not in indexed, "Duplicate " + label, key)
            indexed[key] = record
        return indexed

    def stage(self, label, action):
        try: action()
        except Exception as error:
            self.errors.append(dict(message="Validation stage did not finish: " + label,
                                    exception_type=type(error).__name__, detail=str(error)))

    def common(self, row, key, input_hash=True):
        self.equal(row["run_id"], self.run.name, "Run identity mismatch", key)
        self.equal(row["contract_sha256"], self.contract_sha256, "Contract identity mismatch", key)
        if input_hash:
            self.equal(row["input_manifest_sha256"], self.input_manifest_sha256, "Input manifest identity mismatch", key)

    def setup(self):
        self.equal(self.contract["run_id"], self.run.name, "Contract run identity")
        self.equal(self.manifest["run_id"], self.run.name, "Input manifest run identity")
        self.equal(self.contract["input_manifest_sha256"], self.input_manifest_sha256, "Frozen input manifest changed")
        if "source_manifest_sha256" in self.contract:
            self.equal(sha(self.run/"source_manifest.json"), self.contract["source_manifest_sha256"], "Frozen source manifest changed")
            self.optional_artifacts["source_manifest.json"] = sha(self.run/"source_manifest.json")
        if "qualification_sha256" in self.contract:
            qualification_path = self.run/"baseline_qualification.json"
            self.equal(sha(qualification_path), self.contract["qualification_sha256"], "Frozen D0 qualification changed")
            self.equal(read(qualification_path)["gate_decision"], "pass", "D1 comparison has no passing D0 gate")
            self.optional_artifacts["baseline_qualification.json"] = sha(qualification_path)
        self.methods = self.index(self.contract["methods"], lambda r: r["method_id"], "contract method")
        self.inputs = self.index(self.manifest["inputs"], lambda r: r["input_id"], "input")
        self.snapshots = defaultdict(list)
        for key, item in self.inputs.items():
            self.equal(item["training_seed"], 1101, "Unexpected training trajectory", key)
            self.check(item["update_step"] in (100, 500, 2500), "Unexpected development snapshot", key)
            self.equal(item["partition"], "development", "Input partition must remain development", key)
            self.equal(item["logical_numel"], 818048, "Logical input was sliced or old padding was included", key)
            self.equal(item["input_dtype"], "torch.bfloat16", "Input dtype", key)
            self.equal(item["already_divided_by_world_size"], True, "Input predivision identity", key)
            self.snapshots[item["snapshot_id"]].append(item)
        self.equal(len(self.inputs), 12, "Expected twelve worker tensors")
        self.equal(len(self.snapshots), 3, "Expected three development snapshots")
        for snapshot, group in self.snapshots.items():
            group.sort(key=lambda r: r["worker_id"])
            self.equal([r["worker_id"] for r in group], list(range(4)), "Missing/duplicate worker", snapshot)
            self.equal(len({r["update_step"] for r in group}), 1, "Mixed update steps in one snapshot", snapshot)
        self.expected = set()
        for method_id, method in self.methods.items():
            self.check(type(method["deterministic"]) is bool, "Method deterministic flag missing", method_id)
            wanted = 1 if method["deterministic"] else 8
            self.equal(method["repeats"], wanted, "Repeat count differs from frozen bounded plan", method_id)
            if identity_family(method):
                self.equal(method["deterministic"], True, "BF16 identity must not fabricate random repetitions", method_id)
            for repeat in range(wanted):
                for item in self.inputs.values():
                    self.expected.add(("local", item["snapshot_id"], item["worker_id"], method_id, repeat))
                for snapshot in self.snapshots:
                    self.expected.add(("one_shot_codec_sum", snapshot, None, method_id, repeat))

    def packets(self):
        self.packets_by_id = self.index(self.accounting, lambda r: r["packet_id"], "packet identity")
        self.packets_by_key = self.index(self.accounting,
            lambda r: (r["snapshot_id"], r["worker_id"], r["method_id"], r["repeat_id"]), "packet execution")
        for packet_id, row in self.packets_by_id.items():
            self.common(row, packet_id, input_hash=False)
            source = self.inputs[row["input_id"]]
            method = self.methods[row["method_id"]]
            self.equal((row["snapshot_id"], row["worker_id"]), (source["snapshot_id"], source["worker_id"]),
                       "Packet source/worker identity", packet_id)
            self.check(("local", row["snapshot_id"], row["worker_id"], row["method_id"], row["repeat_id"]) in self.expected,
                       "Packet outside frozen comparison matrix", packet_id)
            self.equal(row["logical_numel"], source["logical_numel"], "Packet logical length", packet_id)
            self.equal(row["padded_numel"], source["padding_by_method"][row["method_id"]], "Packet padding differs from contract", packet_id)
            fields = ("payload_bytes", "scale_bytes", "header_bytes", "alignment_bytes", "packet_total_bytes",
                      "per_input_auxiliary_bytes", "setup_or_persistent_state_bytes")
            self.check(all(type(row[name]) is int and row[name]>=0 for name in fields), "Noninteger/negative packet accounting", packet_id)
            total = sum(row[name] for name in ("payload_bytes", "scale_bytes", "header_bytes", "alignment_bytes"))
            self.equal(row["packet_total_bytes"], total, "Actual packet components do not sum", packet_id)
            self.check(row["per_input_auxiliary_bytes"] <= total, "Auxiliary bytes not included in packet components", packet_id)
            self.close(row["effective_bits_per_logical_element"], 8*total/row["logical_numel"] if row["logical_numel"] else None,
                       "Effective bits per logical element", packet_id)
            self.check(isinstance(row["packet_sha256"], str) and bool(SHA.fullmatch(row["packet_sha256"])), "Packet SHA256 missing", packet_id)
            self.check(type(row["independent_decode_pass"]) is bool, "Independent decoder result missing", packet_id)
            self.check(isinstance(row["component_shapes_dtypes"], dict) and bool(row["component_shapes_dtypes"]),
                       "Actual component shape/dtype evidence missing", packet_id)
            self.check(isinstance(row["byte_measurement_method"], str) and bool(row["byte_measurement_method"]),
                       "Actual byte measurement method missing", packet_id)
            if "packet_total_bytes" in method:
                self.equal(total, method["packet_total_bytes"], "Packet budget changed after configuration freeze", packet_id)
            if method["family"] == "block_uniform":
                p, q, g = row["padded_numel"], method["stored_code_bits"], method["group_size"]
                self.equal((row["payload_bytes"], row["scale_bytes"], row["header_bytes"], row["alignment_bytes"]),
                           ((p*q+7)//8, 4*(p//g), 16, 0), "Uniform actual wire format", packet_id)
            elif identity_family(method):
                self.equal((row["payload_bytes"], row["scale_bytes"], row["header_bytes"], row["alignment_bytes"]),
                           (2*row["logical_numel"], 0, 0, 0), "BF16 identity actual wire format", packet_id)

    def raw_records(self):
        self.records = self.index(self.raw, lambda r: (r["record_type"], r["snapshot_id"], r["worker_id"],
                                                    r["method_id"], r["repeat_id"]), "raw measurement")
        observed = set(self.records)
        self.missing = sorted(self.expected-observed, key=str)
        self.check(not (observed-self.expected), "Unexpected raw identities outside frozen matrix", observed-self.expected)
        for key, row in self.records.items():
            self.common(row, key)
            self.check(row["status"] in ("pass", "fail", "skipped", "not_run"), "Unknown raw status", key)
            self.check(type(row["repeat_id"]) is int, "Repeat ID is not an integer", key)
            if row["record_type"] == "local":
                source = self.inputs[row["input_id"]]
                self.equal((row["snapshot_id"], row["worker_id"]), (source["snapshot_id"], source["worker_id"]),
                           "Raw input identity, including failed records", key)
            elif row["record_type"] == "one_shot_codec_sum":
                self.equal(Counter(row["input_ids"]), Counter(r["input_id"] for r in self.snapshots[row["snapshot_id"]]),
                           "Raw aggregate input identity, including failed records", key)
            if row["status"] != PASS:
                self.failed.append(dict(identity=safe(key), status=row["status"], reason=row.get("failure_reason")))
                self.check(bool(row.get("failure_reason")), "Failed/skipped row has no reason", key)
                continue
            self.equal(row.get("failure_reason"), None, "Passing row carries failure reason", key)
            self.check(bool(row.get("rng_state_id")), "Random/deterministic state identity missing", key)
            if row["record_type"] == "local": self.local(row, key)
            elif row["record_type"] == "one_shot_codec_sum": self.aggregate(row, key)
            else: self.check(False, "Unknown record type", key)
        referenced = {r["packet_id"] for r in self.raw if r["record_type"] == "local" and r.get("packet_id") is not None}
        self.equal(set(self.packets_by_id)-referenced, set(), "Packet has no local record")
        input_energies, target_energies = {}, {}
        for key, row in self.records.items():
            if row["status"] != PASS: continue
            if row["record_type"] == "local":
                identity = row["input_id"]
                reference = input_energies.setdefault(identity, row["input_energy"])
                self.close(row["input_energy"], reference, "Fixed input energy changed across methods/repeats", key)
            elif row["record_type"] == "one_shot_codec_sum":
                identity = row["snapshot_id"]
                reference = target_energies.setdefault(identity, row["target_energy"])
                self.close(row["target_energy"], reference, "Fixed target energy changed across methods/repeats", key)

    def numeric(self, row, fields, key, nonnegative=True):
        for field in fields:
            value = row[field]
            self.check(finite_number(value), "Missing/nonfinite FP64 metric " + field, key)
            if nonnegative: self.check(value >= 0, "Negative metric " + field, key)

    def local(self, row, key):
        source = self.inputs[row["input_id"]]
        self.equal((row["snapshot_id"], row["worker_id"]), (source["snapshot_id"], source["worker_id"]), "Local input identity", key)
        d = source["logical_numel"]
        self.equal(row["logical_numel"], d, "Local metric includes padding or a slice", key)
        self.numeric(row, ("squared_error", "input_energy", "max_abs_error"), key)
        self.equal(row["nonfinite_count"], 0, "Passing local output contains nonfinite values", key)
        self.close(row["mse"], row["squared_error"]/d if d else None, "Local MSE denominator", key)
        energy = row["input_energy"]
        self.close(row["normalized_squared_error"], row["squared_error"]/energy if energy else None, "Local normalization denominator", key)
        if not energy: self.check(bool(row["undefined_metric_reason"]), "Undefined local normalization has no reason", key)
        maximum_squared = row["max_abs_error"]**2
        self.check(row["squared_error"] >= maximum_squared*(1-1e-10), "Local SSE below maximum error squared", key)
        self.check(row["squared_error"] <= d*maximum_squared*(1+1e-10), "Local SSE above max-error bound", key)
        packet = self.packets_by_id[row["packet_id"]]
        self.equal((packet["snapshot_id"], packet["worker_id"], packet["method_id"], packet["repeat_id"]),
                   key[1:], "Local metric linked to wrong packet", key)
        self.equal(packet["input_id"], row["input_id"], "Local metric packet input identity", key)
        self.equal(row["packet_total_bytes"], packet["packet_total_bytes"], "Local packet bytes differ from actual accounting", key)
        self.equal(packet["independent_decode_pass"], True, "Passing local metric lacks independent packet decode", key)
        if identity_family(self.methods[row["method_id"]]):
            for field in ("squared_error", "mse", "max_abs_error"):
                self.equal(row[field], 0, "BF16 identity is not exact", (key, field))
        optional_path = row.get("decoded_output_path")
        optional_hash = row.get("decoded_output_sha256", row.get("decoded_output_hash"))
        if optional_path is not None or optional_hash is not None:
            self.check(isinstance(optional_path, str) and isinstance(optional_hash, str),
                       "Optional decoded artifact needs both path and SHA256", key)
            path = Path(optional_path)
            if not path.is_absolute(): path = self.run/path
            self.equal(sha(path), optional_hash, "Saved decoded output artifact changed", key)
            self.optional_artifacts[str(path)] = optional_hash

    def aggregate(self, row, key):
        snapshot, method_id, repeat = row["snapshot_id"], row["method_id"], row["repeat_id"]
        inputs = self.snapshots[snapshot]
        self.equal(row["worker_count"], 4, "One-shot aggregation must reference four workers", key)
        self.equal(row["worker_id"], None, "Aggregate worker ID must be null", key)
        self.equal(Counter(row["input_ids"]), Counter(r["input_id"] for r in inputs), "Aggregate input coverage", key)
        locals_ = [self.records[("local", snapshot, worker, method_id, repeat)] for worker in range(4)]
        self.check(all(r["status"] == PASS for r in locals_), "Passing aggregate uses a failed local output", key)
        packet_ids = [r["packet_id"] for r in locals_]
        self.equal(Counter(row["packet_ids"]), Counter(packet_ids), "Aggregate did not reuse the same four local packets", key)
        self.equal(len(set(row["packet_ids"])), 4, "Duplicate aggregate packet", key)
        self.equal(row["one_shot_packet_bytes_sum"], sum(self.packets_by_id[p]["packet_total_bytes"] for p in packet_ids),
                   "One-shot packet byte sum or spurious ring multiplier", key)
        self.equal(row["logical_numel"], inputs[0]["logical_numel"], "Aggregate logical length", key)
        self.numeric(row, ("aggregate_squared_error", "aggregate_mse", "source_energy", "target_energy",
                           "local_error_sum", "aggregate_max_abs_error"), key)
        self.numeric(row, ("cross_term",), key, nonnegative=False)
        self.close(row["source_energy"], math.fsum(r["input_energy"] for r in locals_), "Aggregate source energy differs from four locals", key)
        self.close(row["local_error_sum"], math.fsum(r["squared_error"] for r in locals_), "Aggregate local SSE sum differs from four locals", key)
        sse, source, target = row["aggregate_squared_error"], row["source_energy"], row["target_energy"]
        self.close(row["aggregate_mse"], sse/row["logical_numel"], "Aggregate MSE denominator", key)
        self.close(row["source_normalized_error"], sse/source if source else None, "Aggregate source normalization", key)
        self.close(row["target_relative_error"], sse/target if target else None, "Aggregate target normalization", key)
        ratio = target/source if source else None
        self.close(row["cancellation_ratio"], ratio, "Cancellation ratio denominator", key)
        self.equal(row["near_cancellation"], ratio is not None and ratio < 1e-6, "Near-cancellation flag", key)
        self.close(row["cross_term"], sse-row["local_error_sum"], "Cross term identity", key)
        self.check(sse <= 4*row["local_error_sum"]*(1+1e-10), "Four-worker SSE exceeds Cauchy bound", key)
        self.check(target <= 4*source*(1+1e-10), "Target energy exceeds four-worker Cauchy bound", key)
        if not source or not target: self.check(bool(row["undefined_metric_reason"]), "Undefined aggregate normalization has no reason", key)
        self.check(row["aggregate_comparability_status"] in ("valid", "invalid", "incomplete", "not_run"),
                   "Aggregate comparability status missing/unknown", key)
        if identity_family(self.methods[method_id]):
            for field in ("aggregate_squared_error", "aggregate_mse", "local_error_sum", "cross_term", "aggregate_max_abs_error"):
                self.equal(row[field], 0, "BF16 identity one-shot FP64 diagnostic must be exact", (key, field))

    def validate(self):
        self.stage("contract and input coverage", self.setup)
        self.stage("actual packet accounting", self.packets)
        self.stage("raw metrics and four-worker linkage", self.raw_records)
        return self

    def recompute_decoded(self):
        """Recompute in CPU FP64 from trusted input fixtures and saved BF16 tensors."""
        import torch
        self.recomputed = dict(enabled=True, local_rows_checked=0, aggregate_rows_checked=0,
                               input_tensors_checked=0, decoded_tensors_checked=0)
        loaded = {}
        for input_id, item in self.inputs.items():
            path = Path(item["fixture_path"])
            self.equal(sha(path), item["fixture_sha256"], "Input fixture archive changed before recomputation", input_id)
            fixture = torch.load(path, map_location="cpu", weights_only=False)
            value = fixture["input_logical"]
            self.equal((str(value.dtype), list(value.shape)), ("torch.bfloat16", [item["logical_numel"]]),
                       "Recompute input must be the original logical BF16 vector", input_id)
            tensor_bytes = value.contiguous().view(torch.uint8).numpy().tobytes()
            identity = dict(shape=list(value.shape), dtype=str(value.dtype), bytes=len(tensor_bytes),
                            sha256=hashlib.sha256(tensor_bytes).hexdigest())
            self.equal(identity, item["input_identity"], "Logical input tensor identity changed", input_id)
            loaded[input_id] = value.double()
            self.recomputed["input_tensors_checked"] += 1
        for snapshot, inputs in self.snapshots.items():
            source = torch.stack([loaded[item["input_id"]] for item in inputs])
            target = source.sum(0)
            source_energy, target_energy = float(source.square().sum()), float(target.square().sum())
            for method_id, method in self.methods.items():
                for repeat in range(method["repeats"]):
                    decoded = []
                    complete = True
                    for item in inputs:
                        key = ("local", snapshot, item["worker_id"], method_id, repeat)
                        row = self.records.get(key)
                        if row is None or row["status"] != PASS:
                            complete = False
                            continue
                        path_value = row.get("decoded_output_path")
                        if not self.check(isinstance(path_value, str), "Requested recomputation lacks a saved output tensor", key):
                            complete = False
                            continue
                        path = Path(path_value)
                        if not path.is_absolute(): path = self.run/path
                        expected_hash = row.get("decoded_output_sha256", row.get("decoded_output_hash"))
                        self.equal(sha(path), expected_hash, "Decoded tensor archive changed", key)
                        value = torch.load(path, map_location="cpu", weights_only=True)
                        if not self.check(isinstance(value, torch.Tensor), "Saved output must be a torch.save Tensor", key):
                            complete = False
                            continue
                        self.equal((str(value.dtype), list(value.shape)), ("torch.bfloat16", [item["logical_numel"]]),
                                   "Saved output dtype or logical shape changed", key)
                        nonfinite = int((~torch.isfinite(value)).sum())
                        self.equal(nonfinite, row["nonfinite_count"], "Recomputed nonfinite count", key)
                        self.equal(nonfinite, 0, "Passing saved output is nonfinite", key)
                        y = value.double()
                        decoded.append(y)
                        x = loaded[item["input_id"]]
                        error = y-x
                        sse, energy = float(error.square().sum()), float(x.square().sum())
                        expected = dict(squared_error=sse, mse=sse/x.numel(), input_energy=energy,
                                        normalized_squared_error=sse/energy if energy else None,
                                        max_abs_error=float(error.abs().max()))
                        for field, metric in expected.items(): self.close(row[field], metric, "Recomputed local FP64 metric", (key, field))
                        self.recomputed["local_rows_checked"] += 1
                        self.recomputed["decoded_tensors_checked"] += 1
                    aggregate_key = ("one_shot_codec_sum", snapshot, None, method_id, repeat)
                    aggregate = self.records.get(aggregate_key)
                    if not complete or len(decoded) != 4 or aggregate is None or aggregate["status"] != PASS:
                        continue
                    ys = torch.stack(decoded)
                    error = ys.sum(0)-target
                    sse, local_sum = float(error.square().sum()), float((ys-source).square().sum())
                    ratio = target_energy/source_energy if source_energy else None
                    expected = dict(aggregate_squared_error=sse, aggregate_mse=sse/target.numel(),
                        source_energy=source_energy, target_energy=target_energy,
                        source_normalized_error=sse/source_energy if source_energy else None,
                        target_relative_error=sse/target_energy if target_energy else None,
                        cancellation_ratio=ratio, local_error_sum=local_sum, cross_term=sse-local_sum,
                        aggregate_max_abs_error=float(error.abs().max()))
                    for field, metric in expected.items(): self.close(aggregate[field], metric, "Recomputed one-shot FP64 metric", (aggregate_key, field))
                    self.recomputed["aggregate_rows_checked"] += 1

    def group_summary(self, record_type, snapshot, method_id, expected, observed, metrics,
                      *, input_id=None, worker_id=None, exclusions=None):
        method = self.methods[method_id]
        valid = [r for r in observed if r is not None and r.get("status") == PASS]
        missing = sum(r is None for r in observed)
        failures = len(observed)-missing-len(valid)
        comparability = ([r.get("aggregate_comparability_status") for r in valid]
                         if record_type == "one_shot_codec_sum" else ["valid"]*len(valid))
        comparable = bool(valid) and all(value == "valid" for value in comparability)
        complete = len(valid) == expected and missing == 0 and failures == 0
        return dict(schema_version="bounded_discovery_summary_v1", run_id=self.run.name,
            contract_sha256=self.contract_sha256, input_manifest_sha256=self.input_manifest_sha256,
            summary_id=f"{record_type}/{snapshot}/{input_id or 'all_workers'}/{method_id}",
            record_type=record_type, snapshot_id=snapshot, worker_id=worker_id, input_id=input_id,
            method_id=method_id, family=method["family"], budget_role=method["budget_role"],
            deterministic=method["deterministic"], partition="development",
            development_selected=method["family"] == "block_uniform",
            expected_repeats=expected, valid_repeats=len(valid), failed_repeats=failures, missing_repeats=missing,
            valid_repeat_ids=[r["repeat_id"] for r in valid],
            status="invalid" if self.errors else "complete" if complete else "incomplete",
            comparison_eligible=complete and comparable and not self.errors,
            aggregate_comparability_status=("valid" if comparable else
                "invalid" if comparability and all(v == "invalid" for v in comparability) else "incomplete")
                if record_type == "one_shot_codec_sum" else None,
            statistics_object=("Repeated stochastic outputs for this one fixed input, not independent training samples"
                if record_type == "local" else "Per-repeat four-worker totals for one fixed development snapshot"),
            statistics={field: statistics_of([r.get(field) for r in valid]) for field in metrics},
            failure_or_missing_details=exclusions or [
                dict(repeat_id=i, status="missing" if r is None else r["status"], reason=None if r is None else r.get("failure_reason"))
                for i, r in enumerate(observed) if r is None or r.get("status") != PASS])

    def summaries(self):
        result = []
        for input_id, item in sorted(self.inputs.items()):
            for method_id, method in self.methods.items():
                group = [self.records.get(("local", item["snapshot_id"], item["worker_id"], method_id, repeat))
                         for repeat in range(method["repeats"])]
                result.append(self.group_summary("local", item["snapshot_id"], method_id, method["repeats"], group,
                    (*LOCAL_FIELDS, "packet_total_bytes"), input_id=input_id, worker_id=item["worker_id"]))
        for snapshot, inputs in sorted(self.snapshots.items()):
            for method_id, method in self.methods.items():
                weighted, exclusions = [], []
                for repeat in range(method["repeats"]):
                    group = [self.records.get(("local", snapshot, item["worker_id"], method_id, repeat)) for item in inputs]
                    if not all(r is not None and r.get("status") == PASS for r in group):
                        absent = any(r is None for r in group)
                        weighted.append(None if absent else dict(status="fail", repeat_id=repeat))
                        exclusions.append(dict(repeat_id=repeat, status="missing" if absent else "fail",
                            worker_statuses=[dict(worker_id=item["worker_id"], status="missing" if row is None else row["status"],
                                reason=None if row is None else row.get("failure_reason")) for item, row in zip(inputs, group)]))
                        continue
                    sse = math.fsum(r["squared_error"] for r in group)
                    energy = math.fsum(r["input_energy"] for r in group)
                    logical = sum(r["logical_numel"] for r in group)
                    byte_sum = sum(r["packet_total_bytes"] for r in group)
                    weighted.append(dict(status=PASS, repeat_id=repeat, squared_error=sse, input_energy=energy,
                        weighted_normalized_squared_error=sse/energy if energy else None,
                        weighted_mse=sse/logical if logical else None,
                        max_abs_error=max(r["max_abs_error"] for r in group),
                        packet_bytes_mean_per_worker=byte_sum/len(group),
                        packet_bytes_sum=byte_sum, worker_count=len(group)))
                summary = self.group_summary("snapshot_local", snapshot, method_id, method["repeats"], weighted,
                    ("squared_error", "input_energy", "weighted_normalized_squared_error", "weighted_mse",
                     "max_abs_error", "packet_bytes_mean_per_worker", "packet_bytes_sum"), exclusions=exclusions)
                summary["aggregation_rule"] = "For each repeat first sum four worker SSE and energy, then divide; summarize these repeat values."
                summary["per_repeat"] = [r for r in weighted if r is not None and r.get("status") == PASS]
                result.append(summary)
                aggregate = [self.records.get(("one_shot_codec_sum", snapshot, None, method_id, repeat))
                             for repeat in range(method["repeats"])]
                summary = self.group_summary("one_shot_codec_sum", snapshot, method_id, method["repeats"], aggregate, AGG_FIELDS)
                summary["per_repeat"] = [dict(repeat_id=r["repeat_id"], **{field:r[field] for field in AGG_FIELDS})
                                         for r in aggregate if r is not None and r.get("status") == PASS]
                result.append(summary)
        return result

def render_figures(analysis, summaries, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    figures = []
    specifications = (
        ("snapshot_local", "packet_bytes_mean_per_worker", "weighted_normalized_squared_error",
         "Mean serialized packet per worker (MiB)", "Sum local SSE / sum input energy", "local_error_vs_actual_bytes.png"),
        ("one_shot_codec_sum", "one_shot_packet_bytes_sum", "source_normalized_error",
         "Sum of four serialized packets (MiB)", "One-shot aggregate SSE / source energy", "one_shot_error_vs_actual_bytes.png"))
    method_ids = list(analysis.methods)
    colors = {name: plt.get_cmap("tab10")(index % 10) for index, name in enumerate(method_ids)}
    snapshots = sorted(analysis.snapshots)
    for record_type, xfield, yfield, xlabel, ylabel, filename in specifications:
        selected = [r for r in summaries if r["record_type"] == record_type]
        eligible = [r for r in selected if r["comparison_eligible"] and r["statistics"][yfield]["mean"] is not None]
        if not eligible: continue
        fig, axes = plt.subplots(1, len(snapshots), figsize=(14, 5.8), sharey=True, squeeze=False)
        positives = [r["statistics"][yfield]["min"] for r in eligible if r["statistics"][yfield]["min"] > 0]
        linthresh = min(positives)/10 if positives else 1e-8
        upper = max(r["statistics"][yfield]["max"] for r in eligible)
        legend = []
        for method_id in method_ids:
            method = analysis.methods[method_id]
            higher = "higher_budget" in str(method["budget_role"])
            marker = "^" if higher else "D" if method["family"] == "official_dynamiq_fixed_codec" else "X" if identity_family(method) else "o"
            label = method_id + (" [higher budget]" if higher else "")
            legend.append(Line2D([], [], color=colors[method_id], marker=marker, linestyle="None", label=label,
                                 markerfacecolor="none" if higher else colors[method_id]))
        for axis, snapshot in zip(axes[0], snapshots):
            for row in eligible:
                if row["snapshot_id"] != snapshot: continue
                method = analysis.methods[row["method_id"]]
                color = colors[row["method_id"]]
                higher = "higher_budget" in str(method["budget_role"])
                marker = "^" if higher else "D" if method["family"] == "official_dynamiq_fixed_codec" else "X" if identity_family(method) else "o"
                xs, ys = row["statistics"][xfield], row["statistics"][yfield]
                x, y = xs["mean"]/2**20, ys["mean"]
                axis.vlines(x, ys["min"], ys["max"], color=color, alpha=.35, linewidth=5)
                deviation = ys["sample_std"] or 0.
                axis.errorbar(x, y, yerr=[[min(deviation, y)], [deviation]], fmt=marker,
                    color=color, markerfacecolor="none" if higher else color, markersize=7,
                    capsize=3, linewidth=1.2, zorder=3)
            omitted = sum(r["snapshot_id"] == snapshot and
                          (not r["comparison_eligible"] or r["statistics"][yfield]["mean"] is None) for r in selected)
            if omitted: axis.text(.02, .97, f"{omitted} incomplete / ineligible / undefined groups\nSee summary records", transform=axis.transAxes,
                                   va="top", fontsize=8, color="#8a3333")
            axis.set_title(snapshot.replace("seed1101_", ""), fontsize=11)
            axis.set_xlabel(xlabel, fontsize=9)
            axis.set_yscale("symlog", linthresh=linthresh)
            axis.set_ylim(0, upper*1.3 if upper else 1.)
            axis.grid(True, alpha=.2)
        axes[0, 0].set_ylabel(ylabel, fontsize=10)
        fig.suptitle("Fixed development snapshots; independent random repeats", fontsize=12)
        fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(.5, .055), frameon=False)
        fig.text(.5, .017, "Markers: repeat mean. Thin bars: sample SD (clipped at zero). Light spans: min-max. BF16 identity: one deterministic result.",
                 ha="center", fontsize=8)
        fig.subplots_adjust(left=.085, right=.99, top=.88, bottom=.30, wspace=.14)
        path = destination/"figures"/filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream: fig.savefig(stream, format="png", dpi=180)
        plt.close(fig)
        figures.append(dict(path=path.relative_to(analysis.run).as_posix(), sha256=sha(path),
                            record_type=record_type, x_metric=xfield, y_metric=yfield))
    return figures


def comparison_status(analysis, record_type):
    if analysis.errors or not hasattr(analysis, "records"): return "invalid"
    wanted = {key for key in analysis.expected if key[0] == record_type}
    selected = [analysis.records.get(key) for key in wanted]
    if analysis.errors: return "invalid"
    if any(row is None or row["status"] != PASS for row in selected): return "incomplete"
    if record_type == "one_shot_codec_sum":
        states = {row["aggregate_comparability_status"] for row in selected}
        if states == {"invalid"}: return "invalid"
        if states != {"valid"}: return "incomplete"
    return "valid"


def choose_output(run):
    names = ("metrics_summary.jsonl", "summary_analysis.json",
             "figures/local_error_vs_actual_bytes.png", "figures/one_shot_error_vs_actual_bytes.png")
    if not any((run/name).exists() for name in names): return run, 1
    revision = 2
    while (run/"summary_revisions"/f"revision_{revision:03d}").exists(): revision += 1
    target = run/"summary_revisions"/f"revision_{revision:03d}"
    target.mkdir(parents=True, exist_ok=False)
    return target, revision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--verify-only", action="store_true", help="Validate raw records and any root summary without writing files")
    parser.add_argument("--recompute-decoded", action="store_true", help="Reload saved CPU BF16 tensor outputs and verified logical input fixtures; recompute FP64 diagnostics")
    args = parser.parse_args()
    run = (ROOT/args.run_dir).resolve()
    if not run.is_relative_to(ROOT/"results/bounded_discovery_v1"):
        raise ValueError("Only a new bounded_discovery_v1 run is allowed")
    analysis = Analysis(run).validate()
    analysis.recomputed = dict(enabled=False, local_rows_checked=0, aggregate_rows_checked=0,
                               input_tensors_checked=0, decoded_tensors_checked=0)
    if args.recompute_decoded and not analysis.errors:
        analysis.stage("independent saved-tensor recomputation", analysis.recompute_decoded)
    summaries = analysis.summaries() if not analysis.errors else None
    existing_matches = None
    if args.verify_only and (run/"metrics_summary.jsonl").exists() and summaries is not None:
        existing_matches = rows(run/"metrics_summary.jsonl") == summaries
        analysis.check(existing_matches, "Existing root metrics_summary differs from independent recomputation")
    local_status = comparison_status(analysis, "local") if hasattr(analysis, "expected") else "invalid"
    aggregate_status = comparison_status(analysis, "one_shot_codec_sum") if hasattr(analysis, "expected") else "invalid"
    status = "invalid" if analysis.errors else "valid" if local_status == aggregate_status == "valid" else "incomplete"
    report = dict(schema_version="bounded_discovery_analysis_v1", run_id=run.name, status=status,
        validation_status="fail" if analysis.errors else "pass", validation_error_count=len(analysis.errors),
        validation_errors=analysis.errors, check_count=analysis.checks,
        local_comparison_status=local_status, aggregate_comparison_status=aggregate_status,
        source_sha256=analysis.source_sha256, optional_artifact_sha256=analysis.optional_artifacts,
        generator_path=Path(__file__).relative_to(ROOT).as_posix(), generator_sha256=sha(__file__),
        generated_at=datetime.now(timezone.utc).isoformat(), command=[sys.executable, *sys.argv],
        coverage=dict(expected_raw_records=len(getattr(analysis, "expected", [])), observed_raw_records=len(analysis.raw),
                      packet_records=len(analysis.accounting), missing_raw_records=safe(analysis.missing),
                      failed_or_skipped_records=analysis.failed, method_count=len(getattr(analysis, "methods", {})),
                      worker_inputs=len(getattr(analysis, "inputs", {})), snapshots=len(getattr(analysis, "snapshots", {}))),
        tensor_recomputation=analysis.recomputed, root_existing_summary_matches=existing_matches,
        statistical_scope="Random-repeat dispersion on three fixed development snapshots; no statistical generalization or paired cross-method randomness claim.",
        local_snapshot_aggregation="Sum worker SSE and energy within each repeat, divide once, then summarize repeat values.",
        timing_performed=False, figures=[], summary_path=None, summary_sha256=None,
        output_revision=None, verification_only=args.verify_only)
    if not args.verify_only:
        destination, revision = choose_output(run)
        report["output_revision"] = revision
        if summaries is not None and not analysis.errors:
            path = destination/"metrics_summary.jsonl"
            with path.open("x", encoding="utf-8") as stream:
                for row in summaries: stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False)+"\n")
            report["summary_path"] = path.relative_to(run).as_posix()
            report["summary_sha256"] = sha(path)
            report["figures"] = render_figures(analysis, summaries, destination)
        with (destination/"summary_analysis.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write("\n")
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return 1 if status == "invalid" else 2 if status == "incomplete" else 0


if __name__ == "__main__":
    raise SystemExit(main())
