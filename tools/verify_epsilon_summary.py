#!/usr/bin/env python3
"""Verify the published epsilon diagnostic's recorded scalar summaries.

Usage: python tools/verify_epsilon_summary.py --run-dir PATH_TO_EPSILON_RUN

Uses only the Python standard library. Reads metrics_raw.jsonl (or its lossless
.jsonl.gz counterpart), summary.json, and paired_summary.json. No files are
written. This checks consistency of recorded metrics, not tensors, CUDA kernels,
binary provenance, domain safety, training quality, or communication performance.
The fixed coverage and thresholds belong to the archived epsilon_diagnostic_v1
experiment; this is not a general validator for differently designed experiments.
"""

import argparse
import gzip
import json
import math
from pathlib import Path
import statistics
import sys


CONDITIONS = ("control_add1e7", "diagnostic_add0", "uniform_reused")
SNAPSHOTS = (
    "seed1101_update000100",
    "seed1101_update000500",
    "seed1101_update002500",
)
REGIONS = ("full", "H", "rest")
WORKERS = range(4)
REPEATS = range(8)
EXPECTED_PAIRS = 96


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    require(
        isinstance(expected, (int, float))
        and not isinstance(expected, bool)
        and math.isfinite(expected)
        and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-18),
        f"Mismatch in {label}: calculated={actual!r}, recorded={expected!r}",
    )


def read_json(path):
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def verify(run_dir):
    summary = read_json(run_dir / "summary.json")
    paired = read_json(run_dir / "paired_summary.json")
    raw_path = run_dir / "metrics_raw.jsonl"
    opener = open
    if not raw_path.is_file():
        raw_path = run_dir / "metrics_raw.jsonl.gz"
        opener = gzip.open
    with opener(raw_path, "rt", encoding="utf-8-sig") as stream:
        records = [json.loads(line) for line in stream if line.strip()]

    require(summary["execution_status"] == "complete", "Run is not complete")
    require(paired["run_id"] == summary["run_id"], "Paired run_id mismatch")
    require(paired["contract_sha256"] == summary["contract_sha256"], "Contract mismatch")
    require(len(records) == 1080, f"Expected 1080 raw records; found {len(records)}")
    local = {}
    one_shot = {}
    pair_ids = set()
    for record in records:
        require(record["run_id"] == summary["run_id"], "Raw run_id mismatch")
        require(record["contract_sha256"] == summary["contract_sha256"], "Raw contract mismatch")
        require(record["status"] == "pass", "Raw record is not pass")
        require(record["nonfinite_count"] == 0, "Raw nonfinite output count is nonzero")
        value = record["sse"]
        require(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0,
                "Invalid raw SSE")
        snapshot, worker, repeat, condition, region = (
            record["snapshot_id"], record["worker_id"], record["repeat_id"],
            record["condition_id"], record["region_id"],
        )
        require(snapshot in SNAPSHOTS and repeat in REPEATS
                and condition in CONDITIONS and region in REGIONS, "Unexpected raw coverage")
        key = (snapshot, worker, repeat, condition, region)
        if record["record_type"] == "local":
            require(worker in WORKERS, "Invalid local worker")
            require(key not in local, f"Duplicate local record: {key}")
            expected_bytes = 843628 if condition == "uniform_reused" else 875704
            require(record["packet_bytes"] == expected_bytes, "Packet budget mismatch")
            local[key] = record
            if condition == CONDITIONS[0] and region == "full":
                require(record["pair_id"] not in pair_ids, "Duplicate pair_id")
                pair_ids.add(record["pair_id"])
        else:
            require(record["record_type"] == "one_shot" and worker is None,
                    "Unexpected record_type or one-shot worker")
            require(key not in one_shot, f"Duplicate one-shot record: {key}")
            one_shot[key] = record
    require(len(local) == 864 and len(one_shot) == 216, "Raw type counts mismatch")
    require(len(pair_ids) == EXPECTED_PAIRS, "Expected 96 distinct local pairs")

    groups = {}
    for group in paired["groups"]:
        key = (group["record_type"], group["snapshot_id"], group["worker_id"], group["region_id"])
        require(key not in groups, "Duplicate paired summary group")
        groups[key] = group
    require(len(groups) == 54, "Expected 54 paired summary groups")

    # Aggregate the four workers within each repeat before computing means.
    totals = {}
    improved_pairs = 0
    for snapshot in SNAPSHOTS:
        for repeat in REPEATS:
            for worker in WORKERS:
                a, b, u = [local[(snapshot, worker, repeat, c, "full")] for c in CONDITIONS]
                require(a["pair_id"] == b["pair_id"] == u["pair_id"], "A/B/U pair_id mismatch")
                improved_pairs += b["sse"] < a["sse"]
            for condition in CONDITIONS:
                for region in REGIONS:
                    values = [local[(snapshot, w, repeat, condition, region)]["sse"]
                              for w in WORKERS]
                    total = math.fsum(values)
                    totals[(snapshot, repeat, condition, region)] = total
                    aggregate = one_shot[(snapshot, None, repeat, condition, region)]
                    close(total, aggregate["local_error_sum"], "one-shot local_error_sum")
                    close(aggregate["sse"] - total, aggregate["cross_term"], "one-shot cross_term")
                for worker in WORKERS:
                    get = lambda region: local[(snapshot, worker, repeat, condition, region)]["sse"]
                    close(get("H") + get("rest"), get("full"), "H + rest = full")
    require(improved_pairs == EXPECTED_PAIRS, "Recorded 96/96 improvement does not hold")

    results = {}
    for snapshot in SNAPSHOTS:
        regions = {}
        for region in REGIONS:
            columns = [[totals[(snapshot, r, c, region)] for r in REPEATS] for c in CONDITIONS]
            a, b, u = columns
            means = [statistics.fmean(column) for column in columns]
            delta = [x - y for x, y in zip(a, b)]
            before = [x - y for x, y in zip(a, u)]
            after = [x - y for x, y in zip(b, u)]
            series = {
                "control_sse": a, "diagnostic_sse": b, "uniform_sse": u,
                "paired_delta": delta, "gap_before": before, "gap_after": after,
                "relative_reduction": [d / x for d, x in zip(delta, a)],
            }
            group = groups[("snapshot_local", snapshot, None, region)]
            require(group["repeat_count"] == 8, "Paired repeat_count mismatch")
            for name, values in series.items():
                calculated = {
                    "mean": statistics.fmean(values), "sample_std": statistics.stdev(values),
                    "min": min(values), "max": max(values),
                }
                for field, value in calculated.items():
                    close(value, group["statistics"][name][field], f"{snapshot}/{region}/{name}/{field}")
            improvements = sum(d > 0 for d in delta)
            require(group["paired_repeats_improved"] == improvements, "Paired direction mismatch")
            closure = None
            if snapshot == SNAPSHOTS[0] and means[0] > means[2]:
                closure = statistics.fmean(delta) / statistics.fmean(before)
                close(closure, group["gap_closure"], f"{snapshot}/{region}/gap_closure")
            regions[region] = {
                "mean_control_sse": means[0], "mean_diagnostic_sse": means[1],
                "mean_uniform_sse": means[2],
                "relative_reduction_ratio_of_means": (means[0] - means[1]) / means[0],
                "gap_closure": closure, "paired_repeats_improved": improvements,
            }
        results[snapshot] = regions

    first = results[SNAPSHOTS[0]]
    workers_improved = sum(
        statistics.fmean(local[(SNAPSHOTS[0], w, r, CONDITIONS[1], "full")]["sse"] for r in REPEATS)
        < statistics.fmean(local[(SNAPSHOTS[0], w, r, CONDITIONS[0], "full")]["sse"] for r in REPEATS)
        for w in WORKERS
    )
    for document in (summary, paired):
        close(first["full"]["gap_closure"], document["update100"]["gap_closure"], "update100 gap_closure")
        close(first["H"]["gap_closure"], document["update100"]["gap_closure_H"], "update100 H closure")
        require(document["update100"]["workers_improved"] == workers_improved, "Worker direction mismatch")
        require(document["update100"]["paired_repeats_improved"] == first["full"]["paired_repeats_improved"],
                "Snapshot direction mismatch")
    for snapshot, short in zip(SNAPSHOTS[1:], ("update500", "update2500")):
        change = -results[snapshot]["full"]["relative_reduction_ratio_of_means"]
        close(change, summary["other_snapshots_relative_sse_change"][short], short)
        close(change, paired["other_snapshots_relative_sse_change"][snapshot], snapshot)

    checks = {
        "gap_closure_at_least_half": first["full"]["gap_closure"] >= 0.5,
        "gap_closure_H_at_least_half": first["H"]["gap_closure"] >= 0.5,
        "at_least_three_workers_improve": workers_improved >= 3,
        "at_least_seven_paired_repeats_improve": first["full"]["paired_repeats_improved"] >= 7,
        "other_snapshots_regression_at_most_two_percent": all(
            -results[s]["full"]["relative_reduction_ratio_of_means"] <= 0.02 for s in SNAPSHOTS[1:]),
    }
    require(checks == summary["continuation_checks"], "Continuation checks mismatch")
    gate = "met" if all(checks.values()) else "not_met"
    closure = first["full"]["gap_closure"]
    outcome = "partially_closes_gap" if 0 < closure < 1 else "closes_or_reverses_gap" if closure >= 1 else "changes_without_closing_gap"
    for document in (summary, paired):
        require(document["continuation_gate"] == gate, "Continuation gate mismatch")
        require(document["outcome"] == outcome, "Outcome mismatch")
    for field in ("planned_pairs", "completed_control_pairs", "completed_diagnostic_pairs", "valid_diagnostic_pairs"):
        require(summary["coverage"][field] == EXPECTED_PAIRS, f"Coverage mismatch: {field}")
    require(summary["coverage"]["failed_pairs"] == summary["coverage"]["missing_pairs"] == 0,
            "Failed or missing pairs recorded")

    return {
        "status": "pass", "scope": "recorded_scalar_metrics_only",
        "run_id": summary["run_id"], "metrics_path": str(raw_path),
        "raw_records": len(records), "local_pairs": EXPECTED_PAIRS,
        "local_pairs_improved": improved_pairs,
        "snapshot_full_results": {s: results[s]["full"] for s in SNAPSHOTS},
        "update100_H_gap_closure": first["H"]["gap_closure"],
        "outcome": outcome, "continuation_gate": gate,
        "tensor_or_kernel_verification": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = verify(args.run_dir.resolve())
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as error:
        print(json.dumps({"status": "fail", "error": str(error)}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
