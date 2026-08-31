#!/usr/bin/env python3
"""Fail-closed integrity audit for the frozen JurisDrive RQ2--RQ4 results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify_reference(reference: dict[str, Any], label: str) -> None:
    path = Path(reference["path"])
    require(path.is_file(), f"{label}: missing file {path}")
    require(sha256_file(path) == reference["sha256"], f"{label}: SHA-256 mismatch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    base = args.base_dir.resolve()
    output = args.output_dir.resolve()
    require(base.is_dir(), f"base directory does not exist: {base}")
    require(not output.exists(), f"refusing to overwrite completion audit: {output}")

    checks: list[dict[str, Any]] = []

    def checked(name: str, detail: Any) -> None:
        checks.append({"check": name, "status": "pass", "detail": detail})

    frozen = read_json(base / "manifest.json")
    require(frozen["selection_frozen"] is True, "24-case selection is not frozen")
    require(frozen["selected_cases"] == 24, "frozen selection must contain 24 cases")
    require(frozen["compile_valid_cases"] == 24, "all 24 frozen cases must compile")
    verify_reference(frozen["frozen_config"], "frozen topology config")
    checked("frozen_topology_selection", "24 selected; 24 compile-valid; 0 blocked")

    rq2_dir = base / "rq2_publication_results_v2"
    rq2_manifest = read_json(rq2_dir / "manifest.json")
    for name, reference in rq2_manifest["inputs"].items():
        if name == "graphs":
            graph_root = Path(reference["path"])
            graph_paths = list(graph_root.glob("*.json"))
            require(len(graph_paths) == reference["files"] == 381, "RQ2 graph file count drift")
            require(tree_sha256(graph_paths, graph_root) == reference["tree_sha256"], "RQ2 graph tree SHA mismatch")
        else:
            verify_reference(reference, f"RQ2 inputs.{name}")
    for name, reference in rq2_manifest["outputs"].items():
        verify_reference(reference, f"RQ2 outputs.{name}")
    rq2 = read_json(rq2_dir / "rq2_summary.json")
    require(rq2["total"] == 381, "RQ2 denominator must be 381")
    require(rq2["strata"] == {"accept": 281, "reject": 74, "unresolved": 26}, "RQ2 strata drift")
    require(rq2["accept_semantics"]["evaluated"] == 281, "RQ2 ACCEPT semantic denominator drift")
    require(rq2["reject_selective_safety"]["n"] == 74, "RQ2 REJECT denominator drift")
    require(rq2["unresolved_selective_safety"]["n"] == 26, "RQ2 UNRESOLVED denominator drift")
    require(rq2["accept_semantics"]["collision_evidence_grounding_coverage"]["present"] == 170, "RQ2 collision-evidence coverage drift")
    require(rq2["accept_semantics"]["dual_human_evidence_quote_alignment"]["aligned"] == 161, "RQ2 quote-alignment count drift")
    require(rq2["accept_semantics"]["evidence_span_semantic_sufficiency"]["direct_human_ratings_n"] == 0, "RQ2 human sufficiency-rating boundary drift")
    checked("rq2_denominator_and_hashes", "381 = 281 ACCEPT + 74 REJECT + 26 UNRESOLVED")

    rq3_dir = base / "rq3_fidelity96"
    rq3_manifest = read_json(rq3_dir / "manifest.json")
    verify_reference(rq3_manifest["schedule"], "RQ3 schedule")
    verify_reference(rq3_manifest["records"], "RQ3 records")
    rq3_rows = read_jsonl(Path(rq3_manifest["records"]["path"]))
    require(len(rq3_rows) == 96, "RQ3 must contain exactly 96 records")
    require(len({row["run_id"] for row in rq3_rows}) == 96, "RQ3 run IDs are not unique")
    require(len({row["candidate_id"] for row in rq3_rows}) == 24, "RQ3 must contain 24 unique cases")
    for row in rq3_rows:
        require(row["execution_status"] == "completed", f"RQ3 incomplete run: {row['run_id']}")
        for field in (
            "contract_compile_pass",
            "carla_launch_complete",
            "run_complete",
            "actor_target_correct",
            "lane_topology_valid",
            "event_order_valid",
            "hard_constraint_pass",
        ):
            require(row[field] is True, f"RQ3 {field} failed: {row['run_id']}")
    require(Counter(row["topology"] for row in rq3_rows) == {
        "rear_end": 24,
        "intersection_crossing_turning": 24,
        "lane_change_side_swipe": 24,
        "head_on_centerline_intrusion": 24,
    }, "RQ3 topology balance drift")
    require(Counter(row["source_stage"] for row in rq3_rows) == {"rule": 48, "qwen": 48}, "RQ3 route balance drift")
    rq3_summary = rq3_manifest["summary"]
    require(rq3_summary["completed_runs"] == 96, "RQ3 manifest completion drift")
    require(rq3_summary["replay"]["complete_same_seed_pairs"] == 48, "RQ3 replay denominator drift")
    require(rq3_summary["replay"]["exact_core_metric_pairs"] == 48, "RQ3 core replay mismatch")
    require(rq3_summary["replay"]["exact_telemetry_hash_pairs"] == 22, "RQ3 raw replay hash count drift")
    require(rq3_summary["map_asset_fallback_runs"] == 96, "RQ3 fallback disclosure count drift")
    checked("rq3_execution", "96/96 completed; 24 unique; 48/48 core replay pairs; raw telemetry SHA 22/48")
    rq4_dir = base / "rq4_publication_results_v4"
    rq4_manifest = read_json(rq4_dir / "manifest.json")
    require(rq4_manifest["base_artifacts"] == 168, "RQ4 base denominator drift")
    require(rq4_manifest["clean_controls"] == 24, "RQ4 clean-control denominator drift")
    require(rq4_manifest["faults"] == 144, "RQ4 fault denominator drift")
    require(rq4_manifest["mutable_faults"] == 72, "RQ4 mutable denominator drift")
    require(rq4_manifest["immutable_or_evidence_conflict_faults"] == 72, "RQ4 immutable denominator drift")
    require(rq4_manifest["completed_rows"] == 840, "RQ4 completed method rows drift")
    for name, reference in rq4_manifest["artifacts"].items():
        verify_reference(reference, f"RQ4 artifact.{name}")
    rq4_rows = read_jsonl(Path(rq4_manifest["artifacts"]["records"]["path"]))
    require(len(rq4_rows) == 840, "RQ4 records must contain exactly 840 rows")
    methods = set(rq4_manifest["methods"])
    require(set(row["method"] for row in rq4_rows) == methods, "RQ4 method set drift")
    by_trial: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rq4_rows:
        require(row["execution_status"] == "completed", f"RQ4 incomplete row: {row['trial_id']} / {row['method']}")
        require(row["injection_verified"] is True, f"RQ4 unverified injection: {row['trial_id']}")
        by_trial[row["trial_id"]].append(row)
    require(len(by_trial) == 168, "RQ4 must contain 168 unique base trials")
    require(all(len(rows) == 5 for rows in by_trial.values()), "each RQ4 trial must have five method rows")
    representatives = [rows[0] for rows in by_trial.values()]
    require(sum(row["trial_kind"] == "clean_control" for row in representatives) == 24, "RQ4 clean trial count drift")
    require(sum(row["trial_kind"] != "clean_control" for row in representatives) == 144, "RQ4 fault trial count drift")
    guarded_probes = [
        row for row in rq4_rows
        if row["method"] == "guarded_bounded_repair" and row.get("immutable_edit_attempted")
    ]
    require(len(guarded_probes) == 48, "RQ4 explicit immutable/structural guard-probe count drift")
    require(all(row.get("immutable_edit_rejected") is True for row in guarded_probes), "RQ4 guard probe was not rejected")
    require(all(row.get("immutable_guard_path") for row in guarded_probes), "RQ4 guard probe path missing")
    require(all(row.get("immutable_guard_notes") for row in guarded_probes), "RQ4 guard probe audit note missing")

    expected_inputs = {
        "materialization": base / "rq4_materialized168_v3" / "materialization_records.jsonl",
        "mutable_reruns": base / "rq4_mutable72_reruns_v3" / "mutable_rerun_records.jsonl",
        "vlm_observations": base / "rq4_vlm336" / "vlm_observations.jsonl",
        "guarded_repairs": base / "rq4_guarded_repair72_runs" / "guarded_repair_records.jsonl",
        "unconstrained_repairs": base / "rq4_unconstrained_repair49_runs" / "unconstrained_repair_records.jsonl",
    }
    for name, path in expected_inputs.items():
        require(path.is_file(), f"RQ4 input missing: {path}")
        require(sha256_file(path) == rq4_manifest["input_hashes"][name], f"RQ4 input SHA mismatch: {name}")

    vlm_manifest = read_json(base / "rq4_vlm336" / "manifest.json")
    require(vlm_manifest["completed_observations"] == 504, "RQ4 VLM observation count drift")
    require(vlm_manifest["failed_observations"] == 0, "RQ4 VLM contains failed observations")
    require(len(read_jsonl(Path(vlm_manifest["records"]["path"]))) == 504, "RQ4 VLM JSONL count drift")
    guarded_manifest = read_json(base / "rq4_guarded_repair72_runs" / "manifest.json")
    require(guarded_manifest["completed_repair_runs"] == 72, "guarded repair completion drift")
    require(guarded_manifest["post_repair_passed"] == 72, "guarded repair pass count drift")
    require(guarded_manifest["post_repair_regressions"] == 0, "guarded repair regression count drift")
    checked("rq4_execution", "168 base artifacts x 5 methods = 840 completed rows; 504 VLM observations; 72/72 guarded reruns; 48/48 explicit guard probes rejected")

    method_table = list(csv.DictReader((rq4_dir / "rq4_method_summary.csv").open(encoding="utf-8-sig")))
    require(len(method_table) == 5, "RQ4 paper method table must contain five methods")
    checked("rq4_report_table", "five method rows and hash-linked paper outputs")

    output.mkdir(parents=True)
    audit = {
        "version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "base_dir": str(base),
        "status": "complete_with_disclosed_limitations",
        "checks": checks,
        "rq2_status": "completed_denominator_safe; semantic performance is weak; collision evidence exists for 170/281, exact dual-human quote alignment is 161/281, and independent human sufficiency ratings n=0",
        "rq3_status": "completed; all 96 runs used the disclosed Town_Safebench_Light fallback",
        "rq4_status": "completed controlled suite; guarded repair is provenance-manifest-guided rollback, not unconstrained generative repair",
    }
    audit_path = output / "completion_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = """# RQ2--RQ4 completion audit

Status: **complete with disclosed limitations**

- RQ2: 381/381 evaluated under denominator-safe rules (281 ACCEPT semantics, 74 REJECT leakage, 26 UNRESOLVED abstention). Collision evidence exists in 170/281 ACCEPT graphs and exact dual-human quote alignment holds for 161/281; there is no separate human sufficiency rating. The low semantic scores are results, not a blocker.
- RQ3: 24 unique cases and 96/96 CARLA executions completed. All hard checks passed; 48/48 replay pairs matched on normalized core metrics, while 22/48 raw telemetry hashes matched. All runs used the disclosed `Town_Safebench_Light` fallback.
- RQ4: 24 controls + 144 verified faults = 168 artifacts; five methods produced 840/840 completed rows. The VLM stage contains 504/504 observations with zero API failures. Guarded bounded rollback completed 72/72 CARLA reruns with no regression.

The frozen pre-execution plan retains historical `blocked_*` counts by design. Those fields describe the state before materialization and execution; this completion audit and the hash-linked final manifests supersede them for reporting current completion.
"""
    report_path = output / "completion_audit.md"
    report_path.write_text(report, encoding="utf-8")
    audit["outputs"] = {
        "json": {"path": str(audit_path), "sha256": sha256_file(audit_path)},
        "markdown": {"path": str(report_path), "sha256": sha256_file(report_path)},
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "checks": len(checks), "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
