#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

RESULT_NAME_RE = re.compile(r"zeroshot_test_(\d+)_result\.json$")
RAW_NAME_RE = re.compile(r"zeroshot_test_(\d+)\.json$")

FIELDS = (
    "accident_datetime",
    "location",
    "weather_or_environment",
    "road_type",
    "vehicle_type",
    "blood_alcohol_content",
    "accident_trajectory",
    "casualty_result",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def natural_sort_key(path: Path) -> tuple[int, str]:
    match = RESULT_NAME_RE.fullmatch(path.name)
    if match:
        return int(match.group(1)), path.name
    return 2**63 - 1, path.name


def nonempty(value: Any) -> bool:
    return value not in (None, "", [], {})


def readiness_tier(parsed: dict[str, Any]) -> str:
    has_vehicle = nonempty(parsed.get("vehicle_type"))
    has_trajectory = nonempty(parsed.get("accident_trajectory"))
    has_road_context = nonempty(parsed.get("location")) or nonempty(parsed.get("road_type"))
    if has_vehicle and has_trajectory and has_road_context:
        return "A_minimum_grounded"
    if has_vehicle and has_trajectory:
        return "B_defaults_needed"
    return "C_reextract_or_review"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def count_matching_files(directory: Path, pattern: re.Pattern[str]) -> int:
    if not directory.exists():
        return 0
    return sum(1 for path in directory.iterdir() if path.is_file() and pattern.fullmatch(path.name))


def iter_candidate_files(full_run_dir: Path) -> Iterable[tuple[str, Path]]:
    sources = (
        ("rule", full_run_dir / "output" / "car_to_car"),
        ("llm", full_run_dir / "ambiguous_done" / "car_to_car"),
    )
    for stage, directory in sources:
        for path in sorted(directory.glob("zeroshot_test_*_result.json"), key=natural_sort_key):
            if RESULT_NAME_RE.fullmatch(path.name):
                yield stage, path


def build_manifest_row(
    stage: str,
    path: Path,
    record: dict[str, Any],
    *,
    full_run_dir: Path | None = None,
) -> dict[str, Any]:
    parsed = record.get("parsed") if isinstance(record.get("parsed"), dict) else {}
    postprocess = record.get("postprocess") if isinstance(record.get("postprocess"), dict) else {}
    rule = postprocess.get("rule") if isinstance(postprocess.get("rule"), dict) else {}
    llm = postprocess.get("llm") if isinstance(postprocess.get("llm"), dict) else {}
    match = RESULT_NAME_RE.fullmatch(path.name)
    candidate_id = int(match.group(1)) if match else None
    return {
        "candidate_id": candidate_id,
        "result_file": path.name,
        "source_stage": stage,
        "source_path": (
            path.relative_to(full_run_dir).as_posix()
            if full_run_dir is not None and path.is_relative_to(full_run_dir)
            else str(path)
        ),
        "input_file": record.get("input_file"),
        "source_text_length": len(record.get("source_text") or ""),
        "readiness_tier": readiness_tier(parsed),
        "parsed": {field: parsed.get(field) for field in FIELDS},
        "rule": {
            "label": rule.get("label"),
            "score": rule.get("score"),
            "reason": rule.get("reason"),
        },
        "llm": {
            "label": llm.get("label"),
            "confidence": llm.get("confidence"),
            "accident_type": llm.get("accident_type"),
            "reason": llm.get("reason"),
            "evidence": llm.get("evidence"),
        }
        if stage == "llm"
        else None,
    }


def audit(
    *,
    full_run_dir: Path,
    raw_dir: Path,
    zeroshot_dir: Path,
    output_dir: Path,
    record_local_paths: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rule_summary_path = full_run_dir / "summary.json"
    llm_summary_path = full_run_dir / "ambiguous_done" / "summary.json"
    rule_summary = read_json(rule_summary_path)
    llm_summary = read_json(llm_summary_path)

    raw_count = count_matching_files(raw_dir, RAW_NAME_RE)
    zeroshot_count = count_matching_files(zeroshot_dir, RESULT_NAME_RE)

    field_counts: Counter[str] = Counter()
    readiness_counts: Counter[str] = Counter()
    stage_counts: Counter[str] = Counter()
    llm_confidence: Counter[str] = Counter()
    llm_accident_type: Counter[str] = Counter()
    manifest_rows: list[dict[str, Any]] = []

    for stage, path in iter_candidate_files(full_run_dir):
        record = read_json(path)
        row = build_manifest_row(stage, path, record, full_run_dir=full_run_dir)
        manifest_rows.append(row)
        stage_counts[stage] += 1
        readiness_counts[row["readiness_tier"]] += 1
        for field, value in row["parsed"].items():
            if nonempty(value):
                field_counts[field] += 1
        if stage == "llm" and row["llm"]:
            llm_confidence[row["llm"].get("confidence") or "missing"] += 1
            llm_accident_type[row["llm"].get("accident_type") or "missing"] += 1

    final_car_to_car = len(manifest_rows)
    final_not_car_to_car = int(rule_summary["not_car_to_car"]) + int(llm_summary["not_car_to_car"])
    final_unresolved = int(llm_summary["ambiguous"])
    total = int(rule_summary["selected_files"])

    integrity_checks = {
        "raw_equals_zeroshot": raw_count == zeroshot_count,
        "zeroshot_equals_rule_input": zeroshot_count == total,
        "rule_branches_sum_to_total": (
            int(rule_summary["car_to_car"])
            + int(rule_summary["not_car_to_car"])
            + int(rule_summary["ambiguous"])
            == total
        ),
        "llm_branches_sum_to_routed": (
            int(llm_summary["car_to_car"])
            + int(llm_summary["not_car_to_car"])
            + int(llm_summary["ambiguous"])
            == int(rule_summary["ambiguous"])
        ),
        "final_branches_sum_to_total": final_car_to_car + final_not_car_to_car + final_unresolved == total,
        "candidate_manifest_matches_final_car_count": final_car_to_car
        == int(rule_summary["car_to_car"]) + int(llm_summary["car_to_car"]),
    }

    audit_payload = {
        "generated_at": utc_now_iso(),
        "paths": (
            {
                "raw_dir": str(raw_dir),
                "zeroshot_dir": str(zeroshot_dir),
                "full_run_dir": str(full_run_dir),
            }
            if record_local_paths
            else {"redacted": True}
        ),
        "source_hashes": {
            "rule_summary_sha256": sha256_file(rule_summary_path),
            "llm_summary_sha256": sha256_file(llm_summary_path),
        },
        "counts": {
            "raw_records": raw_count,
            "zeroshot_records": zeroshot_count,
            "rule_car_to_car": int(rule_summary["car_to_car"]),
            "rule_not_car_to_car": int(rule_summary["not_car_to_car"]),
            "routed_to_llm": int(rule_summary["ambiguous"]),
            "llm_car_to_car": int(llm_summary["car_to_car"]),
            "llm_not_car_to_car": int(llm_summary["not_car_to_car"]),
            "llm_unresolved": int(llm_summary["ambiguous"]),
            "final_car_to_car": final_car_to_car,
            "final_not_car_to_car": final_not_car_to_car,
            "final_unresolved": final_unresolved,
        },
        "rates_percent_of_total": {
            "routed_to_llm": round(int(rule_summary["ambiguous"]) / total * 100, 3),
            "final_car_to_car": round(final_car_to_car / total * 100, 3),
            "final_not_car_to_car": round(final_not_car_to_car / total * 100, 3),
            "final_unresolved": round(final_unresolved / total * 100, 3),
        },
        "rates_percent_of_routed": {
            "llm_car_to_car": round(int(llm_summary["car_to_car"]) / int(rule_summary["ambiguous"]) * 100, 3),
            "llm_not_car_to_car": round(
                int(llm_summary["not_car_to_car"]) / int(rule_summary["ambiguous"]) * 100, 3
            ),
            "llm_unresolved": round(int(llm_summary["ambiguous"]) / int(rule_summary["ambiguous"]) * 100, 3),
        },
        "candidate_source_stage": dict(stage_counts),
        "candidate_field_coverage": {
            field: {
                "count": field_counts[field],
                "percent": round(field_counts[field] / final_car_to_car * 100, 2),
            }
            for field in FIELDS
        },
        "scenario_contract_readiness": {
            tier: {
                "count": readiness_counts[tier],
                "percent": round(readiness_counts[tier] / final_car_to_car * 100, 2),
            }
            for tier in ("A_minimum_grounded", "B_defaults_needed", "C_reextract_or_review")
        },
        "llm_candidate_distribution": {
            "confidence": dict(llm_confidence),
            "accident_type": dict(llm_accident_type),
        },
        "integrity_checks": integrity_checks,
    }

    manifest_path = output_dir / "final_car_to_car_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    audit_path = output_dir / "current_data_audit.json"
    audit_path.write_text(json.dumps(audit_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    csv_path = output_dir / "n_stage_counts.csv"
    rows = [
        ("zeroshot_input", total, 100.0),
        ("rule_car_to_car", int(rule_summary["car_to_car"]), int(rule_summary["car_to_car"]) / total * 100),
        ("rule_not_car_to_car", int(rule_summary["not_car_to_car"]), int(rule_summary["not_car_to_car"]) / total * 100),
        ("routed_to_llm", int(rule_summary["ambiguous"]), int(rule_summary["ambiguous"]) / total * 100),
        ("llm_car_to_car", int(llm_summary["car_to_car"]), int(llm_summary["car_to_car"]) / total * 100),
        ("llm_not_car_to_car", int(llm_summary["not_car_to_car"]), int(llm_summary["not_car_to_car"]) / total * 100),
        ("llm_unresolved", int(llm_summary["ambiguous"]), int(llm_summary["ambiguous"]) / total * 100),
        ("final_car_to_car", final_car_to_car, final_car_to_car / total * 100),
        ("minimum_grounded", readiness_counts["A_minimum_grounded"], readiness_counts["A_minimum_grounded"] / total * 100),
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("stage", "count", "percent_of_total"))
        for stage, count, percent in rows:
            writer.writerow((stage, count, f"{percent:.3f}"))

    return audit_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the current N-stage LocalLLM filtering outputs.")
    parser.add_argument("--full-run-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--zeroshot-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--record-local-paths",
        action="store_true",
        help="Include absolute input paths in the local audit report (redacted by default).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = audit(
        full_run_dir=args.full_run_dir.expanduser().resolve(),
        raw_dir=args.raw_dir.expanduser().resolve(),
        zeroshot_dir=args.zeroshot_dir.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        record_local_paths=args.record_local_paths,
    )
    failed_checks = [name for name, passed in payload["integrity_checks"].items() if not passed]
    print(json.dumps(payload["counts"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["scenario_contract_readiness"], ensure_ascii=False, indent=2))
    if failed_checks:
        print(f"Integrity checks failed: {failed_checks}")
        return 1
    print("All integrity checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
