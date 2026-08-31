#!/usr/bin/env python3
"""Build the four legacy pending-era IEEE IoT-J tables from frozen summaries.

The script deliberately keeps measured and preregistered values in separate
columns.  A value marked ``PENDING`` is never converted into a numeric result.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLD = (
    REPO_ROOT
    / "results"
    / "frozen"
    / "rq1_selective_metrics_table.csv"
)
DEFAULT_OUT = REPO_ROOT / "artifacts" / "paper_tables"


def percent(value: str | float, digits: int = 2) -> str:
    return f"{100.0 * float(value):.{digits}f}%"


def interval(value: str, low: str, high: str) -> str:
    return f"{float(value):.4f} [{float(low):.4f}, {float(high):.4f}]"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_gold(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["method"]: row for row in rows}


def build_tables(gold_csv: Path, output_dir: Path) -> dict[str, object]:
    gold = read_gold(gold_csv)

    table_i = [
        {"Cohort or stage": "Source judgments (N1)", "n": 76291, "Status/share": "100.000%"},
        {"Cohort or stage": "Rule ACCEPT", "n": 2471, "Status/share": "3.239%"},
        {"Cohort or stage": "Rule REJECT", "n": 71296, "Status/share": "93.453%"},
        {"Cohort or stage": "Routed to Qwen", "n": 2524, "Status/share": "3.308%"},
        {"Cohort or stage": "Qwen ACCEPT / REJECT", "n": "431 / 1,357", "Status/share": "0.565% / 1.779%"},
        {"Cohort or stage": "Final UNRESOLVED", "n": 736, "Status/share": "0.965%"},
        {"Cohort or stage": "Dual-human binary consensus", "n": "743 / 900", "Status/share": "281 ACCEPT + 462 REJECT"},
        {"Cohort or stage": "Additional review", "n": "157 / 900", "Status/share": "155 shared uncertain + 2 discordant"},
    ]
    write_csv(output_dir / "table_i_corpus_and_gold.csv", list(table_i[0]), table_i)

    table_ii: list[dict[str, object]] = []
    display_names = {
        "rule_only": "Rule Only",
        "qwen_only": "Qwen Only",
        "hybrid": "Selective Hybrid",
        "hybrid_forced_reject": "Hybrid, Forced REJECT",
    }
    for method in ("rule_only", "qwen_only", "hybrid", "hybrid_forced_reject"):
        row = gold[method]
        table_ii.append(
            {
                "Method": display_names[method],
                "Consensus n/covered": f"{row['consensus_n']}/{row['consensus_covered']}",
                "Coverage": percent(row["coverage"]),
                "Precision [95% CI]": interval(row["precision"], row["precision_ci_low"], row["precision_ci_high"]),
                "Recall [95% CI]": interval(row["recall"], row["recall_ci_low"], row["recall_ci_high"]),
                "F1 [95% CI]": interval(row["f1"], row["f1_ci_low"], row["f1_ci_high"]),
                "MCC [95% CI]": interval(row["mcc"], row["mcc_ci_low"], row["mcc_ci_high"]),
                "TP/TN/FP/FN": f"{row['tp']}/{row['tn']}/{row['fp']}/{row['fn']}",
                "Full-900 unresolved recall": percent(row["unresolved_detection_recall"]),
            }
        )
    write_csv(output_dir / "table_ii_selective_gold_metrics.csv", list(table_ii[0]), table_ii)

    table_iii = [
        {"Evaluation block": "N4 graph schema + exact-span integrity", "Denominator": 400, "Result/status": "400/400 (measured; not semantic accuracy)"},
        {"Evaluation block": "N4 critical relation resolved", "Denominator": 400, "Result/status": "150/400 measured"},
        {"Evaluation block": "N5 contracts: defaults/review/blocked", "Denominator": 400, "Result/status": "104/189/107 measured"},
        {"Evaluation block": "RQ2 semantic review tasks", "Denominator": 381, "Result/status": "PENDING human scoring"},
        {"Evaluation block": "RQ3 unique legal cases", "Denominator": 24, "Result/status": "PENDING topology confirmation"},
        {"Evaluation block": "RQ3 CARLA fidelity runs", "Denominator": 96, "Result/status": "PENDING; 24 cases x 2 seeds x 2 repeats"},
        {"Evaluation block": "RQ4 clean controls", "Denominator": 24, "Result/status": "PENDING"},
        {"Evaluation block": "RQ4 injected faults", "Denominator": 144, "Result/status": "PENDING; 72 mutable + 72 immutable"},
        {"Evaluation block": "RQ4 N6 assurance artifacts", "Denominator": 168, "Result/status": "PENDING; controls + faults"},
    ]
    write_csv(output_dir / "table_iii_grounding_fidelity_assurance.csv", list(table_iii[0]), table_iii)

    table_iv = [
        {"Workload": "Ambiguous Qwen, concurrency 8", "n": 400, "Mean/P95 latency": "5.42/7.08 s", "Throughput": "1.455 req/s", "Load profile": "GPU/VRAM"},
        {"Workload": "CARLA, 12 workers", "n": 200, "Mean/P95 latency": "224.74/343.27 s", "Throughput": "2.179 run/min", "Load profile": "CPU/RAM"},
        {"Workload": "N6 VLM, concurrency 1", "n": 30, "Mean/P95 latency": "1.93/2.36 s", "Throughput": "0.514 req/s", "Load profile": "GPU/VRAM"},
        {"Workload": "N6 VLM, concurrency 8", "n": 170, "Mean/P95 latency": "6.01/7.42 s", "Throughput": "1.320 req/s", "Load profile": "GPU/VRAM"},
    ]
    write_csv(output_dir / "table_iv_runtime_and_load.csv", list(table_iv[0]), table_iv)

    manifest = {
        "schema_version": "jurisdrive-ieee-iotj-tables-v1",
        "gold_metrics_source": str(gold_csv.resolve()),
        "tables": {
            "I": {"file": "table_i_corpus_and_gold.csv", "status": "measured"},
            "II": {"file": "table_ii_selective_gold_metrics.csv", "status": "measured on 743 consensus; full-set uncertainty columns use n=900"},
            "III": {"file": "table_iii_grounding_fidelity_assurance.csv", "status": "mixed measured and explicitly pending"},
            "IV": {"file": "table_iv_runtime_and_load.csv", "status": "measured steady-state workloads"},
        },
        "denominator_guard": {
            "gold_classification": 900,
            "binary_consensus": 743,
            "static_graph_contract": 400,
            "unique_scenarios": 24,
            "fidelity_runs": 96,
            "runtime_repetitions": 200,
            "assurance_artifacts": 168,
        },
    }
    (output_dir / "tables_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-metrics", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    manifest = build_tables(args.gold_metrics, args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
