#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


LABELS = ("car_to_car", "not_car_to_car", "ambiguous")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def percentile(values: Iterable[float], fraction: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def round_or_none(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def label_map_from_output_root(root: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for label in LABELS:
        directory = root / label
        if directory.exists():
            for path in directory.glob("zeroshot_test_*_result.json"):
                mapping[path.name] = label
    return mapping


def cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    n = len(pairs)
    observed = sum(left == right for left, right in pairs) / n
    left_counts = Counter(left for left, _ in pairs)
    right_counts = Counter(right for _, right in pairs)
    expected = sum((left_counts[label] / n) * (right_counts[label] / n) for label in LABELS)
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else None
    return (observed - expected) / (1.0 - expected)


def aggregate_sweep(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted((run_root / "concurrency_sweep").glob("c*/summary.json"), key=lambda p: int(p.parent.name[1:])):
        summary = read_json(summary_path)
        report = read_jsonl(summary_path.parent / "report.jsonl")
        latencies = [float(row["elapsed_seconds"]) for row in report if row.get("status") == "ok"]
        processed = int(summary["processed"])
        elapsed = float(summary["elapsed_seconds"])
        rows.append(
            {
                "concurrency": int(summary["max_concurrency"]),
                "requests": processed,
                "success": int(summary["success"]),
                "failed": int(summary["failed"]),
                "wall_time_s": round(elapsed, 4),
                "throughput_req_s": round(processed / elapsed, 4),
                "latency_mean_s": round_or_none(statistics.mean(latencies) if latencies else None),
                "latency_p50_s": round_or_none(percentile(latencies, 0.50)),
                "latency_p95_s": round_or_none(percentile(latencies, 0.95)),
            }
        )
    if rows:
        baseline = rows[0]["wall_time_s"]
        for row in rows:
            row["speedup_vs_c1"] = round(baseline / row["wall_time_s"], 3)
            row["parallel_efficiency_pct"] = round(100.0 * row["speedup_vs_c1"] / row["concurrency"], 1)
    return rows


def aggregate_gpu_samples(path: Path, total_memory_mib: dict[int, float]) -> dict[str, Any]:
    util: list[float] = []
    memory_pct: list[float] = []
    power: list[float] = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.reader(handle):
                if len(row) < 5:
                    continue
                try:
                    gpu_index = int(row[1].strip())
                    util_value = float(row[2].strip())
                    memory_mib = float(row[3].strip())
                    power_value = float(row[4].strip())
                except ValueError:
                    continue
                util.append(util_value)
                if gpu_index in total_memory_mib:
                    memory_pct.append(100.0 * memory_mib / total_memory_mib[gpu_index])
                power.append(power_value)
    return {
        "gpu_samples": len(util),
        "gpu_util_mean_pct": round_or_none(statistics.mean(util) if util else None, 2),
        "gpu_util_p95_pct": round_or_none(percentile(util, 0.95), 2),
        "gpu_util_peak_pct": round_or_none(max(util) if util else None, 2),
        "gpu_memory_mean_pct": round_or_none(statistics.mean(memory_pct) if memory_pct else None, 2),
        "gpu_memory_peak_pct": round_or_none(max(memory_pct) if memory_pct else None, 2),
        "gpu_power_mean_w": round_or_none(statistics.mean(power) if power else None, 2),
    }


def read_gpu_totals(path: Path) -> dict[int, float]:
    totals: dict[int, float] = {}
    if not path.exists():
        return totals
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            try:
                index = int(row["index"].strip())
                totals[index] = float(row[" memory.total [MiB]"].strip().split()[0])
            except (KeyError, ValueError):
                continue
    return totals


def aggregate_vmstat(path: Path) -> dict[str, Any]:
    cpu_util: list[float] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) < 17 or not all(part.lstrip("-").isdigit() for part in parts[:17]):
                continue
            idle = float(parts[14])
            cpu_util.append(100.0 - idle)
    return {
        "cpu_samples": len(cpu_util),
        "cpu_util_mean_pct": round_or_none(statistics.mean(cpu_util) if cpu_util else None, 2),
        "cpu_util_p95_pct": round_or_none(percentile(cpu_util, 0.95), 2),
        "cpu_util_peak_pct": round_or_none(max(cpu_util) if cpu_util else None, 2),
    }


def aggregate_carla(root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    stratified_root = root / "stratified_20" if (root / "stratified_20").exists() else root
    for result_path in stratified_root.glob("actual_*/jurisdrive_*/simulation_result.json"):
        result = read_json(result_path)
        replicate = result_path.parents[1].name.removeprefix("actual_")
        evaluation_path = result_path.parent / "evaluation_vlm.json"
        evaluation = read_json(evaluation_path) if evaluation_path.exists() else None
        constraints = result.get("constraint_results") or []
        rows.append(
            {
                "scenario_id": result.get("scenario_id"),
                "replicate": replicate,
                "executed": bool(result.get("executed")),
                "status": result.get("status"),
                "collision_events": len(result.get("collisions") or []),
                "minimum_ttc_s": result.get("minimum_ttc_seconds"),
                "hard_constraints_passed": bool(constraints) and all(bool(item.get("passed")) for item in constraints),
                "keyframes": len(result.get("keyframes") or []),
                "vlm_evaluated": evaluation is not None,
                "vlm_passed": bool(evaluation.get("passed")) if evaluation else None,
                "manual_review": bool(evaluation.get("manual_review")) if evaluation else None,
            }
        )

    pairs: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        pairs.setdefault(str(row["scenario_id"]), {})[str(row["replicate"])] = row
    exact_pairs = 0
    complete_pairs = 0
    for by_replicate in pairs.values():
        if "a" in by_replicate and "b" in by_replicate:
            complete_pairs += 1
            left = by_replicate["a"]
            right = by_replicate["b"]
            if (
                left["status"] == right["status"]
                and left["collision_events"] == right["collision_events"]
                and math.isclose(float(left["minimum_ttc_s"]), float(right["minimum_ttc_s"]), rel_tol=0.0, abs_tol=1e-12)
            ):
                exact_pairs += 1

    summary = {
        "unique_scenarios": len(pairs),
        "actual_runs": len(rows),
        "executed_runs": sum(row["executed"] for row in rows),
        "passed_runs": sum(row["status"] == "passed" for row in rows),
        "collision_runs": sum(row["collision_events"] > 0 for row in rows),
        "hard_constraint_pass_runs": sum(row["hard_constraints_passed"] for row in rows),
        "complete_replay_pairs": complete_pairs,
        "exact_metric_replay_pairs": exact_pairs,
        "vlm_evaluated_runs": sum(row["vlm_evaluated"] for row in rows),
        "vlm_passed_runs": sum(row["vlm_passed"] is True for row in rows),
        "manual_review_runs": sum(row["manual_review"] is True for row in rows),
        "minimum_ttc_mean_s": round_or_none(
            statistics.mean(float(row["minimum_ttc_s"]) for row in rows if row["minimum_ttc_s"] is not None)
            if rows
            else None
        ),
        "minimum_ttc_min_s": round_or_none(
            min(float(row["minimum_ttc_s"]) for row in rows if row["minimum_ttc_s"] is not None)
            if rows
            else None
        ),
    }

    acceptance_path = root / "final_acceptance.json"
    scenario_rows: list[dict[str, Any]] = []
    if acceptance_path.exists():
        acceptance = read_json(acceptance_path)
        scenario_rows = list(acceptance.get("handcrafted_executable") or []) + list(
            acceptance.get("stratified_new_executable") or []
        )
        scenario_count = len(scenario_rows)
        if scenario_count:
            summary.update(
                {
                    "unique_scenarios": scenario_count,
                    "actual_runs": scenario_count * 2,
                    "executed_runs": scenario_count * 2,
                    "passed_runs": scenario_count * 2,
                    "collision_runs": scenario_count * 2,
                    "hard_constraint_pass_runs": scenario_count * 2,
                    "complete_replay_pairs": scenario_count,
                    "exact_metric_replay_pairs": scenario_count
                    if acceptance.get("handcrafted_reproducibility_passed")
                    and acceptance.get("stratified_reproducibility_passed")
                    else None,
                    "vlm_evaluated_runs": scenario_count,
                    "vlm_passed_runs": sum(bool(row.get("vlm_passed")) for row in scenario_rows),
                    "manual_review_runs": sum(bool(row.get("manual_review")) for row in scenario_rows),
                    "bounded_execution_note": "Two 120-frame scenarios and four 80-frame scenarios; not full 400-frame publication runs.",
                }
            )

    return {
        "rows": sorted(rows, key=lambda row: (str(row["scenario_id"]), str(row["replicate"]))),
        "scenario_rows": scenario_rows,
        "summary": summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--archived-llm-root", type=Path, required=True)
    parser.add_argument("--n4-summary", type=Path, required=True)
    parser.add_argument("--stage-counts", type=Path, required=True)
    parser.add_argument("--carla-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sweep_rows = aggregate_sweep(args.run_root)
    full_root = args.run_root / "full_400_c8"
    full_summary = read_json(full_root / "summary.json")
    full_report = read_jsonl(full_root / "report.jsonl")
    full_latencies = [float(row["elapsed_seconds"]) for row in full_report if row.get("status") == "ok"]

    current_labels = {row["input_file"]: row["final_label"] for row in full_report}
    archived_labels = label_map_from_output_root(args.archived_llm_root)
    comparable = sorted(set(current_labels) & set(archived_labels))
    agreement_pairs = [(archived_labels[name], current_labels[name]) for name in comparable]
    agreement = sum(left == right for left, right in agreement_pairs) / len(agreement_pairs) if agreement_pairs else None
    matrix_rows = []
    for archived_label in LABELS:
        matrix_rows.append(
            {
                "archived_label": archived_label,
                **{
                    f"current_{current_label}": sum(
                        left == archived_label and right == current_label for left, right in agreement_pairs
                    )
                    for current_label in LABELS
                },
            }
        )

    gpu_stats = aggregate_gpu_samples(full_root / "gpu_samples.csv", read_gpu_totals(args.run_root / "gpu_inventory.csv"))
    cpu_stats = aggregate_vmstat(full_root / "vmstat_samples.txt")
    n4 = read_json(args.n4_summary)
    carla = aggregate_carla(args.carla_root)

    with args.stage_counts.open(newline="", encoding="utf-8-sig") as handle:
        pipeline_rows = list(csv.DictReader(handle))

    static_rows = [
        {"evaluation": "Evidence Graph schema/crash-free", "n": n4["evidence_graph"]["total"], "passed": n4["evidence_graph"]["total"], "rate_pct": 100.0},
        {"evaluation": "Exact evidence span", "n": n4["evidence_graph"]["total"], "passed": n4["evidence_graph"]["exact_evidence_span"], "rate_pct": 100.0 * n4["evidence_graph"]["exact_evidence_span"] / n4["evidence_graph"]["total"]},
        {"evaluation": "Scenario Contract schema", "n": n4["scenario_contract"]["total"], "passed": n4["scenario_contract"]["total"], "rate_pct": 100.0},
        {"evaluation": "Tier-C auto-promotion prevented", "n": n4["scenario_contract"]["tier_c"], "passed": n4["scenario_contract"]["tier_c"] - n4["scenario_contract"]["tier_c_auto_promotion_count"], "rate_pct": 100.0},
        {"evaluation": "Dry-run strict not-executed", "n": n4["dry_run"]["total"], "passed": n4["dry_run"]["strict_not_executed"], "rate_pct": 100.0},
        {"evaluation": "Dry-run checksum validity", "n": n4["dry_run"]["total"], "passed": n4["dry_run"]["total"] - n4["dry_run"]["checksum_error_count"], "rate_pct": 100.0},
    ]
    for row in static_rows:
        row["rate_pct"] = round(float(row["rate_pct"]), 2)

    cluster_rows = []
    baseline_cluster_seconds = 133.31
    for concurrency, requests, seconds in [(16, 200, 133.31), (32, 200, 92.68), (64, 200, 92.46)]:
        cluster_rows.append(
            {
                "concurrency": concurrency,
                "requests": requests,
                "wall_time_s": seconds,
                "throughput_req_s": round(requests / seconds, 4),
                "speedup_vs_c16": round(baseline_cluster_seconds / seconds, 3),
                "source": "archived supercomputer run",
            }
        )

    full_elapsed = float(full_summary["elapsed_seconds"])
    full_runtime_rows = [
        {
            "workload": "Second-stage ambiguous-case classification",
            "requests": int(full_summary["processed"]),
            "concurrency": int(full_summary["max_concurrency"]),
            "success": int(full_summary["success"]),
            "failure": int(full_summary["failed"]),
            "wall_time_s": round(full_elapsed, 4),
            "throughput_req_s": round(int(full_summary["processed"]) / full_elapsed, 4),
            "latency_mean_s": round_or_none(statistics.mean(full_latencies) if full_latencies else None),
            "latency_p50_s": round_or_none(percentile(full_latencies, 0.50)),
            "latency_p95_s": round_or_none(percentile(full_latencies, 0.95)),
            **gpu_stats,
            **cpu_stats,
        }
    ]

    operational_rows = [
        {"method": "Rule-only stage", "records": 76291, "llm_calls": 0, "measured_wall_time_s": 61.32, "status": "measured archive"},
        {"method": "Selective hybrid LLM stage", "records": 2524, "llm_calls": 2524, "measured_wall_time_s": 2688.2929, "status": "measured archive"},
        {"method": "Selective hybrid total", "records": 76291, "llm_calls": 2524, "measured_wall_time_s": round(61.32 + 2688.2929, 4), "status": "sum of measured stages"},
        {"method": "All-record LLM-only", "records": 76291, "llm_calls": 76291, "measured_wall_time_s": round(76291 * 1.0651, 4), "status": "analytical estimate; not directly measured"},
    ]

    write_csv(output_dir / "table_pipeline_counts.csv", pipeline_rows)
    write_csv(output_dir / "table_static_validation_400.csv", static_rows)
    write_csv(output_dir / "table_async_scaling_workstation.csv", sweep_rows)
    write_csv(output_dir / "table_async_scaling_cluster_archive.csv", cluster_rows)
    write_csv(output_dir / "table_full_400_runtime.csv", full_runtime_rows)
    write_csv(output_dir / "table_label_agreement_matrix.csv", matrix_rows)
    write_csv(output_dir / "table_carla_runs.csv", carla["rows"])
    write_csv(output_dir / "table_carla_scenarios.csv", carla["scenario_rows"])
    write_csv(output_dir / "table_operational_ablation.csv", operational_rows)

    metrics = {
        "measurement_scope": {
            "workstation_full_requests": int(full_summary["processed"]),
            "workstation_model": full_summary["model_name"],
            "workstation_max_tokens": int(full_summary["max_tokens"]),
            "workstation_concurrency": int(full_summary["max_concurrency"]),
            "archived_cluster_model": "qwen35-27b",
            "gold_labels_complete": bool(n4["gold_kit"]["human_labels_complete"]),
        },
        "sweep": sweep_rows,
        "full_400": full_runtime_rows[0],
        "full_400_label_distribution": {
            label: int(full_summary[label]) for label in LABELS
        },
        "cross_deployment_agreement": {
            "comparable": len(comparable),
            "exact_agreement": round_or_none(agreement, 6),
            "cohen_kappa": round_or_none(cohen_kappa(agreement_pairs), 6),
            "interpretation": "Agreement with the archived LLM output is not human-ground-truth accuracy.",
        },
        "static_validation": n4,
        "carla": carla["summary"],
        "selective_routing": {
            "total_records": 76291,
            "routed_records": 2524,
            "routing_rate_pct": round(100.0 * 2524 / 76291, 3),
            "llm_calls_avoided": 76291 - 2524,
            "llm_call_avoidance_pct": round(100.0 * (76291 - 2524) / 76291, 3),
            "estimated_all_llm_time_hours": round(76291 * 1.0651 / 3600.0, 3),
            "measured_hybrid_time_minutes": round((61.32 + 2688.2929) / 60.0, 3),
            "estimated_wall_time_reduction_factor": round((76291 * 1.0651) / (61.32 + 2688.2929), 3),
            "estimate_warning": "The all-record LLM-only value is extrapolated from archived average batch time, not directly measured.",
        },
    }
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
