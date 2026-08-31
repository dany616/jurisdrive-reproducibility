from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

from .io import DEFAULT_FULL_RUN, read_json, sha256_file, write_json, write_jsonl

STRATA = {
    "rule_car": ("output/car_to_car", 200, "car_to_car"),
    "rule_not": ("output/not_car_to_car", 200, "not_car_to_car"),
    "qwen_car": ("ambiguous_done/car_to_car", 150, "car_to_car"),
    "qwen_not": ("ambiguous_done/not_car_to_car", 150, "not_car_to_car"),
    "unresolved": ("ambiguous_done/ambiguous", 200, "unresolved"),
}


def _candidate_id(path: Path) -> int:
    return int(path.stem.removeprefix("zeroshot_test_").removesuffix("_result"))


def blinded_annotation_task(row: dict[str, Any]) -> dict[str, Any]:
    """Return only the fields a human annotator may see before labeling."""

    return {
        "candidate_id": row["candidate_id"],
        "source_file_sha256": row["source_file_sha256"],
        "source_text": row.get("source_text"),
    }


def blank_human_annotation(row: dict[str, Any]) -> dict[str, Any]:
    """Create a label template without route, model, or prediction leakage."""

    return {
        "candidate_id": row["candidate_id"],
        "source_file_sha256": row["source_file_sha256"],
        "label": None,
        "vehicle_count": None,
        "collision_agent": None,
        "collision_target": None,
        "legal_status": None,
        "evidence_quotes": [],
        "notes": None,
    }


def sample_gold(
    output_dir: Path,
    *,
    full_run_dir: Path = DEFAULT_FULL_RUN,
    seed: int = 20260728,
) -> dict[str, Any]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    strata_summary: dict[str, Any] = {}
    for stratum, (relative, requested, predicted) in STRATA.items():
        directory = full_run_dir / relative
        files = sorted(directory.glob("zeroshot_test_*_result.json"), key=_candidate_id)
        if len(files) < requested:
            raise ValueError(f"{stratum}: requested {requested}, found {len(files)}")
        selected = sorted(rng.sample(files, requested), key=_candidate_id)
        strata_summary[stratum] = {
            "population": len(files),
            "sampled": len(selected),
            "predicted_label": predicted,
        }
        for path in selected:
            record = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "candidate_id": _candidate_id(path),
                    "stratum": stratum,
                    "predicted_label": predicted,
                    "source_path": str(path),
                    "source_file_sha256": sha256_file(path),
                    "source_text": record.get("source_text"),
                    "rule": (record.get("postprocess") or {}).get("rule"),
                    "qwen": (record.get("postprocess") or {}).get("llm"),
                }
            )
    rows.sort(key=lambda row: (row["stratum"], row["candidate_id"]))
    tasks_path = output_dir / "annotation_tasks.jsonl"
    write_jsonl(tasks_path, rows)
    blinded_tasks_path = output_dir / "annotation_tasks_blinded.jsonl"
    write_jsonl(blinded_tasks_path, [blinded_annotation_task(row) for row in rows])
    label_rows = [blank_human_annotation(row) for row in rows]
    write_jsonl(output_dir / "annotator_a.jsonl", label_rows)
    write_jsonl(output_dir / "annotator_b.jsonl", label_rows)
    write_jsonl(output_dir / "adjudicated.jsonl", label_rows)
    write_jsonl(
        output_dir / "predictions_rule_only.jsonl",
        [
            {
                "candidate_id": row["candidate_id"],
                "prediction": (
                    row["predicted_label"] if row["stratum"] in {"rule_car", "rule_not"} else "abstain"
                ),
            }
            for row in rows
        ],
    )
    write_jsonl(
        output_dir / "predictions_hybrid.jsonl",
        [
            {
                "candidate_id": row["candidate_id"],
                "prediction": (
                    "abstain"
                    if row["predicted_label"] == "unresolved"
                    else row["predicted_label"]
                ),
            }
            for row in rows
        ],
    )
    write_jsonl(
        output_dir / "predictions_qwen_only.jsonl",
        [{"candidate_id": row["candidate_id"], "prediction": None} for row in rows],
    )
    summary = {
        "seed": seed,
        "total": len(rows),
        "strata": strata_summary,
        "tasks_sha256": sha256_file(tasks_path),
        "blinded_tasks_sha256": sha256_file(blinded_tasks_path),
        "blinded_task_fields": [
            "candidate_id",
            "source_file_sha256",
            "source_text",
        ],
        "human_labels_complete": False,
        "metrics_generated": False,
    }
    write_json(output_dir / "sampling_summary.json", summary)
    return summary


def _load_labels(path: Path) -> dict[int, dict[str, Any]]:
    labels: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            candidate_id = int(row["candidate_id"])
            if candidate_id in labels:
                raise ValueError(f"duplicate candidate_id in {path}: {candidate_id}")
            labels[candidate_id] = row
    return labels


def _require_complete(
    labels: dict[int, dict[str, Any]],
    path: Path,
    *,
    allowed: set[str] | None = None,
) -> None:
    allowed = allowed or {"car_to_car", "not_car_to_car"}
    missing = [candidate_id for candidate_id, row in labels.items() if row.get("label") not in allowed]
    if missing:
        raise ValueError(
            f"{path} has {len(missing)} incomplete/invalid labels; no metrics were generated"
        )


def cohens_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    if len(labels_a) != len(labels_b) or not labels_a:
        raise ValueError("label sequences must have equal non-zero length")
    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / len(labels_a)
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    expected = sum(
        counts_a[label] / len(labels_a) * counts_b[label] / len(labels_b)
        for label in set(counts_a) | set(counts_b)
    )
    return 1.0 if expected == 1.0 else (observed - expected) / (1.0 - expected)


def binary_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("label sequences must have equal non-zero length")
    positive = "car_to_car"
    tp = sum(t == positive and p == positive for t, p in zip(y_true, y_pred))
    tn = sum(t != positive and p != positive for t, p in zip(y_true, y_pred))
    fp = sum(t != positive and p == positive for t, p in zip(y_true, y_pred))
    fn = sum(t == positive and p != positive for t, p in zip(y_true, y_pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denominator if denominator else 0.0
    return {
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "false_acceptance_rate": fp / (tp + fp) if tp + fp else 0.0,
        "false_negative_rate": fn / (tp + fn) if tp + fn else 0.0,
    }


def weighted_binary_metrics(
    y_true: list[str], y_pred: list[str], weights: list[float]
) -> dict[str, Any]:
    if len(y_true) != len(y_pred) or len(y_true) != len(weights) or not y_true:
        raise ValueError("truth, prediction, and weight sequences must have equal non-zero length")
    positive = "car_to_car"
    tp = sum(weight for truth, pred, weight in zip(y_true, y_pred, weights) if truth == positive and pred == positive)
    tn = sum(weight for truth, pred, weight in zip(y_true, y_pred, weights) if truth != positive and pred != positive)
    fp = sum(weight for truth, pred, weight in zip(y_true, y_pred, weights) if truth != positive and pred == positive)
    fn = sum(weight for truth, pred, weight in zip(y_true, y_pred, weights) if truth == positive and pred != positive)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denominator if denominator else 0.0
    return {
        "weighted_confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "false_acceptance_rate": fp / (tp + fp) if tp + fp else 0.0,
        "false_negative_rate": fn / (tp + fn) if tp + fn else 0.0,
    }


def selective_metrics(
    truth_by_id: dict[int, str],
    predictions_by_id: dict[int, dict[str, Any]],
    weights_by_id: dict[int, float] | None = None,
) -> dict[str, Any]:
    valid = {"car_to_car", "not_car_to_car"}
    covered_ids = [
        candidate_id
        for candidate_id in sorted(truth_by_id)
        if predictions_by_id.get(candidate_id, {}).get("prediction") in valid
    ]
    pending = sum(
        predictions_by_id.get(candidate_id, {}).get("prediction") is None
        for candidate_id in truth_by_id
    )
    if not covered_ids:
        return {
            "status": "pending" if pending else "no_coverage",
            "coverage": 0.0,
            "covered": 0,
            "total": len(truth_by_id),
            "risk": None,
            "binary_metrics_on_covered": None,
        }
    truth = [truth_by_id[candidate_id] for candidate_id in covered_ids]
    predictions = [
        predictions_by_id[candidate_id]["prediction"] for candidate_id in covered_ids
    ]
    errors = sum(expected != observed for expected, observed in zip(truth, predictions))
    weights_by_id = weights_by_id or {candidate_id: 1.0 for candidate_id in truth_by_id}
    total_weight = sum(weights_by_id[candidate_id] for candidate_id in truth_by_id)
    covered_weights = [weights_by_id[candidate_id] for candidate_id in covered_ids]
    covered_weight = sum(covered_weights)
    weighted_errors = sum(
        weight
        for expected, observed, weight in zip(truth, predictions, covered_weights)
        if expected != observed
    )
    return {
        "status": "complete" if len(covered_ids) == len(truth_by_id) else "selective",
        "coverage": len(covered_ids) / len(truth_by_id),
        "covered": len(covered_ids),
        "total": len(truth_by_id),
        "risk": errors / len(covered_ids),
        "binary_metrics_on_covered": binary_metrics(truth, predictions),
        "population_weighted": {
            "coverage": covered_weight / total_weight if total_weight else 0.0,
            "risk": weighted_errors / covered_weight if covered_weight else None,
            "binary_metrics_on_covered": weighted_binary_metrics(
                truth, predictions, covered_weights
            ),
        },
    }


def gold_status(gold_dir: Path) -> dict[str, Any]:
    tasks = _load_labels(gold_dir / "annotation_tasks.jsonl")
    task_ids = set(tasks)
    roles: dict[str, Any] = {}
    loaded_roles: dict[str, dict[int, dict[str, Any]]] = {}
    for role in ("annotator_a", "annotator_b", "adjudicated"):
        path = gold_dir / f"{role}.jsonl"
        labels = _load_labels(path)
        loaded_roles[role] = labels
        values = Counter(row.get("label") or "missing" for row in labels.values())
        roles[role] = {
            "rows": len(labels),
            "id_set_matches_tasks": set(labels) == task_ids,
            "labels": dict(sorted(values.items())),
            "binary_complete": sum(values[label] for label in ("car_to_car", "not_car_to_car")),
            "missing": values["missing"],
            "uncertain": values["uncertain"],
        }
    a = loaded_roles["annotator_a"]
    b = loaded_roles["annotator_b"]
    binary_overlap = [
        candidate_id
        for candidate_id in sorted(task_ids)
        if a[candidate_id].get("label") in {"car_to_car", "not_car_to_car"}
        and b[candidate_id].get("label") in {"car_to_car", "not_car_to_car"}
    ]
    disagreements = [
        candidate_id
        for candidate_id in binary_overlap
        if a[candidate_id]["label"] != b[candidate_id]["label"]
    ]
    prediction_status = {}
    for method in ("rule_only", "qwen_only", "hybrid"):
        predictions = _load_labels(gold_dir / f"predictions_{method}.jsonl")
        counts = Counter(row.get("prediction") or "missing" for row in predictions.values())
        prediction_status[method] = {
            "rows": len(predictions),
            "id_set_matches_tasks": set(predictions) == task_ids,
            "predictions": dict(sorted(counts.items())),
        }
    return {
        "gold_dir": str(gold_dir.resolve()),
        "task_count": len(tasks),
        "roles": roles,
        "inter_annotator": {
            "binary_overlap": len(binary_overlap),
            "binary_overlap_rate": len(binary_overlap) / len(tasks) if tasks else 0.0,
            "disagreements": len(disagreements),
            "cohens_kappa": cohens_kappa(
                [a[candidate_id]["label"] for candidate_id in binary_overlap],
                [b[candidate_id]["label"] for candidate_id in binary_overlap],
            )
            if binary_overlap
            else None,
        },
        "predictions": prediction_status,
        "metrics_ready": roles["adjudicated"]["binary_complete"] == len(tasks),
    }


def benchmark_gold(gold_dir: Path, output_path: Path) -> dict[str, Any]:
    tasks = _load_labels(gold_dir / "annotation_tasks.jsonl")
    annotator_a = _load_labels(gold_dir / "annotator_a.jsonl")
    annotator_b = _load_labels(gold_dir / "annotator_b.jsonl")
    adjudicated = _load_labels(gold_dir / "adjudicated.jsonl")
    task_ids = set(tasks)
    for path, labels in (
        (gold_dir / "annotator_a.jsonl", annotator_a),
        (gold_dir / "annotator_b.jsonl", annotator_b),
        (gold_dir / "adjudicated.jsonl", adjudicated),
    ):
        if set(labels) != task_ids:
            raise ValueError(f"{path} candidate IDs do not exactly match annotation_tasks.jsonl")
    _require_complete(
        annotator_a,
        gold_dir / "annotator_a.jsonl",
        allowed={"car_to_car", "not_car_to_car", "uncertain"},
    )
    _require_complete(
        annotator_b,
        gold_dir / "annotator_b.jsonl",
        allowed={"car_to_car", "not_car_to_car", "uncertain"},
    )
    _require_complete(adjudicated, gold_dir / "adjudicated.jsonl")
    ids = sorted(task_ids)
    if len(ids) != 900:
        raise ValueError(f"Expected 900 common labels, found {len(ids)}")
    truth_by_id = {
        candidate_id: adjudicated[candidate_id]["label"] for candidate_id in ids
    }
    method_files = {
        "rule_only": gold_dir / "predictions_rule_only.jsonl",
        "qwen_only": gold_dir / "predictions_qwen_only.jsonl",
        "hybrid": gold_dir / "predictions_hybrid.jsonl",
    }
    summary = read_json(gold_dir / "sampling_summary.json")
    weights_by_id = {
        candidate_id: float(summary["strata"][tasks[candidate_id]["stratum"]]["population"])
        / float(summary["strata"][tasks[candidate_id]["stratum"]]["sampled"])
        for candidate_id in ids
    }
    agreement_ids = [
        candidate_id
        for candidate_id in ids
        if annotator_a[candidate_id]["label"] in {"car_to_car", "not_car_to_car"}
        and annotator_b[candidate_id]["label"] in {"car_to_car", "not_car_to_car"}
    ]
    payload = {
        "sample_size": len(ids),
        "agreement_sample_size": len(agreement_ids),
        "agreement_coverage": len(agreement_ids) / len(ids),
        "annotator_a_uncertain": sum(annotator_a[candidate_id]["label"] == "uncertain" for candidate_id in ids),
        "annotator_b_uncertain": sum(annotator_b[candidate_id]["label"] == "uncertain" for candidate_id in ids),
        "cohens_kappa": cohens_kappa(
            [annotator_a[candidate_id]["label"] for candidate_id in agreement_ids],
            [annotator_b[candidate_id]["label"] for candidate_id in agreement_ids],
        )
        if agreement_ids
        else None,
        "methods": {
            method: selective_metrics(
                truth_by_id, _load_labels(path), weights_by_id=weights_by_id
            )
            for method, path in method_files.items()
        },
        "abstention_policy": "excluded from covered-set metrics and reported through coverage",
        "sampling_policy": "unweighted sample metrics and inverse-probability population-weighted metrics are both reported",
    }
    write_json(output_path, payload)
    return payload
