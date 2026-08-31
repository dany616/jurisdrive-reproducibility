"""Execution and denominator-safe analysis helpers for sanitized RQ4 Task A.

This module keeps model-visible request reconstruction separate from the private
label mapping used after inference.  It has no dependency outside the Python
standard library so the frozen suite can run from the WSL execution host.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


VALID_STATUSES = frozenset({"PASS", "FAIL", "MANUAL_REVIEW"})
VALID_ISSUE_CODES = frozenset(
    {
        "ACTOR_RELATION",
        "TRAJECTORY_GEOMETRY",
        "EVENT_SEQUENCE",
        "COLLISION_EVIDENCE",
        "MODALITY_ALIGNMENT",
        "INSUFFICIENT_EVIDENCE",
    }
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def validate_verdict(value: Any) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(value, Mapping):
        return None, "verdict is not an object"
    status = value.get("status")
    issue_codes = value.get("issue_codes")
    rationale = value.get("rationale")
    if status not in VALID_STATUSES:
        return None, f"invalid status: {status!r}"
    if not isinstance(issue_codes, list) or any(code not in VALID_ISSUE_CODES for code in issue_codes):
        return None, "invalid issue_codes"
    if len(set(issue_codes)) != len(issue_codes):
        return None, "duplicate issue_codes"
    if not isinstance(rationale, str) or len(rationale) > 600:
        return None, "invalid rationale"
    return {
        "status": str(status),
        "issue_codes": [str(code) for code in issue_codes],
        "rationale": rationale,
    }, None


def response_content(response: Mapping[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("response has no choices[0].message.content") from exc
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text", "")) for item in content if isinstance(item, Mapping)
        ).strip()
    return str(content).strip()


def parse_verdict_response(response: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str | None, str]:
    try:
        text = response_content(response)
    except ValueError as exc:
        return None, str(exc), ""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"JSON parse error: {exc}", text
    verdict, error = validate_verdict(parsed)
    return verdict, error, text


def conservative_repeat_consensus(statuses: Sequence[str | None]) -> str:
    """Require three identical valid outputs; otherwise route to review."""

    if len(statuses) != 3:
        return "MANUAL_REVIEW"
    normalized = [status if status in VALID_STATUSES else None for status in statuses]
    if normalized[0] is not None and len(set(normalized)) == 1:
        return str(normalized[0])
    return "MANUAL_REVIEW"


def _rate(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def task_a_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute full-denominator selective detection metrics.

    FAIL is a defect detection, PASS is acceptance, and MANUAL_REVIEW is an
    abstention.  Reviews remain outside decisive confusion cells but remain in
    the fault/clean denominators used for recall and specificity.
    """

    total = len(rows)
    faults = sum(bool(row["is_fault"]) for row in rows)
    clean = total - faults
    tp = sum(bool(row["is_fault"]) and row["status"] == "FAIL" for row in rows)
    fn = sum(bool(row["is_fault"]) and row["status"] == "PASS" for row in rows)
    fp = sum(not bool(row["is_fault"]) and row["status"] == "FAIL" for row in rows)
    tn = sum(not bool(row["is_fault"]) and row["status"] == "PASS" for row in rows)
    review_fault = sum(bool(row["is_fault"]) and row["status"] == "MANUAL_REVIEW" for row in rows)
    review_clean = sum(not bool(row["is_fault"]) and row["status"] == "MANUAL_REVIEW" for row in rows)
    review = review_fault + review_clean
    decisive = tp + tn + fp + fn
    precision = _rate(tp, tp + fp)
    recall = _rate(tp, faults)
    specificity = _rate(tn, clean)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    balanced = (
        (recall + specificity) / 2
        if recall is not None and specificity is not None
        else None
    )
    return {
        "n": total,
        "faults": faults,
        "clean_controls": clean,
        "decisive_confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "manual_review": {"fault": review_fault, "clean": review_clean, "total": review},
        "precision": precision,
        "recall_full_fault_denominator": recall,
        "f1_full_fault_denominator": f1,
        "specificity_full_clean_denominator": specificity,
        "balanced_accuracy": balanced,
        "false_acceptance_rate": _rate(fn, faults),
        "false_rejection_rate": _rate(fp, clean),
        "coverage": _rate(decisive, total),
        "abstention_rate": _rate(review, total),
        "manual_review_rate": _rate(review, total),
        "decisive_selective_recall": _rate(tp, tp + fn),
        "decisive_selective_specificity": _rate(tn, tn + fp),
    }


BOOTSTRAP_METRICS = (
    "precision",
    "recall_full_fault_denominator",
    "f1_full_fault_denominator",
    "specificity_full_clean_denominator",
    "balanced_accuracy",
    "false_acceptance_rate",
    "false_rejection_rate",
    "coverage",
    "abstention_rate",
)


def _percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def judgment_cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]], *, samples: int, seed: int
) -> dict[str, Any]:
    by_cluster: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cluster[str(row["judgment_slot"])].append(row)
    clusters = sorted(by_cluster)
    if len(clusters) != 24:
        raise ValueError(f"expected 24 judgment clusters, found {len(clusters)}")
    rng = random.Random(seed)
    distributions: dict[str, list[float]] = {name: [] for name in BOOTSTRAP_METRICS}
    for _ in range(samples):
        sampled: list[Mapping[str, Any]] = []
        for _slot in range(len(clusters)):
            sampled.extend(by_cluster[rng.choice(clusters)])
        metrics = task_a_metrics(sampled)
        for name in BOOTSTRAP_METRICS:
            value = metrics.get(name)
            if isinstance(value, (int, float)):
                distributions[name].append(float(value))
    return {
        "method": "judgment-cluster percentile bootstrap",
        "primary_unit": "judgment_slot",
        "clusters": len(clusters),
        "samples": samples,
        "seed": seed,
        "metrics": {
            name: {
                "lower": _percentile(values, 0.025),
                "upper": _percentile(values, 0.975),
                "valid_replicates": len(values),
            }
            for name, values in distributions.items()
        },
    }


def exact_mcnemar_p(b: int, c: int) -> float:
    discordant = b + c
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(0, min(b, c) + 1)) / (2**discordant)
    return min(1.0, 2 * tail)


def paired_common_coverage_test(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    left_by_id = {str(row["opaque_artifact_id"]): row for row in left}
    right_by_id = {str(row["opaque_artifact_id"]): row for row in right}
    common_ids = sorted(set(left_by_id) & set(right_by_id))
    paired: list[tuple[str, float, float]] = []
    b = c = 0
    for artifact_id in common_ids:
        lrow, rrow = left_by_id[artifact_id], right_by_id[artifact_id]
        if lrow["status"] == "MANUAL_REVIEW" or rrow["status"] == "MANUAL_REVIEW":
            continue
        expected = "FAIL" if bool(lrow["is_fault"]) else "PASS"
        lcorrect = 1.0 if lrow["status"] == expected else 0.0
        rcorrect = 1.0 if rrow["status"] == expected else 0.0
        if lcorrect and not rcorrect:
            b += 1
        elif rcorrect and not lcorrect:
            c += 1
        paired.append((str(lrow["judgment_slot"]), lcorrect, rcorrect))
    by_cluster: dict[str, list[float]] = defaultdict(list)
    for cluster, lcorrect, rcorrect in paired:
        by_cluster[cluster].append(lcorrect - rcorrect)
    cluster_differences = [sum(values) / len(values) for _, values in sorted(by_cluster.items())]
    observed = sum(cluster_differences) / len(cluster_differences) if cluster_differences else 0.0
    rng = random.Random(seed)
    extreme = 0
    for _ in range(samples):
        permuted = sum(value * (-1 if rng.random() < 0.5 else 1) for value in cluster_differences)
        statistic = permuted / len(cluster_differences) if cluster_differences else 0.0
        if abs(statistic) >= abs(observed) - 1e-15:
            extreme += 1
    return {
        "common_decisive_artifacts": len(paired),
        "common_judgment_clusters": len(cluster_differences),
        "left_only_correct_b": b,
        "right_only_correct_c": c,
        "row_level_exact_mcnemar_p_descriptive": exact_mcnemar_p(b, c),
        "cluster_mean_accuracy_difference_left_minus_right": observed,
        "cluster_sign_flip": {
            "samples": samples,
            "seed": seed,
            "two_sided_p": (extreme + 1) / (samples + 1),
        },
        "scope": "paired inference uses only artifacts decisively covered by both methods; judgment is the resampling unit",
    }


def per_fault_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for fault_type in sorted({str(row.get("fault_type")) for row in rows if row.get("fault_type")}):
        stratum = [row for row in rows if row.get("fault_type") == fault_type]
        result[fault_type] = {
            "n": len(stratum),
            "fail": sum(row["status"] == "FAIL" for row in stratum),
            "pass": sum(row["status"] == "PASS" for row in stratum),
            "manual_review": sum(row["status"] == "MANUAL_REVIEW" for row in stratum),
            "detection_rate": _rate(sum(row["status"] == "FAIL" for row in stratum), len(stratum)),
            "false_acceptance_rate": _rate(sum(row["status"] == "PASS" for row in stratum), len(stratum)),
            "review_rate": _rate(sum(row["status"] == "MANUAL_REVIEW" for row in stratum), len(stratum)),
        }
    return result


def repeat_agreement(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_key: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[(str(row["request_id"]), str(row["method_code"]))].append(row)
    counts = Counter()
    for group in by_key.values():
        statuses = [row.get("status") for row in sorted(group, key=lambda item: int(item["order_seed"]))]
        if len(statuses) != 3:
            counts["incomplete"] += 1
        elif len(set(statuses)) == 1 and statuses[0] in VALID_STATUSES:
            counts["exact_status_agreement"] += 1
        else:
            counts["status_disagreement"] += 1
    total = len(by_key)
    return {
        "requests": total,
        **dict(counts),
        "exact_status_agreement_rate": _rate(counts["exact_status_agreement"], total),
    }

