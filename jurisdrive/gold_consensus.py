from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CANONICAL_LABELS = {"ACCEPT", "REJECT", "UNRESOLVED"}
BINARY_LABELS = {"ACCEPT", "REJECT"}
LABEL_ALIASES = {
    "accept": "ACCEPT",
    "accepted": "ACCEPT",
    "car_to_car": "ACCEPT",
    "reject": "REJECT",
    "rejected": "REJECT",
    "not_car_to_car": "REJECT",
    "unresolved": "UNRESOLVED",
    "uncertain": "UNRESOLVED",
    "abstain": "UNRESOLVED",
    "abstained": "UNRESOLVED",
}
SEMANTIC_FIELDS = (
    "vehicle_count",
    "collision_agent",
    "collision_target",
    "legal_status",
    "evidence_quotes",
)
PROTOCOL_VERSION = "jurisdrive-dual-human-consensus-v1"


def normalize_label(value: Any, *, allow_missing: bool = False) -> str | None:
    """Normalize legacy/UI labels to the paper-facing three-way decision enum."""

    if value is None or (isinstance(value, str) and not value.strip()):
        if allow_missing:
            return None
        raise ValueError("decision label is missing")
    text = str(value).strip()
    upper = text.upper()
    if upper in CANONICAL_LABELS:
        return upper
    normalized = LABEL_ALIASES.get(text.lower())
    if normalized is None:
        raise ValueError(f"unsupported decision label: {value!r}")
    return normalized


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record_sha256(row: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json(row).encode("utf-8"))


def read_jsonl_index(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                candidate_id = int(row["candidate_id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid candidate_id") from exc
            if candidate_id in rows:
                raise ValueError(f"{path}:{line_number}: duplicate candidate_id {candidate_id}")
            rows[candidate_id] = row
    return rows


def _atomic_write_text(path: Path, text: str, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing freeze artifact: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]], *, overwrite: bool) -> None:
    payload = "".join(canonical_json(dict(row)) + "\n" for row in rows)
    _atomic_write_text(path, payload, overwrite=overwrite)


def write_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool) -> None:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    _atomic_write_text(path, text, overwrite=overwrite)


def _validate_id_sets(
    tasks: Mapping[int, Mapping[str, Any]],
    annotator_a: Mapping[int, Mapping[str, Any]],
    annotator_b: Mapping[int, Mapping[str, Any]],
) -> None:
    expected = set(tasks)
    for name, rows in (("annotator_a", annotator_a), ("annotator_b", annotator_b)):
        missing = sorted(expected - set(rows))
        unexpected = sorted(set(rows) - expected)
        if missing or unexpected:
            raise ValueError(
                f"{name} candidate IDs differ from tasks: "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )


def _validate_source_hash(task: Mapping[str, Any], annotation: Mapping[str, Any], role: str) -> None:
    task_hash = task.get("source_file_sha256")
    annotation_hash = annotation.get("source_file_sha256")
    if task_hash and annotation_hash and task_hash != annotation_hash:
        raise ValueError(
            f"candidate {task.get('candidate_id')}: {role} source hash differs from task"
        )


def _semantic_reference(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    agreed: dict[str, Any] = {}
    disagreements: list[str] = []
    for field in SEMANTIC_FIELDS:
        if left.get(field) == right.get(field):
            agreed[field] = left.get(field)
        else:
            disagreements.append(field)
    return agreed, disagreements


def _largest_remainder_allocation(group_sizes: Mapping[str, int], requested: int) -> dict[str, int]:
    total = sum(group_sizes.values())
    if requested < 0 or requested > total:
        raise ValueError(f"semantic review sample {requested} exceeds pool {total}")
    if not requested or not total:
        return {key: 0 for key in group_sizes}
    raw = {key: requested * size / total for key, size in group_sizes.items()}
    allocated = {key: min(group_sizes[key], math.floor(value)) for key, value in raw.items()}
    remaining = requested - sum(allocated.values())
    order = sorted(
        group_sizes,
        key=lambda key: (raw[key] - allocated[key], group_sizes[key], key),
        reverse=True,
    )
    while remaining:
        progressed = False
        for key in order:
            if allocated[key] < group_sizes[key]:
                allocated[key] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            raise RuntimeError("unable to allocate semantic review sample")
    return allocated


def _semantic_review_rows(
    consensus_rows: Sequence[Mapping[str, Any]],
    review_rows: Sequence[Mapping[str, Any]],
    tasks: Mapping[int, Mapping[str, Any]],
    *,
    sample_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in consensus_rows:
        if row["gold_label"] != "ACCEPT":
            continue
        rows.append(
            {
                "candidate_id": row["candidate_id"],
                "stratum": row.get("stratum"),
                "source_file_sha256": row.get("source_file_sha256"),
                "source_path": tasks[int(row["candidate_id"])].get("source_path"),
                "source_text": tasks[int(row["candidate_id"])].get("source_text"),
                "reference_label": "ACCEPT",
                "sampling_role": "all_consensus_accept",
                "semantic_reference": row.get("semantic_reference", {}),
                "semantic_disagreement_fields": row.get("semantic_disagreement_fields", []),
                "semantic_review_status": (
                    "dual_human_fields_agree"
                    if not row.get("semantic_disagreement_fields")
                    else "requires_semantic_review"
                ),
            }
        )

    pool: list[dict[str, Any]] = []
    for row in consensus_rows:
        if row["gold_label"] == "REJECT":
            pool.append(dict(row))
    for row in review_rows:
        pool.append(dict(row))

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pool:
        group = f"{row.get('stratum', 'unknown')}|{row['gold_label']}"
        grouped[group].append(row)
    allocation = _largest_remainder_allocation(
        {group: len(group_rows) for group, group_rows in grouped.items()}, sample_size
    )
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for group in sorted(grouped):
        group_rows = sorted(grouped[group], key=lambda row: int(row["candidate_id"]))
        selected.extend(rng.sample(group_rows, allocation[group]))
    for row in sorted(selected, key=lambda item: int(item["candidate_id"])):
        task = tasks[int(row["candidate_id"])]
        rows.append(
            {
                "candidate_id": row["candidate_id"],
                "stratum": row.get("stratum"),
                "source_file_sha256": row.get("source_file_sha256"),
                "source_path": task.get("source_path"),
                "source_text": task.get("source_text"),
                "reference_label": row["gold_label"],
                "sampling_role": "stratified_reject_or_review_sample",
                "semantic_review_status": "requires_semantic_review",
                "source_text_sha256": sha256_bytes(
                    str(task.get("source_text") or "").encode("utf-8")
                ),
                "vehicle_entities": [],
                "collision_agent": None,
                "collision_target": None,
                "legal_status": None,
                "evidence_span_sufficient": None,
                "unsupported_relation_count": None,
                "notes": None,
            }
        )
    return sorted(rows, key=lambda row: int(row["candidate_id"]))


def freeze_dual_human_consensus(
    *,
    tasks_path: Path,
    annotator_a_path: Path,
    annotator_b_path: Path,
    output_dir: Path,
    expected_total: int | None = 900,
    expected_consensus: int | None = 743,
    expected_review: int | None = 157,
    semantic_review_sample_size: int = 100,
    semantic_sample_seed: int = 20260823,
    protocol_version: str = PROTOCOL_VERSION,
    protocol_statement: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Freeze dual-human binary consensus and keep every uncertain case unresolved."""

    protocol_version = str(protocol_version).strip()
    if not protocol_version:
        raise ValueError("protocol_version must be non-empty")
    tasks = read_jsonl_index(tasks_path)
    annotator_a = read_jsonl_index(annotator_a_path)
    annotator_b = read_jsonl_index(annotator_b_path)
    _validate_id_sets(tasks, annotator_a, annotator_b)
    if expected_total is not None and len(tasks) != expected_total:
        raise ValueError(f"expected {expected_total} tasks, found {len(tasks)}")

    consensus_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    pair_counts: Counter[str] = Counter()
    for candidate_id in sorted(tasks):
        task = tasks[candidate_id]
        left = annotator_a[candidate_id]
        right = annotator_b[candidate_id]
        _validate_source_hash(task, left, "annotator_a")
        _validate_source_hash(task, right, "annotator_b")
        left_label = normalize_label(left.get("label"))
        right_label = normalize_label(right.get("label"))
        pair_counts[f"{left_label}|{right_label}"] += 1
        base = {
            "candidate_id": candidate_id,
            "stratum": task.get("stratum"),
            "source_file_sha256": task.get("source_file_sha256"),
            "annotator_a_record_sha256": record_sha256(left),
            "annotator_b_record_sha256": record_sha256(right),
        }
        if left_label == right_label and left_label in BINARY_LABELS:
            semantic_reference, disagreements = _semantic_reference(left, right)
            consensus_rows.append(
                {
                    **base,
                    "gold_label": left_label,
                    "reference_status": "dual_human_consensus",
                    "semantic_reference": semantic_reference,
                    "semantic_disagreement_fields": disagreements,
                }
            )
            continue

        if left_label == right_label == "UNRESOLVED":
            reason = "common_uncertain"
        elif "UNRESOLVED" in {left_label, right_label}:
            reason = "one_annotator_uncertain"
        else:
            reason = "binary_label_disagreement"
        review_rows.append(
            {
                **base,
                "gold_label": "UNRESOLVED",
                "reference_status": "additional_review_required",
                "review_reason": reason,
                "annotator_a_label": left_label,
                "annotator_b_label": right_label,
                "adjudication_label": None,
                "adjudication_notes": None,
            }
        )

    if expected_consensus is not None and len(consensus_rows) != expected_consensus:
        raise ValueError(
            f"expected {expected_consensus} consensus rows, found {len(consensus_rows)}"
        )
    if expected_review is not None and len(review_rows) != expected_review:
        raise ValueError(f"expected {expected_review} review rows, found {len(review_rows)}")
    if any(row["gold_label"] != "UNRESOLVED" for row in review_rows):
        raise AssertionError("review candidates must never be merged into a binary label")

    full_rows = sorted(
        [*consensus_rows, *review_rows], key=lambda row: int(row["candidate_id"])
    )
    semantic_rows = _semantic_review_rows(
        consensus_rows,
        review_rows,
        tasks,
        sample_size=semantic_review_sample_size,
        seed=semantic_sample_seed,
    )
    output_paths = {
        "consensus_gold": output_dir / "consensus_gold_743.jsonl",
        "additional_review_queue": output_dir / "additional_review_queue_157.jsonl",
        "full_selective_reference": output_dir / "full_selective_reference_900.jsonl",
        "semantic_review_tasks": output_dir / "semantic_review_tasks.jsonl",
    }
    manifest_path = output_dir / "consensus_freeze_manifest.json"
    if not overwrite:
        existing = [path for path in [*output_paths.values(), manifest_path] if path.exists()]
        if existing:
            raise FileExistsError(
                "refusing to overwrite existing freeze artifacts: "
                + ", ".join(str(path) for path in existing)
            )
    write_jsonl(output_paths["consensus_gold"], consensus_rows, overwrite=overwrite)
    write_jsonl(output_paths["additional_review_queue"], review_rows, overwrite=overwrite)
    write_jsonl(output_paths["full_selective_reference"], full_rows, overwrite=overwrite)
    write_jsonl(output_paths["semantic_review_tasks"], semantic_rows, overwrite=overwrite)

    input_paths = {
        "annotation_tasks": tasks_path,
        "annotator_a": annotator_a_path,
        "annotator_b": annotator_b_path,
    }
    manifest: dict[str, Any] = {
        "protocol_version": protocol_version,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "labels": sorted(CANONICAL_LABELS),
        "policy": {
            "binary_gold": "only exact ACCEPT/REJECT agreement from both annotators",
            "uncertain": "never coerced to REJECT; routed to additional review",
            "full_reference": (
                "review-pending cases are represented as UNRESOLVED and must not be used "
                "as binary truth"
            ),
        },
        "counts": {
            "total": len(tasks),
            "consensus": len(consensus_rows),
            "consensus_accept": sum(row["gold_label"] == "ACCEPT" for row in consensus_rows),
            "consensus_reject": sum(row["gold_label"] == "REJECT" for row in consensus_rows),
            "additional_review": len(review_rows),
            "common_uncertain": sum(
                row["review_reason"] == "common_uncertain" for row in review_rows
            ),
            "one_annotator_uncertain": sum(
                row["review_reason"] == "one_annotator_uncertain" for row in review_rows
            ),
            "binary_label_disagreement": sum(
                row["review_reason"] == "binary_label_disagreement" for row in review_rows
            ),
            "semantic_review_tasks": len(semantic_rows),
        },
        "annotator_pair_counts": dict(sorted(pair_counts.items())),
        "semantic_sampling": {
            "all_consensus_accept": sum(
                row["sampling_role"] == "all_consensus_accept" for row in semantic_rows
            ),
            "stratified_reject_or_review": semantic_review_sample_size,
            "seed": semantic_sample_seed,
        },
        "inputs": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "rows": len(read_jsonl_index(path)),
            }
            for name, path in input_paths.items()
        },
        "outputs": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "rows": len(read_jsonl_index(path)),
            }
            for name, path in output_paths.items()
        },
    }
    if protocol_statement is not None:
        manifest["protocol_statement"] = dict(protocol_statement)
    manifest["freeze_digest"] = sha256_bytes(
        canonical_json(
            {
                "protocol_version": manifest["protocol_version"],
                "protocol_statement": manifest.get("protocol_statement"),
                "policy": manifest["policy"],
                "counts": manifest["counts"],
                "inputs": {
                    name: value["sha256"] for name, value in manifest["inputs"].items()
                },
                "outputs": {
                    name: value["sha256"] for name, value in manifest["outputs"].items()
                },
                "semantic_sampling": manifest["semantic_sampling"],
            }
        ).encode("utf-8")
    )
    write_json(manifest_path, manifest, overwrite=overwrite)
    return manifest


def apply_additional_review(
    *,
    full_reference_path: Path,
    adjudication_path: Path,
    output_path: Path,
    manifest_path: Path,
    expected_review: int | None = 157,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Merge only the queued review decisions; UNRESOLVED remains a valid final state."""

    full_rows = read_jsonl_index(full_reference_path)
    adjudication = read_jsonl_index(adjudication_path)
    review_ids = {
        candidate_id
        for candidate_id, row in full_rows.items()
        if row.get("reference_status") == "additional_review_required"
    }
    if expected_review is not None and len(review_ids) != expected_review:
        raise ValueError(f"expected {expected_review} queued rows, found {len(review_ids)}")
    if set(adjudication) != review_ids:
        raise ValueError(
            "adjudication IDs must exactly match the additional-review queue: "
            f"missing={len(review_ids - set(adjudication))}, "
            f"unexpected={len(set(adjudication) - review_ids)}"
        )
    merged: list[dict[str, Any]] = []
    for candidate_id in sorted(full_rows):
        row = dict(full_rows[candidate_id])
        if candidate_id not in review_ids:
            merged.append(row)
            continue
        decision = normalize_label(adjudication[candidate_id].get("label"))
        row.update(
            {
                "gold_label": decision,
                "reference_status": (
                    "adjudicated_unresolved"
                    if decision == "UNRESOLVED"
                    else "adjudicated_binary"
                ),
                "adjudication_record_sha256": record_sha256(adjudication[candidate_id]),
            }
        )
        merged.append(row)
    if not overwrite:
        existing = [path for path in (output_path, manifest_path) if path.exists()]
        if existing:
            raise FileExistsError(
                "refusing to overwrite existing adjudication artifacts: "
                + ", ".join(str(path) for path in existing)
            )
    write_jsonl(output_path, merged, overwrite=overwrite)
    counts = Counter(row["gold_label"] for row in merged)
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "policy": "Additional review may end in ACCEPT, REJECT, or UNRESOLVED; no coercion is applied.",
        "counts": {"total": len(merged), **dict(sorted(counts.items()))},
        "inputs": {
            "full_reference": {
                "path": str(full_reference_path.resolve()),
                "sha256": sha256_file(full_reference_path),
            },
            "adjudication": {
                "path": str(adjudication_path.resolve()),
                "sha256": sha256_file(adjudication_path),
                "rows": len(adjudication),
            },
        },
        "output": {
            "path": str(output_path.resolve()),
            "sha256": sha256_file(output_path),
            "rows": len(merged),
        },
    }
    write_json(manifest_path, manifest, overwrite=overwrite)
    return manifest


def _binary_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, Any]:
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("truth and prediction must have equal non-zero length")
    tp = sum(t == "ACCEPT" and p == "ACCEPT" for t, p in zip(y_true, y_pred))
    tn = sum(t == "REJECT" and p == "REJECT" for t, p in zip(y_true, y_pred))
    fp = sum(t == "REJECT" and p == "ACCEPT" for t, p in zip(y_true, y_pred))
    fn = sum(t == "ACCEPT" and p == "REJECT" for t, p in zip(y_true, y_pred))
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
    }


def _method_metrics(
    truth_by_id: Mapping[int, str], prediction_by_id: Mapping[int, str]
) -> dict[str, Any]:
    ids = sorted(truth_by_id)
    covered = [candidate_id for candidate_id in ids if prediction_by_id[candidate_id] in BINARY_LABELS]
    result: dict[str, Any] = {
        "n": len(ids),
        "covered": len(covered),
        "abstained": len(ids) - len(covered),
        "coverage": len(covered) / len(ids) if ids else 0.0,
        "abstention_rate": (len(ids) - len(covered)) / len(ids) if ids else 0.0,
    }
    if not covered:
        result.update({"selective_risk": None, "binary_metrics_on_covered": None})
        return result
    truth = [truth_by_id[candidate_id] for candidate_id in covered]
    predictions = [prediction_by_id[candidate_id] for candidate_id in covered]
    errors = sum(left != right for left, right in zip(truth, predictions))
    result.update(
        {
            "selective_risk": errors / len(covered),
            "binary_metrics_on_covered": _binary_metrics(truth, predictions),
        }
    )
    return result


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def stratified_bootstrap_ci(
    truth_by_id: Mapping[int, str],
    prediction_by_id: Mapping[int, str],
    *,
    samples: int = 10_000,
    seed: int = 20260823,
) -> dict[str, Any]:
    if samples <= 0:
        return {"samples": 0, "seed": seed, "confidence": 0.95, "metrics": {}}
    strata = {
        label: [candidate_id for candidate_id, truth in truth_by_id.items() if truth == label]
        for label in sorted(BINARY_LABELS)
    }
    if not all(strata.values()):
        raise ValueError("stratified bootstrap requires both ACCEPT and REJECT gold rows")
    rng = random.Random(seed)
    collected: dict[str, list[float]] = defaultdict(list)
    for _ in range(samples):
        sampled_ids: list[int] = []
        for label in sorted(strata):
            ids = strata[label]
            sampled_ids.extend(rng.choice(ids) for _ in ids)
        sampled_truth = {index: truth_by_id[candidate_id] for index, candidate_id in enumerate(sampled_ids)}
        sampled_predictions = {
            index: prediction_by_id[candidate_id] for index, candidate_id in enumerate(sampled_ids)
        }
        result = _method_metrics(sampled_truth, sampled_predictions)
        for key in ("coverage", "abstention_rate", "selective_risk"):
            if result.get(key) is not None:
                collected[key].append(float(result[key]))
        binary = result.get("binary_metrics_on_covered") or {}
        for key in ("precision", "recall", "f1", "mcc"):
            if binary.get(key) is not None:
                collected[key].append(float(binary[key]))
    return {
        "samples": samples,
        "seed": seed,
        "confidence": 0.95,
        "method": "class-stratified percentile bootstrap",
        "metrics": {
            key: {"low": _percentile(values, 0.025), "high": _percentile(values, 0.975)}
            for key, values in sorted(collected.items())
        },
    }


def _binomial_probability(n: int, k: int) -> float:
    return math.comb(n, k) * (0.5**n)


def mcnemar_exact(
    truth_by_id: Mapping[int, str],
    prediction_a: Mapping[int, str],
    prediction_b: Mapping[int, str],
) -> dict[str, Any]:
    common = [
        candidate_id
        for candidate_id in sorted(truth_by_id)
        if prediction_a[candidate_id] in BINARY_LABELS
        and prediction_b[candidate_id] in BINARY_LABELS
    ]
    a_correct_b_wrong = sum(
        prediction_a[candidate_id] == truth_by_id[candidate_id]
        and prediction_b[candidate_id] != truth_by_id[candidate_id]
        for candidate_id in common
    )
    a_wrong_b_correct = sum(
        prediction_a[candidate_id] != truth_by_id[candidate_id]
        and prediction_b[candidate_id] == truth_by_id[candidate_id]
        for candidate_id in common
    )
    discordant = a_correct_b_wrong + a_wrong_b_correct
    if not discordant:
        p_value = 1.0
    else:
        tail = sum(
            _binomial_probability(discordant, k)
            for k in range(0, min(a_correct_b_wrong, a_wrong_b_correct) + 1)
        )
        p_value = min(1.0, 2.0 * tail)
    return {
        "common_coverage_n": len(common),
        "a_correct_b_wrong": a_correct_b_wrong,
        "a_wrong_b_correct": a_wrong_b_correct,
        "discordant_n": discordant,
        "two_sided_exact_p": p_value,
    }


def _load_predictions(path: Path, expected_ids: set[int]) -> dict[int, str]:
    rows = read_jsonl_index(path)
    if set(rows) != expected_ids:
        raise ValueError(
            f"{path}: prediction IDs differ from reference "
            f"(missing={len(expected_ids - set(rows))}, unexpected={len(set(rows) - expected_ids)})"
        )
    predictions: dict[int, str] = {}
    for candidate_id, row in rows.items():
        predictions[candidate_id] = normalize_label(row.get("prediction"))  # type: ignore[assignment]
    return predictions


def _unresolved_detection(
    full_truth: Mapping[int, str], predictions: Mapping[int, str]
) -> dict[str, Any]:
    tp = sum(
        truth == "UNRESOLVED" and predictions[candidate_id] == "UNRESOLVED"
        for candidate_id, truth in full_truth.items()
    )
    fp = sum(
        truth in BINARY_LABELS and predictions[candidate_id] == "UNRESOLVED"
        for candidate_id, truth in full_truth.items()
    )
    fn = sum(
        truth == "UNRESOLVED" and predictions[candidate_id] in BINARY_LABELS
        for candidate_id, truth in full_truth.items()
    )
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "interpretation": (
            "Review-pending rows are uncertainty targets, not negative binary ground truth."
        ),
    }


def _full_selective_metrics(
    full_truth: Mapping[int, str], predictions: Mapping[int, str]
) -> dict[str, Any]:
    ids = sorted(full_truth)
    covered = [candidate_id for candidate_id in ids if predictions[candidate_id] in BINARY_LABELS]
    binary_truth = {
        candidate_id: truth
        for candidate_id, truth in full_truth.items()
        if truth in BINARY_LABELS
    }
    binary_predictions = {
        candidate_id: predictions[candidate_id] for candidate_id in binary_truth
    }
    return {
        "n": len(ids),
        "covered": len(covered),
        "abstained": len(ids) - len(covered),
        "coverage": len(covered) / len(ids) if ids else 0.0,
        "abstention_rate": (len(ids) - len(covered)) / len(ids) if ids else 0.0,
        "reference_label_counts": dict(Counter(full_truth.values())),
        "binary_reference_evaluation": _method_metrics(
            binary_truth, binary_predictions
        ),
        "unresolved_detection": _unresolved_detection(full_truth, predictions),
        "binary_scoring_policy": (
            "UNRESOLVED rows are excluded from binary confusion counts and are evaluated "
            "only as review/abstention targets."
        ),
    }


def evaluate_selective_protocol(
    *,
    consensus_path: Path,
    full_reference_path: Path,
    prediction_paths: Mapping[str, Path],
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20260823,
    forced_reject_from: str | None = "hybrid",
    protocol_version: str = PROTOCOL_VERSION,
) -> dict[str, Any]:
    protocol_version = str(protocol_version).strip()
    if not protocol_version:
        raise ValueError("protocol_version must be non-empty")
    consensus_rows = read_jsonl_index(consensus_path)
    full_rows = read_jsonl_index(full_reference_path)
    consensus_truth = {
        candidate_id: normalize_label(row.get("gold_label"))
        for candidate_id, row in consensus_rows.items()
    }
    if any(label not in BINARY_LABELS for label in consensus_truth.values()):
        raise ValueError("consensus file contains a non-binary decision")
    full_truth = {
        candidate_id: normalize_label(row.get("gold_label"))
        for candidate_id, row in full_rows.items()
    }
    if not set(consensus_rows).issubset(full_rows):
        raise ValueError("consensus IDs are not a subset of full reference IDs")
    predictions = {
        method: _load_predictions(path, set(full_rows))
        for method, path in prediction_paths.items()
    }
    derived_methods: dict[str, str] = {}
    if forced_reject_from:
        if forced_reject_from not in predictions:
            raise ValueError(f"forced-reject source method not found: {forced_reject_from}")
        name = f"{forced_reject_from}_forced_reject"
        predictions[name] = {
            candidate_id: ("REJECT" if label == "UNRESOLVED" else label)
            for candidate_id, label in predictions[forced_reject_from].items()
        }
        derived_methods[name] = (
            f"derived from {forced_reject_from}; UNRESOLVED predictions were coerced to REJECT"
        )

    methods: dict[str, Any] = {}
    for method, prediction_by_id in predictions.items():
        consensus_predictions = {
            candidate_id: prediction_by_id[candidate_id] for candidate_id in consensus_truth
        }
        consensus_metrics = _method_metrics(consensus_truth, consensus_predictions)
        consensus_metrics["bootstrap_95_ci"] = stratified_bootstrap_ci(
            consensus_truth,
            consensus_predictions,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        )
        full_metrics = _full_selective_metrics(full_truth, prediction_by_id)
        methods[method] = {
            "consensus_evaluation": consensus_metrics,
            "full_set_selective_evaluation": full_metrics,
        }

    method_names = sorted(predictions)
    comparisons: list[dict[str, Any]] = []
    for index, method_a in enumerate(method_names):
        for method_b in method_names[index + 1 :]:
            comparisons.append(
                {
                    "method_a": method_a,
                    "method_b": method_b,
                    **mcnemar_exact(
                        consensus_truth,
                        {candidate_id: predictions[method_a][candidate_id] for candidate_id in consensus_truth},
                        {candidate_id: predictions[method_b][candidate_id] for candidate_id in consensus_truth},
                    ),
                }
            )
    return {
        "protocol_version": protocol_version,
        "consensus_n": len(consensus_truth),
        "full_reference_n": len(full_truth),
        "full_reference_unresolved_n": sum(
            label == "UNRESOLVED" for label in full_truth.values()
        ),
        "labels": sorted(CANONICAL_LABELS),
        "methods": methods,
        "derived_methods": derived_methods,
        "mcnemar_common_coverage": comparisons,
        "reporting_policy": {
            "consensus": "Precision, recall, F1, MCC, risk, coverage, and 95% CI on binary consensus only.",
            "full_set": "Coverage, abstention, review-target detection, and unresolved count on all rows.",
            "uncertain": "UNRESOLVED is never scored as REJECT.",
        },
    }


def evaluate_graph_semantics(
    semantic_reference_path: Path, prediction_path: Path
) -> dict[str, Any]:
    """Evaluate flat semantic graph exports without conflating span integrity and meaning."""

    references = read_jsonl_index(semantic_reference_path)
    predictions = read_jsonl_index(prediction_path)
    if set(references) != set(predictions):
        raise ValueError("semantic prediction IDs must exactly match semantic reference IDs")
    reviewed_ids = [
        candidate_id
        for candidate_id, row in references.items()
        if row.get("semantic_review_status")
        in {"dual_human_fields_agree", "human_semantic_review_complete"}
    ]
    if not reviewed_ids:
        return {
            "status": "pending_semantic_review",
            "total_tasks": len(references),
            "evaluated": 0,
        }
    entity_tp = entity_fp = entity_fn = 0
    agent_match = target_match = legal_match = 0
    evidence_sufficient_values: list[bool] = []
    unsupported = relation_total = resolver_abstained = 0
    for candidate_id in reviewed_ids:
        row = references[candidate_id]
        reference = row.get("semantic_reference") or row
        predicted = predictions[candidate_id]
        gold_entities = set(reference.get("vehicle_entities") or [])
        if not gold_entities:
            gold_entities = {
                value
                for value in (
                    reference.get("collision_agent"),
                    reference.get("collision_target"),
                )
                if value
            }
        predicted_entities = set(predicted.get("vehicle_entities") or [])
        if not predicted_entities:
            predicted_entities = {
                value
                for value in (
                    predicted.get("collision_agent"),
                    predicted.get("collision_target"),
                )
                if value
            }
        entity_tp += len(gold_entities & predicted_entities)
        entity_fp += len(predicted_entities - gold_entities)
        entity_fn += len(gold_entities - predicted_entities)
        agent_match += predicted.get("collision_agent") == reference.get("collision_agent")
        target_match += predicted.get("collision_target") == reference.get("collision_target")
        legal_match += predicted.get("legal_status") == reference.get("legal_status")
        if reference.get("evidence_span_sufficient") is not None:
            evidence_sufficient_values.append(
                bool(predicted.get("evidence_span_sufficient"))
                == bool(reference.get("evidence_span_sufficient"))
            )
        unsupported += int(predicted.get("unsupported_relation_count") or 0)
        relation_total += int(predicted.get("relation_count") or 0)
        resolver_abstained += str(predicted.get("resolver_status") or "").lower() in {
            "abstain",
            "abstained",
            "unresolved",
        }
    entity_precision = entity_tp / (entity_tp + entity_fp) if entity_tp + entity_fp else 0.0
    entity_recall = entity_tp / (entity_tp + entity_fn) if entity_tp + entity_fn else 0.0
    entity_f1 = (
        2 * entity_precision * entity_recall / (entity_precision + entity_recall)
        if entity_precision + entity_recall
        else 0.0
    )
    return {
        "status": "complete" if len(reviewed_ids) == len(references) else "partial",
        "total_tasks": len(references),
        "evaluated": len(reviewed_ids),
        "vehicle_entity": {
            "precision": entity_precision,
            "recall": entity_recall,
            "f1": entity_f1,
            "tp": entity_tp,
            "fp": entity_fp,
            "fn": entity_fn,
        },
        "collision_agent_exact_match": agent_match / len(reviewed_ids),
        "collision_target_exact_match": target_match / len(reviewed_ids),
        "legal_status_accuracy": legal_match / len(reviewed_ids),
        "evidence_span_semantic_sufficiency_accuracy": (
            sum(evidence_sufficient_values) / len(evidence_sufficient_values)
            if evidence_sufficient_values
            else None
        ),
        "evidence_span_semantic_sufficiency_n": len(evidence_sufficient_values),
        "unsupported_relation_rate": unsupported / relation_total if relation_total else None,
        "unsupported_relations": unsupported,
        "relation_total": relation_total,
        "resolver_abstention_rate": resolver_abstained / len(reviewed_ids),
        "interpretation": (
            "Exact source offsets are provenance-integrity checks and are not counted as "
            "semantic correctness."
        ),
    }
