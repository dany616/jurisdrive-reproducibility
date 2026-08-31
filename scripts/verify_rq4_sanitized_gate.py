#!/usr/bin/env python3
"""Read-only verification of the sanitized RQ4 first-gate package."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jurisdrive.rq4_sanitized import (  # noqa: E402
    METHOD_CODES,
    forbidden_path_hits,
    recursive_forbidden_hits,
    sha256_file,
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    experiment_root = args.root.resolve()
    model_visible = experiment_root / "task_a" / "model_visible"
    manifest = read_json(experiment_root / "sanitized_materialization_manifest.json")
    leakage = read_json(experiment_root / "leakage_audit.json")
    budget = read_json(experiment_root / "protocol" / "information_budget.json")
    request_index = read_jsonl(model_visible / "request_index.jsonl")
    private_rows = read_jsonl(experiment_root / "private" / "opaque_id_mapping.jsonl")
    samples = read_jsonl(experiment_root / "private" / "rendered_sample_index.jsonl")
    file_hashes = read_jsonl(experiment_root / "audit" / "model_visible_file_hashes.jsonl")

    require(manifest["status"] == "FIRST_GATE_PASS_NO_MODEL_CALLS", "unexpected gate status")
    require(manifest["acceptance_gate"]["model_calls"] == 0, "model call count is not zero")
    require(len(private_rows) == 168, "private identity mapping must contain 168 rows")
    require(len({row["opaque_artifact_id"] for row in private_rows}) == 168, "opaque IDs are not unique")
    require(len({row["judgment_slot"] for row in private_rows}) == 24, "judgment denominator drift")
    require(sum(row["trial_kind"] == "clean_control" for row in private_rows) == 24, "control denominator drift")
    require(sum(row["trial_kind"] != "clean_control" for row in private_rows) == 144, "injected denominator drift")

    require(len(request_index) == 1344, "request-template denominator drift")
    by_artifact: dict[str, set[str]] = defaultdict(set)
    code_counts = Counter()
    recursive_hits: list[dict[str, str]] = []
    for row in request_index:
        request_path = model_visible / row["request_path"]
        require(request_path.is_file(), f"missing request: {request_path}")
        require(sha256_file(request_path) == row["request_sha256"], f"request hash mismatch: {request_path}")
        by_artifact[row["opaque_artifact_id"]].add(row["method_code"])
        code_counts[row["method_code"]] += 1
        recursive_hits.extend(recursive_forbidden_hits(read_json(request_path), origin=row["request_path"]))
    require(len(by_artifact) == 168, "request artifact denominator drift")
    require(all(codes == set(METHOD_CODES.values()) for codes in by_artifact.values()), "an artifact lacks one or more method budgets")
    require(all(code_counts[code] == 168 for code in METHOD_CODES.values()), "method request counts are not all 168")

    visible_paths = [str(path.relative_to(model_visible)).replace("\\", "/") for path in model_visible.rglob("*")]
    path_hits = forbidden_path_hits(visible_paths)
    recursive_hits.extend(recursive_forbidden_hits(request_index, origin="request_index.jsonl"))
    require(not path_hits, f"revealing model-visible paths found: {path_hits[:3]}")
    require(not recursive_hits, f"forbidden Task-A tokens/fields found: {recursive_hits[:3]}")

    require(len(samples) == 48, "rendered sample denominator drift")
    require(len({(row["method"], row["fault_type"]) for row in samples}) == 48, "sample method/family coverage drift")
    for row in samples:
        path = experiment_root / row["sample_path"]
        require(path.is_file(), f"missing rendered sample: {path}")
        require(sha256_file(path) == row["sample_sha256"], f"sample hash mismatch: {path}")
        require(not recursive_forbidden_hits(read_json(path), origin=row["sample_path"]), f"rendered sample leakage: {path}")

    for row in file_hashes:
        path = model_visible / row["path"]
        require(path.is_file(), f"missing hashed model-visible file: {path}")
        require(sha256_file(path) == row["sha256"], f"model-visible file hash mismatch: {path}")

    require(leakage["status"] == "PASS" and leakage["forbidden_hits"] == 0, "stored leakage gate is not zero-hit PASS")
    require(all(budget["invariants"].values()), "information-budget invariant failed")
    require(read_json(experiment_root / "task_b" / "task_manifest.json")["contributes_to_task_a_detection_metrics"] is False, "Task B mixed into Task A")
    require(read_json(experiment_root / "task_c" / "task_manifest.json")["contributes_to_task_a_detection_metrics"] is False, "Task C mixed into Task A")

    result = {
        "status": "PASS",
        "judgments": 24,
        "artifacts": 168,
        "request_templates_verified": len(request_index),
        "rendered_samples_verified": len(samples),
        "model_visible_file_hashes_verified": len(file_hashes),
        "forbidden_hits": 0,
        "path_hits": 0,
        "model_calls": 0,
        "task_a_b_c_separated": True,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

