#!/usr/bin/env python3
"""Build N4 graphs for the frozen semantic tasks and export honest RQ2 predictions.

The semantic reference is used only to select task IDs and to supply the immutable
source text.  Gold semantic fields are never copied into the predictions.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jurisdrive.evidence import (  # noqa: E402
    OpenAICompatibleResolver,
    build_evidence_graph,
    validate_evidence_spans,
)
from jurisdrive.gold_consensus import evaluate_graph_semantics  # noqa: E402
from jurisdrive.io import (  # noqa: E402
    iter_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from jurisdrive.models import NodeType, RelationType  # noqa: E402


def _collision_semantics(graph: Any) -> tuple[str | None, str | None, str]:
    nodes = {node.id: node for node in graph.nodes}
    evidence = {item.id: item for item in graph.evidence}
    collisions = [
        edge
        for edge in graph.edges
        if edge.relation == RelationType.COLLIDES_WITH and edge.supported
    ]
    if not collisions:
        return None, None, "unknown"
    edge = collisions[0]
    statuses = [
        evidence[evidence_id].legal_status.value
        for evidence_id in edge.evidence_ids
        if evidence_id in evidence
    ]
    legal_status = statuses[0] if statuses else "unknown"
    return nodes[edge.source].label, nodes[edge.target].label, legal_status


def _flat_prediction(graph: Any) -> dict[str, Any]:
    agent, target, legal_status = _collision_semantics(graph)
    relation_edges = [
        edge for edge in graph.edges if edge.relation != RelationType.SUPPORTED_BY
    ]
    evidence = {item.id: item for item in graph.evidence}
    unsupported = 0
    for edge in relation_edges:
        if not edge.supported or not edge.evidence_ids:
            unsupported += 1
            continue
        if any(
            evidence_id not in evidence or not evidence[evidence_id].supported
            for evidence_id in edge.evidence_ids
        ):
            unsupported += 1
    collision_edges = [
        edge for edge in relation_edges if edge.relation == RelationType.COLLIDES_WITH
    ]
    evidence_span_sufficient = bool(collision_edges) and all(
        edge.evidence_ids
        and all(
            evidence_id in evidence and evidence[evidence_id].supported
            for evidence_id in edge.evidence_ids
        )
        for edge in collision_edges
    )
    return {
        "candidate_id": graph.candidate_id,
        "vehicle_entities": [
            node.label for node in graph.nodes if node.type == NodeType.VEHICLE
        ],
        "collision_agent": agent,
        "collision_target": target,
        "legal_status": legal_status,
        "evidence_span_sufficient": evidence_span_sufficient,
        "unsupported_relation_count": unsupported,
        "relation_count": len(relation_edges),
        "resolver_status": "unresolved" if graph.critical_unresolved else "resolved",
        "critical_unresolved": list(graph.critical_unresolved),
        "review_required": list(graph.review_required),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolver-endpoint")
    parser.add_argument("--resolver-model", default="qwen35-vlm")
    parser.add_argument(
        "--resolver-workers",
        type=int,
        default=1,
        help="Bounded parallel graph builds/resolver calls (1-8).",
    )
    args = parser.parse_args()
    if not 1 <= args.resolver_workers <= 8:
        parser.error("--resolver-workers must be between 1 and 8")

    reference = args.semantic_reference.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite RQ2 output: {output_dir}")
    graph_dir = output_dir / "graphs"
    graph_dir.mkdir(parents=True)
    resolver = (
        OpenAICompatibleResolver(args.resolver_endpoint, args.resolver_model)
        if args.resolver_endpoint
        else None
    )

    rows = list(iter_jsonl(reference))
    candidate_ids = [int(row["candidate_id"]) for row in rows]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("semantic reference contains duplicate candidate IDs")

    def process_row(row: dict[str, Any]) -> tuple[Any | None, dict[str, Any] | None, dict[str, Any] | None]:
        candidate_id = int(row["candidate_id"])
        source_text = str(row.get("source_text") or "")
        if not source_text:
            return None, None, {"candidate_id": candidate_id, "error": "source_text_missing"}
        # ``source_file_sha256`` freezes the original JSON file, not the decoded
        # ``source_text`` member.  The graph records its own source-text digest.
        source_name = Path(str(row.get("source_path") or f"candidate_{candidate_id}.json")).name
        record = {
            "source_text": source_text,
            "input_file": source_name,
            "_manifest": {"candidate_id": candidate_id},
        }
        try:
            graph = build_evidence_graph(record, resolver=resolver)
            span_errors = validate_evidence_spans(graph, source_text)
            if span_errors:
                raise ValueError("; ".join(span_errors))
            prediction = _flat_prediction(graph)
            return graph, prediction, None
        except Exception as exc:  # preserve a complete audit without inventing a row
            return None, None, {"candidate_id": candidate_id, "error": str(exc)}

    if args.resolver_workers == 1:
        processed = [process_row(row) for row in rows]
    else:
        with ThreadPoolExecutor(max_workers=args.resolver_workers) as executor:
            processed = list(executor.map(process_row, rows))

    predictions: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    errors: list[dict[str, Any]] = []
    for graph, prediction, error in processed:
        if error is not None:
            errors.append(error)
            continue
        assert graph is not None and prediction is not None
        write_json(graph_dir / f"{graph.scenario_id}.json", graph)
        predictions.append(prediction)
        counts["schema_valid"] += 1
        counts["critical_resolved" if not graph.critical_unresolved else "critical_unresolved"] += 1
        counts[f"legal_{prediction['legal_status']}"] += 1

    if errors:
        write_json(output_dir / "errors.json", errors)
        raise RuntimeError(f"RQ2 graph construction failed for {len(errors)} tasks")
    predictions.sort(key=lambda row: int(row["candidate_id"]))
    prediction_path = output_dir / "graph_semantic_predictions.jsonl"
    write_jsonl(prediction_path, predictions)
    metrics = evaluate_graph_semantics(reference, prediction_path)
    write_json(output_dir / "graph_semantic_metrics.json", metrics)
    manifest = {
        "version": "1.0",
        "experiment_id": "rq2_n4_graph_semantics_381",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "semantic_reference": {
            "path": str(reference),
            "sha256": sha256_file(reference),
            "rows": len(rows),
        },
        "resolver": resolver.name if resolver else "none",
        "resolver_endpoint": args.resolver_endpoint,
        "resolver_model": args.resolver_model if resolver else None,
        "resolver_workers": args.resolver_workers,
        "implementation": {
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "evidence_builder_sha256": sha256_file(PROJECT_ROOT / "jurisdrive" / "evidence.py"),
            "semantic_evaluator_sha256": sha256_file(
                PROJECT_ROOT / "jurisdrive" / "gold_consensus.py"
            ),
        },
        "counts": dict(counts),
        "evaluated_now": metrics.get("evaluated", 0),
        "pending_human_semantic_review": len(rows) - int(metrics.get("evaluated", 0)),
        "outputs": {
            "predictions": {"path": str(prediction_path), "sha256": sha256_file(prediction_path)},
            "metrics": {
                "path": str(output_dir / "graph_semantic_metrics.json"),
                "sha256": sha256_file(output_dir / "graph_semantic_metrics.json"),
            },
        },
        "claim_boundaries": [
            "Gold semantic fields were not used to populate predictions.",
            "Exact offsets are provenance-integrity checks, not semantic accuracy.",
            "Rows pending human semantic review are excluded from semantic metrics.",
        ],
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps({"manifest": manifest, "metrics": metrics}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
