#!/usr/bin/env python3
"""Produce denominator-safe RQ2 semantic and selective-safety results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jurisdrive.gold_consensus import evaluate_graph_semantics  # noqa: E402
from jurisdrive.experiments import read_jsonl  # noqa: E402
from jurisdrive.io import sha256_file, write_json  # noqa: E402


def _rate(n: int, d: int) -> float | None:
    return n / d if d else None


def _critical_resolved(row: dict[str, Any]) -> bool:
    return bool(
        row.get("resolver_status") == "resolved"
        and row.get("collision_agent")
        and row.get("collision_target")
        and len(row.get("vehicle_entities") or []) >= 2
        and not row.get("critical_unresolved")
    )


def _quote_alignment(predicted_quotes: list[str], human_quotes: list[str]) -> bool:
    """Return whether a collision span contains, or is contained by, an agreed quote.

    Both quote sets are exact substrings of the same frozen source text.  This is
    intentionally an exact-containment proxy; it is not a substitute for a
    separate human rating of arbitrary evidence-span sufficiency.
    """

    return any(
        predicted and human and (predicted in human or human in predicted)
        for predicted in predicted_quotes
        for human in human_quotes
    )


def _collision_evidence_quotes(graph: dict[str, Any]) -> list[str]:
    evidence_by_id = {
        str(item["id"]): str(item.get("quote") or "")
        for item in graph.get("evidence") or []
    }
    return [
        evidence_by_id[evidence_id]
        for edge in graph.get("edges") or []
        if edge.get("relation") == "collides_with" and bool(edge.get("supported"))
        for evidence_id in edge.get("evidence_ids") or []
        if evidence_id in evidence_by_id and evidence_by_id[evidence_id]
    ]


def _tree_sha256(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-reference", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--protocol-version",
        default="jurisdrive-dual-human-consensus-v1",
    )
    parser.add_argument("--protocol-statement", type=Path)
    args = parser.parse_args()
    protocol_version = str(args.protocol_version).strip()
    if not protocol_version:
        raise ValueError("protocol_version must be non-empty")
    protocol_statement = (
        args.protocol_statement.resolve() if args.protocol_statement is not None else None
    )
    if protocol_statement is not None and not protocol_statement.is_file():
        raise FileNotFoundError(f"protocol statement not found: {protocol_statement}")
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite RQ2 publication summary: {output_dir}")
    output_dir.mkdir(parents=True)
    references = [dict(row) for row in read_jsonl(args.semantic_reference.resolve())]
    predictions = {int(row["candidate_id"]): dict(row) for row in read_jsonl(args.predictions.resolve())}
    graph_dir = args.graph_dir.resolve()
    if len(references) != 381 or len(predictions) != 381:
        raise ValueError(f"expected 381 tasks and predictions; got {len(references)}, {len(predictions)}")
    accepted = [row for row in references if row["reference_label"] == "ACCEPT"]
    rejected = [row for row in references if row["reference_label"] == "REJECT"]
    unresolved = [row for row in references if row["reference_label"] == "UNRESOLVED"]
    if (len(accepted), len(rejected), len(unresolved)) != (281, 74, 26):
        raise ValueError("RQ2 frozen strata changed")
    accepted_metrics = evaluate_graph_semantics(
        args.semantic_reference.resolve(), args.predictions.resolve()
    )
    graph_paths = sorted(graph_dir.glob("jurisdrive_*.json"))
    if len(graph_paths) != 381:
        raise ValueError(f"expected 381 graph files; got {len(graph_paths)}")
    graph_by_id: dict[int, dict[str, Any]] = {}
    for path in graph_paths:
        graph = json.loads(path.read_text(encoding="utf-8"))
        candidate_id = int(graph["candidate_id"])
        if candidate_id in graph_by_id:
            raise ValueError(f"duplicate graph candidate ID: {candidate_id}")
        graph_by_id[candidate_id] = graph
    if set(graph_by_id) != set(predictions):
        raise ValueError("graph IDs must exactly match prediction IDs")

    evidence_expected = [
        row for row in accepted if (row.get("semantic_reference") or {}).get("evidence_quotes")
    ]
    collision_evidence_present = 0
    evidence_quote_aligned = 0
    for row in evidence_expected:
        predicted_quotes = _collision_evidence_quotes(graph_by_id[int(row["candidate_id"])])
        collision_evidence_present += bool(predicted_quotes)
        human_quotes = [
            str(quote)
            for quote in (row.get("semantic_reference") or {}).get("evidence_quotes") or []
        ]
        evidence_quote_aligned += _quote_alignment(predicted_quotes, human_quotes)
    reject_leakage_rows = [
        row for row in rejected if _critical_resolved(predictions[int(row["candidate_id"])])
    ]
    unresolved_abstained = [
        row
        for row in unresolved
        if not _critical_resolved(predictions[int(row["candidate_id"])])
    ]
    accepted_resolved = [
        row for row in accepted if _critical_resolved(predictions[int(row["candidate_id"])])
    ]
    selective_safe = len(accepted_resolved) + (len(rejected) - len(reject_leakage_rows)) + len(unresolved_abstained)
    summary = {
        "version": "1.0",
        "protocol_version": protocol_version,
        "experiment_id": "rq2_n4_graph_semantics_381_denominator_safe",
        "total": 381,
        "strata": {"accept": 281, "reject": 74, "unresolved": 26},
        "accept_semantics": {
            **accepted_metrics,
            "critical_resolved": len(accepted_resolved),
            "critical_resolved_rate": _rate(len(accepted_resolved), len(accepted)),
            "collision_evidence_grounding_coverage": {
                "n": len(evidence_expected),
                "present": collision_evidence_present,
                "rate": _rate(collision_evidence_present, len(evidence_expected)),
            },
            "dual_human_evidence_quote_alignment": {
                "n": len(evidence_expected),
                "aligned": evidence_quote_aligned,
                "rate": _rate(evidence_quote_aligned, len(evidence_expected)),
                "conditional_on_collision_evidence": _rate(
                    evidence_quote_aligned, collision_evidence_present
                ),
                "operational_definition": (
                    "At least one supported compiled collision-evidence quote and one "
                    "dual-human-agreed exact quote have an exact containment relation."
                ),
            },
            "evidence_span_semantic_sufficiency": {
                "status": "proxy_only_no_separate_human_rating",
                "direct_human_ratings_n": 0,
                "proxy": "dual_human_evidence_quote_alignment",
            },
        },
        "reject_selective_safety": {
            "n": len(rejected),
            "executable_graph_leakage": len(reject_leakage_rows),
            "leakage_rate": _rate(len(reject_leakage_rows), len(rejected)),
            "safe_rejection_or_abstention": len(rejected) - len(reject_leakage_rows),
            "specificity": _rate(len(rejected) - len(reject_leakage_rows), len(rejected)),
            "leaked_candidate_ids": [int(row["candidate_id"]) for row in reject_leakage_rows],
        },
        "unresolved_selective_safety": {
            "n": len(unresolved),
            "abstained": len(unresolved_abstained),
            "abstention_rate": _rate(len(unresolved_abstained), len(unresolved)),
            "advanced_without_resolution": len(unresolved) - len(unresolved_abstained),
            "advanced_candidate_ids": [
                int(row["candidate_id"])
                for row in unresolved
                if row not in unresolved_abstained
            ],
        },
        "overall_selective_semantic_safety": {
            "safe": selective_safe,
            "n": len(references),
            "rate": _rate(selective_safe, len(references)),
            "definition": "ACCEPT has critical graph; REJECT does not leak an executable graph; UNRESOLVED abstains",
        },
        "claim_boundary": (
            "Agent/target/entity/legal semantic accuracy is scored only on 281 dual-human ACCEPT references. "
            "The 74 REJECT and 26 UNRESOLVED cases are evaluated as leakage/abstention, not assigned fabricated semantic fields."
        ),
    }
    summary_path = output_dir / "rq2_summary.json"
    write_json(summary_path, summary)
    table_path = output_dir / "rq2_summary.csv"
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Measure", "n", "Result"])
        writer.writerow(["Vehicle entity precision (ACCEPT)", 281, accepted_metrics["vehicle_entity"]["precision"]])
        writer.writerow(["Vehicle entity recall (ACCEPT)", 281, accepted_metrics["vehicle_entity"]["recall"]])
        writer.writerow(["Vehicle entity F1 (ACCEPT)", 281, accepted_metrics["vehicle_entity"]["f1"]])
        writer.writerow(["Collision agent exact match (ACCEPT)", 281, accepted_metrics["collision_agent_exact_match"]])
        writer.writerow(["Collision target exact match (ACCEPT)", 281, accepted_metrics["collision_target_exact_match"]])
        writer.writerow(["Legal status accuracy (ACCEPT)", 281, accepted_metrics["legal_status_accuracy"]])
        writer.writerow(["Collision evidence grounding coverage (ACCEPT)", len(evidence_expected), _rate(collision_evidence_present, len(evidence_expected))])
        writer.writerow(["Dual-human evidence quote alignment (ACCEPT)", len(evidence_expected), _rate(evidence_quote_aligned, len(evidence_expected))])
        writer.writerow(["Direct human evidence-sufficiency ratings", 0, "not_scored"])
        writer.writerow(["Unsupported relation rate (ACCEPT)", accepted_metrics["relation_total"], accepted_metrics["unsupported_relation_rate"]])
        writer.writerow(["Resolver abstention rate (ACCEPT)", 281, accepted_metrics["resolver_abstention_rate"]])
        writer.writerow(["REJECT executable graph leakage", 74, _rate(len(reject_leakage_rows), 74)])
        writer.writerow(["UNRESOLVED abstention", 26, _rate(len(unresolved_abstained), 26)])
        writer.writerow(["Overall selective semantic safety", 381, _rate(selective_safe, 381)])
    paper = [
        "# RQ2 Evidence-Graph Semantic Evaluation",
        "",
        f"On 281 dual-human ACCEPT references, vehicle entity precision/recall/F1 were {accepted_metrics['vehicle_entity']['precision']:.4f}/{accepted_metrics['vehicle_entity']['recall']:.4f}/{accepted_metrics['vehicle_entity']['f1']:.4f}; collision-agent and collision-target exact match were {accepted_metrics['collision_agent_exact_match']:.4f} and {accepted_metrics['collision_target_exact_match']:.4f}.",
        "",
        f"Dual-human-agreed exact evidence quotes were available for {len(evidence_expected)} ACCEPT cases. Supported collision evidence was present in {collision_evidence_present}/{len(evidence_expected)} graphs ({_rate(collision_evidence_present, len(evidence_expected)):.4f}), and {evidence_quote_aligned}/{len(evidence_expected)} ({_rate(evidence_quote_aligned, len(evidence_expected)):.4f}) had exact quote containment alignment. This alignment is an explicit proxy; no separate human sufficiency rating was collected. Legal-status accuracy was {accepted_metrics['legal_status_accuracy']:.4f}, exposing a substantive compiler failure rather than a positive result.",
        "",
        f"For the negative/selective strata, executable-graph leakage was {len(reject_leakage_rows)}/74 REJECT cases, while {len(unresolved_abstained)}/26 UNRESOLVED cases remained abstained. These cases were not assigned fabricated actor/target references.",
        "",
        "Exact offsets remain provenance-integrity evidence and are not counted as semantic correctness.",
    ]
    paper_path = output_dir / "rq2_paper_results.md"
    paper_path.write_text("\n".join(paper) + "\n", encoding="utf-8")
    manifest = {
        "version": "1.0",
        "protocol_version": protocol_version,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "inputs": {
            "semantic_reference": {"path": str(args.semantic_reference.resolve()), "sha256": sha256_file(args.semantic_reference.resolve())},
            "predictions": {"path": str(args.predictions.resolve()), "sha256": sha256_file(args.predictions.resolve())},
            "graphs": {
                "path": str(graph_dir),
                "files": len(graph_paths),
                "tree_sha256": _tree_sha256(graph_paths, graph_dir),
            },
        },
        "outputs": {
            "summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
            "table": {"path": str(table_path), "sha256": sha256_file(table_path)},
            "paper": {"path": str(paper_path), "sha256": sha256_file(paper_path)},
        },
        "human_claim": (
            "Only author-certified, source-only dual-human ACCEPT semantic fields "
            "from the bound protocol freeze are treated as semantic reference."
            if protocol_version != "jurisdrive-dual-human-consensus-v1"
            else "Only existing dual-human ACCEPT semantic fields are treated as semantic gold."
        ),
    }
    if protocol_statement is not None:
        manifest["protocol_statement"] = {
            "path": str(protocol_statement),
            "sha256": sha256_file(protocol_statement),
            "status": "author-approved-not-interaction-log-verified",
        }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps({"summary": summary, "manifest": manifest}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
