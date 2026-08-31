#!/usr/bin/env python3
"""Compile the 24 suggested RQ3 cases without claiming human preregistration."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jurisdrive.contract import bind_topology_profile, compile_contract  # noqa: E402
from jurisdrive.evidence import (  # noqa: E402
    OpenAICompatibleResolver,
    build_evidence_graph,
    validate_evidence_spans,
)
from jurisdrive.experiments import write_experiment_plan  # noqa: E402
from jurisdrive.io import iter_jsonl, read_json, sha256_file, write_json, write_jsonl  # noqa: E402
from jurisdrive.simulator import DryRunBackend, write_bundle  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suggestions", type=Path, required=True)
    parser.add_argument("--semantic-tasks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--preregistration-template",
        type=Path,
        default=PROJECT_ROOT / "configs" / "carla_assurance_24_preregistration.json",
    )
    parser.add_argument("--resolver-endpoint")
    parser.add_argument("--resolver-model", default="qwen35-vlm")
    args = parser.parse_args()

    suggestions_path = args.suggestions.resolve()
    tasks_path = args.semantic_tasks.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite RQ3 preparation: {output_dir}")
    graph_dir = output_dir / "graphs"
    contract_dir = output_dir / "contracts"
    bundle_dir = output_dir / "dry_run_bundles"
    graph_dir.mkdir(parents=True)
    contract_dir.mkdir()

    suggestions = read_json(suggestions_path)
    cases = suggestions.get("cases") or []
    if len(cases) != 24:
        raise ValueError("suggestion file must contain exactly 24 cases")
    tasks = {int(row["candidate_id"]): row for row in iter_jsonl(tasks_path)}
    resolver = (
        OpenAICompatibleResolver(args.resolver_endpoint, args.resolver_model)
        if args.resolver_endpoint
        else None
    )
    backend = DryRunBackend()
    review_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    for case in cases:
        candidate_id = int(case["candidate_id"])
        if candidate_id not in tasks:
            raise ValueError(f"candidate {candidate_id} is absent from the frozen semantic tasks")
        task = tasks[candidate_id]
        source_text = str(task.get("source_text") or "")
        record = {
            "source_text": source_text,
            "input_file": Path(str(task.get("source_path") or f"candidate_{candidate_id}.json")).name,
            "_manifest": {"candidate_id": candidate_id},
        }
        graph = build_evidence_graph(record, resolver=resolver)
        span_errors = validate_evidence_spans(graph, source_text)
        if span_errors:
            raise ValueError(f"candidate {candidate_id}: " + "; ".join(span_errors))
        write_json(graph_dir / f"{graph.scenario_id}.json", graph)
        readiness_tier = (
            "B_defaults_needed" if not graph.critical_unresolved else "C_reextract_or_review"
        )
        contract = compile_contract(
            graph,
            source_text=source_text,
            readiness_tier=readiness_tier,
        )
        collision_evidence_ids = (
            list(contract.collision_constraints[0].evidence_ids)
            if contract.collision_constraints
            else []
        )
        contract = bind_topology_profile(
            contract,
            str(case["topology"]),
            evidence_ids=collision_evidence_ids,
        )
        write_json(contract_dir / f"{contract.scenario_id}.json", contract)
        compiled = backend.compile(contract)
        dry_result = backend.run(compiled)
        write_bundle(bundle_dir, graph, contract, compiled, dry_result)

        compile_errors = backend.validate(compiled)
        counts["graphs_schema_valid"] += 1
        counts[f"contract_{contract.status.value}"] += 1
        counts["compile_valid" if not compile_errors else "compile_blocked"] += 1
        counts["critical_resolved" if not graph.critical_unresolved else "critical_unresolved"] += 1
        review_rows.append(
            {
                "slot_id": case["slot_id"],
                "candidate_id": candidate_id,
                "scenario_id": graph.scenario_id,
                "topology": case["topology"],
                "source_stage": case["source_stage"],
                "evidence_quote": case["evidence_quote"],
                "source_sha256": graph.source_text_sha256,
                "graph_path": str((graph_dir / f"{graph.scenario_id}.json").resolve()),
                "contract_path": str((contract_dir / f"{contract.scenario_id}.json").resolve()),
                "dry_run_bundle_path": str((bundle_dir / contract.scenario_id).resolve()),
                "critical_unresolved": list(graph.critical_unresolved),
                "review_required": list(graph.review_required),
                "contract_status": contract.status.value,
                "contract_review_issues": list(contract.review_issues),
                "compile_valid": not compile_errors,
                "compile_errors": compile_errors,
                "human_topology_confirmed": False,
                "publication_selection_frozen": False,
                "execution_scope": "exploratory_only_until_author_topology_confirmation",
            }
        )

    review_path = output_dir / "candidate_contract_review.jsonl"
    write_jsonl(review_path, review_rows)
    preregistration = read_json(args.preregistration_template.resolve())
    review_by_slot = {row["slot_id"]: row for row in review_rows}
    for slot in preregistration["cases"]:
        prepared = review_by_slot[slot["slot_id"]]
        slot.update(
            {
                "candidate_id": prepared["candidate_id"],
                "scenario_id": prepared["scenario_id"],
                "human_topology_confirmed": False,
                "contract_path": prepared["contract_path"],
                "clean_bundle_path": None,
            }
        )
    preregistration["selection_frozen"] = False
    preregistration["frozen_at_utc"] = None
    preregistration.setdefault("notes", []).append(
        "Candidate IDs and contract paths are populated for author review only; topology confirmation and publication freeze remain pending."
    )
    preregistration_path = output_dir / "preregistration_candidate_draft.json"
    write_json(preregistration_path, preregistration)
    plan_manifest = write_experiment_plan(
        preregistration_path,
        output_dir / "pending_experiment_plan",
        allow_pending=True,
    )
    manifest = {
        "version": "1.0",
        "experiment_id": "rq3_candidate24_contract_preparation",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "resolver": resolver.name if resolver else "none",
        "inputs": {
            "suggestions": {"path": str(suggestions_path), "sha256": sha256_file(suggestions_path)},
            "semantic_tasks": {"path": str(tasks_path), "sha256": sha256_file(tasks_path)},
        },
        "counts": dict(counts),
        "review_rows": {"path": str(review_path), "sha256": sha256_file(review_path)},
        "preregistration_candidate_draft": {
            "path": str(preregistration_path),
            "sha256": sha256_file(preregistration_path),
        },
        "pending_plan": plan_manifest,
        "execution_authorized_for_publication_protocol": False,
        "claim_boundaries": [
            "Candidate topology labels remain AI suggestions until author confirmation.",
            "Dry-run bundles contain no CARLA telemetry or simulated outcomes.",
            "Blocked contracts are retained in the failure taxonomy and are not replaced.",
        ],
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
