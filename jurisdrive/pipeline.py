from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .contract import compile_contract
from .evidence import OpenAICompatibleResolver, build_evidence_graph, validate_evidence_spans
from .io import DEFAULT_FULL_RUN, iter_jsonl, load_candidate, read_json, write_json
from .models import EvidenceGraphV1, ScenarioContractV1
from .simulator import DryRunBackend, write_bundle


def select_manifest_rows(
    manifest_path: Path,
    *,
    limit: int | None = None,
    tier_counts: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    rows = list(iter_jsonl(manifest_path))
    if tier_counts:
        aliases = {
            "A": "A_minimum_grounded",
            "B": "B_defaults_needed",
            "C": "C_reextract_or_review",
        }
        selected: list[dict[str, Any]] = []
        for short_name, count in tier_counts.items():
            tier = aliases.get(short_name, short_name)
            matches = [row for row in rows if row.get("readiness_tier") == tier]
            if len(matches) < count:
                raise ValueError(f"{tier}: requested {count}, found {len(matches)}")
            selected.extend(matches[:count])
        return selected
    return rows[:limit] if limit is not None else rows


def build_graph_batch(
    rows: Iterable[dict[str, Any]],
    output_dir: Path,
    *,
    full_run_dir: Path = DEFAULT_FULL_RUN,
    resolver_endpoint: str | None = None,
    resolver_model: str | None = None,
) -> dict[str, Any]:
    if resolver_endpoint and not resolver_model:
        raise ValueError(
            "resolver_model is required with resolver_endpoint; use the exact "
            "model ID returned by the server's /v1/models endpoint"
        )
    resolver = (
        OpenAICompatibleResolver(resolver_endpoint, resolver_model)
        if resolver_endpoint
        else None
    )
    counts: Counter[str] = Counter()
    crashes: list[dict[str, Any]] = []
    for row in rows:
        counts["attempted"] += 1
        try:
            record = load_candidate(row, full_run_dir=full_run_dir)
            graph = build_evidence_graph(record, resolver=resolver)
            source_text = str(record.get("source_text") or "")
            span_errors = validate_evidence_spans(graph, source_text)
            if span_errors:
                raise ValueError("; ".join(span_errors))
            write_json(output_dir / f"{graph.scenario_id}.json", graph)
            counts["schema_valid"] += 1
            if graph.critical_unresolved:
                counts["critical_unresolved"] += 1
            else:
                counts["critical_resolved"] += 1
        except Exception as exc:
            crashes.append({"candidate_id": row.get("candidate_id"), "error": str(exc)})
    summary = {
        **counts,
        "crashes": len(crashes),
        "crash_details": crashes,
        "resolver": resolver.name if resolver else "none",
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def build_contract_batch(
    rows: Iterable[dict[str, Any]],
    graph_dir: Path,
    output_dir: Path,
    *,
    full_run_dir: Path = DEFAULT_FULL_RUN,
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    crashes: list[dict[str, Any]] = []
    for row in rows:
        counts["attempted"] += 1
        candidate_id = int(row["candidate_id"])
        try:
            record = load_candidate(row, full_run_dir=full_run_dir)
            graph = EvidenceGraphV1.model_validate(
                read_json(graph_dir / f"jurisdrive_{candidate_id}.json")
            )
            contract = compile_contract(
                graph,
                source_text=str(record.get("source_text") or ""),
                readiness_tier=str(row["readiness_tier"]),
            )
            write_json(output_dir / f"{contract.scenario_id}.json", contract)
            counts["schema_valid"] += 1
            counts[contract.status.value] += 1
        except Exception as exc:
            crashes.append({"candidate_id": candidate_id, "error": str(exc)})
    summary = {**counts, "crashes": len(crashes), "crash_details": crashes}
    write_json(output_dir / "summary.json", summary)
    return summary


def compile_dry_run_batch(
    rows: Iterable[dict[str, Any]],
    graph_dir: Path,
    contract_dir: Path,
    bundle_dir: Path,
) -> dict[str, Any]:
    backend = DryRunBackend()
    counts: Counter[str] = Counter()
    crashes: list[dict[str, Any]] = []
    for row in rows:
        counts["attempted"] += 1
        candidate_id = int(row["candidate_id"])
        try:
            graph = EvidenceGraphV1.model_validate(
                read_json(graph_dir / f"jurisdrive_{candidate_id}.json")
            )
            contract = ScenarioContractV1.model_validate(
                read_json(contract_dir / f"jurisdrive_{candidate_id}.json")
            )
            compiled = backend.compile(contract)
            result = backend.run(compiled)
            write_bundle(bundle_dir, graph, contract, compiled, result)
            counts["bundled"] += 1
            counts["compile_valid" if compiled["compile_valid"] else "compile_invalid"] += 1
            if result.executed:
                counts["unexpected_executed"] += 1
            if (
                result.actor_states is not None
                or result.collisions is not None
                or result.minimum_ttc_seconds is not None
                or result.keyframes is not None
            ):
                counts["unexpected_simulation_metrics"] += 1
        except Exception as exc:
            crashes.append({"candidate_id": candidate_id, "error": str(exc)})
    summary = {**counts, "crashes": len(crashes), "crash_details": crashes}
    write_json(bundle_dir / "summary.json", summary)
    return summary
