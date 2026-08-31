from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .assurance import MockEvaluator, VlmEvaluator
from .gold import benchmark_gold, gold_status, sample_gold
from .io import DEFAULT_ARTIFACTS, DEFAULT_FULL_RUN, DEFAULT_MANIFEST, read_json, write_json
from .models import ScenarioContractV1, SimulationResultV1
from .pipeline import (
    build_contract_batch,
    build_graph_batch,
    compile_dry_run_batch,
    select_manifest_rows,
)
from .simulator import CarlaBackend


def parse_tier_counts(value: str | None) -> dict[str, int] | None:
    if not value:
        return None
    result: dict[str, int] = {}
    for item in value.split(","):
        tier, count = item.split("=", 1)
        result[tier.strip()] = int(count)
    return result


def add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--full-run-dir",
        type=Path,
        default=DEFAULT_FULL_RUN,
        help="N0-N3 full_run directory used to resolve portable manifest rows",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--tier-counts",
        default=None,
        help="Deterministic first-N strata, e.g. A=200,B=100,C=100",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m jurisdrive")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gold = subparsers.add_parser("sample-gold")
    gold.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACTS / "gold_kit")
    gold.add_argument("--full-run-dir", type=Path, default=None)
    gold.add_argument("--seed", type=int, default=20260728)

    graph = subparsers.add_parser("build-graph")
    add_selection_arguments(graph)
    graph.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACTS / "graphs")
    graph.add_argument("--resolver-endpoint", default=None)
    graph.add_argument(
        "--resolver-model",
        default=None,
        help="Exact served model ID; required when --resolver-endpoint is used",
    )

    contract = subparsers.add_parser("build-contract")
    add_selection_arguments(contract)
    contract.add_argument("--graph-dir", type=Path, default=DEFAULT_ARTIFACTS / "graphs")
    contract.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACTS / "contracts")

    compile_parser = subparsers.add_parser("compile")
    add_selection_arguments(compile_parser)
    compile_parser.add_argument("--backend", choices=("dry-run",), default="dry-run")
    compile_parser.add_argument("--graph-dir", type=Path, default=DEFAULT_ARTIFACTS / "graphs")
    compile_parser.add_argument(
        "--contract-dir", type=Path, default=DEFAULT_ARTIFACTS / "contracts"
    )
    compile_parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACTS / "bundles")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--backend", choices=("carla",), required=True)
    run_parser.add_argument("--bundle-dir", type=Path, required=True)
    run_parser.add_argument("--host", default="127.0.0.1")
    run_parser.add_argument("--port", type=int, default=2000)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--evaluator", choices=("mock", "vlm"), default="mock")
    evaluate.add_argument("--bundle-dir", type=Path, required=True)
    evaluate.add_argument("--result", type=Path, default=None)
    evaluate.add_argument("--endpoint", default=None)
    evaluate.add_argument("--model", default=None)
    evaluate.add_argument("--timeout", type=float, default=180.0)
    evaluate.add_argument("--keyframe", type=Path, action="append", default=[])

    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--gold-dir", type=Path, default=DEFAULT_ARTIFACTS / "gold_kit")
    benchmark.add_argument(
        "--output", type=Path, default=DEFAULT_ARTIFACTS / "gold_kit" / "metrics.json"
    )

    status = subparsers.add_parser("gold-status")
    status.add_argument("--gold-dir", type=Path, default=DEFAULT_ARTIFACTS / "gold_kit")
    status.add_argument("--output", type=Path, default=None)
    return parser


def _rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    return select_manifest_rows(
        args.manifest,
        limit=args.limit,
        tier_counts=parse_tier_counts(args.tier_counts),
    )


def command_run(args: argparse.Namespace) -> dict[str, Any]:
    contract = ScenarioContractV1.model_validate(read_json(args.bundle_dir / "contract.json"))
    backend = CarlaBackend(args.bundle_dir, host=args.host, port=args.port)
    compiled = backend.compile(contract)
    result = backend.run(compiled)
    output = args.bundle_dir / "simulation_result.json"
    write_json(output, result)
    return {"scenario_id": contract.scenario_id, "result": str(output), "executed": True}


def command_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    contract = ScenarioContractV1.model_validate(read_json(args.bundle_dir / "contract.json"))
    result_path = args.result or args.bundle_dir / "dry_run_report.json"
    result = SimulationResultV1.model_validate(read_json(result_path))
    if args.evaluator == "mock":
        evaluator = MockEvaluator()
    else:
        if not args.endpoint or not args.model:
            raise SystemExit("--endpoint and --model are required for --evaluator vlm")
        evaluator = VlmEvaluator(
            args.endpoint,
            args.model,
            timeout=args.timeout,
            bundle_dir=args.bundle_dir,
            keyframes=args.keyframe,
        )
    report = evaluator.evaluate(contract, result)
    output = args.bundle_dir / f"evaluation_{evaluator.name}.json"
    write_json(output, report)
    if isinstance(evaluator, VlmEvaluator):
        write_json(args.bundle_dir / "evaluation_vlm_request.json", evaluator.last_request)
        write_json(args.bundle_dir / "evaluation_vlm_raw_response.json", evaluator.last_response)
    return report.model_dump(mode="json")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "sample-gold":
        kwargs: dict[str, Any] = {"seed": args.seed}
        if args.full_run_dir is not None:
            kwargs["full_run_dir"] = args.full_run_dir
        result = sample_gold(args.output_dir, **kwargs)
    elif args.command == "build-graph":
        result = build_graph_batch(
            _rows(args),
            args.output_dir,
            full_run_dir=args.full_run_dir,
            resolver_endpoint=args.resolver_endpoint,
            resolver_model=args.resolver_model,
        )
    elif args.command == "build-contract":
        result = build_contract_batch(
            _rows(args),
            args.graph_dir,
            args.output_dir,
            full_run_dir=args.full_run_dir,
        )
    elif args.command == "compile":
        result = compile_dry_run_batch(
            _rows(args),
            args.graph_dir,
            args.contract_dir,
            args.output_dir,
        )
    elif args.command == "run":
        result = command_run(args)
    elif args.command == "evaluate":
        result = command_evaluate(args)
    elif args.command == "benchmark":
        result = benchmark_gold(args.gold_dir, args.output)
    elif args.command == "gold-status":
        result = gold_status(args.gold_dir)
        if args.output is not None:
            write_json(args.output, result)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
