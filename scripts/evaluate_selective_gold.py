#!/usr/bin/env python3
"""Evaluate JurisDrive's consensus and full-set selective gold protocols."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jurisdrive.gold_consensus import (  # noqa: E402
    PROTOCOL_VERSION,
    evaluate_selective_protocol,
    sha256_file,
    write_json,
)


def prediction_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("prediction must be METHOD=PATH")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("prediction must be METHOD=PATH")
    return name.strip(), Path(raw_path)


def write_metrics_csv(path: Path, payload: dict) -> None:
    rows: list[dict] = []
    for method, result in payload["methods"].items():
        consensus = result["consensus_evaluation"]
        binary = consensus.get("binary_metrics_on_covered") or {}
        confusion = binary.get("confusion") or {}
        full = result["full_set_selective_evaluation"]
        unresolved = full["unresolved_detection"]
        ci = consensus["bootstrap_95_ci"]["metrics"]
        rows.append(
            {
                "method": method,
                "consensus_n": consensus["n"],
                "consensus_covered": consensus["covered"],
                "coverage": consensus["coverage"],
                "abstention_rate": consensus["abstention_rate"],
                "selective_risk": consensus["selective_risk"],
                "precision": binary.get("precision"),
                "precision_ci_low": (ci.get("precision") or {}).get("low"),
                "precision_ci_high": (ci.get("precision") or {}).get("high"),
                "recall": binary.get("recall"),
                "recall_ci_low": (ci.get("recall") or {}).get("low"),
                "recall_ci_high": (ci.get("recall") or {}).get("high"),
                "f1": binary.get("f1"),
                "f1_ci_low": (ci.get("f1") or {}).get("low"),
                "f1_ci_high": (ci.get("f1") or {}).get("high"),
                "mcc": binary.get("mcc"),
                "mcc_ci_low": (ci.get("mcc") or {}).get("low"),
                "mcc_ci_high": (ci.get("mcc") or {}).get("high"),
                "tp": confusion.get("tp"),
                "tn": confusion.get("tn"),
                "fp": confusion.get("fp"),
                "fn": confusion.get("fn"),
                "full_n": full["n"],
                "full_coverage": full["coverage"],
                "full_abstention_rate": full["abstention_rate"],
                "unresolved_detection_precision": unresolved["precision"],
                "unresolved_detection_recall": unresolved["recall"],
                "unresolved_detection_f1": unresolved["f1"],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_mcnemar_csv(path: Path, payload: dict) -> None:
    rows = payload["mcnemar_common_coverage"]
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method_a",
        "method_b",
        "common_coverage_n",
        "a_correct_b_wrong",
        "a_wrong_b_correct",
        "discordant_n",
        "two_sided_exact_p",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consensus", type=Path, required=True)
    parser.add_argument("--full-reference", type=Path, required=True)
    parser.add_argument(
        "--prediction",
        type=prediction_argument,
        action="append",
        required=True,
        metavar="METHOD=PATH",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260823)
    parser.add_argument("--protocol-version", default=PROTOCOL_VERSION)
    parser.add_argument(
        "--forced-reject-from",
        default="hybrid",
        help="Derive a forced-binary baseline by mapping this method's UNRESOLVED to REJECT; use 'none' to disable.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    prediction_paths = {name: path.resolve() for name, path in args.prediction}
    if len(prediction_paths) != len(args.prediction):
        raise SystemExit("duplicate prediction method name")
    forced_reject_from = (
        None if args.forced_reject_from.lower() == "none" else args.forced_reject_from
    )
    payload = evaluate_selective_protocol(
        consensus_path=args.consensus.resolve(),
        full_reference_path=args.full_reference.resolve(),
        prediction_paths=prediction_paths,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        forced_reject_from=forced_reject_from,
        protocol_version=args.protocol_version,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "metrics": output_dir / "selective_metrics.json",
        "table": output_dir / "selective_metrics_table.csv",
        "mcnemar": output_dir / "mcnemar_common_coverage.csv",
    }
    manifest_path = output_dir / "selective_evaluation_manifest.json"
    for path in [*outputs.values(), manifest_path]:
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite existing evaluation artifact: {path}")
    write_json(outputs["metrics"], payload, overwrite=args.overwrite)
    write_metrics_csv(outputs["table"], payload)
    write_mcnemar_csv(outputs["mcnemar"], payload)
    manifest = {
        "protocol_version": payload["protocol_version"],
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "configuration": {
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
            "forced_reject_from": forced_reject_from,
        },
        "inputs": {
            "consensus": {"path": str(args.consensus.resolve()), "sha256": sha256_file(args.consensus.resolve())},
            "full_reference": {"path": str(args.full_reference.resolve()), "sha256": sha256_file(args.full_reference.resolve())},
            "predictions": {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in prediction_paths.items()
            },
        },
        "outputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in outputs.items()
        },
    }
    write_json(
        manifest_path,
        manifest,
        overwrite=args.overwrite,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
