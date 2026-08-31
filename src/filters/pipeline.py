#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prompt_templates import build_llm_input_payload
from rule_filter import LABEL_AMBIGUOUS, LABEL_CAR_TO_CAR, LABEL_NOT_CAR_TO_CAR, classify_record

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR.parent / "zeroshot_done"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "output"
DEFAULT_REPORT_PATH = SCRIPT_DIR / "report.jsonl"
DEFAULT_SUMMARY_PATH = SCRIPT_DIR / "summary.json"
DEFAULT_LLM_CANDIDATES_PATH = SCRIPT_DIR / "llm_candidates.jsonl"
INPUT_NAME_RE = re.compile(r"zeroshot_test_(\d+)_result\.json$")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def natural_sort_key(path: Path) -> tuple[int, str]:
    match = INPUT_NAME_RE.fullmatch(path.name)
    if match:
        return int(match.group(1)), path.name
    return sys.maxsize, path.name


def discover_input_files(input_dir: Path) -> list[Path]:
    files = [path for path in input_dir.glob("zeroshot_test_*_result.json") if INPUT_NAME_RE.fullmatch(path.name)]
    return sorted(files, key=natural_sort_key)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def ensure_output_dirs(output_root: Path) -> dict[str, Path]:
    directories = {
        LABEL_CAR_TO_CAR: output_root / LABEL_CAR_TO_CAR,
        LABEL_NOT_CAR_TO_CAR: output_root / LABEL_NOT_CAR_TO_CAR,
        LABEL_AMBIGUOUS: output_root / LABEL_AMBIGUOUS,
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def enrich_record(record: dict[str, Any], rule_result: dict[str, Any], llm_result: dict[str, Any] | None, llm_raw: str | None, final_label: str) -> dict[str, Any]:
    enriched = dict(record)
    enriched["postprocess"] = {
        "rule": rule_result,
        "llm": llm_result,
        "llm_raw": llm_raw,
        "final_label": final_label,
        "processed_at": utc_now_iso(),
    }
    return enriched


def build_report_row(input_path: Path, record: dict[str, Any], rule_result: dict[str, Any], final_label: str) -> dict[str, Any]:
    return {
        "input_file": input_path.name,
        "source_input_file": record.get("input_file"),
        "rule_label": rule_result.get("label"),
        "rule_score": rule_result.get("score"),
        "rule_reason": rule_result.get("reason"),
        "matched_positive_keywords": rule_result.get("matched_positive_keywords"),
        "matched_negative_keywords": rule_result.get("matched_negative_keywords"),
        "matched_patterns": rule_result.get("matched_patterns"),
        "vehicle_mentions": rule_result.get("vehicle_mentions"),
        "collision_hits": rule_result.get("collision_hits"),
        "work_hits": rule_result.get("work_hits"),
        "facility_hits": rule_result.get("facility_hits"),
        "used_llm": False,
        "llm_label": None,
        "final_label": final_label,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rule-based first-pass filter for car-to-car accident extraction.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--llm-candidates-path", type=Path, default=DEFAULT_LLM_CANDIDATES_PATH)
    parser.add_argument("--start-index", type=int, default=1, help="1-based start index in natural sort order.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--reset-report", action="store_true", help="Remove previous report/candidate files before running.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    report_path = args.report_path.expanduser().resolve()
    summary_path = args.summary_path.expanduser().resolve()
    llm_candidates_path = args.llm_candidates_path.expanduser().resolve()

    files = discover_input_files(input_dir)
    if not files:
        raise SystemExit(f"No input files found in {input_dir}")
    if args.start_index < 1:
        raise SystemExit("--start-index must be at least 1")

    start_offset = args.start_index - 1
    files = files[start_offset:]
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise SystemExit("No files selected after applying start/limit")

    if args.reset_report:
        for path in [report_path, llm_candidates_path]:
            if path.exists():
                path.unlink()

    output_dirs = ensure_output_dirs(output_root)
    counters: Counter[str] = Counter()
    total = len(files)
    started_at = utc_now_iso()
    failed_files: list[dict[str, str]] = []

    print(f"처리 대상 파일 수: {total}", flush=True)
    for index, input_path in enumerate(files, start=1):
        try:
            record = read_json(input_path)
            rule_result = classify_record(record)
            final_label = rule_result["label"]
            destination = output_dirs[final_label] / input_path.name

            if args.skip_existing and destination.exists():
                counters["skipped"] += 1
                if index <= 10 or index % 1000 == 0:
                    print(f"[{index}/{total}] skipped -> {destination.name}", flush=True)
                continue

            llm_result = None
            llm_raw = None
            enriched = enrich_record(
                record=record,
                rule_result=rule_result,
                llm_result=llm_result,
                llm_raw=llm_raw,
                final_label=final_label,
            )
            write_json(destination, enriched)
            append_jsonl(report_path, build_report_row(input_path, record, rule_result, final_label))

            if final_label == LABEL_AMBIGUOUS:
                candidate = {
                    "input_file": input_path.name,
                    "final_label": final_label,
                    "prompt_payload": build_llm_input_payload(record, rule_result),
                }
                append_jsonl(llm_candidates_path, candidate)

            counters[final_label] += 1
            counters["processed"] += 1
            if index <= 10 or index % 1000 == 0:
                print(f"[{index}/{total}] {input_path.name} -> {final_label}", flush=True)
        except Exception as exc:
            failed_files.append({"input_file": input_path.name, "error": str(exc) or exc.__class__.__name__})
            counters["failed"] += 1
            print(f"[{index}/{total}] failed -> {input_path.name} ({exc})", flush=True)

    summary = {
        "started_at": started_at,
        "finished_at": utc_now_iso(),
        "input_dir": str(input_dir),
        "output_root": str(output_root),
        "report_path": str(report_path),
        "llm_candidates_path": str(llm_candidates_path),
        "start_index": args.start_index,
        "limit": args.limit,
        "selected_files": total,
        "processed": counters["processed"],
        "skipped": counters["skipped"],
        "car_to_car": counters[LABEL_CAR_TO_CAR],
        "not_car_to_car": counters[LABEL_NOT_CAR_TO_CAR],
        "ambiguous": counters[LABEL_AMBIGUOUS],
        "failed": counters["failed"],
        "failed_files": failed_files,
    }
    write_json(summary_path, summary)

    print("완료", flush=True)
    print(f"Processed: {counters['processed']}", flush=True)
    print(f"Skipped: {counters['skipped']}", flush=True)
    print(f"Car-to-car: {counters[LABEL_CAR_TO_CAR]}", flush=True)
    print(f"Not car-to-car: {counters[LABEL_NOT_CAR_TO_CAR]}", flush=True)
    print(f"Ambiguous: {counters[LABEL_AMBIGUOUS]}", flush=True)
    print(f"Failed: {counters['failed']}", flush=True)
    print(f"Summary: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
