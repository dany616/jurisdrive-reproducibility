from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path(
    os.environ.get(
        "JURISDRIVE_MANIFEST",
        REPO_ROOT / "artifacts" / "audit" / "final_car_to_car_manifest.jsonl",
    )
)
DEFAULT_FULL_RUN = Path(
    os.environ.get(
        "JURISDRIVE_FULL_RUN_DIR",
        REPO_ROOT / "data" / "full_run",
    )
)
DEFAULT_ARTIFACTS = REPO_ROOT / "artifacts"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            yield value


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def candidate_source_path(
    row: dict[str, Any],
    *,
    full_run_dir: Path | None = None,
) -> Path:
    """Resolve a manifest row without relying on a machine-specific path.

    Fresh manifests use paths relative to ``full_run_dir``. Archived local
    manifests may still contain an absolute source path, which remains a
    compatibility fallback but is never required by the public package.
    """

    source_stage = str(row.get("source_stage") or "")
    result_file = str(row.get("result_file") or "")
    candidates: list[Path] = []

    if full_run_dir is not None and result_file:
        stage_directory = {
            "rule": Path("output") / "car_to_car",
            "llm": Path("ambiguous_done") / "car_to_car",
        }.get(source_stage)
        if stage_directory is not None:
            candidates.append(Path(full_run_dir) / stage_directory / result_file)

    historical = row.get("source_path")
    if historical:
        historical_path = Path(str(historical))
        if not historical_path.is_absolute() and full_run_dir is not None:
            historical_path = Path(full_run_dir) / historical_path
        candidates.append(historical_path)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    attempted = ", ".join(str(path) for path in candidates) or "<none>"
    raise FileNotFoundError(
        "Candidate source file was not found. "
        f"Attempted: {attempted}. Set --full-run-dir or "
        "JURISDRIVE_FULL_RUN_DIR to the reproduced N0-N3 full_run directory."
    )


def load_candidate(
    row: dict[str, Any],
    *,
    full_run_dir: Path | None = None,
) -> dict[str, Any]:
    source_path = candidate_source_path(row, full_run_dir=full_run_dir)
    record = read_json(source_path)
    record["_manifest"] = row
    return record


def stable_id(prefix: str, *parts: object) -> str:
    raw = "|".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:12]}"
