#!/usr/bin/env python3
"""Populate the 900-case Qwen-only prediction file without reading human labels."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import request

SCHEMA = {
    "type": "object",
    "properties": {
        "accident_type": {
            "type": "string",
            "enum": [
                "car_to_car",
                "vehicle_to_person",
                "single_vehicle",
                "vehicle_to_structure",
                "construction_equipment_work_accident",
                "other_or_unknown",
            ],
        },
        "is_car_to_car": {"type": "boolean"},
        "label": {"type": "string", "enum": ["car_to_car", "not_car_to_car", "ambiguous"]},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "reason": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
    },
    "required": ["accident_type", "is_car_to_car", "label", "confidence", "reason", "evidence"],
    "additionalProperties": False,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def get_json(url: str) -> dict:
    with request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(url: str, payload: dict, timeout: float) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    http_request = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    with request.urlopen(http_request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_json_content(content: str) -> dict:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(stripped)


def classify(
    base_url: str,
    model: str,
    task: dict,
    max_tokens: int,
    timeout: float,
    retries: int,
) -> dict:
    started = time.perf_counter()
    candidate_id = int(task["candidate_id"])
    content = None
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = post_json(
                f"{base_url.rstrip('/')}/chat/completions",
                {
                "model": model,
                "temperature": 0.0,
                "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "당신은 한국 판결문 후처리 분류기입니다. 목표는 차대차 사고만 남기고 나머지는 배제하는 것입니다. "
                            "단순 키워드가 아니라 실제 충돌 대상이 다른 차량인지 판단하세요. 보행자, 자전거/오토바이, "
                            "단독사고, 구조물 충돌, 작업기계 사고는 car_to_car가 아닙니다. 정보가 부족하면 ambiguous를 "
                            "반환하고 반드시 JSON 객체 하나만 반환하세요."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "아래 원문만 보고 차대차 사고 여부를 판정하세요. 서로 다른 도로 차량 2대 이상과 차량 간 "
                            "접촉 근거가 있어야 car_to_car입니다. 원문 외의 규칙 결과나 기존 모델 판정은 제공되지 않습니다.\n\n"
                            f"반환 스키마:\n{json.dumps(SCHEMA, ensure_ascii=False)}\n\n"
                            f"원문:\n{task.get('source_text') or ''}"
                        ),
                    },
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "gold_qwen_only_prediction", "schema": SCHEMA},
                },
                },
                timeout,
            )
            content = response["choices"][0]["message"]["content"]
            parsed = parse_json_content(content)
            raw_label = parsed["label"]
            prediction = "abstain" if raw_label == "ambiguous" else raw_label
            return {
                "candidate_id": candidate_id,
                "prediction": prediction,
                "raw_label": raw_label,
                "confidence": parsed.get("confidence"),
                "reason": parsed.get("reason"),
                "evidence": parsed.get("evidence") or [],
                "raw_content": content,
                "model": model,
                "recorded_at_utc": utc_now(),
                "elapsed_seconds": round(time.perf_counter() - started, 4),
                "attempts": attempt,
                "error": None,
            }
        except Exception as exc:
            last_error = str(exc).strip() or exc.__class__.__name__
    return {
            "candidate_id": candidate_id,
            "prediction": None,
            "raw_label": None,
            "confidence": None,
            "reason": None,
            "evidence": [],
            "raw_content": content,
            "model": model,
            "recorded_at_utc": utc_now(),
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "attempts": retries,
            "error": last_error,
        }


def atomic_write(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


async def main_async(args: argparse.Namespace) -> int:
    tasks = [json.loads(line) for line in args.tasks.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit is not None:
        tasks = tasks[: args.limit]
    base_url = args.api_base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    model = args.model
    if not model:
        models = get_json(f"{base_url}/models").get("data") or []
        if not models:
            raise SystemExit("No served model was returned by /v1/models")
        model = models[0]["id"]

    existing = {}
    if args.resume and args.output.exists():
        existing = {
            int(row["candidate_id"]): row
            for row in (json.loads(line) for line in args.output.read_text(encoding="utf-8").splitlines() if line.strip())
            if row.get("prediction") in {"car_to_car", "not_car_to_car", "abstain"}
        }
    selected = [task for task in tasks if int(task["candidate_id"]) not in existing]
    semaphore = asyncio.Semaphore(args.concurrency)

    async def one(task: dict) -> dict:
        async with semaphore:
            return await asyncio.to_thread(
                classify,
                base_url,
                model,
                task,
                args.max_tokens,
                args.timeout,
                args.retries,
            )

    started = time.perf_counter()
    results = await asyncio.gather(*(one(task) for task in selected))
    combined = {**existing, **{int(row["candidate_id"]): row for row in results}}
    ordered = [combined.get(int(task["candidate_id"]), {"candidate_id": int(task["candidate_id"]), "prediction": None}) for task in tasks]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(args.output, ordered)
    summary = {
        "model": model,
        "tasks": len(tasks),
        "newly_processed": len(results),
        "success": sum(row.get("prediction") is not None for row in ordered),
        "failed_or_missing": sum(row.get("prediction") is None for row in ordered),
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "output": str(args.output.resolve()),
    }
    summary_path = args.output.with_name("predictions_qwen_only_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["failed_or_missing"] == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default=None, help="Omit to discover the first served model from /v1/models.")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
