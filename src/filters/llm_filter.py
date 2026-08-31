from __future__ import annotations

import json
from typing import Any

from prompt_templates import build_system_prompt, build_user_prompt

VALID_LABELS = {"car_to_car", "not_car_to_car", "ambiguous"}
VALID_CONFIDENCE = {"high", "medium", "low"}
VALID_ACCIDENT_TYPES = {
    "car_to_car",
    "vehicle_to_person",
    "single_vehicle",
    "vehicle_to_structure",
    "construction_equipment_work_accident",
    "other_or_unknown",
}


def fallback_llm_result(reason: str = "LLM response unavailable") -> dict[str, Any]:
    return {
        "accident_type": "other_or_unknown",
        "is_car_to_car": False,
        "label": "ambiguous",
        "confidence": "low",
        "reason": reason,
        "evidence": [],
    }


def extract_json_object(raw_text: str) -> str:
    stripped = raw_text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in LLM response")
    return stripped[start : end + 1]


def normalize_llm_result(raw_text: str) -> dict[str, Any]:
    try:
        payload = json.loads(extract_json_object(raw_text))
    except Exception as exc:
        return fallback_llm_result(reason=f"Failed to parse LLM response: {exc}")

    result = fallback_llm_result()
    if payload.get("accident_type") in VALID_ACCIDENT_TYPES:
        result["accident_type"] = payload["accident_type"]
    if isinstance(payload.get("is_car_to_car"), bool):
        result["is_car_to_car"] = payload["is_car_to_car"]
    if payload.get("label") in VALID_LABELS:
        result["label"] = payload["label"]
    if payload.get("confidence") in VALID_CONFIDENCE:
        result["confidence"] = payload["confidence"]
    if isinstance(payload.get("reason"), str) and payload["reason"].strip():
        result["reason"] = payload["reason"].strip()
    if isinstance(payload.get("evidence"), list):
        result["evidence"] = [str(item).strip() for item in payload["evidence"] if str(item).strip()]
    return result


def build_llm_messages(record: dict[str, Any], rule_result: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": build_user_prompt(record, rule_result)},
    ]


def classify_with_llm(*, client: Any, model_name: str, record: dict[str, Any], rule_result: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    response = client.chat.completions.create(
        model=model_name,
        temperature=0.0,
        messages=build_llm_messages(record, rule_result),
    )
    raw_content = response.choices[0].message.content
    return raw_content, normalize_llm_result(raw_content)
