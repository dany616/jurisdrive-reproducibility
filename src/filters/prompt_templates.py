from __future__ import annotations

import json
from typing import Any

LLM_OUTPUT_SCHEMA = {
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
        "label": {
            "type": "string",
            "enum": ["car_to_car", "not_car_to_car", "ambiguous"],
        },
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"],
        },
        "reason": {"type": "string"},
        "evidence": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "accident_type",
        "is_car_to_car",
        "label",
        "confidence",
        "reason",
        "evidence",
    ],
    "additionalProperties": False,
}


def build_system_prompt() -> str:
    return (
        "당신은 한국 판결문 후처리 분류기입니다. "
        "목표는 차대차 사고만 남기고 나머지는 배제하는 것입니다. "
        "단순 키워드가 아니라 실제 충돌 대상이 다른 차량인지 판단하세요. "
        "다음 경우는 car_to_car가 아닙니다: 보행자 충돌, 자전거/오토바이 충돌, "
        "단독사고, 건물/벽/가드레일/전봇대 등 구조물 충돌, 지게차/굴착기/크레인 등 작업사고. "
        "정보가 부족하면 ambiguous를 반환하세요. "
        "반드시 JSON 객체 하나만 반환하세요."
    )


def build_llm_input_payload(record: dict[str, Any], rule_result: dict[str, Any]) -> dict[str, Any]:
    parsed = record.get("parsed") if isinstance(record.get("parsed"), dict) else {}
    return {
        "input_file": record.get("input_file"),
        "source_text": record.get("source_text"),
        "source_mode": record.get("source_mode"),
        "parsed": {
            "accident_trajectory": parsed.get("accident_trajectory"),
            "vehicle_type": parsed.get("vehicle_type"),
            "location": parsed.get("location"),
            "road_type": parsed.get("road_type"),
            "weather_or_environment": parsed.get("weather_or_environment"),
        },
        "rule_summary": {
            "label": rule_result.get("label"),
            "score": rule_result.get("score"),
            "reason": rule_result.get("reason"),
            "matched_positive_keywords": rule_result.get("matched_positive_keywords"),
            "matched_negative_keywords": rule_result.get("matched_negative_keywords"),
            "matched_patterns": rule_result.get("matched_patterns"),
        },
    }


def build_user_prompt(record: dict[str, Any], rule_result: dict[str, Any]) -> str:
    payload = build_llm_input_payload(record, rule_result)
    schema_text = json.dumps(LLM_OUTPUT_SCHEMA, ensure_ascii=False, indent=2)
    payload_text = json.dumps(payload, ensure_ascii=False, indent=2)
    return (
        "아래 입력을 보고 차대차 사고 여부를 판정하세요.\n"
        "판정 기준:\n"
        "1. 서로 다른 차량 2대 이상이 등장해야 합니다.\n"
        "2. 충돌 대상이 다른 차량이어야 합니다.\n"
        "3. 보행자, 자전거, 오토바이, 구조물, 작업기계 충돌은 not_car_to_car입니다.\n"
        "4. 정보가 불충분하면 ambiguous입니다.\n\n"
        f"반환 스키마:\n{schema_text}\n\n"
        f"입력 데이터:\n{payload_text}"
    )
