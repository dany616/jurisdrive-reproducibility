from __future__ import annotations

import re
from typing import Any

LABEL_CAR_TO_CAR = "car_to_car"
LABEL_NOT_CAR_TO_CAR = "not_car_to_car"
LABEL_AMBIGUOUS = "ambiguous"

CAR_VEHICLE_KEYWORDS = [
    "차량",
    "자동차",
    "승용차",
    "승합차",
    "화물차",
    "버스",
    "택시",
    "트럭",
    "레미콘",
    "트레일러",
    "덤프트럭",
    "냉동탑차",
    "카고트럭",
    "승용",
    "SUV",
    "지프",
    "승합",
]

ROLE_PATTERNS = {
    "원고 차량": re.compile(r"원고\s*(?:의\s*)?(?:차량|차|자동차)"),
    "피고 차량": re.compile(r"피고\s*(?:의\s*)?(?:차량|차|자동차)"),
    "피해 차량": re.compile(r"피해(?:자)?\s*(?:의\s*)?(?:차량|차|자동차)"),
    "가해 차량": re.compile(r"가해(?:자)?\s*(?:의\s*)?(?:차량|차|자동차)"),
    "상대 차량": re.compile(r"상대\s*(?:차량|차)"),
    "앞 차량": re.compile(r"(?:전방|앞)\s*(?:차량|차)"),
    "뒤 차량": re.compile(r"(?:후방|뒤)\s*(?:차량|차)"),
    "피고인 차량": re.compile(r"피고인\s*(?:의\s*)?(?:차량|차|자동차)"),
}

PAIR_CONTEXT_PATTERNS = {
    "원고/피고 차량": re.compile(r"원고.{0,12}(?:차량|차).{0,24}피고.{0,12}(?:차량|차)|피고.{0,12}(?:차량|차).{0,24}원고.{0,12}(?:차량|차)"),
    "피해/가해 차량": re.compile(r"피해.{0,12}(?:차량|차).{0,24}가해.{0,12}(?:차량|차)|가해.{0,12}(?:차량|차).{0,24}피해.{0,12}(?:차량|차)"),
    "상대 차량 표현": re.compile(r"(?:다른|상대|앞서|전방의)\s*(?:차량|차)"),
    "다중 차량 표현": re.compile(r"(?:1차|2차)\s*충돌|연쇄\s*추돌|다중\s*추돌"),
}

COLLISION_KEYWORDS = [
    "추돌",
    "충돌",
    "접촉",
    "충격",
    "들이받",
    "들이받고",
    "부딪",
    "부딪혀",
    "긁",
    "스치",
]

ACCIDENT_KEYWORDS = [
    "교통사고",
    "사고",
    "충돌",
    "추돌",
    "접촉",
    "차선변경",
    "유턴",
    "좌회전",
    "우회전",
    "직진",
    "정차",
    "신호위반",
]

FACILITY_KEYWORDS = [
    "건물",
    "벽",
    "담장",
    "가드레일",
    "중앙분리대",
    "전봇대",
    "신호등",
    "구조물",
    "시설물",
    "펜스",
    "연석",
    "가로수",
    "방호벽",
    "교각",
    "난간",
]

PEDESTRIAN_KEYWORDS = [
    "보행자",
    "행인",
    "보행 중",
]

TWO_WHEELER_KEYWORDS = [
    "오토바이",
    "원동기장치자전거",
    "이륜차",
    "자전거",
    "킥보드",
]

WORK_MACHINERY_KEYWORDS = [
    "크레인",
    "굴착기",
    "포크레인",
    "지게차",
    "파일드라이버",
    "항타기",
    "불도저",
    "로더",
]

WORK_CONTEXT_KEYWORDS = [
    "작업",
    "공사",
    "현장",
    "하역",
    "적재",
    "조립",
    "신호수",
    "현장소장",
    "배달업무",
]

NON_TRAFFIC_CONTEXT_KEYWORDS = [
    "지급",
    "투자",
    "계약",
    "매매",
    "도급",
    "공사대금",
    "지연손해금",
    "소장",
    "송달",
    "보험금",
]

VEHICLE_TO_VEHICLE_PATTERN = re.compile(
    r"(?:차량|자동차|승용차|화물차|버스|택시|트럭).{0,35}"
    r"(?:추돌|충돌|접촉|충격|들이받|부딪).{0,50}"
    r"(?:다른|상대|피해|가해|원고|피고|전방|앞서)?.{0,20}"
    r"(?:차량|자동차|승용차|화물차|버스|택시|트럭)"
)

VEHICLE_TO_PERSON_PATTERN = re.compile(
    r"(?:차량|자동차|승용차|화물차|버스|택시|트럭).{0,30}(?:보행자|행인|자전거|오토바이|원동기장치자전거|이륜차)"
)

VEHICLE_TO_FACILITY_PATTERN = re.compile(
    r"(?:차량|자동차|승용차|화물차|버스|택시|트럭).{0,30}(?:건물|벽|담장|가드레일|중앙분리대|전봇대|신호등|구조물|시설물|펜스|연석|가로수|방호벽|교각|난간)"
)

WORK_ACCIDENT_PATTERN = re.compile(
    r"(?:크레인|굴착기|포크레인|지게차|파일드라이버|항타기|불도저|로더).{0,40}(?:작업|공사|현장|하역|적재|조립)"
)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _get_parsed(record: dict[str, Any]) -> dict[str, Any]:
    parsed = record.get("parsed")
    return parsed if isinstance(parsed, dict) else {}


def build_analysis_text(record: dict[str, Any]) -> str:
    parsed = _get_parsed(record)
    sections = [
        record.get("source_text") or "",
        parsed.get("accident_trajectory") or "",
        parsed.get("vehicle_type") or "",
        parsed.get("location") or "",
        parsed.get("road_type") or "",
        parsed.get("weather_or_environment") or "",
    ]
    return _normalize_text("\n".join(str(section) for section in sections if section))


def _count_occurrences(text: str, keywords: list[str]) -> tuple[int, list[str]]:
    matches = [keyword for keyword in keywords if keyword in text]
    return len(matches), matches


def _count_pattern_matches(text: str, patterns: dict[str, re.Pattern[str]]) -> tuple[int, list[str]]:
    matches = [label for label, pattern in patterns.items() if pattern.search(text)]
    return len(matches), matches


def _generic_vehicle_count(text: str) -> int:
    count = 0
    for keyword in CAR_VEHICLE_KEYWORDS:
        count += text.count(keyword)
    return count


def _vehicle_mentions(text: str, role_hits: int) -> int:
    return max(role_hits, min(_generic_vehicle_count(text), 8))


def classify_record(record: dict[str, Any]) -> dict[str, Any]:
    text = build_analysis_text(record)
    if not text:
        return {
            "label": LABEL_NOT_CAR_TO_CAR,
            "score": -5,
            "vehicle_mentions": 0,
            "collision_hits": 0,
            "work_hits": 0,
            "facility_hits": 0,
            "reason": ["분석 가능한 텍스트가 없음"],
            "matched_positive_keywords": [],
            "matched_negative_keywords": [],
            "matched_patterns": [],
        }

    role_hits, matched_roles = _count_pattern_matches(text, ROLE_PATTERNS)
    pair_context_hits, matched_pair_patterns = _count_pattern_matches(text, PAIR_CONTEXT_PATTERNS)
    collision_hits, matched_collision_keywords = _count_occurrences(text, COLLISION_KEYWORDS)
    accident_hits, matched_accident_keywords = _count_occurrences(text, ACCIDENT_KEYWORDS)
    facility_hits, matched_facility_keywords = _count_occurrences(text, FACILITY_KEYWORDS)
    person_hits, matched_person_keywords = _count_occurrences(text, PEDESTRIAN_KEYWORDS)
    two_wheeler_hits, matched_two_wheeler_keywords = _count_occurrences(text, TWO_WHEELER_KEYWORDS)
    machinery_hits, matched_machinery_keywords = _count_occurrences(text, WORK_MACHINERY_KEYWORDS)
    work_context_hits, matched_work_context_keywords = _count_occurrences(text, WORK_CONTEXT_KEYWORDS)
    non_traffic_hits, matched_non_traffic_keywords = _count_occurrences(text, NON_TRAFFIC_CONTEXT_KEYWORDS)
    vehicle_mentions = _vehicle_mentions(text, role_hits)

    matched_patterns: list[str] = []
    if VEHICLE_TO_VEHICLE_PATTERN.search(text):
        matched_patterns.append("vehicle_to_vehicle_collision_pattern")
    if VEHICLE_TO_PERSON_PATTERN.search(text):
        matched_patterns.append("vehicle_to_person_pattern")
    if VEHICLE_TO_FACILITY_PATTERN.search(text):
        matched_patterns.append("vehicle_to_facility_pattern")
    if WORK_ACCIDENT_PATTERN.search(text):
        matched_patterns.append("work_accident_pattern")
    matched_patterns.extend(matched_pair_patterns)

    positive_keywords = sorted(set(matched_roles + matched_collision_keywords + matched_accident_keywords))
    negative_keywords = sorted(
        set(
            matched_facility_keywords
            + matched_person_keywords
            + matched_two_wheeler_keywords
            + matched_machinery_keywords
            + matched_work_context_keywords
            + matched_non_traffic_keywords
        )
    )

    work_hits = machinery_hits + work_context_hits
    negative_total = facility_hits + person_hits + two_wheeler_hits + work_hits

    reasons: list[str] = []
    if vehicle_mentions >= 2:
        reasons.append(f"차량 언급 {vehicle_mentions}회 수준으로 2대 이상 가능성이 높음")
    elif vehicle_mentions == 1:
        reasons.append("차량 언급은 있으나 1대 수준으로 보임")
    else:
        reasons.append("차량 언급이 거의 없음")

    if collision_hits:
        reasons.append(f"충돌/접촉 표현 {collision_hits}개 확인")
    else:
        reasons.append("충돌/접촉 표현이 약함")

    if pair_context_hits:
        reasons.append("상대 차량 또는 다중 차량 문맥 확인")
    if facility_hits:
        reasons.append("구조물 충돌 신호 존재")
    if person_hits:
        reasons.append("보행자/사람 충돌 신호 존재")
    if two_wheeler_hits:
        reasons.append("자전거/오토바이 등 비자동차 신호 존재")
    if work_hits:
        reasons.append("작업기계 또는 작업현장 문맥이 강함")
    if non_traffic_hits and accident_hits == 0 and collision_hits == 0 and vehicle_mentions == 0:
        reasons.append("교통사고보다는 일반 민사/상사 문맥이 강함")

    score = (
        vehicle_mentions * 2
        + collision_hits * 3
        + accident_hits
        + pair_context_hits * 2
        + (4 if "vehicle_to_vehicle_collision_pattern" in matched_patterns else 0)
        - facility_hits * 3
        - person_hits * 4
        - two_wheeler_hits * 4
        - work_hits * 2
        - non_traffic_hits
        - (4 if "vehicle_to_person_pattern" in matched_patterns else 0)
        - (4 if "vehicle_to_facility_pattern" in matched_patterns else 0)
        - (4 if "work_accident_pattern" in matched_patterns else 0)
    )

    if "vehicle_to_person_pattern" in matched_patterns or person_hits > 0:
        label = LABEL_NOT_CAR_TO_CAR
    elif two_wheeler_hits > 0:
        label = LABEL_NOT_CAR_TO_CAR
    elif "vehicle_to_facility_pattern" in matched_patterns or facility_hits >= 2 or (facility_hits >= 1 and "vehicle_to_vehicle_collision_pattern" not in matched_patterns and pair_context_hits == 0):
        label = LABEL_NOT_CAR_TO_CAR
    elif "work_accident_pattern" in matched_patterns or (machinery_hits > 0 and work_context_hits > 0):
        label = LABEL_NOT_CAR_TO_CAR
    elif vehicle_mentions == 0 and collision_hits == 0 and accident_hits == 0:
        label = LABEL_NOT_CAR_TO_CAR
    elif "vehicle_to_vehicle_collision_pattern" in matched_patterns and negative_total == 0:
        label = LABEL_CAR_TO_CAR
    elif vehicle_mentions >= 2 and collision_hits >= 1 and pair_context_hits >= 1 and negative_total == 0:
        label = LABEL_CAR_TO_CAR
    elif vehicle_mentions >= 2 and collision_hits >= 1 and negative_total == 0:
        label = LABEL_AMBIGUOUS
    elif vehicle_mentions >= 1 and (collision_hits >= 1 or accident_hits >= 1):
        label = LABEL_AMBIGUOUS
    else:
        label = LABEL_NOT_CAR_TO_CAR

    return {
        "label": label,
        "score": score,
        "vehicle_mentions": vehicle_mentions,
        "collision_hits": collision_hits,
        "work_hits": work_hits,
        "facility_hits": facility_hits,
        "reason": reasons,
        "matched_positive_keywords": positive_keywords,
        "matched_negative_keywords": negative_keywords,
        "matched_patterns": matched_patterns,
    }
