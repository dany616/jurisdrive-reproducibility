from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol
from urllib import request

from .io import sha256_text, stable_id
from .models import (
    EvidenceGraphV1,
    EvidenceRef,
    GraphEdge,
    GraphNode,
    LegalStatus,
    NodeType,
    ProvenanceValue,
    RelationType,
)

COLLISION_TERMS = (
    "충격",
    "충돌",
    "추돌",
    "접촉",
    "들이받",
    "부딪",
)
MANEUVER_TERMS = (
    ("signal_wait", "신호대기"),
    ("stopped", "정차"),
    ("lane_change", "차선변경"),
    ("lane_change", "차선 변경"),
    ("lane_change", "차선을 변경"),
    ("lane_change", "차로 변경"),
    ("lane_change", "차로를 변경"),
    ("lane_change", "진로변경"),
    ("lane_change", "진로 변경"),
    ("lane_change", "진로를 변경"),
    ("left_turn", "좌회전"),
    ("right_turn", "우회전"),
    ("merge", "합류"),
    ("centerline_intrusion", "중앙선"),
)
VEHICLE_ROLE_RE = re.compile(
    r"(?:피고(?:인)?|원고|피해자|가해자|상대방|선행|후행|앞|뒤)"
    r"(?:의|인\s+운전의?|가\s+운전하는|가\s+운전하던)?\s*"
    r"(?:차량|자동차|승용차|승합차|화물차량?|트럭|버스|오토바이)"
)
MODEL_VEHICLE_RE = re.compile(
    r"(?:[A-Z]\s+)?[A-Z가-힣0-9()·-]{2,15}\s+"
    r"(?:승용차|승합차|화물차량?|트럭|버스|오토바이)"
)
POSSESSIVE_VEHICLE_RE = re.compile(
    r"(?:자신의|피고인의|원고의|피해자의|피고인\s+운전|원고\s+운전)\s*차량"
)
DRIVER_RE = re.compile(r"([가-힣A-Za-z0-9·-]{1,20})가\s+운전하던")


@dataclass(frozen=True)
class SentenceSpan:
    text: str
    start: int
    end: int


class RelationResolver(Protocol):
    name: str

    def resolve(self, graph: EvidenceGraphV1, source_text: str) -> dict[str, Any] | None: ...


class OpenAICompatibleResolver:
    name = "qwen_openai_compatible"

    def __init__(self, endpoint: str, model: str, timeout: float = 120.0) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout = timeout

    def resolve(self, graph: EvidenceGraphV1, source_text: str) -> dict[str, Any] | None:
        entity_ids = [node.id for node in graph.nodes if node.type == NodeType.VEHICLE]
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 256,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Resolve only the missing collision agent and target. "
                        "Use existing entity IDs and an exact quote from the source. "
                        "Return one JSON object with agent_id, target_id, quote, confidence."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "entity_ids": entity_ids,
                            "unresolved": graph.critical_unresolved,
                            "source_text": source_text,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        url = self.endpoint
        if url.endswith("/v1"):
            url = f"{url}/chat/completions"
        elif not url.endswith("/chat/completions"):
            url = f"{url}/v1/chat/completions"
        http_request = request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
            method="POST",
        )
        with request.urlopen(http_request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        content = body["choices"][0]["message"].get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("resolver response contained no parseable content")
        return json.loads(content)


def sentence_spans(text: str) -> list[SentenceSpan]:
    spans: list[SentenceSpan] = []
    for match in re.finditer(r"[^\n]+(?:\n|$)", text):
        line = match.group(0).rstrip("\n")
        line_offset = match.start()
        for sentence in re.finditer(r"[^.!?。]+(?:[.!?。]|$)", line):
            value = sentence.group(0).strip()
            if not value:
                continue
            leading = len(sentence.group(0)) - len(sentence.group(0).lstrip())
            start = line_offset + sentence.start() + leading
            spans.append(SentenceSpan(value, start, start + len(value)))
    return spans or [SentenceSpan(text, 0, len(text))]


def detect_legal_status(text: str) -> LegalStatus:
    if any(term in text for term in ("주장", "항소이유", "원고는", "피고는")):
        return LegalStatus.PARTY_CLAIM
    if any(term in text for term in ("인정사실", "범죄사실", "공소사실", "인정된다")):
        return LegalStatus.ACCEPTED_FACT
    if any(term in text for term in ("판단", "살피건대", "법원은", "보건대")):
        return LegalStatus.COURT_REASONING
    return LegalStatus.UNKNOWN


def _evidence(
    source_file: str,
    source_text: str,
    span: SentenceSpan,
    extractor: str,
    confidence: float,
) -> EvidenceRef:
    supported = source_text[span.start : span.end] == span.text
    return EvidenceRef(
        id=stable_id("ev", source_file, span.start, span.end, span.text),
        source_file=source_file,
        quote=span.text,
        start=span.start,
        end=span.end,
        source_text_sha256=sha256_text(source_text),
        extractor=extractor,
        confidence=confidence,
        provenance=ProvenanceValue.OBSERVED,
        legal_status=detect_legal_status(span.text),
        supported=supported,
    )


def validate_evidence_spans(graph: EvidenceGraphV1, source_text: str) -> list[str]:
    errors: list[str] = []
    for evidence in graph.evidence:
        if evidence.source_text_sha256 != sha256_text(source_text):
            errors.append(f"{evidence.id}: source hash mismatch")
        if source_text[evidence.start : evidence.end] != evidence.quote:
            errors.append(f"{evidence.id}: quote/offset mismatch")
    evidence_by_id = {item.id: item for item in graph.evidence}
    for edge in graph.edges:
        if edge.provenance == ProvenanceValue.OBSERVED:
            if not edge.evidence_ids:
                errors.append(f"{edge.id}: observed relation has no evidence")
            for evidence_id in edge.evidence_ids:
                if evidence_id not in evidence_by_id or not evidence_by_id[evidence_id].supported:
                    errors.append(f"{edge.id}: unsupported evidence {evidence_id}")
    return errors


def _vehicle_mentions(text: str) -> list[str]:
    mentions: list[str] = []
    for regex in (VEHICLE_ROLE_RE, MODEL_VEHICLE_RE, POSSESSIVE_VEHICLE_RE):
        for match in regex.finditer(text):
            value = match.group(0).strip()
            if value and value not in mentions:
                mentions.append(value)
    return [
        value
        for value in mentions
        if not any(value != other and value in other for other in mentions)
    ]


def _mentions_in_clause(clause: str, aliases: dict[str, str]) -> list[str]:
    found: list[tuple[int, str]] = []
    for alias, entity_id in aliases.items():
        offset = clause.find(alias)
        if offset >= 0:
            found.append((offset, entity_id))
    return [entity_id for _, entity_id in sorted(found) if entity_id]


def _collision_roles(clause: str, aliases: dict[str, str]) -> tuple[str, str] | None:
    verb_offsets = [clause.rfind(term) for term in COLLISION_TERMS]
    verb_offset = max(verb_offsets)
    if verb_offset < 0:
        return None
    mentions: list[tuple[int, int, str]] = []
    for alias, entity_id in aliases.items():
        for match in re.finditer(re.escape(alias), clause):
            if match.start() < verb_offset:
                mentions.append((match.start(), match.end(), entity_id))
    mentions.sort()
    if len({item[2] for item in mentions}) < 2:
        return None

    object_candidates: list[tuple[int, str]] = []
    instrument_candidates: list[tuple[int, str]] = []
    for start, end, entity_id in mentions:
        suffix = clause[end:verb_offset]
        object_match = re.search(r"(?:을|를)(?:\s|,|$)", suffix)
        instrument_match = re.search(r"(?:으로|로)(?:\s|,|$)", suffix)
        if object_match and object_match.start() <= 35:
            object_candidates.append((start, entity_id))
        if instrument_match and instrument_match.start() <= 35:
            instrument_candidates.append((start, entity_id))

    target = object_candidates[-1][1] if object_candidates else None
    agent = instrument_candidates[-1][1] if instrument_candidates else None
    if agent and target and agent != target:
        return agent, target

    ordered = list(dict.fromkeys(item[2] for item in mentions))
    # Subject-first clauses such as "피고차량이 ... 원고차량을 접촉" are safe.
    subject_match = re.search(r"(이|가)(?:\s|,)", clause[mentions[0][1] : verb_offset])
    if subject_match and target and ordered[0] != target:
        return ordered[0], target
    return None


def _append_resolver_relation(
    graph: EvidenceGraphV1,
    source_text: str,
    resolved: dict[str, Any],
    resolver_name: str,
) -> None:
    vehicle_ids = {node.id for node in graph.nodes if node.type == NodeType.VEHICLE}
    agent = resolved.get("agent_id")
    target = resolved.get("target_id")
    quote = resolved.get("quote")
    if agent not in vehicle_ids or target not in vehicle_ids or agent == target:
        graph.review_required.append("resolver returned invalid entity IDs")
        return
    if not isinstance(quote, str) or quote not in source_text:
        graph.review_required.append("resolver returned a quote absent from source")
        return
    start = source_text.index(quote)
    span = SentenceSpan(quote, start, start + len(quote))
    evidence = _evidence(
        graph.source_file,
        source_text,
        span,
        resolver_name,
        float(resolved.get("confidence", 0.5)),
    )
    graph.evidence.append(evidence)
    graph.edges.append(
        GraphEdge(
            id=stable_id("edge", graph.scenario_id, "collision", agent, target),
            source=agent,
            target=target,
            relation=RelationType.COLLIDES_WITH,
            provenance=ProvenanceValue.INFERRED,
            evidence_ids=[evidence.id],
            confidence=evidence.confidence,
            supported=True,
        )
    )
    graph.critical_unresolved = [
        issue
        for issue in graph.critical_unresolved
        if issue not in {"collision_agent", "collision_target"}
    ]


def build_evidence_graph(
    record: dict[str, Any],
    *,
    resolver: RelationResolver | None = None,
) -> EvidenceGraphV1:
    manifest = record.get("_manifest") if isinstance(record.get("_manifest"), dict) else {}
    source_text = str(record.get("source_text") or "")
    source_file = str(record.get("input_file") or manifest.get("result_file") or "unknown")
    candidate_id = int(manifest.get("candidate_id") or 0)
    scenario_id = f"jurisdrive_{candidate_id}"
    nodes = [
        GraphNode(
            id=f"judgment_{candidate_id}",
            type=NodeType.JUDGMENT,
            label=source_file,
            attributes={"candidate_id": candidate_id},
        )
    ]
    edges: list[GraphEdge] = []
    evidence: list[EvidenceRef] = []
    review_required: list[str] = []

    aliases: dict[str, str] = {}
    mentions = _vehicle_mentions(source_text)
    for index, mention in enumerate(mentions, start=1):
        entity_id = f"vehicle_{index}"
        aliases[mention] = entity_id
        mention_start = source_text.index(mention)
        mention_span = SentenceSpan(mention, mention_start, mention_start + len(mention))
        mention_evidence = _evidence(
            source_file,
            source_text,
            mention_span,
            "rule_vehicle_mention",
            0.95,
        )
        evidence.append(mention_evidence)
        nodes.append(
            GraphNode(
                id=entity_id,
                type=NodeType.VEHICLE,
                label=mention,
                attributes={"aliases": [mention]},
                evidence_ids=[mention_evidence.id],
            )
        )
        evidence_node_id = f"evidence_{mention_evidence.id}"
        nodes.append(
            GraphNode(
                id=evidence_node_id,
                type=NodeType.EVIDENCE,
                label=mention,
                attributes={"start": mention_start, "end": mention_start + len(mention)},
                evidence_ids=[mention_evidence.id],
            )
        )
        edges.append(
            GraphEdge(
                id=stable_id("edge", entity_id, evidence_node_id, "supported_by"),
                source=entity_id,
                target=evidence_node_id,
                relation=RelationType.SUPPORTED_BY,
                provenance=ProvenanceValue.OBSERVED,
                evidence_ids=[mention_evidence.id],
                confidence=0.95,
            )
        )

    collision_spans = [
        span for span in sentence_spans(source_text) if any(term in span.text for term in COLLISION_TERMS)
    ]
    for index, span in enumerate(collision_spans, start=1):
        ref = _evidence(source_file, source_text, span, "rule_collision_clause", 0.9)
        collision_term_offset = max(span.text.rfind(term) for term in COLLISION_TERMS)
        evidence.append(ref)
        event_id = f"collision_event_{index}"
        nodes.append(
            GraphNode(
                id=event_id,
                type=NodeType.EVENT,
                label="vehicle_collision",
                attributes={
                    "order": index,
                    "source_start": span.start + max(collision_term_offset, 0),
                    "legal_status": ref.legal_status.value,
                },
                evidence_ids=[ref.id],
            )
        )
        evidence_node_id = f"evidence_{ref.id}"
        nodes.append(
            GraphNode(
                id=evidence_node_id,
                type=NodeType.EVIDENCE,
                label=span.text[:80],
                attributes={"start": span.start, "end": span.end},
                evidence_ids=[ref.id],
            )
        )
        edges.append(
            GraphEdge(
                id=stable_id("edge", event_id, evidence_node_id, "supported_by"),
                source=event_id,
                target=evidence_node_id,
                relation=RelationType.SUPPORTED_BY,
                provenance=ProvenanceValue.OBSERVED,
                evidence_ids=[ref.id],
                confidence=0.9,
            )
        )
        roles = _collision_roles(span.text, aliases)
        if roles:
            agent_id, target_id = roles
            edges.append(
                GraphEdge(
                    id=stable_id("edge", scenario_id, index, agent_id, target_id),
                    source=agent_id,
                    target=target_id,
                    relation=RelationType.COLLIDES_WITH,
                    provenance=ProvenanceValue.OBSERVED,
                    evidence_ids=[ref.id],
                    confidence=0.9,
                )
            )

    maneuver_index = 0
    for kind, term in MANEUVER_TERMS:
        for match in re.finditer(re.escape(term), source_text):
            maneuver_index += 1
            span = SentenceSpan(match.group(0), match.start(), match.end())
            ref = _evidence(
                source_file,
                source_text,
                span,
                "rule_maneuver_term",
                0.8,
            )
            if ref.id not in {item.id for item in evidence}:
                evidence.append(ref)
            event_id = f"maneuver_event_{maneuver_index}"
            nodes.append(
                GraphNode(
                    id=event_id,
                    type=NodeType.EVENT,
                    label=kind,
                    attributes={
                        "source_start": match.start(),
                        "legal_status": ref.legal_status.value,
                    },
                    evidence_ids=[ref.id],
                )
            )
            evidence_node_id = f"evidence_{ref.id}"
            if not any(node.id == evidence_node_id for node in nodes):
                nodes.append(
                    GraphNode(
                        id=evidence_node_id,
                        type=NodeType.EVIDENCE,
                        label=term,
                        attributes={"start": match.start(), "end": match.end()},
                        evidence_ids=[ref.id],
                    )
                )
            edges.append(
                GraphEdge(
                    id=stable_id("edge", event_id, evidence_node_id, "supported_by"),
                    source=event_id,
                    target=evidence_node_id,
                    relation=RelationType.SUPPORTED_BY,
                    provenance=ProvenanceValue.OBSERVED,
                    evidence_ids=[ref.id],
                    confidence=0.8,
                )
            )

    # Add conservative driver relations when one person and one vehicle are explicit in a sentence.
    person_index = 0
    for span in sentence_spans(source_text):
        if "운전" not in span.text:
            continue
        person_match = re.search(r"(피고인|피해자(?:\s+[A-Z가-힣0-9]+)?|원고(?:\s+[A-Z가-힣0-9]+)?)", span.text)
        clause_entities = list(dict.fromkeys(_mentions_in_clause(span.text, aliases)))
        if not person_match or len(clause_entities) != 1:
            continue
        person_index += 1
        person_id = f"person_{person_index}"
        person_label = person_match.group(1)
        ref = _evidence(source_file, source_text, span, "rule_driver_clause", 0.85)
        if ref.id not in {item.id for item in evidence}:
            evidence.append(ref)
        nodes.append(
            GraphNode(
                id=person_id,
                type=NodeType.PERSON,
                label=person_label,
                evidence_ids=[ref.id],
            )
        )
        edges.append(
            GraphEdge(
                id=stable_id("edge", person_id, clause_entities[0], "drives"),
                source=person_id,
                target=clause_entities[0],
                relation=RelationType.DRIVES,
                provenance=ProvenanceValue.OBSERVED,
                evidence_ids=[ref.id],
                confidence=0.85,
            )
        )

    # Link a role alias to a named vehicle when the same sentence explicitly introduces both.
    for span in sentence_spans(source_text):
        if "운전" not in span.text:
            continue
        clause_entities = list(dict.fromkeys(_mentions_in_clause(span.text, aliases)))
        if len(clause_entities) != 2:
            continue
        labels = {
            entity_id: next(node.label for node in nodes if node.id == entity_id)
            for entity_id in clause_entities
        }
        role_ids = [
            entity_id
            for entity_id, label in labels.items()
            if any(role in label for role in ("피고", "원고", "자신의"))
        ]
        model_ids = [entity_id for entity_id in clause_entities if entity_id not in role_ids]
        if len(role_ids) == 1 and len(model_ids) == 1:
            ref = _evidence(source_file, source_text, span, "rule_same_vehicle_clause", 0.85)
            if ref.id not in {item.id for item in evidence}:
                evidence.append(ref)
            edges.append(
                GraphEdge(
                    id=stable_id("edge", role_ids[0], model_ids[0], "same_as"),
                    source=role_ids[0],
                    target=model_ids[0],
                    relation=RelationType.SAME_AS,
                    provenance=ProvenanceValue.OBSERVED,
                    evidence_ids=[ref.id],
                    confidence=0.85,
                )
            )

    event_nodes = sorted(
        (node for node in nodes if node.type == NodeType.EVENT),
        key=lambda node: (int(node.attributes.get("source_start", 10**12)), node.id),
    )
    for previous, current in zip(event_nodes, event_nodes[1:]):
        shared_evidence = previous.evidence_ids + current.evidence_ids
        edges.append(
            GraphEdge(
                id=stable_id("edge", previous.id, current.id, "precedes"),
                source=previous.id,
                target=current.id,
                relation=RelationType.PRECEDES,
                provenance=ProvenanceValue.OBSERVED,
                evidence_ids=shared_evidence,
                confidence=0.8,
            )
        )

    collision_edges = [edge for edge in edges if edge.relation == RelationType.COLLIDES_WITH]
    vehicle_nodes = [node for node in nodes if node.type == NodeType.VEHICLE]
    unresolved: list[str] = []
    if len(vehicle_nodes) < 2:
        unresolved.append("two_vehicle_entities")
    if not collision_edges:
        unresolved.extend(["collision_agent", "collision_target"])

    graph = EvidenceGraphV1(
        scenario_id=scenario_id,
        candidate_id=candidate_id,
        source_file=source_file,
        source_text_sha256=sha256_text(source_text),
        nodes=nodes,
        edges=edges,
        evidence=evidence,
        critical_unresolved=list(dict.fromkeys(unresolved)),
        review_required=review_required,
    )
    if graph.critical_unresolved and resolver is not None:
        try:
            resolved = resolver.resolve(graph, source_text)
            if resolved:
                _append_resolver_relation(graph, source_text, resolved, resolver.name)
        except Exception as exc:
            graph.review_required.append(f"resolver failure: {exc}")

    graph.review_required.extend(validate_evidence_spans(graph, source_text))
    return EvidenceGraphV1.model_validate(graph.model_dump())
