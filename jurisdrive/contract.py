from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .io import REPO_ROOT, sha256_text
from .models import (
    CollisionConstraint,
    ContractStatus,
    EvidenceGraphV1,
    FieldValue,
    MapBinding,
    NodeType,
    ProvenanceValue,
    RelationType,
    ScenarioActor,
    ScenarioContractV1,
    ScenarioEvent,
    SensorPlan,
)

DEFAULT_POLICY = REPO_ROOT / "configs" / "scenario_defaults.yaml"

SUPPORTED_TOPOLOGIES = {
    "rear_end",
    "intersection_crossing_turning",
    "lane_change_side_swipe",
    "head_on_centerline_intrusion",
}


def load_policy(path: Path = DEFAULT_POLICY) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid policy: {path}")
    return value


def _blueprint(label: str, policy: dict[str, Any]) -> str:
    lowered = label.lower()
    fallbacks = policy["blueprint_fallbacks"]
    if any(token in lowered for token in ("트럭", "화물")):
        return str(fallbacks["truck"])
    if "버스" in lowered:
        return str(fallbacks["bus"])
    if any(token in lowered for token in ("suv", "스포티지", "쏘렌토")):
        return str(fallbacks["suv"])
    return str(fallbacks["generic"])


def _map_archetype(source_text: str) -> str:
    if any(token in source_text for token in ("교차로", "사거리", "삼거리")):
        return "intersection"
    if any(token in source_text for token in ("차로", "차선변경", "진로변경")):
        return "multi_lane"
    if any(token in source_text for token in ("직진", "도로")):
        return "straight_road"
    return "unknown"


def compile_contract(
    graph: EvidenceGraphV1,
    *,
    source_text: str,
    readiness_tier: str,
    policy: dict[str, Any] | None = None,
) -> ScenarioContractV1:
    policy = policy or load_policy()
    vehicle_nodes = [node for node in graph.nodes if node.type == NodeType.VEHICLE]
    collisions = [
        edge
        for edge in graph.edges
        if edge.relation == RelationType.COLLIDES_WITH and edge.supported
    ]
    collision = collisions[0] if collisions else None
    actors: list[ScenarioActor] = []
    immutable_paths: list[str] = []
    for node in vehicle_nodes:
        if collision and node.id == collision.source:
            role = "ego"
        elif collision and node.id == collision.target:
            role = "target"
        else:
            role = "other"
        actors.append(
            ScenarioActor(
                id=node.id,
                role=role,
                vehicle_type=FieldValue(
                    value=node.label,
                    provenance=ProvenanceValue.OBSERVED
                    if node.evidence_ids
                    else ProvenanceValue.INFERRED,
                    evidence_ids=node.evidence_ids,
                    confidence=0.8,
                ),
                blueprint=FieldValue(
                    value=_blueprint(node.label, policy),
                    provenance=ProvenanceValue.DEFAULTED,
                    confidence=0.6,
                ),
                lane_position=FieldValue(
                    value="lane_relative",
                    provenance=ProvenanceValue.DEFAULTED,
                    confidence=0.5,
                ),
                initial_speed_mps=FieldValue(
                    value=8.0 if role == "ego" else 3.0,
                    provenance=ProvenanceValue.DEFAULTED,
                    confidence=0.4,
                ),
            )
        )
        if node.evidence_ids:
            immutable_paths.append(f"actors.{node.id}.vehicle_type")

    archetype = _map_archetype(source_text)
    map_name = str(policy["map_catalog"][archetype])
    events: list[ScenarioEvent] = []
    constraints: list[CollisionConstraint] = []
    if collision:
        graph_events = sorted(
            (node for node in graph.nodes if node.type == NodeType.EVENT),
            key=lambda node: (
                int(node.attributes.get("source_start", 10**12)),
                node.id,
            ),
        )
        for order, node in enumerate(graph_events, start=1):
            is_collision = node.label == "vehicle_collision"
            events.append(
                ScenarioEvent(
                    id=f"event_{order}_{node.label}",
                    kind="collision" if is_collision else node.label,
                    actor_id=collision.source,
                    target_id=collision.target if is_collision else None,
                    order=order,
                    description=FieldValue(
                        value=(
                            "required collision between grounded actors"
                            if is_collision
                            else f"grounded pre-collision event: {node.label}"
                        ),
                        provenance=ProvenanceValue.OBSERVED,
                        evidence_ids=list(node.evidence_ids),
                        confidence=0.9 if is_collision else 0.8,
                    ),
                )
            )
            if node.evidence_ids:
                immutable_paths.append(f"event_sequence.{order - 1}.description")
        if not events:
            events.append(
                ScenarioEvent(
                    id="event_collision_1",
                    kind="collision",
                    actor_id=collision.source,
                    target_id=collision.target,
                    order=1,
                    description=FieldValue(
                        value="required collision between grounded actors",
                        provenance=collision.provenance,
                        evidence_ids=collision.evidence_ids,
                        confidence=collision.confidence,
                    ),
                )
            )
        constraints.append(
            CollisionConstraint(
                actor_id=collision.source,
                target_id=collision.target,
                provenance=collision.provenance,
                evidence_ids=collision.evidence_ids,
            )
        )
        if collision.provenance == ProvenanceValue.OBSERVED:
            immutable_paths.append("collision_constraints.0")

    issues = list(graph.critical_unresolved) + list(graph.review_required)
    collision_event_nodes = [
        node
        for node in graph.nodes
        if node.type == NodeType.EVENT and node.label == "vehicle_collision"
    ]
    if any(
        node.attributes.get("legal_status") == "party_claim"
        for node in collision_event_nodes
    ):
        issues.append("collision is grounded only in a party claim")
    if readiness_tier.startswith("C_"):
        issues.append("Tier C cannot be promoted automatically")
    if len(actors) < 2:
        issues.append("contract requires at least two vehicle actors")
    if not collision:
        issues.append("grounded collision agent/target is missing")

    if len(actors) < 2:
        status = ContractStatus.BLOCKED
    elif issues:
        status = ContractStatus.NEEDS_REVIEW
    else:
        status = ContractStatus.NEEDS_DEFAULTS

    simulation = policy["simulation"]
    sensors = policy["sensors"]
    graph_json = json.dumps(graph.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
    return ScenarioContractV1(
        scenario_id=graph.scenario_id,
        candidate_id=graph.candidate_id,
        source_graph_sha256=sha256_text(graph_json),
        readiness_tier=readiness_tier,
        status=status,
        actors=actors,
        map_binding=MapBinding(
            archetype=FieldValue(
                value=archetype,
                provenance=ProvenanceValue.INFERRED
                if archetype != "unknown"
                else ProvenanceValue.UNKNOWN,
                confidence=0.7 if archetype != "unknown" else 0.2,
            ),
            carla_map=FieldValue(
                value=map_name,
                provenance=ProvenanceValue.DEFAULTED,
                confidence=0.5,
            ),
        ),
        topology=FieldValue(
            value={
                "intersection": "intersection_crossing_turning",
                "multi_lane": "lane_change_side_swipe",
            }.get(archetype, "unknown"),
            provenance=(
                ProvenanceValue.INFERRED
                if archetype in {"intersection", "multi_lane"}
                else ProvenanceValue.UNKNOWN
            ),
            confidence=0.55 if archetype in {"intersection", "multi_lane"} else 0.0,
        ),
        event_sequence=events,
        collision_constraints=constraints,
        sensors=SensorPlan(
            collision=bool(sensors["collision"]),
            rgb=bool(sensors["rgb"]),
            semantic_segmentation=bool(sensors["semantic_segmentation"]),
            telemetry_hz=int(simulation["telemetry_hz"]),
        ),
        seed=int(simulation["seed"]) + graph.candidate_id,
        fixed_delta_seconds=float(simulation["fixed_delta_seconds"]),
        duration_seconds=float(simulation["duration_seconds"]),
        policy_version=str(policy["version"]),
        immutable_paths=sorted(set(immutable_paths)),
        review_issues=list(dict.fromkeys(issues)),
    )


def bind_topology_profile(
    contract: ScenarioContractV1,
    topology: str,
    *,
    evidence_ids: list[str] | None = None,
) -> ScenarioContractV1:
    """Bind an evidence-cued topology while keeping physical choices defaulted.

    The topology is an interpretation supported by the cited collision span.  CARLA
    speeds, lane placements, and maneuver controls remain explicit defaults rather
    than being promoted to legal facts.
    """

    if topology not in SUPPORTED_TOPOLOGIES:
        raise ValueError(f"unsupported RQ3 topology: {topology}")
    evidence_ids = list(evidence_ids or [])
    actors = [actor.model_copy(deep=True) for actor in contract.actors]
    ego = next((actor for actor in actors if actor.role == "ego"), None)
    target = next((actor for actor in actors if actor.role == "target"), None)
    if ego is None or target is None:
        return contract.model_copy(
            update={
                "topology": FieldValue(
                    value=topology,
                    provenance=ProvenanceValue.INFERRED,
                    evidence_ids=evidence_ids,
                    confidence=0.8,
                )
            }
        )

    speed_defaults = {
        "rear_end": (10.0, 0.0),
        "intersection_crossing_turning": (8.0, 8.0),
        "lane_change_side_swipe": (9.0, 7.0),
        "head_on_centerline_intrusion": (8.0, 8.0),
    }
    maneuver_defaults = {
        "rear_end": ("following_closure", "stopped_or_slow_lead"),
        "intersection_crossing_turning": ("crossing_approach", "turning_or_crossing_approach"),
        "lane_change_side_swipe": ("lane_change", "lane_keep"),
        "head_on_centerline_intrusion": ("centerline_intrusion", "opposing_approach"),
    }
    ego_speed, target_speed = speed_defaults[topology]
    ego.initial_speed_mps = FieldValue(
        value=ego_speed, provenance=ProvenanceValue.DEFAULTED, confidence=0.5
    )
    target.initial_speed_mps = FieldValue(
        value=target_speed, provenance=ProvenanceValue.DEFAULTED, confidence=0.5
    )
    ego_maneuver, target_maneuver = maneuver_defaults[topology]
    maneuvers = {
        ego.id: FieldValue(
            value=ego_maneuver, provenance=ProvenanceValue.DEFAULTED, confidence=0.5
        ),
        target.id: FieldValue(
            value=target_maneuver, provenance=ProvenanceValue.DEFAULTED, confidence=0.5
        ),
    }
    # Town03 is the bounded intersection map for the workstation protocol.  The
    # larger Town05 repeatedly hit UE4 render-fence crashes on this CARLA 0.9.13
    # build and is retained as a recorded failed map attempt, not silently retried.
    map_name = "Town03" if topology == "intersection_crossing_turning" else "Town04"
    archetype = {
        "rear_end": "straight_road",
        "intersection_crossing_turning": "intersection",
        "lane_change_side_swipe": "multi_lane",
        "head_on_centerline_intrusion": "straight_road",
    }[topology]
    return contract.model_copy(
        update={
            "actors": actors,
            "map_binding": MapBinding(
                archetype=FieldValue(
                    value=archetype,
                    provenance=ProvenanceValue.INFERRED,
                    evidence_ids=evidence_ids,
                    confidence=0.8,
                ),
                carla_map=FieldValue(
                    value=map_name,
                    provenance=ProvenanceValue.DEFAULTED,
                    confidence=0.6,
                ),
            ),
            "topology": FieldValue(
                value=topology,
                provenance=ProvenanceValue.INFERRED,
                evidence_ids=evidence_ids,
                confidence=0.8,
            ),
            "maneuver_by_actor": maneuvers,
        }
    )
