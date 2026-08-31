from __future__ import annotations

try:
    from enum import StrEnum
except ImportError:  # Python 3.8 ChatScene compatibility profile
    from enum import Enum

    class StrEnum(str, Enum):
        def __str__(self) -> str:
            return self.value
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProvenanceValue(StrEnum):
    OBSERVED = "observed"
    INFERRED = "inferred"
    DEFAULTED = "defaulted"
    UNKNOWN = "unknown"


class LegalStatus(StrEnum):
    ACCEPTED_FACT = "accepted_fact"
    PARTY_CLAIM = "party_claim"
    COURT_REASONING = "court_reasoning"
    UNKNOWN = "unknown"


class NodeType(StrEnum):
    JUDGMENT = "judgment"
    EVENT = "event"
    VEHICLE = "vehicle"
    PERSON = "person"
    ROAD = "road"
    SIGNAL = "signal"
    EVIDENCE = "evidence"


class RelationType(StrEnum):
    DRIVES = "drives"
    COLLIDES_WITH = "collides_with"
    SAME_AS = "same_as"
    PRECEDES = "precedes"
    SUPPORTED_BY = "supported_by"
    LEGAL_STATUS = "legal_status"


class ContractStatus(StrEnum):
    READY = "ready"
    NEEDS_DEFAULTS = "needs_defaults"
    NEEDS_REVIEW = "needs_review"
    BLOCKED = "blocked"


class ExecutionStatus(StrEnum):
    NOT_EXECUTED = "not_executed"
    COMPILED = "compiled"
    RAN = "ran"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"


class EvidenceRef(StrictModel):
    id: str
    source_file: str
    quote: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    source_text_sha256: str
    extractor: str
    confidence: float = Field(ge=0, le=1)
    provenance: ProvenanceValue
    legal_status: LegalStatus
    supported: bool

    @model_validator(mode="after")
    def check_offsets(self) -> "EvidenceRef":
        if self.end < self.start:
            raise ValueError("evidence end must be >= start")
        if self.end - self.start != len(self.quote):
            raise ValueError("evidence offsets must match quote length")
        return self


class GraphNode(StrictModel):
    id: str
    type: NodeType
    label: str
    attributes: dict[str, Any] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)


class GraphEdge(StrictModel):
    id: str
    source: str
    target: str
    relation: RelationType
    provenance: ProvenanceValue
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    supported: bool = True


class EvidenceGraphV1(StrictModel):
    version: Literal["1.0"] = "1.0"
    scenario_id: str
    candidate_id: int
    source_file: str
    source_text_sha256: str
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    evidence: list[EvidenceRef]
    critical_unresolved: list[str] = Field(default_factory=list)
    review_required: list[str] = Field(default_factory=list)


class FieldValue(StrictModel):
    value: Any | None = None
    provenance: ProvenanceValue
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0, le=1)


class ScenarioActor(StrictModel):
    id: str
    role: Literal["ego", "target", "other"]
    vehicle_type: FieldValue
    blueprint: FieldValue
    lane_position: FieldValue
    initial_speed_mps: FieldValue


class MapBinding(StrictModel):
    archetype: FieldValue
    carla_map: FieldValue


class ScenarioEvent(StrictModel):
    id: str
    kind: str
    actor_id: str
    target_id: str | None = None
    order: int = Field(ge=0)
    description: FieldValue


class CollisionConstraint(StrictModel):
    actor_id: str
    target_id: str
    required: bool = True
    provenance: ProvenanceValue
    evidence_ids: list[str] = Field(default_factory=list)


class SensorPlan(StrictModel):
    collision: bool = True
    rgb: bool = True
    semantic_segmentation: bool = False
    telemetry_hz: int = Field(default=20, gt=0)


class ScenarioContractV1(StrictModel):
    version: Literal["1.0"] = "1.0"
    scenario_id: str
    candidate_id: int
    source_graph_sha256: str
    readiness_tier: str
    status: ContractStatus
    actors: list[ScenarioActor]
    map_binding: MapBinding
    topology: FieldValue = Field(
        default_factory=lambda: FieldValue(
            value="unknown",
            provenance=ProvenanceValue.UNKNOWN,
            confidence=0.0,
        )
    )
    maneuver_by_actor: dict[str, FieldValue] = Field(default_factory=dict)
    event_sequence: list[ScenarioEvent]
    collision_constraints: list[CollisionConstraint]
    sensors: SensorPlan
    seed: int
    fixed_delta_seconds: float = Field(default=0.05, gt=0)
    duration_seconds: float = Field(default=20.0, gt=0)
    policy_version: str
    immutable_paths: list[str] = Field(default_factory=list)
    review_issues: list[str] = Field(default_factory=list)


class ConstraintResult(StrictModel):
    name: str
    passed: bool | None
    expected: Any | None = None
    observed: Any | None = None
    reason: str | None = None


class ActorState(StrictModel):
    frame: int
    actor_id: str
    timestamp_seconds: float
    location: dict[str, float]
    rotation: dict[str, float]
    speed_mps: float
    control: dict[str, float | bool]


class CollisionEvent(StrictModel):
    frame: int
    actor_id: str
    other_actor_id: str | None
    impulse: dict[str, float]


class SimulationResultV1(StrictModel):
    version: Literal["1.0"] = "1.0"
    scenario_id: str
    backend: str
    executed: bool
    status: ExecutionStatus
    actor_states: list[ActorState] | None = None
    collisions: list[CollisionEvent] | None = None
    minimum_ttc_seconds: float | None = None
    constraint_results: list[ConstraintResult]
    keyframes: list[str] | None = None
    logs: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def protect_not_executed(self) -> "SimulationResultV1":
        if not self.executed:
            if self.status != ExecutionStatus.NOT_EXECUTED:
                raise ValueError("non-executed result must use not_executed status")
            if any(
                value is not None
                for value in (
                    self.actor_states,
                    self.collisions,
                    self.minimum_ttc_seconds,
                    self.keyframes,
                )
            ):
                raise ValueError("simulation metrics must be null when not executed")
        return self


class EvaluationFailure(StrictModel):
    attribute: str
    expected: Any | None = None
    observed: Any | None = None
    evidence: str | None = None
    repair_instruction: str | None = None


class RepairInstruction(StrictModel):
    path: str
    value: Any
    reason: str | None = None


class EvaluationReport(StrictModel):
    scenario_id: str
    evaluator: str
    passed: bool | None
    failures: list[EvaluationFailure] = Field(default_factory=list)
    repair_instructions: list[RepairInstruction] = Field(default_factory=list)
    manual_review: bool = False
    notes: list[str] = Field(default_factory=list)
