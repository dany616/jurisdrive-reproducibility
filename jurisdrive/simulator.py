from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Protocol

import yaml

from .io import sha256_file, write_json
from .models import (
    ConstraintResult,
    ContractStatus,
    EvidenceGraphV1,
    ExecutionStatus,
    ScenarioContractV1,
    SimulationResultV1,
)


class SimulatorBackend(Protocol):
    name: str

    def compile(self, contract: ScenarioContractV1) -> dict[str, Any]: ...

    def validate(self, compiled: dict[str, Any]) -> list[str]: ...

    def run(self, compiled: dict[str, Any]) -> SimulationResultV1: ...


def validate_contract(contract: ScenarioContractV1) -> list[str]:
    errors: list[str] = []
    actor_ids = [actor.id for actor in contract.actors]
    if len(actor_ids) != len(set(actor_ids)):
        errors.append("actor IDs must be unique")
    if len(actor_ids) < 2:
        errors.append("at least two actors are required")
    for actor in contract.actors:
        if not actor.blueprint.value:
            errors.append(f"{actor.id}: blueprint is missing")
    for constraint in contract.collision_constraints:
        if constraint.actor_id not in actor_ids or constraint.target_id not in actor_ids:
            errors.append("collision constraint references an unknown actor")
        if constraint.actor_id == constraint.target_id:
            errors.append("collision actor and target must differ")
    orders = [event.order for event in contract.event_sequence]
    if orders != sorted(orders) or len(orders) != len(set(orders)):
        errors.append("event order must be unique and ascending")
    if not contract.map_binding.carla_map.value:
        errors.append("CARLA map binding is missing")
    if not contract.sensors.collision:
        errors.append("collision sensor is required")
    if contract.readiness_tier.startswith("C_"):
        errors.append("Tier C is review-only and cannot be compiled")
    if contract.status in {ContractStatus.BLOCKED, ContractStatus.NEEDS_REVIEW}:
        errors.append(f"contract status is {contract.status.value}")
    return errors


def render_scenic(contract: ScenarioContractV1) -> str:
    map_name = str(contract.map_binding.carla_map.value)
    scenic_map_root = Path(
        os.environ.get(
            "JURISDRIVE_SCENIC_MAP_ROOT",
            "/opt/CARLA_0.9.13/CarlaUE4/Content/Carla/Maps/OpenDrive",
        )
    )
    # Scenic/ChatScene runs in the Linux compatibility environment even when
    # this source is rendered by the Windows-side annotation workspace.
    scenic_map_path = (scenic_map_root / f"{map_name}.xodr").as_posix()
    actor_lines = []
    for actor in contract.actors:
        scenic_name = "ego" if actor.role == "ego" else actor.id
        actor_lines.append(
            f'{scenic_name} = Car with blueprint "{actor.blueprint.value}"  '
            f'# contract actor: {actor.id}; position sampled on the bound road network'
        )
    return "\n".join(
        [
            "# Generated from ScenarioContractV1; do not treat defaults as legal facts.",
            f"param map = {json.dumps(scenic_map_path)}",
            f"param carla_map = {json.dumps(map_name)}",
            f"param seed = {contract.seed}",
            f"param fixed_delta_seconds = {contract.fixed_delta_seconds}",
            "model scenic.simulators.carla.model",
            *actor_lines,
            "",
            "# Collision/event constraints are enforced by the external runner and telemetry guard.",
        ]
    )


class DryRunBackend:
    name = "dry-run"

    def compile(self, contract: ScenarioContractV1) -> dict[str, Any]:
        errors = validate_contract(contract)
        return {
            "contract": contract,
            "scenic_source": render_scenic(contract),
            "compile_valid": not errors,
            "validation_errors": errors,
        }

    def validate(self, compiled: dict[str, Any]) -> list[str]:
        return list(compiled["validation_errors"])

    def run(self, compiled: dict[str, Any]) -> SimulationResultV1:
        contract: ScenarioContractV1 = compiled["contract"]
        errors = self.validate(compiled)
        return SimulationResultV1(
            scenario_id=contract.scenario_id,
            backend=self.name,
            executed=False,
            status=ExecutionStatus.NOT_EXECUTED,
            actor_states=None,
            collisions=None,
            minimum_ttc_seconds=None,
            constraint_results=[
                ConstraintResult(
                    name="compile_valid",
                    passed=not errors,
                    expected=True,
                    observed=not errors,
                    reason="; ".join(errors) if errors else None,
                )
            ],
            keyframes=None,
            logs=["Dry-run validation only; CARLA was not started."],
            errors=errors,
        )


def telemetry_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "JurisDriveTelemetryFrameV1",
        "type": "object",
        "required": ["frame", "timestamp_seconds", "actors"],
        "properties": {
            "frame": {"type": "integer"},
            "timestamp_seconds": {"type": "number"},
            "actors": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["actor_id", "location", "rotation", "speed_mps", "control"],
                },
            },
            "collision_events": {"type": "array"},
            "minimum_ttc_seconds": {"type": ["number", "null"]},
        },
        "additionalProperties": False,
    }


def write_bundle(
    output_dir: Path,
    graph: EvidenceGraphV1,
    contract: ScenarioContractV1,
    compiled: dict[str, Any],
    result: SimulationResultV1,
) -> Path:
    bundle = output_dir / contract.scenario_id
    bundle.mkdir(parents=True, exist_ok=True)
    write_json(bundle / "evidence_graph.json", graph)
    write_json(bundle / "contract.json", contract)
    write_json(bundle / "dry_run_report.json", result)
    write_json(bundle / "telemetry_schema.json", telemetry_schema())
    (bundle / "scenario.scenic").write_text(compiled["scenic_source"] + "\n", encoding="utf-8")
    run_config = {
        "backend": "carla",
        "host": "127.0.0.1",
        "port": 2000,
        "traffic_manager_port": 8000,
        "synchronous_mode": True,
        "fixed_delta_seconds": contract.fixed_delta_seconds,
        "duration_seconds": contract.duration_seconds,
        "seed": contract.seed,
        "compile_valid": compiled["compile_valid"],
    }
    (bundle / "run_config.yaml").write_text(
        yaml.safe_dump(run_config, sort_keys=False),
        encoding="utf-8",
    )
    checksum_files = sorted(path for path in bundle.iterdir() if path.name != "checksums.sha256")
    checksum_lines = [f"{sha256_file(path)}  {path.name}" for path in checksum_files]
    (bundle / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    return bundle


def minimum_ttc(
    ego_position: tuple[float, float],
    ego_velocity: tuple[float, float],
    target_position: tuple[float, float],
    target_velocity: tuple[float, float],
) -> float | None:
    relative_position = (
        target_position[0] - ego_position[0],
        target_position[1] - ego_position[1],
    )
    relative_velocity = (
        ego_velocity[0] - target_velocity[0],
        ego_velocity[1] - target_velocity[1],
    )
    closing = relative_position[0] * relative_velocity[0] + relative_position[1] * relative_velocity[1]
    speed_squared = relative_velocity[0] ** 2 + relative_velocity[1] ** 2
    if closing <= 0 or speed_squared <= 1e-9:
        return None
    ttc = closing / speed_squared
    closest = (
        relative_position[0] - relative_velocity[0] * ttc,
        relative_position[1] - relative_velocity[1] * ttc,
    )
    return ttc if math.hypot(*closest) < 3.0 else None


class CarlaBackend:
    """External-only CARLA backend with lazy dependency loading."""

    name = "carla"

    def __init__(self, bundle_dir: Path, host: str = "127.0.0.1", port: int = 2000) -> None:
        self.bundle_dir = bundle_dir
        self.host = host
        self.port = port

    def compile(self, contract: ScenarioContractV1) -> dict[str, Any]:
        errors = validate_contract(contract)
        return {"contract": contract, "compile_valid": not errors, "validation_errors": errors}

    def validate(self, compiled: dict[str, Any]) -> list[str]:
        return list(compiled["validation_errors"])

    def run(self, compiled: dict[str, Any]) -> SimulationResultV1:
        if self.validate(compiled):
            raise ValueError("Cannot execute an invalid contract")
        try:
            import carla  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "CARLA Python API is external-only; use requirements-carla.txt in Python 3.10"
            ) from exc

        contract: ScenarioContractV1 = compiled["contract"]
        execution_profile = os.environ.get(
            "JURISDRIVE_EXECUTION_PROFILE", "contract_collision"
        )
        client = carla.Client(self.host, self.port)
        client.set_timeout(20.0)
        requested_map = str(contract.map_binding.carla_map.value)
        world_acquire_started = time.perf_counter()
        reuse_loaded_world = os.environ.get("JURISDRIVE_REUSE_LOADED_WORLD", "0") == "1"
        world_acquire_mode = "loaded"
        if reuse_loaded_world:
            current_world = client.get_world()
            current_map = str(current_world.get_map().name).split("/")[-1]
            if current_map == requested_map.split("/")[-1]:
                world = current_world
                world_acquire_mode = "reused"
            else:
                world = client.load_world(requested_map)
        else:
            world = client.load_world(requested_map)
        world_acquire_seconds = time.perf_counter() - world_acquire_started
        original_settings = world.get_settings()
        traffic_manager = None
        actors: list[Any] = []
        actor_ids: list[int] = []
        sensor_actors: list[Any] = []
        states: list[dict[str, Any]] = []
        collisions: list[dict[str, Any]] = []
        keyframes: list[str] = []
        try:
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = contract.fixed_delta_seconds
            world.apply_settings(settings)
            blueprint_library = world.get_blueprint_library()
            spawn_points = world.get_map().get_spawn_points()
            if len(spawn_points) < len(contract.actors):
                raise RuntimeError("Not enough CARLA spawn points")
            actor_by_contract_id: dict[str, Any] = {}
            collision_pair_ids: tuple[str, str] | None = None
            pair_transforms: dict[str, Any] = {}
            velocity_vectors: dict[str, Any] = {}
            topology_spawn_observation: dict[str, Any] | None = None
            if execution_profile == "contract_collision":
                required_constraint = next(
                    (item for item in contract.collision_constraints if item.required),
                    None,
                )
                if required_constraint is None:
                    raise RuntimeError("contract_collision profile requires a collision constraint")
                collision_pair_ids = (
                    required_constraint.actor_id,
                    required_constraint.target_id,
                )
                base_transform = spawn_points[0]
                forward = base_transform.get_forward_vector()
                actor_transform = carla.Transform(
                    carla.Location(
                        x=base_transform.location.x - forward.x * 12.0,
                        y=base_transform.location.y - forward.y * 12.0,
                        z=base_transform.location.z + 0.5,
                    ),
                    carla.Rotation(
                        pitch=base_transform.rotation.pitch,
                        yaw=base_transform.rotation.yaw,
                        roll=base_transform.rotation.roll,
                    ),
                )
                target_transform = carla.Transform(
                    carla.Location(
                        x=base_transform.location.x + forward.x * 12.0,
                        y=base_transform.location.y + forward.y * 12.0,
                        z=base_transform.location.z + 0.5,
                    ),
                    carla.Rotation(
                        pitch=base_transform.rotation.pitch,
                        yaw=base_transform.rotation.yaw + 180.0,
                        roll=base_transform.rotation.roll,
                    ),
                )
                pair_transforms = {
                    collision_pair_ids[0]: actor_transform,
                    collision_pair_ids[1]: target_transform,
                }
            elif execution_profile == "topology_contract":
                required_constraint = next(
                    (item for item in contract.collision_constraints if item.required),
                    None,
                )
                if required_constraint is None:
                    raise RuntimeError("topology_contract profile requires a collision constraint")
                collision_pair_ids = (
                    required_constraint.actor_id,
                    required_constraint.target_id,
                )
                topology = str(contract.topology.value or "unknown")
                if topology not in {
                    "rear_end",
                    "intersection_crossing_turning",
                    "lane_change_side_swipe",
                    "head_on_centerline_intrusion",
                }:
                    raise RuntimeError(f"unsupported contract topology: {topology}")
                base_transform = spawn_points[0]
                forward = base_transform.get_forward_vector()
                right = carla.Vector3D(x=-forward.y, y=forward.x, z=0.0)

                def profiled_transform(
                    along: float, lateral: float, yaw_delta: float = 0.0
                ) -> Any:
                    return carla.Transform(
                        carla.Location(
                            x=base_transform.location.x + forward.x * along + right.x * lateral,
                            y=base_transform.location.y + forward.y * along + right.y * lateral,
                            z=base_transform.location.z + 0.5,
                        ),
                        carla.Rotation(
                            pitch=base_transform.rotation.pitch,
                            yaw=base_transform.rotation.yaw + yaw_delta,
                            roll=base_transform.rotation.roll,
                        ),
                    )

                actor_id, target_id = collision_pair_ids
                actor_spec = next(item for item in contract.actors if item.id == actor_id)
                target_spec = next(item for item in contract.actors if item.id == target_id)
                actor_speed = float(actor_spec.initial_speed_mps.value or 0.0)
                target_speed = float(target_spec.initial_speed_mps.value or 0.0)
                if topology == "rear_end":
                    pair_transforms = {
                        actor_id: profiled_transform(-18.0, 0.0),
                        target_id: profiled_transform(0.0, 0.0),
                    }
                    velocity_vectors = {
                        actor_id: carla.Vector3D(forward.x * actor_speed, forward.y * actor_speed, 0.0),
                        target_id: carla.Vector3D(forward.x * target_speed, forward.y * target_speed, 0.0),
                    }
                elif topology == "head_on_centerline_intrusion":
                    pair_transforms = {
                        actor_id: profiled_transform(-14.0, 0.0),
                        target_id: profiled_transform(14.0, 0.0, 180.0),
                    }
                    velocity_vectors = {
                        actor_id: carla.Vector3D(forward.x * actor_speed, forward.y * actor_speed, 0.0),
                        target_id: carla.Vector3D(-forward.x * target_speed, -forward.y * target_speed, 0.0),
                    }
                elif topology == "intersection_crossing_turning":
                    pair_transforms = {
                        actor_id: profiled_transform(-10.0, 0.0),
                        target_id: profiled_transform(0.0, -10.0, 90.0),
                    }
                    velocity_vectors = {
                        actor_id: carla.Vector3D(forward.x * actor_speed, forward.y * actor_speed, 0.0),
                        target_id: carla.Vector3D(right.x * target_speed, right.y * target_speed, 0.0),
                    }
                else:  # lane_change_side_swipe
                    pair_transforms = {
                        actor_id: profiled_transform(-6.0, 3.5),
                        target_id: profiled_transform(0.0, 0.0),
                    }
                    velocity_vectors = {
                        actor_id: carla.Vector3D(
                            forward.x * actor_speed - right.x * 1.5,
                            forward.y * actor_speed - right.y * 1.5,
                            0.0,
                        ),
                        target_id: carla.Vector3D(forward.x * target_speed, forward.y * target_speed, 0.0),
                    }
                carla_map = world.get_map()
                lane_checks = {}
                for contract_id, transform in pair_transforms.items():
                    waypoint = carla_map.get_waypoint(
                        transform.location,
                        project_to_road=True,
                        lane_type=carla.LaneType.Driving,
                    )
                    projected_distance = transform.location.distance(waypoint.transform.location)
                    lane_checks[contract_id] = {
                        "road_id": waypoint.road_id,
                        "lane_id": waypoint.lane_id,
                        "projected_distance_m": projected_distance,
                        "within_driving_lane_tolerance": projected_distance <= 4.0,
                    }
                topology_spawn_observation = {
                    "topology": topology,
                    "lane_checks": lane_checks,
                    "all_within_driving_lane_tolerance": all(
                        value["within_driving_lane_tolerance"] for value in lane_checks.values()
                    ),
                }
            elif execution_profile == "traffic_manager":
                traffic_manager = client.get_trafficmanager(8000)
                traffic_manager.set_synchronous_mode(True)
                traffic_manager.set_random_device_seed(contract.seed)
            else:
                raise ValueError(f"unknown execution profile: {execution_profile}")

            for index, actor_spec in enumerate(contract.actors):
                blueprint = blueprint_library.find(str(actor_spec.blueprint.value))
                spawn_transform = pair_transforms.get(
                    actor_spec.id,
                    spawn_points[(index + 10) % len(spawn_points)],
                )
                actor = world.try_spawn_actor(blueprint, spawn_transform)
                if actor is None:
                    raise RuntimeError(f"Failed to spawn {actor_spec.id}")
                if traffic_manager is not None:
                    actor.set_autopilot(True, traffic_manager.get_port())
                else:
                    actor.apply_control(carla.VehicleControl(brake=1.0))
                actor_by_contract_id[actor_spec.id] = actor
                actors.append(actor)
                actor_ids.append(actor.id)

            carla_to_contract = {
                actor.id: contract_id for contract_id, actor in actor_by_contract_id.items()
            }
            collision_blueprint = blueprint_library.find("sensor.other.collision")
            for contract_id, actor in actor_by_contract_id.items():
                sensor = world.spawn_actor(collision_blueprint, carla.Transform(), attach_to=actor)
                actors.append(sensor)
                actor_ids.append(sensor.id)
                sensor_actors.append(sensor)

                def on_collision(event: Any, source_id: str = contract_id) -> None:
                    impulse = event.normal_impulse
                    collisions.append(
                        {
                            "frame": event.frame,
                            "actor_id": source_id,
                            "other_actor_id": carla_to_contract.get(event.other_actor.id),
                            "impulse": {"x": impulse.x, "y": impulse.y, "z": impulse.z},
                        }
                    )

                sensor.listen(on_collision)

            ego_spec = next(
                (actor for actor in contract.actors if actor.role == "ego"),
                contract.actors[0],
            )
            capture_camera = os.environ.get("JURISDRIVE_CAPTURE_CAMERA", "1") != "0"
            keyframe_dir = self.bundle_dir / "keyframes"
            keyframe_dir.mkdir(parents=True, exist_ok=True)
            captured_frames: list[tuple[int, str]] = []

            def on_image(image: Any) -> None:
                path = keyframe_dir / f"frame_{image.frame:08d}.png"
                image.save_to_disk(str(path))
                captured_frames.append((image.frame, str(path.relative_to(self.bundle_dir))))

            if capture_camera:
                camera_blueprint = blueprint_library.find("sensor.camera.rgb")
                camera_blueprint.set_attribute("image_size_x", "800")
                camera_blueprint.set_attribute("image_size_y", "450")
                camera_blueprint.set_attribute(
                    "sensor_tick", str(contract.fixed_delta_seconds * 10)
                )
                camera = world.spawn_actor(
                    camera_blueprint,
                    carla.Transform(
                        carla.Location(x=-8.0, z=4.0),
                        carla.Rotation(pitch=-15.0),
                    ),
                    attach_to=actor_by_contract_id[ego_spec.id],
                )
                actors.append(camera)
                actor_ids.append(camera.id)
                sensor_actors.append(camera)
                camera.listen(on_image)

            def apply_contract_collision_motion(pair_collision_observed: bool) -> None:
                commands = []
                for actor_spec in contract.actors:
                    actor = actor_by_contract_id[actor_spec.id]
                    if actor_spec.id in (collision_pair_ids or ()) and not pair_collision_observed:
                        forward = actor.get_transform().get_forward_vector()
                        speed = float(actor_spec.initial_speed_mps.value or 0.0)
                        target_velocity = velocity_vectors.get(
                            actor_spec.id,
                            carla.Vector3D(
                                x=forward.x * speed,
                                y=forward.y * speed,
                                z=0.0,
                            ),
                        )
                        commands.extend(
                            (
                                carla.command.ApplyVehicleControl(
                                    actor.id, carla.VehicleControl()
                                ),
                                carla.command.ApplyTargetVelocity(
                                    actor.id,
                                    target_velocity,
                                ),
                            )
                        )
                    else:
                        commands.append(
                            carla.command.ApplyVehicleControl(
                                actor.id, carla.VehicleControl(brake=1.0)
                            )
                        )
                        if pair_collision_observed:
                            commands.append(
                                carla.command.ApplyTargetVelocity(
                                    actor.id, carla.Vector3D()
                                )
                            )
                responses = client.apply_batch_sync(commands, False)
                errors = [response.error for response in responses if response.has_error()]
                if errors:
                    raise RuntimeError("CARLA motion batch failed: " + "; ".join(errors))

            if collision_pair_ids is not None:
                # Discard one setup tick so queued spawn/control RPCs cannot leak into
                # the first measured frame. Every measured tick then starts from a
                # synchronously acknowledged command batch.
                apply_contract_collision_motion(False)
                world.tick()

            planned_frames = int(contract.duration_seconds / contract.fixed_delta_seconds)
            frame_limit = int(os.environ.get("JURISDRIVE_FRAME_LIMIT", planned_frames))
            frames = min(planned_frames, frame_limit)
            minimum_seen: float | None = None
            telemetry_path = self.bundle_dir / "telemetry.jsonl"
            with telemetry_path.open("w", encoding="utf-8") as telemetry:
                for _ in range(frames):
                    if collision_pair_ids is not None:
                        pair_collision_observed = any(
                            {
                                item["actor_id"],
                                item["other_actor_id"],
                            }
                            == set(collision_pair_ids)
                            for item in collisions
                        )
                        apply_contract_collision_motion(pair_collision_observed)
                    frame = world.tick()
                    snapshot = world.get_snapshot()
                    frame_states = []
                    for contract_id, actor in actor_by_contract_id.items():
                        transform = actor.get_transform()
                        velocity = actor.get_velocity()
                        control = actor.get_control()
                        state = {
                            "frame": frame,
                            "actor_id": contract_id,
                            "timestamp_seconds": snapshot.timestamp.elapsed_seconds,
                            "location": {
                                "x": transform.location.x,
                                "y": transform.location.y,
                                "z": transform.location.z,
                            },
                            "rotation": {
                                "pitch": transform.rotation.pitch,
                                "yaw": transform.rotation.yaw,
                                "roll": transform.rotation.roll,
                            },
                            "speed_mps": math.sqrt(
                                velocity.x**2 + velocity.y**2 + velocity.z**2
                            ),
                            "control": {
                                "throttle": control.throttle,
                                "steer": control.steer,
                                "brake": control.brake,
                                "hand_brake": control.hand_brake,
                            },
                        }
                        states.append(state)
                        frame_states.append(state)
                    for constraint in contract.collision_constraints:
                        ego_actor = actor_by_contract_id[constraint.actor_id]
                        target_actor = actor_by_contract_id[constraint.target_id]
                        ego_location = ego_actor.get_location()
                        target_location = target_actor.get_location()
                        ego_velocity = ego_actor.get_velocity()
                        target_velocity = target_actor.get_velocity()
                        current_ttc = minimum_ttc(
                            (ego_location.x, ego_location.y),
                            (ego_velocity.x, ego_velocity.y),
                            (target_location.x, target_location.y),
                            (target_velocity.x, target_velocity.y),
                        )
                        if current_ttc is not None:
                            minimum_seen = (
                                current_ttc
                                if minimum_seen is None
                                else min(minimum_seen, current_ttc)
                            )
                    telemetry.write(
                        json.dumps(
                            {
                                "frame": frame,
                                "timestamp_seconds": snapshot.timestamp.elapsed_seconds,
                                "actors": frame_states,
                                "collision_events": [
                                    item for item in collisions if item["frame"] == frame
                                ],
                                "minimum_ttc_seconds": minimum_seen,
                            }
                        )
                        + "\n"
                    )
            for sensor in sensor_actors:
                try:
                    sensor.stop()
                except RuntimeError:
                    pass
            sensor_actors.clear()
            if collisions and captured_frames:
                collision_frames = [collision["frame"] for collision in collisions]
                peak_collision = max(
                    collisions,
                    key=lambda collision: sum(
                        float(collision["impulse"][axis]) ** 2
                        for axis in ("x", "y", "z")
                    ),
                )
                target_frames = (
                    min(collision_frames) - 20,
                    peak_collision["frame"],
                    max(collision_frames) + 20,
                )
                for target_frame in target_frames:
                    _, path = min(
                        captured_frames, key=lambda item: abs(item[0] - target_frame)
                    )
                    if path not in keyframes:
                        keyframes.append(path)
            if not keyframes and captured_frames:
                representative_indices = (0, len(captured_frames) // 2, len(captured_frames) - 1)
                keyframes = list(
                    dict.fromkeys(captured_frames[index][1] for index in representative_indices)
                )
            constraint_results = []
            for constraint in contract.collision_constraints:
                matched = any(
                    {
                        collision["actor_id"],
                        collision["other_actor_id"],
                    }
                    == {constraint.actor_id, constraint.target_id}
                    for collision in collisions
                )
                constraint_results.append(
                    {
                        "name": "collision_target",
                        "passed": matched,
                        "expected": {
                            "actor_id": constraint.actor_id,
                            "target_id": constraint.target_id,
                        },
                        "observed": collisions,
                        "reason": None if matched else "required actor-target collision was not observed",
                    }
                )
            if execution_profile == "topology_contract":
                topology_passed = bool(
                    topology_spawn_observation
                    and topology_spawn_observation["all_within_driving_lane_tolerance"]
                )
                constraint_results.append(
                    {
                        "name": "lane_topology_valid",
                        "passed": topology_passed,
                        "expected": str(contract.topology.value),
                        "observed": topology_spawn_observation,
                        "reason": None
                        if topology_passed
                        else "one or more profiled spawn points exceeded the driving-lane tolerance",
                    }
                )
                first_state_frame = min((state["frame"] for state in states), default=None)
                first_collision_frame = min(
                    (collision["frame"] for collision in collisions), default=None
                )
                event_order_passed = (
                    first_state_frame is not None
                    and first_collision_frame is not None
                    and first_collision_frame >= first_state_frame
                )
                constraint_results.append(
                    {
                        "name": "event_order_valid",
                        "passed": event_order_passed,
                        "expected": "initial state precedes required collision",
                        "observed": {
                            "first_state_frame": first_state_frame,
                            "first_collision_frame": first_collision_frame,
                        },
                        "reason": None if event_order_passed else "required collision was absent or predated telemetry",
                    }
                )
            return SimulationResultV1.model_validate(
                {
                    "scenario_id": contract.scenario_id,
                    "backend": self.name,
                    "executed": True,
                    "status": (
                        ExecutionStatus.PASSED
                        if all(item["passed"] for item in constraint_results)
                        else ExecutionStatus.FAILED
                    ),
                    "actor_states": states,
                    "collisions": collisions,
                    "minimum_ttc_seconds": minimum_seen,
                    "constraint_results": constraint_results,
                    "keyframes": keyframes,
                    "logs": [
                        str(telemetry_path.relative_to(self.bundle_dir)),
                        f"camera_capture={str(capture_camera).lower()}",
                        f"executed_frames={frames}/{planned_frames}",
                        f"execution_profile={execution_profile}",
                        f"world_acquire_mode={world_acquire_mode}",
                        f"world_acquire_seconds={world_acquire_seconds:.6f}",
                    ],
                    "errors": [],
                }
            )
        finally:
            for sensor in sensor_actors:
                try:
                    sensor.stop()
                except RuntimeError:
                    pass
            try:
                if actor_ids:
                    client.apply_batch_sync(
                        [carla.command.DestroyActor(actor_id) for actor_id in actor_ids],
                        True,
                    )
                if traffic_manager is not None:
                    traffic_manager.set_synchronous_mode(False)
                world.apply_settings(original_settings)
            except RuntimeError:
                pass
