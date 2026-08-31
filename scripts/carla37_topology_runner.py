#!/usr/bin/env python3
"""Python-3.7-compatible CARLA 0.9.13 runner for RQ3 topology contracts.

This bridge intentionally uses only the standard library and the CARLA wheel.  It
does not import the main Pydantic-based JurisDrive package.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time


TOPOLOGIES = {
    "rear_end",
    "intersection_crossing_turning",
    "lane_change_side_swipe",
    "head_on_centerline_intrusion",
}


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def minimum_ttc(ego_position, ego_velocity, target_position, target_velocity):
    relative_position = (
        target_position[0] - ego_position[0],
        target_position[1] - ego_position[1],
    )
    relative_velocity = (
        ego_velocity[0] - target_velocity[0],
        ego_velocity[1] - target_velocity[1],
    )
    closing = (
        relative_position[0] * relative_velocity[0]
        + relative_position[1] * relative_velocity[1]
    )
    speed_squared = relative_velocity[0] ** 2 + relative_velocity[1] ** 2
    if closing <= 0 or speed_squared <= 1e-9:
        return None
    value = closing / speed_squared
    closest = (
        relative_position[0] - relative_velocity[0] * value,
        relative_position[1] - relative_velocity[1] * value,
    )
    return value if math.hypot(*closest) < 3.0 else None


def field_value(value, default=None):
    if isinstance(value, dict):
        return value.get("value", default)
    return default


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carla-api", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--frame-limit", type=int, default=80)
    parser.add_argument("--post-collision-frames", type=int, default=20)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--map-fallback-from",
        help="Original preregistered map when a defaulted runtime map binding is used.",
    )
    parser.add_argument(
        "--force-load-world",
        action="store_true",
        help="Load the contract map directly without querying the server's default map.",
    )
    parser.add_argument(
        "--reuse-only",
        action="store_true",
        help="Require the server to already host the contract map; never call load_world().",
    )
    args = parser.parse_args()

    carla_api = os.path.abspath(args.carla_api)
    sys.path.insert(0, carla_api)
    import carla

    output_dir = os.path.abspath(args.output_dir)
    if os.path.exists(output_dir):
        raise FileExistsError("refusing to overwrite run directory: " + output_dir)
    os.makedirs(output_dir)
    keyframe_dir = os.path.join(output_dir, "keyframes")
    os.makedirs(keyframe_dir)
    contract = read_json(args.contract)
    shutil.copy2(args.contract, os.path.join(output_dir, "contract.json"))
    topology = str(field_value(contract.get("topology"), "unknown"))
    if topology not in TOPOLOGIES:
        raise ValueError("unsupported or missing topology: " + topology)
    seed = int(args.seed if args.seed is not None else contract.get("seed", 0))

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    requested_map = str(field_value(contract["map_binding"]["carla_map"]))
    acquire_started = time.perf_counter()
    if args.force_load_world and args.reuse_only:
        raise ValueError("--force-load-world and --reuse-only are mutually exclusive")
    if args.force_load_world:
        world = client.load_world(requested_map)
        acquire_mode = "forced_loaded"
    else:
        current_world = client.get_world()
        current_map = str(current_world.get_map().name).split("/")[-1]
        if current_map == requested_map.split("/")[-1]:
            world = current_world
            acquire_mode = "reused"
        elif args.reuse_only:
            raise RuntimeError(
                "dedicated server map mismatch: current={} requested={}".format(
                    current_map, requested_map
                )
            )
        else:
            world = client.load_world(requested_map)
            acquire_mode = "loaded"
    acquire_seconds = time.perf_counter() - acquire_started
    original_settings = world.get_settings()
    actors = []
    sensors = []
    actor_by_id = {}
    collision_events = []
    captured_frames = []
    states = []
    velocity_samples = {}
    minimum_seen = None
    asset_fallbacks = []
    telemetry_path = os.path.join(output_dir, "telemetry.jsonl")

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = float(contract.get("fixed_delta_seconds", 0.05))
        settings.no_rendering_mode = False
        world.apply_settings(settings)
        blueprint_library = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("CARLA map has no spawn points")
        constraint = next(
            (item for item in contract.get("collision_constraints", []) if item.get("required", True)),
            None,
        )
        if not constraint:
            raise RuntimeError("topology contract requires a collision constraint")
        actor_id = constraint["actor_id"]
        target_id = constraint["target_id"]
        pair = (actor_id, target_id)
        specs = {item["id"]: item for item in contract["actors"]}
        if actor_id not in specs or target_id not in specs:
            raise RuntimeError("collision pair is absent from actors")
        base = spawn_points[seed % len(spawn_points)]
        forward = base.get_forward_vector()
        right = carla.Vector3D(x=-forward.y, y=forward.x, z=0.0)
        map_object = world.get_map()

        def profiled_transform(along, lateral, yaw_delta=0.0):
            return carla.Transform(
                carla.Location(
                    x=base.location.x + forward.x * along + right.x * lateral,
                    y=base.location.y + forward.y * along + right.y * lateral,
                    z=base.location.z + 0.5,
                ),
                carla.Rotation(
                    pitch=base.rotation.pitch,
                    yaw=base.rotation.yaw + yaw_delta,
                    roll=base.rotation.roll,
                ),
            )

        actor_speed = float(field_value(specs[actor_id]["initial_speed_mps"], 0.0) or 0.0)
        target_speed = float(field_value(specs[target_id]["initial_speed_mps"], 0.0) or 0.0)
        if topology == "rear_end":
            transforms = {
                actor_id: profiled_transform(-18.0, 0.0),
                target_id: profiled_transform(0.0, 0.0),
            }
            velocities = {
                actor_id: carla.Vector3D(forward.x * actor_speed, forward.y * actor_speed, 0.0),
                target_id: carla.Vector3D(forward.x * target_speed, forward.y * target_speed, 0.0),
            }
        elif topology == "head_on_centerline_intrusion":
            transforms = {
                actor_id: profiled_transform(-14.0, 0.0),
                target_id: profiled_transform(14.0, 0.0, 180.0),
            }
            velocities = {
                actor_id: carla.Vector3D(forward.x * actor_speed, forward.y * actor_speed, 0.0),
                target_id: carla.Vector3D(-forward.x * target_speed, -forward.y * target_speed, 0.0),
            }
        elif topology == "intersection_crossing_turning":
            junction_waypoints = [
                waypoint
                for waypoint in map_object.generate_waypoints(3.0)
                if waypoint.is_junction
            ]
            crossing_pairs = []
            for first_index, first in enumerate(junction_waypoints):
                for second in junction_waypoints[first_index + 1 :]:
                    distance = first.transform.location.distance(second.transform.location)
                    yaw_difference = abs(
                        ((first.transform.rotation.yaw - second.transform.rotation.yaw + 180.0) % 360.0)
                        - 180.0
                    )
                    if 10.0 <= distance <= 24.0 and 55.0 <= yaw_difference <= 125.0:
                        crossing_pairs.append((first, second))
                        if len(crossing_pairs) >= 32:
                            break
                if len(crossing_pairs) >= 32:
                    break
            if not crossing_pairs:
                raise RuntimeError("no perpendicular junction waypoint pair found")
            crossing_pair = crossing_pairs[seed % len(crossing_pairs)]
            first, second = crossing_pair
            first_transform = first.transform
            second_transform = second.transform
            first_transform.location.z += 0.5
            second_transform.location.z += 0.5
            transforms = {actor_id: first_transform, target_id: second_transform}
            center_x = (first_transform.location.x + second_transform.location.x) / 2.0
            center_y = (first_transform.location.y + second_transform.location.y) / 2.0

            def toward_center(transform, speed):
                delta_x = center_x - transform.location.x
                delta_y = center_y - transform.location.y
                length = max(math.hypot(delta_x, delta_y), 1e-6)
                return carla.Vector3D(
                    delta_x / length * speed, delta_y / length * speed, 0.0
                )

            velocities = {
                actor_id: toward_center(first_transform, actor_speed),
                target_id: toward_center(second_transform, target_speed),
            }
        else:
            transforms = {
                actor_id: profiled_transform(-6.0, 3.5),
                target_id: profiled_transform(0.0, 0.0),
            }
            velocities = {
                actor_id: carla.Vector3D(
                    forward.x * actor_speed - right.x * 1.5,
                    forward.y * actor_speed - right.y * 1.5,
                    0.0,
                ),
                target_id: carla.Vector3D(forward.x * target_speed, forward.y * target_speed, 0.0),
            }

        # Controlled RQ4 pose faults are encoded only in mutable/defaulted lane
        # fields.  Move the affected actor outside the profiled relation so a
        # CARLA rerun, rather than materialization alone, verifies the phenotype.
        injected_lane_fault_ids = set()
        collision_omission_active = False
        for contract_id, spec in specs.items():
            lane_fault = field_value(spec.get("lane_position"))
            if lane_fault in (
                "adjacent_lane_fault",
                "collision_omission_fault",
                "pose_perturbation_fault",
                "map_lane_mismatch_fault",
            ):
                transform = transforms.get(contract_id)
                if transform is not None:
                    displacement = {
                        "collision_omission_fault": 30.0,
                        "pose_perturbation_fault": 30.0,
                        "map_lane_mismatch_fault": 30.0,
                    }.get(lane_fault, 12.0)
                    transform.location.x += right.x * displacement
                    transform.location.y += right.y * displacement
                    if lane_fault == "collision_omission_fault":
                        collision_omission_active = True
                    if lane_fault in ("pose_perturbation_fault", "map_lane_mismatch_fault"):
                        injected_lane_fault_ids.add(contract_id)

        if collision_omission_active:
            velocities[actor_id] = carla.Vector3D()
            velocities[target_id] = carla.Vector3D()

        lane_checks = {}
        for contract_id, transform in transforms.items():
            waypoint = map_object.get_waypoint(
                transform.location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            # Lane validity is a road-plane relationship.  Custom CARLA maps may
            # use a spawn Z offset that differs from the OpenDRIVE reference
            # surface, so including Z creates false lane mismatches.
            projected_distance = math.hypot(
                transform.location.x - waypoint.transform.location.x,
                transform.location.y - waypoint.transform.location.y,
            )
            lane_checks[contract_id] = {
                "road_id": waypoint.road_id,
                "lane_id": waypoint.lane_id,
                "projected_distance_m": projected_distance,
                "within_driving_lane_tolerance": (
                    projected_distance <= 4.0 and contract_id not in injected_lane_fault_ids
                ),
                "controlled_lane_fault": contract_id in injected_lane_fault_ids,
            }

        fallback_blueprints = ("vehicle.tesla.model3", "vehicle.audi.tt")
        for index, spec in enumerate(contract["actors"]):
            contract_id = spec["id"]
            requested_blueprint = str(
                field_value(spec.get("blueprint"), fallback_blueprints[0])
            )
            try:
                blueprint = blueprint_library.find(requested_blueprint)
            except IndexError:
                blueprint = blueprint_library.find(fallback_blueprints[0])
                asset_fallbacks.append(
                    {
                        "actor_id": contract_id,
                        "requested": requested_blueprint,
                        "used": fallback_blueprints[0],
                    }
                )
            transform = transforms.get(contract_id, spawn_points[(index + seed + 10) % len(spawn_points)])
            actor = world.try_spawn_actor(blueprint, transform)
            for fallback_blueprint in fallback_blueprints:
                if actor is not None:
                    break
                blueprint = blueprint_library.find(fallback_blueprint)
                retry_transform = carla.Transform(transform.location, transform.rotation)
                retry_transform.location.z += 0.5
                actor = world.try_spawn_actor(blueprint, retry_transform)
                if actor is not None:
                    asset_fallbacks.append(
                        {"actor_id": contract_id, "requested": requested_blueprint, "used": fallback_blueprint}
                    )
            if actor is None:
                raise RuntimeError("failed to spawn contract actor " + contract_id)
            actor.apply_control(carla.VehicleControl(brake=1.0))
            actors.append(actor)
            actor_by_id[contract_id] = actor

        carla_to_contract = {actor.id: contract_id for contract_id, actor in actor_by_id.items()}
        collision_blueprint = blueprint_library.find("sensor.other.collision")
        for contract_id, actor in actor_by_id.items():
            sensor = world.spawn_actor(collision_blueprint, carla.Transform(), attach_to=actor)
            actors.append(sensor)
            sensors.append(sensor)

            def on_collision(event, source_id=contract_id):
                impulse = event.normal_impulse
                collision_events.append(
                    {
                        "frame": event.frame,
                        "actor_id": source_id,
                        "other_actor_id": carla_to_contract.get(event.other_actor.id),
                        "impulse": {"x": impulse.x, "y": impulse.y, "z": impulse.z},
                    }
                )

            sensor.listen(on_collision)

        ego_id = next(
            (item["id"] for item in contract["actors"] if item.get("role") == "ego"),
            actor_id,
        )
        camera_blueprint = blueprint_library.find("sensor.camera.rgb")
        camera_blueprint.set_attribute("image_size_x", "800")
        camera_blueprint.set_attribute("image_size_y", "450")
        camera_blueprint.set_attribute(
            "sensor_tick", str(float(contract.get("fixed_delta_seconds", 0.05)) * 5.0)
        )
        camera = world.spawn_actor(
            camera_blueprint,
            carla.Transform(
                carla.Location(x=-8.0, z=4.0), carla.Rotation(pitch=-15.0)
            ),
            attach_to=actor_by_id[ego_id],
        )
        actors.append(camera)
        sensors.append(camera)

        def on_image(image):
            name = "frame_{:08d}.png".format(image.frame)
            path = os.path.join(keyframe_dir, name)
            image.save_to_disk(path)
            captured_frames.append((image.frame, os.path.join("keyframes", name).replace("\\", "/")))

        camera.listen(on_image)

        def apply_motion(collision_seen):
            commands = []
            for contract_id, actor in actor_by_id.items():
                if contract_id in pair and not collision_seen:
                    commands.append(
                        carla.command.ApplyVehicleControl(actor.id, carla.VehicleControl())
                    )
                    commands.append(
                        carla.command.ApplyTargetVelocity(actor.id, velocities[contract_id])
                    )
                else:
                    commands.append(
                        carla.command.ApplyVehicleControl(actor.id, carla.VehicleControl(brake=1.0))
                    )
                    if collision_seen:
                        commands.append(
                            carla.command.ApplyTargetVelocity(actor.id, carla.Vector3D())
                        )
            responses = client.apply_batch_sync(commands, False)
            errors = [response.error for response in responses if response.has_error()]
            if errors:
                raise RuntimeError("CARLA motion batch failed: " + "; ".join(errors))

        apply_motion(False)
        run_start_frame = world.tick()
        fixed_delta = float(contract.get("fixed_delta_seconds", 0.05))
        planned_frames = int(float(contract.get("duration_seconds", 20.0)) / fixed_delta)
        frames = min(planned_frames, args.frame_limit)
        with open(telemetry_path, "w", encoding="utf-8") as telemetry:
            for _ in range(frames):
                pair_collision_seen = any(
                    {item["actor_id"], item["other_actor_id"]} == set(pair)
                    for item in collision_events
                )
                apply_motion(pair_collision_seen)
                frame = world.tick()
                relative_frame = frame - run_start_frame
                snapshot = world.get_snapshot()
                frame_states = []
                for contract_id, actor in actor_by_id.items():
                    transform = actor.get_transform()
                    velocity = actor.get_velocity()
                    control = actor.get_control()
                    speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
                    state = {
                        "frame": relative_frame,
                        "actor_id": contract_id,
                        "timestamp_seconds": relative_frame * fixed_delta,
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
                        "speed_mps": speed,
                        "control": {
                            "throttle": control.throttle,
                            "steer": control.steer,
                            "brake": control.brake,
                            "hand_brake": control.hand_brake,
                        },
                    }
                    states.append(state)
                    frame_states.append(state)
                    velocity_samples[(relative_frame, contract_id)] = (velocity.x, velocity.y)
                ego_actor = actor_by_id[actor_id]
                target_actor = actor_by_id[target_id]
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
                    minimum_seen = current_ttc if minimum_seen is None else min(minimum_seen, current_ttc)
                telemetry.write(
                    json.dumps(
                        {
                            "frame": relative_frame,
                            "timestamp_seconds": relative_frame * fixed_delta,
                            "actors": frame_states,
                            "collision_events": [
                                dict(item, frame=item["frame"] - run_start_frame)
                                for item in sorted(
                                    (
                                        event
                                        for event in collision_events
                                        if event["frame"] == frame
                                    ),
                                    key=lambda event: (
                                        event["actor_id"],
                                        str(event["other_actor_id"]),
                                    ),
                                )
                            ],
                            "minimum_ttc_seconds": minimum_seen,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                if collision_events:
                    first_collision_global = min(item["frame"] for item in collision_events)
                    if frame >= first_collision_global + args.post_collision_frames:
                        break
        for sensor in sensors:
            sensor.stop()
        sensors[:] = []
        time.sleep(0.25)

        pair_collisions = [
            item
            for item in collision_events
            if {item["actor_id"], item["other_actor_id"]} == set(pair)
        ]
        collision_frame = min((item["frame"] for item in pair_collisions), default=None)
        collision_frame_relative = (
            collision_frame - run_start_frame if collision_frame is not None else None
        )
        normalized_collisions = [
            dict(item, frame=item["frame"] - run_start_frame)
            for item in sorted(
                collision_events,
                key=lambda event: (
                    event["frame"],
                    event["actor_id"],
                    str(event["other_actor_id"]),
                ),
            )
        ]
        first_state_frame = min((item["frame"] for item in states), default=None)
        keyframes = []
        if captured_frames:
            if collision_frame is not None:
                targets = (collision_frame - 15, collision_frame, collision_frame + 15)
                for target_frame in targets:
                    path = min(captured_frames, key=lambda item: abs(item[0] - target_frame))[1]
                    if path not in keyframes:
                        keyframes.append(path)
            else:
                indices = (0, len(captured_frames) // 2, len(captured_frames) - 1)
                keyframes = list(dict.fromkeys(captured_frames[index][1] for index in indices))
        lane_topology_pass = all(
            value["within_driving_lane_tolerance"] for value in lane_checks.values()
        )
        event_order_pass = (
            first_state_frame is not None
            and collision_frame_relative is not None
            and collision_frame_relative >= first_state_frame
        )
        relative_speed = None
        if collision_frame is not None:
            actor_velocity = velocity_samples.get((collision_frame_relative, actor_id))
            target_velocity = velocity_samples.get((collision_frame_relative, target_id))
            if actor_velocity and target_velocity:
                relative_speed = math.hypot(
                    actor_velocity[0] - target_velocity[0],
                    actor_velocity[1] - target_velocity[1],
                )
        constraints = [
            {
                "name": "collision_target",
                "passed": bool(pair_collisions),
                "expected": {"actor_id": actor_id, "target_id": target_id},
                "observed": [
                    dict(item, frame=item["frame"] - run_start_frame)
                    for item in pair_collisions
                ],
                "reason": None if pair_collisions else "required actor-target collision was not observed",
            },
            {
                "name": "lane_topology_valid",
                "passed": lane_topology_pass,
                "expected": topology,
                "observed": {"topology": topology, "lane_checks": lane_checks},
                "reason": None if lane_topology_pass else "profiled spawn exceeded driving-lane tolerance",
            },
            {
                "name": "event_order_valid",
                "passed": event_order_pass,
                "expected": "initial state precedes required collision",
                "observed": {
                    "first_state_frame": first_state_frame,
                    "first_collision_frame": collision_frame_relative,
                },
                "reason": None if event_order_pass else "collision absent or outside telemetry order",
            },
        ]
        passed = all(item["passed"] for item in constraints)
        result = {
            "version": "1.0",
            "scenario_id": contract["scenario_id"],
            "backend": "carla-0.9.13-python37-topology-bridge",
            "executed": True,
            "status": "passed" if passed else "failed",
            "actor_states": states,
            "collisions": normalized_collisions,
            "minimum_ttc_seconds": minimum_seen,
            "constraint_results": constraints,
            "keyframes": keyframes,
            "logs": [
                "execution_profile=topology_contract",
                "topology=" + topology,
                "world_acquire_mode=" + acquire_mode,
                "world_acquire_seconds={:.6f}".format(acquire_seconds),
                "impact_relative_speed_mps=" + ("null" if relative_speed is None else "{:.6f}".format(relative_speed)),
                "asset_fallbacks=" + json.dumps(asset_fallbacks, ensure_ascii=False),
            ],
            "errors": [],
        }
        result_path = os.path.join(output_dir, "simulation_result.json")
        write_json(result_path, result)
        run_record = {
            "scenario_id": contract["scenario_id"],
            "candidate_id": contract["candidate_id"],
            "topology": topology,
            "requested_map": requested_map,
            "seed": seed,
            "execution_status": "completed",
            "contract_compile_pass": True,
            "carla_launch_complete": True,
            "run_complete": True,
            "actor_target_correct": bool(pair_collisions),
            "lane_topology_valid": lane_topology_pass,
            "event_order_valid": event_order_pass,
            "hard_constraint_pass": passed,
            "minimum_ttc_seconds": minimum_seen,
            "impact_relative_speed_mps": relative_speed,
            "collision_signature": "{}>{}@{}".format(
                actor_id, target_id, collision_frame_relative
            ),
            "telemetry_sha256": sha256_file(telemetry_path),
            "simulation_result_sha256": sha256_file(result_path),
            "map_asset_fallback": (
                {
                    "map": (
                        {"requested": args.map_fallback_from, "used": requested_map}
                        if args.map_fallback_from
                        else None
                    ),
                    "blueprints": asset_fallbacks,
                }
                if args.map_fallback_from or asset_fallbacks
                else None
            ),
            "failure_reason": None if passed else "; ".join(
                item["name"] for item in constraints if not item["passed"]
            ),
        }
        write_json(os.path.join(output_dir, "run_record.json"), run_record)
        write_json(
            os.path.join(output_dir, "run_manifest.json"),
            {
                "carla_api": carla_api,
                "carla_binary_sha256": sha256_file(
                    os.path.join(carla_api, "carla", "libcarla.cp37-win_amd64.pyd")
                ),
                "contract_path": os.path.abspath(args.contract),
                "contract_sha256": sha256_file(args.contract),
                "frame_limit": args.frame_limit,
                "post_collision_frames": args.post_collision_frames,
                "host": args.host,
                "port": args.port,
                "requested_map": requested_map,
                "map_fallback_from": args.map_fallback_from,
                "python": sys.version,
            "claim_boundary": "Executed CARLA telemetry; selection provenance and topology confirmation are governed by the separately frozen preregistration.",
            },
        )
        print(json.dumps(run_record, ensure_ascii=False, indent=2))
        return 0 if passed else 2
    finally:
        try:
            client.set_timeout(2.0)
            if actors:
                client.apply_batch_sync(
                    [carla.command.DestroyActor(actor.id) for actor in reversed(actors)],
                    True,
                )
        except RuntimeError:
            pass
        try:
            world.apply_settings(original_settings)
        except RuntimeError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
