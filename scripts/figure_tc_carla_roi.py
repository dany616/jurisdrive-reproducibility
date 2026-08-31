#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch, Rectangle
from PIL import Image


@dataclass(frozen=True)
class FigureCase:
    candidate_id: int
    map_name: str
    bundle_relative: str
    scene_label: str


CASES = (
    FigureCase(25, "Town01", "repro_fix_a/jurisdrive_25", "Urban approach"),
    FigureCase(
        460,
        "Town04",
        "stratified_20/actual_a/jurisdrive_460",
        "Highway interaction",
    ),
    FigureCase(71, "Town05", "repro_fix_a/jurisdrive_71", "Junction approach"),
)

ACTOR_COLORS = ("#1464b4", "#f28e2b", "#2a9d62", "#8f5bb7")
MAP_BACKGROUND = "#dcebd8"
ROAD_COLOR = "#4c514c"
ROI_COLOR = "#d62728"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _geometry_points(geometry: ET.Element, step: float = 2.0) -> np.ndarray:
    x0 = float(geometry.attrib["x"])
    y0 = float(geometry.attrib["y"])
    heading0 = float(geometry.attrib["hdg"])
    length = float(geometry.attrib["length"])
    count = max(2, int(math.ceil(length / step)) + 1)
    s = np.linspace(0.0, length, count)
    child = next(iter(geometry), None)
    if child is None:
        return np.empty((0, 2))
    kind = child.tag.split("}")[-1]

    if kind == "line":
        x = x0 + s * math.cos(heading0)
        y = y0 + s * math.sin(heading0)
    elif kind == "arc":
        curvature = float(child.attrib["curvature"])
        if abs(curvature) < 1e-12:
            x = x0 + s * math.cos(heading0)
            y = y0 + s * math.sin(heading0)
        else:
            heading = heading0 + curvature * s
            x = x0 + (np.sin(heading) - math.sin(heading0)) / curvature
            y = y0 - (np.cos(heading) - math.cos(heading0)) / curvature
    elif kind == "spiral":
        start = float(child.attrib["curvStart"])
        end = float(child.attrib["curvEnd"])
        curvature = start + (end - start) * s / max(length, 1e-9)
        ds = np.diff(s, prepend=0.0)
        heading = heading0 + np.cumsum(curvature * ds)
        x = x0 + np.cumsum(np.cos(heading) * ds)
        y = y0 + np.cumsum(np.sin(heading) * ds)
    elif kind == "poly3":
        a = float(child.attrib.get("a", 0.0))
        b = float(child.attrib.get("b", 0.0))
        c = float(child.attrib.get("c", 0.0))
        d = float(child.attrib.get("d", 0.0))
        u = s
        v = a + b * s + c * s**2 + d * s**3
        x = x0 + u * math.cos(heading0) - v * math.sin(heading0)
        y = y0 + u * math.sin(heading0) + v * math.cos(heading0)
    elif kind == "paramPoly3":
        normalized = child.attrib.get("pRange", "normalized") == "normalized"
        p = s / max(length, 1e-9) if normalized else s
        au = float(child.attrib.get("aU", 0.0))
        bu = float(child.attrib.get("bU", 0.0))
        cu = float(child.attrib.get("cU", 0.0))
        du = float(child.attrib.get("dU", 0.0))
        av = float(child.attrib.get("aV", 0.0))
        bv = float(child.attrib.get("bV", 0.0))
        cv = float(child.attrib.get("cV", 0.0))
        dv = float(child.attrib.get("dV", 0.0))
        u = au + bu * p + cu * p**2 + du * p**3
        v = av + bv * p + cv * p**2 + dv * p**3
        x = x0 + u * math.cos(heading0) - v * math.sin(heading0)
        y = y0 + u * math.sin(heading0) + v * math.cos(heading0)
    else:
        x = x0 + s * math.cos(heading0)
        y = y0 + s * math.sin(heading0)
    return np.column_stack((x, y))


def load_road_centerlines(xodr_path: Path) -> list[np.ndarray]:
    root = ET.parse(xodr_path).getroot()
    lines: list[np.ndarray] = []
    for road in root.findall("road"):
        plan_view = road.find("planView")
        if plan_view is None:
            continue
        pieces = [_geometry_points(item) for item in plan_view.findall("geometry")]
        pieces = [piece for piece in pieces if len(piece)]
        if pieces:
            points = np.vstack(pieces)
            # OpenDRIVE uses a right-handed frame; CARLA telemetry uses a
            # left-handed frame with the opposite y direction.
            points[:, 1] *= -1.0
            lines.append(points)
    if not lines:
        raise ValueError(f"no road geometry found in {xodr_path}")
    return lines


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def actor_tracks(result: dict) -> dict[str, list[dict]]:
    tracks: dict[str, list[dict]] = {}
    for state in result["actor_states"]:
        tracks.setdefault(state["actor_id"], []).append(state)
    for states in tracks.values():
        states.sort(key=lambda state: state["frame"])
    return tracks


def state_nearest(states: list[dict], frame: int) -> dict:
    return min(states, key=lambda state: abs(int(state["frame"]) - frame))


def impact_summary(result: dict) -> tuple[int, int, tuple[str, str], tuple[float, float]]:
    collisions = result["collisions"]
    if not collisions:
        raise ValueError(f"{result['scenario_id']} has no collision")
    first_frame = min(int(item["frame"]) for item in collisions)
    last_frame = max(int(item["frame"]) for item in collisions)
    peak = max(
        collisions,
        key=lambda item: sum(float(item["impulse"][axis]) ** 2 for axis in ("x", "y", "z")),
    )
    pair = (str(peak["actor_id"]), str(peak["other_actor_id"]))
    tracks = actor_tracks(result)
    left = state_nearest(tracks[pair[0]], int(peak["frame"]))["location"]
    right = state_nearest(tracks[pair[1]], int(peak["frame"]))["location"]
    point = ((float(left["x"]) + float(right["x"])) / 2, (float(left["y"]) + float(right["y"])) / 2)
    return first_frame, last_frame, pair, point


def choose_keyframe(bundle: Path, target_frame: int) -> Path:
    candidates = sorted((bundle / "keyframes").glob("frame_*.png"))
    if not candidates:
        raise FileNotFoundError(f"no keyframes in {bundle}")

    def frame_number(path: Path) -> int:
        match = re.search(r"(\d+)$", path.stem)
        if not match:
            raise ValueError(path)
        return int(match.group(1))

    return min(
        candidates,
        key=lambda path: (
            abs(frame_number(path) - target_frame),
            0 if frame_number(path) >= target_frame else 1,
        ),
    )


def plot_roads(ax: plt.Axes, roads: Iterable[np.ndarray], width: float) -> None:
    for points in roads:
        ax.plot(points[:, 0], points[:, 1], color=ROAD_COLOR, linewidth=width, alpha=0.78, zorder=1)


def padded_bounds(tracks: dict[str, list[dict]], padding: float = 14.0) -> tuple[float, float, float, float]:
    x = [float(state["location"]["x"]) for states in tracks.values() for state in states]
    y = [float(state["location"]["y"]) for states in tracks.values() for state in states]
    xmin, xmax, ymin, ymax = min(x), max(x), min(y), max(y)
    xspan = max(xmax - xmin, 10.0)
    yspan = max(ymax - ymin, 10.0)
    return (
        xmin - max(padding, 0.12 * xspan),
        xmax + max(padding, 0.12 * xspan),
        ymin - max(padding, 0.12 * yspan),
        ymax + max(padding, 0.12 * yspan),
    )


def style_map_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(MAP_BACKGROUND)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def make_figure(run_root: Path, carla_root: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(18.0, 10.2), facecolor="white")
    grid = fig.add_gridspec(
        3,
        5,
        width_ratios=(1.18, 1.28, 1.55, 1.55, 1.55),
        left=0.035,
        right=0.99,
        top=0.905,
        bottom=0.075,
        wspace=0.07,
        hspace=0.18,
    )
    headers = ("CARLA network map", "TC-specific ROI dynamics", "Before", "Impact", "After")
    metadata: dict[str, object] = {"cases": [], "source_run_root": str(run_root)}

    for row, case in enumerate(CASES):
        bundle = run_root / case.bundle_relative
        contract = read_json(bundle / "contract.json")
        result = read_json(bundle / "simulation_result.json")
        tracks = actor_tracks(result)
        first_frame, last_frame, collision_pair, impact_xy = impact_summary(result)
        xodr_path = carla_root / "CarlaUE4" / "Content" / "Carla" / "Maps" / "OpenDrive" / f"{case.map_name}.xodr"
        roads = load_road_centerlines(xodr_path)
        bounds = padded_bounds(tracks)

        map_ax = fig.add_subplot(grid[row, 0])
        roi_ax = fig.add_subplot(grid[row, 1])
        screenshot_axes = [fig.add_subplot(grid[row, column]) for column in (2, 3, 4)]
        if row == 0:
            for column, axis in enumerate((map_ax, roi_ax, *screenshot_axes)):
                axis.set_title(headers[column], fontsize=11.0, fontweight="bold", pad=8)

        style_map_axis(map_ax)
        plot_roads(map_ax, roads, 0.72)
        all_road_points = np.vstack(roads)
        global_xmin = min(float(np.min(all_road_points[:, 0])), bounds[0])
        global_xmax = max(float(np.max(all_road_points[:, 0])), bounds[1])
        global_ymin = min(float(np.min(all_road_points[:, 1])), bounds[2])
        global_ymax = max(float(np.max(all_road_points[:, 1])), bounds[3])
        global_xpad = max((global_xmax - global_xmin) * 0.025, 2.0)
        global_ypad = max((global_ymax - global_ymin) * 0.025, 2.0)
        map_ax.set_xlim(global_xmin - global_xpad, global_xmax + global_xpad)
        map_ax.set_ylim(global_ymin - global_ypad, global_ymax + global_ypad)
        roi_rect = Rectangle(
            (bounds[0], bounds[2]),
            bounds[1] - bounds[0],
            bounds[3] - bounds[2],
            fill=False,
            edgecolor=ROI_COLOR,
            linewidth=1.8,
            linestyle=(0, (4, 2)),
            zorder=5,
        )
        map_ax.add_patch(roi_rect)
        map_ax.scatter(*impact_xy, marker="*", s=52, color=ROI_COLOR, edgecolor="white", linewidth=0.5, zorder=6)
        map_ax.text(
            0.03,
            0.97,
            f"({chr(97 + row)})  TC-{case.candidate_id}\n{case.map_name}",
            transform=map_ax.transAxes,
            ha="left",
            va="top",
            fontsize=10.5,
            fontweight="bold",
            bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "none", "pad": 3.2},
            zorder=10,
        )

        style_map_axis(roi_ax)
        plot_roads(roi_ax, roads, 1.15)
        for actor_index, (actor_id, states) in enumerate(sorted(tracks.items())):
            color = ACTOR_COLORS[actor_index % len(ACTOR_COLORS)]
            x = [float(state["location"]["x"]) for state in states]
            y = [float(state["location"]["y"]) for state in states]
            roi_ax.plot(x, y, color=color, linewidth=2.2, zorder=4)
            roi_ax.scatter(x[0], y[0], s=28, marker="o", color=color, edgecolor="white", linewidth=0.6, zorder=5)
            roi_ax.annotate(
                actor_id.replace("vehicle_", "V"),
                (x[0], y[0]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7.8,
                color="#202020",
                zorder=6,
            )
        roi_ax.scatter(*impact_xy, marker="*", s=135, color=ROI_COLOR, edgecolor="white", linewidth=0.8, zorder=8)
        roi_ax.set_xlim(bounds[0], bounds[1])
        roi_ax.set_ylim(bounds[2], bounds[3])
        roi_ax.text(
            0.03,
            0.04,
            f"{case.scene_label}\nTTCmin = {result['minimum_ttc_seconds']:.3f} s",
            transform=roi_ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.2,
            bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "none", "pad": 2.6},
            zorder=10,
        )

        connector = ConnectionPatch(
            xyA=(bounds[1], impact_xy[1]),
            coordsA=map_ax.transData,
            xyB=(bounds[0], impact_xy[1]),
            coordsB=roi_ax.transData,
            color=ROI_COLOR,
            linewidth=1.0,
            alpha=0.72,
            zorder=12,
        )
        fig.add_artist(connector)

        peak_frame = max(
            result["collisions"],
            key=lambda item: sum(float(item["impulse"][axis]) ** 2 for axis in ("x", "y", "z")),
        )["frame"]
        image_paths = (
            choose_keyframe(bundle, first_frame - 20),
            choose_keyframe(bundle, int(peak_frame)),
            choose_keyframe(bundle, last_frame + 20),
        )
        for column_index, (axis, image_path) in enumerate(zip(screenshot_axes, image_paths)):
            with Image.open(image_path) as image:
                axis.imshow(image.convert("RGB"))
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_linewidth(1.8 if column_index == 1 else 0.8)
                spine.set_edgecolor(ROI_COLOR if column_index == 1 else "#8a8a8a")
            if column_index == 1:
                axis.text(
                    0.975,
                    0.05,
                    "COLLISION",
                    transform=axis.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=8.0,
                    color="white",
                    bbox={"facecolor": ROI_COLOR, "edgecolor": "none", "pad": 2.4},
                )

        metadata["cases"].append(
            {
                "candidate_id": case.candidate_id,
                "map": case.map_name,
                "bundle": str(bundle),
                "collision_pair": list(collision_pair),
                "collision_frame_range": [first_frame, last_frame],
                "minimum_ttc_seconds": result["minimum_ttc_seconds"],
                "roi_bounds_xy": list(bounds),
                "screenshots": [
                    {"path": str(path), "sha256": sha256_file(path)} for path in image_paths
                ],
            }
        )

    fig.suptitle(
        "TC-driven CARLA regions of interest and physics-grounded visual evidence",
        fontsize=16,
        fontweight="bold",
        y=0.972,
    )
    fig.text(
        0.5,
        0.035,
        "Dashed red box: simulation ROI   •   solid colored lines: actor trajectories   •   ★: telemetry collision point   •   images: raw CARLA RGB frames",
        ha="center",
        va="center",
        fontsize=9.3,
        color="#333333",
    )
    png_path = output_dir / "jurisdrive_tc_carla_roi_figure.png"
    pdf_path = output_dir / "jurisdrive_tc_carla_roi_figure.pdf"
    fig.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(pdf_path, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    metadata.update(
        {
            "figure_png": str(png_path),
            "figure_pdf": str(pdf_path),
            "png_sha256": sha256_file(png_path),
            "pdf_sha256": sha256_file(pdf_path),
        }
    )
    metadata_path = output_dir / "jurisdrive_tc_carla_roi_figure_sources.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    caption_path = output_dir / "jurisdrive_tc_carla_roi_figure_caption.txt"
    caption_path.write_text(
        "TC-driven CARLA regions of interest and physics-grounded visual evidence. "
        "The left panels locate each test case on the corresponding CARLA road network; "
        "the center panels show actor trajectories and the telemetry-derived collision point; "
        "the right panels show raw RGB frames before, during, and after impact.\n",
        encoding="utf-8",
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--carla-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    metadata = make_figure(args.run_root, args.carla_root, args.output_dir)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
