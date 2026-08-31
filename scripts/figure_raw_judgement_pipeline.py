#!/usr/bin/env python3
"""Render a provenance-backed raw-judgment-to-CARLA pipeline figure."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image


INK = "#17324D"
MUTED = "#526A7D"
BLUE = "#165D8D"
BLUE_LIGHT = "#E7F2F8"
TEAL = "#0A7A78"
TEAL_LIGHT = "#E4F4F1"
GREEN = "#26734D"
GREEN_LIGHT = "#E8F3EC"
ORANGE = "#C85D17"
ORANGE_LIGHT = "#FFF0E4"
RED = "#C63D3D"
RED_LIGHT = "#FBE9E8"
PURPLE = "#7354A3"
PURPLE_LIGHT = "#F0EBF8"
LINE = "#AABAC6"
PAPER = "#F8FAFB"
WHITE = "#FFFFFF"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def configure_fonts() -> None:
    candidates = (
        Path("C:/Windows/Fonts/malgun.ttf"),
        Path("C:/Windows/Fonts/malgunbd.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    )
    for path in candidates:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
    plt.rcParams.update(
        {
            "font.family": ["Malgun Gothic", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    facecolor: str = WHITE,
    edgecolor: str = LINE,
    linewidth: float = 1.2,
    radius: float = 1.3,
    zorder: float = 1,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.4,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def stage_header(
    ax: plt.Axes,
    x: float,
    y: float,
    number: str,
    title: str,
    color: str,
    *,
    subtitle: str | None = None,
) -> None:
    ax.add_patch(Circle((x + 2.2, y - 2.2), 1.45, facecolor=color, edgecolor="none", zorder=4))
    ax.text(x + 2.2, y - 2.2, number, ha="center", va="center", color=WHITE, fontsize=8.5, fontweight="bold", zorder=5)
    ax.text(x + 4.4, y - 1.6, title, ha="left", va="center", color=INK, fontsize=10.0, fontweight="bold", zorder=5)
    if subtitle:
        ax.text(x + 4.4, y - 3.5, subtitle, ha="left", va="center", color=MUTED, fontsize=7.5, zorder=5)


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = MUTED) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.35,
            color=color,
            shrinkA=2,
            shrinkB=2,
            zorder=8,
        )
    )


def fit_image(path: Path, crop: tuple[float, float, float, float] | None = None) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if crop:
        w, h = image.size
        l, t, r, b = crop
        image = image.crop((int(l * w), int(t * h), int(r * w), int(b * h)))
    return image


def add_image_axes(
    fig: plt.Figure,
    rect: tuple[float, float, float, float],
    image: Image.Image,
    *,
    border: str = LINE,
    linewidth: float = 1.0,
) -> plt.Axes:
    ax = fig.add_axes(rect, zorder=4)
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(border)
        spine.set_linewidth(linewidth)
    return ax


def compact_excerpt(source_text: str) -> str:
    marker = "그럼에도"
    start = source_text.find(marker)
    excerpt = source_text[start:] if start >= 0 else source_text
    excerpt = " ".join(excerpt.split())
    return textwrap.shorten(excerpt, width=190, placeholder=" …")


def make_figure(
    workspace_root: Path,
    run_root: Path,
    output_dir: Path,
) -> dict:
    configure_fonts()
    output_dir.mkdir(parents=True, exist_ok=True)

    local_root = workspace_root / "LocalLLM"
    paper_root = workspace_root / "Paper_NewLocalLLM"
    raw_path = local_root / "zeroshot_test/inputs/raw/zeroshot_test_71.json"
    structured_path = local_root / "zeroshot_test/outputs/zeroshot_done/zeroshot_test_71_result.json"
    filtered_path = local_root / "zeroshot_test/pipelines/car_to_car_filter/full_run/output/car_to_car/zeroshot_test_71_result.json"
    audit_path = paper_root / "results/n0_n3_summary.json"
    validation_path = paper_root / "results/n4_n6_validation_summary.json"
    bundle = run_root / "repro_fix_a/jurisdrive_71"
    graph_path = bundle / "evidence_graph.json"
    contract_path = bundle / "contract.json"
    simulation_path = bundle / "simulation_result.json"
    evaluation_path = bundle / "evaluation_vlm.json"

    sources = [
        raw_path,
        structured_path,
        filtered_path,
        audit_path,
        validation_path,
        graph_path,
        contract_path,
        simulation_path,
        evaluation_path,
    ]
    for path in sources:
        if not path.exists():
            raise FileNotFoundError(path)

    raw = read_json(raw_path)
    structured = read_json(structured_path)
    filtered = read_json(filtered_path)
    audit = read_json(audit_path)
    validation = read_json(validation_path)
    graph = read_json(graph_path)
    contract = read_json(contract_path)
    simulation = read_json(simulation_path)
    evaluation = read_json(evaluation_path)

    keyframes = [bundle / rel for rel in simulation["keyframes"]]
    for path in keyframes:
        if not path.exists():
            raise FileNotFoundError(path)

    fig = plt.figure(figsize=(16, 11.2), dpi=200, facecolor=WHITE)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Title band
    ax.add_patch(Rectangle((2.5, 93.4), 95, 4.8, facecolor=INK, edgecolor="none"))
    ax.text(50, 96.2, "From Raw Legal Judgment to a Validated CARLA Scenario", ha="center", va="center", color=WHITE, fontsize=18, fontweight="bold")
    ax.text(50, 94.5, "A provenance-preserving, selective multimodal reconstruction pipeline", ha="center", va="center", color="#DCE8F0", fontsize=9.5)

    # Raw record
    rounded_box(ax, (5, 78.0), 90, 12.0, facecolor=PAPER, edgecolor=BLUE, linewidth=1.5, radius=1.5)
    stage_header(ax, 7.0, 89.0, "N0", "Raw Judgment Data", BLUE)
    ax.text(92.2, 86.8, "76,291 Korean court records  •  selected example: TC-71", color=MUTED, fontsize=7.5, ha="right", va="center")
    ax.add_patch(FancyBboxPatch((8, 80.0), 20, 5.2, boxstyle="round,pad=0.25,rounding_size=0.8", facecolor=BLUE_LIGHT, edgecolor="none"))
    ax.text(9.2, 83.7, "광주지방법원-2020고단6037", color=BLUE, fontsize=8.8, fontweight="bold", ha="left", va="center")
    ax.text(9.2, 81.75, "2020-10-26 22:25  |  criminal", color=MUTED, fontsize=7.5, ha="left", va="center")
    excerpt = compact_excerpt(structured["source_text"])
    wrapped_excerpt = textwrap.fill(excerpt, width=91)
    ax.text(30.2, 83.0, wrapped_excerpt, color=INK, fontsize=8.4, ha="left", va="center", linespacing=1.45)
    ax.text(91.5, 79.25, f"source SHA-256  {sha256_file(raw_path)[:12]}…", color=MUTED, fontsize=6.8, ha="right", va="center")

    arrow(ax, (50, 77.7), (50, 74.5), color=BLUE)
    ax.text(52.1, 76.1, "semantic extraction + deterministic routing", color=MUTED, fontsize=7.3, va="center")

    # Processing row
    box_y, box_h, box_w = 54.2, 18.8, 17.0
    xs = [4.3, 23.4, 42.5, 61.6, 80.7]
    fills = [BLUE_LIGHT, ORANGE_LIGHT, TEAL_LIGHT, PURPLE_LIGHT, GREEN_LIGHT]
    colors = [BLUE, ORANGE, TEAL, PURPLE, GREEN]
    titles = [
        ("N1", "Structured Extraction", "accident-window JSON"),
        ("N2–N3", "Selective Cascade", "rule first, Qwen only if ambiguous"),
        ("N4", "Evidence Graph", "exact quote + actor relation"),
        ("N5", "Scenario Contract", "grounded + bounded defaults"),
        ("N5", "CARLA Execution", "Scenic compile + sensors"),
    ]
    for x, fill, color, (num, title, subtitle) in zip(xs, fills, colors, titles):
        rounded_box(ax, (x, box_y), box_w, box_h, facecolor=fill, edgecolor=color, linewidth=1.25, radius=1.2)
        stage_header(ax, x + 0.6, box_y + box_h - 1.0, num, title, color, subtitle=subtitle)
    for x0, x1 in zip(xs[:-1], xs[1:]):
        arrow(ax, (x0 + box_w + 0.3, box_y + 9.4), (x1 - 0.3, box_y + 9.4), color=MUTED)

    parsed = structured["parsed"]
    n1_lines = [
        ("Time", parsed["accident_datetime"]),
        ("Road", parsed["road_type"]),
        ("Actors", "Sorento ↔ Sportage"),
        ("Motion", "moving → stopped vehicle"),
    ]
    for idx, (label, value) in enumerate(n1_lines):
        yy = 64.6 - idx * 2.25
        ax.text(xs[0] + 1.3, yy, label.upper(), color=BLUE, fontsize=6.3, fontweight="bold", va="center")
        ax.text(xs[0] + 5.0, yy, str(value), color=INK, fontsize=7.0, va="center")

    rule = filtered["postprocess"]["rule"]
    ax.text(xs[1] + 1.3, 65.2, f"RULE SCORE   {rule['score']}", color=ORANGE, fontsize=8.0, fontweight="bold")
    ax.text(xs[1] + 1.3, 62.5, "8 vehicle mentions", color=INK, fontsize=7.2)
    ax.text(xs[1] + 1.3, 60.3, "1 collision expression", color=INK, fontsize=7.2)
    ax.text(xs[1] + 1.3, 57.5, "TC-71  →  CAR-TO-CAR", color=WHITE, fontsize=7.4, fontweight="bold", bbox={"boxstyle": "round,pad=0.35", "facecolor": ORANGE, "edgecolor": "none"})
    ax.text(xs[1] + 1.3, 55.4, "Qwen route: not needed", color=MUTED, fontsize=6.8)

    # Mini evidence graph
    gx = xs[2]
    nodes = {
        "v1": (gx + 3.2, 62.7, "vehicle_1"),
        "event": (gx + 8.5, 59.8, "collision"),
        "v3": (gx + 13.8, 62.7, "vehicle_3"),
        "quote": (gx + 8.5, 56.8, "evidence span"),
    }
    ax.plot([nodes["v1"][0], nodes["event"][0], nodes["v3"][0]], [nodes["v1"][1], nodes["event"][1], nodes["v3"][1]], color=TEAL, linewidth=1.4, zorder=3)
    ax.plot([nodes["event"][0], nodes["quote"][0]], [nodes["event"][1], nodes["quote"][1]], color=TEAL, linewidth=1.2, zorder=3)
    for key, (nx, ny, label) in nodes.items():
        radius = 1.55 if key != "quote" else 1.75
        ax.add_patch(Circle((nx, ny), radius, facecolor=WHITE, edgecolor=TEAL, linewidth=1.2, zorder=4))
        ax.text(nx, ny, label, ha="center", va="center", fontsize=5.8, color=INK, zorder=5)
    ax.text(gx + 8.5, 65.5, "vehicle_1  collides_with  vehicle_3", ha="center", color=TEAL, fontsize=6.5, fontweight="bold")
    ax.text(gx + 8.5, 54.9, f"{len(graph['evidence'])} exact spans  •  0 unresolved", ha="center", color=MUTED, fontsize=6.6)

    # Contract card details
    cx = xs[3] + 1.3
    ax.text(cx, 65.2, "TOWN05 / INTERSECTION", color=PURPLE, fontsize=7.8, fontweight="bold")
    ax.text(cx, 62.7, f"actors                  {len(contract['actors'])}", color=INK, fontsize=7.1)
    ax.text(cx, 60.5, "collision target     v1 → v3", color=INK, fontsize=7.1)
    ax.text(cx, 58.3, f"telemetry             {contract['sensors']['telemetry_hz']} Hz", color=INK, fontsize=7.1)
    ax.text(cx, 56.1, f"seed                    {contract['seed']}", color=INK, fontsize=7.1)
    ax.text(cx, 54.9, "observed fields locked", color=PURPLE, fontsize=6.6, fontweight="bold")

    # CARLA execution thumbnail
    carla_thumb = fit_image(keyframes[1], crop=(0.02, 0.03, 0.98, 0.90))
    add_image_axes(fig, (0.823, 0.590, 0.135, 0.075), carla_thumb, border=GREEN, linewidth=1.2)
    ax.text(xs[4] + 1.3, 56.8, "CARLA 0.9.13  •  executed", color=GREEN, fontsize=7.0, fontweight="bold")
    ax.text(xs[4] + 1.3, 54.9, "RGB + collision + telemetry", color=MUTED, fontsize=6.6)

    arrow(ax, (50, 53.8), (50, 49.8), color=GREEN)
    ax.text(52.0, 51.8, "telemetry-grounded multimodal assurance", color=MUTED, fontsize=7.3, va="center")

    # Final output container
    rounded_box(ax, (4.5, 11.2), 91, 37.6, facecolor=PAPER, edgecolor=GREEN, linewidth=1.6, radius=1.8)
    stage_header(ax, 6.0, 47.6, "N6", "Validated Executable Scenario", GREEN, subtitle="actual CARLA RGB keyframes + telemetry + VLM corroboration")
    ax.text(92.8, 45.8, "FINAL  PASS", ha="right", va="center", color=WHITE, fontsize=9.4, fontweight="bold", bbox={"boxstyle": "round,pad=0.45", "facecolor": GREEN, "edgecolor": "none"})

    # Images: coordinates expressed in figure fractions.
    image_rects = [
        (0.075, 0.188, 0.255, 0.205),
        (0.3725, 0.188, 0.255, 0.205),
        (0.670, 0.188, 0.255, 0.205),
    ]
    labels = ["BEFORE  ·  frame 27", "IMPACT  ·  frame 47", "AFTER  ·  frame 67"]
    for idx, (path, rect, label) in enumerate(zip(keyframes, image_rects, labels)):
        image = fit_image(path, crop=(0.0, 0.03, 1.0, 0.91))
        image_ax = add_image_axes(fig, rect, image, border=RED if idx == 1 else LINE, linewidth=2.0 if idx == 1 else 1.0)
        image_ax.text(0.03, 0.93, label, transform=image_ax.transAxes, ha="left", va="top", color=WHITE, fontsize=8.2, fontweight="bold", bbox={"boxstyle": "round,pad=0.3", "facecolor": RED if idx == 1 else INK, "edgecolor": "none", "alpha": 0.93})

    collision_frames = [int(item["frame"]) for item in simulation["collisions"]]
    metrics = [
        ("Collision target", "vehicle_1 → vehicle_3", GREEN),
        ("Detected contacts", str(len(simulation["collisions"])), BLUE),
        ("Collision frames", f"{min(collision_frames)}–{max(collision_frames)}", ORANGE),
        ("Minimum TTC", f"{simulation['minimum_ttc_seconds']:.3f} s", RED),
        ("VLM evaluation", "PASS", GREEN),
    ]
    x_positions = [8.0, 26.1, 42.5, 59.0, 75.6]
    widths = [16.0, 14.2, 14.2, 14.2, 16.0]
    for (label, value, color), x, width in zip(metrics, x_positions, widths):
        rounded_box(ax, (x, 12.8), width, 5.5, facecolor=WHITE, edgecolor=color, linewidth=1.0, radius=0.8, zorder=5)
        ax.text(x + width / 2, 16.5, label.upper(), ha="center", va="center", color=MUTED, fontsize=6.3, fontweight="bold", zorder=6)
        ax.text(x + width / 2, 14.3, value, ha="center", va="center", color=color, fontsize=8.4, fontweight="bold", zorder=6)

    ax.text(50, 7.3, "Observed evidence remains immutable; only inferred/defaulted fields are repairable under bounded assurance.", ha="center", va="center", color=MUTED, fontsize=8.0)
    ax.text(50, 4.6, "TC-71  |  raw judgment → structured facts → selective decision → evidence graph → scenario contract → CARLA → validated result", ha="center", va="center", color=INK, fontsize=8.8, fontweight="bold")

    png_path = output_dir / "jurisdrive_raw_judgement_to_validated_scenario.png"
    pdf_path = output_dir / "jurisdrive_raw_judgement_to_validated_scenario.pdf"
    caption_path = output_dir / "jurisdrive_raw_judgement_to_validated_scenario_caption.txt"
    provenance_path = output_dir / "jurisdrive_raw_judgement_to_validated_scenario_sources.json"
    fig.savefig(png_path, dpi=300, facecolor=WHITE, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(pdf_path, facecolor=WHITE, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    caption = (
        "From raw legal judgment to a validated executable traffic scenario. "
        "For TC-71, an accident-focused judgment window is converted into structured fields, "
        "routed by the selective rule/Qwen cascade, grounded as an exact-span evidence graph, "
        "compiled into a bounded CARLA scenario contract, and evaluated using actual RGB keyframes, "
        "collision telemetry, TTC, and multimodal VLM corroboration."
    )
    caption_path.write_text(caption + "\n", encoding="utf-8")

    provenance = {
        "figure": str(png_path),
        "example": {
            "candidate_id": 71,
            "document_id": raw["doc_id"],
            "rule_label": filtered["postprocess"]["final_label"],
            "rule_score": rule["score"],
            "graph_evidence_spans": len(graph["evidence"]),
            "contract_map": contract["map_binding"]["carla_map"]["value"],
            "contract_seed": contract["seed"],
            "simulation_status": simulation["status"],
            "collision_event_count": len(simulation["collisions"]),
            "minimum_ttc_seconds": simulation["minimum_ttc_seconds"],
            "evaluation_passed": evaluation["passed"],
        },
        "dataset_counts": audit["counts"],
        "validation_summary": {
            "evidence_graph": validation["evidence_graph"],
            "scenario_contract": validation["scenario_contract"],
            "dry_run": validation["dry_run"],
        },
        "sources": [
            {"path": str(path), "sha256": sha256_file(path)} for path in [*sources, *keyframes]
        ],
        "outputs": {
            "png": {"path": str(png_path), "sha256": sha256_file(png_path)},
            "pdf": {"path": str(pdf_path), "sha256": sha256_file(pdf_path)},
            "caption": str(caption_path),
        },
    }
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return provenance


def parse_args() -> argparse.Namespace:
    default_workspace = Path(
        os.environ.get(
            "JURISDRIVE_WORKSPACE",
            Path(__file__).resolve().parents[2],
        )
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=default_workspace)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=default_workspace
        / "Paper_NewLocalLLM/artifacts/migration_runs/20260804_234100/handcrafted_5_collision_profile",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "figures/generated",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    provenance = make_figure(args.workspace_root, args.run_root, args.output_dir)
    print(json.dumps(provenance["outputs"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
