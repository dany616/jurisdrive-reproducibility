#!/usr/bin/env python3

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_DIR / "figures"

FONT_REGULAR = "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc"
FONT_BOLD = "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Bold.ttc"

BG = "#F7F8FA"
PAPER = "#FFFFFF"
INK = "#18313D"
MUTED = "#647982"
LINE = "#C9D4D8"
NAVY = "#173F5F"
TEAL = "#16877C"
TEAL_LIGHT = "#E5F4F1"
GREEN = "#3B8F64"
GREEN_LIGHT = "#E8F4EC"
ORANGE = "#E5852A"
ORANGE_LIGHT = "#FFF0DA"
BLUE = "#3478B9"
BLUE_LIGHT = "#E9F1FA"
RED = "#C74A4A"
RED_LIGHT = "#FBE9E8"
GRAY = "#7B8C93"
GRAY_LIGHT = "#EEF1F2"
YELLOW = "#D9A321"
YELLOW_LIGHT = "#FFF7D9"

WIDTH = 3000
HEIGHT = 1688


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size=size)


def text_center(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    *,
    size: int,
    color: str = INK,
    bold: bool = False,
    spacing: int = 8,
) -> None:
    fnt = font(size, bold)
    bounds = draw.multiline_textbbox((0, 0), text, font=fnt, spacing=spacing, align="center")
    text_width = bounds[2] - bounds[0]
    text_height = bounds[3] - bounds[1]
    x1, y1, x2, y2 = box
    draw.multiline_text(
        ((x1 + x2 - text_width) / 2, (y1 + y2 - text_height) / 2 - bounds[1]),
        text,
        font=fnt,
        fill=color,
        spacing=spacing,
        align="center",
    )


def rounded_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    fill: str = PAPER,
    outline: str = LINE,
    width: int = 3,
    radius: int = 28,
    dashed: bool = False,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill)
    if not dashed:
        draw.rounded_rectangle(box, radius=radius, outline=outline, width=width)
        return

    x1, y1, x2, y2 = box
    dash = 22
    gap = 13
    for x in range(x1 + radius, x2 - radius, dash + gap):
        draw.line((x, y1, min(x + dash, x2 - radius), y1), fill=outline, width=width)
        draw.line((x, y2, min(x + dash, x2 - radius), y2), fill=outline, width=width)
    for y in range(y1 + radius, y2 - radius, dash + gap):
        draw.line((x1, y, x1, min(y + dash, y2 - radius)), fill=outline, width=width)
        draw.line((x2, y, x2, min(y + dash, y2 - radius)), fill=outline, width=width)


def pill(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    *,
    fill: str,
    color: str,
    size: int = 25,
    outline: str | None = None,
) -> None:
    draw.rounded_rectangle(box, radius=(box[3] - box[1]) // 2, fill=fill, outline=outline, width=2)
    text_center(draw, box, text, size=size, color=color, bold=True)


def arrow_head(
    draw: ImageDraw.ImageDraw,
    point: tuple[int, int],
    angle: float,
    *,
    color: str,
    size: int = 20,
) -> None:
    x, y = point
    points = [(x, y)]
    for delta in (2.55, -2.55):
        points.append(
            (
                x + int(size * math.cos(angle + delta)),
                y + int(size * math.sin(angle + delta)),
            )
        )
    draw.polygon(points, fill=color)


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    color: str = GRAY,
    width: int = 7,
    label: str | None = None,
    label_offset: tuple[int, int] = (0, -38),
) -> None:
    draw.line((*start, *end), fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    arrow_head(draw, end, angle, color=color, size=23)
    if label:
        bounds = draw.textbbox((0, 0), label, font=font(24, True))
        midpoint = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
        x = midpoint[0] - (bounds[2] - bounds[0]) // 2 + label_offset[0]
        y = midpoint[1] + label_offset[1]
        draw.rounded_rectangle(
            (x - 12, y - 5, x + bounds[2] - bounds[0] + 12, y + bounds[3] - bounds[1] + 8),
            radius=10,
            fill=BG,
        )
        draw.text((x, y), label, font=font(24, True), fill=color)


def poly_arrow(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[int, int]],
    *,
    color: str,
    width: int = 7,
    label: str | None = None,
    label_xy: tuple[int, int] | None = None,
) -> None:
    draw.line(points, fill=color, width=width, joint="curve")
    end = points[-1]
    previous = points[-2]
    angle = math.atan2(end[1] - previous[1], end[0] - previous[0])
    arrow_head(draw, end, angle, color=color, size=23)
    if label and label_xy:
        pill(
            draw,
            (label_xy[0], label_xy[1], label_xy[0] + 310, label_xy[1] + 52),
            label,
            fill=PAPER,
            color=color,
            size=23,
            outline=color,
        )


def icon_document(draw: ImageDraw.ImageDraw, center: tuple[int, int], color: str) -> None:
    x, y = center
    draw.rounded_rectangle((x - 38, y - 48, x + 32, y + 48), radius=9, outline=color, width=6)
    draw.polygon([(x + 6, y - 48), (x + 32, y - 22), (x + 6, y - 22)], fill=color)
    for offset in (-8, 10, 28):
        draw.line((x - 24, y + offset, x + 17, y + offset), fill=color, width=5)


def icon_filter(draw: ImageDraw.ImageDraw, center: tuple[int, int], color: str) -> None:
    x, y = center
    draw.polygon(
        [
            (x - 50, y - 42),
            (x + 50, y - 42),
            (x + 18, y - 4),
            (x + 18, y + 40),
            (x - 18, y + 50),
            (x - 18, y - 4),
        ],
        outline=color,
        fill=None,
    )
    draw.line((x - 50, y - 42, x + 50, y - 42), fill=color, width=6)
    draw.line((x - 18, y - 4, x + 18, y - 4), fill=color, width=6)


def icon_network(draw: ImageDraw.ImageDraw, center: tuple[int, int], color: str) -> None:
    x, y = center
    nodes = [(x, y - 45), (x - 50, y + 10), (x + 50, y + 10), (x - 22, y + 52), (x + 28, y + 52)]
    edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 4), (3, 4)]
    for first, second in edges:
        draw.line((*nodes[first], *nodes[second]), fill=color, width=5)
    for nx, ny in nodes:
        draw.ellipse((nx - 12, ny - 12, nx + 12, ny + 12), fill=PAPER, outline=color, width=5)


def icon_contract(draw: ImageDraw.ImageDraw, center: tuple[int, int], color: str) -> None:
    x, y = center
    draw.rounded_rectangle((x - 42, y - 50, x + 42, y + 50), radius=8, outline=color, width=6)
    for offset in (-22, 2, 26):
        draw.rectangle((x - 25, y + offset - 5, x - 14, y + offset + 6), outline=color, width=3)
        draw.line((x - 5, y + offset, x + 27, y + offset), fill=color, width=4)
    draw.line((x - 23, y - 30, x - 17, y - 24), fill=color, width=4)
    draw.line((x - 17, y - 24, x - 8, y - 38), fill=color, width=4)


def icon_car(draw: ImageDraw.ImageDraw, center: tuple[int, int], color: str) -> None:
    x, y = center
    draw.rounded_rectangle((x - 58, y - 15, x + 58, y + 30), radius=13, fill=color)
    draw.polygon([(x - 35, y - 15), (x - 18, y - 45), (x + 28, y - 45), (x + 45, y - 15)], fill=color)
    draw.ellipse((x - 40, y + 17, x - 17, y + 40), fill=INK)
    draw.ellipse((x + 20, y + 17, x + 43, y + 40), fill=INK)
    draw.rectangle((x - 10, y - 38, x + 20, y - 18), fill=PAPER)


def icon_feedback(draw: ImageDraw.ImageDraw, center: tuple[int, int], color: str) -> None:
    x, y = center
    draw.arc((x - 52, y - 52, x + 52, y + 52), 35, 285, fill=color, width=8)
    arrow_head(draw, (x + 24, y - 45), -0.35, color=color, size=18)
    draw.ellipse((x - 12, y - 12, x + 12, y + 12), fill=color)
    draw.line((x, y, x + 28, y + 18), fill=PAPER, width=5)


def stage_header(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    number: str,
    title: str,
    *,
    color: str,
    status: str,
) -> None:
    x, y = xy
    draw.ellipse((x, y, x + 58, y + 58), fill=color)
    text_center(draw, (x, y, x + 58, y + 58), number, size=26, color=PAPER, bold=True)
    draw.text((x + 76, y + 4), title, font=font(31, True), fill=INK)
    status_bounds = draw.textbbox((0, 0), status, font=font(19, True))
    status_width = status_bounds[2] - status_bounds[0] + 28
    pill(
        draw,
        (x + 76, y + 48, x + 76 + status_width, y + 84),
        status,
        fill=color,
        color=PAPER,
        size=19,
    )


def render_architecture_refined(*, implementation_status: bool = False) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)

    draw.text((90, 54), "JurisDrive: Evidence-Carrying Legal-to-CARLA Framework", font=font(61, True), fill=INK)
    draw.text(
        (92, 134),
        "Selective legal-event intelligence coupled with executable simulation assurance",
        font=font(29),
        fill=MUTED,
    )
    pill(draw, (2390, 66, 2640, 122), "VERIFIED", fill=TEAL, color=PAPER, size=23)
    pill(draw, (2660, 66, 2900, 122), "PROPOSED", fill=BLUE_LIGHT, color=BLUE, size=23, outline=BLUE)

    rounded_panel(draw, (70, 218, 2930, 780), fill="#F3FAF8", outline="#B8DBD5", width=3, radius=34)
    draw.text((105, 242), "A  SELECTIVE INTELLIGENCE", font=font(30, True), fill=TEAL)
    draw.text((542, 246), "Implemented and batch-verified", font=font(25), fill=MUTED)
    pill(draw, (2480, 238, 2888, 290), "76,291 records · 0 failures", fill=TEAL_LIGHT, color=TEAL, size=22)

    arrow(draw, (650, 518), (748, 518), color=TEAL)
    arrow(draw, (1365, 518), (1463, 518), color=ORANGE)
    arrow(draw, (2075, 518), (2173, 518), color=GREEN)

    data_box = (105, 322, 650, 718)
    rule_box = (760, 322, 1365, 718)
    qwen_box = (1475, 322, 2075, 718)
    cohort_box = (2185, 322, 2895, 718)
    rounded_panel(draw, data_box, outline=TEAL, width=4)
    rounded_panel(draw, rule_box, outline=TEAL, width=4)
    rounded_panel(draw, qwen_box, outline=ORANGE, width=4)
    rounded_panel(draw, cohort_box, outline=GREEN, width=4)

    icon_document(draw, (170, 408), TEAL)
    stage_header(draw, (222, 372), "1", "Judgment Data", color=TEAL, status="N0-N1 · VERIFIED")
    pill(draw, (145, 500, 610, 564), "Raw judgments     76,291", fill=TEAL_LIGHT, color=TEAL, size=25)
    pill(draw, (145, 580, 610, 644), "Structured records  76,291", fill=GRAY_LIGHT, color=INK, size=25)
    draw.text((162, 662), "ID-aligned · no missing files", font=font(22), fill=MUTED)

    icon_filter(draw, (832, 408), TEAL)
    stage_header(draw, (892, 372), "2", "Rule Triage", color=TEAL, status="N2 · VERIFIED")
    draw.text((810, 493), "Deterministic three-way decision", font=font(25, True), fill=INK)
    pill(draw, (805, 548, 975, 606), "ACCEPT", fill=GREEN_LIGHT, color=GREEN, size=21)
    draw.text((1000, 556), "2,471", font=font(27, True), fill=GREEN)
    pill(draw, (805, 620, 975, 678), "REJECT", fill=GRAY_LIGHT, color=GRAY, size=21)
    draw.text((1000, 628), "71,296", font=font(27, True), fill=GRAY)
    pill(draw, (1135, 584, 1325, 642), "DEFER 2,524", fill=ORANGE_LIGHT, color=ORANGE, size=21)

    icon_network(draw, (1548, 408), ORANGE)
    stage_header(draw, (1608, 372), "3", "Qwen Resolver", color=ORANGE, status="N3 · VERIFIED")
    draw.text((1520, 493), "Bounded relation judgment", font=font(25, True), fill=INK)
    pill(draw, (1520, 548, 1750, 606), "+431 ACCEPT", fill=GREEN_LIGHT, color=GREEN, size=21)
    pill(draw, (1770, 548, 2025, 606), "1,357 REJECT", fill=GRAY_LIGHT, color=GRAY, size=21)
    pill(draw, (1520, 626, 2025, 684), "736 ABSTAIN · low-confidence review", fill=RED_LIGHT, color=RED, size=21)

    draw.text((2230, 370), "Selective output cohort", font=font(31, True), fill=INK)
    draw.text((2232, 424), "Only evidence-bearing candidates advance.", font=font(23), fill=MUTED)
    pill(draw, (2230, 490, 2850, 574), "2,902  FINAL CAR-TO-CAR", fill=GREEN, color=PAPER, size=30)
    draw.text((2250, 612), "Rule 2,471", font=font(24, True), fill=TEAL)
    draw.text((2460, 612), "+", font=font(24, True), fill=MUTED)
    draw.text((2510, 612), "Qwen 431", font=font(24, True), fill=ORANGE)
    draw.text((2690, 612), "=", font=font(24, True), fill=MUTED)
    draw.text((2740, 612), "2,902", font=font(24, True), fill=GREEN)
    draw.text((2250, 663), "LLM routing ratio: 3.308%", font=font(23), fill=MUTED)

    rounded_panel(
        draw,
        (70, 826, 2930, 1495),
        fill="#F4F7FC",
        outline="#B9CEE5",
        width=3,
        radius=34,
        dashed=not implementation_status,
    )
    draw.text((105, 850), "B  EXECUTABLE ASSURANCE", font=font(30, True), fill=BLUE)
    assurance_caption = (
        "Static implementation verified; external CARLA/VLM pending"
        if implementation_status
        else "Implementation and evaluation target"
    )
    draw.text((522, 854), assurance_caption, font=font(25), fill=MUTED)
    pill(draw, (2465, 846, 2888, 898), "2,370 minimum-grounded", fill=BLUE_LIGHT, color=BLUE, size=22)

    evidence_box = (105, 935, 755, 1388)
    contract_box = (830, 935, 1480, 1388)
    simulator_box = (1555, 935, 2185, 1388)
    assurance_box = (2260, 935, 2895, 1388)

    arrow(draw, (755, 1158), (818, 1158), color=BLUE)
    arrow(draw, (1480, 1158), (1543, 1158), color=BLUE)
    arrow(draw, (2185, 1158), (2248, 1158), color=BLUE)

    rounded_panel(draw, evidence_box, outline=BLUE, width=4, dashed=not implementation_status)
    rounded_panel(draw, contract_box, outline=BLUE, width=4, dashed=not implementation_status)
    rounded_panel(draw, simulator_box, outline=BLUE, width=4, dashed=not implementation_status)
    rounded_panel(draw, assurance_box, outline=RED, width=4, dashed=True)

    icon_network(draw, (175, 1018), BLUE)
    stage_header(
        draw,
        (235, 980),
        "4",
        "Evidence Graph",
        color=BLUE,
        status="N4 · STATIC VERIFIED" if implementation_status else "N4 · TO IMPLEMENT",
    )
    nodes = [
        (220, 1160, "vehicle"),
        (415, 1100, "event"),
        (595, 1165, "target"),
        (415, 1280, "evidence"),
    ]
    for nx, ny, label in nodes:
        draw.line((415, 1100, nx, ny), fill="#8CAFD2", width=4)
        draw.ellipse((nx - 39, ny - 39, nx + 39, ny + 39), fill=BLUE_LIGHT, outline=BLUE, width=3)
        text_center(draw, (nx - 65, ny + 45, nx + 65, ny + 78), label, size=20, color=INK)
    draw.text((145, 1354), "actor · target · order · supported_by", font=font(19), fill=MUTED)

    icon_contract(draw, (905, 1018), BLUE)
    stage_header(
        draw,
        (965, 980),
        "5",
        "Scenario Contract",
        color=BLUE,
        status="N4 · STATIC VERIFIED" if implementation_status else "N4 · TO IMPLEMENT",
    )
    contract_rows = [
        ("actors / map", "observed"),
        ("maneuvers", "inferred"),
        ("speed / pose", "unknown"),
        ("fallback assets", "defaulted"),
    ]
    y = 1100
    state_colors = {
        "observed": GREEN,
        "inferred": BLUE,
        "unknown": RED,
        "defaulted": YELLOW,
    }
    for field, state in contract_rows:
        draw.text((895, y + 7), field, font=font(22), fill=INK)
        pill(
            draw,
            (1190, y, 1420, y + 48),
            state,
            fill=state_colors[state],
            color=PAPER,
            size=19,
        )
        y += 62
    draw.text((895, 1350), "hard / soft / unknown guard", font=font(22, True), fill=BLUE)

    icon_car(draw, (1635, 1018), BLUE)
    stage_header(
        draw,
        (1705, 980),
        "6",
        "Simulation Stack",
        color=BLUE,
        status="N5 · DRY-RUN VERIFIED" if implementation_status else "N5 · TO IMPLEMENT",
    )
    stack_rows = (
        [
            ("Contract adapter", "#DCEAF7"),
            ("Scenic source generator", "#C7DDF2"),
            ("CARLA external runner", "#AFCFEA"),
        ]
        if implementation_status
        else [
            ("ChatScene adapter", "#DCEAF7"),
            ("Scenic compiler", "#C7DDF2"),
            ("CARLA runtime", "#AFCFEA"),
        ]
    )
    y = 1095
    for label, fill in stack_rows:
        draw.rounded_rectangle((1625, y, 2115, y + 66), radius=14, fill=fill, outline=BLUE, width=2)
        text_center(draw, (1625, y, 2115, y + 66), label, size=23, color=NAVY, bold=True)
        y += 82
    stack_status = (
        "dry-run · external runtime pending"
        if implementation_status
        else "compile · run · telemetry"
    )
    pill(draw, (1650, 1350, 2088, 1406), stack_status, fill=BLUE, color=PAPER, size=19)

    icon_feedback(draw, (2340, 1018), RED)
    stage_header(
        draw,
        (2400, 980),
        "7",
        "Assurance Loop",
        color=RED,
        status="N6 · MOCK VERIFIED" if implementation_status else "N6 · TO IMPLEMENT",
    )
    loop_steps = [
        ("1", "Telemetry + keyframes"),
        ("2", "Semantic observation"),
        ("3", "Evaluator report"),
        ("4", "Attribute-level repair"),
    ]
    y = 1090
    for number, label in loop_steps:
        draw.ellipse((2325, y, 2367, y + 42), fill=RED)
        text_center(draw, (2325, y, 2367, y + 42), number, size=19, color=PAPER, bold=True)
        draw.text((2385, y + 5), label, font=font(22), fill=INK)
        if number != "4":
            draw.line((2346, y + 43, 2346, y + 58), fill=RED, width=4)
        y += 64
    pill(draw, (2325, 1350, 2828, 1406), "pass  |  repair  |  manual review", fill=RED_LIGHT, color=RED, size=20)

    poly_arrow(
        draw,
        [(2575, 1415), (2575, 1460), (1160, 1460), (1160, 1404)],
        color=RED,
        width=6,
        label="failed attribute only",
        label_xy=(1650, 1429),
    )

    draw.text((90, 1530), "Reference design patterns:", font=font(23, True), fill=MUTED)
    sources = [
        ("EI-Drive", "modular boundaries"),
        ("Virtual Roads", "data-to-simulation tiers"),
        ("ChatSync", "graph-grounded reasoning"),
        ("FuzzCoder", "deterministic + LLM"),
        ("Closed-Loop UAV", "execution feedback"),
    ]
    x = 400
    for title, pattern in sources:
        width = 470 if title == "Closed-Loop UAV" else 430
        draw.rounded_rectangle((x, 1512, x + width, 1588), radius=16, fill=GRAY_LIGHT)
        draw.text((x + 18, 1522), title, font=font(21, True), fill=INK)
        draw.text((x + 18, 1553), pattern, font=font(18), fill=MUTED)
        x += width + 18
    draw.text(
        (90, 1628),
        (
            "Solid N4-N5 modules passed static/dry-run validation; dashed N6/CARLA-VLM execution remains external."
            if implementation_status
            else "Solid modules are implemented and batch-verified; dashed modules are implementation targets."
        ),
        font=font(20),
        fill=MUTED,
    )

    output_name = (
        "fig1_jurisdrive_architecture_implementation_status.png"
        if implementation_status
        else "fig1_jurisdrive_architecture_refined.png"
    )
    image.save(OUTPUT_DIR / output_name, quality=96)


def branch_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    title: str,
    count: str,
    rate: str,
    color: str,
    fill: str,
) -> None:
    rounded_panel(draw, box, fill=PAPER, outline=color, width=4, radius=22)
    draw.rectangle((box[0], box[1], box[0] + 14, box[3]), fill=color)
    draw.text((box[0] + 38, box[1] + 22), title, font=font(25, True), fill=INK)
    draw.text((box[0] + 38, box[1] + 66), count, font=font(37, True), fill=color)
    pill(
        draw,
        (box[2] - 205, box[1] + 58, box[2] - 25, box[1] + 108),
        rate,
        fill=fill,
        color=color,
        size=19,
    )


def render_filter_refined() -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)

    draw.text((90, 54), "N-Stage Selective Filtering: 76,291 Legal Cases", font=font(61, True), fill=INK)
    draw.text(
        (92, 134),
        "Deterministic triage resolves 96.692%; Qwen is invoked only for the 3.308% deferred set",
        font=font(29),
        fill=MUTED,
    )

    pill(draw, (2100, 58, 2500, 118), "LLM ROUTING 3.308%", fill=ORANGE_LIGHT, color=ORANGE, size=23)
    pill(draw, (2520, 58, 2910, 118), "ZERO BATCH FAILURES", fill=TEAL_LIGHT, color=TEAL, size=23)

    rounded_panel(draw, (70, 218, 2930, 1175), fill=PAPER, outline=LINE, width=3, radius=32)
    draw.text((105, 246), "A  SELECTIVE CASCADE", font=font(29, True), fill=NAVY)
    draw.text((490, 251), "Accept · reject · defer decisions retain an explicit reason trail", font=font(24), fill=MUTED)

    input_box = (110, 440, 550, 790)
    rule_box = (700, 382, 1190, 848)
    qwen_box = (1910, 715, 2320, 1045)

    arrow(draw, (550, 615), (688, 615), color=TEAL, label="all records")
    poly_arrow(draw, [(1190, 505), (1275, 505), (1275, 405), (1370, 405)], color=GREEN, width=7)
    poly_arrow(draw, [(1190, 615), (1275, 615), (1275, 650), (1370, 650)], color=GRAY, width=7)
    poly_arrow(draw, [(1190, 742), (1275, 742), (1275, 895), (1370, 895)], color=ORANGE, width=7)
    arrow(draw, (1810, 895), (1898, 895), color=ORANGE)
    poly_arrow(draw, [(2320, 810), (2390, 810), (2390, 700), (2470, 700)], color=GREEN, width=7)
    poly_arrow(draw, [(2320, 895), (2410, 895), (2410, 895), (2470, 895)], color=GRAY, width=7)
    poly_arrow(draw, [(2320, 980), (2390, 980), (2390, 1090), (2470, 1090)], color=RED, width=7)

    rounded_panel(draw, input_box, outline=TEAL, width=5, radius=28)
    icon_document(draw, (205, 528), TEAL)
    draw.text((275, 486), "ZEROSHOT", font=font(28, True), fill=TEAL)
    draw.text((275, 528), "structured input", font=font(24), fill=INK)
    text_center(draw, (140, 610, 520, 720), "76,291", size=55, color=INK, bold=True)
    pill(draw, (155, 724, 505, 777), "100.000% population", fill=TEAL_LIGHT, color=TEAL, size=20)

    rounded_panel(draw, rule_box, outline=TEAL, width=5, radius=28)
    icon_filter(draw, (800, 478), TEAL)
    draw.text((875, 440), "RULE TRIAGE", font=font(29, True), fill=TEAL)
    draw.text((875, 486), "deterministic", font=font(23), fill=MUTED)
    draw.line((750, 560, 1140, 560), fill=LINE, width=2)
    rule_rows = [
        ("accept", "2,471", GREEN),
        ("reject", "71,296", GRAY),
        ("defer", "2,524", ORANGE),
    ]
    y = 590
    for label, count, color in rule_rows:
        pill(draw, (760, y, 930, y + 53), label.upper(), fill=color, color=PAPER, size=20)
        draw.text((970, y + 4), count, font=font(31, True), fill=color)
        y += 72
    draw.text((765, 810), "96.692% resolved without LLM", font=font(23, True), fill=TEAL)

    branch_card(
        draw,
        (1370, 320, 1810, 490),
        title="DIRECT ACCEPT",
        count="2,471",
        rate="3.239%",
        color=GREEN,
        fill=GREEN_LIGHT,
    )
    branch_card(
        draw,
        (1370, 565, 1810, 735),
        title="DIRECT REJECT",
        count="71,296",
        rate="93.453%",
        color=GRAY,
        fill=GRAY_LIGHT,
    )
    branch_card(
        draw,
        (1370, 810, 1810, 980),
        title="ROUTED TO QWEN",
        count="2,524",
        rate="3.308%",
        color=ORANGE,
        fill=ORANGE_LIGHT,
    )

    rounded_panel(draw, qwen_box, outline=ORANGE, width=5, radius=28)
    icon_network(draw, (1995, 802), ORANGE)
    draw.text((2070, 760), "QWEN", font=font(29, True), fill=ORANGE)
    draw.text((2070, 804), "bounded resolver", font=font(23), fill=MUTED)
    draw.line((1950, 867, 2280, 867), fill=LINE, width=2)
    qwen_rows = [
        ("accept", "431", GREEN),
        ("reject", "1,357", GRAY),
        ("abstain", "736", RED),
    ]
    y = 895
    for label, count, color in qwen_rows:
        draw.text((1960, y), label.upper(), font=font(21, True), fill=color)
        bounds = draw.textbbox((0, 0), count, font=font(27, True))
        draw.text((2260 - (bounds[2] - bounds[0]), y - 4), count, font=font(27, True), fill=color)
        y += 53

    branch_card(
        draw,
        (2470, 615, 2890, 785),
        title="QWEN ACCEPT",
        count="431",
        rate="17.076%",
        color=GREEN,
        fill=GREEN_LIGHT,
    )
    branch_card(
        draw,
        (2470, 810, 2890, 980),
        title="QWEN REJECT",
        count="1,357",
        rate="53.764%",
        color=GRAY,
        fill=GRAY_LIGHT,
    )
    branch_card(
        draw,
        (2470, 1005, 2890, 1170),
        title="ABSTAIN",
        count="736",
        rate="29.160%",
        color=RED,
        fill=RED_LIGHT,
    )

    rounded_panel(draw, (70, 1215, 2930, 1585), fill="#F5F7F8", outline=LINE, width=3, radius=32)
    draw.text((105, 1242), "B  FINAL ACCOUNTING", font=font(29, True), fill=NAVY)
    draw.text((465, 1247), "All branches reconcile to the original 76,291 records", font=font(24), fill=MUTED)

    summary_boxes = [
        ((110, 1320, 700, 1515), "CAR-TO-CAR", "2,902", "3.804% of input", GREEN, GREEN_LIGHT),
        ((750, 1320, 1340, 1515), "NOT CAR-TO-CAR", "72,653", "95.231% of input", GRAY, GRAY_LIGHT),
        ((1390, 1320, 1900, 1515), "UNRESOLVED", "736", "0.965% of input", RED, RED_LIGHT),
    ]
    for box, title, count, rate, color, fill in summary_boxes:
        rounded_panel(draw, box, fill=PAPER, outline=color, width=4, radius=24)
        draw.text((box[0] + 30, box[1] + 24), title, font=font(23, True), fill=color)
        draw.text((box[0] + 30, box[1] + 70), count, font=font(45, True), fill=INK)
        pill(draw, (box[0] + 30, box[1] + 137, box[2] - 30, box[1] + 181), rate, fill=fill, color=color, size=19)

    readiness_box = (1950, 1288, 2888, 1535)
    rounded_panel(draw, readiness_box, fill=PAPER, outline=BLUE, width=4, radius=24)
    draw.text((1985, 1318), "Scenario Contract readiness within 2,902 candidates", font=font(23, True), fill=INK)
    bar_x1, bar_y1, bar_x2, bar_y2 = 1985, 1382, 2850, 1448
    total_width = bar_x2 - bar_x1
    a_width = round(total_width * 0.8167)
    b_width = round(total_width * 0.0693)
    draw.rounded_rectangle((bar_x1, bar_y1, bar_x2, bar_y2), radius=15, fill=GRAY_LIGHT)
    draw.rounded_rectangle((bar_x1, bar_y1, bar_x1 + a_width, bar_y2), radius=15, fill=BLUE)
    draw.rectangle((bar_x1 + a_width - 15, bar_y1, bar_x1 + a_width + b_width, bar_y2), fill=YELLOW)
    draw.rounded_rectangle(
        (bar_x1 + a_width + b_width, bar_y1, bar_x2, bar_y2),
        radius=15,
        fill=RED,
    )
    text_center(draw, (bar_x1, bar_y1, bar_x1 + a_width, bar_y2), "Tier A  2,370 · 81.67%", size=21, color=PAPER, bold=True)
    legend = [
        ("A minimum-grounded", "2,370", BLUE),
        ("B defaults needed", "201", YELLOW),
        ("C review/re-extract", "331", RED),
    ]
    x = 1985
    for label, count, color in legend:
        draw.ellipse((x, 1483, x + 18, 1501), fill=color)
        draw.text((x + 28, 1474), f"{label}  {count}", font=font(18), fill=INK)
        x += 285

    draw.text(
        (90, 1625),
        "Accounting invariant: 2,902 + 72,653 + 736 = 76,291.  Tier A denotes minimum contract grounding, not CARLA readiness.",
        font=font(20),
        fill=MUTED,
    )

    image.save(OUTPUT_DIR / "fig2_n_stage_filter_refined.png", quality=96)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    render_architecture_refined()
    render_architecture_refined(implementation_status=True)
    render_filter_refined()
    print(f"Saved refined figures to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
