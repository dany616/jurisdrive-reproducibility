#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_DIR / "figures"
FONT_REGULAR = "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc"
FONT_BOLD = "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Bold.ttc"

BG = "#F4F0E8"
INK = "#16323A"
MUTED = "#567078"
TEAL = "#0F766E"
TEAL_LIGHT = "#D8EFE9"
ORANGE = "#D97706"
ORANGE_LIGHT = "#FCE8C7"
BLUE = "#2563A8"
BLUE_LIGHT = "#DCE9F7"
RED = "#B94343"
RED_LIGHT = "#F5D8D5"
GRAY = "#E5E2DA"
WHITE = "#FFFDF8"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size=size)


def rounded_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    fill: str,
    outline: str,
    width: int = 4,
    radius: int = 24,
    dashed: bool = False,
) -> None:
    if not dashed:
        draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)
        return
    draw.rounded_rectangle(box, radius=radius, fill=fill)
    x1, y1, x2, y2 = box
    dash = 18
    gap = 10
    for x in range(x1 + radius, x2 - radius, dash + gap):
        draw.line((x, y1, min(x + dash, x2 - radius), y1), fill=outline, width=width)
        draw.line((x, y2, min(x + dash, x2 - radius), y2), fill=outline, width=width)
    for y in range(y1 + radius, y2 - radius, dash + gap):
        draw.line((x1, y, x1, min(y + dash, y2 - radius)), fill=outline, width=width)
        draw.line((x2, y, x2, min(y + dash, y2 - radius)), fill=outline, width=width)


def centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    *,
    size: int,
    color: str = INK,
    bold: bool = False,
    spacing: int = 6,
) -> None:
    fnt = font(size, bold)
    bounds = draw.multiline_textbbox((0, 0), text, font=fnt, spacing=spacing, align="center")
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    x1, y1, x2, y2 = box
    draw.multiline_text(
        ((x1 + x2 - width) / 2, (y1 + y2 - height) / 2 - bounds[1]),
        text,
        font=fnt,
        fill=color,
        spacing=spacing,
        align="center",
    )


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: str = MUTED) -> None:
    draw.line((*start, *end), fill=color, width=8)
    x, y = end
    draw.polygon([(x, y), (x - 22, y - 14), (x - 22, y + 14)], fill=color)


def badge(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    fill: str,
    color: str,
    width: int,
) -> None:
    x, y = xy
    draw.rounded_rectangle((x, y, x + width, y + 66), radius=20, fill=fill)
    centered_text(draw, (x, y, x + width, y + 66), text, size=27, color=color, bold=True)


def render_architecture() -> None:
    image = Image.new("RGB", (2400, 1350), BG)
    draw = ImageDraw.Draw(image)

    draw.text((90, 55), "JurisDrive: Evidence-Carrying Legal-to-CARLA Framework", font=font(58, True), fill=INK)
    draw.text(
        (92, 130),
        "Solid border: implemented and batch-verified  ·  Dashed border: implementation target",
        font=font(27),
        fill=MUTED,
    )

    boxes = [
        (60, 245, 380, 700),
        (430, 245, 790, 700),
        (840, 245, 1200, 700),
        (1250, 245, 1590, 700),
        (1640, 245, 1960, 700),
        (2010, 245, 2360, 700),
    ]
    specs = [
        ("1", "Judgment Data", "Zeroshot structured\nrecords", "76,291", TEAL_LIGHT, TEAL, False),
        ("2", "Rule Filter", "Deterministic\nthree-way decision", "2,524 routed", TEAL_LIGHT, TEAL, False),
        ("3", "Qwen Resolver", "Ambiguous-only\nrelation judgment", "2,902 final car", ORANGE_LIGHT, ORANGE, False),
        ("4", "Scenario Layer", "Evidence Graph\n+ Contract", "2,370 minimum", BLUE_LIGHT, BLUE, True),
        ("5", "Simulation", "ChatScene / Scenic\n→ CARLA", "compile · run", BLUE_LIGHT, BLUE, True),
        ("6", "Assurance", "Telemetry + VLM\n→ attribute repair", "closed loop", RED_LIGHT, RED, True),
    ]

    for box, (number, title, body, metric, fill, color, dashed) in zip(boxes, specs):
        rounded_box(draw, box, fill=WHITE, outline=color, dashed=dashed)
        x1, y1, x2, _ = box
        box_width = x2 - x1
        title_size = 31 if box_width <= 350 else 33
        body_size = 28 if box_width <= 350 else 31
        draw.ellipse((x1 + 24, y1 + 20, x1 + 90, y1 + 86), fill=color)
        centered_text(draw, (x1 + 24, y1 + 20, x1 + 90, y1 + 86), number, size=30, color=WHITE, bold=True)
        draw.text((x1 + 108, y1 + 31), title, font=font(title_size, True), fill=INK)
        centered_text(draw, (x1 + 25, y1 + 120, x2 - 25, y1 + 280), body, size=body_size, color=INK)
        badge(draw, (x1 + 30, y1 + 330), metric, fill=fill, color=color, width=x2 - x1 - 60)

    for left, right in zip(boxes[:-1], boxes[1:]):
        arrow(draw, (left[2] + 10, 475), (right[0] - 12, 475))

    rounded_box(draw, (450, 760, 1270, 1080), fill=WHITE, outline=TEAL, width=3)
    draw.text((485, 792), "Verified N-stage filtering", font=font(34, True), fill=INK)
    draw.text((490, 855), "Rule", font=font(30, True), fill=TEAL)
    draw.text((630, 855), "2,471 accept", font=font(29), fill=INK)
    draw.text((875, 855), "71,296 reject", font=font(29), fill=INK)
    draw.text((490, 920), "Qwen", font=font(30, True), fill=ORANGE)
    draw.text((630, 920), "+431 accept", font=font(29), fill=INK)
    draw.text((875, 920), "1,357 reject", font=font(29), fill=INK)
    draw.text((630, 978), "736 unresolved → manual/re-extraction queue", font=font(27), fill=RED)

    rounded_box(draw, (1330, 760, 2350, 1080), fill=WHITE, outline=BLUE, width=3, dashed=True)
    draw.text((1365, 792), "Implementation and evaluation targets", font=font(34, True), fill=INK)
    target_lines = [
        "Evidence span provenance + actor/relation graph",
        "Scenario Contract with hard/soft/unknown fields",
        "CARLA compile rate, run rate, and constraint satisfaction",
        "Telemetry/VLM evaluation with attribute-level repair",
    ]
    for index, line in enumerate(target_lines):
        y = 855 + index * 55
        draw.ellipse((1368, y + 7, 1382, y + 21), fill=BLUE)
        draw.text((1400, y), line, font=font(27), fill=INK)

    draw.text((90, 1165), "Design patterns", font=font(30, True), fill=MUTED)
    references = [
        ("EI-Drive", "modular CARLA architecture"),
        ("Virtual Roads", "raw data → twin → safety"),
        ("ChatSync / FuzzCoder", "bounded reasoning + hybrid execution"),
        ("Closed-Loop UAV", "simulation → evaluator → repair"),
    ]
    x = 90
    for title, subtitle in references:
        draw.rounded_rectangle((x, 1210, x + 520, 1300), radius=18, fill=GRAY)
        draw.text((x + 24, 1224), title, font=font(25, True), fill=INK)
        draw.text((x + 24, 1260), subtitle, font=font(21), fill=MUTED)
        x += 560

    image.save(OUTPUT_DIR / "fig1_jurisdrive_architecture.png", quality=95)


def render_filter_funnel() -> None:
    image = Image.new("RGB", (2000, 1200), BG)
    draw = ImageDraw.Draw(image)
    draw.text((90, 55), "N-stage Selective Filtering From 76,291 Records", font=font(56, True), fill=INK)
    draw.text((92, 132), "Only uncertain cases consume LLM inference.", font=font(28), fill=MUTED)

    stages = [
        (160, 240, 1840, 385, TEAL, "Zeroshot structured records", "76,291", "100.000%"),
        (300, 430, 1700, 575, TEAL, "Rule decision", "2,471 accept  ·  71,296 reject  ·  2,524 route", "3.308% routed"),
        (450, 620, 1550, 765, ORANGE, "Qwen on ambiguous only", "431 accept  ·  1,357 reject  ·  736 unresolved", "29.160% remain unresolved"),
        (600, 810, 1400, 955, BLUE, "Final car-to-car candidate set", "2,902", "3.804% of input"),
        (735, 1000, 1265, 1135, BLUE, "Minimum grounded subset", "2,370", "81.67% of candidates"),
    ]
    for x1, y1, x2, y2, color, title, metric, rate in stages:
        draw.rounded_rectangle((x1, y1, x2, y2), radius=28, fill=WHITE, outline=color, width=5)
        draw.text((x1 + 36, y1 + 24), title, font=font(31, True), fill=INK)
        draw.text((x1 + 36, y1 + 74), metric, font=font(28), fill=color)
        rate_bounds = draw.textbbox((0, 0), rate, font=font(25, True))
        draw.text((x2 - 36 - (rate_bounds[2] - rate_bounds[0]), y1 + 86), rate, font=font(25, True), fill=MUTED)

    image.save(OUTPUT_DIR / "fig2_n_stage_filter_funnel.png", quality=95)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    render_architecture()
    render_filter_funnel()
    print(f"Saved figures to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
