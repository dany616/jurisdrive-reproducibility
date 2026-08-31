#!/usr/bin/env python3
"""Smoke-test an OpenAI-compatible vLLM endpoint with text and image inputs."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import time
import urllib.request
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def request_json(url: str, payload: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="GET" if payload is None else "POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
        return {"status": response.status, "elapsed_seconds": time.perf_counter() - started, "body": body}


def response_content(response: dict[str, Any]) -> dict[str, Any]:
    text = response["body"]["choices"][0]["message"]["content"]
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object response")
    return value


def schema(name: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def create_test_image(path: Path) -> str:
    image = Image.new("RGB", (960, 480), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 72)
    except OSError:
        font = ImageFont.load_default()
    draw.rectangle((55, 75, 285, 305), fill="#e53935")
    draw.ellipse((675, 75, 905, 305), fill="#1e66f5")
    text = "JurisDrive VLM"
    box = draw.textbbox((0, 0), text, font=font)
    draw.text(((960 - (box[2] - box[0])) / 2, 350), text, fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    api_base = args.api_base.rstrip("/")
    if not api_base.endswith("/v1"):
        api_base += "/v1"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_path = args.output_dir / "vlm_smoke.png"

    models = request_json(f"{api_base}/models", None, args.timeout)
    model_ids = [item["id"] for item in models["body"].get("data", []) if item.get("id")]
    if not model_ids:
        raise RuntimeError("The endpoint returned no served model ID")
    model = model_ids[0]

    common = {
        "model": model,
        "temperature": 0,
        "seed": 42,
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    text_payload = {
        **common,
        "messages": [
            {
                "role": "user",
                "content": "Return JSON confirming this is a text-only JurisDrive API smoke test.",
            }
        ],
        "response_format": schema(
            "jurisdrive_text_smoke",
            {
                "status": {"type": "string", "enum": ["ok"]},
                "mode": {"type": "string", "enum": ["text_only"]},
            },
            ["status", "mode"],
        ),
    }
    text_response = request_json(f"{api_base}/chat/completions", text_payload, args.timeout)
    text_value = response_content(text_response)
    text_passed = text_value == {"status": "ok", "mode": "text_only"}

    image_url = create_test_image(image_path)
    image_payload = {
        **common,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {
                        "type": "text",
                        "text": "Inspect the image. Report whether an image is present, its center text, and the two dominant shape colors.",
                    },
                ],
            }
        ],
        "response_format": schema(
            "jurisdrive_image_smoke",
            {
                "image_present": {"type": "boolean"},
                "center_text": {"type": "string"},
                "colors": {"type": "array", "items": {"type": "string"}, "minItems": 2},
            },
            ["image_present", "center_text", "colors"],
        ),
    }
    image_response = request_json(f"{api_base}/chat/completions", image_payload, args.timeout)
    image_value = response_content(image_response)
    image_passed = bool(image_value.get("image_present")) and "jurisdrive" in str(
        image_value.get("center_text", "")
    ).lower()

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "api_base": api_base,
        "served_model_id": model,
        "request_settings": {"temperature": 0, "seed": 42, "max_tokens": 256, "thinking": False},
        "text": {
            "passed": text_passed,
            "http_status": text_response["status"],
            "elapsed_seconds": round(text_response["elapsed_seconds"], 3),
            "parsed": text_value,
        },
        "image": {
            "passed": image_passed,
            "http_status": image_response["status"],
            "elapsed_seconds": round(image_response["elapsed_seconds"], 3),
            "path": str(image_path),
            "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            "parsed": image_value,
        },
        "passed": text_passed and image_passed,
    }
    (args.output_dir / "vllm_smoke.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "vllm_text_raw.json").write_text(
        json.dumps(text_response["body"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "vllm_image_raw.json").write_text(
        json.dumps(image_response["body"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
