#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import json
import re
import socket
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib import error, request

SCRIPT_DIR = Path(__file__).resolve().parent
ZEROSHOT_DIR = SCRIPT_DIR.parent
if str(ZEROSHOT_DIR) not in sys.path:
    sys.path.insert(0, str(ZEROSHOT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from Structured_output_test import VLLMJobProxyServer, get_job_info
from llm_filter import build_llm_messages, fallback_llm_result, normalize_llm_result
from prompt_templates import LLM_OUTPUT_SCHEMA

try:
    from openai import OpenAI
except ImportError:
    class _ChatCompletionsAPI:
        def __init__(self, base_url: str, api_key: str, timeout: float = 600.0) -> None:
            self.base_url = base_url.rstrip("/")
            self.api_key = api_key
            self.timeout = timeout

        def create(
            self,
            *,
            model: str,
            temperature: float,
            messages: list[dict[str, str]],
            max_tokens: int | None = None,
            extra_body: dict[str, Any] | None = None,
        ) -> SimpleNamespace:
            payload: dict[str, Any] = {
                "model": model,
                "temperature": temperature,
                "messages": messages,
            }
            if max_tokens is not None:
                payload["max_tokens"] = max_tokens
            if extra_body:
                payload.update(extra_body)

            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            http_request = request.Request(
                url=f"{self.base_url}/chat/completions",
                data=body,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with request.urlopen(http_request, timeout=self.timeout) as response:
                    raw = response.read().decode("utf-8")
            except error.HTTPError as exc:
                details = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"HTTP {exc.code}: {details}") from exc
            except error.URLError as exc:
                raise RuntimeError(f"request failed: {exc.reason}") from exc

            data = json.loads(raw)
            content = data["choices"][0]["message"]["content"]
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )


    class _ChatAPI:
        def __init__(self, base_url: str, api_key: str) -> None:
            self.completions = _ChatCompletionsAPI(base_url=base_url, api_key=api_key)


    class OpenAI:
        def __init__(self, *, base_url: str, api_key: str) -> None:
            self.chat = _ChatAPI(base_url=base_url, api_key=api_key)


DEFAULT_INPUT_DIR = SCRIPT_DIR / "full_run" / "output" / "ambiguous"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "full_run" / "ambiguous_done"
DEFAULT_REPORT_PATH = DEFAULT_OUTPUT_ROOT / "report.jsonl"
DEFAULT_REASON_LOG_PATH = DEFAULT_OUTPUT_ROOT / "reason_log.jsonl"
DEFAULT_SUMMARY_PATH = DEFAULT_OUTPUT_ROOT / "summary.json"
DEFAULT_PROCESS_LOG_PATH = DEFAULT_OUTPUT_ROOT / "process.log"
DEFAULT_TIME_PATH = DEFAULT_OUTPUT_ROOT / "time.txt"
DEFAULT_MODEL_NAME = "qwen35-27b"
DEFAULT_PROXY_PORT = 18001
INPUT_NAME_RE = re.compile(r"zeroshot_test_(\d+)_result\.json$")

LABELS = ("car_to_car", "not_car_to_car", "ambiguous")

LOG_LOCK = threading.Lock()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def kst_now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def natural_sort_key(path: Path) -> tuple[int, str]:
    match = INPUT_NAME_RE.fullmatch(path.name)
    if match:
        return int(match.group(1)), path.name
    return sys.maxsize, path.name


def discover_input_files(input_dir: Path) -> list[Path]:
    files = [path for path in input_dir.glob("zeroshot_test_*_result.json") if INPUT_NAME_RE.fullmatch(path.name)]
    return sorted(files, key=natural_sort_key)


def normalize_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"
    return normalized


def reserve_port(preferred_port: int) -> int:
    if preferred_port > 0:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", preferred_port))
                return preferred_port
            except OSError:
                pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def build_client(
    *,
    api_base_url: str | None,
    job_id: str | None,
    ensure_job: bool,
    proxy_port: int,
) -> tuple[OpenAI, VLLMJobProxyServer | None, Any | None]:
    if api_base_url:
        client = OpenAI(base_url=normalize_base_url(api_base_url), api_key="EMPTY")
        return client, None, None

    job_info = get_job_info(job_id=job_id, ensure=ensure_job)
    job_info.require_running()
    port = reserve_port(proxy_port)
    proxy = VLLMJobProxyServer(job_id=job_info.job_id, port=port).start()
    client = OpenAI(base_url=proxy.base_url, api_key="EMPTY")
    return client, proxy, job_info


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with LOG_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def append_process_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{kst_now_text()}] {message}\n"
    with LOG_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)


def ensure_output_dirs(output_root: Path) -> dict[str, Path]:
    directories = {label: output_root / label for label in LABELS}
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def existing_output_path(output_root: Path, name: str) -> Path | None:
    for label in LABELS:
        candidate = output_root / label / name
        if candidate.exists():
            return candidate
    return None


def extract_llm_result(
    client: OpenAI,
    *,
    model_name: str,
    record: dict[str, Any],
    rule_result: dict[str, Any],
    max_tokens: int,
) -> tuple[str, dict[str, Any]]:
    response = client.chat.completions.create(
        model=model_name,
        temperature=0.0,
        max_tokens=max_tokens,
        messages=build_llm_messages(record, rule_result),
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "car_to_car_second_stage",
                    "schema": LLM_OUTPUT_SCHEMA,
                },
            },
        },
    )
    raw_content = response.choices[0].message.content
    return raw_content, normalize_llm_result(raw_content)


def build_report_row(
    *,
    input_path: Path,
    record: dict[str, Any],
    rule_result: dict[str, Any],
    llm_result: dict[str, Any],
    final_label: str,
    elapsed_seconds: float,
    status: str,
    error_message: str | None,
) -> dict[str, Any]:
    return {
        "input_file": input_path.name,
        "source_input_file": record.get("input_file"),
        "status": status,
        "rule_label": rule_result.get("label"),
        "rule_score": rule_result.get("score"),
        "rule_reason": rule_result.get("reason"),
        "llm_label": llm_result.get("label"),
        "llm_confidence": llm_result.get("confidence"),
        "llm_reason": llm_result.get("reason"),
        "llm_evidence": llm_result.get("evidence"),
        "final_label": final_label,
        "elapsed_seconds": round(elapsed_seconds, 4),
        "error": error_message,
    }


def build_reason_row(
    *,
    input_path: Path,
    llm_result: dict[str, Any],
    final_label: str,
) -> dict[str, Any]:
    return {
        "input_file": input_path.name,
        "final_label": final_label,
        "llm_label": llm_result.get("label"),
        "confidence": llm_result.get("confidence"),
        "reason": llm_result.get("reason"),
        "evidence": llm_result.get("evidence"),
    }


def process_one_file_sync(
    *,
    input_path: Path,
    output_root: Path,
    output_dirs: dict[str, Path],
    report_path: Path,
    reason_log_path: Path,
    process_log_path: Path,
    index: int,
    total: int,
    skip_existing: bool,
    client: OpenAI,
    model_name: str,
    max_tokens: int,
) -> dict[str, Any]:
    existing = existing_output_path(output_root, input_path.name)
    if skip_existing and existing is not None:
        append_process_log(process_log_path, f"[{index}/{total}] skipped {input_path.name} -> {existing.parent.name}")
        print(f"[{index}/{total}] skipped -> {existing.name}", flush=True)
        return {"status": "skipped", "label": existing.parent.name, "elapsed_seconds": 0.0}

    started = time.perf_counter()
    record = read_json(input_path)
    rule_result = ((record.get("postprocess") or {}).get("rule")) or {}
    raw_content: str | None = None
    error_message: str | None = None

    try:
        raw_content, llm_result = extract_llm_result(
            client,
            model_name=model_name,
            record=record,
            rule_result=rule_result,
            max_tokens=max_tokens,
        )
        final_label = llm_result.get("label") or "ambiguous"
        status = "ok"
    except Exception as exc:
        llm_result = fallback_llm_result(reason=f"LLM call failed: {str(exc).strip() or exc.__class__.__name__}")
        final_label = "ambiguous"
        status = "error"
        error_message = str(exc).strip() or exc.__class__.__name__

    elapsed_seconds = time.perf_counter() - started
    destination = output_dirs[final_label] / input_path.name

    enriched = dict(record)
    postprocess = dict(record.get("postprocess") or {})
    postprocess["llm"] = llm_result
    postprocess["llm_raw"] = raw_content
    postprocess["final_label"] = final_label
    postprocess["llm_processed_at"] = utc_now_iso()
    postprocess["llm_elapsed_seconds"] = round(elapsed_seconds, 4)
    if error_message:
        postprocess["llm_error"] = error_message
    enriched["postprocess"] = postprocess
    write_json(destination, enriched)

    append_jsonl(
        report_path,
        build_report_row(
            input_path=input_path,
            record=record,
            rule_result=rule_result,
            llm_result=llm_result,
            final_label=final_label,
            elapsed_seconds=elapsed_seconds,
            status=status,
            error_message=error_message,
        ),
    )
    append_jsonl(
        reason_log_path,
        build_reason_row(
            input_path=input_path,
            llm_result=llm_result,
            final_label=final_label,
        ),
    )
    append_process_log(
        process_log_path,
        f"[{index}/{total}] {status} {input_path.name} -> {final_label} ({elapsed_seconds:.2f}s)",
    )
    print(f"[{index}/{total}] {status} -> {destination.name} [{final_label}] ({elapsed_seconds:.2f}s)", flush=True)

    return {
        "status": status,
        "label": final_label,
        "elapsed_seconds": elapsed_seconds,
    }


async def process_one_file(
    input_path: Path,
    sem: asyncio.Semaphore,
    *,
    output_root: Path,
    output_dirs: dict[str, Path],
    report_path: Path,
    reason_log_path: Path,
    process_log_path: Path,
    index: int,
    total: int,
    skip_existing: bool,
    client: OpenAI,
    model_name: str,
    max_tokens: int,
) -> dict[str, Any]:
    async with sem:
        return await asyncio.to_thread(
            process_one_file_sync,
            input_path=input_path,
            output_root=output_root,
            output_dirs=output_dirs,
            report_path=report_path,
            reason_log_path=reason_log_path,
            process_log_path=process_log_path,
            index=index,
            total=total,
            skip_existing=skip_existing,
            client=client,
            model_name=model_name,
            max_tokens=max_tokens,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM second-stage classification for ambiguous car-to-car cases.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--ensure-job", action="store_true")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--api-base-url", default=None)
    parser.add_argument("--proxy-port", type=int, default=DEFAULT_PROXY_PORT)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--reset-logs", action="store_true")
    return parser.parse_args()


async def async_main() -> int:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    report_path = (output_root / "report.jsonl").resolve()
    reason_log_path = (output_root / "reason_log.jsonl").resolve()
    summary_path = (output_root / "summary.json").resolve()
    process_log_path = (output_root / "process.log").resolve()
    time_path = (output_root / "time.txt").resolve()

    if args.max_concurrency < 1:
        raise SystemExit("--max-concurrency must be at least 1")
    if args.start_index < 1:
        raise SystemExit("--start-index must be at least 1")

    files = discover_input_files(input_dir)
    files = files[args.start_index - 1 :]
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"No ambiguous input files found in {input_dir}")

    output_dirs = ensure_output_dirs(output_root)
    if args.reset_logs:
        for path in [report_path, reason_log_path, process_log_path]:
            if path.exists():
                path.unlink()

    started_at = utc_now_iso()
    append_process_log(process_log_path, f"run_started total={len(files)} job_id={args.job_id or 'auto'} model={args.model_name}")

    client: OpenAI
    proxy: VLLMJobProxyServer | None = None
    job_info: Any | None = None
    client, proxy, job_info = build_client(
        api_base_url=args.api_base_url,
        job_id=args.job_id,
        ensure_job=args.ensure_job,
        proxy_port=args.proxy_port,
    )
    job_id_value = job_info.job_id if job_info is not None else None

    try:
        sem = asyncio.Semaphore(args.max_concurrency)
        total = len(files)
        tasks = [
            process_one_file(
                input_path,
                sem,
                output_root=output_root,
                output_dirs=output_dirs,
                report_path=report_path,
                reason_log_path=reason_log_path,
                process_log_path=process_log_path,
                index=index,
                total=total,
                skip_existing=args.skip_existing,
                client=client,
                model_name=args.model_name,
                max_tokens=args.max_tokens,
            )
            for index, input_path in enumerate(files, start=1)
        ]
        run_started = time.perf_counter()
        results = await asyncio.gather(*tasks, return_exceptions=False)
        elapsed_seconds = time.perf_counter() - run_started
    finally:
        if proxy is not None:
            proxy.stop()

    processed = sum(1 for result in results if result["status"] in {"ok", "error"})
    skipped = sum(1 for result in results if result["status"] == "skipped")
    success = sum(1 for result in results if result["status"] == "ok")
    failed = sum(1 for result in results if result["status"] == "error")
    car_to_car = sum(1 for result in results if result["label"] == "car_to_car")
    not_car_to_car = sum(1 for result in results if result["label"] == "not_car_to_car")
    ambiguous = sum(1 for result in results if result["label"] == "ambiguous")

    summary = {
        "started_at": started_at,
        "finished_at": utc_now_iso(),
        "job_id": job_id_value,
        "model_name": args.model_name,
        "input_dir": str(input_dir),
        "output_root": str(output_root),
        "selected_files": len(files),
        "processed": processed,
        "skipped": skipped,
        "success": success,
        "failed": failed,
        "car_to_car": car_to_car,
        "not_car_to_car": not_car_to_car,
        "ambiguous": ambiguous,
        "max_tokens": args.max_tokens,
        "max_concurrency": args.max_concurrency,
        "elapsed_seconds": round(elapsed_seconds, 4),
        "avg_seconds_per_processed": round(elapsed_seconds / processed, 4) if processed else None,
        "report_path": str(report_path),
        "reason_log_path": str(reason_log_path),
        "process_log_path": str(process_log_path),
        "time_path": str(time_path),
    }
    write_json(summary_path, summary)

    time_lines = [
        f"started_at={started_at}",
        f"finished_at={summary['finished_at']}",
        f"elapsed_seconds={summary['elapsed_seconds']}",
        f"avg_seconds_per_processed={summary['avg_seconds_per_processed']}",
        f"selected_files={len(files)}",
        f"processed={processed}",
        f"success={success}",
        f"failed={failed}",
        f"skipped={skipped}",
        f"car_to_car={car_to_car}",
        f"not_car_to_car={not_car_to_car}",
        f"ambiguous={ambiguous}",
    ]
    time_path.write_text("\n".join(time_lines) + "\n", encoding="utf-8")
    append_process_log(process_log_path, f"run_finished processed={processed} success={success} failed={failed} elapsed={elapsed_seconds:.2f}s")

    print("완료", flush=True)
    print(f"Processed: {processed}", flush=True)
    print(f"Success: {success}", flush=True)
    print(f"Failed: {failed}", flush=True)
    print(f"Skipped: {skipped}", flush=True)
    print(f"Car-to-car: {car_to_car}", flush=True)
    print(f"Not car-to-car: {not_car_to_car}", flush=True)
    print(f"Ambiguous: {ambiguous}", flush=True)
    print(f"Elapsed: {elapsed_seconds:.2f}s", flush=True)
    print(f"Summary: {summary_path}", flush=True)
    return 0


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())
