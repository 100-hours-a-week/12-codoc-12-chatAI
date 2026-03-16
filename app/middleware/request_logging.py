import json
import logging
import os
import time
import traceback
import uuid
from datetime import datetime, timezone
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse


logger = logging.getLogger("request")


def _iso_now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _truncate_payload(payload: object, max_bytes: int) -> tuple[object, bool]:
    if payload is None:
        return None, False
    try:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    except Exception:  # noqa: BLE001
        raw = str(payload).encode("utf-8", errors="replace")
    if len(raw) <= max_bytes:
        return payload, False
    trimmed = raw[:max_bytes].decode("utf-8", errors="replace")
    return {"_truncated": True, "preview": trimmed, "size": len(raw)}, True


def _get_otel_trace_id() -> str | None:
    try:
        from opentelemetry import trace  # type: ignore
    except Exception:  # noqa: BLE001
        return None
    span = trace.get_current_span()
    if not span:
        return None
    ctx = span.get_span_context()
    if not ctx or not ctx.trace_id:
        return None
    return f"{ctx.trace_id:032x}"


async def request_logging_middleware(request: Request, call_next: Callable) -> Response:
    start = time.perf_counter()

    trace_id = _get_otel_trace_id()
    if not trace_id:
        trace_id = request.headers.get("X-Request-Id") or f"req-{uuid.uuid4().hex[:12]}"
    user_id = request.headers.get("X-User-Id")

    service = os.getenv("SERVICE_NAME", "ai-api")
    app_env = os.getenv("APP_ENV")
    release = os.getenv("APP_RELEASE")

    error_type = None
    stacktrace = None
    status_code = 500
    message = "Request processed"
    body_payload: object | None = None
    body_truncated = False
    max_body_bytes = int(os.getenv("LOG_BODY_MAX_BYTES", "2048"))

    try:
        body_bytes = await request.body()
        request._body = body_bytes  # allow downstream handlers to read again
        if body_bytes:
            body_text = body_bytes.decode("utf-8", errors="replace")
            try:
                body_payload = json.loads(body_text)
            except json.JSONDecodeError:
                body_payload = {"_raw": body_text}
    except Exception:  # noqa: BLE001
        body_payload = {"_raw": "<failed to read body>"}

    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception as exc:  # noqa: BLE001
        error_type = exc.__class__.__name__
        stacktrace = traceback.format_exc()
        message = "Unhandled exception"
        logger.exception("Unhandled exception", extra={"trace_id": trace_id})
        response = JSONResponse(status_code=500, content={"detail": "Internal Server Error"})
    finally:
        latency_ms = int((time.perf_counter() - start) * 1000)
        body_payload, body_truncated = _truncate_payload(body_payload, max_body_bytes)
        payload = {
            "timestamp": _iso_now_utc(),
            "level": "ERROR" if error_type else "INFO",
            "service": service,
            "env": app_env,
            "release": release,
            "trace_id": trace_id,
            "path": request.url.path,
            "method": request.method,
            "status": status_code,
            "latency_ms": latency_ms,
            "message": message,
            "context": {
                "user_id": user_id or None,
                "client_ip": None,  # PII 보호
                "extra": {
                    "body": body_payload,
                    "body_truncated": body_truncated,
                },
            },
        }
        if error_type:
            payload["error"] = {
                "type": error_type,
                "stacktrace": stacktrace,
            }

        logger.log(
            logging.ERROR if error_type else logging.INFO,
            message,
            extra={"json_data": payload},
        )

    return response
