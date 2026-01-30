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


async def request_logging_middleware(request: Request, call_next: Callable) -> Response:
    start = time.perf_counter()

    trace_id = request.headers.get("X-Request-Id") or f"req-{uuid.uuid4().hex[:12]}"
    user_id = request.headers.get("X-User-Id")

    service = os.getenv("SERVICE_NAME", "ai-api")
    app_env = os.getenv("APP_ENV")
    release = os.getenv("APP_RELEASE")

    error_type = None
    stacktrace = None
    status_code = 500
    message = "Request processed"

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
                "extra": {},
            },
            "error": {
                "type": error_type,
                "stacktrace": stacktrace,
            },
        }

        logger.log(
            logging.ERROR if error_type else logging.INFO,
            message,
            extra={"json_data": payload},
        )

    return response
