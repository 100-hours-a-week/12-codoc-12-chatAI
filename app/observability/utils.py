import time
from typing import Tuple

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_client import REGISTRY, Counter, Gauge, Histogram
from prometheus_client.openmetrics.exposition import CONTENT_TYPE_LATEST, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Match
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR
from starlette.types import ASGIApp

INFO = Gauge(
    "fastapi_app_info",
    "FastAPI application information.",
    ["app_name"],
)
REQUESTS = Counter(
    "fastapi_requests_total",
    "Total count of requests by method and path.",
    ["method", "path", "app_name"],
)
RESPONSES = Counter(
    "fastapi_responses_total",
    "Total count of responses by method, path and status codes.",
    ["method", "path", "status_code", "app_name"],
)
REQUESTS_PROCESSING_TIME = Histogram(
    "fastapi_requests_duration_seconds",
    "Histogram of requests processing time by path (in seconds).",
    ["method", "path", "app_name"],
)
EXCEPTIONS = Counter(
    "fastapi_exceptions_total",
    "Total count of exceptions raised by path and exception type.",
    ["method", "path", "exception_type", "app_name"],
)
REQUESTS_IN_PROGRESS = Gauge(
    "fastapi_requests_in_progress",
    "Gauge of requests by method and path currently being processed.",
    ["method", "path", "app_name"],
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, app_name: str = "app_chatai") -> None:
        super().__init__(app)
        self.app_name = app_name
        INFO.labels(app_name=self.app_name).set(1)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        method = request.method
        path, is_handled_path = self.get_path(request)

        if not is_handled_path:
            return await call_next(request)

        labels = {"method": method, "path": path, "app_name": self.app_name}
        REQUESTS_IN_PROGRESS.labels(**labels).inc()
        REQUESTS.labels(**labels).inc()
        before_time = time.perf_counter()
        status_code = HTTP_500_INTERNAL_SERVER_ERROR

        try:
            response = await call_next(request)
            status_code = response.status_code
            after_time = time.perf_counter()
            span = trace.get_current_span()
            trace_id = trace.format_trace_id(span.get_span_context().trace_id)

            exemplar = {"TraceID": trace_id} if trace_id and int(trace_id, 16) != 0 else None
            if exemplar:
                REQUESTS_PROCESSING_TIME.labels(**labels).observe(after_time - before_time, exemplar=exemplar)
            else:
                REQUESTS_PROCESSING_TIME.labels(**labels).observe(after_time - before_time)
            return response
        except BaseException as exc:
            EXCEPTIONS.labels(
                method=method,
                path=path,
                exception_type=type(exc).__name__,
                app_name=self.app_name,
            ).inc()
            raise
        finally:
            RESPONSES.labels(
                method=method,
                path=path,
                status_code=str(status_code),
                app_name=self.app_name,
            ).inc()
            REQUESTS_IN_PROGRESS.labels(**labels).dec()

    @staticmethod
    def get_path(request: Request) -> Tuple[str, bool]:
        return resolve_request_path(request)


def resolve_request_path(request: Request) -> Tuple[str, bool]:
    for route in request.app.routes:
        match, _ = route.matches(request.scope)
        if match == Match.FULL:
            return route.path, True
    return request.url.path, False


def record_exception_metric(request: Request, exc: BaseException, app_name: str) -> None:
    path, is_handled_path = resolve_request_path(request)
    if not is_handled_path:
        return

    EXCEPTIONS.labels(
        method=request.method,
        path=path,
        exception_type=type(exc).__name__,
        app_name=app_name,
    ).inc()


def metrics(_: Request) -> Response:
    return Response(generate_latest(REGISTRY), headers={"Content-Type": CONTENT_TYPE_LATEST})


def setting_otlp(app: ASGIApp, app_name: str, endpoint: str, log_correlation: bool = False) -> None:
    resource = Resource.create(attributes={"service.name": app_name})

    tracer = TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer)
    tracer.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True)))

    # Keep our JSON logger intact; we only want OTEL context attached to records.
    if log_correlation:
        LoggingInstrumentor().instrument(set_logging_format=False)

    FastAPIInstrumentor.instrument_app(app, tracer_provider=tracer)
