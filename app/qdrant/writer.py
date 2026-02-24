import json
import logging
import os
from datetime import datetime, timezone
from typing import Iterable, Optional

from qdrant_client import QdrantClient, models

from app.common.config import settings

logger = logging.getLogger(__name__)
OBS_ENABLED = os.getenv("QDRANT_OBS", "false").lower() == "true"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _extract_run_id(points: Iterable[models.PointStruct]) -> Optional[str]:
    for p in points:
        payload = getattr(p, "payload", None) or {}
        tags = payload.get("tags") or []
        for t in tags:
            if isinstance(t, str) and t.startswith("run="):
                return t.split("=", 1)[1]
    return None

def _point_ids(points: Iterable[models.PointStruct]) -> list[str]:
    return [str(p.id) for p in points]

def _log_obs(
    event: str,
    target: str,
    collection: str,
    points: Iterable[models.PointStruct],
    exc: Optional[Exception] = None,
    endpoint: Optional[dict[str, str | int]] = None,
) -> None:
    if not OBS_ENABLED:
        return
    pts = list(points)
    payload = {
        "ts": _now_iso(),
        "event": event,
        "target": target,
        "collection": collection,
        "point_ids": _point_ids(pts),
        "run_id": _extract_run_id(pts),
        "error": str(exc) if exc else None,
    }
    if endpoint:
        payload["endpoint"] = endpoint
    logger.info("QDRANT_OBS %s", json.dumps(payload, ensure_ascii=False))


class QdrantWriter:
    """
    primary 동기 write + secondary best-effort write.
    secondary 실패는 로컬 JSONL 로그에 적재하여 재시도한다.
    """

    def __init__(
        self,
        primary: QdrantClient,
        secondary: Optional[QdrantClient],
        retry_log_path: str = "/tmp/qdrant_retry.log",
        primary_host: str = "",
        primary_port: int = 0,
        secondary_host: str = "",
        secondary_port: int = 0,
    ):
        self.primary = primary
        self.secondary = secondary
        self.retry_log_path = retry_log_path
        self.primary_host = primary_host
        self.primary_port = primary_port
        self.secondary_host = secondary_host
        self.secondary_port = secondary_port

    def is_already_exists_error(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        return "already exists" in msg

    def ensure_collection(self, collection_name: str, vector_size: int) -> None:
        if not self.primary.collection_exists(collection_name):
            try:
                self.primary.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size, distance=models.Distance.COSINE
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                # already exists race -> ignore
                if not self.is_already_exists_error(exc):
                    raise

        if self.secondary and not self.secondary.collection_exists(collection_name):
            try:
                self.secondary.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size, distance=models.Distance.COSINE
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                # already exists race -> ignore
                if not self.is_already_exists_error(exc):
                    raise

    def upsert(self, collection_name: str, points: Iterable[models.PointStruct]) -> None:
        points = list(points)
        # primary 동기 write
        try:
            self.primary.upsert(collection_name=collection_name, points=points)
            _log_obs(
                "write_ok",
                "primary",
                collection_name,
                points,
                endpoint={"host": self.primary_host, "port": self.primary_port},
            )
        except Exception as exc:  # noqa: BLE001
            # 컷오버 중 클라이언트 실패 방지를 위해 primary 실패는 큐잉
            _log_obs(
                "write_err",
                "primary",
                collection_name,
                points,
                exc,
                endpoint={"host": self.primary_host, "port": self.primary_port},
            )
            self._append_retry_log(collection_name, points, exc, target="primary")
            return

        # secondary best-effort write
        if self.secondary is None:
            return

        try:
            self.secondary.upsert(collection_name=collection_name, points=points)
            _log_obs(
                "write_ok",
                "secondary",
                collection_name,
                points,
                endpoint={"host": self.secondary_host, "port": self.secondary_port},
            )
        except Exception as exc:  # noqa: BLE001
            _log_obs(
                "write_err",
                "secondary",
                collection_name,
                points,
                exc,
                endpoint={"host": self.secondary_host, "port": self.secondary_port},
            )
            self._append_retry_log(collection_name, points, exc, target="secondary")

    def _append_retry_log(
        self,
        collection_name: str,
        points: Iterable[models.PointStruct],
        exc: Exception,
        target: str = "secondary",
    ) -> None:
        points = list(points)
        payload = {
            "ts": _now_iso(),
            "collection": collection_name,
            "target": target,
            "error": str(exc),
            "points": [p.model_dump() for p in points],
        }
        try:
            with open(self.retry_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as exc:  # noqa: BLE001
            # 로깅 실패는 상위로 전파하지 않음
            logger.exception("retry log write failed: %s", self.retry_log_path, exc_info=exc)


def build_writer() -> QdrantWriter:
    retry_log_path = os.getenv("QDRANT_RETRY_LOG", "/tmp/qdrant_retry.log")
    primary = QdrantClient(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        api_key=settings.QDRANT_API_KEY if settings.QDRANT_API_KEY else None,
        https=False,
    )

    secondary_host = getattr(settings, "QDRANT_SECONDARY_HOST", "")
    secondary_port = int(getattr(settings, "QDRANT_SECONDARY_PORT", "0") or 0)
    secondary_key = getattr(settings, "QDRANT_SECONDARY_API_KEY", "")

    secondary = None
    if secondary_host and secondary_port:
        secondary = QdrantClient(
            host=secondary_host,
            port=secondary_port,
            api_key=secondary_key if secondary_key else None,
            https=False,
        )

    return QdrantWriter(
        primary=primary,
        secondary=secondary,
        retry_log_path=retry_log_path,
        primary_host=settings.QDRANT_HOST,
        primary_port=settings.QDRANT_PORT,
        secondary_host=secondary_host,
        secondary_port=secondary_port,
    )
