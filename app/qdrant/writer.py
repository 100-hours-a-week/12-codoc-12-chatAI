import json
from datetime import datetime, timezone
from typing import Iterable, Optional

from qdrant_client import QdrantClient, models

from app.common.config import settings


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    ):
        self.primary = primary
        self.secondary = secondary
        self.retry_log_path = retry_log_path

    def ensure_collection(self, collection_name: str, vector_size: int) -> None:
        if not self.primary.collection_exists(collection_name):
            self.primary.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size, distance=models.Distance.COSINE
                ),
            )

        if self.secondary and not self.secondary.collection_exists(collection_name):
            self.secondary.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size, distance=models.Distance.COSINE
                ),
            )

    def upsert(self, collection_name: str, points: Iterable[models.PointStruct]) -> None:
        # primary 동기 write
        try:
            self.primary.upsert(collection_name=collection_name, points=points)
        except Exception as exc:  # noqa: BLE001
            # 컷오버 중 클라이언트 실패 방지를 위해 primary 실패는 큐잉
            self._append_retry_log(collection_name, points, exc)
            return

        # secondary best-effort write
        if self.secondary is None:
            return

        try:
            self.secondary.upsert(collection_name=collection_name, points=points)
        except Exception as exc:  # noqa: BLE001
            self._append_retry_log(collection_name, points, exc)

    def _append_retry_log(
        self,
        collection_name: str,
        points: Iterable[models.PointStruct],
        exc: Exception,
    ) -> None:
        payload = {
            "ts": _now_iso(),
            "collection": collection_name,
            "error": str(exc),
            "points": [p.model_dump() for p in points],
        }
        try:
            with open(self.retry_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            # 로깅 실패는 상위로 전파하지 않음
            pass


def build_writer() -> QdrantWriter:
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

    return QdrantWriter(primary=primary, secondary=secondary)
