import json
import logging
import os
import time
from typing import List

from qdrant_client import QdrantClient, models

from app.common.config import settings

logger = logging.getLogger(__name__)
OBS_ENABLED = os.getenv("QDRANT_OBS", "false").lower() == "true"
RETRY_LOG_PATH = os.getenv("QDRANT_RETRY_LOG", "/tmp/qdrant_retry.log")
RETRY_PROCESSING_PATH = RETRY_LOG_PATH + ".processing"
ENABLE_DRAIN_WORKER = os.getenv("ENABLE_DRAIN_WORKER", "false").lower() == "true"
DRAIN_INTERVAL_SEC = int(os.getenv("QDRANT_DRAIN_INTERVAL_SEC", "5"))
MAX_BATCH = int(os.getenv("QDRANT_DRAIN_MAX_BATCH", "200"))


def _load_lines(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return f.readlines()


def _write_lines(path: str, lines: List[str]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.writelines(lines)
    os.replace(tmp, path)


def _acquire_processing_file() -> str:
    """
    재시도 로그를 처리 파일로 이동해 동시 append 유실을 방지한다.
    처리 파일이 남아 있으면 먼저 처리한다.
    """
    if os.path.exists(RETRY_PROCESSING_PATH):
        return RETRY_PROCESSING_PATH
    if not os.path.exists(RETRY_LOG_PATH):
        return ""
    try:
        os.replace(RETRY_LOG_PATH, RETRY_PROCESSING_PATH)
        return RETRY_PROCESSING_PATH
    except Exception:
        return ""


def _make_client() -> QdrantClient:
    host = getattr(settings, "QDRANT_SECONDARY_HOST", "")
    port = int(getattr(settings, "QDRANT_SECONDARY_PORT", "0") or 0)
    key = getattr(settings, "QDRANT_SECONDARY_API_KEY", "")
    if not host or not port:
        raise RuntimeError("secondary qdrant not configured")
    return QdrantClient(host=host, port=port, api_key=key if key else None, https=False)


def _ensure_user_memories_collection(client: QdrantClient, collection: str) -> None:
    if collection != settings.USER_MEMORIES_COLLECTION:
        return
    if client.collection_exists(collection):
        return
    try:
        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(
                size=settings.VECTOR_SIZE, distance=models.Distance.COSINE
            ),
        )
    except Exception as exc:  # noqa: BLE001
        if "already exists" not in str(exc).lower():
            raise


def _parse_points(items):
    points = []
    for p in items:
        points.append(models.PointStruct(**p))
    return points

def _extract_run_id(points):
    for p in points:
        payload = getattr(p, "payload", None) or {}
        tags = payload.get("tags") or []
        for t in tags:
            if isinstance(t, str) and t.startswith("run="):
                return t.split("=", 1)[1]
    return None

def _point_ids(points):
    return [str(p.id) for p in points]

def _log_obs(event: str, collection: str, points, exc: Exception | None = None) -> None:
    if not OBS_ENABLED:
        return
    payload = {
        "ts": time.time(),
        "event": event,
        "target": "secondary",
        "collection": collection,
        "point_ids": _point_ids(points),
        "run_id": _extract_run_id(points),
        "error": str(exc) if exc else None,
    }
    logger.info("QDRANT_OBS %s", json.dumps(payload, ensure_ascii=False))

def drain_once(client: QdrantClient) -> int:
    processing_path = _acquire_processing_file()
    if not processing_path:
        return 0

    lines = _load_lines(processing_path)
    if not lines:
        return 0

    processed = 0
    remaining = []

    for line in lines:
        if processed >= MAX_BATCH:
            remaining.append(line)
            continue
        collection = None
        points = []
        try:
            payload = json.loads(line)
            collection = payload.get("collection")
            points = _parse_points(payload.get("points", []))
            if collection and points:
                _ensure_user_memories_collection(client, collection)
                client.upsert(collection_name=collection, points=points)
                _log_obs("drain_ok", collection, points)
                processed += 1
            else:
                remaining.append(line)
        except Exception as exc:
            # 실패 라인은 다음 재시도 대상
            try:
                _log_obs("drain_err", collection or "unknown", points or [], exc)
            except Exception:
                pass
            remaining.append(line)

    # 남은 라인은 메인 로그에 append하여 신규 기록을 덮어쓰지 않음
    if remaining:
        with open(RETRY_LOG_PATH, "a", encoding="utf-8") as f:
            f.writelines(remaining)
    try:
        os.remove(processing_path)
    except Exception:
        pass
    return processed


def run_forever() -> None:
    if not ENABLE_DRAIN_WORKER:
        return
    while True:
        try:
            client = _make_client()
            drained = drain_once(client)
            if drained == 0:
                time.sleep(DRAIN_INTERVAL_SEC)
        except Exception:
            time.sleep(DRAIN_INTERVAL_SEC)


if __name__ == "__main__":
    run_forever()
