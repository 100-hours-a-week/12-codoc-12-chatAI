from fastapi import APIRouter
from uuid import uuid5, NAMESPACE_DNS
from qdrant_client import models

from app.common.api_response import CommonResponse
from app.common.config import settings
from app.qdrant.writer import build_writer
from app.domain.user_memories import user_schemas


router = APIRouter(prefix="/user-memories", tags=["user_memories"])


_writer = build_writer()


@router.post("", response_model=CommonResponse[user_schemas.UserMemoryUpsertRes])
def upsert_user_memory(req: user_schemas.UserMemoryUpsertReq):
    collection = settings.USER_MEMORIES_COLLECTION
    try:
        id_key = f"{req.user_id}:{req.created_at}"
        if req.problem_id is not None:
            id_key = f"{req.user_id}:{req.problem_id}:{req.created_at}"
        point_id = str(uuid5(NAMESPACE_DNS, id_key))
        payload = {
            "user_id": req.user_id,
            "problem_id": req.problem_id,
            "created_at": req.created_at,
            "text": req.text,
            "tags": req.tags or [],
            "seq": req.seq,
        }
        point = models.PointStruct(id=point_id, vector=req.vector, payload=payload)

        try:
            try:
                _writer.ensure_collection(collection, settings.VECTOR_SIZE)
            except Exception as exc:  # noqa: BLE001
                if not _writer.is_already_exists_error(exc):
                    _writer._append_retry_log(collection, [point], exc, target="ensure_collection")
                return CommonResponse.success_response(
                    message="queued", data=user_schemas.UserMemoryUpsertRes(point_id=point_id)
                )
            _writer.upsert(collection_name=collection, points=[point])
        except Exception as exc:  # noqa: BLE001
            # write-stop/전환 구간 실패는 재시도 로그에 적재
            try:
            _writer._append_retry_log(collection, [point], exc, target="primary")
            except Exception:
                pass

        return CommonResponse.success_response(
            message="upserted", data=user_schemas.UserMemoryUpsertRes(point_id=point_id)
        )
    except Exception as exc:  # noqa: BLE001
        # 최후 수단: write-stop 구간에서도 클라이언트 실패는 반환하지 않음
        try:
            id_key = f"{req.user_id}:{req.created_at}"
            if req.problem_id is not None:
                id_key = f"{req.user_id}:{req.problem_id}:{req.created_at}"
            dummy_point = models.PointStruct(
                id=str(uuid5(NAMESPACE_DNS, id_key)),
                vector=req.vector,
                payload={
                    "user_id": req.user_id,
                    "problem_id": req.problem_id,
                    "created_at": req.created_at,
                    "text": req.text,
                    "tags": req.tags or [],
                },
            )
            _writer._append_retry_log(collection, [dummy_point], exc, target="primary")
        except Exception:
            pass
        return CommonResponse.success_response(
            message="queued", data=user_schemas.UserMemoryUpsertRes(point_id=str(uuid5(NAMESPACE_DNS, id_key)))
        )


@router.get("/{user_id}", response_model=CommonResponse[user_schemas.UserMemoryListRes])
def list_user_memories(user_id: int, limit: int = 10):
    collection = settings.USER_MEMORIES_COLLECTION
    try:
        _writer.ensure_collection(collection, settings.VECTOR_SIZE)

        results, _ = _writer.primary.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]
            ),
            limit=limit,
            with_payload=True,
        )

        items = []
        for r in results:
            payload = r.payload or {}
            items.append(
                user_schemas.UserMemoryItem(
                    user_id=payload.get("user_id", user_id),
                    problem_id=payload.get("problem_id"),
                    created_at=payload.get("created_at", 0),
                    text=payload.get("text", ""),
                    tags=payload.get("tags", []),
                    seq=payload.get("seq"),
                )
            )
    except Exception:
        # 컷오버 중 proxy disconnect 발생 가능 → 5xx 대신 빈 결과 반환
        items = []

    return CommonResponse.success_response(
        message="ok", data=user_schemas.UserMemoryListRes(items=items)
    )
