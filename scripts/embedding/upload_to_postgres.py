#!/usr/bin/env python3
"""
Local problem asset files -> PostgreSQL batch uploader.

Default filename convention:
  {problem_id}_{slot}.{ext}
  - slot 1 -> COMMON_JSON
  - slot 2 -> PROBLEM_MD
  - slot 3 -> QUIZ_JSON
  - slot 4 -> EMBED_JSON

Example:
  python scripts/embedding/upload_problem_assets_to_postgres.py \
    --root-dir /Users/chanmi/Documents/KTB_CODOC/codoc-embedding/v2_problem
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import func

# Allow running this file directly: `python scripts/embedding/upload_problem_assets_to_postgres.py ...`
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.common.db import SessionLocal
from app.db.models import (
    EmbeddingJob,
    EmbeddingStatus,
    ParagraphType,
    Problem,
    ProblemAsset,
    ProblemAssetFileType,
    ReviewStatus,
)


FILE_SLOT_MAP: dict[str, ProblemAssetFileType] = {
    "1": ProblemAssetFileType.COMMON_JSON,
    "2": ProblemAssetFileType.PROBLEM_MD,
    "3": ProblemAssetFileType.QUIZ_JSON,
    "4": ProblemAssetFileType.EMBED_JSON,
}

LEGACY_FILE_RE = re.compile(r"^(?P<problem_id>\d+)_(?P<slot>[1-4])\.(?P<ext>json|md)$", re.IGNORECASE)
INVALID_ESCAPE_RE = re.compile(r'\\(?!["\\/bfnrtu])')


def sha256_hex(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load_file_payload(path: Path, file_type: ProblemAssetFileType) -> tuple[dict[str, Any] | None, str | None]:
    raw = path.read_bytes()
    text = raw.decode("utf-8")

    if file_type in {
        ProblemAssetFileType.COMMON_JSON,
        ProblemAssetFileType.QUIZ_JSON,
        ProblemAssetFileType.EMBED_JSON,
    }:
        try:
            return json.loads(text), None
        except json.JSONDecodeError as e:
            # Some source files contain invalid backslash escapes (e.g. "\("),
            # so convert them to literal backslashes and retry.
            sanitized = INVALID_ESCAPE_RE.sub(r"\\\\", text)
            try:
                return json.loads(sanitized), None
            except json.JSONDecodeError:
                raise ValueError(f"JSON parsing failed: {path} -> {e}") from e
    return None, text


def parse_legacy_filename(path: Path) -> tuple[int, ProblemAssetFileType] | None:
    match = LEGACY_FILE_RE.match(path.name)
    if not match:
        return None
    problem_id = int(match.group("problem_id"))
    slot = match.group("slot")
    return problem_id, FILE_SLOT_MAP[slot]


def pick_problem_meta(problem_id: int, asset_rows: list[dict[str, Any]]) -> tuple[str, int]:
    default_title = f"Problem {problem_id}"
    default_difficulty = 0

    def extract_from_payload(payload: Any) -> tuple[str | None, int | None]:
        if isinstance(payload, dict):
            title = payload.get("title")
            difficulty = payload.get("difficulty")
            parsed_difficulty = None
            if isinstance(difficulty, int):
                parsed_difficulty = difficulty
            elif isinstance(difficulty, str) and difficulty.isdigit():
                parsed_difficulty = int(difficulty)
            return (title if isinstance(title, str) else None), parsed_difficulty

        if isinstance(payload, list):
            for item in payload:
                title, difficulty = extract_from_payload(item)
                if title is not None or difficulty is not None:
                    return title, difficulty
        return None, None

    # difficulty/title 우선순위: EMBED_JSON -> COMMON_JSON -> QUIZ_JSON
    priority = [
        ProblemAssetFileType.EMBED_JSON,
        ProblemAssetFileType.COMMON_JSON,
        ProblemAssetFileType.QUIZ_JSON,
    ]
    rows_by_type = {row["file_type"]: row for row in asset_rows}

    for file_type in priority:
        row = rows_by_type.get(file_type)
        if not row:
            continue
        title, difficulty = extract_from_payload(row.get("json_body"))
        if title:
            default_title = title
        if difficulty is not None:
            default_difficulty = difficulty
        if default_title != f"Problem {problem_id}" or default_difficulty != 0:
            return default_title, default_difficulty

    for row in asset_rows:
        title, difficulty = extract_from_payload(row.get("json_body"))
        if title:
            default_title = title
        if difficulty is not None:
            default_difficulty = difficulty
        if default_title != f"Problem {problem_id}":
            break

    return default_title, default_difficulty


def pick_source_hash(asset_rows: list[dict[str, Any]]) -> str:
    by_type = {row["file_type"]: row["content_hash"] for row in asset_rows}
    # 요구사항: 최초 source_hash는 임베딩 소스(EMBED_JSON)의 content_hash를 복사
    if ProblemAssetFileType.EMBED_JSON in by_type:
        return by_type[ProblemAssetFileType.EMBED_JSON]
    raise ValueError("EMBED_JSON asset가 없어 source_hash를 설정할 수 없습니다.")


async def upsert_problem_bundle(problem_id: int, asset_rows: list[dict[str, Any]], dry_run: bool) -> None:
    title, difficulty = pick_problem_meta(problem_id, asset_rows)
    source_hash = pick_source_hash(asset_rows)

    async with SessionLocal() as session:
        problem_stmt = insert(Problem).values(
            problem_id=problem_id,
            title=title,
            difficulty=difficulty,
        )
        problem_stmt = problem_stmt.on_conflict_do_update(
            index_elements=[Problem.problem_id],
            set_={
                "title": problem_stmt.excluded.title,
                "difficulty": problem_stmt.excluded.difficulty,
                "updated_at": func.now(),
            },
        )

        await session.execute(problem_stmt)

        for row in asset_rows:
            asset_stmt = insert(ProblemAsset).values(
                problem_id=row["problem_id"],
                file_type=row["file_type"],
                json_body=row["json_body"],
                md_body=row["md_body"],
                content_hash=row["content_hash"],
            )
            asset_stmt = asset_stmt.on_conflict_do_update(
                index_elements=[ProblemAsset.problem_id, ProblemAsset.file_type],
                set_={
                    "json_body": asset_stmt.excluded.json_body,
                    "md_body": asset_stmt.excluded.md_body,
                    "content_hash": asset_stmt.excluded.content_hash,
                    "updated_at": asset_stmt.excluded.updated_at,
                },
            )
            await session.execute(asset_stmt)

        for paragraph_type in ParagraphType:
            job_stmt = insert(EmbeddingJob).values(
                problem_id=problem_id,
                paragraph_type=paragraph_type,
                review_status=ReviewStatus.DRAFT,
                embedding_status=EmbeddingStatus.PENDING,
                source_hash=source_hash,
                last_error=None,
            )
            job_stmt = job_stmt.on_conflict_do_update(
                index_elements=[EmbeddingJob.problem_id, EmbeddingJob.paragraph_type],
                set_={
                    "review_status": job_stmt.excluded.review_status,
                    "embedding_status": job_stmt.excluded.embedding_status,
                    "source_hash": job_stmt.excluded.source_hash,
                    "last_error": None,
                    "updated_at": func.now(),
                },
            )
            await session.execute(job_stmt)

        if dry_run:
            await session.rollback()
            return
        await session.commit()


def collect_assets(root_dir: Path) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    dedup: dict[tuple[int, ProblemAssetFileType], dict[str, Any]] = {}

    for path in sorted(root_dir.rglob("*")):
        if not path.is_file():
            continue
        parsed = parse_legacy_filename(path)
        if not parsed:
            continue

        problem_id, file_type = parsed
        raw = path.read_bytes()
        json_body, md_body = load_file_payload(path, file_type)

        row = {
            "problem_id": problem_id,
            "file_type": file_type,
            "json_body": json_body,
            "md_body": md_body,
            "content_hash": sha256_hex(raw),
            "source_path": str(path),
        }
        key = (problem_id, file_type)
        if key in dedup:
            print(
                f"[WARN] duplicated asset detected for problem_id={problem_id}, file_type={file_type.value}. "
                f"replace: {dedup[key]['source_path']} -> {row['source_path']}"
            )
        dedup[key] = row

    for row in dedup.values():
        grouped.setdefault(row["problem_id"], []).append(row)

    return grouped


async def main() -> None:
    parser = argparse.ArgumentParser(description="Upload local problem assets into PostgreSQL.")
    parser.add_argument("--root-dir", required=True, help="Root directory containing files (e.g. 65_1.json).")
    parser.add_argument("--dry-run", action="store_true", help="Parse and SQL execute, but rollback transaction.")
    args = parser.parse_args()

    root_dir = Path(args.root_dir).expanduser().resolve()
    if not root_dir.exists() or not root_dir.is_dir():
        raise ValueError(f"Invalid --root-dir: {root_dir}")

    grouped = collect_assets(root_dir)
    if not grouped:
        print("업로드할 파일을 찾지 못했습니다. 파일명 패턴 {problem_id}_{slot}.(json|md) 확인하세요.")
        return

    print(f"문제 수: {len(grouped)}")
    total_assets = 0
    for problem_id, asset_rows in grouped.items():
        if len(asset_rows) != 4:
            print(f"[WARN] problem_id={problem_id}: detected assets={len(asset_rows)} (expected 4)")
        print(f"[INFO] upsert problem_id={problem_id} (PK)")
        await upsert_problem_bundle(problem_id, asset_rows, dry_run=args.dry_run)
        total_assets += len(asset_rows)
        print(f"[OK] problem_id={problem_id}, assets={len(asset_rows)}")

    print(f"완료: problems={len(grouped)}, assets={total_assets}, dry_run={args.dry_run}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
