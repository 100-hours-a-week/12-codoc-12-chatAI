import json
import os
import uuid
import sys
from pathlib import Path
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv

# .env 로드
load_dotenv()

# ==================== 설정 (공통) ====================
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "Problems")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "384"))

# ==================== [핵심] 임베딩 모델 로드 및 함수 ====================
# 모델은 파일이 import될 때 한 번만 메모리에 올라갑니다.
print(f"🔧 임베딩 모델 로드 중: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

def get_embeddings(text: str) -> List[float]:
    """
    텍스트를 벡터로 변환하여 리스트로 반환 (nodes.py에서 사용)
    """
    if not text:
        return [0.0] * VECTOR_SIZE
    return model.encode(text).tolist()

# ==================== 데이터 처리 함수 ====================

def load_and_embed_json(file_path: str, client: QdrantClient) -> List[PointStruct]:
    """
    JSON 파일을 읽고 문단별로 임베딩해서 PointStruct 리스트 반환
    """
    print(f"\n📖 파일 읽기: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents = data if isinstance(data, list) else [data]
    points = []

    for doc in documents:
        problem_id = doc.get("problem_id")
        title = doc.get("title", "")

        # 💡 [Lv1 에러 해결] 문자열 형태의 난이도 처리
        raw_difficulty = doc.get("difficulty", "0")
        if isinstance(raw_difficulty, str) and "Lv" in raw_difficulty:
            difficulty = int(raw_difficulty.replace("Lv", ""))
        else:
            try:
                difficulty = int(raw_difficulty) if raw_difficulty else 0
            except ValueError:
                difficulty = 0

        tags = doc.get("tags", [])
        print(f"  문제: {problem_id} - {title}")

        answer_guides = doc.get("answer_guides", [])
        for idx, guide in enumerate(answer_guides, 1):
            content = guide.get("content", "")
            if not content:
                continue

            # 임베딩 생성 (위에서 만든 함수 활용)
            vector = get_embeddings(content)

            payload = {
                "problem_id": int(problem_id) if problem_id else 0,
                "title": title,
                "difficulty": difficulty,
                "tags": tags,
                "paragraph_type": guide.get("paragraph_type", ""),
                "paragraph_order": idx,
                "content": content,
                "essential_keywords": " ".join(guide.get("essential_keywords", [])),
                "chatbot_answer_guide": guide.get("chatbot_answer_guide", ""),
            }

            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=payload,
            ))
            print(f"✅ 문단 {idx} 임베딩 완료")

    return points

# ==================== 메인 실행 구역 ====================
# 터미널에서 'python embed_and_upload.py'를 칠 때만 실행됩니다.
if __name__ == "__main__":
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

    # 1. Qdrant 클라이언트 초기화 (로컬 전용 설정)
    client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        api_key=QDRANT_API_KEY if QDRANT_API_KEY else None,
        https=False # 로컬은 필수
    )

    # 2. 콜렉션 생성/확인
    try:
        client.get_collection(COLLECTION_NAME)
        print(f"✅ 기존 콜렉션 사용: {COLLECTION_NAME}")
    except Exception:
        print(f"⚙️ 새로운 콜렉션 생성 중: {COLLECTION_NAME}")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

    # 3. 경로 설정
    test_file = sys.argv[1] if len(sys.argv) > 1 else "/Users/joyejin/Desktop/dataset/1/1-258712.json"

    # 4. 실행
    points = load_and_embed_json(test_file, client)

    if points:
        print(f"\n⬆ Qdrant 업로드 중... ({len(points)} 포인트)")
        client.upsert(collection_name=COLLECTION_NAME, points=points)

        # 5. 검증
        info = client.get_collection(COLLECTION_NAME)
        print(f"✅ 검증 완료! 현재 포인트 수: {info.points_count}")
    else:
        print("⚠️ 업로드할 데이터가 없습니다.")