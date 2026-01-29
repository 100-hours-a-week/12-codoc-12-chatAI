#!/usr/bin/env python3
"""
Kakao dataset Lv2 임베딩 및 Qdrant 업로드 스크립트
- 파일에서 문단 데이터 읽기
- sentence-transformers로 임베딩
- Qdrant에 저장
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv
import uuid

# .env 파일 로드
load_dotenv()

# ==================== 설정 ====================
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "Problems")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "384"))

QDRANT_URL = f"http://{QDRANT_HOST}:{QDRANT_PORT}"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ==================== 초기화 ====================
print("🔧 초기화 중...")

# Qdrant 클라이언트
client = QdrantClient(
    host=QDRANT_HOST,
    port=QDRANT_PORT,
    api_key=QDRANT_API_KEY if QDRANT_API_KEY else None,
)

# 임베딩 모델
model = SentenceTransformer(MODEL_NAME)
embedding_dim = model.get_sentence_embedding_dimension()
print(f"✅ 모델 로드됨: {MODEL_NAME}")
print(f"✅ 임베딩 차원: {embedding_dim}")

# Qdrant 콜렉션 확인/생성
try:
    collection_info = client.get_collection(COLLECTION_NAME)
    print(f"✅ 기존 콜렉션 사용: {COLLECTION_NAME}")
except Exception:
    print(f"⚙️ 새로운 콜렉션 생성 중: {COLLECTION_NAME}")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=embedding_dim,
            distance=Distance.COSINE,
        ),
    )
    print(f"✅ 콜렉션 생성 완료")


# ==================== 데이터 로드 및 임베딩 ====================
def load_and_embed_json(file_path: str) -> List[PointStruct]:
    """
    JSON 파일을 읽고 문단별로 임베딩해서 PointStruct 리스트 반환
    """
    print(f"\n📖 파일 읽기: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 단일 문서 또는 리스트 처리
    if isinstance(data, list):
        documents = data
    else:
        documents = [data]
    
    points = []
    
    # 기존 포인트 수를 조회해서 누적 ID 시작점 설정
    try:
        collection_info = client.get_collection(COLLECTION_NAME)
        point_id = collection_info.points_count  # 기존 포인트 수
        print(f"📍 기존 포인트: {point_id}개 | 새 포인트는 ID {point_id + 1}부터 시작")
    except Exception as e:
        point_id = 0
        print(f"⚠️ 콜렉션 정보 조회 실패, ID 0부터 시작: {e}")
    
    for doc in documents:
        problem_id = doc.get("problem_id")
        title = doc.get("title", "")
        difficulty = int(doc.get("difficulty", 0)) if doc.get("difficulty") else 0
        tags = doc.get("tags", [])
        
        print(f"  문제: {problem_id} - {title}")
        
        # 문단 처리
        answer_guides = doc.get("answer_guides", [])
        for idx, guide in enumerate(answer_guides, 1):
            paragraph_type = guide.get("paragraph_type", "")
            content = guide.get("content", "")
            essential_keywords = guide.get("essential_keywords", [])
            chatbot_answer_guide = guide.get("chatbot_answer_guide", "")
            
            if not content:
                print(f"    ⚠️ 문단 {idx} ({paragraph_type}): 내용 없음, 스킵")
                continue
            
            # 임베딩 생성
            embedding = model.encode(content)
            
            # Qdrant Point 구성
            payload = {
                "problem_id": int(problem_id),
                "title": title,
                "difficulty": difficulty,
                "tags": tags,
                "paragraph_type": paragraph_type,
                "paragraph_order": idx,
                "content": content,
                "essential_keywords": " ".join(essential_keywords) if essential_keywords else "",
                "chatbot_answer_guide": chatbot_answer_guide,
            }
            
            point_uuid = str(uuid.uuid4()) # 랜덤한 고유 ID 생성
            point = PointStruct(
                id=point_uuid,
                vector=embedding.tolist(),
                payload=payload,
            )
            points.append(point)
            
            print(f"✅ 문단 {idx} ({paragraph_type}): 임베딩 완료 (ID: {point_uuid})")
    
    return points


def upload_to_qdrant(points: List[PointStruct]) -> None:
    """
    포인트들을 Qdrant에 업로드
    """
    if not points:
        print("⚠️ 업로드할 포인트가 없습니다.")
        return
    
    print(f"\n⬆ Qdrant 업로드 중... ({len(points)} 포인트)")
    
    try:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )
        print(f"✅ {len(points)}개 포인트 업로드 완료!")
    except Exception as e:
        print(f"❌ 업로드 실패: {e}")
        raise


# ==================== 메인 실행 ====================
if __name__ == "__main__":
    import sys
    
    # 파일 경로 (상대경로 또는 명령행 인자)
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    else:
        # 기본값: 현재 디렉토리 기준 상대경로
        test_file = "./programmers_data2/nested_json/81301_nested.json"
    
    # 상대경로를 절대경로로 변환
    test_file = os.path.abspath(test_file)
    
    # 파일 존재 확인
    if not os.path.exists(test_file):
        print(f"❌ 파일을 찾을 수 없습니다: {test_file}")
        print(f"💡 팁: python embed_and_upload.py <JSON 파일 경로>")
        exit(1)
    
    # 1. 로드 및 임베딩
    points = load_and_embed_json(test_file)
    
    # 2. Qdrant 업로드
    upload_to_qdrant(points)
    
    # 3. 검증
    print("\n🔍 업로드 결과 검증...")
    try:
        collection_info = client.get_collection(COLLECTION_NAME)
        print(f"✅ 콜렉션 포인트 수: {collection_info.points_count}")
        print(f"✅ 벡터 차원: {collection_info.config.params.vectors.size}")
    except Exception as e:
        print(f"❌ 검증 실패: {e}")
    
    print("\n✨ 완료!")
