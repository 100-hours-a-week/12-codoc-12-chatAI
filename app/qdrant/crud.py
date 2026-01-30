from qdrant_client import QdrantClient, models
from app.common.config import QDRANT_URL, QDRANT_API_KEY # 설정값 연동

# 클라이언트 초기화
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

async def get_paragraph_by_type(problem_id: int, paragraph_type: str):
    print(f"🚀 Qdrant 조회 시작: problem_id={problem_id} ({type(problem_id)}), type={paragraph_type}")
    
    """
    Collection 1(problems)에서 problem_id와 paragraph_type이 일치하는 
    지문 원문 및 가이드 데이터를 검색합니다.
    """
    # 1. 메타데이터 필터링을 사용한 검색
    # scroll API는 벡터 검색 없이 조건(Filter)만으로 데이터를 가져올 때 효율적입니다.
    results, _ = client.scroll(
        collection_name="Problems",
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(key="problem_id", match=models.MatchValue(value=problem_id)),
                models.FieldCondition(key="paragraph_type", match=models.MatchValue(value=paragraph_type))
            ]
        ),
        limit=1,
        with_payload=True # content, essential_keywords 등을 가져오기 위해 필수
    )

    if results:
        # 가져온 페이로드의 키들을 출력하여 확인
        print(f"✅ 데이터 발견! Keys: {list(results[0].payload.keys())}")
        return results[0].payload
    
    print(f"⚠️ 결과 없음: {paragraph_type} 타입의 {problem_id}번 문제를 찾지 못했습니다.")
    return None