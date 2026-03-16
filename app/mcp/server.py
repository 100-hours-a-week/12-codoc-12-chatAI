"""
Codoc MCP Server
----------------
내부 툴: retrieve_paragraph, retrieve_concept, retrieve_user_history
외부 툴: web_search (Tavily, 추후 활성화)

실행: python -m app.mcp.server
포트: MCP_SERVER_PORT (기본 8001)
"""

import os
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from qdrant_client import QdrantClient
from qdrant_client import models as qdrant_models
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv(override=True)

mcp = FastMCP(
    "codoc-tools",
    host="0.0.0.0",
    port=int(os.getenv("MCP_SERVER_PORT", "8001")),
)

# ── Qdrant 클라이언트 (MCP 서버 전용) ─────────────────────────────────────────
_qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL", f"http://{os.getenv('QDRANT_HOST', 'localhost')}:{os.getenv('QDRANT_PORT', '6333')}"),
    api_key=os.getenv("QDRANT_API_KEY") or None,
)

# ── 임베딩 모델 (서버 시작 시 1회 로드) ────────────────────────────────────────
# bot_service.py와 동일한 모델 사용 (벡터 공간 일치)
print("🔄 임베딩 모델 로딩 중 (BAAI/bge-m3)...")
_embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
print("✅ 임베딩 모델 로딩 완료")

# ── 내부 툴 1: 현재 문단 힌트 ──────────────────────────────────────────────────
@mcp.tool()
async def retrieve_paragraph(problem_id: int, paragraph_type: str) -> str:
    """
    현재 유저가 풀고 있는 문단의 힌트와 답변 가이드를 가져옵니다.
    유저가 현재 단계에서 막히거나 힌트를 요청할 때 사용하세요.

    Args:
        problem_id: 문제 ID
        paragraph_type: 단계명 (BACKGROUND, GOAL, STRATEGY, INSIGHT 중 하나)
    """
    try:
        results, _ = _qdrant.scroll(
            collection_name="Problems",
            scroll_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="problem_id",
                        match=qdrant_models.MatchValue(value=problem_id),
                    ),
                    qdrant_models.FieldCondition(
                        key="paragraph_type",
                        match=qdrant_models.MatchValue(value=paragraph_type),
                    ),
                ]
            ),
            limit=1,
            with_payload=True,
        )
        if not results:
            return f"[{paragraph_type}] 단계의 힌트 데이터를 찾을 수 없습니다."

        payload = results[0].payload
        guide = payload.get("chatbot_answer_guide", "")
        summary = payload.get("essential_summary", "")
        return (
            f"[{paragraph_type} 단계 힌트]\n"
            f"핵심 요약: {summary}\n"
            f"답변 가이드: {guide}"
        )
    except Exception as e:
        return f"힌트 조회 중 오류가 발생했습니다: {e}"


# ── 내부 툴 2: 알고리즘/자료구조 개념 (유사도 검색) ────────────────────────────
@mcp.tool()
async def retrieve_concept(query: str) -> str:
    """
    Algo_concepts 컬렉션에서 유사도 검색으로 알고리즘/자료구조 개념을 가져옵니다.
    유저가 특정 개념(BFS, DFS, DP, 스택, 큐 등)에 대해 헷갈려할 때 사용하세요.

    Args:
        query: 검색할 개념 설명 또는 질문 (예: "BFS로 최단경로 구하는 방법", "DP 메모이제이션이란")
    """
    try:
        # Algo_Concepts 컬렉션 존재 여부 확인
        collections = [c.name for c in _qdrant.get_collections().collections]
        if "Algo_Concepts" not in collections:
            return (
                "[안내] 개념 데이터베이스가 아직 준비 중입니다.\n"
                f"'{query}'에 대해서는 제가 직접 설명드릴게요."
            )

        # 쿼리 임베딩 후 유사도 검색
        query_vector = _embeddings.embed_query(query)
        results = _qdrant.search(
            collection_name="Algo_Concepts",
            query_vector=query_vector,
            limit=3,
            with_payload=True,
            score_threshold=0.5,  # 유사도 임계값: 너무 관련없는 문서 제외
        )

        if not results:
            return f"'{query}'와 관련된 개념 문서를 찾지 못했습니다."

        docs = []
        for r in results:
            title = r.payload.get("title", "")
            content = r.payload.get("content", "")
            score = round(r.score, 3)
            docs.append(f"## {title} (유사도: {score})\n{content}")

        return "\n\n---\n\n".join(docs)
    except Exception as e:
        return f"개념 조회 중 오류가 발생했습니다: {e}"


# ── 내부 툴 3: 유저 학습 히스토리 ──────────────────────────────────────────────
@mcp.tool()
async def retrieve_user_history(user_id: int, problem_id: int, session_id: str) -> str:
    """
    유저의 현재 세션 대화 히스토리와 과거 취약점을 요약합니다.
    유저의 학습 패턴이나 과거 실수를 참고하여 개인화된 조언이 필요할 때 사용하세요.

    Args:
        user_id: 유저 ID
        problem_id: 현재 문제 ID
        session_id: 현재 세션 ID
    """
    try:
        redis_url = os.getenv("REDIS_URL")
        redis_key = f"ai:chatbot:user:{user_id}:problem:{problem_id}:{session_id}"
        history = RedisChatMessageHistory(session_id=redis_key, url=redis_url)
        messages = history.messages

        if not messages:
            return "현재 세션의 대화 기록이 없습니다."

        # 단계별 정답/오답 패턴 요약
        stage_summary = {}
        for msg in messages:
            if isinstance(msg, HumanMessage):
                p_type = msg.additional_kwargs.get("paragraph_type")
                is_correct = msg.additional_kwargs.get("is_correct", False)
                if p_type:
                    if p_type not in stage_summary:
                        stage_summary[p_type] = {"attempts": 0, "correct": False}
                    stage_summary[p_type]["attempts"] += 1
                    if is_correct:
                        stage_summary[p_type]["correct"] = True

        lines = ["[유저 학습 패턴]"]
        for stage, info in stage_summary.items():
            status = "✅ 정답" if info["correct"] else f"❌ {info['attempts']}회 시도 중"
            lines.append(f"- {stage}: {status}")

        total_turns = len(messages) // 2
        lines.append(f"\n총 {total_turns}번의 대화가 이루어졌습니다.")

        return "\n".join(lines)
    except Exception as e:
        return f"히스토리 조회 중 오류가 발생했습니다: {e}"


# ── 외부 툴: 웹 검색 (Tavily, 추후 활성화) ────────────────────────────────────
# @mcp.tool()
# async def web_search(query: str) -> str:
#     """
#     기술 블로그, 유튜브 강의 등 외부 자료를 검색합니다.
#     유저가 개념을 더 깊이 공부하고 싶어할 때 사용하세요.
#     """
#     from tavily import TavilyClient
#     client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
#     results = client.search(query=query, max_results=3)
#     ...


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
