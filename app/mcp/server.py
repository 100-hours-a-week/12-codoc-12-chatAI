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
        paragraph_type: 단계명 (BACKGROUND, GOAL, STRATEGY, INSIGHT 중 하나임)
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
    유저의 현재 세션 대화 히스토리와 최근 1주일간의 학습 기억을 요약합니다.
    유저의 학습 패턴이나 과거 실수를 참고하여 개인화된 조언이 필요할 때 사용하세요.

    Args:
        user_id: 유저 ID
        problem_id: 현재 문제 ID
        session_id: 현재 세션 ID
    """
    import time
    lines = []

    # ── Part 1: Redis 현재 세션 히스토리 ──────────────────────────────────────
    try:
        redis_url = os.getenv("REDIS_URL")
        redis_key = f"ai:chatbot:user:{user_id}:problem:{problem_id}:{session_id}"
        history = RedisChatMessageHistory(session_id=redis_key, url=redis_url)
        messages = history.messages

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

        lines.append("[현재 세션 학습 패턴]")
        if stage_summary:
            for stage, info in stage_summary.items():
                status = "✅ 정답" if info["correct"] else f"❌ {info['attempts']}회 시도 중"
                lines.append(f"- {stage}: {status}")
            total_turns = len(messages) // 2
            lines.append(f"총 {total_turns}번의 대화가 이루어졌습니다.")
        else:
            lines.append("- 현재 세션의 대화 기록이 없습니다.")
    except Exception as e:
        lines.append(f"[현재 세션] 조회 중 오류: {e}")

    # ── Part 2: Qdrant User_memories 최근 1주일 ────────────────────────────────
    try:
        one_week_ago = int(time.time()) - 7 * 24 * 3600
        results, _ = _qdrant.scroll(
            collection_name="User_memories",
            scroll_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="user_id",
                        match=qdrant_models.MatchValue(value=user_id),
                    ),
                    qdrant_models.FieldCondition(
                        key="created_at",
                        range=qdrant_models.Range(gte=one_week_ago),
                    ),
                ]
            ),
            limit=5,
            with_payload=True,
            with_vectors=False,
        )

        lines.append("\n[최근 1주일 학습 기억]")
        if not results:
            lines.append("- 최근 1주일 내 학습 기록이 없습니다.")
        else:
            for r in results:
                p = r.payload
                prob_id = p.get("problem_id", "?")
                error_summary = p.get("error_summary", "")
                weak_tags = p.get("weak_tags", [])
                error_paragraph = p.get("error_paragraph", "")
                created_at = p.get("created_at", 0)
                date_str = time.strftime("%m/%d", time.localtime(created_at)) if created_at else "?"

                lines.append(f"\n• [{date_str}] 문제 #{prob_id}")
                if error_summary:
                    lines.append(f"  실수 요약: {error_summary}")
                if weak_tags:
                    lines.append(f"  취약 태그: {', '.join(weak_tags)}")
                if error_paragraph:
                    lines.append(f"  어려웠던 단계: {error_paragraph}")
    except Exception as e:
        lines.append(f"[학습 기억] 조회 중 오류: {e}")

    return "\n".join(lines)


# ── 외부 툴 1: 웹 검색 (Tavily) ────────────────────────────────────────────────
@mcp.tool()
async def web_search(query: str) -> str:
    """
    기술 블로그, 공식 문서, 관련 페이지 등 외부 자료를 검색하여 링크를 제공합니다.
    유저가 개념을 더 깊이 공부하고 싶거나 참고 자료를 원할 때 사용하세요.

    Args:
        query: 검색 키워드 (예: "BFS 알고리즘 Python 구현", "동적 프로그래밍 예시")
    """
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        response = client.search(
            query=query,
            max_results=5,
            search_depth="basic",
        )

        results = response.get("results", [])
        if not results:
            return f"'{query}'에 대한 검색 결과를 찾을 수 없습니다."

        lines = [f"['{query}' 관련 외부 자료]\n"]
        for i, r in enumerate(results, 1):
            title = r.get("title", "제목 없음")
            url = r.get("url", "")
            snippet = r.get("content", "")[:120].strip()
            lines.append(f"{i}. **{title}**\n   링크: {url}\n   요약: {snippet}...")

        return "\n\n".join(lines)
    except Exception as e:
        return f"웹 검색 중 오류가 발생했습니다: {e}"


# ── 외부 툴 2: Pseudo 코드 생성 (Groq API) ─────────────────────────────────────
@mcp.tool()
async def generate_pseudocode(problem_description: str) -> str:
    """
    알고리즘 문제에 대한 언어 독립적인 의사코드(pseudo code)를 생성합니다.
    유저가 구현 방법을 모르거나 코드 구조에 대한 가이드가 필요할 때 사용하세요.
    특정 프로그래밍 언어 문법 없이 누구나 이해할 수 있는 로직 흐름을 제공합니다.

    Args:
        problem_description: 풀어야 할 알고리즘 문제나 구현할 기능 설명
    """
    try:
        from groq import Groq

        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        prompt = (
            f"다음 알고리즘 문제에 대한 의사코드(pseudo code)를 작성해주세요.\n"
            f"규칙:\n"
            f"- 특정 프로그래밍 언어(Python, Java, C++ 등)의 문법을 사용하지 마세요\n"
            f"- 자연어(한국어)와 기호를 섞어서 누구나 이해할 수 있게 작성하세요\n"
            f"- 예: '큐에 시작점 추가', 'visited[x][y] = True로 표시', 'while 큐가 비지 않은 동안'\n"
            f"- 라이브러리 임포트, 타입 선언 등 언어 특유의 코드는 절대 포함하지 마세요\n"
            f"- 로직 흐름(입력→처리→출력)이 명확히 보이도록 단계별로 작성\n"
            f"- 코드 블록 없이 pseudo code 텍스트만 출력\n\n"
            f"문제: {problem_description}"
        )

        completion = client.chat.completions.create(
            model="moonshotai/kimi-k2-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=600,
        )

        pseudocode = completion.choices[0].message.content.strip()
        return f"```pseudocode\n{pseudocode}\n```"
    except Exception as e:
        return f"의사코드 생성 중 오류가 발생했습니다: {e}"


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
