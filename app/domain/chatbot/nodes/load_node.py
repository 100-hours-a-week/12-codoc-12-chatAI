from app.domain.chatbot.bot_state import ChatBotState
from app.qdrant.crud import get_paragraph_by_type

async def load_problem_node(state: ChatBotState) -> dict:
    """
    스프링에서 받은 problem_id와 current_node를 기반으로 
    Qdrant(Collection 1)에서 단계별 지문 정보를 로드하여 State에 저장합니다.
    """
    problem_id = state.get("problem_id")
    paragraph_type = state.get("paragraph_type") # 예: 'BACKGROUND', 'GOAL', 'CONSTRAINT' 등

    print(f"🔍 Qdrant 조회 시도: ID={problem_id}, Node={paragraph_type}") # 로그 추가

    # 1. Qdrant에서 해당 문제의 현재 단계 데이터 조회 (필터링 기반) -> 추후 유사도 검색으로 변경
    paragraph_data = await get_paragraph_by_type(
        problem_id=problem_id,
        paragraph_type=paragraph_type
    )

    # 데이터가 없을 경우에 대한 예외 처리
    if not paragraph_data:
        raise ValueError(f"Qdrant에서 데이터를 찾을 수 없습니다. (ID: {problem_id}, Node: {paragraph_type})")

    print(f"✅ 지문 로드 성공: {paragraph_data['content'][:20]}...") # 성공 로그

    # 2. 조회된 데이터를 State 형식에 맞춰 반환
    # 반환되는 dict는 LangGraph에 의해 기존 State와 병합(merge)됩니다.
    return {
        "content": paragraph_data.get("content"),
        "essential_keywords": paragraph_data.get("essential_keywords"),
        "chatbot_answer_guide": paragraph_data.get("chatbot_answer_guide"),
        "paragraph_order": paragraph_data.get("paragraph_order"),
        "retry_count": 0  # 새로운 단계 진입 시 시도 횟수 초기화
    }