from typing import Annotated, List, TypedDict, Optional
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[List, add_messages]
    problem_id: str
    current_stage: str       # CONTEXT, GOAL, STRATEGY, INSIGHT
    intent: str              # CARD_PROGRESS (카드 진행), GENERAL_INQUIRY (일반 질문)
    is_validated: bool       # 키워드/의미 검증 통과 여부
    retrieved_content: dict  # Qdrant에서 가져온 원문, 가이드, 키워드 전체