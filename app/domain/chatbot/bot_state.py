from typing import Annotated, List, TypedDict, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class ChatBotState(TypedDict):
    # 1. 대화 기록 (V1에서는 이 리스트가 메모리 역할을 함)
    # 랭그래프 내부에서 노드를 거칠 때마다 유저/AI 메시지가 여기에 누적됩니다.
    messages: Annotated[List[BaseMessage], add_messages]
    
    # 2. 스프링 요청 데이터
    user_id: int
    problem_id: int
    run_id: int
    
    # 3. 흐름 제어 (스프링의 current_node와 매칭)
    # 스프링이 보낸 'CONSTRAINT' 등 노드 정보를 저장하여 적절한 프롬프트를 선택
    current_node: str 
    paragraph_order: int
    
    # 4. 벡터 DB(Collection 1) 데이터
    content: str               # 문단 원문
    essential_keywords: str    # 정답 판단 기준
    chatbot_answer_guide: str  # 답변 가이드
    
    # 5. 상태 판단 필드
    is_correct: bool           # 정답 여부