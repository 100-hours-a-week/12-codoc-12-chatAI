from langgraph.graph import StateGraph, END
from app.domain.chatbot.bot_state import ChatBotState
from app.domain.chatbot.nodes.load_node import load_problem_node
from app.domain.chatbot.nodes.tutor_node import tutor_node
from app.domain.chatbot.nodes.analyzer_node import analyzer_node

def define_graph():
    # 1. StateGraph 초기화
    workflow = StateGraph(ChatBotState)

    # 2. 노드 등록
    workflow.add_node("load_problem", load_problem_node)
    workflow.add_node("analyze_answer", analyzer_node)
    workflow.add_node("tutor_question", tutor_node)

    # 3. 엣지 연결 (흐름 재설계)
    
    # 시작 -> 문제 데이터 로드
    workflow.set_entry_point("load_problem")
    
    # 1단계: 문제를 먼저 로드합니다.
    workflow.add_edge("load_problem", "analyze_answer")
    
    # 2단계: 유저의 답변을 분석하여 is_correct와 analysis_reason을 생성합니다.
    # (첫 진입 시 유저 답변이 없다면 analyzer_node 내에서 pass 하도록 설계)
    workflow.add_edge("analyze_answer", "tutor_question")
    
    # 3단계: 분석 결과를 바탕으로 튜터가 답변(SSE 스트리밍)을 생성합니다.
    workflow.add_edge("tutor_question", END)

    # 4. 그래프 컴파일
    return workflow.compile()

# 외부(router.py)에서 사용할 그래프 인스턴스
chatbot_graph = define_graph()