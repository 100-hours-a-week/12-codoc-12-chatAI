from langgraph.graph import StateGraph, END
from app.domain.chatbot.bot_state import ChatBotState
from app.domain.chatbot.nodes.load_node import load_problem_node
from app.domain.chatbot.nodes.tutor_node import tutor_node
from app.domain.chatbot.nodes.analyzer_node import analyzer_node

def define_graph():
    # 1. StateGraph 초기화
    workflow = StateGraph(ChatBotState)

    # 2. 노드 등록 (도메인/nodes 폴더에서 가져온 함수들)
    workflow.add_node("load_problem", load_problem_node)
    workflow.add_node("tutor_question", tutor_node)
    workflow.add_node("analyze_answer", analyzer_node)

    # 3. 엣지 연결 (흐름 설계)
    # 시작 -> 문제 데이터 로드
    workflow.set_entry_point("load_problem")
    
    # 문제 로드 완료 -> 유저에게 질문 던지기
    workflow.add_edge("load_problem", "tutor_question")
    
    # 유저 답변이 오면 분석 노드로 이동
    # (실제 런타임에서는 유저 입력이 들어올 때까지 대기하다가 analyze로 넘어갑니다)
    workflow.add_edge("tutor_question", "analyze_answer")
    
    # 결과를 GET으로 반환
    workflow.add_edge("analyze_answer", END)

    # 5. 그래프 컴파일
    return workflow.compile()

# 외부(router.py)에서 사용할 그래프 인스턴스
chatbot_graph = define_graph()