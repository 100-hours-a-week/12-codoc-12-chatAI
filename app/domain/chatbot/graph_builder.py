from langgraph.graph import StateGraph, START, END
from app.domain.chatbot.bot_state import ChatBotState
from app.domain.chatbot.nodes.load_node import load_problem_node
from app.domain.chatbot.nodes.tutor_node import tutor_node
from app.domain.chatbot.nodes.analyzer_node import analyzer_node
from app.domain.chatbot.nodes.finalizer_node import finalizer_node
from app.domain.chatbot.nodes.knowledge_node import knowledge_node


def _route_by_message_type(state: ChatBotState) -> str:
    """
    message_type에 따라 분기합니다.
    - ANSWER: load_problem → 분석 → 튜터 흐름
    - QUESTION: Knowledge Node (MCP 에이전트) - load_problem 스킵
    """
    if state.get("message_type") == "QUESTION":
        return "knowledge"
    return "load_problem"


def _route_after_tutor(state: ChatBotState) -> str:
    """
    INSIGHT 단계에서 정답을 맞혔으면 Finalizer로, 아니면 종료합니다.
    """
    if state.get("paragraph_type") == "INSIGHT" and state.get("is_correct", False):
        return "finalizer"
    return END


def define_graph():
    # 1. StateGraph 초기화
    workflow = StateGraph(ChatBotState)

    # 2. 노드 등록
    workflow.add_node("load_problem", load_problem_node)
    workflow.add_node("analyze_answer", analyzer_node)
    workflow.add_node("tutor_question", tutor_node)
    workflow.add_node("finalizer", finalizer_node)
    workflow.add_node("knowledge", knowledge_node)

    # 3. 엣지 연결

    # START에서 message_type으로 분기 (load_problem 실행 전)
    workflow.add_conditional_edges(
        START,
        _route_by_message_type,
        {"load_problem": "load_problem", "knowledge": "knowledge"},
    )

    # QUESTION 흐름: knowledge → END
    workflow.add_edge("knowledge", END)

    # ANSWER 흐름: load_problem → analyze_answer → tutor_question
    workflow.add_edge("load_problem", "analyze_answer")
    workflow.add_edge("analyze_answer", "tutor_question")

    # tutor_question 후 INSIGHT 완료 여부로 분기
    workflow.add_conditional_edges(
        "tutor_question",
        _route_after_tutor,
        {"finalizer": "finalizer", END: END},
    )

    workflow.add_edge("finalizer", END)

    # 4. 그래프 컴파일
    return workflow.compile()

# 외부(router.py)에서 사용할 그래프 인스턴스
chatbot_graph = define_graph()