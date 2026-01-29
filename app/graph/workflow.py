from langgraph.graph import StateGraph, START, END
from app.graph.state import AgentState
from app.graph.nodes import(
    intent_router_node, semantic_validator_node, helper_node, stage_manager_node
)

def route_by_intent(state: AgentState):
    # 일반 질문은 헬퍼로, 카드 진행은 검증으로
    if state["intent"] == "GENERAL_INQUIRY":
        return "general_helper"
    return "validator"

def route_by_validation(state: AgentState):
    if state["is_validated"]:
        return "manager"
    return "socratic_helper"

workflow = StateGraph(AgentState)

workflow.add_node("intent_router", intent_router_node)
workflow.add_node("semantic_validator", semantic_validator_node)
workflow.add_node("helper", helper_node)
workflow.add_node("stage_manager", stage_manager_node)

# --- 에지(Edge) 연결 ---
workflow.add_edge(START, "intent_router")

workflow.add_conditional_edges(
    "intent_router",
    route_by_intent,
    {
        "general_helper": "helper",
        "validator": "semantic_validator"
    }
)

workflow.add_conditional_edges(
    "semantic_validator",
    route_by_validation,
    {
        "manager": "stage_manager",
        "socratic_helper": "helper"
    }
)

workflow.add_edge("helper", END)
workflow.add_edge("stage_manager", END)

app = workflow.compile()