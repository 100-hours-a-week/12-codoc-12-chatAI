import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from qdrant_client import QdrantClient, models
from app.graph.state import AgentState
from scripts.embedding import embed_and_upload as get_embeddings

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=api_key  # 직접 명시
)
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    https=False,
    port=6333,
    api_key=os.getenv("QDRANT_API_KEY")
)

# --- [Node 1: Intent Router] ---
def intent_router_node(state: AgentState):
    # 사용자 입력이 문제 풀이인지 일반 질문인지 판별
    last_message = state["messages"][-1].content

    system_prompt= """
    사용자의 메시지를 분석하여 의도를 분류하세요.
    - CARD_PROGRESS: 현재 단계의 문제를 해결하려 하거나 설명에 답변하는 경우
    - GENERAL_INQUIRY: "그리디가 뭐야?", "BFS 예시 알려줘" 같이 알고리즘 개념 자체를 묻는 경우
    결과는 반드시 'CARD_PROGRESS' 또는 'GENERAL_INQUIRY' 중 하나만 출력하세요.
    """

    response = llm.invoke([("system", system_prompt), ("human", last_message)])
    intent = response.content.strip().upper()

    query_vector = get_embeddings.model.encode(last_message).tolist()

    # Qdrant 검색 (전역 변수로 선언된 COLLECTION_NAME 사용)
    query_response = client.query_points(
        collection_name=get_embeddings.COLLECTION_NAME,
        query=query_vector,
        query_filter=models.Filter(must=[
            # 스크립트에서 int(problem_id)로 저장했으므로, 검색 시에도 int로 형변환 필수
            models.FieldCondition(key="problem_id", match=models.MatchValue(value=int(state["problem_id"]))),
            models.FieldCondition(key="paragraph_type", match=models.MatchValue(value=state["current_stage"]))
        ]),
        limit=1
    )

    # 결과 추출
    if query_response.points:
        retrieved = query_response.points[0].payload
        print(f"✅ [DEBUG] 데이터를 성공적으로 가져왔습니다: {retrieved.get('title')}")
    else:
        retrieved = {}
        print(f"⚠️ [DEBUG] 데이터를 찾지 못했습니다. 필터를 확인하세요.")

    return {"intent": intent, "retrieved_content": retrieved}

# --- [Node 2: Semantic Validator] ---
def semantic_validator_node(state: AgentState):
    # 카드 진행 상황일 때 유저의 답변이 필수 키워드와 의미적으로 일치하는지 검증
    if state["intent"] == "GENERAL_INQUIRY":
        return {"is_validated": False}

    last_message = state["messages"][-1].content
    essential_keywords = state["retrieved_content"].get("essential_keywords",[])

    system_prompt = f"""
    사용자의 답변이 아래 [필수 키워드]들의 의미를 충분히 포함하고 있는지 검증하세요.
    [필수 키워드]: {', '.join(essential_keywords)}
    
    결과를 'TRUE' 또는 'FALSE'로만 대답하세요. 단어의 일치가 아니라 '의미적 충분함'을 기준으로 합니다.
    """

    response = llm.invoke([("system", system_prompt), ("human", last_message)])
    is_validated = response.content.strip().upper() == "TRUE"

    return {"is_validated": is_validated}

# --- [Node 3: Socratic/General Helper] ---
def helper_node(state: AgentState):
    last_message = state["messages"][-1].content
    intent = state["intent"]
    is_validated = state["is_validated"]
    retrieved = state.get("retrieved_content", {})

    if intent == "GENERAL_INQUIRY":
        system_prompt = "개념을 설명하고 '다시 문제로 돌아갈까요?'라고 하세요."
    elif not is_validated:
        # 가이드가 없을 때를 대비한 기본 지침 강화
        guide = retrieved.get("chatbot_answer_guide", "유저가 문제 상황을 더 구체적으로 설명하도록 유도하세요.")

        system_prompt = f"""
        너는 소크라테스식 질문법을 쓰는 튜터야.
        [규칙]
        1. 절대로 유저의 말을 그대로 반복하거나 '~인 것 같네요'라며 동의만 하지 마.
        2. 정답을 직접 알려주지 마.
        3. 아래 [코칭 가이드]를 바탕으로 유저에게 '스스로 생각할 거리'를 던지는 짧은 질문을 해.
        4. 너무 길게 이야기 하지말고 간략하지만 자세하게 이야기해줘
        
        [코칭 가이드]: {guide}
        """
    else:
        return {"messages" : [("assistant", "정확해요! 그럼 이제 다음으로 넘어가 볼까요?")]}

    # 유저의 말을 단순히 던지는 게 아니라, '이 답변에 대해 질문하라'고 명시
    response = llm.invoke([
        ("system", system_prompt),
        ("human", f"유저의 답변: {last_message}\n위 답변을 보고 튜터로서 질문해줘.")
    ])

    return {"messages": [response]}

# --- [Node 4: State/Stage Manager] ---
def stage_manager_node(state: AgentState):
    # 조건 충족 시 단계 업데이트
    stages = ["CONTEXT", "GOAL", "STRATEGY", "INSIGHT", "COMPLETE"]
    current_idx = stages.index(state["current_stage"])

    if state["is_validated"] and current_idx < len(stages) -1:
        next_stage = stages[current_idx+1]
        return {"current_stage": next_stage, "is_validated": False}

    return {}
