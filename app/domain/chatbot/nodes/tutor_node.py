from app.domain.chatbot.bot_state import ChatBotState
from app.domain.chatbot.prompts import PROMPTS
from app.common.config import llm

async def tutor_node(state: ChatBotState) -> dict:
    """
    현재 단계(current_node)에 맞는 프롬프트를 사용하여 
    유저에게 단계별 가이드 질문을 생성합니다.
    """
    # 1. 현재 단계에 맞는 프롬프트 템플릿 선택
    current_step = state.get("current_node")
    prompt_template = PROMPTS.get(current_step)
    
    if not prompt_template:
        raise ValueError(f"정의되지 않은 노드 단계입니다: {current_step}")

    # 2. 체인 구성 (프롬프트 | 모델)
    chain = prompt_template | llm

    # 3. 답변 생성
    # load_node에서 저장한 가이드 정보와 유저의 마지막 메시지를 주입합니다.
    response = await chain.ainvoke({
        "content": state.get("content"),
        "chatbot_answer_guide": state.get("chatbot_answer_guide"),
        "essential_keywords": state.get("essential_keywords"), # 내부 참조용으로만 사용
        "user_message": state["messages"][-1].content
    })

    # 4. 생성된 AI 메시지를 State에 추가하도록 반환
    return {"messages": [response]}