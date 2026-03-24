from app.domain.chatbot.bot_state import ChatBotState
from app.domain.chatbot.prompts import PROMPTS, STAGE_INSTRUCTIONS
from app.common.config import chatbot, llm
import asyncio

async def tutor_node(state: ChatBotState) -> dict:
    """
    현재 단계(current_node)에 맞는 프롬프트를 사용하여 
    유저에게 단계별 가이드 질문을 생성합니다.
    """
    
    await asyncio.sleep(3.0)
    
    # 1. 현재 단계에 맞는 프롬프트 템플릿 선택
    current_type = state.get("paragraph_type")
    prompt_template = PROMPTS.get(current_type)
    analyzer_reason = state.get("analyzer_reason", "")
    is_correct = state.get("is_correct", False)
    user_level = state.get("user_level", "newbie")
    
    if not prompt_template:
        raise ValueError(f"정의되지 않은 노드 단계입니다: {current_type}")
    
    # 유저 학습 기억 컨텍스트 빌드
    user_memory = state.get("user_memory")
    if user_memory:
        weak_tags = user_memory.get("weak_tags") or []
        error_paragraph = user_memory.get("error_paragraph") or []
        error_summary = user_memory.get("error_summary") or ""
        user_memory_context = (
            f"- 취약 태그: {', '.join(weak_tags) if weak_tags else '없음'}\n"
            f"- 자주 틀리는 단계: {', '.join(error_paragraph) if isinstance(error_paragraph, list) else error_paragraph or '없음'}\n"
            f"- 실수 요약: {error_summary or '없음'}"
        )
    else:
        user_memory_context = "학습 기억 없음 (첫 세션이거나 최근 1주일 내 기록 없음)"

    print(f"🧠 [Tutor] user_memory_context:\n{user_memory_context}")

    # is_correct에 따라 해당 케이스 지시문만 선택
    raw_instruction = STAGE_INSTRUCTIONS.get((current_type, is_correct), "")
    stage_instruction = raw_instruction.format(analysis_reason=analyzer_reason)

    print(f"📋 [Tutor] stage_instruction key: ({current_type}, {is_correct})")

    # invoke에 사용할 변수들 정리
    invoke_params = {
        "content": state.get("content"),
        "chatbot_answer_guide": state.get("chatbot_answer_guide"),
        "essential_keywords": ", ".join(state.get("essential_keywords", [])),
        "user_message": state["messages"][-1].content,
        "stage_instruction": stage_instruction,
        "user_level": user_level,
        "user_memory_context": user_memory_context,
    }
    
    # 2. 체인 구성 (프롬프트 | 모델)
    chain = prompt_template | chatbot

    # 3. 답변 생성 (429 에러 대응 재시도 로직 추가)
    try:
        response = await chain.ainvoke(invoke_params)
    except Exception as e:
        if "429" in str(e):
            print("⚠️ [Tutor] Rate Limit(429) 감지: 2.5초 후 재시도합니다...")
            await asyncio.sleep(2.5)  # 분석 노드보다 조금 더 길게 대기
            response = await chain.ainvoke(invoke_params)
        else:
            print(f"❌ [Tutor] 예상치 못한 에러 발생: {e}")
            raise e

    # 4. 생성된 AI 메시지를 State에 추가하도록 반환
    return {"messages": [response]}