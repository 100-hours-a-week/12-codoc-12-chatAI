from langchain_core.messages import HumanMessage
from app.domain.chatbot.bot_state import ChatBotState
from app.domain.chatbot.prompts import FINALIZER_PROMPT
from app.common.config import llm
import asyncio

STAGE_ORDER = ["BACKGROUND", "GOAL", "STRATEGY", "INSIGHT"]
STAGE_LABELS = {
    "BACKGROUND": "배경 파악",
    "GOAL": "목표 설정",
    "STRATEGY": "전략 수립",
    "INSIGHT": "예외 및 검증",
}

async def finalizer_node(state: ChatBotState) -> dict:
    """
    INSIGHT 단계 정답 후 호출됩니다.
    state["messages"]에서 단계별 유저 답변을 수집하여
    전체 학습 흐름을 정리한 마무리 메시지를 생성합니다.
    """
    await asyncio.sleep(0.5)

    # 1. 단계별 유저 답변 수집
    # - Redis에서 로드된 past_messages는 additional_kwargs에 paragraph_type, is_correct 포함
    # - 정답 메시지를 우선 사용하고, 없으면 마지막 오답으로 fallback
    # - 현재 INSIGHT 메시지는 additional_kwargs 없이 마지막 HumanMessage로 존재
    stage_answers = {}
    current_paragraph_type = state.get("paragraph_type")  # "INSIGHT"

    for msg in state["messages"]:
        if not isinstance(msg, HumanMessage):
            continue
        p_type = msg.additional_kwargs.get("paragraph_type")
        if p_type:
            is_msg_correct = msg.additional_kwargs.get("is_correct", False)
            # 아직 해당 단계 답변이 없거나, 이번이 정답이면 업데이트 (정답 우선)
            if p_type not in stage_answers or is_msg_correct:
                stage_answers[p_type] = msg.content
        elif current_paragraph_type and current_paragraph_type not in stage_answers:
            # 현재 INSIGHT 메시지(아직 Redis 미저장, is_correct=True 확정)
            stage_answers[current_paragraph_type] = msg.content

    # 2. 정렬된 문자열 포맷팅
    stage_lines = []
    for stage in STAGE_ORDER:
        label = STAGE_LABELS[stage]
        answer = stage_answers.get(stage, "(답변 없음)")
        stage_lines.append(f"- [{label}] {answer}")
    formatted_answers = "\n".join(stage_lines)

    print(f"📋 [Finalizer] 수집된 단계별 답변:\n{formatted_answers}")

    # 3. 마무리 메시지 생성
    chain = FINALIZER_PROMPT | llm

    try:
        response = await chain.ainvoke({
            "stage_answers": formatted_answers,
            "user_level": state.get("user_level", "newbie"),
            "content": state.get("content", ""),
        })
    except Exception as e:
        if "429" in str(e):
            print("⚠️ [Finalizer] Rate Limit(429) 감지: 2.5초 후 재시도합니다...")
            await asyncio.sleep(2.5)
            response = await chain.ainvoke({
                "stage_answers": formatted_answers,
                "user_level": state.get("user_level", "newbie"),
                "content": state.get("content", ""),
            })
        else:
            print(f"❌ [Finalizer] 예상치 못한 에러 발생: {e}")
            raise e

    return {"messages": [response]}
