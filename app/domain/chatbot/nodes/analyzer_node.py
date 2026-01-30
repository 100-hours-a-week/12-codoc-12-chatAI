from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from app.domain.chatbot.bot_state import ChatBotState
from app.common.config import llm

# 판단을 위한 전용 프롬프트 설계
ANALYZER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """당신은 유저의 답변이 문제 해결을 위한 핵심 내용을 포함하고 있는지 판단하는 검증가입니다.
    
    [핵심 키워드/조건]:
    {essential_keywords}
    
    [판단 기준]:
    1. 유저의 답변이 핵심 키워드의 의미를 충분히 담고 있는가?
    2. 단순히 모르겠다는 답변이 아니라, 스스로 사고하여 결론에 도달했는가?
    
    반드시 아래 JSON 형식으로만 응답하세요:
    {{
        "is_correct": true 또는 false,
        "reason": "판단 이유를 간략히 설명"
    }}"""),
    ("user", "유저 답변: {user_message}")
])

async def analyzer_node(state: ChatBotState) -> dict:
    """
    유저의 답변과 essential_keywords를 대조하여 정답 여부를 판별합니다.
    """
    # 1. 분석용 체인 구성 (JSON 출력 강제)
    chain = ANALYZER_PROMPT | llm | JsonOutputParser()
    
    # 2. 유저의 마지막 메시지와 키워드 대조
    result = await chain.ainvoke({
        "essential_keywords": state.get("essential_keywords"),
        "user_message": state["messages"][-1].content
    })
    
    is_correct = result.get("is_correct", False)
    
    current_answer = ""
    # 3. 결과 업데이트
    # is_correct가 False일 경우 retry_count를 1 증가시킵니다.
    new_retry_count = state.get("retry_count", 0)
    
    if is_correct:
        current_answer = state.get("essential_keywords")
    else:
        new_retry_count += 1
    
    return {
        "is_correct": is_correct,
        "retry_count": new_retry_count,
        "current_answer": current_answer
    }