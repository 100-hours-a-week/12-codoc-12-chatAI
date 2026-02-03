from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from app.domain.chatbot.bot_state import ChatBotState
from app.common.config import llm

# semantic 판단을 위한 전용 프롬프트 설계
# ANALYZER_PROMPT = ChatPromptTemplate.from_messages([
#     ("system", """당신은 유저의 답변이 문제 해결을 위한 핵심 내용을 포함하고 있는지 판단하는 검증가입니다.
    
#     [핵심 키워드/조건]:
#     {essential_keywords}
    
#     [판단 기준]:
#     1. 유저의 답변이 핵심 키워드의 의미를 충분히 담고 있는가?
#     2. 단순히 모르겠다는 답변이 아니라, 스스로 사고하여 결론에 도달했는가?
    
#     반드시 아래 JSON 형식으로만 응답하세요:
#     {{
#         "is_correct": true 또는 false,
#         "reason": "판단 이유를 간략히 설명"
#     }}"""),
#     ("user", "유저 답변: {user_message}")
# ])

# 하이브리드 검증 방시
ANALYZER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """당신은 유저의 코딩 테스트 사고 과정을 검증하는 전문 AI 튜터입니다.
    유저의 답변이 [핵심 키워드]를 포함하고 있는지, 그리고 문제 해결 의도가 있는지 판단하세요.

    [핵심 키워드]: {essential_keywords}

    [판단 가이드라인 - 반드시 준수]:
    1. **키워드 우선 원칙**: 유저의 답변에 [핵심 키워드] 중 하나라도 직접 포함되어 있거나, 그와 기술적으로 동등한 의미(예: 딕셔너리-해시맵, 2차원 배열-리스트-표)를 가진 단어가 있다면 무조건 'true'로 판정합니다.
    2. **너그러운 의미 해석**: 유저가 완벽한 문장이 아닌 단답형(예: "딕셔너리요", "if문")으로 대답했더라도, 핵심 개념을 언급했다면 사고에 성공한 것으로 간주합니다.
    3. **오답 케이스**: 키워드가 아예 없거나, "모르겠어요", "힌트 주세요" 같이 사고를 포기한 답변인 경우에만 'false'를 반환하세요.
    4. **맥락 고려**: 핵심 키워드가 문장 속에 녹아있다면, 문법이 조금 틀리더라도 정답으로 인정합니다.

    반드시 아래 JSON 형식으로만 응답하세요:
    {{
        "is_correct": true 또는 false,
        "reason": "판단 이유를 1문장으로 설명"
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