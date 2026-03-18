from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from app.domain.chatbot.bot_state import ChatBotState
from app.common.config import chatbot, llm
import asyncio

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
    
    [유저의 수준] : {level_guide}

    [핵심 키워드]: {essential_keywords}

    [판단 가이드라인 - 반드시 준수]:
    1. **키워드 우선 원칙**: 유저의 답변에 [핵심 키워드] 중 하나라도 직접 포함되어 있거나, 그와 기술적으로 동등한 의미(예: 딕셔너리-해시맵, 2차원 배열-리스트-표)를 가진 단어가 있다면 무조건 'true'로 판정합니다.
    2. **너그러운 의미 해석**: 유저가 완벽한 문장이 아닌 단답형(예: "딕셔너리요", "if문")으로 대답했더라도, 핵심 개념을 언급했다면 사고에 성공한 것으로 간주합니다.
    3. **오답 케이스**: 키워드가 아예 없거나, "모르겠어요", "힌트 주세요" 같이 사고를 포기하거나 [핵심 키워드] 자체를 물어보는 답변인 경우에만 'false'를 반환하세요.
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
    if not state.get("messages") or len(state["messages"]) == 0:
        return {"is_correct": False, "analyzer_reason": "유저 답변이 없습니다."}
    
    await asyncio.sleep(0.5)
    
    user_level = state.get("user_level", "newbie")
    essential_keywords = state.get("essential_keywords", [])
    essential_summary = state.get("essential_summary", "")
    
    # 레벨별 동적 가이드라인 설정
    if user_level == "newbie":
        level_guide = f"""
        [입문자 판정 가이드]:
        1. 핵심 키워드({essential_keywords})의 '의미'가 담겨 있다면 용어가 틀려도 정답입니다.
        2. 비유(예: 갈라진다, 바구니 등)를 적극 수용하고, 정답 시 기술 용어로 교정해줄 준비를 하세요.
        3. 정답 기준: 키워드 의미의 60% 이상 부합 시 True.
        """
    elif user_level == "pupil":
        level_guide = f"""
        [중급자 판정 가이드]:
        1. 핵심 키워드({essential_keywords})가 직접 언급되거나 매우 근접한 용어여야 합니다.
        2. 키워드가 빠졌다면 구체적으로 어떤 개념이 빠졌는지 이유(reason)에 적어주세요.
        3. 정답 기준: 주요 키워드 포함 및 개념 이해 시 True.
        """
    else: # specialist
        level_guide = f"""
        [고급자 판정 가이드]:
        1. 개별 단어 매칭보다 전체 로직({essential_summary})의 인과관계가 맞는지 확인하세요.
        2. 유저가 효율적인 알고리즘이나 최적화 관점에서 설명해도 로직이 맞다면 정답입니다.
        3. 정답 기준: 전체 요약문과의 논리적 일치도 80% 이상 시 True.
        """
        
    # 1. 분석용 체인 구성 (JSON 출력 강제)
    chain = ANALYZER_PROMPT | chatbot | JsonOutputParser()
    
    # ainvoke에 전달할 공통 인자 설정
    invoke_params = {
        "level_guide": level_guide,
        "essential_keywords": essential_keywords,
        "user_message": state["messages"][-1].content
    }

    # 3. 유저의 답변 분석 실행 (429 에러 대응 재시도 로직 적용)
    try:
        result = await chain.ainvoke(invoke_params)
        
        # result가 dict인지 먼저 확인하는 방어 코드 추가
        if not isinstance(result, dict):
            print(f"DEBUG: result type is {type(result)}")
            # 만약 객체라면 dict로 변환을 시도하거나 에러 처리
            raise TypeError(f"Expected dict, got {type(result)}")
        
    except Exception as e:
        if "429" in str(e):
            print("⚠️ API Rate Limit(429) 감지: 2초 후 재시도합니다.")
            await asyncio.sleep(2)
            # 재시도
            result = await chain.ainvoke(invoke_params)
        else:
            # 429 외의 에러는 그대로 발생시킴
            print(f"❌ 분석 노드에서 예외 발생: {e}")
            raise e
    
    # 4. 결과 파싱 및 상태 업데이트
    is_correct = result.get("is_correct", False)
    current_answer = ""
    new_retry_count = state.get("retry_count", 0)
    
    # 결과 업데이트 로직
    if is_correct:
        # 레벨에 따라 정답으로 저장할 텍스트를 선택
        if user_level == "specialist":
            current_answer = essential_summary
        else:
            # 리스트를 문자열로 합쳐서 저장
            current_answer = ", ".join(essential_keywords) if isinstance(essential_keywords, list) else essential_keywords
    else:
        new_retry_count += 1
    
    return {
        "is_correct": is_correct,
        "retry_count": new_retry_count,
        "current_answer": current_answer,
        "analyzer_reason": result.get("reason", "") 
    }