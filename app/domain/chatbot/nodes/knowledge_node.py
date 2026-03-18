import os
import asyncio
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

from app.domain.chatbot.bot_state import ChatBotState
from app.domain.chatbot.prompts import KNOWLEDGE_SYSTEM_PROMPT, KNOWLEDGE_SYNTHESIS_PROMPT
from app.common.config import llm, chatbot

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8001")


async def _run_agent(llm, tools, system_prompt, messages):
    agent = create_agent(llm, tools)
    # 에이전트에는 현재 유저 메시지만 전달 (과거 대화 포함 시 Gemini가 직접 답변하려는 경향)
    # 과거 대화 컨텍스트는 이후 EXAONE 합성 단계에서 활용
    last_human = next(
        (m for m in reversed(messages) if isinstance(m, HumanMessage)),
        messages[-1],
    )
    return await agent.ainvoke({
        "messages": [SystemMessage(content=system_prompt), last_human]
    })


async def knowledge_node(state: ChatBotState) -> dict:
    """
    message_type == "QUESTION" 일 때 호출됩니다.
    Gemini(llm)가 ReAct 에이전트로 MCP 툴을 호출하여 정보를 수집하고,
    EXAONE(chatbot)이 수집된 정보를 바탕으로 최종 답변을 생성합니다.
    """
    user_level = state.get("user_level", "newbie")
    paragraph_type = state.get("paragraph_type", "BACKGROUND")
    problem_id = state.get("problem_id")
    user_id = state.get("user_id")
    session_id = state.get("session_id", "")

    # Gemini 에이전트용 시스템 프롬프트 (툴 호출 판단용)
    system_prompt = KNOWLEDGE_SYSTEM_PROMPT.format(
        user_level=user_level,
        paragraph_type=paragraph_type,
    ) + (
        f"\n[현재 컨텍스트]\n"
        f"- problem_id: {problem_id}\n"
        f"- paragraph_type: {paragraph_type}\n"
        f"- user_id: {user_id}\n"
        f"- session_id: {session_id}\n"
        "도구 호출 시 위 값을 파라미터로 사용하세요."
    )

    async def run_with_client():
        mcp_client = MultiServerMCPClient(
            {
                "codoc": {
                    "url": f"{MCP_SERVER_URL}/mcp",
                    "transport": "streamable_http",
                }
            }
        )
        tools = await mcp_client.get_tools()
        print(f"📦 [Knowledge] 로드된 MCP 툴 ({len(tools)}개): {[t.name for t in tools]}")
        return await _run_agent(llm, tools, system_prompt, state["messages"])

    try:
        result = await run_with_client()
    except Exception as e:
        if "429" in str(e):
            print("⚠️ [Knowledge] Rate Limit(429) 감지: 2.5초 후 재시도합니다..")
            await asyncio.sleep(2.5)
            result = await run_with_client()
        else:
            print(f"❌ [Knowledge] 예상치 못한 에러: {e}")
            raise e

    # ── Step 2: 툴 결과를 EXAONE에게 넘겨 최종 답변 생성 ──────────────────────
    def _extract_tool_content(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                item.get("text", str(item)) if isinstance(item, dict) else str(item)
                for item in content
            )
        return str(content)

    tool_calls = [
        m.tool_calls for m in result["messages"]
        if hasattr(m, "tool_calls") and m.tool_calls
    ]
    for calls in tool_calls:
        for call in calls:
            print(f"🔧 [MCP Tool Called] {call['name']}({call['args']})")

    tool_results = [
        _extract_tool_content(m.content)
        for m in result["messages"]
        if isinstance(m, ToolMessage)
    ]

    if tool_results:
        synthesis_prompt = KNOWLEDGE_SYNTHESIS_PROMPT.format(
            user_level=user_level,
            paragraph_type=paragraph_type,
            tool_results="\n\n---\n\n".join(tool_results),
        )
        # state["messages"]에는 Redis 히스토리 + 현재 유저 메시지가 포함되어 있음
        # 고정 버튼 메시지만으론 컨텍스트가 부족하므로 대화 흐름 전체를 전달
        final_message = await chatbot.ainvoke([
            SystemMessage(content=synthesis_prompt),
            *state["messages"],
        ])
    else:
        # 툴이 호출되지 않은 경우 Gemini 답변을 폴백으로 사용
        print("⚠️ [Knowledge] 툴 미호출 - Gemini 답변 폴백 사용")
        ai_messages = [m for m in result["messages"] if hasattr(m, "type") and m.type == "ai"]
        final_message = ai_messages[-1] if ai_messages else result["messages"][-1]

    return {"messages": [final_message]}
