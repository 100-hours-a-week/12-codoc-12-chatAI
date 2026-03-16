import os
import asyncio
from langchain_core.messages import SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

from app.domain.chatbot.bot_state import ChatBotState
from app.domain.chatbot.prompts import KNOWLEDGE_SYSTEM_PROMPT
from app.common.config import llm

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8001")


async def knowledge_node(state: ChatBotState) -> dict:
    """
    message_type == "QUESTION" 일 때 호출됩니다.
    MCP 서버의 툴을 활용하는 ReAct 에이전트가 유저의 질문에 답변합니다.
    답변 후 유저가 현재 단계로 돌아가도록 유도합니다.
    """
    user_level = state.get("user_level", "newbie")
    paragraph_type = state.get("paragraph_type", "BACKGROUND")
    problem_id = state.get("problem_id")
    user_id = state.get("user_id")
    session_id = state.get("session_id", "")

    # 시스템 프롬프트 포매팅
    system_prompt = KNOWLEDGE_SYSTEM_PROMPT.format(
        user_level=user_level,
        paragraph_type=paragraph_type,
    )

    # MCP 툴 호출 시 에이전트가 참고할 컨텍스트를 시스템 메시지에 주입
    context_hint = (
        f"\n[현재 컨텍스트]\n"
        f"- problem_id: {problem_id}\n"
        f"- paragraph_type: {paragraph_type}\n"
        f"- user_id: {user_id}\n"
        f"- session_id: {session_id}\n"
        "도구 호출 시 위 값을 파라미터로 사용하세요."
    )

    try:
        async with MultiServerMCPClient(
            {
                "codoc": {
                    "url": f"{MCP_SERVER_URL}/mcp",
                    "transport": "streamable_http",
                }
            }
        ) as mcp_client:
            tools = mcp_client.get_tools()

            agent = create_agent(
                llm,
                tools,
                prompt=system_prompt + context_hint,
            )

            result = await agent.ainvoke({"messages": state["messages"]})

    except Exception as e:
        if "429" in str(e):
            print("⚠️ [Knowledge] Rate Limit(429) 감지: 2.5초 후 재시도합니다...")
            await asyncio.sleep(2.5)
            async with MultiServerMCPClient(
                {
                    "codoc": {
                        "url": f"{MCP_SERVER_URL}/mcp",
                        "transport": "streamable_http",
                    }
                }
            ) as mcp_client:
                tools = mcp_client.get_tools()
                agent = create_agent(llm, tools, prompt=system_prompt + context_hint)
                result = await agent.ainvoke({"messages": state["messages"]})
        else:
            print(f"❌ [Knowledge] 예상치 못한 에러: {e}")
            raise e

    # ReAct 에이전트의 최종 AI 메시지만 반환
    ai_messages = [m for m in result["messages"] if hasattr(m, "type") and m.type == "ai"]
    final_message = ai_messages[-1] if ai_messages else result["messages"][-1]

    return {"messages": [final_message]}
