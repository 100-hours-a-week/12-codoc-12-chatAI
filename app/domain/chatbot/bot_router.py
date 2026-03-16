from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import StreamingResponse
from app.domain.chatbot.bot_service import bot_service
from app.common.config import llm
from app.common.api_response import CommonResponse
from .graph_builder import chatbot_graph 
from .bot_state import ChatBotState
from langchain_core.messages import HumanMessage
from langfuse.decorators import langfuse_context, observe
from . import bot_schemas
from typing import Dict, Any
from enum import Enum
import asyncio
import json


router = APIRouter(prefix="/chatbot", tags=["chatbot"])


@router.post("", response_model=CommonResponse[bot_schemas.UserMsgCreateRes])
@observe(name="Chatbot-Full-Flow", capture_output=False) #trace 자동 생성
async def chat(post : bot_schemas.UserMsgCreateReq):    
    """사용자 메시지를 받아 LangGraph 워크플로우를 실행하고 SSE로 응답 스트리밍"""  
     
    # Langfuse 컨텍스트에 사용자 ID 및 태그 추가 및 업데이트
    langfuse_context.update_current_trace(
        user_id=post.user_id,
        tags=["chatbot", "dev"]
    )
    
    trace_id = langfuse_context.get_current_trace_id()

    sse_headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive"
    }
    
    return StreamingResponse(
        bot_service.run_and_stream(post, trace_id=trace_id),
        media_type="text/event-stream",
        headers=sse_headers
    )

@router.delete("/{run_id}")
async def cancel_workflow(run_id: int):
    if bot_service.cancel_run(run_id):
        return CommonResponse.success_response(message="워크플로우가 취소되었습니다.")
    else:
        return CommonResponse.fail_response(code="NOT_FOUND", message="워크플로우를 찾을 수 없습니다.")

@router.post("/sessions", response_model=CommonResponse[bot_schemas.ExpireSessionRes])
async def expire_session(
    post: bot_schemas.ExpireSessionReq, 
    background_tasks: BackgroundTasks
):
    background_tasks.add_task(
        bot_service.from_redis_to_user_memories,
        session_id=post.session_id, 
        user_id=post.user_id, 
        problem_id=post.problem_id
    )
    return CommonResponse.success_response(message="세션이 만료되어 Redis에서 User Memories로 이동하는 작업이 백그라운드에서 처리됩니다.")
