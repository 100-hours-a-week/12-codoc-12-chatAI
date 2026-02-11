from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.domain.chatbot.bot_service import bot_service, WorkflowStatus
from app.common.config import llm
from app.common.api_response import CommonResponse
from langfuse.langchain import CallbackHandler
from .graph_builder import chatbot_graph 
from .bot_state import ChatBotState
from langchain_core.messages import HumanMessage
from . import bot_schemas
from typing import Dict, Any
from enum import Enum
import asyncio
import json


router = APIRouter(prefix="/chatbot", tags=["chatbot"])

langfuse_handler = CallbackHandler()

@router.post("", response_model=CommonResponse[bot_schemas.UserMsgCreateRes])
async def chat(post : bot_schemas.UserMsgCreateReq):    
    """사용자 메시지를 받아 LangGraph 워크플로우를 실행하고 SSE로 응답 스트리밍"""    
    sse_headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive"
    }
    
    return StreamingResponse(
        bot_service.run_and_stream(post),
        media_type="text/event-stream",
        headers=sse_headers
    )

# 워크플로우 취소 API 엔드포인트 추가
# @router.delete("/{run_id}")
# async def cancel_workflow(run_id: int):
#     if run_id in bot_controller.workflow_status:
#         bot_controller.workflow_status[run_id]["status"] = WorkflowStatus.CANCELED
#         return CommonResponse.success_response(message="워크플로우가 취소되었습니다.")
#     else:
#         return CommonResponse.error_response(code="NOT_FOUND", message="워크플로우를 찾을 수 없습니다.")