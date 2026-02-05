from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import StreamingResponse
from app.domain.chatbot.bot_controller import bot_controller, WorkflowStatus
from app.common.config import llm
from app.common.api_response import CommonResponse
from .graph_builder import chatbot_graph 
from .bot_state import ChatBotState
from langchain_core.messages import HumanMessage
from . import bot_schemas
from typing import Dict, Any
from enum import Enum
import asyncio
import json


router = APIRouter(prefix="/chatbot", tags=["chatbot"])

@router.post("", response_model=CommonResponse[bot_schemas.UserMsgCreateRes])
async def chat(
    post : bot_schemas.UserMsgCreateReq,
    background_tasks : BackgroundTasks
)-> CommonResponse[Dict[str, Any]]:
    # 메세지 받고 백그라운드에서 워크플로우 실행
    
    run_id = post.run_id
    bot_controller.workflow_status[run_id] = {
        "status": WorkflowStatus.ACCEPTED,
        "user_level": post.user_level,
        "paragraph_type": post.paragraph_type if post.paragraph_type else "BACKGROUND"
    }    
    background_tasks.add_task(bot_controller.execute_workflow, run_id, post) 
    
    return CommonResponse.success_response(
        message="채팅 요청이 접수되었습니다.",
        data=bot_schemas.UserMsgCreateRes(
            run_id=run_id,
            status=WorkflowStatus.ACCEPTED
        )
    )

@router.get("/{run_id}/stream")
async def get_chat_stream(run_id: int):
    
    if run_id not in bot_controller.workflow_status:
            # 에러 메시지 생성기
            async def error_generator():
                yield f'event: error\ndata: {json.dumps({"code": "NOT_FOUND", "message": "워크플로우를 찾을 수 없습니다."}, ensure_ascii=False)}\n\n'
                
            return StreamingResponse(
                error_generator(),
                media_type="text/event-stream"
            )
                
    return StreamingResponse(
        bot_controller.get_event_generator(run_id), 
        media_type="text/event-stream")

# 워크플로우 취소 API 엔드포인트 추가
# @router.delete("/{run_id}")
# async def cancel_workflow(run_id: int):
#     if run_id in bot_controller.workflow_status:
#         bot_controller.workflow_status[run_id]["status"] = WorkflowStatus.CANCELED
#         return CommonResponse.success_response(message="워크플로우가 취소되었습니다.")
#     else:
#         return CommonResponse.error_response(code="NOT_FOUND", message="워크플로우를 찾을 수 없습니다.")