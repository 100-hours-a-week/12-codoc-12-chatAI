from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import StreamingResponse
from app.common.config import llm
from app.common.api_response import CommonResponse
from . import bot_schemas
from typing import Dict, Any
from enum import Enum
import asyncio
import json


router = APIRouter(prefix="/chatbot", tags=["chatbot"])

class WorkflowStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"
    
# 실행 중인 워크 플로우 상태 저장 -> 실제론 DB 사용
workflow_status: Dict[str, Dict[str, Any]] = {}

async def execute_langgraph_workflow(run_id: int, request:bot_schemas.UserMsgCreateReq):
    # 여기에 LangGraph 워크플로우 실행 로직을 구현합니다.
    # 예시로, run_id와 request 데이터를 사용하여 워크플로우를 실행한다고 가정합니다.
    # 실제 구현은 LangGraph의 API 또는 SDK를 사용하여 작성해야 합니다.
    
    try:
        workflow_status[run_id]["status"] = WorkflowStatus.PROCESSING
        
        # LangGraph 워크플로우 실행 로직 추가
        
        workflow_status[run_id]["status"] = WorkflowStatus.COMPLETED
        workflow_status[run_id]["result"] = "작업 완료"

    except asyncio.CancelledError:
        workflow_status[run_id]["status"] = WorkflowStatus.CANCELED
    except Exception as e:
        workflow_status[run_id]["status"] = WorkflowStatus.FAILED
        workflow_status[run_id]["error"] = str(e)

@router.post("", response_model=CommonResponse[bot_schemas.UserMsgCreateRes])
async def chat(
    post : bot_schemas.UserMsgCreateReq,
    background_tasks : BackgroundTasks
)-> CommonResponse[Dict[str, Any]]:
    # 메세지 받고 백그라운드에서 워크플로우 실행
    
    run_id = post.run_id
    
    workflow_status[run_id] = {
        "status": WorkflowStatus.ACCEPTED,
        "user_id" : post.user_id,
        "problem_id" : post.problem_id,
        "prompt" : post.user_message
    }   

    background_tasks.add_task(execute_langgraph_workflow, run_id, post)
    
    return CommonResponse.success_response(
        message="채팅 요청이 접수되었습니다.",
        data=bot_schemas.UserMsgCreateRes(
            run_id=run_id,
            status=WorkflowStatus.ACCEPTED
        )
    )

@router.get("/{run_id}/stream", response_model=CommonResponse[Dict[str, Any]])
async def get_chat_stream(run_id: int):
    
    if run_id not in workflow_status:
        return StreamingResponse(
            content=iter([
                f'event: error\ndata: {json.dumps({"code": "NOT_FOUND", "message": "워크플로우를 찾을 수 없습니다."}, ensure_ascii=False)}\n\n'
            ]),
            media_type="text/event-stream"
        )
        
    async def generate():
        yield f'event: status\ndata: {json.dumps({"code": "SUCCESS", "message": "OK", "result": {"status": WorkflowStatus.PROCESSING, "message": "작업 처리 중..."}}, ensure_ascii=False)}\n\n'
    
        try:
            accumulated_text = ""    
            async for chunk in llm.astream(workflow_status[run_id].get("prompt", "")):
                if workflow_status[run_id]["status"] == WorkflowStatus.CANCELED:
                        yield f'event: error\ndata: {json.dumps({"code": "CANCELED", "message": "작업이 취소되었습니다."}, ensure_ascii=False)}\n\n'
                        break
                    
                token_text = chunk.text
                accumulated_text += token_text
                
                yield f'event: token\ndata: {json.dumps({"code": "SUCCESS", "message": "OK", "result": {"text": token_text}}, ensure_ascii=False)}\n\n'
                
            final_response = {
                "status": WorkflowStatus.COMPLETED,
                "ai_message": accumulated_text,
                "current_node": workflow_status[run_id].get("current_node", ""),
                 "is_correct": workflow_status[run_id].get("is_correct", False),
                "current_answer": workflow_status[run_id].get("current_answer", "")
            }
            yield f'event: final\ndata: {json.dumps({"code": "SUCCESS", "message": "OK", "result": final_response}, ensure_ascii=False)}\n\n'
        
        except Exception as e:
            workflow_status[run_id]["status"] = WorkflowStatus.FAILED
            workflow_status[run_id]["error"] = str(e)
            yield f'event: error\ndata: {json.dumps({"code": "FAILED", "message": str(e)}, ensure_ascii=False)}\n\n'
        
    return StreamingResponse(generate(), media_type="text/event-stream")

# 워크플로우 취소 API 엔드포인트 추가
