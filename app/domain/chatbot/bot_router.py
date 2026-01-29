from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import StreamingResponse
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

class WorkflowStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"
    
# 실행 중인 워크 플로우 상태 저장 -> 실제론 DB 사용
workflow_status: Dict[str, Dict[str, Any]] = {}

async def execute_langgraph_workflow(run_id: int, request:bot_schemas.UserMsgCreateReq):
    
    try:
        workflow_status[run_id]["status"] = WorkflowStatus.PROCESSING
        
        # 1. 초기 State 구성
        initial_state: ChatBotState = {
            "messages": [HumanMessage(content=request.user_message)],
            "user_id": request.user_id,
            "problem_id": request.problem_id,
            "run_id": run_id,
            "current_node": request.current_node,
            "is_correct": False
        }
        # 2. 그래프 실행 및 이벤트 캐치
        # 스트리밍 API에서 참조할 수 있도록 이벤트를 생성합니다.
        workflow_status[run_id]["event_generator"] = chatbot_graph.astream_events(
            initial_state, version="v2"
        )
        
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
    workflow_status[run_id] = {"status": WorkflowStatus.ACCEPTED}
    
    # 워크플로우 백그라운드 실행
    background_tasks.add_task(execute_langgraph_workflow, run_id, post)
    
    
    # workflow_status[run_id] = {
    #     "status": WorkflowStatus.ACCEPTED,
    #     "user_id" : post.user_id,
    #     "problem_id" : post.problem_id,
    #     "prompt" : post.user_message
    # }   
    
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
            # 1. 제너레이터 대기 (타임아웃 추가)
            wait_time = 0
            while "event_generator" not in workflow_status[run_id]:
                await asyncio.sleep(0.1)
                wait_time += 0.1
            if wait_time > 5.0: # 5초 이상 대기 시 에러 처리
                raise TimeoutError("워크플로우 시작이 지연되고 있습니다.")
            if workflow_status[run_id].get("status") == WorkflowStatus.FAILED:
                raise Exception(workflow_status[run_id].get("error", "알 수 없는 에러"))
                
            accumulated_text = ""    
            async for event in workflow_status[run_id]["event_generator"]:
                kind = event["event"]
                node_name = event.get("metadata", {}).get("langgraph_node", "") # 현재 실행 중인 노드 이름
                
                # 1. 노드 시작 알림 (현재 어떤 단계인지)
                # 현재 실행 노드 정보 업데이트
                if kind == "on_chain_start" and event["name"] in ["tutor_question", "analyze_answer"]:
                    workflow_status[run_id]["current_node"] = event["name"]
                    yield f'event: status\ndata: {json.dumps({"current_node": event["name"]}, ensure_ascii=False)}\n\n'

                # 2. 실시간 토큰 스트리밍
                elif kind == "on_chat_model_stream":
                    token = event["data"]["chunk"].content
                    if token:
                        if node_name == "tutor_question": 
                            accumulated_text += token
                            yield f'event: token\ndata: {json.dumps({"text": token}, ensure_ascii=False)}\n\n'
                        elif node_name == "analyze_answer":
                            # [로그]터미널에 실시간으로 분석 토큰 출력 (줄바꿈 없이)
                            print(f"\033[94m[Analyzer Log]\033[0m {token}", end="", flush=True)

                # 3. 분석 결과 전송
                elif kind == "on_chain_end" and event["name"] == "analyze_answer":
                    print("\n") # 로그 가독성을 위한 줄바꿈
                    output = event["data"]["output"]
                    # output의 형태에 따라 조정 필요
                    is_correct = output.get("is_correct", False) if isinstance(output, dict) else False
                    workflow_status[run_id]["is_correct"] = is_correct
                
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
        finally:
            # 3. 작업 종료 후 메모리 정리 (선택 사항)
            # del workflow_status[run_id]
            pass
        
    return StreamingResponse(generate(), media_type="text/event-stream")

# 워크플로우 취소 API 엔드포인트 추가
