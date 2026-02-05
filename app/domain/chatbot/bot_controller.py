import asyncio
import json
from enum import Enum
from typing import Dict, Any
from langchain_core.messages import HumanMessage
from .graph_builder import chatbot_graph
from .bot_schemas import UserMsgCreateReq, UserMsgCreateRes
from .bot_state import ChatBotState

class WorkflowStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"
    
class ChatBotController:
    def __init__(self):
        # 실행 중인 상태 저장 (v2부터는 Redis 이용)
        self.workflow_status: Dict[str, Dict[str, Any]] = {}

    async def execute_workflow(self, run_id: int, request: UserMsgCreateReq):
        """LangGraph 워크플로우를 실행하고 이벤트를 셋업합니다."""
        try:
            self.workflow_status[run_id]["status"] = WorkflowStatus.PROCESSING
            initial_state : ChatBotState = {
                "messages": [HumanMessage(content=request.user_message)],
                "user_id": request.user_id,
                "user_level": request.user_level,
                "problem_id": request.problem_id,
                "run_id": run_id,
                "paragraph_type": request.paragraph_type if request.paragraph_type else "BACKGROUND",
                "is_correct": False
            }
            self.workflow_status[run_id]["paragraph_type"] = initial_state["paragraph_type"]
            
            # 이벤트를 제너레이터 형태로 저장
            self.workflow_status[run_id]["event_generator"] = chatbot_graph.astream_events(
                initial_state, version="v2"
            )
            self.workflow_status[run_id]["status"] = WorkflowStatus.COMPLETED
        except Exception as e:
            self.workflow_status[run_id]["status"] = WorkflowStatus.FAILED
            self.workflow_status[run_id]["error"] = str(e)

    async def get_event_generator(self, run_id: int):
        """스트리밍 응답을 위한 제너레이터를 생성합니다."""
        try:
        # 워크플로우 시작 대기 로직
            wait_time = 0
            while "event_generator" not in self.workflow_status.get(run_id, {}):
                await asyncio.sleep(0.1)
                wait_time += 0.1
                if wait_time > 5.0: raise TimeoutError("워크플로우 시작 지연")

            accumulated_text = ""
            async for event in self.workflow_status[run_id]["event_generator"]:
                kind = event["event"]
                node_name = event.get("metadata", {}).get("langgraph_node", "")

                # 1. 노드 시작 알림
                if kind == "on_chain_start" and event["name"] == "tutor_question":
                    current_type = self.workflow_status[run_id].get("paragraph_type", "UNKNOWN")
                    yield f'event: status\ndata: {json.dumps({"paragraph_type": current_type}, ensure_ascii=False)}\n\n'

                # 2. 토큰 스트리밍 (Tutor는 유저에게, Analyzer는 로그로)
                elif kind == "on_chat_model_stream":
                    content = event["data"].get("chunk", None)
                    if content and hasattr(content, 'content'):
                        token = content.content
                        # 튜터 노드의 응답만 클라이언트로 스트리밍
                        if node_name == "tutor_question":
                            accumulated_text += token
                            yield f'event: token\ndata: {json.dumps({"text": token}, ensure_ascii=False)}\n\n'
                        # 분석 노드의 로그는 서버 콘솔에만 기록
                        elif node_name == "analyze_answer":
                            print(f"\033[94m[Analyzer Log]\033[0m {token}", end="", flush=True)

                # 3. 분석 결과 처리
                elif kind == "on_chain_end" and event["name"] == "analyze_answer":
                    print("\n[Analysis Complete]")
                    output = event["data"]["output"]
                    # analyzer_node가 리턴한 dict 값들을 workflow_status에 업데이트
                    if isinstance(output, dict):
                        self.workflow_status[run_id]["is_correct"] = output.get("is_correct", False)
                        self.workflow_status[run_id]["current_answer"] = output.get("current_answer", "")
                        self.workflow_status[run_id]["analyzer_reason"] = output.get("analyzer_reason", "")

            # 4. 최종 결과 전송
            final_data = {
                "status": WorkflowStatus.COMPLETED,
                "ai_message": accumulated_text,
                "paragraph_type": self.workflow_status[run_id].get("paragraph_type", ""),
                "is_correct": self.workflow_status[run_id].get("is_correct", False),
                "current_answer": self.workflow_status[run_id].get("current_answer", "")
            }
            yield f'event: final\ndata: {json.dumps({"code": "SUCCESS", "message": "OK", "result": final_data}, ensure_ascii=False)}\n\n'
        
        except asyncio.CancelledError:
            # 유저가 브라우저 나가거나 연결 끊었을 때
            print(f"⚠️ 연결 취소됨 (run_id: {run_id})")
            self.workflow_status[run_id]["status"] = WorkflowStatus.CANCELED  
                      
        except Exception as e:
            # 예상치 못한 서버 에러 발생 시
            print(f"❌ 스트리밍 에러: {e}")
            
            self.workflow_status[run_id]["status"] = WorkflowStatus.FAILED
            self.workflow_status[run_id]["error"] = str(e)
            yield f'event: error\ndata: {json.dumps({"code": "FAILED", "message": str(e)}, ensure_ascii=False)}\n\n'
        
        finally:
            # 작업 종료 후 메모리 정리
            if run_id in self.workflow_status:
                del self.workflow_status[run_id]
            print(f"스트리밍 종료 (run_id: {run_id})")


# 컨트롤러 인스턴스 생성
bot_controller = ChatBotController()