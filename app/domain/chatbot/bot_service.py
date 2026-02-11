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
    
class ChatBotService:
    def __init__(self):
        # 실행 중인 상태 저장 (v2부터는 Redis 이용)
        self.workflow_status: Dict[str, Dict[str, Any]] = {}

    async def run_and_stream(self, request: UserMsgCreateReq):
        """LangGraph 워크플로우를 실행하고 생성된 이벤트를 SSE 형식으로 즉시 반환"""
        
        run_id = request.run_id
        
        initial_state : ChatBotState = {
            "messages": [HumanMessage(content=request.user_message)],
            "user_id": request.user_id,
            "user_level": request.user_level,
            "problem_id": request.problem_id,
            "run_id": run_id,
            "paragraph_type": request.paragraph_type if request.paragraph_type else "BACKGROUND",
            "is_correct": False
        }
        
        self.workflow_status[run_id] = {
            "status": WorkflowStatus.PROCESSING,
            "is_correct": False,
            "current_answer": "",
            "paragraph_type": initial_state["paragraph_type"]
        }
        
        accumulated_text = ""
        
        try:
            async for event in chatbot_graph.astream_events(initial_state, version="v2"):
                kind = event["event"]
                node_name = event.get("metadata", {}).get("langgraph_node", "")
                
                # 1. 노드 시작 알림
                if kind == "on_chain_start" and event["name"] == "tutor_question":
                    yield f'event: status\ndata: {json.dumps({"paragraph_type": initial_state["paragraph_type"], "status": "PROCESSING"}, ensure_ascii=False)}\n\n'
                    
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
                        
                # 3. 분석 결과 처리(tutor_question 시작되기 전 발생)
                elif kind == "on_chain_end" and event["name"] == "analyze_answer":
                    print("\n[Analysis Complete]")
                    output = event["data"]["output"]
                    # analyzer_node가 리턴한 dict 값들을 workflow_status에 업데이트
                    if isinstance(output, dict):
                        self.workflow_status[run_id].update({
                            "is_correct": output.get("is_correct", False),
                            "current_answer": output.get("current_answer", ""),
                            "analyzer_reason": output.get("analyzer_reason", "")
                        })
                        
            result_info = self.workflow_status.get(run_id)
            final_data = {
                "code": "SUCCESS",
                "message": "OK",
                "result": {
                    "status": WorkflowStatus.COMPLETED,
                    "ai_message": accumulated_text,
                    "paragraph_type": result_info.get("paragraph_type", "") if result_info else "",
                    "is_correct": result_info.get("is_correct", False) if result_info else False
                }
            }
            yield f'event: final\ndata: {json.dumps(final_data, ensure_ascii=False)}\n\n'
        
        except asyncio.CancelledError:
            # 유저가 브라우저 나가거나 연결 끊었을 때
            print(f"⚠️ 연결 취소됨 (run_id: {run_id})")
            self.workflow_status[run_id]["status"] = WorkflowStatus.CANCELED
        
        except Exception as e:
            print(f"❌ 에러 발생: {e}")
            error_msg = str(e)
            # 429 에러 등에 대한 메시지 처리
            if "429" in error_msg:
                error_msg = "현재 사용량이 많아 응답이 지연되고 있습니다. 잠시 후 다시 시도해주세요."
            
            yield f'event: error\ndata: {json.dumps({"code": "FAILED", "message": error_msg}, ensure_ascii=False)}\n\n'
            
        finally:
            # 작업 종료 후 메모리 정리
            if run_id in self.workflow_status:
                del self.workflow_status[run_id]
            print(f"워크플로우 종료 (run_id: {run_id})")


# 컨트롤러 인스턴스 생성
bot_service = ChatBotService()