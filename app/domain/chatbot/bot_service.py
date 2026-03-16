import asyncio
import json
import os
import time
import uuid
from enum import Enum
from typing import Dict, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from app.common.config import llm
from qdrant_client.http import models
from qdrant_client import QdrantClient

from langfuse import Langfuse
langfuse_client = Langfuse()

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
        # self.workflow_status: Dict[str, Dict[str, Any]] = {}
        self.workflow_status = {}
        self.redis_url = os.getenv("REDIS_URL")
        
        model_name = "BAAI/bge-m3"
        encode_kwargs = {"normalize_embeddings": True}
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name, 
            model_kwargs={'device': 'cpu'},
            encode_kwargs=encode_kwargs
        )
        
        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY", "")
        )
        print("ChatBotService : Embedding Model & Qdrant Client Loaded")

    def _ensure_user_memories_collection(self, vector_size: int) -> None:
        collection_name = "User_memories"

        try:
            if hasattr(self.qdrant_client, "collection_exists"):
                exists = self.qdrant_client.collection_exists(collection_name=collection_name)
            else:
                self.qdrant_client.get_collection(collection_name=collection_name)
                exists = True
        except Exception:
            exists = False

        if exists:
            return

        try:
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
            print(f"[Qdrant] '{collection_name}' 컬렉션 생성 완료 (vector_size={vector_size})")
        except Exception:
            # 동시 요청으로 이미 생성됐을 수 있으므로 한 번 더 확인
            if hasattr(self.qdrant_client, "collection_exists") and self.qdrant_client.collection_exists(collection_name=collection_name):
                return
            raise
        
    async def from_redis_to_user_memories(self, user_id, problem_id, session_id):
        print(f"🔔 [Event] 세션 만료 알림 수신: {session_id}")
        
        redis_history_key = f"ai:chatbot:user:{user_id}:problem:{problem_id}:{session_id}"
        history = RedisChatMessageHistory(session_id=redis_history_key, url=self.redis_url)
        messages = history.messages
        
        if len(messages) < 2 : return
        
        chat_text = "\n".join([f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}" for m in messages])
        
        extract_prompt = ChatPromptTemplate.from_template(
            "당신은 학생의 학습 상태를 분석하는 전문가입니다. 아래 대화 내역을 분석하여 JSON 형식으로만 응답하세요.\n\n"
            "대화 내용:\n{chat_history}\n\n"
            "다음 필드를 포함한 JSON을 생성하세요:\n"
            "1. error_summary: 학생의 취약점이나 실수에 대한 한 줄 요약 (2-3줄 가능)\n"
            "2. weak_tags: 학생이 어려워하는 개념 알고리즘이나 자료구조 리스트 (예: ['DFS', '시간복잡도'])\n"
            "3. error_paragraph: 요약카드 문단 중 틀린 부분 (BACKGROUND, GOAL, RULE, CONSTRAINT 중 선택)\n"
            "4. graph_tags: 지식 그래프 노드 (예: ['BFS', 'Queue'])\n"
            "5. graph_edge: 지식 간의 관계 정의 (예: ['TIME', 'LOGIC_ERROR'])\n"
            "6. metric: 이번 세션의 가장 강점인 지수 (ACCURACY, INDEPENDENCE, SPEED, CONSISTENCY, REPORT 중 하나)\n"
        )
        
        chain = extract_prompt | llm
        response = await chain.ainvoke({"chat_history": chat_text})
        
        raw_content = response.content.replace("```json", "").replace("```", "").strip()   
        extracted = json.loads(raw_content)
        
        try:
            vector = self.embeddings.embed_query(extracted.get("error_summary", ""))
            self._ensure_user_memories_collection(vector_size=len(vector))
            
            payload = {
                "user_id": user_id,
                "problem_id": problem_id,
                "error_summary" : extracted.get("error_summary", ""),
                "weak_tags" : extracted.get("weak_tags", []),
                "error_paragraph" : extracted.get("error_paragraph", ""),
                "graph_tags" : extracted.get("graph_tags", []),
                "graph_edge" : extracted.get("graph_edge", []),
                "metric" : extracted.get("metric", {}),
                "scores" : {
                    "accuracy_score" : 0,
                    "independence_score" : 0,
                    "speed_score" : 0,
                    "consistency_score" : 0
                },
                "created_at": int(time.time())
            }
            
            self.qdrant_client.upsert(
                collection_name="User_memories",
                points=[
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload=payload
                    )
                ]
            )
            print(f"저장 완료! Timestamp: {payload['created_at']}")
            print(f"[User_Memories] 저장 완료 (User_Id: {user_id})")

            # Qdrant 저장 성공 후에만 Redis 세션 히스토리 삭제
            try:
                history.clear()
                print(f"[Redis] 세션 히스토리 삭제 완료 (key: {history.key})")
            except Exception as clear_error:
                print(f"[Redis] history.clear() 실패, direct delete 시도: {clear_error}")
                delete_result = history.redis_client.delete(history.key)
                print(f"[Redis] direct delete result: {delete_result} (key: {history.key})")
            
        except Exception as e:
            print(f"저장 실패 : {e}")

    def cancel_run(self, run_id: int) -> bool:
        run_info = self.workflow_status.get(run_id)
        if not run_info:
            return False

        run_info["status"] = WorkflowStatus.CANCELED
        run_info["cancel_requested"] = True

        run_task = run_info.get("task")
        if run_task and not run_task.done():
            run_task.cancel()

        return True

    async def run_and_stream(self, request: UserMsgCreateReq, trace_id: str = None):
        """LangGraph 워크플로우를 실행하고 생성된 이벤트를 SSE 형식으로 즉시 반환"""
        
        run_id = request.run_id
        user_id = request.user_id
        problem_id = request.problem_id
        session_id = request.session_id
        
        print(f"🔍 [Start] Run ID: {run_id} / Session ID: {session_id}")   
        
        # 1. Redis 히스토리 관리 (전체 대화 문맥 유지용)
        # 키 형식 예: "ai:chatbot:user:1:problem:1:abc-123-def"
        redis_history_key = f"ai:chatbot:user:{user_id}:problem:{problem_id}:{session_id}"
        
        history = RedisChatMessageHistory(
            session_id=redis_history_key,
            url=self.redis_url,
            ttl=60 * 60 * 24, # 24시간 보관
        )
        # 2. [두 번째 흐름] 이전 대화 내역 불러오기 (문맥 반영)
        all_messages = history.messages
        past_messages = all_messages[-10:] if len(all_messages) > 10 else all_messages
        print(f"📋 past_messages 개수: {len(past_messages)}")

             
        initial_state : ChatBotState = {
            # "messages": [HumanMessage(content=request.user_message)],
            "messages": past_messages + [HumanMessage(content=request.user_message)],            
            "user_id": request.user_id,
            "user_level": request.user_level,
            "problem_id": request.problem_id,
            "run_id": run_id,
            "paragraph_type": request.paragraph_type if request.paragraph_type else "BACKGROUND",
            "is_correct": False
        }

        final_data = None
        current_task = asyncio.current_task()
        
        self.workflow_status[run_id] = {
            "status": WorkflowStatus.PROCESSING,
            "is_correct": False,
            "current_answer": "",
            "paragraph_type": initial_state["paragraph_type"],
            "cancel_requested": False,
            "task": current_task
        }
        
        accumulated_text = ""
        
        try:
            
            async for event in chatbot_graph.astream_events(
                initial_state, 
                version="v2",
                config={}
            ):
                run_info = self.workflow_status.get(run_id, {})
                if run_info.get("cancel_requested") or run_info.get("status") == WorkflowStatus.CANCELED:
                    print(f"🛑 취소 요청 감지 (run_id: {run_id})")
                    raise asyncio.CancelledError()

                kind = event["event"]
                node_metadata = event.get("metadata", {})
                node_name = node_metadata.get("langgraph_node", "") if node_metadata else ""
                
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

            run_info = self.workflow_status.get(run_id, {})
            if run_info.get("cancel_requested") or run_info.get("status") == WorkflowStatus.CANCELED:
                print(f"🛑 완료 직전 취소 요청 감지 (run_id: {run_id})")
                raise asyncio.CancelledError()

            self.workflow_status[run_id]["status"] = WorkflowStatus.COMPLETED
                        
            # 4. [두 번째 흐름] 답변 완료 시 Redis에 한 쌍(run_id) 저장
            # Redis에 차곡차곡 쌓이며, 다음번 astream_events 호출 시 past_messages로 들어감
            
            actual_redis_key = history.key  # RedisChatMessageHistory 인스턴스가 실제로 사용하는 키 확인

            result1 = history.add_message(HumanMessage(
                content=request.user_message,
                additional_kwargs={
                    "run_id" : request.run_id,
                    "paragraph_type": initial_state["paragraph_type"]
                }
            ))
            result2 = history.add_message(AIMessage(
                content=accumulated_text,
                additional_kwargs={
                    "run_id" : request.run_id,
                    "paragraph_type": initial_state["paragraph_type"]
                }
            ))
            
            result3 = history.redis_client.expire(actual_redis_key, 60 * 60 * 24) # 24시간 보관
            
            print(f"🔍 Actual Redis Key: {actual_redis_key}")  # 실제로 저장된 키 확인
            print(f"🔍 expire result: {result3}")  # 1이면 성공, 0이면 키 없음

            print(f"✅ Redis 저장 완료 / key: {actual_redis_key}")
            print(f"📋 RPUSH result: {result1}, {result2}")  # 1, 2 이렇게 나와야 정상

                        
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
            run_info = self.workflow_status.get(run_id)
            if run_info is not None:
                run_info["status"] = WorkflowStatus.CANCELED
        
        except Exception as e:
            print(f"❌ 에러 발생: {e}")
            run_info = self.workflow_status.get(run_id)
            if run_info is not None:
                run_info["status"] = WorkflowStatus.FAILED
            error_msg = str(e)
            # 429 에러 등에 대한 메시지 처리
            if "429" in error_msg:
                error_msg = "현재 사용량이 많아 응답이 지연되고 있습니다. 잠시 후 다시 시도해주세요."
            
            yield f'event: error\ndata: {json.dumps({"code": "FAILED", "message": error_msg}, ensure_ascii=False)}\n\n'
            
        finally:
            print(f"🏁 [Finally] Text Length: {len(accumulated_text)} / Trace ID: {trace_id}")
            
            if trace_id:
                try:
                    # 전역 클라이언트 사용
                    langfuse_client.trace(id=trace_id).update(
                        output=final_data if final_data else "No response generated"
                    )
                    langfuse_client.flush() # 전송 강제
                    print("✅ Langfuse Output Updated Successfully")
                except Exception as lf_e:
                    print(f"🔥 Langfuse Update Failed: {lf_e}")
                    # 혹시 인증키 문제인지 확인
                    if not os.getenv("LANGFUSE_PUBLIC_KEY"):
                        print("🔥 Warning: LANGFUSE_PUBLIC_KEY not found in env")
            
            # 작업 종료 후 메모리 정리
            if run_id in self.workflow_status:
                del self.workflow_status[run_id]
            print(f"워크플로우 종료 (run_id: {run_id})")


# 컨트롤러 인스턴스 생성
bot_service = ChatBotService()
