from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from . import bot_schemas
import json
from app.common.config import llm  # 원래 쓰시던 llm 가져오기
from app.graph.workflow import app as langgraph_app

router = APIRouter(prefix="/chatbot", tags=["chatbot"])
workflow_status = {}

@router.post("")
async def chat(post: bot_schemas.UserMsgCreateReq):
    workflow_status[post.run_id] = post.model_dump()
    return {"run_id": post.run_id, "status": "ACCEPTED"}

@router.get("/{run_id}/stream")
async def get_chat_stream(run_id: int):
    if run_id not in workflow_status:
        raise HTTPException(status_code=404, detail="Run ID not found")

    async def generate():
        data = workflow_status[run_id]

        # 1. 랭그래프를 'invoke'로 실행 (이벤트 추적 없이 결과만 딱 가져옵니다)
        # 이 과정에서 Qdrant 검색, 의도 파악, 검증이 한꺼번에 일어납니다.
        initial_state = {
            "messages": [("human", data["user_message"])],
            "problem_id": str(data["problem_id"]),
            "current_stage": data["current_node"]
        }

        yield f'event: status\ndata: {json.dumps({"message": "데이터 분석 중..."})}\n\n'

        # 💡 그래프 로직 실행 (검색 결과 등을 담은 최종 state를 가져옴)
        final_state = langgraph_app.invoke(initial_state)

        # 2. 그래프가 찾아온 데이터와 가이드 추출
        retrieved = final_state.get("retrieved_content", {})
        guide = retrieved.get("chatbot_answer_guide", "유저 스스로 생각하도록 유도하세요.")
        intent = final_state.get("intent", "CARD_PROGRESS")

        # 3. 원래 파트장님이 하시던 방식대로 llm.astream 실행!
        system_prompt = f"""
        너는 소크라테스식 질문법을 쓰는 튜터야.
        [가이드]: {guide}
        [의도]: {intent}
        [제약]: 반드시 2문장 이내로 짧게 질문할 것. 정답을 직접 주지 말 것.
        """

        user_msg = data["user_message"]

        try:
            # 💡 익숙한 그 방식 그대로!
            async for chunk in llm.astream([("system", system_prompt), ("human", user_msg)]):
                token = chunk.content
                if token:
                    yield f'event: token\ndata: {json.dumps({"result": {"text": token}}, ensure_ascii=False)}\n\n'

            yield f'event: final\ndata: {json.dumps({"status": "COMPLETED", "current_node": final_state.get("current_stage")})}\n\n'

        except Exception as e:
            yield f'event: error\ndata: {json.dumps({"message": str(e)})}\n\n'

    return StreamingResponse(generate(), media_type="text/event-stream")