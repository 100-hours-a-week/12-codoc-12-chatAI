import time
import json
import os
from datetime import datetime
from app.common.config import llm
from test_data import TEST_SCENARIOS
import openai

# ================= CONFIGURATION =================

# 1. 결과 저장 경로
SAVE_DIR = "scripts/model_test"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 2. 모델 리스트 (테스트할 6개 모델 ID)
MODELS = [
    "Qwen/Qwen2.5-14B-Instruct", # 밸런스
    "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct", # 한국어 중점
    "naver-hyperclovax/HyperCLOVAX-SEED-Think-14B",
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503", # 추론 중점
    "google/gemma-3-12b-it",
    "LGAI-EXAONE/EXAONE-3.5-32B-Instruct" # 한국어 중점
]

# 3. 연결 설정
BASE_URL = "https://qmswz6hlysjk6w-8000.proxy.runpod.net/v1" # RunPod IP로 변경
client = openai.OpenAI(api_key="EMPTY", base_url=BASE_URL)
# =================================================

def get_qualitative_evaluation(user_msg, response_text, keywords, current_step):
    # 다음 단계 매핑 (정답 시 유도 확인용)
    next_step_map = {
        "BACKGROUND": "GOAL",
        "GOAL": "STRATEGY",
        "STRATEGY": "INSIGHT",
        "INSIGHT": "완료"
    }
    target_next = next_step_map.get(current_step, "")
    
    # Gemini 사용
    judge_prompt = f"""
    당신은 AI 응답 품질 평가 전문가입니다. 다음 응답을 1~5점 척도로 평가하세요.
    
    [평가 기준]
    1. 은닉 규칙: "{keywords}" 단어를 직접 언급하지 않았는가?
    2. 단계 유도: 마지막에 "[{target_next} 이후의 다음 단계]"를 명확히 제안했는가?
    단계 명칭: BACKGROUND(배경 설정) → GOAL(목표 설정) → STRATEGY(전략 수립) → INSIGHT(통찰 도출) → 완료
    3. 톤앤매너: 중급자 수준에 맞게 논리적이고 전문적인가?
    4. 환각 체크: 문제 지문에 없는 내용을 지어내거나 잘못된 알고리즘/사실을 말하지 않았는가?
    
    [유저 메시지]
    {user_msg}
    
    [대상 응답]
    {response_text}
    
    모든 내용은 반드시 한국어로 대답하세요.
    
    결과를 반드시 JSON 형식으로만 응답하세요: 
    {{"keyword_score": 점수, "step_score": 점수, "tone_score": 점수, "hallucination_score": 점수, "reason": "이유"}}
    """
    
    try:
        # gemini 호출
        judge_res = llm.invoke(judge_prompt)
        content = judge_res.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        return json.loads(content)    
    except Exception as e:
        return {"error": f"평가 중 오류 발생: {str(e)}"}
    
def run_tutor_batch_test(model_name):
    print(f"🚀 [{model_name}] 배치 테스트 시작 (총 {len(TEST_SCENARIOS)}개 시나리오)")    
    
    # 모델별 파일에 저장
    file_safe_name = model_name.replace("/", "_")
    output_file = os.path.join(SAVE_DIR, f"results_{file_safe_name}.jsonl")
    
    for i, scenario in enumerate(TEST_SCENARIOS):
        print(f"🔄 [{i+1}/{len(TEST_SCENARIOS)}] 테스트 중: {scenario['step']} | 정답여부: {scenario['is_correct']}")
    
        # tutor_node 전용 시스템 프롬프트 구성
        system_prompt = f"""
        당신은 친절한 코딩 테스트 문제 풀이 도우미입니다. 유저 수준: {scenario['level']}.
        답변은 반드시 모두 한국어로 대답하세요.
        
        [현재 단계: {scenario['step']}]
        지침: {scenario['guide']}
        
        [규칙]
        - 은닉 규칙: {scenario['keywords']} 단어를 직접 언급하지 마세요.        
        - 단계 이동: 정답(is_correct=True) 시 반드시 다음 단계 명칭을 포함하여 순서대로 유도하세요.
        단계 명칭: BACKGROUND(배경 설정) → GOAL(목표 설정) → STRATEGY(전략 수립) → INSIGHT(통찰 도출) → 완료
        
        
        [문제 지문]
        {scenario['content']}        
        """

        start_time = time.time()
        try:
            # 스트리밍 모드로 TTFT 측정
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"유저 답변: {scenario['user_msg']}\n(판단 결과: {scenario['is_correct']})"}   
                ],
                stream=True,
                temperature=0.7
            )

            first_token_time = None
            full_content = ""
            
            for chunk in response:
                if chunk.choices[0].delta.content:
                    if first_token_time is None:
                        first_token_time = time.time()
                    full_content += chunk.choices[0].delta.content

            end_time = time.time()

            # 지표 계산
            ttft = first_token_time - start_time if first_token_time else 0
            char_count = len(full_content)
            estimated_tokens = char_count * 1.5 # 대략적인 토큰 수 계산
            total_time = end_time - start_time
            tps = estimated_tokens / total_time if total_time > 0 else 0
            
            scores = get_qualitative_evaluation(scenario['user_msg'], full_content, scenario['keywords'], scenario['step'])

            # 결과 데이터 구성
            result = {
                "step": scenario['step'],
                "model" : model_name,
                "quantitative_metrics": {
                    "ttft": round(ttft, 3),
                    "tpot": round((total_time - ttft) / len(full_content), 4) if len(full_content) > 0 else 0,             
                    "tps": round(tps, 2),
                    "total_latency": round(total_time, 3),
                    "token_count": len(full_content.split())
                },
                "qualitative_evaluation": {
                    "keyword_hidden_check": scores.get("keyword_score", 0),
                    "step_transition_check": scores.get("step_score", 0),
                    "tone_consistency_check": scores.get("tone_score", 0),
                    "hallucination_check": scores.get("hallucination_score", 0),  # 1~5점 척도
                    "reason": scores.get("reason", "")
                },
                "data": {
                    "input": scenario['user_msg'],
                    "output": full_content,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            
            print(f"✅ 완료: {output_file} 저장됨")

        except Exception as e:
            print(f"❌ 시나리오 에러 ({scenario['step']}): {e}")
        
    print(f"✅ [{model_name}] 모든 시나리오 완료! 결과: {output_file}")
    
if __name__ == "__main__":
    current_model = MODELS[5] # 지금 RunPod에 올린 모델
    run_tutor_batch_test(current_model)