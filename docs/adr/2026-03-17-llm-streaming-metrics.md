# ADR: LLM 스트리밍 지표 정의

## 상태
승인

## 배경
`chatAI` 는 SSE 기반으로 답변을 스트리밍한다. 기존 FastAPI 공통 대시보드만으로는 LLM 응답 특성을 보기 어려워서, 스트리밍 품질을 직접 보여주는 지표가 추가로 필요하다.

## 결정
다음 4개 지표를 표준 지표로 추가한다.

### 1. TTFT
- 메트릭: `codoc_llm_ttft_seconds`
- 정의: 요청 시작 시점부터 첫 토큰을 클라이언트에 전송하기 직전까지의 시간

### 2. TPS
- 메트릭: `codoc_llm_tokens_per_second`
- 정의: 첫 토큰 이후 생성 구간에서의 초당 출력 토큰 수
- 계산식: `completion_tokens / (last_token_time - first_token_time)`

### 3. TPOT
- 메트릭: `codoc_llm_tpot_seconds`
- 정의: 첫 토큰 이후 출력 토큰 하나를 생성하는 데 걸린 평균 시간
- 계산식: `(last_token_time - first_token_time) / completion_tokens`

### 4. Total Latency
- 메트릭: `codoc_llm_total_latency_seconds`
- 정의: 요청 시작부터 최종 응답 종료까지 걸린 전체 시간

## 라벨
모든 지표는 아래 라벨을 공통으로 사용한다.

- `route`
- `model`
- `status`
- `app_name`

## 구현 원칙
- 측정 시점은 SSE generator 내부에서 관리한다.
- `TTFT` 는 첫 토큰이 실제로 스트리밍되는 시점 기준으로 계산한다.
- `TPS` 와 `TPOT` 는 완료 후 누적 응답 텍스트의 토큰 수를 기준으로 계산한다.
- 토큰 수 계산이 실패하면 공백 분리 기반 fallback 을 사용한다.

## 결과
- 기존 FastAPI 공통 지표와 별개로, LLM 품질 전용 패널을 안정적으로 추가할 수 있다.
- 모델 변경 시에도 동일 정의를 유지할 수 있다.
