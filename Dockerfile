# 파이썬 3.12 버전 환경을 가져옴
FROM python:3.12-slim

# 작업 폴더를 /app으로 설정
WORKDIR /app

# 필요한 라이브러리 목록을 복사하고 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 현재 폴더의 모든 소스 코드를 도커 내부로 복사
COPY . .

# 서버 실행(main.py 실행)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]