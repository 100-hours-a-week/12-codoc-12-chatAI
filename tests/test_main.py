from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    # GET 요청을 보냄
    response = client.get("/healthcheck")
    
    assert response.status_code == 200
    
    # 반환된 JSON 메시지가 {"status": "ok"} 인지 검증
    assert response.json() == {"status": "ok"}