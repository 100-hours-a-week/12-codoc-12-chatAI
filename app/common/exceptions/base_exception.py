from typing import Optional,TypeVar

# 비즈니스 로직 에러 담당 -> Custom
class BusinessException(Exception):
    def __init__(self, errorCode: str, message : str):
        self.errorCode = errorCode
        self.message = message
        #Exception 클래스도 메시지를 출력할 수 있도록 부모 클래스에도 전달
        super().__init__(f"[{errorCode}] {message}")