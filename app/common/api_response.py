from typing import Generic, Optional, TypeVar
from pydantic import BaseModel

T = TypeVar("T")

class CommonResponse(BaseModel, Generic[T]):
    code : str
    message : str
    data : Optional[T] = None   
    
    model_config = {"arbitrary_types_allowed": True}
    
    @classmethod # 클래스 이용해 새 객체 생성하는 Factory Method 역할(클래스 이름으로 바로 호출가능)
    def success_response(cls, message:str, data : Optional[T] = None):
        return cls(code="SUCCESS", message=message, data = data)

    @classmethod
    def fail_response(cls, code: str, message:str, data : Optional[T] = None):
        return cls(code=code, message=message, data = data)