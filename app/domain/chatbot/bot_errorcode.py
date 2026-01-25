from app.common.exceptions.base_exception import BusinessException

# [400] 사용자 입력 관련 에러
class EmptyPromptException(BusinessException):
    def __init__(self, message: str = "질문 내용이 비어있습니다."):
        super().__init__(code="CHAT_400_1", message=message)

class TokenLimitExceededException(BusinessException):
    def __init__(self, max_token: int, message: str = None):
        if message is None:
            message = f"입력 길이가 너무 깁니다. (최대 {max_token} 토큰)"
        super().__init__(code="CHAT_400_2", message=message)

class InvalidSessionException(BusinessException):
    def __init__(self, message: str = "유효하지 않거나 만료된 채팅 세션입니다."):
        super().__init__(code="CHAT_400_3", message=message)
        
# [500/502] 외부 서비스 및 서버 내부 에러
class LLMGenerationException(BusinessException):
    def __init__(self, message: str = "AI 응답 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."):
        # 502 Bad Gateway: 외부 서비스(OpenAI)가 이상한 응답을 줬을 때 적절
        super().__init__(code="CHAT_502_1", message=message)

class VectorDbSearchException(BusinessException):
    def __init__(self, message: str = "지식 베이스 검색 중 오류가 발생했습니다."):
        super().__init__(code="CHAT_500_1", message=message)

class PolicyViolationException(BusinessException):
    def __init__(self, message: str = "부적절한 콘텐츠가 감지되어 답변을 생성할 수 없습니다."):
        super().__init__(code="CHAT_403_1", message=message)