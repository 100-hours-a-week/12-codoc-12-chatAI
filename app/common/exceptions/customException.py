from app.common.exceptions.BaseException import BusinessException

# 401: 인증 실패 (토큰 없음, 만료 등)
class CredentialException(BusinessException):
    def __init__(self, message: str = "자격 증명을 검증할 수 없습니다."):
        super().__init__(code="AUTH_401", message=message)

# 403: 권한 없음 (관리자만 접근 가능 등)
class UnauthorizedException(BusinessException):
    def __init__(self, message: str = "해당 권한을 가진 사용자가 아닙니다."):
        super().__init__(code="AUTH_403", message=message)