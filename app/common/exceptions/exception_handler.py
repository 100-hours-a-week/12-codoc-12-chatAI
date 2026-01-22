from fastapi import Request, FastAPI
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from app.common.exceptions.BaseException import BusinessException
from common.apiResponse import CommonResponse

def register_exception_handlers(app: FastAPI):
    
    # HTTP 에러 처리(ex. 404 NOT FOUND)
    @app.exception_handler(StarletteHTTPException)
    async def unicorn_exception_handler(request: Request, exc: StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=CommonResponse.fail_response(
                code=str(exc.status_code),
                message=exc.detail
            ).model_dump()
        )

    # 데이터 검증 에러 처리(Pydantic Validation Error)
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content=CommonResponse.fail_response(
                code="VALIDATION_ERROR",
                message="Validation failed",
                data=exc.errors()
            ).model_dump()
        )

    # 그 외 서버 내부 에러(500)
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content=CommonResponse.fail_response(
                code="INTERNAL_SERVER_ERROR",
                message=str(exc)
            ).model_dump()
        )

    # 커스텀 비즈니스 로직 에러 처리
    @app.exception_handler(BusinessException)
    async def business_exception_handler(request: Request, exc: BusinessException):
        return JSONResponse(
            status_code=400,
            content=CommonResponse.fail_response(
                code=exc.errorCode,
                message=exc.message
            ).model_dump()
        )