import json

from pydantic import BaseModel
from fastapi import Header, Response


class ErrorDetail(BaseModel):
    message: str
    type: str
    code: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


def error_constructor(message: str, error_type: str, code: str, status_code: int) -> Response:
    """
    Example:
        message="Invalid authentication",
        type="invalid_request_error",
        code="invalid_api_key"
        status_code=401
    """
    error_response = ErrorResponse(
        error=ErrorDetail(
            message=message,
            type=error_type,
            code=code
        )
    )
    return Response(
        status_code=status_code,
        content=json.dumps(error_response.model_dump()),
        media_type="application/json"
    )
