from pydantic import BaseModel


class ErrorResponse(BaseModel):
    detail: str


class Pagination(BaseModel):
    page: int = 1
    page_size: int = 50
    total: int = 0

