from typing import Optional, Any
from pydantic import BaseModel


class Response(BaseModel):
    message: str
    success: bool
    data: Optional[dict[str, Any]] = None
