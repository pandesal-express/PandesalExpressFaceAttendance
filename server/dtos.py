import datetime
from typing import Any, Optional
from pydantic import BaseModel


class DepartmentDto(BaseModel):
    id: str
    name: str

class EmployeeDto(BaseModel):
    id: str
    email: str
    firstName: str
    lastName: str
    position: str
    department: DepartmentDto
    departmentId: Optional[str] = None
    storeId: Optional[str] = None

class FaceRegisterRequestDto(BaseModel):
    email: str
    firstName: str
    lastName: str
    position: str
    departmentId: str
    timeLogged: datetime.datetime
    storeId: Optional[str] = None

class ApiResponseDto(BaseModel):
    message: str
    success: bool
    statusCode: int
    data: Optional[dict[str, Any]] = None
