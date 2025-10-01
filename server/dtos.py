import datetime
from typing import Optional
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

class AuthResponseDto(BaseModel):
    token: str
    refreshToken: Optional[str] = None
    expiration: datetime.datetime
    user: EmployeeDto

class FaceRegisterRequest(BaseModel):
    email: str
    firstName: str
    lastName: str
    position: str
    departmentId: str
    storeId: Optional[str] = None