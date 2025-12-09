from pydantic import BaseModel

class UserCaseRequest(BaseModel):
    data: str
