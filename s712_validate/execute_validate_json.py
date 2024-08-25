from pydantic import BaseModel, Field, HttpUrl, validator
from typing import Dict, Any

# VO 클래스 정의 (pydantic을 사용하여 유효성 검사 포함)
class UserProjectVO(BaseModel):
  id: str = Field(..., min_length=1, description="Unique identifier for the user")
  pw: str = Field(..., min_length=8, description="Password for the user")
  projectName: str = Field(..., min_length=1, description="Name of the project")
  githubUrl: HttpUrl = Field(..., description="GitHub repository URL")
  hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Hyperparameters for the machine learning model")
  dataPath: str = Field(..., min_length=1, description="File path to the training data")

  # 추가적인 유효성 검사가 필요한 경우, validator를 사용할 수 있습니다.
  @validator('pw')
  def password_strength(cls, v):
    if len(v) < 8:
      raise ValueError('Password must be at least 8 characters long')
    return v

# 예제 JSON 데이터 (dict 형태)
data = {
  "id": "user123",
  "pw": "securePassword123",
  "projectName": "MyMLProject",
  "githubUrl": "https://github.com/user123/mymlproject",
  "hyperparameters": {
    "learning_rate": 0.01,
    "batch_size": 32
  },
  "dataPath": "/data/mymlproject/dataset.csv"
}

# VO 객체 생성 및 유효성 검사
try:
  user_project_vo = UserProjectVO(**data)
  print("VO JSON data is valid.")
except ValueError as e:
  print("VO JSON data is invalid:", e)
