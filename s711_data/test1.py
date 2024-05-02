/your_project
│
├── app
│   ├── main.py          # FastAPI 애플리케이션 엔트리 포인트
│   ├── dependencies.py  # 의존성 관리 파일
│   ├── api              # 애플리케이션 엔드포인트
│   │   └── api_v1       # API 버전 관리
│   │       └── endpoints
│   │           └── data.py
│   ├── domain           # 도메인 모델
│   │   └── models.py    # 엔터티 정의
│   ├── application      # 애플리케이션 비즈니스 로직
│   │   └── services.py  # 서비스 로직
│   └── adapters         # 어댑터
│       └── repository.py# 데이터베이스 리포지토리
│
└── requirements.txt     # 필요한 라이브러리


from pydantic import BaseModel
from typing import List, Optional

class DataModel(BaseModel):
    id: str
    content: str
    
    
    
    
    
    
from typing import List
from domain.models import DataModel
from adapters.repository import AbstractRepository

class DataService:
    def __init__(self, repo: AbstractRepository):
        self.repo = repo

    def fetch_data(self) -> List[DataModel]:
        return self.repo.get_all_data()






from pymongo import MongoClient
from domain.models import DataModel

class AbstractRepository:
    def get_all_data(self) -> List[DataModel]:
        raise NotImplementedError

class MongoRepository(AbstractRepository):
    def __init__(self):
        self.client = MongoClient("mongodb://localhost:27017/")
        self.db = self.client['your_database_name']

    def get_all_data(self) -> List[DataModel]:
        collection = self.db['your_collection_name']
        return [DataModel(**data) for data in collection.find({})]
        
        
        
        
        
from fastapi import APIRouter, Depends
from application.services import DataService
from adapters.repository import MongoRepository

router = APIRouter()

def get_service():
    repo = MongoRepository()
    return DataService(repo)

@router.get("/data", response_model=List[DataModel])
def get_data(service: DataService = Depends(get_service)):
    return service.fetch_data()
    
    
    
    
from fastapi import FastAPI
from api.api_v1.endpoints import data

app = FastAPI()

app.include_router(data.router, prefix="/v1")







