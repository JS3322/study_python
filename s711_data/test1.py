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



from fastapi import Header, HTTPException, Security

def verify_knox_token(x_knox_token: str = Header(...)):
    # 예제용으로 토큰을 "secret-token"으로 가정
    valid_token = "secret-token"
    if x_knox_token != valid_token:
        raise HTTPException(status_code=403, detail="Invalid or missing token")
        
        
        

from fastapi import APIRouter, Depends
from dependencies import verify_knox_token
from application.services import DataService
from adapters.repository import MongoRepository
from domain.models import DataModel
from typing import List

router = APIRouter()

def get_service():
    repo = MongoRepository()
    return DataService(repo)

@router.get("/data", response_model=List[DataModel], dependencies=[Depends(verify_knox_token)])
def get_data(service: DataService = Depends(get_service)):
    return service.fetch_data()
    
    
    
    
from fastapi import Header, HTTPException, Security
from pymongo.errors import PyMongoError
from adapters.repository import MongoRepository

def verify_knox_token(x_knox_token: str = Header(None)):
    if x_knox_token is None:
        raise HTTPException(status_code=400, detail="No 'knox' token provided in the header")
    
    # 예제용으로 토큰을 "secret-token"으로 가정
    valid_token = "secret-token"
    if x_knox_token != valid_token:
        raise HTTPException(status_code=403, detail="Invalid 'knox' token")

def get_database():
    try:
        return MongoRepository()
    except PyMongoError as e:
        raise HTTPException(status_code=503, detail="Database connection failed")
        
        
        


from fastapi import APIRouter, Depends
from dependencies import verify_knox_token, get_database
from application.services import DataService
from domain.models import DataModel
from typing import List

router = APIRouter()

@router.get("/data", response_model=List[DataModel], dependencies=[Depends(verify_knox_token)])
def get_data(service: DataService = Depends(DataService, dependency_overrides={MongoRepository: get_database})):
    return service.fetch_data()
    
    
    
    

from typing import List
from domain.models import DataModel
from adapters.repository import MongoRepository

class DataService:
    def __init__(self, repo: MongoRepository):
        self.repo = repo

    def fetch_data(self) -> List[DataModel]:
        try:
            return self.repo.get_all_data()
        except Exception as e:
            raise HTTPException(status_code=500, detail="Failed to fetch data from database")
            
            
            
            
            
from fastapi import HTTPException, status

# Global Level Exceptions
class DatabaseConnectionError(HTTPException):
    def __init__(self):
        super().__init__(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database connection failed")

class TokenValidationError(HTTPException):
    def __init__(self):
        super().__init__(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid 'knox' token")

class MissingTokenError(HTTPException):
    def __init__(self):
        super().__init__(status_code=status.HTTP_400_BAD_REQUEST, detail="No 'knox' token provided in the header")

# Application Layer Exceptions
class DataFetchError(HTTPException):
    def __init__(self):
        super().__init__(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to fetch data from database")
        
        
        
        

from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from exceptions import DatabaseConnectionError, TokenValidationError, MissingTokenError, DataFetchError

async def db_connection_error_handler(request: Request, exc: DatabaseConnectionError):
    return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={"detail": exc.detail})

async def token_validation_error_handler(request: Request, exc: TokenValidationError):
    return JSONResponse(status_code=status.HTTP_403_FORBIDDEN, content={"detail": exc.detail})

async def missing_token_error_handler(request: Request, exc: MissingTokenError):
    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": exc.detail})

async def data_fetch_error_handler(request: Request, exc: DataFetchError):
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"detail": exc.detail})

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content={"detail": exc.errors()})
    
    
    
    
    
    

from fastapi import FastAPI
from error_handlers import (
    db_connection_error_handler,
    token_validation_error_handler,
    missing_token_error_handler,
    data_fetch_error_handler,
    validation_exception_handler
)
from api.api_v1.endpoints import data

app = FastAPI()

# Register error handlers
app.add_exception_handler(DatabaseConnectionError, db_connection_error_handler)
app.add_exception_handler(TokenValidationError, token_validation_error_handler)
app.add_exception_handler(MissingTokenError, missing_token_error_handler)
app.add_exception_handler(DataFetchError, data_fetch_error_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)

app.include_router(data.router, prefix="/v1")






    









