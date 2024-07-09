from typing import List

from fastapi import HttpException, status, APIRouter
from src.ai.adapter.ExecuteDeepLearningAdapter import ExecuteDeepLearningAdapter
import logging

router = APIROuter()
logger = logging.getLogger("default")

@router.post("/layer1/classification", response_mode=List)
async def execute_layer1_classification():
    return {"message: API call successfully"}