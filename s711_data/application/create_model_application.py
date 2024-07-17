import logging
import asyncio
from src.s711_data.domain.service.calculate_process_service import execute_compute_process

def execute_create_model():
    try:
    	logging.info(" :: START :: execute_create_model")
    	asyncio.run(execute_compute_process())
    except Exception as e:
        raise Exception(e)
