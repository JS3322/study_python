import logging
import asyncio
from src.s711_data.domain.service.calculate_process_service import execute_compute_process
from src.s711_data.domain.service.calculate_process_service import generator_example

def execute_create_model():
    try:
    	logging.info(" :: START :: execute_create_model")
    	asyncio.run(execute_compute_process())
        for value in gen:
            print(value)
    except Exception as e:
        raise Exception(e)
