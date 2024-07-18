import logging
import asyncio
from src.s711_data.domain.service.calculate_process_service import execute_compute_process
from src.s711_data.domain.service.calculate_process_service import generator_example

def execute_create_model():
    try:
    	logging.info(" :: START :: execute_create_model")
    	asyncio.run(execute_compute_process())
        
        # 한 번에 하나의 값을 반환하고, 함수 상태를 유지한 채로 중단
        # iterator구현 또는 메모리 절약 위해 큰 데이터 스퀀스를 생성 용도
        for value in gen:
            print(value)
    except Exception as e:
        raise Exception(e)
