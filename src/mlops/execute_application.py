import logging
import asyncio

@execution_time_decorator
def execute_create_model():
    try:
    	logging.info(" :: RUN :: execute_create_model{func.name}")})
        asyncio.run(execute_compute_process())
        gen = generator_example()
        for value in gen:
            print(value)
            
	except Exception as e:
    	raise Exception(e)            