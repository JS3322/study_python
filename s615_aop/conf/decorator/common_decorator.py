import time
import inspect
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s :::: $(name)s :::: %(levelname)s :::: %(message)s')
logger = logging.getLogger(__name__)

# 데코레이터를 호출하면 함수를 내부 함수로 시간을 측정하고 원래 함수의 결과 반환
# execution_time_decorator 함수는 데코레이터 역할을 하고, 함수를 인자로 받아서 확장 또는 수정
# 내부 함수를 정의하는 이유 : 위치 인자, 키워드 인자를 사용하면 원래 함수가 어떤 인자를 받든 데코레이터가 이를 처리 가능 + 캡슐화
def execution_time_decorator(func)
	
    # 내부함수 wrapper는 원래 함수의 동작을 확장하는 역할
    # 내부함수 wrapper는 *args : 위치 인자, **kwargs : 키워드 인자를 받음
    # func(a,b,c) 형태로 위치 인자 *args는 a,b,c 튜플로 받음
    # func() 키워드 인자들을 딕셔너리로 받음
	def wrapper(*args, **kwargs):
        
        for arg in args:
            print(arg)
        for key, value in kwargs.imtes()
        	print(f"{key} = {value}")
            
        sig = inspect.signature(func)
        bound_args = sig.bind_partial(*args, **kwargs).arguments
        
    	start_time = time.time()
        # 원래 함수 func 호출 및 결과 저장
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        # func.__name__ : 데코레이터가 적용된 함수
        # 4f : 소수점 4자리까지 포맷팅
        print(f"Execution time of{func.__name__}: {execution_time:.4f}seconds")
        logger.debug(f"Execution time of {func.__name__}: {execution_time:.4f} seconds)
        # 원래 함수 반환
        return result
	return wrapper        