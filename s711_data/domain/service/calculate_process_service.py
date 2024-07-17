import asyncio

async def execute_compute_process():
    
    tasks = [compute_square(i) for i in range(1,21)]
    # asyncio.gather(*tasks)는 주어진 비동기 작업을 리스트 병렬로 실행한 것을 모든 작업이 완료될 때까지 기다림
    # 각 작업이 완료 된 후 gather에 순서대로 반환하여 순서 보장
    results = await asyncio.gather(*tasks)
    print(f"Results: {results}")
    
async def compute_square(x):
    
    print(f"Compute square of {x}")
    result = x * x
    # 1초 대기하지만 어떤 작업이 먼저 완료될지는 이벤트 루프 스케쥴링에 따라 다름
    await asyncio.sleep(1)
    print(f"square of {x} is {result}")
    return result