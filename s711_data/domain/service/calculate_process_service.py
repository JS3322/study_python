import asyncio

async def execute_compute_process():
    
    tasks = [compute_square(i) for i in range(1,21)]
    

    
async def compute_square(x):
    print(f"Compute square of {x}")
    result = x * x
    # 1초 대기하지만 어떤 작업이 먼저 완료될지는 이벤트 루프 스케쥴링에 따라 다름
    await asyncio.sleep(1)    