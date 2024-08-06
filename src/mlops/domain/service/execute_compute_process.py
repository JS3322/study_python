

@execution_time_decorator
async def execute_compute_process():
    tasks = [compute_square(i) for i in range(1, 21)]
    results = await asyncio.gather(*tasks)

@execution_time_decorator
async def compute_square(x):
    result = x * x
    await asyncio.sleep(1)
    return result