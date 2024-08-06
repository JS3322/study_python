

@execution_time_decorator
async def execute_compute_process():
    tasks = [compute_square(i) for i in range(1, 21)]
    results = await asyncio.gather(*tasks)

@execution_time_decorator
async def compute_square(x):
    result = x * x
    await asyncio.sleep(1)
    run_loop_info = asyncio.get_running_loop()
    get_loop_info = asyncio.get_event_loop()
    return result